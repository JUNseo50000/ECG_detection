import decimal
import optuna
import os
import argparse
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import KLDivergence, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (TensorBoard, CSVLogger)
from model import get_student_model, get_teacher_model
from datasets import ECGSequence

''' version check for requirements'''
import numpy as np
import sklearn
import scipy
import h5py
print("os Version:", os.__version__)
print("argparse Version:", argparse.__version__)
print("pandas Version:", pd.__version__)
print("tensorflow Version:", tf.__version__)
print("numpy Version:", np.__version__)
print("decimal version", decimal.__version__)
print("sklearn version", sklearn.__version__)
print("scipy version", scipy.__version__)
print("h5py version", h5py.__version__)
print("optuna version", optuna.__version__)
exit()


def verify_teacher_model(teacher_model, valid_seq):
    '''
    Check pre-trained teacher model works well

    Args:
        teacher_model: pre-trained teacher model
        valid_seq: validation sequence to calculate validation error
    Returns:
        avg_loss : average validation error of pre-trained teacher model
    '''
    # Use BCE to compare teacher outputs with ground truth
    bce_loss_fn = BinaryCrossentropy(from_logits=True)
    total_loss = 0.0
    batch_count = 0

    print("\n--- Verifying Teacher Model Predictions ---")
    for x_batch, y_batch in valid_seq:
        teacher_preds = teacher_model(x_batch, training=False)  # Get teacher predictions
        batch_loss = bce_loss_fn(y_batch, teacher_preds).numpy()  # Compute binary cross-entropy loss

        # Print per-batch loss
        print(f"Batch {batch_count + 1}, Teacher Loss: {batch_loss}")

        total_loss += batch_loss
        batch_count += 1

        # print("teacher_preds[0] : ", teacher_preds[0])

    # Calculate and print the average loss over all batches in validation set
    avg_loss = total_loss / batch_count
    print(f"\nAverage Teacher Model Loss: {avg_loss}")

    return avg_loss

def objective(trial, args):
    start_alpha_range = 0.8
    end_alpha_range = 0.9
    alpha_step = 0.1
    start_temperature_range = 1
    end_temperature_range = 20
    temperature_step = 1
    num_output_classes = 6
    lr = 0.001

    alpha = trial.suggest_float("alpha", start_alpha_range, end_alpha_range, step=alpha_step)
    temperature = trial.suggest_int("temperature", start_temperature_range, end_temperature_range, step=temperature_step)

    student_model = get_student_model(num_output_classes)
    student_model.compile(optimizer=Adam(learning_rate=lr))

    val_loss = run_training(args, student_model, alpha, temperature)
    return val_loss

def grid_search(args):
    num_output_classes = 6
    lr = 0.001

    # Define ranges for alpha and temperature
    alpha_values = [0.3, 0.5, 0.7, 0.9]
    temperature_values = [1, 5, 10, 15]

    # DataFrame to store results for each combination of alpha and temperature
    results = []

    for alpha in alpha_values:
        for temperature in temperature_values:
            print(f"Training with alpha={alpha} and temperature={temperature}")

            # Reload model and compile with updated alpha and temperature
            student_model = get_student_model(num_output_classes)
            student_model.compile(optimizer=Adam(learning_rate=lr))  # Adjust learning rate as needed

            # Train model and capture validation loss
            val_loss = run_training(args, student_model, alpha, temperature)

            # Save results
            results.append({
                'alpha': alpha,
                'temperature': temperature,
                'val_loss': val_loss
            })

    # Convert results to DataFrame and display
    results_df = pd.DataFrame(results)
    print("\nGrid Search Results:")
    print(results_df.sort_values(by="val_loss"))

def run_training(args, student_model, alpha, temperature):
    '''
    Train the student model to find optimal hyperparameters combination
    '''

    # Set up your training and validation sequences
    train_seq, valid_seq = ECGSequence.get_train_and_val(
        args.path_to_hdf5, args.dataset_name, args.path_to_csv, 64, args.val_split, n_leads=args.n_leads
    )

    teacher_model = get_teacher_model(train_seq.n_classes, weights_path=args.path_to_teacher_hdf5)
    teacher_model.trainable = False

    epochs = 10  # Limit epochs to save time
    best_val_loss = float('inf')
    consecutive_increase_count = 0
    patience_limit = 3

    for epoch in range(epochs):
        train_loss = 0.0
        for x_batch, y_batch in train_seq:
            with tf.GradientTape() as tape:
                teacher_preds = teacher_model(x_batch, training=False)
                student_preds = student_model(x_batch, training=True)

                loss_value = distillation_loss(student_preds, y_batch, teacher_preds, temperature, alpha)

            grads = tape.gradient(loss_value, student_model.trainable_variables)
            student_model.optimizer.apply_gradients(zip(grads, student_model.trainable_variables))

            train_loss += loss_value.numpy()

        # Validation loss calculation
        val_loss = 0.0
        for x_batch_val, y_batch_val in valid_seq:
            teacher_preds_val = teacher_model(x_batch_val, training=False)
            student_preds_val = student_model(x_batch_val, training=False)
            val_loss += distillation_loss(student_preds_val, y_batch_val, teacher_preds_val,
                                          temperature, alpha).numpy()

        val_loss /= len(valid_seq)
        print(f"Epoch {epoch+1}/{epochs}, alpha={alpha}, temperature={temperature}, Validation Loss: {val_loss}")

        # Track the best validation loss for each combination
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            consecutive_increase_count = 0  # Reset counter if we find a new best
        else:
            consecutive_increase_count += 1
            print(f"Validation loss increased for {consecutive_increase_count} consecutive epochs.")

        # Check if we should stop training
        if consecutive_increase_count >= patience_limit:
            print("Validation loss has increased for {} consecutive epochs. Stopping training.".format(patience_limit))
            break

    return best_val_loss

def quantize_student_model(student_model):
    # Convert the model to a quantized tflite format
    converter = tf.lite.TFLiteConverter.from_keras_model(student_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable default optimizations
    quantized_tflite_model = converter.convert()

    return quantized_tflite_model

def distillation_loss(student_output, labels, teacher_output, temperature=10, alpha=0.5):
    if labels.shape != student_output.shape:
        labels = labels[:student_output.shape[0], :student_output.shape[1]]

    # Define soft loss using KLDivergence
    # If the output is tremendous, use log_softmax
    kl_div = KLDivergence(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    soft_loss = kl_div(tf.nn.softmax(teacher_output / temperature),
                       tf.nn.softmax(student_output / temperature)) * (temperature ** 2)

    # Define hard loss for binary classification
    hard_loss = BinaryCrossentropy(from_logits=True)(labels, student_output)

    # print(f"Soft Loss: {soft_loss.numpy()}, Hard Loss: {hard_loss.numpy()}")

    # Combine losses
    loss = (1.0 - alpha) * hard_loss + alpha * soft_loss

    return loss


def train_student_model(args):
    # Set hyperparameters
    lr = 0.001
    batch_size = 64

    # With high temperature, student can learn more soft prob distribution
    temperature = 10
    # Small Student fit with Low alpha. vice versa
    alpha = 1.0

    # Initialize optimizer
    opt = Adam(learning_rate=lr)

    # Load data sequences
    train_seq, valid_seq = ECGSequence.get_train_and_val(
        args.path_to_hdf5, args.dataset_name, args.path_to_csv, batch_size, args.val_split, n_leads=args.n_leads)

    # Load teacher model for distillation
    teacher_model = get_teacher_model(train_seq.n_classes, weights_path=args.path_to_teacher_hdf5)
    teacher_model.trainable = False  # Freeze teacher model weights

    # verify_teacher_model(teacher_model, valid_seq)
    # return

    # Load student model
    student_model = get_student_model(train_seq.n_classes)
    student_model.compile(optimizer=opt)  # No loss specified here for custom training loop

    # Define callbacks
    callbacks = [
        TensorBoard(log_dir='./logs', write_graph=False),
        CSVLogger('training.log', append=False),
        # ModelCheckpoint(os.path.join(args.path_to_save, 'backup_model_last.keras')),
        # ModelCheckpoint(os.path.join(args.path_to_save, 'backup_model_best.keras'), save_best_only=True)
    ]
    # callbacks = [
    #     ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, min_lr=lr / 100),
    #     EarlyStopping(patience=9, min_delta=1e-5),
    #     TensorBoard(log_dir='./logs', write_graph=False),
    #     CSVLogger('training.log', append=False),
    #     ModelCheckpoint('./backup_model_best.keras', save_best_only=True, monitor='val_loss'),
    # ]

    for callback in callbacks:
        callback.set_model(student_model)

    # Initialize CSV Logger manually
    csv_logger = CSVLogger('training.log', append=False)
    csv_logger.set_model(student_model)  # Link the student model
    csv_logger.on_train_begin()  # Initialize the CSV file

    # Custom training loop
    epochs = 70
    best_val_loss = float('inf')
    consecutive_increase_count = 0
    patience_limit = 14

    total_batches = len(train_seq)

    # print(f"Total number of batches per epoch: {total_batches}")
    # print(f"len(valid_seq): {len(valid_seq)}")
    # return

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = 0.0
        for batch_idx, (x_batch, y_batch) in enumerate(train_seq):
            if batch_idx >= total_batches:
                break

            with tf.GradientTape() as tape:
                # Get teacher and student predictions
                teacher_preds = teacher_model(x_batch, training=False)
                student_preds = student_model(x_batch, training=True)

                # Calculate distillation loss
                loss_value = distillation_loss(student_preds, y_batch, teacher_preds,
                                               temperature=temperature, alpha=alpha)

            # Backpropagation
            grads = tape.gradient(loss_value, student_model.trainable_variables)
            opt.apply_gradients(zip(grads, student_model.trainable_variables))

            train_loss += loss_value.numpy()

            print(f"Batch {batch_idx + 1}/{total_batches}, Loss: {loss_value.numpy()}")

        train_loss /= len(train_seq)
        print(f"Training loss: {train_loss}")

        # Validation
        val_loss = 0.0
        for x_batch_val, y_batch_val in valid_seq:
            teacher_preds_val = teacher_model(x_batch_val, training=False)
            student_preds_val = student_model(x_batch_val, training=False)
            val_loss += distillation_loss(student_preds_val, y_batch_val, teacher_preds_val,
                                          temperature=temperature, alpha=alpha).numpy()

        val_loss /= len(valid_seq)
        print(f"Validation loss: {val_loss}")

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            student_model.save(os.path.join(args.path_to_save, "best_model.keras"))
            consecutive_increase_count = 0  # Reset counter if we find a new best
        else:
            consecutive_increase_count += 1
            print(f"Validation loss increased for {consecutive_increase_count} consecutive epochs.")

        # Check if we should stop training
        if consecutive_increase_count >= patience_limit:
            print("Validation loss has increased for {} consecutive epochs. Stopping training.".format(patience_limit))
            break

        # Log the metrics at the end of each epoch
        logs = {'val_loss': val_loss, 'loss': train_loss}
        csv_logger.on_epoch_end(epoch, logs=logs)

    csv_logger.on_train_end()
    # Save final model - consider quantization
    if args.quantization == True:
        best_model_path = os.path.join(args.path_to_save, "best_model.keras")
        quantized_model = quantize_student_model(load_model(best_model_path))
        quantized_model.save(os.path.join(args.path_to_save, "quantized_model.keras"))

    print("Training completed. Final model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train student model with knowledge distillation.')
    parser.add_argument('path_to_hdf5', type=str, help='Path to HDF5 file containing tracings')
    parser.add_argument('path_to_csv', type=str, help='Path to CSV file containing annotations')
    parser.add_argument('path_to_teacher_hdf5', type=str, help='Path to HDF5 file containing pre trained teacher weight')
    parser.add_argument('path_to_student_hdf5', type=str, help='Path to HDF5 file containing pre trained teacher weight')
    parser.add_argument('path_to_save', type=str, default='./', help='Path to HDF5 file to save the weight')
    parser.add_argument('--val_split', type=float, default=0.02, help='Validation split ratio')
    parser.add_argument('--dataset_name', type=str, default='tracings', help='Dataset name in HDF5 file')
    parser.add_argument('--n_leads', type=int, default=12, help='Number of leads')
    parser.add_argument('--quantization', type=bool, default=False, help='Whether to apply model Quantization')
    args = parser.parse_args()

    ''' Train '''
    train_student_model(args)

    ''' For finding good hyperparameters - alpha, temperature '''
   # grid_search(args)

    ''' Optuna or finding good hyperparameters - alpha, temperature '''
   # study = optuna.create_study(direction="minimize")
   # study.optimize(lambda trial: objective(trial, args), n_trials=30)
   # trials_df = study.trials_dataframe()
   # sorted_trials_df = trials_df.sort_values(by="value").head(30)
   # print(sorted_trials_df[['number', 'value', 'params_alpha', 'params_temperature']])
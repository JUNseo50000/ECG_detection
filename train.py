import decimal
import optuna
import os
import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import KLDivergence, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (TensorBoard, CSVLogger)
from model import get_student_model, get_teacher_model
from datasets import ECGSequence


# ''' version check for requirements'''
# import numpy as np
# import sklearn
# import scipy
# import h5py
# print("os Version:", os.__version__)
# print("argparse Version:", argparse.__version__)
# print("pandas Version:", pd.__version__)
# print("tensorflow Version:", tf.__version__)
# print("numpy Version:", np.__version__)
# print("decimal version", decimal.__version__)
# print("sklearn version", sklearn.__version__)
# print("scipy version", scipy.__version__)
# print("h5py version", h5py.__version__)
# print("optuna version", optuna.__version__)
# exit()


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
    start_alpha_range = 0.1
    end_alpha_range = 1.0
    alpha_step = 0.05
    start_temperature_range = 1
    end_temperature_range = 20
    temperature_step = 1
    num_output_classes = 6
    lr = 0.001

    alpha = trial.suggest_float("alpha", start_alpha_range, end_alpha_range, step=alpha_step)
    temperature = trial.suggest_int("temperature", start_temperature_range, end_temperature_range,
                                    step=temperature_step)

    student_model = get_student_model(num_output_classes)
    student_model.compile(optimizer=Adam(learning_rate=lr))

    val_loss = run_training(args, student_model, alpha, temperature)
    return val_loss


def grid_search(args):
    num_output_classes = 6
    lr = 0.001

    # Define ranges for alpha and temperature
    alpha_values = [0.3, 0.5, 0.7, 0.9]
    # alpha_values = [0.5, 0.7, 0.9]
    temperature_values = [1, 5, 10, 15]
    # temperature_values = [10]

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

    teacher_model = get_teacher_model(train_seq.n_classes, weights_path=args.path_to_teacher)
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
        print(f"Epoch {epoch + 1}/{epochs}, alpha={alpha}, temperature={temperature}, Validation Loss: {val_loss}")

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


def quantize_model(model):
    # Convert the model to a quantized tflite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable default optimizations
    quantized_tflite_model = converter.convert()

    # Wrap the quantized TFLite model in a tf.keras.Model
    quantized_keras_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=model.input_shape[1:]),
        tf.keras.layers.Lambda(lambda x: tf.convert_to_tensor(quantized_tflite_model, dtype=tf.float32))
    ])

    quantized_keras_model.save("./pre_trained/quantized_model.keras", save_format="keras")

    return quantized_keras_model


def quantize_and_save_model(model: keras.Model, save_path: str):
    """
    Quantizes the weights of a Keras model to int8 and saves the quantized model to a Keras file.

    Args:
        model (keras.Model): Pre-trained Keras model to be quantized.
        save_path (str): Path to save the quantized Keras model.
    """
    # Step 1: Convert weights to int8 format
    quantized_weights = []
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            quantized_layer_weights = []
            for weight in layer.get_weights():
                # Skip if the weight is all zeros or very small
                max_val = np.max(np.abs(weight))
                if max_val == 0:
                    quantized_layer_weights.append(weight)  # Keep original weights
                    continue
                # Scale weights to int8
                scale = 127 / max_val
                quantized_weight = np.round(weight * scale).astype(np.int8)
                # Store the scale as float32 for potential dequantization
                dequantized_weight = quantized_weight.astype(np.float32) / scale
                quantized_layer_weights.append(dequantized_weight)
            quantized_weights.append((layer.name, quantized_layer_weights))
        else:
            quantized_weights.append((layer.name, None))

    # Step 2: Apply the quantized weights back to the model
    for layer_name, weights in quantized_weights:
        if weights:
            layer = model.get_layer(name=layer_name)
            try:
                layer.set_weights(weights)
            except ValueError:
                print(f"Skipping layer '{layer_name}' as it does not accept the quantized weights.")

    # Step 3: Save the quantized model to the specified path
    model.save(save_path)
    print(f"Quantized model saved at: {save_path}")


def distillation_loss(student_output, labels, teacher_output, temperature=10, alpha=0.5):
    # if labels.shape != student_output.shape:
    #     labels = labels[:student_output.shape[0], :student_output.shape[1]]
    if labels.shape != student_output.shape:
        print(f"Labels shape: {labels.shape}")
        print(f"Student output shape: {student_output.shape}")

        raise ValueError("Labels and student output must have the same shape. Please preprocess your data.")

    # Define soft loss using KLDivergence
    # If the output is tremendous, use log_softmax
    kl_div = KLDivergence(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    soft_loss = kl_div(tf.nn.softmax(teacher_output / temperature),
                       tf.nn.softmax(student_output / temperature)) * (temperature ** 2)

    # Define hard loss for binary classification
    # todo: check
    # hard_loss = BinaryCrossentropy(from_logits=True)(labels, student_output)
    hard_loss = BinaryCrossentropy(from_logits=False)(labels, student_output)

    # print(f"Soft Loss: {soft_loss.numpy()}, Hard Loss: {hard_loss.numpy()}")

    # Combine losses
    loss = (1.0 - alpha) * hard_loss + alpha * soft_loss

    return loss


def train_student_model(args):
    # Set hyperparameters
    lr = 0.001
    batch_size = 128
    # batch_size = 64

    # With high temperature, student can learn more soft prob distribution
    temperature = 10
    # Small Student fit with Low alpha. vice versa
    alpha = 0.9

    # Initialize optimizer
    opt = Adam(learning_rate=lr)

    # Load data sequences
    train_seq, valid_seq = ECGSequence.get_train_and_val(
        args.path_to_hdf5, args.dataset_name, args.path_to_csv, batch_size, args.val_split, n_leads=args.n_leads)

    # Load teacher model for distillation
    teacher_model = get_teacher_model(train_seq.n_classes, weights_path=args.path_to_teacher)
    teacher_model.trainable = False  # Freeze teacher model weights

    # verify_teacher_model(teacher_model, valid_seq)
    # return

    # Load student model
    student_model = get_student_model(train_seq.n_classes, weights_path=args.path_to_student, n_leads=args.n_leads)
    student_model.compile(optimizer=opt)  # No loss specified here for custom training loop

    # Define callbacks
    callbacks = [
        TensorBoard(log_dir='./logs', write_graph=False),
        CSVLogger('training.log', append=False),
    ]

    for callback in callbacks:
        callback.set_model(student_model)

    # Initialize CSV Logger manually
    csv_logger = CSVLogger('training.log', append=False)
    csv_logger.set_model(student_model)  # Link the student model
    csv_logger.on_train_begin()  # Initialize the CSV file

    # Custom training loop
    epochs = 100
    best_val_loss = float('inf')
    consecutive_increase_count = 0
    patience_limit = 100

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

                if 4 <= args.n_leads < 9:
                    x_batch = x_batch[:, :, 6:]
                elif args.n_leads <= 3:
                    x_batch = x_batch[:, :, 4]

                student_preds = student_model(x_batch, training=True)
                # Calculate distillation loss
                loss_value = distillation_loss(student_preds, y_batch, teacher_preds,
                                               temperature=temperature, alpha=alpha)

            # Backpropagation
            grads = tape.gradient(loss_value, student_model.trainable_variables)
            opt.apply_gradients(zip(grads, student_model.trainable_variables))

            train_loss += loss_value.numpy()

            # print(f"Batch {batch_idx + 1}/{total_batches}, Loss: {loss_value.numpy()}")

        train_loss /= len(train_seq)
        print(f"Training loss: {train_loss}")

        # Validation
        val_loss = 0.0
        for x_batch_val, y_batch_val in valid_seq:
            # print(f"x_batch_val.shape: {x_batch_val.shape}")

            teacher_preds_val = teacher_model(x_batch_val, training=False)

            if 4 <= args.n_leads < 9:
                x_batch_val = x_batch_val[:, :, 0:6]
            elif args.n_leads <= 3:
                x_batch_val = x_batch_val[:, :, 4]
            student_preds_val = student_model(x_batch_val, training=False)

            # print(f"Labels shape: {y_batch_val.shape}")
            # print(f"Student output shape: {student_preds_val.shape}")

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
    if args.quantization:
        best_model_path = os.path.join(args.path_to_save, "best_model.keras")
        quantize_and_save_model(load_model(best_model_path), os.path.join(args.path_to_save, "quantized_model.keras"))

    print("Training completed. Final model saved.")


def compare_n_leads(args):
    best_val_losses = []
    final_val_losses = []
    for i in range(0, 12):
        # Set hyperparameters
        print(f"\n Start with lead {i}")
        lr = 0.001
        batch_size = 128

        # With high temperature, student can learn more soft prob distribution
        temperature = 10
        # Small Student fit with Low alpha. vice versa
        alpha = 0.9

        # Initialize optimizer
        opt = Adam(learning_rate=lr)

        # Load data sequences
        train_seq, valid_seq = ECGSequence.get_train_and_val(
            args.path_to_hdf5, args.dataset_name, args.path_to_csv, batch_size, args.val_split, n_leads=args.n_leads)

        # Load teacher model for distillation
        teacher_model = get_teacher_model(train_seq.n_classes, weights_path=args.path_to_teacher)
        teacher_model.trainable = False  # Freeze teacher model weights

        # Load student model
        student_model = get_student_model(train_seq.n_classes, weights_path=args.path_to_student, n_leads=args.n_leads)
        student_model.compile(optimizer=opt)  # No loss specified here for custom training loop

        # Define callbacks
        callbacks = [
            TensorBoard(log_dir='./logs', write_graph=False),
            CSVLogger('training.log', append=False),
        ]

        for callback in callbacks:
            callback.set_model(student_model)

        # Initialize CSV Logger manually
        csv_logger = CSVLogger('training.log', append=False)
        csv_logger.set_model(student_model)  # Link the student model
        csv_logger.on_train_begin()  # Initialize the CSV file

        # Custom training loop
        epochs = 10
        best_val_loss = float('inf')
        final_val_loss = float('inf')
        consecutive_increase_count = 0
        patience_limit = 10

        total_batches = len(train_seq)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss = 0.0
            for batch_idx, (x_batch, y_batch) in enumerate(train_seq):
                if batch_idx >= total_batches:
                    break

                with tf.GradientTape() as tape:
                    # Get teacher and student predictions
                    teacher_preds = teacher_model(x_batch, training=False)

                    x_batch = x_batch[:, :, i]

                    student_preds = student_model(x_batch, training=True)
                    # Calculate distillation loss
                    loss_value = distillation_loss(student_preds, y_batch, teacher_preds,
                                                   temperature=temperature, alpha=alpha)

                # Backpropagation
                grads = tape.gradient(loss_value, student_model.trainable_variables)
                opt.apply_gradients(zip(grads, student_model.trainable_variables))

                train_loss += loss_value.numpy()

                # print(f"Batch {batch_idx + 1}/{total_batches}, Loss: {loss_value.numpy()}")

            train_loss /= len(train_seq)
            print(f"Training loss: {train_loss}")

            # Validation
            val_loss = 0.0
            for x_batch_val, y_batch_val in valid_seq:
                # print(f"x_batch_val.shape: {x_batch_val.shape}")

                teacher_preds_val = teacher_model(x_batch_val, training=False)

                if 4 <= args.n_leads < 9:
                    x_batch_val = x_batch_val[:, :, 0:6]
                elif args.n_leads <= 3:
                    x_batch_val = x_batch_val[:, :, 4]
                student_preds_val = student_model(x_batch_val, training=False)

                val_loss += distillation_loss(student_preds_val, y_batch_val, teacher_preds_val,
                                              temperature=temperature, alpha=alpha).numpy()

            val_loss /= len(valid_seq)
            final_val_loss = val_loss
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
                print("Validation loss has increased for {} consecutive epochs. Stopping training.".format(
                    patience_limit))
                break

            # Log the metrics at the end of each epoch
            logs = {'val_loss': val_loss, 'loss': train_loss}
            csv_logger.on_epoch_end(epoch, logs=logs)
        best_val_losses.append(best_val_loss)
        final_val_losses.append(final_val_loss)

        csv_logger.on_train_end()

    for i in range(0, 12):
        print(f"----------Lead {i}----------")
        print(f"best_val_loss: {best_val_losses[i]}")
        print(f"final_val_loss: {final_val_losses[i]}")


if __name__ == "__main__":
    print("Hello")
    parser = argparse.ArgumentParser(description='Train student model with knowledge distillation.')
    parser.add_argument('path_to_hdf5', type=str, help='Path to HDF5 file containing tracings')
    parser.add_argument('path_to_csv', type=str, help='Path to CSV file containing annotations')
    parser.add_argument('path_to_teacher', type=str, help='Path to HDF5 file containing pre trained teacher weight')
    parser.add_argument('--path_to_student', type=str, default=None,
                        help='Path to HDF5 file containing pre trained student weight')
    parser.add_argument('--path_to_save', type=str, default='./pre_trained/',
                        help='Path to HDF5 file to save the weight')
    parser.add_argument('--val_split', type=float, default=0.02, help='Validation split ratio')
    parser.add_argument('--dataset_name', type=str, default='tracings', help='Dataset name in HDF5 file')
    parser.add_argument('--n_leads', type=int, default=12, help='Number of leads')
    parser.add_argument('--quantization', type=bool, default=False, help='Whether to apply model Quantization')
    parser.add_argument("--gpu", type=str, default="0",
                        help="Comma-separated list of GPU device IDs to use (e.g., '0,1')")
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpu_ids = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    # best_model_path = os.path.join(args.path_to_save, "best_model.keras")
    # quantize_and_save_model(load_model(best_model_path), "./pre_trained/quantized_model.keras")
    # exit()

    print(args.path_to_student)

    ''' Train '''
    # train_student_model(args)

    ''' For finding good hyperparameters - alpha, temperature '''
    # grid_search(args)

    ''' Optuna or finding good hyperparameters - alpha, temperature '''
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args), n_trials=30)
    trials_df = study.trials_dataframe()
    sorted_trials_df = trials_df.sort_values(by="value").head(30)
    print(sorted_trials_df[['number', 'value', 'params_alpha', 'params_temperature']])

    ''' compare leads performance '''
    # compare_n_leads(args)
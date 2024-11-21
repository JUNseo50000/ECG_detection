from sklearn.metrics import accuracy_score, recall_score, f1_score
from decimal import Decimal
import numpy as np


from model import get_student_model, get_teacher_model
import argparse
from datasets import ECGSequence


def find_optimal_threshold(args):
    batch_size = 64
    test_seq, _ = ECGSequence.get_train_and_val(
        args.path_to_hdf5, args.dataset_name, args.path_to_csv, batch_size, args.val_split, n_leads=args.n_leads)
    model = get_teacher_model(test_seq.n_classes, weights_path=args.path_to_teacher_hdf5)
    # model = get_student_model(test_seq.n_classes, weights_path=args.path_to_student_hdf5)

    start = Decimal('0.2')
    stop = Decimal('0.85')
    step = Decimal('0.05')
    threshold_range = np.array([float(start + i * step) for i in range(int((stop - start) / step))])  # Thresholds
    print(f"threshold_range: {threshold_range}")

    results = []  # List to store all metrics for each threshold

    for threshold in threshold_range:
        print(f"---------- threshold: {threshold}----------")
        acc, recall, f1 = evaluate_model_performance(model, test_seq, threshold=threshold)

        # Append the results as a dictionary for each threshold
        results.append({
            'threshold': threshold,
            'f1_score': f1,
            'accuracy': acc,
            'recall': recall
        })

    # Sort the results by F1-score in descending order
    results = sorted(results, key=lambda x: x['f1_score'], reverse=True)

    # Print sorted results
    print("Thresholds sorted by F1-score:")
    for res in results:
        print(
            f"Threshold: {res['threshold']}, F1-score: {res['f1_score']}, Accuracy: {res['accuracy']}, Recall: {res['recall']}")

    # Return the best threshold and its metrics
    best_result = results[0]  # Top result after sorting
    best_threshold = best_result['threshold']
    best_f1_score = best_result['f1_score']

    print(f"Optimal Threshold: {best_threshold}")
    print(f"Best F1-score: {best_f1_score}")
    return best_threshold, best_f1_score

def evaluate_model_performance(model, test_seq,):
    all_labels = []
    all_preds = []
    class_thresholds = [0.2, 0.4, 0.2, 0.35, 0.25, 0.2]

    for x_batch, y_batch in test_seq:
        preds = model(x_batch, training=False)  # TensorFlow tensor
        # print(f"preds.shape: {preds.shape}")

        # Convert the TensorFlow tensor to a NumPy array
        preds = preds.numpy()  # Explicit conversion to NumPy array

        # Apply thresholds
        preds = np.array([
            (preds[:, i] > class_thresholds[i]).astype(int) for i in range(len(class_thresholds))
        ]).T  # Transpose to match the original shape (batch_size, n_classes)
        # print(f"preds_np.shape: {preds.shape}")

        all_labels.extend(y_batch)
        all_preds.extend(preds)


    # Convert lists to arrays for metric calculation
    # Multi-label -> No Flatten
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # # Calculate metrics with multi-label handling
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)


    true_positives = np.sum((all_preds == 1) & (all_labels == 1), axis=0) # axis=0 : sum by column
    false_negatives = np.sum((all_preds == 0) & (all_labels == 1), axis=0)
    false_positives = np.sum((all_preds == 1) & (all_labels == 0), axis=0)

    precision_per_class = true_positives / (true_positives + false_positives)
    recall_per_class = true_positives / (true_positives + false_negatives)
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)

    average_precision = np.mean(precision_per_class)
    average_recall = np.mean(recall_per_class)
    average_f1_score = np.mean(f1_per_class)

    print(f"precision_per_class: {precision_per_class}")
    print(f"Recall per class: {recall_per_class}")
    print(f"F1 per class: {f1_per_class}")
    print(f"average_precision: {average_precision}")
    print(f"average_recall: {average_recall}")
    print(f"average_f1_score: {average_f1_score}")

    # recall_samples = recall_score(all_labels, all_preds, average='samples', zero_division=0)
    # recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    # recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    # print(f"recall_samples = {recall_samples}")
    # print(f"recall_micro = {recall_micro}")
    # print(f"recall_macro = {recall_macro}")
    # print(f"Average Recall: {average_recall}")
    # print(f"Recall per class: {recall_per_class}")

    return accuracy, recall, f1

def compare_models(args):
    batch_size = 64
    test_seq, _ = ECGSequence.get_train_and_val(
        args.path_to_hdf5, args.dataset_name, args.path_to_csv, batch_size, args.val_split, n_leads=args.n_leads)

    teacher_model = get_teacher_model(test_seq.n_classes, weights_path=args.path_to_teacher_hdf5)
    student_model = get_student_model(test_seq.n_classes, weights_path=args.path_to_student_hdf5)

    print("\n--- Evaluating Teacher Model ---")
    teacher_accuracy, teacher_recall, teacher_f1 = evaluate_model_performance(teacher_model, test_seq)

    print("\n--- Evaluating Student Model ---")
    student_accuracy, student_recall, student_f1 = evaluate_model_performance(student_model, test_seq)


    print(f"Teacher Model - Accuracy: {teacher_accuracy}, Recall: {teacher_recall}, F1 Score: {teacher_f1}")
    print(f"Student Model - Accuracy: {student_accuracy}, Recall: {student_recall}, F1 Score: {student_f1}")
    print("\n--- Comparison ---")
    print(f"Accuracy Improvement:\t{student_accuracy - teacher_accuracy}")
    print(f"Recall Improvement:\t{student_recall - teacher_recall}")
    print(f"F1 Score Improvement:\t{student_f1 - teacher_f1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train student model with knowledge distillation.')
    parser.add_argument('path_to_hdf5', type=str, help='Path to HDF5 file containing tracings')
    parser.add_argument('path_to_csv', type=str, help='Path to CSV file containing annotations')
    parser.add_argument('path_to_teacher_hdf5', type=str, help='Path to HDF5 file containing pre trained teacher weight')
    parser.add_argument('path_to_student_hdf5', type=str, help='Path to HDF5 file containing pre trained student weight')
    parser.add_argument('--val_split', type=float, default=0, help='Validation split ratio')
    parser.add_argument('--dataset_name', type=str, default='tracings', help='Dataset name in HDF5 file')
    parser.add_argument('--n_leads', type=int, default=12, help='Number of leads')
    parser.add_argument('--quantization', type=bool, default=False, help='Whether to apply model Quantization')
    args = parser.parse_args()

    compare_models(args)
    # find_optimal_threshold(args)
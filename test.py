from sklearn.metrics import accuracy_score, recall_score, f1_score
from decimal import Decimal
import numpy as np
import os

from model import get_student_model, get_teacher_model
import argparse
from datasets import ECGSequence


def find_best_threshold(results):
    # Initialize dictionaries to store the best threshold and value for each metric and class
    best_thresholds = {
        'f1_score': [],
        'fbeta_1dot25_per_class': [],
        'fbeta_1dot5_per_class': [],
        'fbeta_2dot0_per_class': []
    }

    for metric in ['f1_score', 'fbeta_1dot25_per_class', 'fbeta_1dot5_per_class', 'fbeta_2dot0_per_class']:
        best_for_metric = []
        for class_idx in range(len(results[0][metric])):
            best_value = -float('inf')
            best_threshold = None
            for result in results:
                if result[metric][class_idx] > best_value:
                    best_value = result[metric][class_idx]
                    best_threshold = result['threshold']
            best_for_metric.append({
                'class': class_idx,
                'threshold': best_threshold,
                'value': best_value
            })
        best_thresholds[metric] = best_for_metric

    return best_thresholds


def find_optimal_threshold(args):
    batch_size = 64
    test_seq, _ = ECGSequence.get_train_and_val(
        args.path_to_hdf5, args.dataset_name, args.path_to_csv, batch_size, args.val_split, n_leads=args.n_leads)
    # model = get_teacher_model(test_seq.n_classes, weights_path=args.path_to_teacher)
    model = get_student_model(test_seq.n_classes, weights_path=args.path_to_student)

    start = Decimal('0.0')
    stop = Decimal('1.0')
    step = Decimal('0.01')
    threshold_range = np.array([float(start + i * step) for i in range(int((stop - start) / step))])  # Thresholds
    print(f"threshold_range: {threshold_range}")

    results = []  # List to store all metrics for each threshold

    for threshold in threshold_range:
        print(f"\n---------- threshold: {threshold}----------")
        f1, fbeta1_1dot5, fbeta1_2dot0, fbeta_1dot25 = evaluate_model_performance(model, test_seq, threshold=threshold)

        # Append the results as a dictionary for each threshold
        results.append({
            # 'threshold': threshold,
            # 'f1_score': f1,
            # 'accuracy': acc,
            # 'recall': recall
            'threshold': threshold,
            'f1_score': f1,
            'fbeta_1dot25_per_class': fbeta_1dot25,
            'fbeta_1dot5_per_class': fbeta1_1dot5,
            'fbeta_2dot0_per_class': fbeta1_2dot0
        })

    find_best_threshold(results)  # Find the best thresholds
    best_thresholds = find_best_threshold(results)

    # Display results
    import pprint
    pprint.pprint(best_thresholds)


def evaluate_model_performance(model, test_seq, n_leads=12, threshold=0.5):
    all_labels = []
    all_preds = []
    class_thresholds = [threshold, threshold, threshold, threshold, threshold, threshold]
    # class_thresholds = [0.2, 0.4, 0.2, 0.35, 0.25, 0.2]

    for x_batch, y_batch in test_seq:
        if 4 <= n_leads < 9:
            x_batch = x_batch[:, :, 0:6]
        elif n_leads <= 3:
            x_batch = x_batch[:, :, 4]

        import time
        start_time = time.time()

        preds = model(x_batch, training=False)  # TensorFlow tensor

        end_time = time.time()
        taken_time = end_time - start_time

        # Convert the TensorFlow tensor to a NumPy array
        preds = preds.numpy()  # Explicit conversion to NumPy array

        # Apply thresholds
        preds = np.array([
            (preds[:, i] > class_thresholds[i]).astype(int) for i in range(len(class_thresholds))
        ]).T  # Transpose to match the original shape (batch_size, n_classes)

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

    true_positives = np.sum((all_preds == 1) & (all_labels == 1), axis=0)  # axis=0 : sum by column
    false_negatives = np.sum((all_preds == 0) & (all_labels == 1), axis=0)
    false_positives = np.sum((all_preds == 1) & (all_labels == 0), axis=0)

    precision_per_class = np.divide(
        true_positives,
        true_positives + false_positives,
        out=np.zeros_like(true_positives, dtype=float),
        where=(true_positives + false_positives) != 0
    )
    recall_per_class = np.divide(
        true_positives,
        true_positives + false_negatives,
        out=np.zeros_like(true_positives, dtype=float),
        where=(true_positives + false_negatives) != 0
    )
    f1_per_class = np.divide(
        2 * (precision_per_class * recall_per_class),
        precision_per_class + recall_per_class,
        out=np.zeros_like(precision_per_class, dtype=float),
        where=(precision_per_class + recall_per_class) != 0
    )
    beta = 1.25
    beta_squared = beta ** 2
    fbeta_1dot25_per_class = np.divide(
        (1 + beta_squared) * (precision_per_class * recall_per_class),
        (beta_squared * precision_per_class) + recall_per_class,
        out=np.zeros_like(precision_per_class, dtype=float),
        where=(beta_squared * precision_per_class + recall_per_class) != 0
    )
    beta = 1.5
    beta_squared = beta ** 2
    fbeta_1dot5_per_class = np.divide(
        (1 + beta_squared) * (precision_per_class * recall_per_class),
        (beta_squared * precision_per_class) + recall_per_class,
        out=np.zeros_like(precision_per_class, dtype=float),
        where=(beta_squared * precision_per_class + recall_per_class) != 0
    )
    beta = 2
    beta_squared = beta ** 2
    fbeta_2doat0_per_class = np.divide(
        (1 + beta_squared) * (precision_per_class * recall_per_class),
        (beta_squared * precision_per_class) + recall_per_class,
        out=np.zeros_like(precision_per_class, dtype=float),
        where=(beta_squared * precision_per_class + recall_per_class) != 0
    )
    average_precision = np.mean(precision_per_class)
    average_recall = np.mean(recall_per_class)
    average_f1_score = np.mean(f1_per_class)

    # print(f"precision_per_class: {precision_per_class}")
    # print(f"Recall per class: {recall_per_class}")
    # print(f"F1 per class: {f1_per_class}")
    # print(f"average_precision: {average_precision}")
    # print(f"average_recall: {average_recall}")
    # print(f"average_f1_score: {average_f1_score}")

    # return accuracy, recall, f1, taken_time

    return f1_per_class, fbeta_1dot25_per_class, fbeta_1dot5_per_class, fbeta_2doat0_per_class


def compare_models(args):
    batch_size = 64
    test_seq, _ = ECGSequence.get_train_and_val(
        args.path_to_hdf5, args.dataset_name, args.path_to_csv, batch_size, args.val_split, n_leads=args.n_leads)

    print("Get models")
    teacher_model = get_teacher_model(test_seq.n_classes, weights_path=args.path_to_teacher)
    student_model = get_student_model(test_seq.n_classes, weights_path=args.path_to_student, n_leads=args.n_leads)
    print("Done")

    print("--- Evaluating Teacher Model ---")
    teacher_accuracy, teacher_recall, teacher_f1, teacher_taken_time = evaluate_model_performance(teacher_model,
                                                                                                  test_seq)

    print("--- Evaluating Student Model ---")
    student_accuracy, student_recall, student_f1, student_taken_time = evaluate_model_performance(student_model,
                                                                                                  test_seq,
                                                                                                  n_leads=args.n_leads)

    # Absolute improvements
    accuracy_improvement = student_accuracy - teacher_accuracy
    recall_improvement = student_recall - teacher_recall
    f1_improvement = student_f1 - teacher_f1
    time_improvement = teacher_taken_time - student_taken_time  # Lower time is better

    # Percentage improvements
    accuracy_percent_improvement = (accuracy_improvement / teacher_accuracy * 100) if teacher_accuracy != 0 else float(
        'inf')
    recall_percent_improvement = (recall_improvement / teacher_recall * 100) if teacher_recall != 0 else float('inf')
    f1_percent_improvement = (f1_improvement / teacher_f1 * 100) if teacher_f1 != 0 else float('inf')
    time_percent_improvement = (time_improvement / teacher_taken_time * 100) if teacher_taken_time != 0 else float(
        'inf')

    print(f"\n--- Comparison in {args.n_leads} leads model ---")
    print(
        f"Teacher Model - Accuracy: {teacher_accuracy}, Recall: {teacher_recall}, F1 Score: {teacher_f1}, Taken Time: {teacher_taken_time}")
    print(
        f"Student Model - Accuracy: {student_accuracy}, Recall: {student_recall}, F1 Score: {student_f1}, Taken Time: {student_taken_time}")
    print(f"Accuracy Improvement:\t{accuracy_improvement:.4f} ({accuracy_percent_improvement:.2f}%)")
    print(f"Recall Improvement:\t{recall_improvement:.4f} ({recall_percent_improvement:.2f}%)")
    print(f"F1 Score Improvement:\t{f1_improvement:.4f} ({f1_percent_improvement:.2f}%)")
    print(f"Taken Time Improvement:\t{time_improvement:.4f} ({time_percent_improvement:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train student model with knowledge distillation.')
    parser.add_argument('path_to_hdf5', type=str, help='Path to HDF5 file containing tracings')
    parser.add_argument('path_to_csv', type=str, help='Path to CSV file containing annotations')
    parser.add_argument('path_to_teacher', type=str, help='Path to HDF5 file containing pre trained teacher weight')
    parser.add_argument('path_to_student', type=str, help='Path to HDF5 file containing pre trained student weight')
    parser.add_argument('--val_split', type=float, default=0, help='Validation split ratio')
    parser.add_argument('--dataset_name', type=str, default='tracings', help='Dataset name in HDF5 file')
    parser.add_argument('--n_leads', type=int, default=12, help='Number of leads')
    parser.add_argument('--quantization', type=bool, default=False, help='Whether to apply model Quantization')
    parser.add_argument("--gpu", type=str, default="-1",
                        help="Comma-separated list of GPU device IDs to use (e.g., '0,1')")
    args = parser.parse_args()

    gpu_ids = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    # compare_models(args)
    find_optimal_threshold(args)
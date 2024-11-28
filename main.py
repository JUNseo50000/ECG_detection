import argparse
from model import get_student_model, get_teacher_model
from datasets import ECGSequence

''' Todo '''
# Compare about quantization - Apply time measuring
# Create main file that combine edge and test sections
# 1개 일 때, 4번 lead말고 다른 것 비교하d기

''' Done '''
# compare percentage
# train student model
# train single and six leads model
# Regulate threshold by F-1 score (Emphasize recall)
# Time measure sectoin add
# think data load if-else -> exam_id
# pre_trained path 

if __name__ == "__main__":
    print("Hello")
    parser = argparse.ArgumentParser(description='Train student model with knowledge distillation.')
    parser.add_argument('path_to_hdf5', type=str, help='Path to HDF5 file containing tracings')
    parser.add_argument('path_to_csv', type=str, help='Path to CSV file containing annotations')
    parser.add_argument('path_to_student', type=str, default=None,
                        help='Path to HDF5 file containing pre trained student weight')
    parser.add_argument('--val_split', type=float, default=0, help='Validation split ratio')
    parser.add_argument('--dataset_name', type=str, default='tracings', help='Dataset name in HDF5 file')
    parser.add_argument('--n_leads', type=int, default=12, help='Number of leads')
    args = parser.parse_args()

    batch_size = 64
    test_seq, _ = ECGSequence.get_train_and_val(
        args.path_to_hdf5, args.dataset_name, args.path_to_csv, batch_size, args.val_split, n_leads=args.n_leads)

    ecg_data = test_seq[0][0]
    print(f"ECG_datas.shape: {ecg_data.shape}")

    ecg_iterator = iter(ecg_data)

    try:
        while True:
            user_input = input("Press Enter to process the next ECG data or type 'q' to quit: ").strip().lower()
            if user_input == 'q':
                print("Exiting program.")
                break
            ecg_data = next(ecg_iterator)


    except StopIteration:
        print("All ECG data processed.")

    # edge에 있는 함수에 넣기

    # 결과물 출력하기
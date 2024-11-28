import argparse
import h5py
import math
import pandas as pd
from tensorflow.keras.utils import Sequence
import numpy as np

'''
the third dimension to the 12 different leads of the ECG exams in the following order
`{DI, DII, DIII, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6}`.
4th :  aVL (Augmented Vector Left)
'''

class ECGSequence(Sequence):
    @classmethod
    def get_train_and_val(cls, path_to_hdf5, hdf5_dset, path_to_csv, batch_size=8, val_split=0.02, n_leads=12):
        n_samples = len(pd.read_csv(path_to_csv))
        print(f"n_samples={n_samples}")
        n_train = math.ceil(n_samples*(1-val_split))
        train_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, end_idx=n_train, n_leads=n_leads)
        valid_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, start_idx=n_train, n_leads=n_leads)
        return train_seq, valid_seq

    def __init__(self, path_to_hdf5, hdf5_dset, path_to_csv=None, batch_size=8,
                 start_idx=0, end_idx=None, n_leads=12):
        if path_to_csv is None:
            self.y = None
        else:
            # Load and align CSV with HDF5 exam_id order
            self.y = self._align_csv_with_hdf5(path_to_hdf5, path_to_csv)

        # Get tracings
        self.f = h5py.File(path_to_hdf5, "r")
        self.x = self.f[hdf5_dset]
        self.batch_size = batch_size
        self.n_leads = n_leads
        if end_idx is None:
            end_idx = len(self.x)
        self.start_idx = start_idx
        self.end_idx = end_idx

    @property
    def n_classes(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        if idx >= self.__len__():  # Bounds check to prevent out-of-range access
            raise IndexError("Index out of range for batches in this epoch.")
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        return np.array(self.x[start:end, :, :]), np.array(self.y[start:end])

    def __len__(self):
        # return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

        # Use integer division to drop last batch if it is incomplete
        return (self.end_idx - self.start_idx) // self.batch_size

    def __del__(self):
        self.f.close()

    @staticmethod
    def check_exam_id_order(path_to_hdf5, path_to_csv):
        # Load exam_id from CSV
        csv_exam_ids = ECGSequence._align_csv_with_hdf5(path_to_hdf5, path_to_csv)

        # Load exam_id from HDF5
        with h5py.File(path_to_hdf5, "r") as f:
            hdf_exam_ids = np.array(f['exam_id'])

        for i in range(0,10):
            print(f"exam_id of hdf and csv : {hdf_exam_ids[i]} & {csv_exam_ids[i]}")
            # if csv_exam_ids[i] != hdf_exam_ids[i]:
            #     print("Mismatch")
            #     break

    @staticmethod
    def _align_csv_with_hdf5(path_to_hdf5, path_to_csv):
        """
        Align CSV data with HDF5 exam_id order.
        """
        # Load exam_id from CSV and HDF5
        csv_data = pd.read_csv(path_to_csv)

        if "exam_id" not in csv_data.columns:
            print("The 'exam_id' column is missing in the CSV.")
            return pd.read_csv(path_to_csv).values

        csv_exam_ids = csv_data["exam_id"].tolist()

        with h5py.File(path_to_hdf5, "r") as f:
            hdf_exam_ids = np.array(f['exam_id'])

        # Create a mapping of exam_id to row in CSV
        valid_hdf_exam_ids = [exam_id for exam_id in hdf_exam_ids if exam_id != 0]
        csv_mapping = {exam_id: row for exam_id, row in zip(csv_exam_ids, csv_data.values)}

        # Ensure all HDF5 exam_ids exist in CSV
        missing_ids = [exam_id for exam_id in valid_hdf_exam_ids if exam_id not in csv_mapping]
        if missing_ids:
            raise KeyError(f"The following exam_ids are missing in CSV: {missing_ids}")

        # Reorder CSV data to match HDF5 exam_id order, excluding exam_id = 0
        ordered_data = [csv_mapping[exam_id] for exam_id in valid_hdf_exam_ids]

        result = np.array(ordered_data)
        result = result[:, 1:]

        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train student model with knowledge distillation.')
    parser.add_argument('path_to_hdf5', type=str, help='Path to HDF5 file containing tracings')
    parser.add_argument('path_to_csv', type=str, help='Path to CSV file containing annotations')
    parser.add_argument('--val_split', type=float, default=0.02, help='Validation split ratio')
    parser.add_argument('--dataset_name', type=str, default='tracings', help='Dataset name in HDF5 file')
    parser.add_argument('--n_leads', type=int, default=12, help='Number of leads')
    args = parser.parse_args()

    # ECGSequence.check_exam_id_order(args.path_to_hdf5, args.path_to_csv)
    train_seq, valid_seq = ECGSequence.get_train_and_val(args.path_to_hdf5, args.dataset_name, args.path_to_csv, 10, args.val_split, 12)

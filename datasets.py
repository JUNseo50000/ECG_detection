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
        with h5py.File(path_to_hdf5, "r") as f:
            n_samples = f[hdf5_dset].shape[0]  # Number of samples is the first dimension size
        # n_samples = len(pd.read_csv(path_to_csv))
        n_train = math.ceil(n_samples*(1-val_split))
        train_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, end_idx=n_train, n_leads=n_leads)
        # valid_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, start_idx=n_train, n_leads=n_leads)
        valid_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, start_idx=n_train, end_idx=n_samples, n_leads=n_leads)
        return train_seq, valid_seq

    def __init__(self, path_to_hdf5, hdf5_dset, path_to_csv=None, batch_size=8,
                 start_idx=0, end_idx=None, n_leads=12):
        if path_to_csv is None:
            self.y = None
        else:
            self.y = pd.read_csv(path_to_csv).values
            # self.y = pd.read_csv(path_to_csv, skiprows=range(1, start_idx + 1), nrows=(end_idx - start_idx)).values
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
        if self.n_leads == 12:
            if self.y is None:
                return np.array(self.x[start:end, :, :])
            else:
                return np.array(self.x[start:end, :, :]), np.array(self.y[start:end])
        elif self.n_leads == 1:
            if self.y is None:
                return np.array(self.x[start:end, :, 4])
            else:
                return np.array(self.x[start:end, :, 4]), np.array(self.y[start:end])

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)
        # Use integer division to drop last batch if it is incomplete
        # return (self.end_idx - self.start_idx) // self.batch_size

    def __del__(self):
        self.f.close()

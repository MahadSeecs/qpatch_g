import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch


class Dataset_RAVDESS(Dataset):
    def __init__(self, args, flag="train", root_path=""):
        """
        ESC-10 custom dataset loader for QCAAPatchTF
        Loads preprocessed .npy features and labels from/.
        """
        self.flag = flag
        self.root_path = root_path

        # Load features and labels
        X = np.load("ravdess_mel_features.npy")
        y = np.load("ravdess_labels.npy")

        if len(y.shape) > 1 and y.shape[1] == 1:
            y = y.squeeze(1)

        # Train/val/test split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        if flag == "train":
            self.X, self.y = X_train, y_train
        elif flag == "val":
            self.X, self.y = X_val, y_val
        else:
            self.X, self.y = X_test, y_test

        # Attributes needed by QCAAPatchTF
        self.max_seq_len = self.X.shape[1]  # 216
        self.enc_in = self.X.shape[2]  # 128
        self.num_samples = self.X.shape[0]

    def __getitem__(self, index):
        # return torch tensors
        x = torch.from_numpy(self.X[index]).float()  # (seq_len, enc_in)
        y = torch.tensor(self.y[index]).long()  # scalar label
        return x, y

    def __len__(self):
        return self.num_samples

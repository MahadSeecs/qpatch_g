import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

class Dataset_SAVEE(Dataset):
    def __init__(self, args, flag="train", root_path=""):
        """
        SAVEE dataset loader for QCAAPatchTF
        Loads preprocessed .npy features and labels.
        """
        self.flag = flag
        self.root_path = root_path

        # Load your precomputed mel features and labels
        X = np.load("savee_mel_features.npy")
        y = np.load("savee_labels.npy")

        # Ensure labels are flattened
        if len(y.shape) > 1 and y.shape[1] == 1:
            y = y.squeeze(1)

        # Split into train/val/test
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

        # Attributes used by QCAAPatchTF
        self.max_seq_len = self.X.shape[1]
        self.enc_in = self.X.shape[2]
        self.num_samples = self.X.shape[0]

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float()
        y = torch.tensor(self.y[index]).long()
        return x, y

    def __len__(self):
        return self.num_samples

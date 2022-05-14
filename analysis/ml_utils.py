# analysis/ml_models.py
"""
This module will implement the machine learning models used in the thesis.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import timedelta

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Setting up the environment to ensure reproducability
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Checking if cuda is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#################################
###### HANDLE MISSING DATA ######
## USE DATE_RANGE FROM PANDAS ###
#################################
class TimeSeriesDataset(Dataset):
    """BTC Dataset"""

    def __init__(self, X, y,  seq_length):
        self.seq_length = seq_length
        self.X = X
        self.y = y

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.y)

    def __getitem__(self, idx):
        """Method for iterating and indexing the dataset."""
        end = (self.y.index[idx]
            - timedelta(days=1)).strftime("%Y-%m-%d")
        start = (self.y.index[idx]
            - timedelta(days=self.seq_length + 1)).strftime("%Y-%m-%d")

        X_item = torch.Tensor(self.X.loc[start:end])
        y_item = torch.tensor(self.y[idx])
        return (X_item, y_item)


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

from helpers import generate_realized_volatility, generate_rolling_realized_variance, interpolate_missing_values
from helpers import generate_log_returns, generate_squared_jumps
from helpers import generate_semi_variance

# Setting up the environment to ensure reproducability
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Checking if cuda is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class ReturnsSeriesDataset(Dataset):
    """Returns Dataset"""

    def __init__(self, X, y,  seq_length):
        self.seq_length = seq_length
        self.X = interpolate_missing_values(X).iloc[:, 0]
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
        return {
            "X": X_item,
            "y": y_item
        }

class RVSeriesDataset(Dataset):
    """BTC Dataset"""

    def __init__(self, data,  seq_length):
        self.seq_length = seq_length
        self.data = data
        self.y = data[self.seq_length:]

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.y) - 2

    def __getitem__(self, idx):
        """Method for iterating and indexing the dataset."""
        end = (self.y.index[idx + 1]
            - timedelta(days=1)).strftime("%Y-%m-%d")
        start = (self.y.index[idx + 1]
            - timedelta(days=self.seq_length + 1)).strftime("%Y-%m-%d")

        X_item = torch.Tensor(self.data.loc[start:end])
        y_item = torch.tensor(self.y[idx + 1])
        return {
            "X": X_item,
            "y": y_item
        }


class HARDataset(Dataset):
    """Class for generating data for neural networks with HAR feature inputs"""
    def __init__(self, data, seq_length):
        self.seq_length = seq_length
        self.data = data
        self.y = data["Daily"].iloc[self.seq_length:]

    def __len__(self):
        return len(self.y) - 2

    def __getitem__(self, idx):
        """Method for iterating and indexing the dataset."""
        end = (self.y.index[idx + 1]
               - timedelta(days=1)).strftime("%Y-%m-%d")
        start = (self.y.index[idx + 1]
                 - timedelta(days=self.seq_length + 1)).strftime("%Y-%m-%d")

        X_daily = torch.Tensor(self.data["Daily"].loc[start:end])
        X_weekly = torch.Tensor(self.data["Weekly"].loc[start:end])
        X_monthly = torch.Tensor(self.data["Monthly"].loc[start:end])
        y_item = torch.tensor(self.y[idx + 1])

        return {
            "X_daily": X_daily,
            "X_weekly": X_weekly,
            "X_monthly": X_monthly,
            "y": y_item
        }

        
class Log_HARDataset(Dataset):
    """Class for generating data for neural networks with HAR feature inputs"""
    def __init__(self, data, seq_length):
        self.seq_length = seq_length
        self.data = data
        self.y = data["Daily"].iloc[self.seq_length:]

    def __len__(self):
        return len(self.y) - 2

    def __getitem__(self, idx):
        """Method for iterating and indexing the dataset."""
        end = (self.y.index[idx + 1]
               - timedelta(days=1)).strftime("%Y-%m-%d")
        start = (self.y.index[idx + 1]
                 - timedelta(days=self.seq_length + 1)).strftime("%Y-%m-%d")

        X_daily = torch.Tensor(self.data["Daily"].loc[start:end])
        X_weekly = torch.Tensor(self.data["Weekly"].loc[start:end])
        X_monthly = torch.Tensor(self.data["Monthly"].loc[start:end])
        y_item = torch.tensor(self.y[idx + 1])

        return {
            "X_daily": X_daily,
            "X_weekly": X_weekly,
            "X_monthly": X_monthly,
            "y": y_item
        }


class HAR_RS_I_Dataset(Dataset):
    """Class for generating data for neural networks with HAR feature inputs"""
    def __init__(self, data, seq_length):
        self.seq_length = seq_length
        self.data = data
        self.y = data["Daily"].iloc[self.seq_length:]

    def __len__(self):
        return len(self.y) - 2

    def __getitem__(self, idx):
        """Method for iterating and indexing the dataset."""
        end = (self.y.index[idx + 1]
               - timedelta(days=1)).strftime("%Y-%m-%d")
        start = (self.y.index[idx + 1]
                 - timedelta(days=self.seq_length + 1)).strftime("%Y-%m-%d")

        X_rsp = torch.Tensor(self.data["RS+"].loc[start:end])
        X_rsn = torch.Tensor(self.data["RS-"].loc[start:end])
        X_daily = torch.Tensor(self.data["Data"].loc[start:end])
        X_weekly = torch.Tensor(self.data["Weekly"].loc[start:end])
        X_monthly = torch.Tensor(self.data["Monthly"].loc[start:end])
        y_item = torch.tensor(self.y[idx + 1])

        return {
            "X_daily": X_daily,
            "RS+": X_rsp,
            "RS-": X_rsn,
            "X_weekly": X_weekly,
            "X_monthly": X_monthly,
            "y": y_item
        }


class HAR_J_Dataset(Dataset):
    """Class for generating data for neural networks with HAR feature inputs"""
    def __init__(self, data, seq_length):
        self.seq_length = seq_length
        self.data = data
        self.y = data["Daily"].iloc[self.seq_length:]

    def __len__(self):
        return len(self.y) - 2

    def __getitem__(self, idx):
        """Method for iterating and indexing the dataset."""
        end = (self.y.index[idx + 1]
               - timedelta(days=1)).strftime("%Y-%m-%d")
        start = (self.y.index[idx + 1]
                 - timedelta(days=self.seq_length + 1)).strftime("%Y-%m-%d")

        X_daily = torch.Tensor(self.data["Daily"].loc[start:end])
        X_jump = torch.Tensor(self.data["Jumps"].loc[start:end])
        X_weekly = torch.Tensor(self.data["Weekly"].loc[start:end])
        X_monthly = torch.Tensor(self.data["Monthly"].loc[start:end])
        y_item = torch.tensor(self.y[idx + 1])

        return {
            "X_daily": X_daily,
            "X_jump": X_jump,
            "X_weekly": X_weekly,
            "X_monthly": X_monthly,
            "y": y_item
        }


def generate_har_dataframe(dataframe):
    df = pd.DataFrame()
    df["Daily"] = generate_realized_volatility(
        data=dataframe,
        resolutions=["5T"],
        feature="Close"
    )
    df["Weekly"] = generate_rolling_realized_variance(
        data=df["Daily"],
        roll=7
    )
    df["Monthly"] = generate_rolling_realized_variance(
        data=df["Daily"],
        roll=30
    )
    df = df[30:]
    return df


def generate_log_har_dataframe(dataframe):
    df = pd.DataFrame()
    df["Daily"] = generate_realized_volatility(
        data=dataframe,
        resolutions=["5T"],
        feature="Close"
    )
    df["Weekly"] = generate_rolling_realized_variance(
        data=df["Daily"],
        roll=7
    )
    df["Monthly"] = generate_rolling_realized_variance(
        data=df["Daily"],
        roll=30
    )
    df = df[30:]
    df = np.log(df)
    return df


def generate_har_j_dataframe(dataframe):
    df = pd.DataFrame()
    df["Daily"] = generate_realized_volatility(
        data=dataframe,
        resolutions=["5T"],
        feature="Close"
    )
    df["Weekly"] = generate_rolling_realized_variance(
        data=df["Daily"],
        roll=7
    )
    df["Monthly"] = generate_rolling_realized_variance(
        data=df["Daily"],
        roll=30
    )

    returns = generate_log_returns(
        data=dataframe,
        feature="Close",
        resolution="5T"
    )
    df["Jumps"] = generate_squared_jumps(
        returns=returns,
        rv=df["Daily"]
    )
    df = df[30:]
    return df


def generate_har_rsi_dataframe(dataframe):
    df = pd.DataFrame()
    df["Daily"] = generate_realized_volatility(
        data=dataframe,
        resolutions=["5T"],
        feature="Close"
    )
    df["Weekly"] = generate_rolling_realized_variance(
        data=df["Daily"],
        roll=7
    )
    df["Monthly"] = generate_rolling_realized_variance(
        data=df["Daily"],
        roll=30
    )

    df["Returns"] = generate_log_returns(
        data=dataframe,
        feature="Close",
        resolution="1D"
    )
    df["RS+"] = generate_semi_variance(
        data=dataframe, 
        resolution="5T",
        sign="positive"
    )
    df["RS-"] = generate_semi_variance(
        data=dataframe, 
        resolution="5T",
        sign="negative"
    )

    df = df[30:]
    return df

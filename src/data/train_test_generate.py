# src/data/train_test_generate.py
"""
Module implementing functions for generating training and testing datasets.

"""
import numpy as np
from src.data.data_generators import generate_realized_var, generate_intraday

def generate_rv_data(dataset, feature: str, n_lags: int):
    """
    Generates a dataset of realized variance.
    :param dataset: BaseDataset - Instance of BaseDataset
    :param feature: str - The feature of which to calculate realized variance.
    :param n_lags: int - The number of lagged variables to include.
    """
    # Generating the realized variance.
    target = dataset.realized_variance(feature, "daily")
    features = np.zeros(shape=(len(target) - n_lags, n_lags))
    for i in range(len(target) - n_lags):
        features[i] = target[i:i+n_lags]
    features = np.asarray(target[n_lags+1:])
    data = list(zip(target, features))
    return data


########### NEEDS TO BE REWRITTEN RUNTIME TOO SLOW ###########
def generate_intraday_data(dataset, feature: str):
    """
    Generates a dataset of intraday features.
    :param dataset: BaseDataset - Instance of BaseDataset
    :param feature: str - The feature to generate the dataset from.
    """
    data = dataset.get_feature(feature)
    features = [generate_intraday(data, date) for date in data.index.date]
    target = dataset.realized_var(data, "daily") 
    return list(zip(target, features))
    
        
        
        

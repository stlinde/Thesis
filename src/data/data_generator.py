# src/data/data_generator.py
"""
This script generates the basic features that are going to be used throughout
the project.
Features:
    Tick Returns
    Hourly Returns
    Daily Returns
    Log Prices
    Log Returns
    Daily Realized Variance
    Sum Over Period
"""
import pandas as pd
import numpy as np

def load_dataset(path: str):
    """
    Loads dataset.
    :param path:        str - The path to the dataset.
    """
    file_type = path.split(".")[-1]
    match file_type:
        case "csv":
            return pd.read_csv(path)
        case "parquet":
            return pd.read_parquet(path)
        case "feather":
            return pd.read_feather(path)
        case _:
            raise Exception(f"Invalid file type. Expected: csv, parquet or feather. Got: {file_type}")

def set_datetime_index(data):
    """
    Converts timestamp to datetime and use as index.
    :param data: pd.DataFrame - The dataframe to be converted.
    """
    data['datetime'] = pd.to_datetime(data["timestamp"], unit='s')
    return data.set_index('datetime')

def sum_period(data, period: str):
    """
    Generates sum of data in day or month.
    :param data:    pandas.Series - The feature to be summed.
    :param period:  str - The period to sum over.
    """
    match period:
        case "daily":
            return data.groupby(pd.Grouper(freq="D")).sum()
        case "monthly":
            return data.groupby(pd.Grouper(freq="M")).sum()
        case _:
            raise Exception(f"Invalid period. Expected: daily or monthly")

def generate_returns(data, period: str, feature: str = "Close"):
    """
    Generates returns based on period.
    :param data:    pandas.DataFrame - DataFrame with datetime index and feature.
    :param period:  str - Period to compute return over.
    :param feature: (Optional) str - The feature to compute the returns off. Default: Close
    """
    match period:
        case "tick":
            return data[feature].pct_change()[1:]
        case "daily":
            return data[feature].resample('D').last().pct_change()[1:]
        case "monthly":
            return data[feature].resample('M').last().pct_change()[1:]
        case _:
            raise Exception(f"Invalid period. Expected: tick, daily or monthly. Got: {period}")

def generate_log_price(data):
    """
    Generates the Logarithmic price of the given feature.
    :param data: pandas.Series - The price data to be computed on. Must be datetime indexed.
    """
    return np.log(data.astype("float64"))

def generate_log_returns(data, period: str = "daily"):
    """
    Generates Logarithmic returns of the given feature.
    :param data:    pandas.Series - The price data to be computed on. Must be datetime indexed.
    :param period:  (Optional) str - The timeframe to compute the log returns over. Either minute or daily.
    """
    log_price = generate_log_price(data)
    match period:
        case "minute":
            return (log_price - log_price.shift(1))[1:]
        case "daily":
            daily_log_price = log_price.resample("D").last()
            return (daily_log_price - daily_log_price.shift(1))[1:]
        case _:
            raise Exception(f"Invalid period. Expected: minute or daily. Got {period}")

def generate_realized_var(data, period: str):
    """
    Generates the Realized Variance for the given period.
    :param data: pandas.Series - The price data to be computed on. Must be datetime indexed.
    :param period: str - The timeframe to generate Realized Variance over. Either daily or monthly.
    """
    log_returns = generate_log_returns(data)
    match period:
        case "daily":
            return log_returns.pow(2).groupby(pd.Grouper(freq="D")).sum()
        case "monthly":
            daily_log_returns = log_returns.resample("D").last()
            return daily_log_returns.pow(2).groupby(pd.Grouper(freq="M")).sum()
        case _:
            raise Exception(f"Invalid period. Expected: daily or monthly. Got: {period}")

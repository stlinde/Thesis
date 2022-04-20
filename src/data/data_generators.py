# src/data/data_generators.py
"""
This script holds functions for generating data for BaseDataset.
Functions:
    load_data:              Loads the data file into pandas DataFrame.
    set_datetime_index:     Sets datetime index on DataFrame
    sum_periods:            Sums the selected feature over a given period.
    generate_returns:       Generates returns of a feature.
    generate_log_prices:    Generates Logarithmic prices of a feature.
    generate_log_returns:   Generates Logarithmic returns of a feature.
    generate_realized_var:  Generates the Realized Variance of a feature.
"""
import pandas as pd
import numpy as np

def load_data(path: str):
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

def generate_returns(data, period: str):
    """
    Generates returns based on period.
    :param data:    pandas.Series - Series with datetime index and feature.
    :param period:  str - Period to compute return over.
    """
    match period:
        case "tick":
            return data.pct_change()[1:]
        case "daily":
            return data.resample('D').last().pct_change()[1:]
        case "monthly":
            return data.resample('M').last().pct_change()[1:]
        case _:
            raise Exception(f"Invalid period. Expected: tick, daily or monthly. Got: {period}")

def generate_log_price(data):
    """
    Generates the Logarithmic price of the given feature.
    :param data:    pandas.Series - The price data to be computed on. Must be datetime indexed.
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

def generate_intraday(data, date):
    """
    Generates an array of intraday movements in a given feature.
    :param data: pandas.Series - The data of the feature. Must have datetime index.
    :param date: pandas.DateTiem - The date for which to generate the array.
    """
    if isinstance(date, str):
        array = data[data.index.date == pd.to_datetime(date).date()]
    else:
        array = data[data.index.date == date]
    return np.asarray(array)

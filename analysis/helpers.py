# analysis/helpers.py
"""
Module implementing various helper functions.
"""
import pandas as pd
import numpy as np

def set_datetime_index(data):
    """Converts timestamp to datetime and use as index.
    :param data:    pd.DataFrame - The dataframe to be converted.
    """
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    return data.set_index('timestamp', drop=True)

def resample_dataframe(data, resolution):
    """Resamples dataframe to specified resolution.
    :param data:        pd.DataFrame - The dataframe to be resampled.
    :param resolution:  str - The resolution to be resampled to.
    """
    if resolution == None:
        return data
    resampled = data.resample(resolution).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })
    return resampled

def realized_quantity(data, func):
    """Applies the function to each day.
    :param data:        pd.Series - The data to be computed upon.
    :param func:    function - The function to be applied.
    """
    return data.groupby(pd.Grouper(freq="1D")).apply(func)

def generate_returns(data, feature: str = "Close", resolution: str = None):
    """Generates returns on a given resolution.
    :param data:        pd.DataFrame - The DataFrame containing the prices.
    :param feature:     str - The feature to compute the returns on.
    :param resolution:  str - The interval to compute returns over.
    """
    return resample_dataframe(data, resolution)[feature].pct_change()[1:] 

def generate_log_prices(data):
    """Generates log transformed prices.
    :param data:    pd.Series - The prices used.
    """
    return np.log(data)

def generate_log_returns(data, feature: str = "Close", resolution: str = None):
    """Generates log returns on a given resolution.
    :param data:        pd.DataFrame - The DataFrame containing the prices.
    :param feature:     str - The feature to compute the returns on.
    :param resolution:  str - The interval to compute returns over.
    """
    temp = generate_log_prices(resample_dataframe(data, resolution)[feature])
    return (temp - temp.shift(1))[1:]

def generate_realized_variance(data, resolutions: list, feature:str = "Close"):
    """Generates the realized variance of a return series
    :param data:        pd.DataFrame - The DataFrame holding the data.
    :param feature:     str - The feature to compute the returns on.
    :param resolutions: list - The intervals to compute the returns over.
    """
    temp = pd.DataFrame()
    for resolution in resolutions:
        temp[resolution] = (
            generate_log_returns(data, feature, resolution)
                .pow(2)
                .groupby(pd.Grouper(freq="D"))
                .sum()
        )
    return temp.mean(axis=1)

def generate_rolling_realized_variance(data, roll: int):
    """Generates rolling average of the realized variance.
    :param data:        pd.Series - The realized variance.
    :param roll:        int - The number of days to compute the mean over.
    """
    return data.rolling(roll).mean()

def bipower_variation(data):
    """Computes the bipower variation in one day.
    :param data:    pd.Series - The series containing the data.
    """
    const = 1 / (2 / np.pi)
    return const * np.sum(np.abs(data) * np.abs(data.shift(1)))

def generate_bpv(data):
    """Generates the standardized realized bipower variation.
    :param data:    pd.DataFrame - The realized variances.
    """
    return realized_quantity(data, bipower_variation)


def generate_squared_jumps(returns, rv):
    """Generates the squared jumps.
    :param returns:     pd.Series - The returns.
    :param rv:          pd.Series - The realized variances
    """
    bpv = generate_bpv(returns)
    rv = rv[1:]
    return [np.maximum(rv[i] - bpv[i], 0) for i in range(rv.shape[0])]

################
## NOT RITGHT ##
################
def generate_realized_quarticity(data):
    """Generates the realized quarticity.
    :param data:        pd.Series - The series containing the realized variances.
    """
    return np.sum(data**4)*data.shape[0] / 3

def generate_semi_variance(rv, returns, sign: str):
    """Generates the realized semi variance.
    :param rv:    pd.Series - The series containing the resalized variance.
    :param returns: pd.Series - The series containing the returns.
    :param sign:    str - positive or negative semivariance.
    """
    if sign == "positive":
        return [rv[i] if returns[i] > 0 else 0 for i in range(len(rv))]
    else:
        return [rv[i] if returns[i] < 0 else 0 for i in range(len(rv))]

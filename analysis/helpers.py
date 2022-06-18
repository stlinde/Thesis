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

def generate_realized_volatility(data, resolutions: list, feature: str = "Close"):
    """Generates the realized volatility of a return series.
    :param data:        pd.DataFrame - The DataFrame holding the data.
    :param feature:     str - The feature to compute the returns on.
    :param resolutions: list - The intervals to compute the returns over.
    """
    return np.sqrt(generate_realized_variance(data, resolutions, feature))

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


def generate_squared_jumps(data, resolution: str):
    """Generates the squared jumps.
    :param data:    pd.DataFrame - The dataframe holding the close prices.
    """
    df = pd.DataFrame()

    df["Returns"] = generate_log_returns(
        data=data,
        feature="Close",
        resolution=resolution
    )
    df["Realized Variance"] = generate_realized_variance(
        data=data,
        feature="Close", 
        resolutions=[resolution]
    )
    df["Bipower Variation"] = generate_bpv(df["Returns"])
    df["jumps"] = np.maximum(
        df["Realized Variance"] - df["Bipower Variation"],
        0
    )
    return df["jumps"]

def realized_quarticity(data):
    """Computes the realized quarticity
    :param data:        pd.Series - The series containing the returns.
    """
    return np.sum(data**4)*data.shape[0] / 3

def generate_realized_quarticity(data):
    """Generates the realized quarticity for a time series.
    :param data:        pd.Series - The series containing the returns.
    """
    return realized_quantity(data, realized_quarticity)


#def generate_semi_variance(data, resolution: str, sign: str):
#    """Generates the realized variance of a return series
#    :param data:        pd.DataFrame - The DataFrame holding the data.
#    :param feature:     str - The feature to compute the returns on.
#    :param resolutions: list - The intervals to compute the returns over.
#    """
#    temp = pd.DataFrame()
#    temp["returns"] = generate_log_returns(data, "Close", resolution)
#    temp["positive"] = temp["returns"].clip(0, np.inf)
#    temp["negative"] = temp["returns"].clip(-np.inf, 0)
#    if sign == "positive":
#        return temp["positive"].pow(2).groupby(pd.Grouper(freq="D")).sum()
#    else:
#        return temp["negative"].pow(2).groupby(pd.Grouper(freq="D")).sum()
            
def generate_semi_variance(data, resolution: str, sign: str):
    """Generates the realized variance of a return series
    :param data:        pd.DataFrame - The DataFrame holding the data.
    :param feature:     str - The feature to compute the returns on.
    :param resolutions: list - The intervals to compute the returns over.
    """
    temp = pd.DataFrame()
    temp["returns"] = generate_log_returns(data, "Close", resolution)
    temp["positive"] = temp["returns"].apply(lambda x: x ** 2 * (x > 0))
    temp["negative"] = temp["returns"].apply(lambda x: x ** 2 * (x < 0))

    if sign == "positive":
        return temp["positive"].groupby(pd.Grouper(freq="D")).sum()
    else:
        return temp["negative"].groupby(pd.Grouper(freq="D")).sum()

def interpolate_missing_values(data):
    """Interpolates missing values in DataFrame.

    :param data:    pd.DataFrame - The dataframe containing missing values.
    """
    time_index = pd.date_range(
        start = data.index[0],
        end = data.index[-1],
        freq = pd.infer_freq(data.index)
    ) 
    temp = pd.DataFrame(index=time_index)
    temp = temp.merge(
        right = data,
        how = 'left',
        left_index = True,
        right_index = True
    )
    return temp.interpolate()

# analysis/stats_test.py
"""
This module will implement utilities for testing the assumptions of the time 
series regression.
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.stats.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Stationarity
def augmented_dickey_fuller(data):
    """Implements the Augmented Dickey-Fuller test to test for stationarity in
    time series.
    :param data:        pd.Series - The time series to be tested.
    """
    return adfuller(data)[1]


# No perfect collinearity between regressors.
def vif(X):
    """Implements the variance inflation factor for testing collinearity.
    :param X:       pd.DataFrame - The X variables of the regression.
    """
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                            for i in range(len(X.columns))]
    return vif_data

# Covariance of residual term and independent variables should be 0

# Homoscedasticity of the residual terms on the independent variables
def breusch_pagan(resids, exog):
    """Implements the Breusch-Pagan test for testing for heteroscedasticity
    in the residuals.
    :param resids:      list - The residuals of the regression.
    :param exog:        pd.DataFrame - The exogenous variables of the regression
    """
    return sm.het_breuschpagan(resids, exog)[1]

def white_test(resids, exog):
    """Implements the White test for testing for heteroscedasticity
    in the residuals.
    :param resids:      list - The residuals of the regression.
    :param exog:        pd.DataFrame - The exogenous variables of the regression
    """
    return sm.het_white(resids, exog)[1]

# No autocorrelation of the residual term.
def ljung_box(data, lags: int):
    """Implementing the Ljung-Box test for testing autocorrelation in a time
    series.
    :param data:        list - The time series to test.
    """
    return sm.acorr_ljungbox(data, lags=lags)

# Normality of residuals
def jarque_bera(resids):
    """Implements the Jarque-Bera test for testing normality in residuals.
    :param resids:      list - The residuals of our regression.
    """
    return stats.jarque_bera(resids)[1]

def shapiro_wilk(resids):
    """Implements the Shapiro-Wilk test for testing normality in residuals.
    :param resids:      list - The residuals of our regression.
    """
    return stats.shapiro(resids)[1]

def kolmogorov_smirnov(resids):
    """Implements the Kolmogorov-Smirnov test for testing normaility in
    residuals
    :param resids:      list - The residuals of the regression.
    """
    return stats.kstest(resids, 'norm')[1]

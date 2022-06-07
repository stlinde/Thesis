# analysis/models.py
"""
Implements the econometric models in the analysis.
Implements functionalities for training and testing.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS


from helpers import ( 
    set_datetime_index,
    generate_realized_volatility, 
    generate_realized_volatility,
    generate_realized_quarticity,
    generate_rolling_realized_variance,
    generate_log_returns,
    generate_semi_variance,
    generate_squared_jumps,
    generate_bpv
)

from evaluation import (
    compute_mse,
    compute_mae,
    compute_mape,
    compute_qlike
) 

TRAIN_SIZE = 0.7
TRAIN_PERIOD = 365

class Model:
    def __init__(self):
        pass 

    def in_sample(self, X, y):
        self.model = OLS(y, X).fit()
        return self.model.summary()

    def residuals(self):
        return self.model.resid

    def in_sample_predict(self, X):
        return self.model.predict(X)

    def one_step_ahead(self, X, y, X_t):
        temp_model = OLS(y, X).fit()
        return temp_model.predict(X_t)

    def out_sample_eval(self, X, y, n_features):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        self.forecasts = []
        for i in range(365, len(X)):
            forecast = self.one_step_ahead(
                X.iloc[i - 365:i - 1, :],
                y.iloc[i - 365:i - 1],
                X.iloc[i, :].values.reshape(1, n_features))
            self.forecasts.append(forecast.item())
        return self.forecasts

    def loss(self, y_test, metric: str):
        y_test = y_test.reset_index(drop=True)
        if metric == "MSE":
            return compute_mse(y_test, self.forecasts)
        elif metric == "MAE":
            return compute_mae(y_test, self.forecasts)
        elif metric == "MAPE":
            return compute_mape(y_test, self.forecasts)
        elif metric == "QLIKE":
            return compute_qlike(y_test, self.forecasts)
        else:
            return "Loss function not implemented"
            

class HAR(Model):
    def __init__(self, data, resolutions):
        self.data = data
        self.resolutions = resolutions
        self.n_features = 4

        self.rv = generate_realized_volatility(
            data = self.data,
            resolutions = self.resolutions,
            feature = "Close"
        )

        # Setting up data needed.
        temp = pd.DataFrame()
        temp["Daily"] = self.rv
        temp["Weekly"] = generate_rolling_realized_variance(
            temp["Daily"], roll=7
        )
        temp["Monthly"] = generate_rolling_realized_variance(
            temp["Daily"], roll=30
        )
        temp = temp.iloc[30:, :]

        # Setting up the X_train, X_test, y_train, y_test
        self.X = sm.add_constant(temp.iloc[:-1, :].reset_index(drop=True))
        self.y = temp['Daily'][1:].reset_index(drop=True)

        # Creating train size
        train_size_int = int(self.X.shape[0] * TRAIN_SIZE // 1)

        self.X_train = self.X.iloc[:train_size_int, :]
        self.X_test = self.X.iloc[train_size_int - TRAIN_PERIOD:, :]
        self.y_train = self.y.iloc[:train_size_int]
        self.y_test = self.y.iloc[train_size_int - TRAIN_PERIOD:]

    def in_sample(self):
        return super().in_sample(self.X_train, self.y_train)

    def residuals(self):
        return super().residuals()

    def in_sample_predict(self):
        return super().in_sample_predict(self.X_train)

    def out_sample_eval(self):
        return super().out_sample_eval(self.X_test, self.y_test, self.n_features)

    def loss(self, metric: str):
        return super().loss(self.y_test[365:], metric=metric) 
        
class LogHAR(Model):
    def __init__(self, data, resolutions):
        self.data = data
        self.resolutions = resolutions
        self.n_features = 4

        self.rv = generate_realized_volatility(
            data = self.data,
            resolutions = self.resolutions,
            feature = "Close"
        )

        # Setting up data needed.
        temp = pd.DataFrame()
        temp["Daily"] = self.rv
        temp["Weekly"] = generate_rolling_realized_variance(
            temp["Daily"], roll=7
        )
        temp["Monthly"] = generate_rolling_realized_variance(
            temp["Daily"], roll=30
        )
        temp = temp.iloc[30:, :]

        # Setting up the X_train, X_test, y_train, y_test
        self.X = np.log(temp.iloc[:-1, :].reset_index(drop=True))
        self.X = sm.add_constant(self.X)
        self.y = np.log(temp['Daily'][1:].reset_index(drop=True))

        # Creating train size
        train_size_int = int(self.X.shape[0] * TRAIN_SIZE // 1)

        self.X_train = self.X.iloc[:train_size_int, :]
        self.X_test = self.X.iloc[train_size_int - TRAIN_PERIOD:, :]
        self.y_train = self.y.iloc[:train_size_int]
        self.y_test = self.y.iloc[train_size_int - TRAIN_PERIOD:]

    def in_sample(self):
        return super().in_sample(self.X_train, self.y_train)

    def residuals(self):
        return super().residuals()

    def in_sample_predict(self):
        return super().in_sample_predict(self.X_train)

    def out_sample_eval(self):
        return super().out_sample_eval(self.X_test, self.y_test, self.n_features)

    def loss(self, metric: str):
        return super().loss(self.y_test[365:], metric=metric) 

class HAR_QF(Model):
    def __init__(self, data, resolutions):
        self.data = data
        self.resolutions = resolutions

        self.rv = generate_realized_volatility(
            data = self.data,
            resolutions = self.resolutions,
            feature = "Close"
        )

        # Setting up data needed.
        temp = pd.DataFrame()
        temp["Daily"] = self.rv
        temp["Weekly"] = generate_rolling_realized_variance(
            temp["Daily"], roll=7
        )
        temp["Monthly"] = generate_rolling_realized_variance(
            temp["Daily"], roll=30
        )
        temp = temp.iloc[30:, :]

        temp['RQ1RV1']  = np.sqrt(
            generate_realized_quarticity(temp['Daily'])) * temp['Daily']
        
        temp['RQ2RV2']  = np.sqrt(
            generate_realized_quarticity(temp['Weekly'])) * temp['Weekly']

        temp['RQ3RV3']  = np.sqrt(
            generate_realized_quarticity(temp['Monthly'])) * temp['Monthly']

        temp.iloc[30:, :]
        self.X = temp.iloc[:-1, -3:].reset_index(drop=True)
        self.X = sm.add_constant(self.X)
        self.y = np.log(temp['Daily'][1:].reset_index(drop=True))

        # Creating train size
        train_size_int = int(self.X.shape[0] *  TRAIN_SIZE // 1)

        self.X_train = self.X.iloc[:train_size_int, :]
        self.X_test = self.X.iloc[train_size_int - TRAIN_PERIOD:, :]
        self.y_train = self.y.iloc[:train_size_int]
        self.y_test = self.y.iloc[train_size_int - TRAIN_PERIOD:]

    def in_sample(self):
        return super().in_sample(self.X_train, self.y_train)

    def residuals(self):
        return super().residuals()

    def in_sample_predict(self):
        return super().in_sample_predict(self.X_train)

    def out_sample_eval(self):
        return super().out_sample_eval(self.X_test, self.y_test)

    def loss(self, metric: str):
        return super().loss(self.y_test[365:], metric=metric) 

class HAR_J(Model):
    def __init__(self, data, resolutions):
        self.data = data
        self.resolutions = resolutions
        self.n_features = 4

        self.rv = generate_realized_volatility(
            data = self.data,
            resolutions = self.resolutions,
            feature = "Close"
        )

        self.returns = generate_log_returns(
            data = self.data,
            feature = "Close",
            resolution = "5T"
        )

        # Setting up data needed.
        temp = pd.DataFrame()
        temp["Daily"] = self.rv
        temp["Weekly"] = generate_rolling_realized_variance(
            temp["Daily"], roll=7
        )
        temp["Monthly"] = generate_rolling_realized_variance(
            temp["Daily"], roll=30
        )
        temp = temp[1:]
        temp["Jumps"] = generate_squared_jumps(self.returns, self.rv)

        temp = temp.iloc[29:, :].reset_index(drop=True)

        # Setting up the X_train, X_test, y_train, y_test
        self.X = sm.add_constant(temp.iloc[:-1, :].reset_index(drop=True))
        self.y = temp['Daily'][1:].reset_index(drop=True)

        # Creating train size
        train_size_int = int(self.X.shape[0] * TRAIN_SIZE // 1)

        self.X_train = self.X.iloc[:train_size_int, :]
        self.X_test = self.X.iloc[train_size_int - TRAIN_PERIOD:, :]
        self.y_train = self.y.iloc[:train_size_int]
        self.y_test = self.y.iloc[train_size_int - TRAIN_PERIOD:]

    def in_sample(self):
        return super().in_sample(self.X_train, self.y_train)

    def residuals(self):
        return super().residuals()

    def in_sample_predict(self):
        return super().in_sample_predict(self.X_train)

    def out_sample_eval(self):
        return super().out_sample_eval(self.X_test, self.y_test, self.n_features)

    def loss(self, metric: str):
        return super().loss(self.y_test[365:], metric)

class HAR_RSI(Model):
    def __init__(self, data, resolutions):
        self.data = data
        self.resolutions = resolutions
        self.n_features = 5

        self.rv = generate_realized_volatility(
            data = self.data,
            resolutions = self.resolutions,
            feature = "Close"
        )

        self.daily_returns = generate_log_returns(
            data = self.data,
            feature = "Close",
            resolution = "1D"
        )

        # Setting up data needed.
        temp = pd.DataFrame()
        temp["Daily"] = self.rv
        temp["Weekly"] = generate_rolling_realized_variance(
            temp["Daily"], roll=7
        )
        temp["Monthly"] = generate_rolling_realized_variance(
            temp["Daily"], roll=30
        )
        temp = temp[1:]

        temp["RS+"] = generate_semi_variance(
            self.data,
            "5T",
            "positive"
        )
        temp["RS-"] = generate_semi_variance(
            self.data,
            "5T",
            "negative"
        )
        temp = temp.iloc[29:, :]

        # Setting up the X_train, X_test, y_train, y_test
        self.X = sm.add_constant(temp.iloc[:-1, 1:].reset_index(drop=True))
        self.y = temp['Daily'][1:].reset_index(drop=True)

        # Creating train size
        train_size_int = int(self.X.shape[0] * TRAIN_SIZE // 1)

        self.X_train = self.X.iloc[:train_size_int, :]
        self.X_test = self.X.iloc[train_size_int - TRAIN_PERIOD:, :]
        self.y_train = self.y.iloc[:train_size_int]
        self.y_test = self.y.iloc[train_size_int - TRAIN_PERIOD:]

    def in_sample(self):
        return super().in_sample(self.X_train, self.y_train)

    def residuals(self):
        return super().residuals()

    def in_sample_predict(self):
        return super().in_sample_predict(self.X_train)

    def out_sample_eval(self):
        return super().out_sample_eval(self.X_test, self.y_test, self.n_features)

    def loss(self, metric: str):
        return super().loss(self.y_test[365:], metric)


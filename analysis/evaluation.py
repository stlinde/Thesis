# analysis/evaluation.py
"""
This module will implement utilities for evaluating out of sample forecasts.
"""
import pandas as pd
import numpy as np

def compute_mse(target, predictions):
    """Computes the mean squared error.
    :param target:        list - The list containing the targets.
    :param predictions:   list - The list containing the predicted values.
    """
    errors = np.square(predictions - target)
    return np.mean(errors)

def compute_mae(target, predictions):
    """Computes the mean squared error.
    :param target:        list - The list containing the targets.
    :param predictions:   list - The list containing the predicted values.
    """
    errors = np.abs(predictions - target)
    return np.mean(errors)

def compute_mase(target, predictions):
    """Computes the mean squared error.
    :param target:        list - The list containing the targets.
    :param predictions:   list - The list containing the predicted values.
    """
    

def compute_mape(target, predictions):
    """Computes the mean squared error.
    :param target:        list - The list containing the targets.
    :param predictions:   list - The list containing the predicted values.
    """
    errors = [np.abs((target[i] - predictions[i])/target[i])
        for i in range(len(target))]
    return np.mean(errors)

def compute_qlike(target, predictions):
    """Computes the mean squared error.
    :param target:        list - The list containing the targets.
    :param predictions:   list - The list containing the predicted values.
    """
    errors = np.log(predictions) + target / predictions
    errors = errors.dropna()
    # errors = [(np.log(predictions[i]) + target[i] / predictions[i])
    #     for i in range(len(target))]
    return np.mean(errors)


def evaluation_table(model, name: str):
    y = model.y_train
    preds = model.in_sample_predict()

    mse = compute_mse(y, preds)
    mae = compute_mae(y, preds)
    qlike = compute_qlike(y, preds)
    losses = [mse, mae, qlike]

    table = pd.DataFrame(columns=[name]) 
    table["Criterion"] = ["MSE", "MAE", "QLIKE"]
    table[name] = losses

    table = table.set_index("Criterion")

    return table


def evaluation_table_out(model, name: str):
    mse = model.loss("MSE")
    mae = model.loss("MAE")
    qlike = model.loss("QLIKE")

    losses = [mse, mae, qlike]

    table = pd.DataFrame(columns=[name]) 
    table["Criterion"] = ["MSE", "MAE", "QLIKE"]
    table[name] = losses

    table = table.set_index("Criterion")

    return table

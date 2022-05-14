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
    errors = [(predictions[i] - target[i])**2 for i in range(len(target))]
    return np.mean(errors)

def compute_mae(target, predictions):
    """Computes the mean squared error.
    :param target:        list - The list containing the targets.
    :param predictions:   list - The list containing the predicted values.
    """
    errors = [np.abs(predictions[i] - target[i]) for i in range(len(target))]
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
    errors = [(np.log(predictions[i]) + target[i] / predictions[i])
        for i in range(len(target))]
    return np.mean(errors)

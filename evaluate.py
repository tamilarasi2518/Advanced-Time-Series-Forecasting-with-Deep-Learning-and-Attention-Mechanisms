"""evaluate.py
Metrics and visualization helpers for model comparison.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true.ravel(), y_pred.ravel())

def summarize(y_true, y_pred):
    return {'rmse': rmse(y_true, y_pred), 'mae': mae(y_true, y_pred)}

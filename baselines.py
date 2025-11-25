"""baselines.py
Implements simple baselines: SARIMAX (statsmodels) and standard LSTM (PyTorch).
"""
import numpy as np
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception as e:
    SARIMAX = None
import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, horizon=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim * horizon)
        self.horizon = horizon
    def forward(self, x):
        out, _ = self.lstm(x)
        # use last output
        last = out[:, -1, :]
        preds = self.fc(last)
        batch = preds.size(0)
        preds = preds.view(batch, self.horizon, -1)
        return preds

def sarimax_forecast(series, order=(1,0,0), seasonal_order=(0,0,0,0), steps=1):
    if SARIMAX is None:
        raise RuntimeError("statsmodels SARIMAX not available. Install statsmodels to use.")
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=steps).predicted_mean
    return np.array(pred)

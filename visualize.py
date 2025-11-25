"""visualize.py
Plots predictions and attention weights.
Uses matplotlib and expects numpy arrays.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_series(true, pred, title='Forecast vs True'):
    # true: (N, horizon, features) or (N,features), take first sample
    true_s = true[0].squeeze()
    pred_s = pred[0].squeeze()
    plt.figure(figsize=(8,3))
    plt.plot(true_s, label='true')
    plt.plot(pred_s, label='pred', linestyle='--')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_attention(weights):
    # weights: (batch, seq_len)
    w = weights[0]
    plt.figure(figsize=(8,2))
    plt.bar(range(len(w)), w)
    plt.title('Attention weights (first sample)')
    plt.tight_layout()
    plt.show()

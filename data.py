"""data.py
Utilities for dataset acquisition and preprocessing.
Supports: 1) synthetic dataset generator, 2) loading CSV file with timestamp column.
Exposes get_datasets(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seq_len=24)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def generate_synthetic(n_series=1, n_steps=500, seed=0):
    np.random.seed(seed)
    t = np.arange(n_steps)
    data = 10*np.sin(2 * np.pi * t / 24) + 0.1 * t + np.random.normal(0, 1, (n_steps, n_series))
    return pd.DataFrame(data, columns=[f"series_{i}" for i in range(n_series)])

def load_csv(path: str, timestamp_col: str=None, value_cols=None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[timestamp_col] if timestamp_col else None)
    if timestamp_col:
        df = df.set_index(timestamp_col)
    if value_cols:
        df = df[value_cols]
    return df

def create_sequences(df: pd.DataFrame, seq_len: int=24, horizon: int=1):
    X, y = [], []
    arr = df.values
    for i in range(len(arr) - seq_len - horizon + 1):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len:i+seq_len+horizon])
    return np.array(X), np.array(y)

def scale_split(df: pd.DataFrame, seq_len:int=24, horizon:int=1,
                train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))
    scaler = StandardScaler()
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_val]
    test = df.iloc[n_val:]
    scaler.fit(train)
    train_s = pd.DataFrame(scaler.transform(train), index=train.index, columns=train.columns)
    val_s = pd.DataFrame(scaler.transform(val), index=val.index, columns=val.columns)
    test_s = pd.DataFrame(scaler.transform(test), index=test.index, columns=test.columns)
    X_train, y_train = create_sequences(train_s, seq_len, horizon)
    X_val, y_val = create_sequences(val_s, seq_len, horizon)
    X_test, y_test = create_sequences(test_s, seq_len, horizon)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler

"""train.py
Training loop for AttentionLSTM with basic logging and checkpoint saving.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
import os
from model import AttentionLSTM
from data import generate_synthetic, scale_split
from typing import Tuple

def to_tensor(x):
    return torch.from_numpy(x).float()

def train(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cpu', ckpt_path='ckpt.pt'):
    model.to(device)
    optimz = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    best_val = float('inf')
    for ep in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimz.zero_grad()
            preds, _ = model(xb)
            loss = criterion(preds, yb.squeeze(1))
            loss.backward()
            optimz.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        val_loss = evaluate(model, val_loader, device=device)
        print(f"Epoch {ep}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model_state': model.state_dict()}, ckpt_path)
    return model

def evaluate(model, loader, device='cpu'):
    model.eval()
    criterion = torch.nn.MSELoss()
    tot = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds, _ = model(xb)
            loss = criterion(preds, yb.squeeze(1))
            tot += loss.item() * xb.size(0)
    return tot / len(loader.dataset)

if __name__ == '__main__':
    # quick demo: synthetic data
    df = generate_synthetic(n_series=1, n_steps=600)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = scale_split(df, seq_len=24, horizon=1)
    # ensure shapes: X=(N,seq,features), y=(N,horizon,features)
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], -1)
    y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], -1)
    train_ds = TensorDataset(to_tensor(X_train), to_tensor(y_train))
    val_ds = TensorDataset(to_tensor(X_val), to_tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    model = AttentionLSTM(input_dim=X_train.shape[-1], hidden_dim=64, num_layers=1, attention_dim=32, horizon=1)
    trained = train(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cpu', ckpt_path='best.pt')

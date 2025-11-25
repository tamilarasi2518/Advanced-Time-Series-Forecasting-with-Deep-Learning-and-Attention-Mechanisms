"""model.py
PyTorch implementation of an LSTM with an attention mechanism.
The AttentionLSTM expects input shape: (batch, seq_len, features)
and outputs predictions of shape (batch, horizon, features).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, attention_dim=32, horizon=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attn_linear = nn.Linear(hidden_dim, attention_dim)
        self.attn_score = nn.Linear(attention_dim, 1, bias=False)
        self.fc = nn.Linear(hidden_dim, input_dim * horizon)
        self.horizon = horizon

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, (h_n, c_n) = self.lstm(x)  # out: (batch, seq_len, hidden_dim)
        # attention weights over time steps
        attn_proj = torch.tanh(self.attn_linear(out))  # (batch, seq_len, attn_dim)
        scores = self.attn_score(attn_proj).squeeze(-1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        context = torch.sum(out * weights, dim=1)  # (batch, hidden_dim)
        # final projection to predict horizon steps for each feature
        preds = self.fc(context)  # (batch, input_dim * horizon)
        batch = preds.size(0)
        preds = preds.view(batch, self.horizon, -1)  # (batch, horizon, input_dim)
        return preds, weights.squeeze(-1)

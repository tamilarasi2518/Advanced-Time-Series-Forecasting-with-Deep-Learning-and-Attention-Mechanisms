import torch
import torch.nn as nn

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, attn_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attn = nn.Linear(hidden_dim, attn_dim)
        self.context = nn.Linear(attn_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        a = torch.tanh(self.attn(out))
        weights = torch.softmax(self.context(a), dim=1)
        context = (weights * out).sum(dim=1)
        return self.fc(context), weights

import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        self.output_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        _, (hn, _) = self.encoder(x)
        # hn shape: (1, batch, hidden_dim)

        # Repeat hidden state for decoder
        seq_len = x.size(1)
        # We need to broadcast or repeat hn to (batch, seq_len, hidden_dim)
        # But decoder expects a sequence.
        # Simple version: use hidden state as input to decoder at each step
        decoder_input = hn.permute(1, 0, 2).repeat(1, seq_len, 1)
        out, _ = self.decoder(decoder_input)
        return out


def train_lstm_cleaner(series, epochs=50, hidden_dim=16, seq_len=10):
    """
    Trains an LSTM Autoencoder on a single series and returns the cleaned series.
    """
    s = series.ffill().bfill().values
    mean, std = s.mean(), s.std()
    s_norm = (s - mean) / (std + 1e-9)

    # Create windows
    windows = []
    for i in range(len(s_norm) - seq_len + 1):
        windows.append(s_norm[i : i + seq_len])

    if not windows:
        return series

    data = torch.FloatTensor(np.array(windows)).unsqueeze(-1)

    model = LSTMAutoencoder(1, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

    # Reconstruct
    model.eval()
    with torch.no_grad():
        reconstructed = model(data).squeeze(-1).numpy()

    # Take the last value of each window for the full series reconstruction
    # (except the first few values)
    res = np.zeros_like(s_norm)
    res[: seq_len - 1] = s_norm[: seq_len - 1]  # First few values stay same
    res[seq_len - 1 :] = reconstructed[:, -1]

    # Unnormalize
    res = res * std + mean
    return pd.Series(res, index=series.index)

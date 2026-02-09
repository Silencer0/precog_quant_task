import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def gan_imputation(df, col="Close", epochs=100, hidden_dim=32):
    """
    Simple GAN for filling NaNs in a series.
    Learns from non-NaN segments and generates values for NaNs.
    """
    series = df[col].copy()
    s = series.values

    # Simple strategy: train on segments without NaNs
    valid_mask = ~np.isnan(s)
    if valid_mask.sum() < 20:
        return df.fillna(0)

    mean, std = np.nanmean(s), np.nanstd(s)
    s_norm = (s - mean) / (std + 1e-9)

    # We use segments of length 10 as vectors
    seg_len = 10
    real_data = []
    for i in range(len(s_norm) - seg_len + 1):
        seg = s_norm[i : i + seg_len]
        if not np.isnan(seg).any():
            real_data.append(seg)

    if not real_data:
        # Fallback if no full valid segments
        return df.ffill().bfill()

    real_data = torch.FloatTensor(np.array(real_data))

    gen = Generator(seg_len, hidden_dim)
    disc = Discriminator(seg_len, hidden_dim)

    d_optimizer = torch.optim.Adam(disc.parameters(), lr=0.001)
    g_optimizer = torch.optim.Adam(gen.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        # Train Discriminator
        d_optimizer.zero_grad()
        real_labels = torch.ones(real_data.size(0), 1)
        fake_labels = torch.zeros(real_data.size(0), 1)

        real_output = disc(real_data)
        d_loss_real = criterion(real_output, real_labels)

        noise = torch.randn(real_data.size(0), seg_len)
        fake_data = gen(noise)
        fake_output = disc(fake_data.detach())
        d_loss_fake = criterion(fake_output, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()
        fake_output = disc(fake_data)
        g_loss = criterion(fake_output, real_labels)
        g_loss.backward()
        g_optimizer.step()

    # Use Generator to fill NaNs
    # In this simple version, we just generate random 'stock-like' noise
    # and blend it. A more advanced version would use conditional GAN.
    # For this research, we'll generate replacements for segments with NaNs.
    df_clean = df.copy()
    s_clean = s_norm.copy()

    for i in range(0, len(s_clean), seg_len):
        seg = s_clean[i : i + seg_len]
        if np.isnan(seg).any():
            # Generate replacement
            with torch.no_grad():
                noise = torch.randn(1, seg_len)
                replacement = gen(noise).numpy().flatten()
                # Blend: keep valid values, fill NaNs
                # (Need to handle end of series if len % seg_len != 0)
                actual_len = min(seg_len, len(s_clean) - i)
                for j in range(actual_len):
                    if np.isnan(s_clean[i + j]):
                        s_clean[i + j] = replacement[j]

    df_clean[col] = s_clean * std + mean
    return df_clean

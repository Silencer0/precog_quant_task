from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PairDiagnostics:
    hedge_ratio: float
    intercept: float
    spread_mean: float
    spread_std: float
    half_life: float | None
    zscore: pd.Series
    spread: pd.Series


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Return (intercept, beta) for y = a + b x."""
    X = np.column_stack([np.ones_like(x), x])
    coeff, *_ = np.linalg.lstsq(X, y, rcond=None)
    a = float(coeff[0])
    b = float(coeff[1])
    return a, b


def _half_life(spread: pd.Series) -> float | None:
    s = spread.dropna()
    if len(s) < 30:
        return None
    y = s.diff().dropna()
    x = s.shift(1).dropna().loc[y.index]
    if len(x) < 30:
        return None
    # y = k * x + eps
    x_np = x.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float)
    k = float(np.dot(x_np, y_np) / (np.dot(x_np, x_np) + 1e-12))
    if k >= 0:
        return None
    hl = float(np.log(2) / (-k))
    return hl


def compute_pair_diagnostics(
    *,
    y: pd.Series,
    x: pd.Series,
    zscore_window: int = 60,
) -> PairDiagnostics:
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    y_np = df["y"].to_numpy(dtype=float)
    x_np = df["x"].to_numpy(dtype=float)
    a, b = _ols(y_np, x_np)
    spread = df["y"] - (a + b * df["x"])

    mean = spread.rolling(zscore_window).mean()
    std = spread.rolling(zscore_window).std(ddof=1)
    z = (spread - mean) / (std + 1e-12)

    return PairDiagnostics(
        hedge_ratio=b,
        intercept=a,
        spread_mean=float(spread.mean()),
        spread_std=float(spread.std(ddof=1)) if len(spread) > 1 else 0.0,
        half_life=_half_life(spread),
        zscore=z.rename("zscore"),
        spread=spread.rename("spread"),
    )


@dataclass(frozen=True)
class PairTradingModel:
    """Statistical arbitrage (pairs) model producing weights for two assets.

    Signals are based on spread z-score.
    """

    entry_z: float = 2.0
    exit_z: float = 0.5
    zscore_window: int = 60
    gross_exposure: float = 1.0

    def compute_weights(
        self,
        *,
        close_y: pd.Series,
        close_x: pd.Series,
    ) -> pd.DataFrame:
        diag = compute_pair_diagnostics(
            y=close_y, x=close_x, zscore_window=self.zscore_window
        )
        z = diag.zscore

        pos = pd.Series(0.0, index=z.index)
        current = 0.0
        for dt, zt in z.items():
            if np.isnan(zt):
                pos.loc[dt] = current
                continue
            if current == 0.0:
                if zt >= self.entry_z:
                    current = -1.0  # short spread
                elif zt <= -self.entry_z:
                    current = 1.0  # long spread
            else:
                if abs(zt) <= self.exit_z:
                    current = 0.0
            pos.loc[dt] = current

        # Spread = y - (a + b x)
        # Long spread: +1 y, -b x
        # Short spread: -1 y, +b x
        w_y = pos * (self.gross_exposure / 2.0)
        w_x = -pos * diag.hedge_ratio * (self.gross_exposure / 2.0)
        w = pd.concat([w_y.rename("Y"), w_x.rename("X")], axis=1)

        gross = w.abs().sum(axis=1).replace(0.0, np.nan)
        w = w.div(gross, axis=0).fillna(0.0)
        return w


def plot_pair_diagnostics(
    *, diag: PairDiagnostics, title: str = "Pair Diagnostics"
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(diag.spread.index, diag.spread.values, label="Spread")
    axes[0].axhline(diag.spread_mean, color="black", linewidth=1, linestyle="--")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(diag.zscore.index, diag.zscore.values, label="Z-Score")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

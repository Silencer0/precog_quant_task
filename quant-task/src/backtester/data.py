from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import pandas as pd


@dataclass(frozen=True)
class AssetData:
    symbol: str
    ohlcv: pd.DataFrame


def _default_cleaned_dir() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../dataset/cleaned")
    )


def load_cleaned_assets(
    *,
    symbols: Iterable[str] | None = None,
    cleaned_dir: str | None = None,
) -> Dict[str, pd.DataFrame]:
    """Load OHLCV CSVs from `dataset/cleaned/`.

    This backtester is intentionally offline-first: it reads local CSV files and does
    not fetch live market data.

    Expected schema per file:
    - filename: e.g. `Asset_001.csv`
    - columns: Date, Open, High, Low, Close, Volume
    - Date parses as datetime

    Returns:
    - dict[symbol, df] where df.index is Date and columns are OHLCV
    """
    data_dir = cleaned_dir or _default_cleaned_dir()
    if symbols is None:
        files = sorted(
            f
            for f in os.listdir(data_dir)
            if f.startswith("Asset_") and f.endswith(".csv")
        )
        symbols = [f.removesuffix(".csv") for f in files]

    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = os.path.join(data_dir, f"{sym}.csv")
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        out[sym] = df
    return out


def align_close_prices(assets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build an aligned Close price matrix (index=date, columns=symbol)."""
    closes: list[pd.DataFrame] = []
    for sym, df in assets.items():
        one: pd.DataFrame = df.loc[:, ["Close"]].copy()
        one.columns = [sym]
        closes.append(one)
    close_df = pd.concat(closes, axis=1).sort_index()
    return close_df

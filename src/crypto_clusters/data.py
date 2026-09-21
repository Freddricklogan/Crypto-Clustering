"""Loading and scaling the price-change table. The dataset is the 41-coin CoinGecko snapshot
distributed with the unsupervised-learning exercise; the schema is validated on load."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

FEATURES = (
    "price_change_percentage_24h",
    "price_change_percentage_7d",
    "price_change_percentage_14d",
    "price_change_percentage_30d",
    "price_change_percentage_60d",
    "price_change_percentage_200d",
    "price_change_percentage_1y",
)
DATA_SOURCE = (
    "CoinGecko price-change percentages for 41 coins over seven horizons (24h to 1y), as "
    "distributed with the Crypto Clustering exercise; vendored in data/crypto_market_data.csv"
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Table:
    coins: list[str]
    raw: FloatArray  # rows = coins, columns = FEATURES
    scaled: FloatArray


def load(path: Path) -> Table:
    df = pd.read_csv(path, index_col="coin_id")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        msg = f"dataset is missing columns: {', '.join(missing)}"
        raise ValueError(msg)
    df = df[list(FEATURES)]
    if df.isna().any().any():
        msg = "dataset has missing values"
        raise ValueError(msg)
    if len(df) < 3:
        msg = "dataset needs at least 3 coins"
        raise ValueError(msg)
    if df.index.duplicated().any():
        msg = "duplicate coin_id"
        raise ValueError(msg)
    raw = df.to_numpy(dtype=np.float64)
    scaled = np.asarray(StandardScaler().fit_transform(raw), dtype=np.float64)
    return Table(coins=[str(c) for c in df.index], raw=raw, scaled=scaled)

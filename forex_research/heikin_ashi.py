from __future__ import annotations

import numpy as np
import pandas as pd


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"heikin_ashi requires columns {sorted(required)}; missing {sorted(missing)}")

    o = df["open"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    lo = df["low"].to_numpy(dtype="float64")
    c = df["close"].to_numpy(dtype="float64")
    n = len(df)

    ha_close = (o + h + lo + c) / 4.0
    ha_open = np.empty(n, dtype="float64")
    if n:
        ha_open[0] = (o[0] + c[0]) / 2.0
        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([lo, ha_open, ha_close])

    return pd.DataFrame(
        {"ha_open": ha_open, "ha_high": ha_high, "ha_low": ha_low, "ha_close": ha_close},
        index=df.index,
    )


def price_series(df: pd.DataFrame, price_basis: str) -> pd.Series:
    if price_basis == "standard":
        return df["close"]
    if price_basis == "heikin_ashi":
        return df["ha_close"]
    raise ValueError(f"Unknown price_basis: {price_basis!r}")


def ohlc_columns(price_basis: str) -> tuple[str, str, str, str]:
    if price_basis == "standard":
        return ("open", "high", "low", "close")
    if price_basis == "heikin_ashi":
        return ("ha_open", "ha_high", "ha_low", "ha_close")
    raise ValueError(f"Unknown price_basis: {price_basis!r}")

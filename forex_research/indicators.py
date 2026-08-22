from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pandas as pd


@contextlib.contextmanager
def quiet_warmup_warnings():
    with warnings.catch_warnings():
        for msg in ("All-NaN slice encountered", "All-NaN axis encountered",
                    "Mean of empty slice", "Degrees of freedom <= 0"):
            warnings.filterwarnings("ignore", message=msg, category=RuntimeWarning)
        yield


def ema(src: pd.Series, period: int) -> pd.Series:
    if period < 1:
        raise ValueError(f"ema period must be >= 1, got {period}")
    return src.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(src: pd.Series, period: int) -> pd.Series:
    if period < 1:
        raise ValueError(f"sma period must be >= 1, got {period}")
    return src.rolling(period, min_periods=period).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = true_range(high, low, close)
    out = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return out.where(out > 0)


def slope(series: pd.Series, lookback: int) -> pd.Series:
    if lookback < 1:
        raise ValueError(f"slope lookback must be >= 1, got {lookback}")
    return (series - series.shift(lookback)) / float(lookback)


def realized_volatility(close: pd.Series, window: int) -> pd.Series:
    logret = np.log(close / close.shift(1))
    return logret.rolling(window, min_periods=window).std()


def rolling_percentile(series: pd.Series, lookback: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods if min_periods is not None else max(2, lookback // 10)
    return series.rolling(lookback, min_periods=mp).rank(pct=True)


def consecutive_true(flag: pd.Series) -> pd.Series:
    b = flag.astype("float64")
    isna = b.isna()
    truth = b.fillna(0.0) > 0.5
    # group id increments whenever the run breaks
    grp = (~truth).cumsum()
    counts = truth.groupby(grp).cumsum()
    out = counts.where(truth, 0.0)
    return out.mask(isna)


def bars_since(flag: pd.Series) -> pd.Series:
    truth = flag.fillna(False).astype(bool)
    idx = np.arange(len(truth), dtype="float64")
    last = pd.Series(np.where(truth, idx, np.nan), index=truth.index).ffill()
    return pd.Series(idx, index=truth.index) - last


def safe_divide(num, den):
    den_arr = np.asarray(den, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.asarray(num, dtype="float64") / np.where(
            (den_arr == 0) | ~np.isfinite(den_arr), np.nan, den_arr
        )
    return out


def sign(series: pd.Series) -> pd.Series:
    return np.sign(series).where(series.notna())

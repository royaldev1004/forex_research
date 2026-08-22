from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import ResearchConfig, timeframe_minutes
from .heikin_ashi import heikin_ashi
from .logging_utils import get_logger

log = get_logger("timeframe_builder")


@dataclass(frozen=True)
class TimeframeSeries:

    timeframe: str
    minutes: int
    frame: pd.DataFrame
    n_incomplete_dropped: int
    n_bins_with_missing_base_bars: int


def _resample_ohlc(base: pd.DataFrame, minutes: int) -> pd.DataFrame:
    idx = base.set_index("bar_open_time")
    agg = (
        idx.resample(f"{minutes}min", label="left", closed="left", origin="start_day")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            tick_volume=("tick_volume", "sum"),
            spread_points_mean=("spread_points", "mean"),
            spread_points_last=("spread_points", "last"),
            n_base_bars=("close", "count"),
        )
        .reset_index()
        .rename(columns={"bar_open_time": "bar_open_time"})
    )
    # Empty bins (weekends, holidays) carry no price information at all.
    return agg[agg["n_base_bars"] > 0].reset_index(drop=True)


def build_timeframe(
    base: pd.DataFrame, timeframe: str, cfg: ResearchConfig
) -> TimeframeSeries:
    minutes = timeframe_minutes(timeframe)
    base_minutes = timeframe_minutes(cfg.base_timeframe)

    if minutes == base_minutes:
        df = base.copy()
        df["n_base_bars"] = 1
        df["spread_points_mean"] = df["spread_points"].astype("float64")
        df["spread_points_last"] = df["spread_points"].astype("float64")
        df = df[
            [
                "bar_open_time", "open", "high", "low", "close",
                "tick_volume", "spread_points_mean", "spread_points_last", "n_base_bars",
            ]
        ].copy()
    else:
        df = _resample_ohlc(base, minutes)

    df["bar_close_time"] = df["bar_open_time"] + pd.Timedelta(minutes=minutes)

    # Drop bins whose nominal close is after the end of observed base data.
    last_base_close = base["bar_close_time"].max()
    incomplete = df["bar_close_time"] > last_base_close
    n_incomplete = int(incomplete.sum())
    if n_incomplete:
        df = df[~incomplete].reset_index(drop=True)

    expected_base_bars = minutes // base_minutes
    n_partial = int((df["n_base_bars"] < expected_base_bars).sum())
    df["expected_base_bars"] = expected_base_bars
    df["base_bar_coverage"] = df["n_base_bars"] / float(expected_base_bars)

    ha = heikin_ashi(df)
    df = pd.concat([df, ha], axis=1)

    df["bar_index"] = range(len(df))
    df = df.reset_index(drop=True)

    if not df["bar_close_time"].is_monotonic_increasing:
        raise RuntimeError(f"{timeframe}: bar_close_time is not monotonic increasing.")

    log.info(
        "Built %-3s : %6d bars | %s -> %s | dropped %d incomplete tail bin(s) | "
        "%d bin(s) with missing base bars",
        timeframe, len(df),
        df["bar_open_time"].iloc[0] if len(df) else "n/a",
        df["bar_close_time"].iloc[-1] if len(df) else "n/a",
        n_incomplete, n_partial,
    )
    return TimeframeSeries(
        timeframe=timeframe,
        minutes=minutes,
        frame=df,
        n_incomplete_dropped=n_incomplete,
        n_bins_with_missing_base_bars=n_partial,
    )


def build_all_timeframes(
    base: pd.DataFrame, cfg: ResearchConfig
) -> dict[str, TimeframeSeries]:
    out: dict[str, TimeframeSeries] = {}
    for tf in cfg.active_timeframes:
        out[tf] = build_timeframe(base, tf, cfg)
    return out


def coverage_report(
    timeframes: dict[str, TimeframeSeries], cfg: ResearchConfig
) -> pd.DataFrame:
    warmup = cfg.warmup_bars()
    rows = []
    for tf, ts in timeframes.items():
        df = ts.frame
        n = len(df)
        first_valid_idx = warmup if n > warmup else None
        rows.append(
            {
                "timeframe": tf,
                "minutes_per_bar": ts.minutes,
                "n_bars": n,
                "first_bar_open_native": str(df["bar_open_time"].iloc[0]) if n else "",
                "last_bar_close_native": str(df["bar_close_time"].iloc[-1]) if n else "",
                "warmup_bars_required": warmup,
                "n_bars_after_warmup": max(0, n - warmup),
                "first_valid_bar_close_native": (
                    str(df["bar_close_time"].iloc[first_valid_idx])
                    if first_valid_idx is not None else ""
                ),
                "warmup_satisfied": n > warmup,
                "n_incomplete_tail_bins_dropped": ts.n_incomplete_dropped,
                "n_bins_with_missing_base_bars": ts.n_bins_with_missing_base_bars,
                "median_base_bar_coverage": (
                    float(df["base_bar_coverage"].median()) if n else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ResearchConfig, timeframe_minutes
from .data_loader import LoadedSeries, epoch_seconds
from .logging_utils import get_logger

log = get_logger("data_quality")


@dataclass
class Check:

    source: str
    timeframe: str
    check: str
    value: Any
    status: str  # "ok" | "info" | "warn" | "fail"
    detail: str = ""


def _weekday_expected_grid(df: pd.DataFrame, tf_minutes: int) -> pd.DatetimeIndex:
    grid = pd.date_range(
        df["bar_open_time"].iloc[0],
        df["bar_open_time"].iloc[-1],
        freq=f"{tf_minutes}min",
    )
    return grid[grid.dayofweek < 5]


def audit_series(series: LoadedSeries, cfg: ResearchConfig) -> list[Check]:
    df = series.frame
    name = series.path.name
    tf = series.timeframe
    tf_min = timeframe_minutes(tf)
    checks: list[Check] = []

    def add(check: str, value: Any, status: str, detail: str = "") -> None:
        checks.append(Check(name, tf, check, value, status, detail))

    add("row_count", len(df), "info")
    add("raw_row_count", series.raw_row_count, "info")
    add("sha256", series.sha256, "info")
    add("first_bar_open_native", str(df["bar_open_time"].iloc[0]), "info")
    add("last_bar_open_native", str(df["bar_open_time"].iloc[-1]), "info")
    add("last_bar_close_native", str(df["bar_close_time"].iloc[-1]), "info")
    add("first_bar_open_utc", str(df["bar_open_time_utc"].iloc[0]), "info")
    add("last_bar_close_utc", str(df["bar_close_time_utc"].iloc[-1]), "info")
    add("timestamp_semantics", cfg.timestamp_semantics, "info",
        "Established empirically; see m1_to_m5_aggregation_* checks.")
    add("source_timezone_mode", cfg.source_timezone_mode, "info", cfg.source_timezone_note)

    # --- ordering / duplicates -----------------------------------------
    monotonic = bool(df["bar_open_time"].is_monotonic_increasing)
    add("timestamps_sorted", monotonic, "ok" if monotonic else "fail")

    n_dup = int(df["bar_open_time"].duplicated().sum())
    add("duplicate_timestamps", n_dup, "ok" if n_dup == 0 else "fail")

    strictly_increasing = bool(df["bar_open_time"].is_monotonic_increasing and n_dup == 0)
    add("timestamps_strictly_increasing", strictly_increasing,
        "ok" if strictly_increasing else "fail")

    # --- timestamp grid alignment ---------------------------------------
    off_grid = int((epoch_seconds(df["bar_open_time"]) % (tf_min * 60) != 0).sum())
    add("bars_off_timeframe_grid", off_grid, "ok" if off_grid == 0 else "warn",
        f"bar_open_time not aligned to a {tf_min}-minute boundary")

    # --- null / invalid prices -------------------------------------------
    price_cols = ["open", "high", "low", "close"]
    n_null = int(df[price_cols].isna().sum().sum())
    add("null_prices", n_null, "ok" if n_null == 0 else "fail")

    n_nonpos = int((df[price_cols] <= 0).sum().sum())
    add("nonpositive_prices", n_nonpos, "ok" if n_nonpos == 0 else "fail")

    invalid = (
        (df["high"] < df["low"])
        | (df["open"] > df["high"])
        | (df["open"] < df["low"])
        | (df["close"] > df["high"])
        | (df["close"] < df["low"])
    )
    n_invalid = int(invalid.sum())
    add("invalid_ohlc_relationships", n_invalid, "ok" if n_invalid == 0 else "fail")

    add("price_min", float(df["low"].min()), "info")
    add("price_max", float(df["high"].max()), "info")

    # --- bid/ask & spread --------------------------------------------------
    add("bid_ask_columns_present", False, "warn",
        "MT5 bar export carries a single (bid-side) OHLC series, not bid/ask. "
        "True bid/ask tick history is not available in this dataset.")
    add("tick_data_available", False, "warn",
        "No tick file supplied; only M1/M5 OHLC bars.")
    n_spread_null = int(df["spread_points"].isna().sum())
    add("spread_column_present", True, "info", "MT5 <SPREAD> in points")
    add("spread_null_count", n_spread_null, "ok" if n_spread_null == 0 else "warn")
    if n_spread_null < len(df):
        pts_per_pip = cfg.pip_size / cfg.point_size
        add("spread_pips_min", float(df["spread_points"].min() / pts_per_pip), "info")
        add("spread_pips_median", float(df["spread_points"].median() / pts_per_pip), "info")
        add("spread_pips_max", float(df["spread_points"].max() / pts_per_pip), "info")

    n_realvol = int((df["real_volume"].fillna(0) != 0).sum())
    add("real_volume_nonzero_bars", n_realvol, "info",
        "MT5 <VOL> is zero for this feed; only tick volume is usable.")

    # --- timezone conversion sanity ---------------------------------------
    n_bad_tz = int(df["bar_open_time_utc"].isna().sum())
    add("utc_conversion_failures", n_bad_tz, "ok" if n_bad_tz == 0 else "fail",
        "NaT means a bar landed in an ambiguous/non-existent local hour.")

    # --- session structure (timezone evidence) -----------------------------
    gaps = df["bar_open_time"].diff()
    big = gaps > pd.Timedelta(hours=2)
    last_before = df["bar_open_time"].shift(1)[big]
    first_after = df["bar_open_time"][big]
    if len(first_after):
        open_mode = first_after.dt.strftime("%a %H:%M").mode()
        close_mode = last_before.dt.strftime("%a %H:%M").mode()
        add("session_gaps_over_2h", int(big.sum()), "info")
        add("modal_week_open_native", open_mode.iloc[0] if len(open_mode) else "n/a", "info")
        add("modal_week_close_native", close_mode.iloc[0] if len(close_mode) else "n/a", "info")

    # --- missing bars ------------------------------------------------------
    present = pd.DatetimeIndex(df["bar_open_time"])
    expected = _weekday_expected_grid(df, tf_min)
    missing = expected.difference(present)
    add("expected_weekday_slots", len(expected), "info")
    add("missing_weekday_bars", len(missing),
        "ok" if len(missing) == 0 else "warn",
        "Weekend slots excluded; remainder are holidays/feed outages.")
    if len(missing):
        by_date = pd.Series(missing).dt.date.value_counts().sort_values(ascending=False)
        add("missing_bars_distinct_dates", int(by_date.size), "info")
        top = "; ".join(f"{d}={int(c)}" for d, c in by_date.head(6).items())
        add("missing_bars_worst_dates", top, "warn")

    # weekend coverage sanity: there should be (almost) no weekend bars
    n_weekend = int((df["bar_open_time"].dt.dayofweek >= 5).sum())
    add("weekend_bars_present", n_weekend, "ok" if n_weekend == 0 else "info")

    # --- gap inventory -----------------------------------------------------
    intra_week = gaps[(gaps > pd.Timedelta(minutes=tf_min)) & (gaps <= pd.Timedelta(hours=2))]
    add("intraweek_gaps_under_2h", int(intra_week.size), "info" if intra_week.size else "ok")
    add("largest_gap", str(gaps.max()), "info")

    return checks


def verify_timestamp_semantics(
    base: LoadedSeries, reference: LoadedSeries, cfg: ResearchConfig
) -> list[Check]:
    checks: list[Check] = []
    fine, coarse = reference.frame, base.frame
    fine_min = timeframe_minutes(reference.timeframe)
    coarse_min = timeframe_minutes(base.timeframe)
    tag = f"{reference.timeframe}_to_{base.timeframe}"

    if coarse_min % fine_min != 0:
        checks.append(Check(reference.path.name, reference.timeframe,
                            f"aggregation_{tag}", "skipped", "info",
                            "timeframes are not integer multiples"))
        return checks

    ratio = coarse_min // fine_min
    lo = max(fine["bar_open_time"].min(), coarse["bar_open_time"].min())
    hi = min(fine["bar_open_time"].max(), coarse["bar_open_time"].max())
    sub = fine[(fine["bar_open_time"] >= lo) & (fine["bar_open_time"] <= hi)]
    if sub.empty:
        checks.append(Check(reference.path.name, reference.timeframe,
                            f"aggregation_{tag}", "no_overlap", "warn"))
        return checks

    idx = sub.set_index("bar_open_time")
    cols = ["open", "high", "low", "close"]
    results: dict[str, float] = {}
    for label in ("left", "right"):
        agg = (
            idx.resample(f"{coarse_min}min", label=label, closed=label, origin="start_day")
            .agg(open=("open", "first"), high=("high", "max"),
                 low=("low", "min"), close=("close", "last"), n=("close", "count"))
            .reset_index()
        )
        agg = agg[agg["n"] == ratio]
        comp = agg.merge(coarse[["bar_open_time"] + cols], on="bar_open_time",
                         suffixes=("_agg", "_src"))
        if comp.empty:
            results[label] = float("nan")
            continue
        eq = np.all(
            comp[[f"{c}_agg" for c in cols]].to_numpy()
            == comp[[f"{c}_src" for c in cols]].to_numpy(),
            axis=1,
        )
        results[label] = float(eq.mean())
        checks.append(Check(reference.path.name, reference.timeframe,
                            f"aggregation_{tag}_label_{label}_bins", int(len(comp)), "info"))
        checks.append(Check(reference.path.name, reference.timeframe,
                            f"aggregation_{tag}_label_{label}_exact_pct",
                            round(results[label] * 100, 4), "info"))

    left, right = results.get("left", float("nan")), results.get("right", float("nan"))
    if np.isfinite(left) and left >= 0.999 and (not np.isfinite(right) or right < 0.5):
        inferred, status = "bar_open", "ok"
        detail = (f"left-labelled aggregation reproduces {left*100:.2f}% of bins exactly "
                  f"vs {right*100:.2f}% right-labelled")
    elif np.isfinite(right) and right >= 0.999 and (not np.isfinite(left) or left < 0.5):
        inferred, status = "bar_close", "ok"
        detail = "right-labelled aggregation matched exactly"
    else:
        inferred, status = "ambiguous", "fail"
        detail = (f"neither labelling dominated (left={left:.4f}, right={right:.4f}); "
                  "timestamp semantics must be confirmed with the data provider")

    checks.append(Check(reference.path.name, reference.timeframe,
                        "inferred_timestamp_semantics", inferred, status, detail))
    checks.append(Check(reference.path.name, reference.timeframe,
                        "configured_timestamp_semantics_matches_inference",
                        inferred == cfg.timestamp_semantics,
                        "ok" if inferred == cfg.timestamp_semantics else "fail"))
    log.info("Timestamp semantics inference (%s): %s (%s)", tag, inferred, detail)
    return checks


def cross_source_consistency(
    base: LoadedSeries, reference: LoadedSeries, cfg: ResearchConfig
) -> list[Check]:
    b, r = base.frame, reference.frame
    lo = max(b["bar_open_time"].min(), r["bar_open_time"].min())
    hi = min(b["bar_open_time"].max(), r["bar_open_time"].max())
    overlap = hi - lo if hi > lo else pd.Timedelta(0)
    return [
        Check(reference.path.name, reference.timeframe, "overlap_start_native", str(lo), "info"),
        Check(reference.path.name, reference.timeframe, "overlap_end_native", str(hi), "info"),
        Check(reference.path.name, reference.timeframe, "overlap_duration_days",
              round(overlap / pd.Timedelta(days=1), 2), "info",
              "Finer-resolution coverage available for later intrabar/tick research."),
    ]


def build_report(checks: list[Check]) -> pd.DataFrame:
    return pd.DataFrame([c.__dict__ for c in checks])[
        ["source", "timeframe", "check", "value", "status", "detail"]
    ]


def audit_all(loaded: dict[str, LoadedSeries], cfg: ResearchConfig) -> pd.DataFrame:
    checks: list[Check] = []
    for series in loaded.values():
        checks.extend(audit_series(series, cfg))

    base = loaded[cfg.base_timeframe]
    for tf, series in loaded.items():
        if tf == cfg.base_timeframe:
            continue
        if timeframe_minutes(tf) < timeframe_minutes(cfg.base_timeframe):
            checks.extend(verify_timestamp_semantics(base, series, cfg))
            checks.extend(cross_source_consistency(base, series, cfg))

    report = build_report(checks)
    n_fail = int((report["status"] == "fail").sum())
    n_warn = int((report["status"] == "warn").sum())
    log.info("Data-quality audit: %d checks, %d fail, %d warn", len(report), n_fail, n_warn)
    return report

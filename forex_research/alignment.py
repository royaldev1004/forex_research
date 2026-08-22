from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ResearchConfig, timeframe_minutes
from .logging_utils import get_logger

log = get_logger("alignment")


class LookaheadError(AssertionError):
    pass


@dataclass
class Alignment:

    decision_time: pd.Series               # base bar_close_time
    decision_bar_index: np.ndarray         # index into the base timeframe frame
    indices: dict[str, np.ndarray]         # timeframe -> aligned bar index (-1 = none)
    source_open: dict[str, np.ndarray]     # timeframe -> aligned bar open time
    source_close: dict[str, np.ndarray]    # timeframe -> aligned bar close time
    valid: np.ndarray                      # bool: every timeframe warmed up and aligned

    def __len__(self) -> int:
        return len(self.decision_time)


def align_timeframes(
    decision_time: pd.Series,
    timeframe_closes: dict[str, pd.Series],
    cfg: ResearchConfig,
    warmup_bars: int | None = None,
) -> Alignment:
    if warmup_bars is None:
        warmup_bars = cfg.warmup_bars()

    dt = pd.DatetimeIndex(decision_time)
    if not dt.is_monotonic_increasing:
        raise LookaheadError("decision_time must be strictly increasing.")
    dt_v = dt.to_numpy()

    indices: dict[str, np.ndarray] = {}
    src_open: dict[str, np.ndarray] = {}
    src_close: dict[str, np.ndarray] = {}
    valid = np.ones(len(dt_v), dtype=bool)

    for tf, closes in timeframe_closes.items():
        c = pd.DatetimeIndex(closes)
        if not c.is_monotonic_increasing:
            raise LookaheadError(f"{tf}: bar_close_time must be increasing.")
        c_v = c.to_numpy()

        # side="right" then -1 => latest bar with close <= decision_time
        idx = np.searchsorted(c_v, dt_v, side="right") - 1
        indices[tf] = idx

        has = idx >= 0
        # Match the source dtype: pandas 3 uses microsecond resolution by
        # default, older versions nanosecond, and mixing the two silently
        # truncates.
        unit = np.datetime_data(c_v.dtype)[0]
        nat = np.datetime64("NaT", unit)
        so = np.full(len(dt_v), nat, dtype=c_v.dtype)
        sc = np.full(len(dt_v), nat, dtype=c_v.dtype)
        # Close times come from the aligned bar; nominal opens are derived from
        # the timeframe duration.
        sc[has] = c_v[idx[has]]
        so[has] = sc[has] - np.timedelta64(timeframe_minutes(tf) * 60, "s").astype(
            f"timedelta64[{unit}]")
        src_open[tf] = so
        src_close[tf] = sc

        valid &= idx >= warmup_bars

        n_missing = int((~has).sum())
        log.info(
            "%-3s aligned: %d/%d rows have a closed bar (warm-up needs index >= %d)",
            tf, len(dt_v) - n_missing, len(dt_v), warmup_bars,
        )

    log.info("Valid decision rows after warm-up on every timeframe: %d/%d",
             int(valid.sum()), len(valid))

    return Alignment(
        decision_time=pd.Series(dt, name="decision_time"),
        decision_bar_index=np.arange(len(dt_v)),
        indices=indices,
        source_open=src_open,
        source_close=src_close,
        valid=valid,
    )


def assert_no_lookahead(alignment: Alignment) -> pd.DataFrame:
    dt = alignment.decision_time.to_numpy()
    rows = []
    violations_total = 0

    for tf, close in alignment.source_close.items():
        has = ~pd.isna(close)
        if has.any():
            delta = (dt[has] - close[has]) / np.timedelta64(1, "m")
            violations = int((delta < 0).sum())
            worst = float(delta.min())
            median_age = float(np.median(delta))
            max_age = float(delta.max())
        else:
            violations, worst, median_age, max_age = 0, float("nan"), float("nan"), float("nan")

        violations_total += violations
        rows.append(
            {
                "timeframe": tf,
                "rows_checked": int(has.sum()),
                "rows_without_closed_bar": int((~has).sum()),
                "violations_source_close_after_decision": violations,
                "min_age_minutes": worst,
                "median_age_minutes": median_age,
                "max_age_minutes": max_age,
                "status": "ok" if violations == 0 else "fail",
            }
        )

    table = pd.DataFrame(rows)
    if violations_total:
        raise LookaheadError(
            f"{violations_total} alignment rows use a source bar that closed after the "
            "decision time. This is a lookahead bug and the run must not continue."
        )
    log.info("No-lookahead assertion passed for %d timeframes.", len(table))
    return table


def build_alignment_map(alignment: Alignment, cfg: ResearchConfig) -> pd.DataFrame:
    out = pd.DataFrame({"decision_time": alignment.decision_time.to_numpy()})
    out.insert(0, "symbol", cfg.symbol)
    for tf in alignment.indices:
        out[f"{tf}__source_bar_index"] = alignment.indices[tf]
        out[f"{tf}__source_bar_open_time"] = alignment.source_open[tf]
        out[f"{tf}__source_bar_close_time"] = alignment.source_close[tf]
        age = (out["decision_time"].to_numpy() - alignment.source_close[tf]) / np.timedelta64(1, "m")
        out[f"{tf}__source_bar_age_minutes"] = age
    out["feature_row_valid"] = alignment.valid
    return out


def take(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype="float64")
    out = np.full(len(idx), np.nan, dtype="float64")
    ok = idx >= 0
    if ok.any():
        out[ok] = arr[idx[ok]]
    return out

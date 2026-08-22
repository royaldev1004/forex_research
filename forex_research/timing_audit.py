from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import INDICATOR_SOURCE_SHIFT, ResearchConfig, timeframe_minutes
from .logging_utils import get_logger
from .timeframe_builder import TimeframeSeries

log = get_logger("timing_audit")


def build_timing_audit(
    timeframes: dict[str, TimeframeSeries],
    decision_time: pd.Series,
    cfg: ResearchConfig,
    n_examples: int = 12,
    seed: int | None = None,
) -> pd.DataFrame:
    dt = pd.DatetimeIndex(decision_time)
    rng = np.random.default_rng(cfg.random_seed if seed is None else seed)

    warm = cfg.warmup_bars()
    lo = min(len(dt) - 1, warm * max(1, timeframe_minutes(max(
        cfg.active_timeframes, key=timeframe_minutes)) // timeframe_minutes(cfg.base_timeframe)))
    candidates = np.arange(lo, len(dt))
    if len(candidates) == 0:
        candidates = np.arange(len(dt))
    picks = np.sort(rng.choice(candidates, size=min(n_examples, len(candidates)), replace=False))

    rows: list[dict] = []
    for tf in cfg.active_timeframes:
        f = timeframes[tf].frame
        closes = pd.DatetimeIndex(f["bar_close_time"]).to_numpy()
        opens = pd.DatetimeIndex(f["bar_open_time"]).to_numpy()
        for i in picks:
            t = dt[i]
            j = int(np.searchsorted(closes, t.to_datetime64(), side="right") - 1)
            if j < 0:
                continue
            for mode, shift in INDICATOR_SOURCE_SHIFT.items():
                k = j - shift
                if k < 0:
                    continue
                lag = (t - pd.Timestamp(closes[k])).total_seconds() / 60.0
                rows.append({
                    "timeframe": tf,
                    "timeframe_minutes": timeframe_minutes(tf),
                    "indicator_source_mode": mode,
                    "is_active_mode": mode == cfg.indicator_source_mode,
                    "decision_time": t,
                    "selected_source_bar_open": pd.Timestamp(opens[j]),
                    "selected_source_bar_close": pd.Timestamp(closes[j]),
                    "indicator_input_bar_open": pd.Timestamp(opens[k]),
                    "indicator_input_bar_close": pd.Timestamp(closes[k]),
                    "effective_lag_minutes": lag,
                    "source_closed_at_or_before_decision": bool(pd.Timestamp(closes[k]) <= t),
                })
    audit = pd.DataFrame(rows)
    if not audit.empty:
        audit = audit.sort_values(
            ["decision_time", "timeframe_minutes", "indicator_source_mode"]
        ).reset_index(drop=True)
    return audit


def lag_distribution(
    timeframes: dict[str, TimeframeSeries], decision_time: pd.Series, cfg: ResearchConfig
) -> pd.DataFrame:
    dt = pd.DatetimeIndex(decision_time).to_numpy()
    rows: list[dict] = []
    for tf in cfg.active_timeframes:
        closes = pd.DatetimeIndex(timeframes[tf].frame["bar_close_time"]).to_numpy()
        j = np.searchsorted(closes, dt, side="right") - 1
        for mode, shift in INDICATOR_SOURCE_SHIFT.items():
            k = j - shift
            ok = k >= 0
            if not ok.any():
                continue
            lag = (dt[ok] - closes[k[ok]]) / np.timedelta64(1, "m")
            rows.append({
                "timeframe": tf,
                "timeframe_minutes": timeframe_minutes(tf),
                "indicator_source_mode": mode,
                "is_active_mode": mode == cfg.indicator_source_mode,
                "rows": int(ok.sum()),
                "min_lag_minutes": float(lag.min()),
                "median_lag_minutes": float(np.median(lag)),
                "mean_lag_minutes": round(float(lag.mean()), 2),
                "max_lag_minutes": float(lag.max()),
                "any_negative_lag": bool((lag < 0).any()),
            })
    return pd.DataFrame(rows)


def write_timing_note(
    path: Path, cfg: ResearchConfig, audit: pd.DataFrame, dist: pd.DataFrame
) -> None:
    active = cfg.indicator_source_mode
    shift = cfg.indicator_source_shift_bars

    def _tbl(df: pd.DataFrame) -> str:
        if df.empty:
            return "_(none)_\n"
        cols = list(df.columns)
        out = ["| " + " | ".join(str(c) for c in cols) + " |",
               "|" + "|".join("---" for _ in cols) + "|"]
        for r in df.itertuples(index=False):
            out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
        return "\n".join(out) + "\n"

    example_t = audit["decision_time"].iloc[0] if not audit.empty else "n/a"
    ex = audit[audit["decision_time"] == example_t] if not audit.empty else audit
    ex_cols = ["timeframe", "indicator_source_mode", "selected_source_bar_close",
               "indicator_input_bar_close", "effective_lag_minutes"]

    md = f"""# Indicator Timing Note

**Active mode: `{active}` (source shift = {shift} bar).**

## The question

At decision time `T`, which raw candle is the newest one that contributes to each
timeframe's EMA/SMA, and is any of it unnecessarily stale?

## The two interpretations

**Interpretation A - post-close research state (`latest_closed`, shift 0).**
At `T` the latest candle that has *fully closed* may contribute. Point-in-time
alignment already selects exactly that candle: it matches on
`bar_close_time <= T`, and the next bar of that timeframe closes strictly after
`T`.

**Interpretation B - forming-bar / Pine-style state (`previous_closed`, shift 1).**
The indicator additionally drops the aligned bar, so the newest contributing
candle is one full bar older.

## Why the default changed to `latest_closed`

The reference indicator scripts write `src = ha_close[1]`. That offset is not a
statement that research should discard a closed candle - in Pine a script is
evaluated *while a bar is forming*, so `[1]` names the most recent **fully closed**
bar. Its purpose is to exclude the forming bar.

This pipeline already excludes the forming bar structurally, by aligning on
`bar_close_time`. Applying the Pine offset on top of that excludes the forming bar
twice and throws away a candle that was genuinely available at `T`.

The project's governing principle is *what information was objectively available
at the historical decision timestamp*, so `latest_closed` is the research default.

**This is a semantics change, not a bug fix in the leakage sense.** Both modes are
point-in-time safe: each reads a bar that closed at or before `T`. Only staleness
differs. `previous_closed` remains fully supported for anyone who wants the literal
Pine offset layered on top of alignment.

## Worked example

Decision time `{example_t}`:

{_tbl(ex[ex_cols]) if not ex.empty else "_(no examples available)_"}

Under `previous_closed` the 4h indicator at this timestamp is driven by a candle
that closed hours before one which had already been available.

## Effective lag across every decision row

Minutes between the decision timestamp and the close of the newest candle
contributing to that timeframe's indicators:

{_tbl(dist)}

`any_negative_lag` is False everywhere in both modes, which is the no-lookahead
property restated: no contributing candle closes after the decision time.

## Consequence of the change

Switching from `previous_closed` to `latest_closed` removes exactly one bar of
staleness per timeframe. The effect is largest on the slowest timeframe, where one
bar is 240 minutes.

## Configuration

```yaml
indicator_source_mode: latest_closed    # research default
# indicator_source_mode: previous_closed  # literal Pine offset on top of alignment
```

`indicator_source_shift_bars` may be set explicitly only if it agrees with the
mode; a contradiction is a hard configuration error, so the semantics cannot be
changed by editing one field and forgetting the other.

## Full audit data

`indicator_timing_audit.csv` carries the per-timestamp, per-timeframe, per-mode
detail behind this note.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    log.info("Wrote indicator timing note: %s", path)

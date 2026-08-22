from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ResearchConfig, timeframe_minutes
from .data_loader import epoch_seconds
from .logging_utils import get_logger
from .timeframe_builder import TimeframeSeries

log = get_logger("event_quality")


@dataclass(frozen=True)
class TradingCalendar:

    base_minutes: int
    full_closure_dates: frozenset
    present_slots: pd.DatetimeIndex

    @property
    def n_closure_dates(self) -> int:
        return len(self.full_closure_dates)


def build_trading_calendar(base: pd.DataFrame, cfg: ResearchConfig) -> TradingCalendar:
    base_minutes = timeframe_minutes(cfg.base_timeframe)
    opens = pd.DatetimeIndex(base["bar_open_time"])

    grid = pd.date_range(opens.min(), opens.max(), freq=f"{base_minutes}min")
    weekday_grid = grid[grid.dayofweek < 5]

    present_dates = set(pd.Series(opens.date).unique())
    expected_dates = set(pd.Series(weekday_grid.date).unique())
    closure_dates = frozenset(expected_dates - present_dates)

    log.info(
        "Trading calendar: %d weekday dates, %d full-day closures (holidays)",
        len(expected_dates), len(closure_dates),
    )
    return TradingCalendar(
        base_minutes=base_minutes,
        full_closure_dates=closure_dates,
        present_slots=opens,
    )


def tradeable_slots_per_bin(
    bar_open: pd.Series, minutes: int, calendar: TradingCalendar
) -> np.ndarray:
    base_min = calendar.base_minutes
    step = base_min * 60
    bin_seconds = minutes * 60

    lo = epoch_seconds(bar_open).to_numpy()
    # every base slot inside every bin, as a flat grid
    per_bin = bin_seconds // step
    offsets = np.arange(per_bin, dtype="int64") * step
    slots = lo[:, None] + offsets[None, :]

    slot_times = pd.to_datetime(slots.reshape(-1), unit="s")
    is_weekday = slot_times.dayofweek < 5
    if calendar.full_closure_dates:
        is_open_date = ~pd.Series(slot_times.date).isin(calendar.full_closure_dates).to_numpy()
    else:
        is_open_date = np.ones(len(slot_times), dtype=bool)

    tradeable = (is_weekday & is_open_date).reshape(slots.shape)
    return tradeable.sum(axis=1).astype("int64")


def annotate_source_quality(
    timeframes: dict[str, TimeframeSeries], calendar: TradingCalendar, cfg: ResearchConfig
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tf, ts in timeframes.items():
        f = ts.frame
        minutes = ts.minutes
        actual = f["n_base_bars"].to_numpy(dtype="int64")

        if minutes == calendar.base_minutes:
            expected = np.ones(len(f), dtype="int64")
        else:
            expected = tradeable_slots_per_bin(f["bar_open_time"], minutes, calendar)

        nominal = minutes // calendar.base_minutes
        expected = np.maximum(expected, 1)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = actual / expected

        complete = actual >= expected
        # short because the market was shut for part of the bar
        closed_short = expected < nominal
        kind = np.where(
            complete,
            np.where(closed_short, "market_closed", "none"),
            "partial_gap",
        )

        f = f.copy()
        f["expected_constituent_bars"] = expected
        f["nominal_constituent_bars"] = nominal
        f["actual_constituent_bars"] = actual
        f["source_bar_completeness_ratio"] = np.clip(ratio, 0.0, 1.0)
        f["source_bar_complete"] = complete
        f["source_bar_gap_kind"] = kind
        out[tf] = f

        n_partial = int((kind == "partial_gap").sum())
        n_closed = int((kind == "market_closed").sum())
        log.info(
            "%-3s source quality: %d complete, %d shortened by market closure, "
            "%d with a partial data gap",
            tf, int(complete.sum()), n_closed, n_partial,
        )
    return out


def source_quality_report(
    annotated: dict[str, pd.DataFrame], calendar: TradingCalendar
) -> pd.DataFrame:
    rows: list[dict] = []
    for tf, f in annotated.items():
        kind = f["source_bar_gap_kind"]
        rows.append({
            "timeframe": tf,
            "n_bars": len(f),
            "n_complete": int(f["source_bar_complete"].sum()),
            "n_market_closed_short": int((kind == "market_closed").sum()),
            "n_partial_gap": int((kind == "partial_gap").sum()),
            "pct_partial_gap": round(float((kind == "partial_gap").mean()) * 100, 4),
            "min_completeness_ratio": round(float(f["source_bar_completeness_ratio"].min()), 4),
            "full_day_closure_dates": calendar.n_closure_dates,
            "worst_bars": "; ".join(
                f"{t:%Y-%m-%d %H:%M}({a}/{e})"
                for t, a, e in f.loc[
                    kind == "partial_gap",
                    ["bar_open_time", "actual_constituent_bars", "expected_constituent_bars"],
                ].sort_values("actual_constituent_bars").head(5).itertuples(index=False)
            ) or "none",
        })
    return pd.DataFrame(rows)


def propagate_quality_to_decisions(
    annotated: dict[str, pd.DataFrame],
    indices: dict[str, np.ndarray],
    n_rows: int,
) -> pd.DataFrame:
    out = pd.DataFrame(index=range(n_rows))
    overall = np.ones(n_rows, dtype=bool)

    for tf, f in annotated.items():
        idx = indices[tf]
        ok = idx >= 0
        safe = np.where(ok, idx, 0)

        complete = f["source_bar_complete"].to_numpy()[safe]
        ratio = f["source_bar_completeness_ratio"].to_numpy()[safe]
        kind = f["source_bar_gap_kind"].to_numpy()[safe]

        complete = np.where(ok, complete, False)
        ratio = np.where(ok, ratio, np.nan)
        kind = np.where(ok, kind, "no_bar")

        out[f"{tf}__source_bar_complete"] = complete
        out[f"{tf}__source_bar_completeness_ratio"] = ratio
        out[f"{tf}__source_bar_gap_kind"] = kind
        overall &= complete

    out["source_quality_ok"] = overall
    return out

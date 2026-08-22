"""Point-in-time alignment: the non-negotiable no-lookahead rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forex_research.alignment import (
    Alignment,
    LookaheadError,
    align_timeframes,
    assert_no_lookahead,
    take,
)
from forex_research.timeframe_builder import build_all_timeframes


def _align(base_bars, cfg, warmup=0) -> tuple[Alignment, dict]:
    tfs = build_all_timeframes(base_bars, cfg)
    a = align_timeframes(
        decision_time=tfs[cfg.base_timeframe].frame["bar_close_time"],
        timeframe_closes={tf: ts.frame["bar_close_time"] for tf, ts in tfs.items()},
        cfg=cfg, warmup_bars=warmup,
    )
    return a, tfs


def test_every_source_bar_closed_at_or_before_decision_time(base_bars, cfg):
    a, _ = _align(base_bars, cfg)
    dt = a.decision_time.to_numpy()
    for tf, close in a.source_close.items():
        has = ~pd.isna(close)
        assert (close[has] <= dt[has]).all(), f"{tf} used a bar that had not closed"


def test_assert_no_lookahead_passes_and_reports(base_bars, cfg):
    a, _ = _align(base_bars, cfg)
    table = assert_no_lookahead(a)
    assert (table["violations_source_close_after_decision"] == 0).all()
    assert (table["status"] == "ok").all()
    assert set(table["timeframe"]) == set(cfg.active_timeframes)


def test_assert_no_lookahead_detects_an_injected_violation(base_bars, cfg):
    """Deliberately break the guarantee and confirm the checker catches it."""
    a, _ = _align(base_bars, cfg)
    broken = a.source_close["1h"].copy()
    broken[500] = broken[500] + np.timedelta64(3, "h")   # a bar from the future
    a.source_close["1h"] = broken
    with pytest.raises(LookaheadError, match="closed after the decision time"):
        assert_no_lookahead(a)


def test_base_timeframe_aligns_to_its_own_bar(base_bars, cfg):
    """At T = close of base bar i, the aligned base bar must be exactly bar i."""
    a, tfs = _align(base_bars, cfg)
    idx = a.indices[cfg.base_timeframe]
    assert (idx == np.arange(len(idx))).all()


def test_higher_timeframe_value_is_the_latest_closed_one(base_bars, cfg):
    """Spot-check the as-of semantics against a brute-force search."""
    a, tfs = _align(base_bars, cfg)
    h1_close = tfs["1h"].frame["bar_close_time"].to_numpy()
    dt = a.decision_time.to_numpy()
    rng = np.random.default_rng(0)
    for i in rng.choice(len(dt), size=60, replace=False):
        expected = np.flatnonzero(h1_close <= dt[i])
        expected_idx = expected[-1] if len(expected) else -1
        assert a.indices["1h"][i] == expected_idx


def test_htf_value_is_not_forward_filled_backwards(base_bars, cfg):
    """An HTF bar must never be visible before its own close time."""
    a, tfs = _align(base_bars, cfg)
    h1 = tfs["1h"].frame
    dt = a.decision_time.to_numpy()
    for bar_i in range(3, min(20, len(h1))):
        close_t = h1["bar_close_time"].iloc[bar_i]
        before = dt < close_t.to_datetime64()
        assert (a.indices["1h"][before] < bar_i).all(), (
            f"1h bar {bar_i} leaked into decisions before it closed at {close_t}")


def test_htf_age_is_bounded_by_one_bar_when_data_is_gapless(base_bars, cfg):
    a, _ = _align(base_bars, cfg)
    dt = a.decision_time.to_numpy()
    for tf, minutes in (("15m", 15), ("1h", 60)):
        close = a.source_close[tf]
        has = ~pd.isna(close)
        age = (dt[has] - close[has]) / np.timedelta64(1, "m")
        assert age.min() >= 0
        assert age.max() < minutes, f"{tf} staleness exceeded one bar on gapless data"


def test_rows_before_first_htf_close_are_marked_invalid(base_bars, cfg):
    a, tfs = _align(base_bars, cfg, warmup=0)
    first_h1_close = tfs["1h"].frame["bar_close_time"].iloc[0]
    early = a.decision_time.to_numpy() < first_h1_close.to_datetime64()
    assert (a.indices["1h"][early] == -1).all()


def test_warmup_marks_early_rows_invalid(base_bars, cfg):
    a, _ = _align(base_bars, cfg, warmup=10)
    assert not a.valid[0]
    assert a.valid[-1]


def test_take_maps_missing_index_to_nan():
    values = np.array([1.0, 2.0, 3.0])
    out = take(values, np.array([-1, 0, 2]))
    assert np.isnan(out[0])
    assert out[1] == 1.0
    assert out[2] == 3.0


def test_align_rejects_unsorted_decision_times(cfg):
    dt = pd.Series(pd.to_datetime(["2025-01-01 01:00", "2025-01-01 00:00"]))
    with pytest.raises(LookaheadError, match="must be strictly increasing"):
        align_timeframes(dt, {"5m": dt.sort_values()}, cfg, warmup_bars=0)


def test_align_rejects_unsorted_timeframe_closes(cfg):
    dt = pd.Series(pd.to_datetime(["2025-01-01 00:00", "2025-01-01 01:00"]))
    bad = pd.Series(pd.to_datetime(["2025-01-01 01:00", "2025-01-01 00:00"]))
    with pytest.raises(LookaheadError, match="must be increasing"):
        align_timeframes(dt, {"1h": bad}, cfg, warmup_bars=0)


def test_alignment_survives_a_data_gap(cfg):
    """Across a weekend hole the aligned HTF bar is stale but never from the future."""
    from conftest import synthetic_base

    a_part = synthetic_base(n=300, start="2025-06-06 12:00")
    b_part = synthetic_base(n=300, start="2025-06-09 00:00", seed=11)
    base = pd.concat([a_part, b_part], ignore_index=True)
    a, _ = _align(base, cfg)
    dt = a.decision_time.to_numpy()
    for tf, close in a.source_close.items():
        has = ~pd.isna(close)
        assert (close[has] <= dt[has]).all()

"""Source-bar completeness: market closures must not be confused with data gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_config, synthetic_base

from forex_research.event_quality import (
    annotate_source_quality,
    build_trading_calendar,
    propagate_quality_to_decisions,
    source_quality_report,
    tradeable_slots_per_bin,
)
from forex_research.pipeline import prepare_dataset
from forex_research.timeframe_builder import build_all_timeframes


def test_gapless_weekday_data_is_fully_complete(cfg):
    bars = synthetic_base(n=864, start="2025-06-02 00:00")   # Mon 00:00, 3 full days
    cal = build_trading_calendar(bars, cfg)
    tfs = build_all_timeframes(bars, cfg)
    ann = annotate_source_quality(tfs, cal, cfg)
    for tf, f in ann.items():
        assert f["source_bar_complete"].all(), f"{tf} wrongly flagged"
        assert (f["source_bar_gap_kind"] == "none").all()
        assert (f["source_bar_completeness_ratio"] == 1.0).all()


def test_expected_counts_match_the_nominal_ratio(cfg):
    bars = synthetic_base(n=864, start="2025-06-02 00:00")
    cal = build_trading_calendar(bars, cfg)
    tfs = build_all_timeframes(bars, cfg)
    ann = annotate_source_quality(tfs, cal, cfg)
    assert (ann["15m"]["expected_constituent_bars"] == 3).all()
    assert (ann["1h"]["expected_constituent_bars"] == 12).all()
    assert (ann["15m"]["nominal_constituent_bars"] == 3).all()
    assert (ann["1h"]["nominal_constituent_bars"] == 12).all()


def test_a_real_data_gap_is_flagged_as_partial_gap(cfg):
    """Drop bars from the middle of an open session."""
    bars = synthetic_base(n=864, start="2025-06-02 00:00")
    drop = (bars["bar_open_time"] >= "2025-06-02 10:05") & \
           (bars["bar_open_time"] < "2025-06-02 10:40")
    holed = bars[~drop].reset_index(drop=True)

    cal = build_trading_calendar(holed, cfg)
    tfs = build_all_timeframes(holed, cfg)
    ann = annotate_source_quality(tfs, cal, cfg)

    h1 = ann["1h"]
    hit = h1[h1["bar_open_time"] == pd.Timestamp("2025-06-02 10:00")]
    assert len(hit) == 1
    assert not bool(hit["source_bar_complete"].iloc[0])
    assert hit["source_bar_gap_kind"].iloc[0] == "partial_gap"
    assert hit["actual_constituent_bars"].iloc[0] < hit["expected_constituent_bars"].iloc[0]


def test_full_day_holiday_is_not_called_a_data_gap(cfg):
    """A whole weekday with no bars is a closure, not thousands of missing bars."""
    a = synthetic_base(n=288, start="2025-06-02 00:00")            # Monday
    c = synthetic_base(n=288, start="2025-06-04 00:00", seed=5)    # Wednesday
    bars = pd.concat([a, c], ignore_index=True)                    # Tuesday absent

    cal = build_trading_calendar(bars, cfg)
    assert pd.Timestamp("2025-06-03").date() in cal.full_closure_dates

    tfs = build_all_timeframes(bars, cfg)
    ann = annotate_source_quality(tfs, cal, cfg)
    for tf, f in ann.items():
        assert (f["source_bar_gap_kind"] != "partial_gap").all(), (
            f"{tf} flagged a full-day closure as a data gap")


def test_weekend_bars_are_not_expected(cfg):
    bars = synthetic_base(n=576, start="2025-06-06 00:00")   # Friday onward
    cal = build_trading_calendar(bars, cfg)
    slots = tradeable_slots_per_bin(
        pd.Series(pd.to_datetime(["2025-06-07 00:00"])), 240, cal)   # a Saturday 4h bin
    assert slots[0] == 0 or slots[0] < 48


def test_quality_propagates_to_decision_rows(cfg):
    bars = synthetic_base(n=900)
    ds = prepare_dataset(bars, cfg)
    cal = build_trading_calendar(bars, cfg)
    ann = annotate_source_quality(ds.timeframes, cal, cfg)
    q = propagate_quality_to_decisions(ann, ds.alignment.indices, ds.n_rows)

    assert len(q) == ds.n_rows
    assert "source_quality_ok" in q.columns
    for tf in cfg.active_timeframes:
        assert f"{tf}__source_bar_complete" in q.columns
        assert f"{tf}__source_bar_completeness_ratio" in q.columns
    # conservative AND: overall is true only when every timeframe is complete
    per_tf = q[[f"{tf}__source_bar_complete" for tf in cfg.active_timeframes]].to_numpy()
    np.testing.assert_array_equal(q["source_quality_ok"].to_numpy(), per_tf.all(axis=1))


def test_incomplete_htf_bar_marks_the_decision_rows_it_covers(cfg):
    bars = synthetic_base(n=900, start="2025-06-02 00:00")
    drop = (bars["bar_open_time"] >= "2025-06-02 10:05") & \
           (bars["bar_open_time"] < "2025-06-02 10:40")
    holed = bars[~drop].reset_index(drop=True)

    ds = prepare_dataset(holed, cfg)
    cal = build_trading_calendar(holed, cfg)
    ann = annotate_source_quality(ds.timeframes, cal, cfg)
    q = propagate_quality_to_decisions(ann, ds.alignment.indices, ds.n_rows)
    assert int((~q["source_quality_ok"]).sum()) > 0


def test_report_separates_closures_from_gaps(cfg):
    bars = synthetic_base(n=864, start="2025-06-02 00:00")
    drop = (bars["bar_open_time"] >= "2025-06-02 10:05") & \
           (bars["bar_open_time"] < "2025-06-02 10:40")
    holed = bars[~drop].reset_index(drop=True)
    cal = build_trading_calendar(holed, cfg)
    ann = annotate_source_quality(build_all_timeframes(holed, cfg), cal, cfg)
    rep = source_quality_report(ann, cal)
    assert set(["timeframe", "n_complete", "n_market_closed_short",
                "n_partial_gap"]).issubset(rep.columns)
    assert rep["n_partial_gap"].sum() > 0

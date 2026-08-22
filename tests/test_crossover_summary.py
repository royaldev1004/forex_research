"""Crossover counting: direction, the three count levels, and no future data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_config, synthetic_base

from forex_research.crossover_analysis import (
    BEARISH,
    BULLISH,
    crossovers_for_timeframe,
    group_episodes,
    summarise,
    summarise_by_session,
)
from forex_research.pipeline import prepare_dataset


def _values(short, long_, times):
    """Two-column indicator frame: ema_2 (shorter) and ema_5 (longer)."""
    return pd.DataFrame({"ema_2": short, "ema_5": long_}), pd.Series(pd.to_datetime(times))


@pytest.fixture
def cfg_two():
    return make_config(ema_periods=(2, 5), sma_periods=(2, 5))


# --------------------------------------------------------------- direction
def test_shorter_rising_above_longer_is_bullish(cfg_two):
    t = pd.date_range("2025-06-02 00:00", periods=4, freq="5min")
    vals, bc = _values([1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 2.0, 2.0], t)
    det = crossovers_for_timeframe("5m", vals, bc, cfg_two, ("ema",))
    assert len(det) == 1
    r = det.iloc[0]
    assert r["direction"] == BULLISH
    assert r["indicator_a_period"] == 2 and r["indicator_b_period"] == 5
    assert r["bar_close_time"] == t[2]


def test_shorter_falling_below_longer_is_bearish(cfg_two):
    t = pd.date_range("2025-06-02 00:00", periods=4, freq="5min")
    vals, bc = _values([3.0, 3.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], t)
    det = crossovers_for_timeframe("5m", vals, bc, cfg_two, ("ema",))
    assert len(det) == 1 and det.iloc[0]["direction"] == BEARISH


def test_no_crossing_when_order_never_changes(cfg_two):
    t = pd.date_range("2025-06-02 00:00", periods=5, freq="5min")
    vals, bc = _values([3.0] * 5, [1.0] * 5, t)
    assert crossovers_for_timeframe("5m", vals, bc, cfg_two, ("ema",)).empty


def test_nan_warmup_does_not_produce_a_crossing(cfg_two):
    t = pd.date_range("2025-06-02 00:00", periods=4, freq="5min")
    vals, bc = _values([np.nan, 1.0, 3.0, 3.0], [np.nan, 2.0, 2.0, 2.0], t)
    det = crossovers_for_timeframe("5m", vals, bc, cfg_two, ("ema",))
    assert len(det) == 1                    # only the real cross at index 2
    assert det.iloc[0]["bar_close_time"] == t[2]


def test_ema_vs_sma_family_pairs_same_periods(cfg_two):
    t = pd.date_range("2025-06-02 00:00", periods=3, freq="5min")
    vals = pd.DataFrame({"ema_2": [1.0, 3.0, 3.0], "ema_5": [1.0, 1.0, 1.0],
                         "sma_2": [2.0, 2.0, 2.0], "sma_5": [2.0, 2.0, 2.0]})
    det = crossovers_for_timeframe("5m", vals, pd.Series(t), cfg_two, ("ema_vs_sma",))
    assert not det.empty
    assert set(det["indicator_a_type"]) == {"ema"}
    assert set(det["indicator_b_type"]) == {"sma"}
    assert (det["indicator_a_period"] == det["indicator_b_period"]).all()


# ------------------------------------------------------------ count levels
def test_pair_events_exceed_unique_timestamps_when_many_pairs_cross():
    """The whole point of separating the two counts."""
    cfg = make_config(ema_periods=(2, 3, 5), sma_periods=(2, 3, 5))
    t = pd.date_range("2025-06-02 00:00", periods=3, freq="5min")
    # all three EMAs start below and jump above the same reference on bar 1
    vals = pd.DataFrame({"ema_2": [1.0, 9.0, 9.0], "ema_3": [1.1, 8.0, 8.0],
                         "ema_5": [5.0, 5.0, 5.0]})
    det = crossovers_for_timeframe("5m", vals, pd.Series(t), cfg, ("ema",))
    counts, _ = summarise(det, {"5m": 5}, 12)
    bull = counts[counts["direction"] == BULLISH].iloc[0]
    assert bull["n_pair_cross_events"] > bull["n_unique_timestamps_with_any_cross"]
    assert bull["n_unique_timestamps_with_any_cross"] == 1


def test_episode_grouping_collapses_adjacent_bars():
    t = pd.to_datetime(["2025-06-02 00:00", "2025-06-02 00:05", "2025-06-02 00:10",
                        "2025-06-02 06:00"])
    ep = group_episodes(pd.Series(t), gap_bars=12, tf_minutes=5)
    assert ep.tolist() == [0, 0, 0, 1]


def test_episode_grouping_respects_the_gap():
    t = pd.to_datetime(["2025-06-02 00:00", "2025-06-02 01:05"])   # 65 min apart
    assert group_episodes(pd.Series(t), gap_bars=12, tf_minutes=5).tolist() == [0, 1]
    assert group_episodes(pd.Series(t), gap_bars=24, tf_minutes=5).tolist() == [0, 0]


def test_episode_count_never_exceeds_unique_timestamps(cfg_two):
    bars = synthetic_base(n=600)
    ds = prepare_dataset(bars, cfg_two)
    tff = ds.tf_features["5m"]
    det = crossovers_for_timeframe("5m", tff.indicator_values, tff.bar_close_time,
                                   cfg_two, ("ema", "sma", "ema_vs_sma"))
    counts, _ = summarise(det, {"5m": 5}, 12)
    assert (counts["n_unique_cross_episodes"]
            <= counts["n_unique_timestamps_with_any_cross"]).all()
    assert (counts["n_unique_timestamps_with_any_cross"]
            <= counts["n_pair_cross_events"]).all()


def test_pair_counts_sum_to_pair_events(cfg_two):
    bars = synthetic_base(n=400)
    ds = prepare_dataset(bars, cfg_two)
    tff = ds.tf_features["5m"]
    det = crossovers_for_timeframe("5m", tff.indicator_values, tff.bar_close_time,
                                   cfg_two, ("ema",))
    counts, pair_counts = summarise(det, {"5m": 5}, 12)
    assert pair_counts["n_cross_events"].sum() == counts["n_pair_cross_events"].sum()


def test_summarise_restricts_to_eligible_timestamps(cfg_two):
    bars = synthetic_base(n=400)
    ds = prepare_dataset(bars, cfg_two)
    tff = ds.tf_features["5m"]
    det = crossovers_for_timeframe("5m", tff.indicator_values, tff.bar_close_time,
                                   cfg_two, ("ema",))
    all_counts, _ = summarise(det, {"5m": 5}, 12)
    half = pd.DatetimeIndex(tff.bar_close_time.iloc[:200])
    part_counts, _ = summarise(det, {"5m": 5}, 12, eligible_times=half)
    assert (part_counts["n_pair_cross_events"].sum()
            <= all_counts["n_pair_cross_events"].sum())


# --------------------------------------------------------------- provenance
def test_crossovers_use_only_indicator_values_no_outcomes(cfg_two):
    """The detail frame must carry no outcome-derived column."""
    bars = synthetic_base(n=300)
    ds = prepare_dataset(bars, cfg_two)
    tff = ds.tf_features["5m"]
    det = crossovers_for_timeframe("5m", tff.indicator_values, tff.bar_close_time,
                                   cfg_two, ("ema",))
    forbidden = ("expansion_", "fwd_", "mfe", "mae", "future", "horizon_")
    assert not [c for c in det.columns if c.lower().startswith(forbidden)]


def test_crossings_are_causal(cfg_two):
    """Changing later bars must not change earlier crossings."""
    bars = synthetic_base(n=500)
    cut = 300
    a = prepare_dataset(bars, cfg_two).tf_features["5m"]
    det_a = crossovers_for_timeframe("5m", a.indicator_values, a.bar_close_time,
                                     cfg_two, ("ema",))
    mutated = bars.copy()
    mutated.loc[cut:, ["open", "high", "low", "close"]] += 0.05
    b = prepare_dataset(mutated, cfg_two).tf_features["5m"]
    det_b = crossovers_for_timeframe("5m", b.indicator_values, b.bar_close_time,
                                     cfg_two, ("ema",))
    cut_time = bars["bar_close_time"].iloc[cut - 1]
    pd.testing.assert_frame_equal(
        det_a[det_a["bar_close_time"] <= cut_time].reset_index(drop=True),
        det_b[det_b["bar_close_time"] <= cut_time].reset_index(drop=True))


def test_by_session_summary(cfg_two):
    t = pd.date_range("2025-06-02 00:00", periods=4, freq="5min")
    vals, bc = _values([1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 2.0, 2.0], t)
    det = crossovers_for_timeframe("5m", vals, bc, cfg_two, ("ema",))
    sess = pd.Series(["asia"] * 4, index=t)
    out = summarise_by_session(det, sess)
    assert not out.empty and out["session"].iloc[0] == "asia"


def test_empty_detail_summarises_to_empty():
    counts, pairs = summarise(pd.DataFrame(), {"5m": 5}, 12)
    assert counts.empty and pairs.empty

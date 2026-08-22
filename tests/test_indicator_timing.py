"""Indicator timing semantics.

Locks down the decision recorded in INDICATOR_TIMING_NOTE.md: which raw candle
feeds an indicator at a decision time, under each source mode, and that neither
mode can reach a candle that had not closed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import ROOT, make_config, synthetic_base

from forex_research.config import INDICATOR_SOURCE_SHIFT, ConfigError, load_config
from forex_research.pipeline import prepare_dataset
from forex_research.timeframe_builder import build_all_timeframes
from forex_research.timing_audit import build_timing_audit, lag_distribution


def test_shift_table_matches_the_documented_semantics():
    assert INDICATOR_SOURCE_SHIFT == {"latest_closed": 0, "previous_closed": 1}


def test_shipped_config_uses_latest_closed():
    """The Day 2 research default is the post-close interpretation."""
    cfg = load_config(ROOT / "configs" / "forex_day1.yaml")
    assert cfg.indicator_source_mode == "latest_closed"
    assert cfg.indicator_source_shift_bars == 0


def test_contradictory_timing_config_is_rejected():
    """Timing must not be settable two ways at once."""
    with pytest.raises(ConfigError, match="contradicts"):
        make_config(indicator_source_mode="latest_closed", indicator_source_shift_bars=1)
    with pytest.raises(ConfigError, match="contradicts"):
        make_config(indicator_source_mode="previous_closed", indicator_source_shift_bars=0)


def test_unknown_mode_rejected():
    with pytest.raises(ConfigError, match="indicator_source_mode"):
        make_config(indicator_source_mode="whenever", indicator_source_shift_bars=0)


@pytest.mark.parametrize("mode,shift", [("latest_closed", 0), ("previous_closed", 1)])
def test_no_contributing_candle_closes_after_the_decision(mode, shift):
    """The no-lookahead guarantee holds in BOTH modes."""
    cfg = make_config(indicator_source_mode=mode, indicator_source_shift_bars=shift)
    bars = synthetic_base(n=900)
    tfs = build_all_timeframes(bars, cfg)
    audit = build_timing_audit(tfs, tfs[cfg.base_timeframe].frame["bar_close_time"], cfg)
    sub = audit[audit["indicator_source_mode"] == mode]
    assert len(sub) > 0
    assert sub["source_closed_at_or_before_decision"].all()
    assert (sub["effective_lag_minutes"] >= 0).all()


def test_latest_closed_uses_the_aligned_bar_itself():
    cfg = make_config(indicator_source_mode="latest_closed", indicator_source_shift_bars=0)
    bars = synthetic_base(n=900)
    tfs = build_all_timeframes(bars, cfg)
    audit = build_timing_audit(tfs, tfs[cfg.base_timeframe].frame["bar_close_time"], cfg)
    sub = audit[audit["indicator_source_mode"] == "latest_closed"]
    assert (sub["indicator_input_bar_close"] == sub["selected_source_bar_close"]).all()


def test_previous_closed_drops_exactly_one_bar():
    cfg = make_config(indicator_source_mode="previous_closed", indicator_source_shift_bars=1)
    bars = synthetic_base(n=900)
    tfs = build_all_timeframes(bars, cfg)
    audit = build_timing_audit(tfs, tfs[cfg.base_timeframe].frame["bar_close_time"], cfg)
    sub = audit[audit["indicator_source_mode"] == "previous_closed"]
    assert (sub["indicator_input_bar_close"] < sub["selected_source_bar_close"]).all()
    for tf, grp in sub.groupby("timeframe"):
        gap = (grp["selected_source_bar_close"] - grp["indicator_input_bar_close"])
        # exactly one bar of that timeframe, except across market gaps where the
        # previous bar is further back in wall-clock time
        assert (gap >= pd.Timedelta(minutes=int(grp["timeframe_minutes"].iloc[0]))).all()


def test_previous_closed_is_strictly_staler_than_latest_closed():
    """This is the quantified cost of the extra shift."""
    cfg = make_config(indicator_source_mode="latest_closed", indicator_source_shift_bars=0)
    bars = synthetic_base(n=1200)
    tfs = build_all_timeframes(bars, cfg)
    dist = lag_distribution(tfs, tfs[cfg.base_timeframe].frame["bar_close_time"], cfg)
    for tf in cfg.active_timeframes:
        a = dist[(dist.timeframe == tf) & (dist.indicator_source_mode == "latest_closed")]
        b = dist[(dist.timeframe == tf) & (dist.indicator_source_mode == "previous_closed")]
        assert b["mean_lag_minutes"].iloc[0] > a["mean_lag_minutes"].iloc[0]
        assert not a["any_negative_lag"].iloc[0]
        assert not b["any_negative_lag"].iloc[0]


def test_latest_closed_indicator_includes_the_aligned_bars_own_close():
    """Direct numeric proof, independent of the audit table.

    With shift 0 the SMA at a bar must equal the mean of that bar's own source
    values including itself; with shift 1 it must equal the mean ending one bar
    earlier.
    """
    from forex_research.heikin_ashi import price_series
    from forex_research.indicators import sma

    bars = synthetic_base(n=400)
    for mode, shift in (("latest_closed", 0), ("previous_closed", 1)):
        cfg = make_config(indicator_source_mode=mode, indicator_source_shift_bars=shift,
                          ema_periods=(3,), sma_periods=(3,))
        tfs = build_all_timeframes(bars, cfg)
        f = tfs["5m"].frame
        px = price_series(f, cfg.price_basis)
        expected = sma(px.shift(shift), 3)

        ds = prepare_dataset(bars, cfg)
        got = ds.tf_features["5m"].indicator_values["sma_3"]
        pd.testing.assert_series_equal(got, expected, check_names=False)

        i = 100
        window = px.iloc[i - 2 - shift: i + 1 - shift]
        assert got.iloc[i] == pytest.approx(window.mean())


def test_switching_mode_changes_feature_values():
    """Guards against the mode silently doing nothing."""
    bars = synthetic_base(n=700)
    a = prepare_dataset(bars, make_config(indicator_source_mode="latest_closed",
                                          indicator_source_shift_bars=0))
    b = prepare_dataset(bars, make_config(indicator_source_mode="previous_closed",
                                          indicator_source_shift_bars=1))
    va = a.tf_features["1h"].indicator_values["ema_9"].to_numpy()
    vb = b.tf_features["1h"].indicator_values["ema_9"].to_numpy()
    m = np.isfinite(va) & np.isfinite(vb)
    assert m.any() and not np.allclose(va[m], vb[m])

"""Single-timeframe, consolidation and cross-timeframe feature semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_config, synthetic_base

from forex_research.consolidation_features import build_consolidation_features
from forex_research.pipeline import build_feature_frame, prepare_dataset
from forex_research.single_tf_features import build_single_tf_features
from forex_research.timeframe_builder import build_timeframe


@pytest.fixture(scope="module")
def cfg_small():
    return make_config()


@pytest.fixture(scope="module")
def built(cfg_small):
    bars = synthetic_base(n=900)
    frame, ds = build_feature_frame(bars, cfg_small)
    return frame, ds, bars


def _ramp(n=400, step=0.0002, start="2025-06-02 00:00"):
    """Strictly rising synthetic bars: the MA stack must order bullishly."""
    t = pd.date_range(start, periods=n, freq="5min")
    close = 1.10 + np.arange(n) * step
    return pd.DataFrame({
        "bar_open_time": t,
        "bar_close_time": t + pd.Timedelta(minutes=5),
        "open": close - step / 2, "high": close + step, "low": close - step, "close": close,
        "tick_volume": np.ones(n), "spread_points": np.full(n, 12.0),
        "n_base_bars": 1, "expected_base_bars": 1, "base_bar_coverage": 1.0,
        "spread_points_mean": 12.0, "spread_points_last": 12.0,
    })


def test_ordering_score_is_bullish_in_an_uptrend(cfg_small):
    bars = _ramp()
    ts = build_timeframe(bars[["bar_open_time", "bar_close_time", "open", "high", "low",
                               "close", "tick_volume", "spread_points"]], "5m", cfg_small)
    f = build_single_tf_features("5m", ts.frame, cfg_small)
    frame = f.features.to_frame(index=ts.frame.index, dtype="float64")
    score = frame["5m__ema_ordering_score"].dropna()
    assert score.iloc[-1] == pytest.approx(1.0), "shorter EMAs must sit above longer ones"
    assert frame["5m__ema_bullish_pair_fraction"].dropna().iloc[-1] == pytest.approx(1.0)
    assert frame["5m__ema_bearish_pair_fraction"].dropna().iloc[-1] == pytest.approx(0.0)


def test_ordering_score_is_bearish_in_a_downtrend(cfg_small):
    bars = _ramp(step=-0.0002)
    ts = build_timeframe(bars[["bar_open_time", "bar_close_time", "open", "high", "low",
                               "close", "tick_volume", "spread_points"]], "5m", cfg_small)
    f = build_single_tf_features("5m", ts.frame, cfg_small)
    frame = f.features.to_frame(index=ts.frame.index, dtype="float64")
    assert frame["5m__ema_ordering_score"].dropna().iloc[-1] == pytest.approx(-1.0)


def test_ribbon_width_is_non_negative_and_widens_when_trending(cfg_small):
    trend = _ramp(step=0.0004)
    flat = _ramp(step=0.0)
    cols = ["bar_open_time", "bar_close_time", "open", "high", "low", "close",
            "tick_volume", "spread_points"]

    def width(bars):
        ts = build_timeframe(bars[cols], "5m", cfg_small)
        f = build_single_tf_features("5m", ts.frame, cfg_small)
        fr = f.features.to_frame(index=ts.frame.index, dtype="float64")
        return fr["5m__ema_ribbon_width_pips"].dropna()

    w_trend = width(trend)
    w_flat = width(flat)
    assert (w_trend >= 0).all() and (w_flat >= -1e-9).all()
    assert w_trend.iloc[-1] > w_flat.iloc[-1]


def test_slope_sign_follows_the_trend(cfg_small):
    up = _ramp(step=0.0003)
    cols = ["bar_open_time", "bar_close_time", "open", "high", "low", "close",
            "tick_volume", "spread_points"]
    ts = build_timeframe(up[cols], "5m", cfg_small)
    f = build_single_tf_features("5m", ts.frame, cfg_small)
    fr = f.features.to_frame(index=ts.frame.index, dtype="float64")
    for p in cfg_small.ema_periods:
        col = f"5m__ema_{p}_slope_atr_{cfg_small.primary_slope_lookback}"
        assert fr[col].dropna().iloc[-1] > 0


def test_crossovers_are_features_not_signals(built):
    """Crossover columns must be counts/recency, never a buy/sell decision."""
    _, ds, _ = built
    cross = [s for s in ds.specs if s.feature_family == "crossover"]
    assert cross
    for s in cross:
        assert s.units in ("count", "count_per_bar", "bars", "boolean", "fraction")
        assert not any(w in s.feature_name.lower() for w in ("buy", "sell", "entry", "signal"))


def test_bars_since_cross_is_non_negative(built):
    frame, ds, _ = built
    col = "5m__ema_bars_since_cross"
    assert frame[col].dropna().min() >= 0


def test_compression_percentile_is_bounded_and_trailing(cfg_small):
    bars = synthetic_base(n=600)
    ts = build_timeframe(bars, "5m", cfg_small)
    f = build_single_tf_features("5m", ts.frame, cfg_small)
    cons = build_consolidation_features("5m", ts.frame, f.atr, cfg_small)
    fr = cons.to_frame(index=ts.frame.index, dtype="float64")
    w = cfg_small.consolidation_windows[0]
    pct = fr[f"5m__compression_percentile_{w}"].dropna()
    assert pct.between(0, 1).all()

    # extending the series with wildly different future data must not change the past
    extra = synthetic_base(n=300, start="2025-06-04 02:00", seed=99).copy()
    extra[["open", "high", "low", "close"]] += 0.05
    longer = pd.concat([bars, extra], ignore_index=True)
    ts2 = build_timeframe(longer, "5m", cfg_small)
    f2 = build_single_tf_features("5m", ts2.frame, cfg_small)
    cons2 = build_consolidation_features("5m", ts2.frame, f2.atr, cfg_small)
    fr2 = cons2.to_frame(index=ts2.frame.index, dtype="float64")
    pd.testing.assert_series_equal(
        fr[f"5m__compression_percentile_{w}"],
        fr2[f"5m__compression_percentile_{w}"].iloc[:len(bars)],
        check_names=False)


def test_is_compressed_is_derived_from_the_trailing_percentile(cfg_small):
    bars = synthetic_base(n=600)
    ts = build_timeframe(bars, "5m", cfg_small)
    f = build_single_tf_features("5m", ts.frame, cfg_small)
    fr = build_consolidation_features("5m", ts.frame, f.atr, cfg_small).to_frame(
        index=ts.frame.index, dtype="float64")
    w = cfg_small.consolidation_windows[0]
    pct = fr[f"5m__compression_percentile_{w}"]
    flag = fr[f"5m__is_compressed_{w}"]
    m = pct.notna()
    expected = (pct[m] <= cfg_small.compression_percentile_threshold).astype(float)
    pd.testing.assert_series_equal(flag[m], expected, check_names=False)


def test_position_in_range_is_bounded(cfg_small):
    bars = synthetic_base(n=400)
    ts = build_timeframe(bars, "5m", cfg_small)
    f = build_single_tf_features("5m", ts.frame, cfg_small)
    fr = build_consolidation_features("5m", ts.frame, f.atr, cfg_small).to_frame(
        index=ts.frame.index, dtype="float64")
    for w in cfg_small.consolidation_windows:
        p = fr[f"5m__position_in_range_{w}"].dropna()
        assert p.between(-1e-9, 1 + 1e-9).all()


def test_cross_tf_same_period_distance_matches_manual_computation(built, cfg_small):
    frame, ds, _ = built
    tf_lo, tf_hi = "5m", "1h"
    key = f"ema_{cfg_small.ema_periods[-1]}"
    col = f"x_{tf_lo}_{tf_hi}__{key}_value_dist_pips"
    assert col in frame.columns

    lo_vals = ds.tf_features[tf_lo].indicator_values[key].to_numpy()
    hi_vals = ds.tf_features[tf_hi].indicator_values[key].to_numpy()
    i_lo = ds.alignment.indices[tf_lo]
    i_hi = ds.alignment.indices[tf_hi]

    for i in (600, 700, 800):
        if i_lo[i] < 0 or i_hi[i] < 0:
            continue
        expected = (lo_vals[i_lo[i]] - hi_vals[i_hi[i]]) / cfg_small.pip_size
        assert frame[col].iloc[i] == pytest.approx(expected, rel=1e-4)


def test_cross_tf_trend_agreement_is_in_range(built):
    frame, _, _ = built
    col = "x_5m_1h__trend_agreement"
    vals = frame[col].dropna().unique()
    assert set(np.unique(vals)) <= {-1.0, 0.0, 1.0}


def test_mtf_counts_never_exceed_the_timeframe_count(built, cfg_small):
    frame, _, _ = built
    n_tf = len(cfg_small.active_timeframes)
    for col in ("mtf__n_timeframes_bullish", "mtf__n_timeframes_bearish",
                "mtf__n_timeframes_compressed"):
        s = frame[col].dropna()
        assert s.min() >= 0 and s.max() <= n_tf


def test_price_to_htf_distance_uses_the_aligned_htf_value(built, cfg_small):
    frame, ds, _ = built
    key = f"sma_{cfg_small.sma_periods[-1]}"
    col = f"p2h_1h__{key}_dist_pips"
    assert col in frame.columns
    hi_vals = ds.tf_features["1h"].indicator_values[key].to_numpy()
    base_px = ds.tf_features["5m"].price.to_numpy()
    i_hi = ds.alignment.indices["1h"]
    i_lo = ds.alignment.indices["5m"]
    for i in (600, 750):
        if i_hi[i] < 0:
            continue
        expected = (base_px[i_lo[i]] - hi_vals[i_hi[i]]) / cfg_small.pip_size
        assert frame[col].iloc[i] == pytest.approx(expected, rel=1e-4)


def test_all_pairs_mode_expands_the_feature_count():
    same = make_config(cross_tf_mode="same_period", timeframes=("5m", "1h"),
                       ema_periods=(2, 5), sma_periods=(2, 5))
    allp = make_config(cross_tf_mode="all_pairs", timeframes=("5m", "1h"),
                       ema_periods=(2, 5), sma_periods=(2, 5))
    bars = synthetic_base(n=400)
    n_same = prepare_dataset(bars, same).n_features
    n_all = prepare_dataset(bars, allp).n_features
    assert n_all > n_same


def test_no_infinities_anywhere(built):
    frame, _, _ = built
    num = frame.select_dtypes(include=[np.number])
    assert not np.isinf(num.to_numpy()).any(), "normalised features must yield NaN, not inf"


def test_feature_names_are_unique_and_namespaced(built, cfg_small):
    _, ds, _ = built
    assert len(set(ds.feature_columns)) == len(ds.feature_columns)
    for tf in cfg_small.active_timeframes:
        assert any(c.startswith(f"{tf}__") for c in ds.feature_columns)


def test_every_family_is_represented(built):
    _, ds, _ = built
    fams = {s.feature_family for s in ds.specs}
    expected = {
        "indicator_distance", "indicator_slope", "ordering", "ribbon_structure",
        "crossover", "volatility", "consolidation", "execution_context",
        "cross_tf_distance", "cross_tf_slope", "cross_tf_direction",
        "cross_tf_structure", "price_to_htf",
    }
    assert expected <= fams, f"missing families: {expected - fams}"

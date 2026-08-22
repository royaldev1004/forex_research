"""Future-outcome correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_config, synthetic_base

from forex_research.outcomes import RangeIndex, build_outcomes, outcome_summary


def _bars(closes, highs=None, lows=None, start="2025-06-02 00:00", freq=5):
    n = len(closes)
    t = pd.date_range(start, periods=n, freq=f"{freq}min")
    closes = np.asarray(closes, dtype=float)
    highs = np.asarray(highs, dtype=float) if highs is not None else closes
    lows = np.asarray(lows, dtype=float) if lows is not None else closes
    return pd.DataFrame({
        "bar_open_time": t,
        "bar_close_time": t + pd.Timedelta(minutes=freq),
        "open": closes, "high": highs, "low": lows, "close": closes,
        "tick_volume": np.ones(n), "spread_points": np.full(n, 12.0),
    })


def test_range_index_matches_bruteforce():
    rng = np.random.default_rng(0)
    v = rng.normal(size=400)
    mx, mn = RangeIndex(v, "max"), RangeIndex(v, "min")
    los = rng.integers(0, 399, 200)
    his = np.minimum(los + rng.integers(0, 60, 200), 399)
    got_max = mx.query_value(los, his)
    got_min = mn.query_value(los, his)
    for k, (lo, hi) in enumerate(zip(los, his)):
        assert got_max[k] == pytest.approx(v[lo:hi + 1].max())
        assert got_min[k] == pytest.approx(v[lo:hi + 1].min())


def test_range_index_empty_window():
    r = RangeIndex(np.array([1.0, 2.0, 3.0]), "max")
    out = r.query_arg(np.array([2]), np.array([1]))
    assert out[0] == -1


def test_forward_return_is_the_close_at_the_horizon():
    cfg = make_config(forward_horizons_minutes=(15,), expansion_thresholds_pips=(5.0,))
    closes = [1.1000, 1.1005, 1.1010, 1.1020, 1.1030, 1.1040]
    out, _ = build_outcomes(_bars(closes), cfg)
    # decision at bar 0 closes at 00:05; window is (00:05, 00:20] -> bars 1,2,3
    assert out["fwd_price_change_15m"].iloc[0] == pytest.approx(1.1020 - 1.1000, abs=1e-12)
    assert out["fwd_pips_15m"].iloc[0] == pytest.approx(20.0, abs=1e-6)


def test_forward_window_excludes_the_decision_bar_itself():
    """The decision bar's own high/low must not count as a future excursion."""
    cfg = make_config(forward_horizons_minutes=(15,), expansion_thresholds_pips=(5.0,))
    closes = [1.1000, 1.1000, 1.1000, 1.1000, 1.1000]
    highs = [1.2000, 1.1000, 1.1000, 1.1000, 1.1000]      # huge spike on bar 0 only
    out, _ = build_outcomes(_bars(closes, highs=highs), cfg)
    assert out["long_mfe_pips_15m"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_excursions_are_measured_from_the_decision_close():
    cfg = make_config(forward_horizons_minutes=(15,), expansion_thresholds_pips=(5.0,))
    closes = [1.1000, 1.1010, 1.0990, 1.1000, 1.1000]
    highs = [1.1000, 1.1015, 1.1000, 1.1000, 1.1000]
    lows = [1.1000, 1.1000, 1.0985, 1.1000, 1.1000]
    out, _ = build_outcomes(_bars(closes, highs, lows), cfg)
    assert out["long_mfe_pips_15m"].iloc[0] == pytest.approx(15.0, abs=1e-6)
    assert out["long_mae_pips_15m"].iloc[0] == pytest.approx(15.0, abs=1e-6)
    assert out["max_future_delta_pips_15m"].iloc[0] == pytest.approx(15.0, abs=1e-6)
    assert out["min_future_delta_pips_15m"].iloc[0] == pytest.approx(-15.0, abs=1e-6)


def test_conventional_excursions_are_never_negative():
    """long/short MFE and MAE are floored at zero by definition."""
    cfg = make_config()
    out, _ = build_outcomes(synthetic_base(n=800, seed=12), cfg)
    for h in cfg.forward_horizons_minutes:
        for col in (f"long_mfe_pips_{h}m", f"long_mae_pips_{h}m",
                    f"short_mfe_pips_{h}m", f"short_mae_pips_{h}m"):
            s = out[col].dropna()
            assert len(s) > 0
            assert (s >= 0).all(), f"{col} contains negative values"


def test_conventional_excursions_derive_from_the_signed_extrema():
    cfg = make_config()
    out, _ = build_outcomes(synthetic_base(n=800, seed=13), cfg)
    for h in cfg.forward_horizons_minutes:
        mx = out[f"max_future_delta_pips_{h}m"]
        mn = out[f"min_future_delta_pips_{h}m"]
        pd.testing.assert_series_equal(
            out[f"long_mfe_pips_{h}m"], mx.clip(lower=0), check_names=False)
        pd.testing.assert_series_equal(
            out[f"long_mae_pips_{h}m"], (-mn).clip(lower=0), check_names=False)
        pd.testing.assert_series_equal(
            out[f"short_mfe_pips_{h}m"], (-mn).clip(lower=0), check_names=False)
        pd.testing.assert_series_equal(
            out[f"short_mae_pips_{h}m"], mx.clip(lower=0), check_names=False)


def test_signed_extrema_preserve_information_that_flooring_would_destroy():
    """A gap away from the reference keeps its sign in the raw delta columns."""
    cfg = make_config(forward_horizons_minutes=(15,), expansion_thresholds_pips=(5.0,))
    closes = [1.1000, 1.1050, 1.1060, 1.1070, 1.1080]
    lows = [1.1000, 1.1045, 1.1055, 1.1065, 1.1075]     # never returns to 1.1000
    highs = [1.1000, 1.1055, 1.1065, 1.1075, 1.1085]
    out, _ = build_outcomes(_bars(closes, highs, lows), cfg)
    assert out["min_future_delta_pips_15m"].iloc[0] > 0       # raw sign retained
    assert out["long_mae_pips_15m"].iloc[0] == pytest.approx(0.0)   # floored
    assert out["long_mfe_pips_15m"].iloc[0] > 0


def test_short_excursions_mirror_long_excursions():
    cfg = make_config()
    out, _ = build_outcomes(synthetic_base(n=400), cfg)
    for h in cfg.forward_horizons_minutes:
        pd.testing.assert_series_equal(
            out[f"short_mfe_pips_{h}m"], out[f"long_mae_pips_{h}m"], check_names=False)
        pd.testing.assert_series_equal(
            out[f"short_mae_pips_{h}m"], out[f"long_mfe_pips_{h}m"], check_names=False)


def test_future_range_is_non_negative():
    cfg = make_config()
    out, _ = build_outcomes(synthetic_base(n=600, seed=8), cfg)
    for h in cfg.forward_horizons_minutes:
        assert (out[f"future_range_pips_{h}m"].dropna() >= -1e-9).all()


def test_expansion_threshold_and_first_touch_order():
    cfg = make_config(forward_horizons_minutes=(60,), expansion_thresholds_pips=(10.0,))
    # up 10 pips at bar 2, down 10 pips at bar 4 -> up must be first
    closes = [1.1000] * 8
    highs = [1.1000, 1.1005, 1.1010, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000]
    lows = [1.1000, 1.1000, 1.1000, 1.1000, 1.0990, 1.0990, 1.0990, 1.0990]
    out, _ = build_outcomes(_bars(closes, highs, lows), cfg)
    assert out["expansion_up_10p_60m"].iloc[0] == 1.0
    assert out["expansion_down_10p_60m"].iloc[0] == 1.0
    assert out["expansion_first_10p_60m"].iloc[0] == 1.0
    assert out["time_to_expansion_up_10p_60m"].iloc[0] == pytest.approx(10.0)
    assert out["time_to_expansion_down_10p_60m"].iloc[0] == pytest.approx(20.0)


def test_expansion_not_reached_is_zero_not_nan():
    cfg = make_config(forward_horizons_minutes=(15,), expansion_thresholds_pips=(50.0,))
    out, _ = build_outcomes(_bars([1.1000] * 6), cfg)
    assert out["expansion_up_50p_15m"].iloc[0] == 0.0
    assert np.isnan(out["time_to_expansion_up_50p_15m"].iloc[0])


def test_same_bar_ambiguity_is_flagged_not_guessed():
    """Both levels inside one bar cannot be ordered without tick data."""
    cfg = make_config(forward_horizons_minutes=(60,), expansion_thresholds_pips=(10.0,))
    closes = [1.1000] * 6
    highs = [1.1000, 1.1010, 1.1000, 1.1000, 1.1000, 1.1000]
    lows = [1.1000, 1.0990, 1.1000, 1.1000, 1.1000, 1.1000]
    out, _ = build_outcomes(_bars(closes, highs, lows), cfg)
    assert out["expansion_first_10p_60m"].iloc[0] == 2.0


def test_first_touch_matches_bruteforce_on_random_data():
    cfg = make_config(forward_horizons_minutes=(60,), expansion_thresholds_pips=(10.0,))
    bars = synthetic_base(n=600, seed=21)
    out, _ = build_outcomes(bars, cfg)
    close_t = bars["bar_close_time"].to_numpy()
    high = bars["high"].to_numpy()
    entry = bars["close"].to_numpy()
    target = entry + 10.0 * cfg.pip_size

    rng = np.random.default_rng(5)
    for i in rng.choice(500, size=60, replace=False):
        end = close_t[i] + np.timedelta64(60, "m")
        stop = np.searchsorted(close_t, end, side="right") - 1
        window = np.arange(i + 1, stop + 1)
        hits = window[high[window] >= target[i]]
        if len(hits):
            expected = (close_t[hits[0]] - close_t[i]) / np.timedelta64(1, "m")
            assert out["expansion_up_10p_60m"].iloc[i] == 1.0
            assert out["time_to_expansion_up_10p_60m"].iloc[i] == pytest.approx(expected)
        else:
            assert out["expansion_up_10p_60m"].iloc[i] == 0.0


def test_horizon_complete_flag_marks_the_tail():
    cfg = make_config(forward_horizons_minutes=(60,), expansion_thresholds_pips=(10.0,))
    bars = synthetic_base(n=100)
    out, _ = build_outcomes(bars, cfg)
    assert out["horizon_complete_60m"].iloc[0] == 1.0
    assert out["horizon_complete_60m"].iloc[-1] == 0.0
    assert np.isnan(out["fwd_pips_60m"].iloc[-1])


def test_horizons_are_wall_clock_not_bar_counts():
    """Across a gap the window must stay a time window, not 12 bars."""
    cfg = make_config(forward_horizons_minutes=(60,), expansion_thresholds_pips=(10.0,))
    a = synthetic_base(n=40, start="2025-06-06 22:00")
    b = synthetic_base(n=40, start="2025-06-09 00:00", seed=4)
    bars = pd.concat([a, b], ignore_index=True)
    out, _ = build_outcomes(bars, cfg)
    # the last bar before the weekend has no bars within 60 minutes
    last_before = len(a) - 1
    assert out["bars_in_window_60m"].iloc[last_before] == 0.0
    assert np.isnan(out["fwd_pips_60m"].iloc[last_before])


def test_outcomes_use_standard_prices_not_heikin_ashi():
    cfg = make_config(forward_horizons_minutes=(15,), expansion_thresholds_pips=(5.0,))
    bars = synthetic_base(n=200)
    out, _ = build_outcomes(bars, cfg)
    # entry reference must be the raw close, so a zero-length move reproduces it
    i = 0
    stop = np.searchsorted(bars["bar_close_time"].to_numpy(),
                           bars["bar_close_time"].to_numpy()[i] + np.timedelta64(15, "m"),
                           side="right") - 1
    expected = bars["close"].to_numpy()[stop] - bars["close"].to_numpy()[i]
    assert out["fwd_price_change_15m"].iloc[i] == pytest.approx(expected, abs=1e-12)


def test_excursion_bounds_hold_wherever_data_exists():
    """-MAE <= forward return <= MFE, exactly, on every observable row."""
    cfg = make_config(forward_horizons_minutes=(60,), expansion_thresholds_pips=(10.0,))
    bars = synthetic_base(n=1500, seed=33)
    out, _ = build_outcomes(bars, cfg)
    d = out.dropna(subset=["fwd_pips_60m", "long_mfe_pips_60m", "long_mae_pips_60m"])
    assert len(d) > 1000
    assert (d["fwd_pips_60m"] <= d["long_mfe_pips_60m"] + 1e-9).all()
    assert (d["fwd_pips_60m"] >= -d["long_mae_pips_60m"] - 1e-9).all()
    # MFE + MAE is the window's high-low range and can never be negative
    assert ((d["long_mfe_pips_60m"] + d["long_mae_pips_60m"]) >= -1e-9).all()


def test_empty_window_yields_nan_not_zero():
    """Market closed for the whole horizon: outcomes must be NaN, not a false 'no move'."""
    cfg = make_config(forward_horizons_minutes=(60,), expansion_thresholds_pips=(10.0,))
    a = synthetic_base(n=30, start="2025-06-06 22:00")
    b = synthetic_base(n=30, start="2025-06-09 00:00", seed=4)
    out, _ = build_outcomes(pd.concat([a, b], ignore_index=True), cfg)
    last = len(a) - 1
    assert out["bars_in_window_60m"].iloc[last] == 0
    assert np.isnan(out["fwd_pips_60m"].iloc[last])
    # flooring must not turn "unobserved" into a spurious zero
    for col in ("long_mfe_pips_60m", "long_mae_pips_60m", "short_mfe_pips_60m",
                "short_mae_pips_60m", "max_future_delta_pips_60m",
                "min_future_delta_pips_60m", "max_abs_move_pips_60m"):
        assert np.isnan(out[col].iloc[last]), f"{col} should be NaN, not a value"
    # an expansion label of 0 here means "not observed", which bars_in_window exposes
    assert out["expansion_up_10p_60m"].iloc[last] == 0.0


def test_first_touch_bruteforce_over_many_rows():
    """Exhaustive comparison of the vectorised binary search against a naive scan."""
    cfg = make_config(forward_horizons_minutes=(120,), expansion_thresholds_pips=(8.0,))
    bars = synthetic_base(n=1200, seed=77)
    out, _ = build_outcomes(bars, cfg)
    close_t = bars["bar_close_time"].to_numpy()
    high, low, entry = (bars["high"].to_numpy(), bars["low"].to_numpy(),
                        bars["close"].to_numpy())

    for i in range(0, 1000, 7):
        end = close_t[i] + np.timedelta64(120, "m")
        stop = np.searchsorted(close_t, end, side="right") - 1
        w = np.arange(i + 1, stop + 1)
        if len(w) == 0:
            continue
        up_hits = w[high[w] >= entry[i] + 8.0 * cfg.pip_size]
        dn_hits = w[low[w] <= entry[i] - 8.0 * cfg.pip_size]
        assert out["expansion_up_8p_120m"].iloc[i] == float(len(up_hits) > 0)
        assert out["expansion_down_8p_120m"].iloc[i] == float(len(dn_hits) > 0)
        if len(up_hits):
            assert out["time_to_expansion_up_8p_120m"].iloc[i] == pytest.approx(
                (close_t[up_hits[0]] - close_t[i]) / np.timedelta64(1, "m"))
        expected_first = (0.0 if not len(up_hits) and not len(dn_hits)
                          else 1.0 if not len(dn_hits)
                          else -1.0 if not len(up_hits)
                          else (2.0 if up_hits[0] == dn_hits[0]
                                else (1.0 if up_hits[0] < dn_hits[0] else -1.0)))
        assert out["expansion_first_8p_120m"].iloc[i] == expected_first


def test_outcome_keys_and_summary():
    cfg = make_config()
    bars = synthetic_base(n=300)
    out, specs = build_outcomes(bars, cfg)
    assert list(out.columns[:2]) == ["symbol", "decision_time"]
    assert (out["symbol"] == cfg.symbol).all()
    assert out["decision_time"].is_monotonic_increasing
    assert len(out) == len(bars)

    summary = outcome_summary(out, cfg, np.ones(len(out), dtype=bool))
    assert set(summary["kind"]) == {"forward_horizon", "expansion_label"}
    assert (summary["n_valid_rows"] == len(out)).all()


def test_unsupported_entry_reference_is_rejected():
    cfg = make_config(outcome_entry_reference="next_bar_open")
    with pytest.raises(ValueError, match="outcome_entry_reference"):
        build_outcomes(synthetic_base(n=50), cfg)

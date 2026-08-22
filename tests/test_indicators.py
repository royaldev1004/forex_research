"""Indicator correctness against hand-computed values on deterministic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forex_research import indicators as ind
from forex_research.heikin_ashi import heikin_ashi


def test_sma_matches_hand_computation():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ind.sma(s, 3)
    assert out.isna().tolist()[:2] == [True, True]
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[3] == pytest.approx(3.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_sma_of_constant_is_the_constant():
    s = pd.Series([7.5] * 20)
    assert ind.sma(s, 8).dropna().eq(7.5).all()


def test_ema_matches_recursion():
    """EMA must follow ta.ema: alpha = 2/(n+1), seeded at the first full window."""
    s = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    period = 3
    alpha = 2 / (period + 1)
    out = ind.ema(s, period)

    # pandas ewm(adjust=False) recursion starts from the first observation and
    # is only exposed once min_periods is met.
    expected = s.iloc[0]
    for i in range(1, len(s)):
        expected = alpha * s.iloc[i] + (1 - alpha) * expected
        if i == period - 1:
            first_visible = expected
    assert out.iloc[period - 1] == pytest.approx(first_visible)
    assert out.iloc[-1] == pytest.approx(expected)
    assert out.iloc[: period - 1].isna().all()


def test_ema_of_constant_is_the_constant():
    s = pd.Series([1.2345] * 40)
    assert ind.ema(s, 17).dropna().sub(1.2345).abs().max() < 1e-12


def test_ema_period_1_is_the_source():
    s = pd.Series([1.0, 5.0, 2.0, 9.0])
    assert ind.ema(s, 1).tolist() == s.tolist()


def test_true_range_and_atr():
    high = pd.Series([10.0, 11.0, 12.0])
    low = pd.Series([9.0, 9.5, 11.0])
    close = pd.Series([9.5, 10.5, 11.5])
    tr = ind.true_range(high, low, close)
    assert tr.iloc[0] == pytest.approx(1.0)          # no previous close
    assert tr.iloc[1] == pytest.approx(1.5)          # 11.0 - 9.5
    assert tr.iloc[2] == pytest.approx(1.5)          # 12.0 - 10.5

    a = ind.atr(high, low, close, 2)
    assert a.iloc[:1].isna().all()
    assert a.dropna().gt(0).all()


def test_atr_zero_becomes_nan():
    """A flat market must not produce a zero divisor for normalised features."""
    n = 10
    flat = pd.Series([1.0] * n)
    a = ind.atr(flat, flat, flat, 3)
    assert a.isna().all()


def test_slope_is_per_bar_rate_of_change():
    s = pd.Series([0.0, 2.0, 4.0, 6.0, 8.0])
    out = ind.slope(s, 2)
    assert out.iloc[2] == pytest.approx(2.0)   # (4-0)/2
    assert out.iloc[4] == pytest.approx(2.0)
    assert out.iloc[:2].isna().all()


def test_slope_of_flat_series_is_zero():
    s = pd.Series([3.0] * 10)
    assert ind.slope(s, 3).dropna().abs().max() == pytest.approx(0.0)


def test_safe_divide_returns_nan_not_inf():
    out = ind.safe_divide(np.array([1.0, 2.0, 3.0]), np.array([2.0, 0.0, np.nan]))
    assert out[0] == pytest.approx(0.5)
    assert np.isnan(out[1])
    assert np.isnan(out[2])


def test_rolling_percentile_is_trailing_only():
    """Appending future values must not change earlier percentile values."""
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=200))
    a = ind.rolling_percentile(s, 50)
    extended = pd.concat([s, pd.Series(rng.normal(size=200) + 50)], ignore_index=True)
    b = ind.rolling_percentile(extended, 50).iloc[:200]
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_rolling_percentile_bounds():
    s = pd.Series(np.arange(100, dtype=float))
    p = ind.rolling_percentile(s, 20).dropna()
    assert p.min() > 0 and p.max() <= 1.0
    # a strictly increasing series is always at the top of its trailing window
    assert p.iloc[-1] == pytest.approx(1.0)


def test_consecutive_true_and_bars_since():
    flag = pd.Series([0, 1, 1, 0, 1, 1, 1], dtype=float)
    assert ind.consecutive_true(flag).tolist() == [0, 1, 2, 0, 1, 2, 3]
    bs = ind.bars_since(flag > 0)
    assert np.isnan(bs.iloc[0])
    assert bs.iloc[1] == 0
    assert bs.iloc[3] == 1
    assert bs.iloc[6] == 0


def test_heikin_ashi_definition():
    df = pd.DataFrame({
        "open": [1.0, 2.0, 3.0],
        "high": [2.0, 3.0, 4.0],
        "low": [0.5, 1.5, 2.5],
        "close": [1.5, 2.5, 3.5],
    })
    ha = heikin_ashi(df)
    assert ha["ha_close"].iloc[0] == pytest.approx((1.0 + 2.0 + 0.5 + 1.5) / 4)
    assert ha["ha_open"].iloc[0] == pytest.approx((1.0 + 1.5) / 2)
    expected_open_1 = (ha["ha_open"].iloc[0] + ha["ha_close"].iloc[0]) / 2
    assert ha["ha_open"].iloc[1] == pytest.approx(expected_open_1)
    assert ha["ha_high"].iloc[1] == pytest.approx(
        max(3.0, ha["ha_open"].iloc[1], ha["ha_close"].iloc[1]))
    assert ha["ha_low"].iloc[1] == pytest.approx(
        min(1.5, ha["ha_open"].iloc[1], ha["ha_close"].iloc[1]))


def test_heikin_ashi_is_causal():
    """Changing a later bar must not change any earlier HA value."""
    df = pd.DataFrame({
        "open": np.linspace(1.0, 2.0, 30),
        "high": np.linspace(1.1, 2.1, 30),
        "low": np.linspace(0.9, 1.9, 30),
        "close": np.linspace(1.05, 2.05, 30),
    })
    a = heikin_ashi(df)
    mutated = df.copy()
    mutated.loc[20:, ["open", "high", "low", "close"]] += 5.0
    b = heikin_ashi(mutated)
    pd.testing.assert_frame_equal(a.iloc[:20], b.iloc[:20])


def test_heikin_ashi_requires_ohlc():
    with pytest.raises(ValueError, match="requires columns"):
        heikin_ashi(pd.DataFrame({"open": [1.0]}))


@pytest.mark.parametrize("period", [0, -1])
def test_invalid_periods_rejected(period):
    s = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        ind.ema(s, period)
    with pytest.raises(ValueError):
        ind.sma(s, period)

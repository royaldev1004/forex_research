"""Timeframe construction, resampling correctness and timestamp integrity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import RAW_M1, RAW_M5, make_config, synthetic_base

from forex_research.config import ConfigError, DataSource, timeframe_minutes
from forex_research.data_loader import epoch_seconds, load_source
from forex_research.timeframe_builder import build_all_timeframes, build_timeframe


def test_ohlc_aggregation_on_a_known_example(cfg):
    """Three 5-minute bars aggregate into one 15-minute bar with the right OHLC."""
    t = pd.date_range("2025-06-02 00:00", periods=3, freq="5min")
    base = pd.DataFrame({
        "bar_open_time": t,
        "bar_close_time": t + pd.Timedelta(minutes=5),
        "open": [1.1000, 1.1010, 1.1005],
        "high": [1.1020, 1.1030, 1.1008],
        "low": [1.0990, 1.1000, 1.0980],
        "close": [1.1010, 1.1005, 1.0985],
        "tick_volume": [10.0, 20.0, 30.0],
        "spread_points": [12.0, 13.0, 14.0],
    })
    ts = build_timeframe(base, "15m", cfg)
    row = ts.frame.iloc[0]

    assert row["open"] == pytest.approx(1.1000)      # first open
    assert row["high"] == pytest.approx(1.1030)      # max high
    assert row["low"] == pytest.approx(1.0980)       # min low
    assert row["close"] == pytest.approx(1.0985)     # last close
    assert row["tick_volume"] == pytest.approx(60.0)
    assert row["n_base_bars"] == 3
    assert row["bar_open_time"] == pd.Timestamp("2025-06-02 00:00")
    assert row["bar_close_time"] == pd.Timestamp("2025-06-02 00:15")


def test_bar_close_time_is_open_plus_duration(cfg, base_bars):
    for tf, ts in build_all_timeframes(base_bars, cfg).items():
        delta = ts.frame["bar_close_time"] - ts.frame["bar_open_time"]
        assert (delta == pd.Timedelta(minutes=timeframe_minutes(tf))).all()


def test_timestamps_sorted_unique_and_on_grid(cfg, base_bars):
    for tf, ts in build_all_timeframes(base_bars, cfg).items():
        f = ts.frame
        assert f["bar_close_time"].is_monotonic_increasing
        assert not f["bar_open_time"].duplicated().any()
        secs = epoch_seconds(f["bar_open_time"])
        assert (secs % (timeframe_minutes(tf) * 60) == 0).all()


def test_incomplete_tail_bin_is_dropped(cfg):
    """A 15m bin that has not finished elapsing must not appear at all."""
    t = pd.date_range("2025-06-02 00:00", periods=4, freq="5min")  # 00:00..00:15
    base = pd.DataFrame({
        "bar_open_time": t,
        "bar_close_time": t + pd.Timedelta(minutes=5),
        "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.10,
        "tick_volume": 1.0, "spread_points": 12.0,
    })
    ts = build_timeframe(base, "15m", cfg)
    # data ends 00:20; the 00:15-00:30 bin closes at 00:30 and cannot be known
    assert ts.n_incomplete_dropped == 1
    assert ts.frame["bar_close_time"].max() == pd.Timestamp("2025-06-02 00:15")


def test_higher_timeframe_never_derived_from_lower_indicators(cfg, base_bars):
    """The 1h close must equal the last 5m close inside that hour, exactly."""
    tfs = build_all_timeframes(base_bars, cfg)
    h1 = tfs["1h"].frame
    b = base_bars.set_index("bar_open_time")
    for row in h1.head(20).itertuples():
        window = b.loc[row.bar_open_time: row.bar_close_time - pd.Timedelta(minutes=5)]
        assert row.close == pytest.approx(window["close"].iloc[-1])
        assert row.open == pytest.approx(window["open"].iloc[0])
        assert row.high == pytest.approx(window["high"].max())
        assert row.low == pytest.approx(window["low"].min())


def test_heikin_ashi_computed_per_timeframe_not_aggregated(cfg, base_bars):
    """1h HA close must come from 1h OHLC, not from averaging 5m HA closes."""
    tfs = build_all_timeframes(base_bars, cfg)
    h1 = tfs["1h"].frame
    expected = (h1["open"] + h1["high"] + h1["low"] + h1["close"]) / 4.0
    pd.testing.assert_series_equal(h1["ha_close"], expected, check_names=False)

    m5 = tfs["5m"].frame.set_index("bar_open_time")
    naive = m5["ha_close"].resample("1h").mean()
    common = h1.set_index("bar_open_time")["ha_close"].index.intersection(naive.index)
    assert not np.allclose(
        h1.set_index("bar_open_time")["ha_close"].loc[common].to_numpy(),
        naive.loc[common].to_numpy(),
    ), "1h HA must differ from a naive average of 5m HA values"


def test_base_timeframe_passes_through_unchanged(cfg, base_bars):
    tfs = build_all_timeframes(base_bars, cfg)
    m5 = tfs["5m"].frame
    assert len(m5) == len(base_bars)
    np.testing.assert_allclose(m5["close"].to_numpy(), base_bars["close"].to_numpy())


def test_gaps_do_not_create_phantom_bars(cfg):
    """A weekend-sized hole must leave no empty bins behind."""
    a = synthetic_base(n=120, start="2025-06-06 20:00")     # Friday evening
    b = synthetic_base(n=120, start="2025-06-09 00:00", seed=9)  # Monday
    base = pd.concat([a, b], ignore_index=True)
    ts = build_timeframe(base, "1h", cfg)
    assert (ts.frame["n_base_bars"] > 0).all()
    assert not ts.frame["bar_open_time"].duplicated().any()


def test_config_rejects_timeframe_below_base():
    with pytest.raises(ConfigError, match="shorter than the base"):
        make_config(base_timeframe="15m", timeframes=("1m", "15m"),
                    sources=(DataSource(path=RAW_M5, timeframe="15m", role="base"),))


def test_config_rejects_unknown_timeframe():
    with pytest.raises(ConfigError, match="Unknown timeframe"):
        make_config(timeframes=("5m", "7m"))


def test_config_rejects_base_not_in_timeframes():
    with pytest.raises(ConfigError, match="must appear in timeframes"):
        make_config(base_timeframe="5m", timeframes=("15m", "1h"))


def test_config_rejects_base_source_timeframe_mismatch():
    with pytest.raises(ConfigError, match="does not match base_timeframe"):
        make_config(sources=(DataSource(path=RAW_M5, timeframe="15m", role="base"),))


@pytest.mark.skipif(not RAW_M5.exists() or not RAW_M1.exists(),
                    reason="raw EUR/USD files not present")
def test_real_m1_aggregates_into_real_m5():
    """The strongest available evidence for bar-open timestamp semantics."""
    cfg = make_config(sources=(
        DataSource(path=RAW_M5, timeframe="5m", role="base"),
        DataSource(path=RAW_M1, timeframe="1m", role="reference"),
    ))
    m5 = load_source(cfg.sources[0], cfg).frame
    m1 = load_source(cfg.sources[1], cfg).frame

    lo = max(m1["bar_open_time"].min(), m5["bar_open_time"].min())
    hi = min(m1["bar_open_time"].max(), m5["bar_open_time"].max())
    sub = m1[(m1["bar_open_time"] >= lo) & (m1["bar_open_time"] <= hi)]

    agg = (sub.set_index("bar_open_time")
           .resample("5min", label="left", closed="left", origin="start_day")
           .agg(open=("open", "first"), high=("high", "max"),
                low=("low", "min"), close=("close", "last"), n=("close", "count"))
           .reset_index())
    agg = agg[agg["n"] == 5]
    comp = agg.merge(m5[["bar_open_time", "open", "high", "low", "close"]],
                     on="bar_open_time", suffixes=("_a", "_b"))
    assert len(comp) > 1000
    cols = ["open", "high", "low", "close"]
    eq = np.all(comp[[f"{c}_a" for c in cols]].to_numpy()
                == comp[[f"{c}_b" for c in cols]].to_numpy(), axis=1)
    assert eq.mean() > 0.999, f"left-labelled match rate {eq.mean():.4f}"

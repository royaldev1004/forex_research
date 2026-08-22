"""Shared fixtures.

Tests run on small deterministic synthetic data so that expected values can be
computed by hand, plus a tiny slice of the real file where schema realism
matters. No test depends on the full dataset being present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forex_research.config import DataSource, ResearchConfig  # noqa: E402

RAW_M5 = ROOT / "data" / "raw" / "EURUSD_M5.csv"
RAW_M1 = ROOT / "data" / "raw" / "EURUSD_M1.csv"


def make_config(**overrides) -> ResearchConfig:
    """Small, fast configuration with toy period lists (research_mode='smoke')."""
    base = dict(
        symbol="EURUSD",
        pip_size=0.0001,
        point_size=0.00001,
        sources=(DataSource(path=RAW_M5, timeframe="5m", role="base"),),
        source_timezone_mode="new_york_plus_7",
        source_timezone_note="test",
        timestamp_semantics="bar_open",
        base_timeframe="5m",
        timeframes=("5m", "15m", "1h"),
        include_daily=False,
        ema_periods=(2, 5, 9),
        sma_periods=(2, 5, 9),
        price_basis="heikin_ashi",
        indicator_source_mode="latest_closed",
        indicator_source_shift_bars=0,
        atr_period=5,
        atr_price_basis="standard",
        slope_lookbacks=(1, 3),
        primary_slope_lookback=3,
        consolidation_windows=(4, 8),
        crossover_windows=(4,),
        dispersion_change_lookback=3,
        entanglement_atr_fraction=0.10,
        compression_percentile_lookback=50,
        compression_percentile_threshold=0.25,
        trend_ordering_threshold=0.5,
        cross_tf_mode="same_period",
        cross_tf_selected_pairs=(),
        cross_tf_per_indicator_features=("value_dist_pips", "value_dist_atr", "slope_diff_atr"),
        wide_per_indicator_features=("dist_pips", "dist_atr", "slope_atr"),
        forward_horizons_minutes=(15, 60),
        expansion_thresholds_pips=(5.0, 20.0),
        outcome_price_basis="standard",
        outcome_entry_reference="decision_bar_close",
        research_mode="smoke",
        output_dir=ROOT / "output" / "test",
        write_full_datasets=False,
        chunk_rows=500,
        sample_rows=50,
        random_seed=7,
        log_level="WARNING",
    )
    base.update(overrides)
    return ResearchConfig(**base)


def synthetic_base(
    n: int = 900, start: str = "2025-06-02 00:00:00", freq_min: int = 5, seed: int = 3
) -> pd.DataFrame:
    """Deterministic gap-free 5-minute bars in the canonical schema.

    Starts on a Monday so weekday-only expectations hold.
    """
    rng = np.random.default_rng(seed)
    open_time = pd.date_range(start, periods=n, freq=f"{freq_min}min")
    steps = rng.normal(0.0, 0.00035, n).cumsum()
    close = 1.10 + steps
    open_ = np.concatenate([[1.10], close[:-1]])
    spread = np.abs(rng.normal(0.00025, 0.00008, n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame({
        "bar_open_time": open_time,
        "bar_close_time": open_time + pd.Timedelta(minutes=freq_min),
        "bar_open_time_utc": (open_time - pd.Timedelta(hours=7)).tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="NaT").tz_convert("UTC"),
        "bar_close_time_utc": (open_time + pd.Timedelta(minutes=freq_min)
                               - pd.Timedelta(hours=7)).tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="NaT").tz_convert("UTC"),
        "open": open_, "high": high, "low": low, "close": close,
        "tick_volume": rng.integers(20, 500, n).astype("float64"),
        "real_volume": np.zeros(n),
        "spread_points": np.full(n, 12.0),
    })


@pytest.fixture
def cfg() -> ResearchConfig:
    return make_config()


@pytest.fixture
def base_bars() -> pd.DataFrame:
    return synthetic_base()


@pytest.fixture
def real_data_available() -> bool:
    return RAW_M5.exists()

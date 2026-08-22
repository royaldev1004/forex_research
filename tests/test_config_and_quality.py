"""Configuration behaviour, missing-data policy and data-quality auditing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from conftest import RAW_M1, RAW_M5, ROOT, make_config, synthetic_base

from forex_research.config import ConfigError, DataSource, load_config
from forex_research.data_loader import DataLoadError, load_all, load_source, to_utc
from forex_research.data_quality import audit_all, verify_timestamp_semantics
from forex_research.validation import ValidationError, check_missing_data_policy
from forex_research.pipeline import prepare_dataset

CONFIG_PATH = ROOT / "configs" / "forex_day1.yaml"


# --------------------------------------------------------------- configuration
def test_shipped_config_loads_and_matches_the_reference_scripts():
    cfg = load_config(CONFIG_PATH)
    expected = (1, 17, 32, 48, 64, 80, 95, 111, 127, 143, 158, 174, 190,
                206, 221, 237, 253, 269, 284, 300)
    assert cfg.ema_periods == expected
    assert cfg.sma_periods == expected
    assert len(cfg.ema_periods) == 20 and len(cfg.sma_periods) == 20
    assert cfg.price_basis == "heikin_ashi"
    assert cfg.indicator_source_mode == "latest_closed"
    assert cfg.indicator_source_shift_bars == 0
    assert cfg.base_timeframe == "5m"
    assert cfg.research_mode == "production"


def test_production_mode_refuses_an_empty_period_list():
    with pytest.raises(ConfigError, match="authoritative period list"):
        make_config(research_mode="production", ema_periods=(), sma_periods=())


def test_smoke_mode_allows_toy_period_lists():
    cfg = make_config(research_mode="smoke", ema_periods=(), sma_periods=())
    assert cfg.indicator_specs() == []


def test_periods_must_be_sorted_and_unique():
    with pytest.raises(ConfigError, match="sorted ascending"):
        make_config(ema_periods=(9, 2, 5))
    with pytest.raises(ConfigError, match="duplicate periods"):
        make_config(ema_periods=(2, 2, 5))


def test_horizons_must_be_multiples_of_the_base_timeframe():
    with pytest.raises(ConfigError, match="not a multiple of the base timeframe"):
        make_config(forward_horizons_minutes=(7,))


def test_primary_slope_lookback_must_be_configured():
    with pytest.raises(ConfigError, match="primary_slope_lookback"):
        make_config(slope_lookbacks=(1, 3), primary_slope_lookback=5)


def test_changing_configuration_needs_no_source_edit(tmp_path):
    """Timeframes and periods come from YAML, so a new study is a config change."""
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["timeframes"] = ["5m", "30m", "4h"]
    raw["ema_periods"] = [3, 21]
    raw["sma_periods"] = [10]
    raw["consolidation_windows"] = [5]
    raw["crossover_windows"] = [5]
    raw["slope_lookbacks"] = [2]
    raw["primary_slope_lookback"] = 2
    raw["sources"] = [{"path": str(RAW_M5), "timeframe": "5m", "role": "base"}]
    p = tmp_path / "custom.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")

    cfg = load_config(p)
    assert cfg.active_timeframes == ("5m", "30m", "4h")
    assert cfg.indicator_specs() == [("ema", 3), ("ema", 21), ("sma", 10)]

    ds = prepare_dataset(synthetic_base(n=800), cfg)
    assert ds.n_features > 0
    assert any(c.startswith("30m__") for c in ds.feature_columns)
    assert any(c.startswith("4h__") for c in ds.feature_columns)


def test_include_daily_toggle_controls_active_timeframes():
    on = make_config(timeframes=("5m", "1h", "1D"), include_daily=True)
    off = make_config(timeframes=("5m", "1h", "1D"), include_daily=False)
    assert "1D" in on.active_timeframes
    assert "1D" not in off.active_timeframes


def test_timeframe_pairs_are_ordered_low_to_high():
    cfg = make_config(timeframes=("5m", "15m", "1h"))
    assert cfg.timeframe_pairs() == [("5m", "15m"), ("5m", "1h"), ("15m", "1h")]


def test_warmup_reflects_the_longest_period():
    cfg = make_config(ema_periods=(2, 300), sma_periods=(2,))
    assert cfg.warmup_bars() > 300


# ------------------------------------------------------------- missing data
def test_missing_data_policy_fails_clearly_when_nothing_is_usable():
    cfg = make_config()
    with pytest.raises(ValidationError, match="warm-up"):
        check_missing_data_policy(pd.Series([False] * 10).to_numpy(), cfg)


def test_missing_data_policy_passes_with_usable_rows():
    cfg = make_config()
    results = check_missing_data_policy(pd.Series([False, True, True]).to_numpy(), cfg)
    assert results[0].value == 2


def test_insufficient_history_for_configured_periods_is_detected():
    """A period list longer than the history must not silently produce garbage."""
    cfg = make_config(ema_periods=(2, 500), sma_periods=(2,), research_mode="smoke")
    ds = prepare_dataset(synthetic_base(n=600), cfg)
    assert ds.alignment.valid.sum() == 0
    with pytest.raises(ValidationError):
        check_missing_data_policy(ds.alignment.valid, cfg)


# ----------------------------------------------------------------- loading
def test_missing_file_raises_clearly():
    cfg = make_config()
    bad = DataSource(path=Path("does_not_exist.csv"), timeframe="5m", role="base")
    with pytest.raises(DataLoadError, match="not found"):
        load_source(bad, cfg)


def test_bad_schema_raises_clearly(tmp_path):
    cfg = make_config()
    p = tmp_path / "bad.csv"
    p.write_text("a\tb\n1\t2\n", encoding="utf-8")
    with pytest.raises(DataLoadError, match="missing expected MT5 columns"):
        load_source(DataSource(path=p, timeframe="5m", role="base"), cfg)


def test_utc_conversion_follows_us_dst():
    s = pd.Series(pd.to_datetime(["2025-07-01 12:00:00", "2025-12-01 12:00:00"]))
    utc = to_utc(s)
    assert str(utc.iloc[0]) == "2025-07-01 09:00:00+00:00"   # UTC+3 in summer
    assert str(utc.iloc[1]) == "2025-12-01 10:00:00+00:00"   # UTC+2 in winter


def test_unknown_timezone_mode_rejected():
    with pytest.raises(DataLoadError, match="Unsupported source_timezone_mode"):
        to_utc(pd.Series(pd.to_datetime(["2025-01-01"])), mode="mars")


# ------------------------------------------------------------ data quality
@pytest.mark.skipif(not RAW_M5.exists() or not RAW_M1.exists(),
                    reason="raw EUR/USD files not present")
def test_real_data_audit_has_no_failures():
    cfg = load_config(CONFIG_PATH)
    report = audit_all(load_all(cfg), cfg)
    failures = report[report["status"] == "fail"]
    assert failures.empty, failures.to_string(index=False)


@pytest.mark.skipif(not RAW_M5.exists() or not RAW_M1.exists(),
                    reason="raw EUR/USD files not present")
def test_timestamp_semantics_inferred_as_bar_open():
    cfg = load_config(CONFIG_PATH)
    loaded = load_all(cfg)
    checks = verify_timestamp_semantics(loaded["5m"], loaded["1m"], cfg)
    inferred = [c for c in checks if c.check == "inferred_timestamp_semantics"]
    assert inferred and inferred[0].value == "bar_open"
    assert inferred[0].status == "ok"


def test_audit_flags_missing_bid_ask():
    cfg = make_config(sources=(DataSource(path=RAW_M5, timeframe="5m", role="base"),))
    if not RAW_M5.exists():
        pytest.skip("raw data not present")
    report = audit_all(load_all(cfg), cfg)
    row = report[report["check"] == "bid_ask_columns_present"]
    assert not row.empty and row["value"].iloc[0] is False
    assert row["status"].iloc[0] == "warn"

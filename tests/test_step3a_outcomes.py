"""Step 3A: outcome joining, exclusions, circularity detection and rate arithmetic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import ROOT

from forex_research.event_definitions import load_step3a_config
from forex_research.outcome_analysis import (
    apply_exclusions,
    attach_outcomes,
    compare_magnitudes,
    compare_rates,
    control_excludes_outcome,
    describe_population,
    event_implies_outcome,
    feature_family_summary,
    outcome_columns,
    parse_expansion_column,
    rate,
)

CONFIG = ROOT / "configs" / "forex_day1.yaml"


@pytest.fixture
def cfg3():
    return load_step3a_config(CONFIG)


# ------------------------------------------------------------- parsing
def test_parse_expansion_column():
    assert parse_expansion_column("expansion_up_20p_240m") == ("up", 20.0, 240)
    assert parse_expansion_column("expansion_down_50p_60m") == ("down", 50.0, 60)
    assert parse_expansion_column("fwd_pips_60m") is None
    assert parse_expansion_column("expansion_first_20p_240m") is None


# ------------------------------------------- logical implication (circularity)
def test_exact_match_is_implied():
    assert event_implies_outcome("up", 20.0, 240, "expansion_up_20p_240m")


def test_horizon_nesting_is_implied():
    """Reaching a level within 60m necessarily reaches it within 240m."""
    assert event_implies_outcome("up", 20.0, 60, "expansion_up_20p_240m")
    assert not event_implies_outcome("up", 20.0, 240, "expansion_up_20p_60m")


def test_threshold_nesting_is_implied():
    """Moving 30 pips necessarily moves 20 pips."""
    assert event_implies_outcome("up", 30.0, 240, "expansion_up_20p_240m")
    assert not event_implies_outcome("up", 20.0, 240, "expansion_up_30p_240m")


def test_opposite_direction_is_never_implied():
    assert not event_implies_outcome("up", 20.0, 60, "expansion_down_20p_240m")


def test_control_exclusion_implication():
    # pool built by removing 20p/240m expanders
    assert control_excludes_outcome(20.0, 240, "expansion_up_20p_240m")
    assert control_excludes_outcome(20.0, 240, "expansion_up_50p_240m")   # 50 implies 20
    assert control_excludes_outcome(20.0, 240, "expansion_up_20p_60m")    # 60m within 240m
    assert not control_excludes_outcome(20.0, 60, "expansion_up_20p_240m")


def test_compare_rates_flags_both_sides():
    ev = pd.DataFrame({"expansion_up_20p_240m": [1.0] * 50})
    ct = pd.DataFrame({"expansion_up_20p_240m": [0.0] * 50})
    r = compare_rates(ev, ct, "expansion_up_20p_240m", "C1", "up_20p_60m", 240,
                      ("up", 20.0, 60), "e", "c", control_exclusion=(20.0, 240))
    assert r["event_side_implied"] and r["control_side_implied"]
    assert r["is_tautological"] and not r["interpretable"]


def test_compare_rates_marks_informative_comparison_interpretable():
    ev = pd.DataFrame({"expansion_up_50p_240m": [1.0] * 30 + [0.0] * 30})
    ct = pd.DataFrame({"expansion_up_50p_240m": [0.0] * 50 + [1.0] * 10})
    r = compare_rates(ev, ct, "expansion_up_50p_240m", "C1", "up_20p_60m", 240,
                      ("up", 20.0, 60), "e", "c", control_exclusion=None)
    assert not r["is_tautological"] and r["interpretable"]
    assert r["event_rate"] == pytest.approx(0.5)
    assert r["control_rate"] == pytest.approx(1 / 6, abs=1e-4)


def test_small_samples_are_not_interpretable():
    ev = pd.DataFrame({"expansion_up_50p_240m": [1.0] * 5})
    ct = pd.DataFrame({"expansion_up_50p_240m": [0.0] * 5})
    r = compare_rates(ev, ct, "expansion_up_50p_240m", "C1", "s", 240, None, "e", "c")
    assert not r["interpretable"]


# ------------------------------------------------------- rate arithmetic
def test_rate_ignores_nan():
    n, r = rate(pd.DataFrame({"x": [1.0, 0.0, np.nan, 1.0]}), "x")
    assert n == 3 and r == pytest.approx(2 / 3)


def test_rate_of_missing_column():
    n, r = rate(pd.DataFrame({"x": [1.0]}), "missing")
    assert n == 0 and np.isnan(r)


def test_descriptive_rates_reproduce_direct_recomputation():
    """Small fixture where the answer can be computed by hand."""
    ev = pd.DataFrame({"expansion_up_20p_240m": [1, 1, 1, 0, 0, 0, 0, 0, 1, 1] * 5})
    ct = pd.DataFrame({"expansion_up_20p_240m": [1, 0, 0, 0, 0] * 10})
    r = compare_rates(ev, ct, "expansion_up_20p_240m", "C", "s", 240, None, "e", "c")
    assert r["event_rate"] == pytest.approx(0.5)
    assert r["control_rate"] == pytest.approx(0.2)
    assert r["absolute_difference_pp"] == pytest.approx(30.0)
    assert r["relative_lift_pct"] == pytest.approx(150.0)


def test_compare_magnitudes():
    ev = pd.DataFrame({"m": [10.0] * 40})
    ct = pd.DataFrame({"m": [4.0] * 40})
    r = compare_magnitudes(ev, ct, "m", "C", "s", 240, "e", "c")
    assert r["mean_difference"] == pytest.approx(6.0)
    assert r["median_difference"] == pytest.approx(6.0)
    assert r["interpretable"]


# --------------------------------------------------------- join & exclude
def test_attach_outcomes_joins_on_the_right_key():
    pop = pd.DataFrame({"event_id": ["a", "b"],
                        "canonical_decision_time": pd.to_datetime(
                            ["2025-06-02 00:05", "2025-06-02 00:15"])})
    oc = pd.DataFrame({"decision_time": pd.date_range("2025-06-02 00:00", periods=4,
                                                      freq="5min"),
                       "fwd_pips_60m": [1.0, 2.0, 3.0, 4.0]})
    out = attach_outcomes(pop, "canonical_decision_time", oc, ["fwd_pips_60m"])
    assert out["fwd_pips_60m"].tolist() == [2.0, 4.0]


def test_exclusions_remove_uninterpretable_windows():
    df = pd.DataFrame({
        "horizon_complete_240m": [1, 0, 1, 1],
        "bars_in_window_240m": [10, 10, 0, 10],
        "max_future_delta_pips_240m": [1.0, 1.0, 1.0, np.nan],
    })
    kept, reasons = apply_exclusions(df, 240, "event", "s")
    assert len(kept) == 1
    lookup = {r["reason"]: r["n_excluded"] for r in reasons}
    assert lookup["incomplete_forward_horizon"] == 1
    assert lookup["market_closed_entire_window"] == 1
    assert lookup["outcome_nan"] == 1
    assert lookup["retained"] == 1


def test_missing_outcome_is_never_read_as_no_movement():
    """A NaN outcome must be excluded, not silently counted as a zero."""
    df = pd.DataFrame({
        "horizon_complete_240m": [1, 1],
        "bars_in_window_240m": [10, 0],
        "max_future_delta_pips_240m": [5.0, np.nan],
        "expansion_up_20p_240m": [0.0, 0.0],
    })
    kept, _ = apply_exclusions(df, 240, "event", "s")
    assert len(kept) == 1
    n, _r = rate(kept, "expansion_up_20p_240m")
    assert n == 1


def test_describe_population_shapes(cfg3):
    n = 60
    df = pd.DataFrame({
        "fwd_pips_240m": np.linspace(-10, 10, n),
        "max_abs_move_pips_240m": np.linspace(1, 50, n),
        "long_mfe_pips_240m": np.linspace(0, 30, n),
        "long_mae_pips_240m": np.linspace(0, 20, n),
        "short_mfe_pips_240m": np.linspace(0, 20, n),
        "short_mae_pips_240m": np.linspace(0, 30, n),
        "future_range_pips_240m": np.linspace(5, 60, n),
        "expansion_up_20p_240m": np.r_[np.ones(30), np.zeros(30)],
        "time_to_expansion_up_20p_240m": np.r_[np.full(30, 60.0), np.full(30, np.nan)],
    })
    out = describe_population(df, 240, cfg3, "event", "s")
    assert not out.empty
    row = out[out["metric"] == "expansion_up_20p_rate"].iloc[0]
    assert row["mean"] == pytest.approx(0.5)
    for q in cfg3.distribution_quantiles:
        assert f"q{int(q * 100):02d}" in out.columns


def test_outcome_columns_cover_the_required_fields():
    cols = outcome_columns(240, (20.0, 30.0))
    for required in ("fwd_pips_240m", "long_mfe_pips_240m", "long_mae_pips_240m",
                     "short_mfe_pips_240m", "short_mae_pips_240m",
                     "max_future_delta_pips_240m", "min_future_delta_pips_240m",
                     "expansion_up_20p_240m", "expansion_first_30p_240m",
                     "time_to_expansion_up_20p_240m", "horizon_complete_240m",
                     "bars_in_window_240m"):
        assert required in cols


# ----------------------------------------------------------- exploratory
def test_feature_family_summary_is_labelled_exploratory():
    rng = np.random.default_rng(0)
    cols = [f"f{i}" for i in range(6)]
    ev = pd.DataFrame(rng.normal(1.0, 1.0, (100, 6)), columns=cols)
    ct = pd.DataFrame(rng.normal(0.0, 1.0, (100, 6)), columns=cols)
    dic = pd.DataFrame({"feature_name": cols,
                        "feature_family": ["volatility"] * 3 + ["ordering"] * 3})
    out = feature_family_summary(ev, ct, dic, ("volatility", "ordering"))
    assert set(out["feature_family"]) == {"volatility", "ordering"}
    assert out["status"].str.contains("EXPLORATORY").all()
    assert (out["mean_abs_smd"] > 0).all()


def test_feature_family_summary_handles_empty():
    assert feature_family_summary(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ()).empty

"""Benchmark states: point-in-time construction, direction mapping, rate arithmetic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import ROOT

from forex_research.benchmark import (
    BenchmarkState,
    StateLeakError,
    assert_state_is_point_in_time,
    build_states,
    cell_key,
    evaluate_state,
    standardised_baseline,
    state_counts,
    unconditional_row,
)
from forex_research.day2_checkpoint import load_day2_config

CONFIG = ROOT / "configs" / "forex_day1.yaml"


@pytest.fixture
def d2():
    return load_day2_config(CONFIG)


def _frame(n=400, seed=0, d2=None):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-06-02 00:00", periods=n, freq="5min")
    return pd.DataFrame({
        "decision_time": t,
        "session": np.where(t.hour < 8, "asia", "london"),
        "hour_bucket": (t.hour // 4) * 4,
        "volatility_bin": rng.integers(0, 3, n),
        d2["compression_column"]: rng.integers(0, 2, n).astype(float),
        d2["crossover_direction_column"]: rng.integers(-3, 4, n).astype(float),
        d2["htf_alignment_column"]: rng.uniform(-1, 1, n),
        "expansion_up_30p_240m": rng.integers(0, 2, n).astype(float),
        "expansion_down_30p_240m": rng.integers(0, 2, n).astype(float),
        "fwd_pips_240m": rng.normal(0, 10, n),
        "long_mfe_pips_240m": rng.uniform(0, 30, n),
        "long_mae_pips_240m": rng.uniform(0, 30, n),
        "short_mfe_pips_240m": rng.uniform(0, 30, n),
        "short_mae_pips_240m": rng.uniform(0, 30, n),
    })


# ------------------------------------------------- point-in-time enforcement
def test_outcome_derived_state_is_rejected():
    with pytest.raises(StateLeakError, match="future information"):
        assert_state_is_point_in_time("bad", ("expansion_up_30p_240m",))
    with pytest.raises(StateLeakError):
        assert_state_is_point_in_time("bad", ("5m__atr_pips", "fwd_pips_240m"))


def test_point_in_time_columns_are_accepted():
    assert_state_is_point_in_time("ok", ("5m__is_compressed_12", "4h__ema_ordering_score"))


def test_every_declared_state_uses_only_day1_features(d2):
    for st in build_states(d2):
        assert_state_is_point_in_time(st.name, st.source_columns)
        assert not [c for c in st.source_columns
                    if c.startswith(("expansion_", "fwd_", "long_", "short_", "max_future",
                                     "min_future", "time_to_", "horizon_", "bars_in_"))]


def test_evaluating_a_leaky_state_raises(d2):
    bad = BenchmarkState("leaky", "up", "uses the label",
                         ("expansion_up_30p_240m",),
                         lambda df: df["expansion_up_30p_240m"] == 1)
    with pytest.raises(StateLeakError):
        bad.evaluate(_frame(d2=d2))


def test_state_ladder_is_complete_for_both_directions(d2):
    states = build_states(d2)
    names = {s.name for s in states}
    assert names == {"S1_crossover_activity", "S2_compression",
                     "S3_compression_and_crossover",
                     "S4_compression_and_htf_alignment",
                     "S5_compression_crossover_htf"}
    for d in ("up", "down"):
        assert len([s for s in states if s.direction == d]) == 5


# ------------------------------------------------------ direction mapping
def test_crossover_direction_maps_correctly(d2):
    df = _frame(d2=d2)
    xcol = d2["crossover_direction_column"]
    states = {(s.name, s.direction): s for s in build_states(d2)}
    up = states[("S1_crossover_activity", "up")].evaluate(df)
    dn = states[("S1_crossover_activity", "down")].evaluate(df)
    np.testing.assert_array_equal(up.to_numpy(), (df[xcol] > 0).to_numpy())
    np.testing.assert_array_equal(dn.to_numpy(), (df[xcol] < 0).to_numpy())
    assert not (up & dn).any()


def test_htf_alignment_maps_correctly(d2):
    df = _frame(d2=d2)
    hcol = d2["htf_alignment_column"]
    thr = float(d2["htf_alignment_min_abs_ordering"])
    states = {(s.name, s.direction): s for s in build_states(d2)}
    up = states[("S4_compression_and_htf_alignment", "up")].evaluate(df)
    dn = states[("S4_compression_and_htf_alignment", "down")].evaluate(df)
    comp = df[d2["compression_column"]] == 1
    np.testing.assert_array_equal(up.to_numpy(), (comp & (df[hcol] >= thr)).to_numpy())
    np.testing.assert_array_equal(dn.to_numpy(), (comp & (df[hcol] <= -thr)).to_numpy())


def test_conjunction_states_are_subsets(d2):
    df = _frame(d2=d2)
    st = {(s.name, s.direction): s for s in build_states(d2)}
    for d in ("up", "down"):
        c = st[("S2_compression", d)].evaluate(df)
        x = st[("S1_crossover_activity", d)].evaluate(df)
        cx = st[("S3_compression_and_crossover", d)].evaluate(df)
        full = st[("S5_compression_crossover_htf", d)].evaluate(df)
        assert (cx <= (c & x)).all() and (cx == (c & x)).all()
        assert (full <= cx).all()


# --------------------------------------------------------- rate arithmetic
def test_rate_reproduces_direct_recomputation(d2):
    df = _frame(n=500, seed=3, d2=d2)
    st = [s for s in build_states(d2)
          if s.name == "S2_compression" and s.direction == "up"][0]
    row = evaluate_state(df, st, 30.0, 240)
    mask = df[d2["compression_column"]] == 1
    expected = df.loc[mask, "expansion_up_30p_240m"].mean()
    assert row["primary_outcome_rate"] == pytest.approx(round(expected, 5))
    assert row["n_observations"] == int(mask.sum())


def test_standardised_baseline_matches_hand_computation():
    """Two cells with known rates and a known state weighting."""
    df = pd.DataFrame({
        "session": ["a"] * 4 + ["b"] * 4,
        "hour_bucket": [0] * 8,
        "volatility_bin": [0] * 8,
        "y": [1.0, 0.0, 1.0, 0.0,   1.0, 1.0, 1.0, 0.0],
    })
    # state = first row of cell a, first row of cell b -> weights 0.5 / 0.5
    mask = pd.Series([True, False, False, False, True, False, False, False])
    base, covered, cells = standardised_baseline(df, mask, "y")
    # non-state cell a = [0,1,0] -> 1/3 ; cell b = [1,1,0] -> 2/3
    assert base == pytest.approx(0.5 * (1 / 3) + 0.5 * (2 / 3))
    assert covered == pytest.approx(1.0)
    assert cells == 2


def test_baseline_excludes_the_state_rows_themselves():
    df = pd.DataFrame({
        "session": ["a"] * 6, "hour_bucket": [0] * 6, "volatility_bin": [0] * 6,
        "y": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    })
    mask = pd.Series([True, True, True, False, False, False])
    base, _, _ = standardised_baseline(df, mask, "y")
    assert base == pytest.approx(0.0)      # comparator is the non-state half only


def test_unconditional_row_matches_plain_mean(d2):
    df = _frame(d2=d2)
    row = unconditional_row(df, "up", 30.0, 240, "primary")
    assert row["primary_outcome_rate"] == pytest.approx(
        round(df["expansion_up_30p_240m"].mean(), 5))
    assert row["n_observations"] == len(df)


def test_cell_key_combines_all_three_context_variables():
    df = pd.DataFrame({"session": ["a", "a"], "hour_bucket": [0, 4],
                       "volatility_bin": [1, 1]})
    k = cell_key(df)
    assert k.iloc[0] != k.iloc[1]
    assert "a" in k.iloc[0] and "h0" in k.iloc[0] and "v1" in k.iloc[0]


def test_state_counts_shape(d2):
    df = _frame(d2=d2)
    counts = state_counts(df, build_states(d2))
    assert len(counts) == 10
    assert (counts["n_observations"] <= len(df)).all()
    assert set(counts["direction"]) == {"up", "down"}


def test_empty_state_yields_nan_rate_not_zero(d2):
    df = _frame(d2=d2)
    df[d2["compression_column"]] = 0.0          # nobody is compressed
    st = [s for s in build_states(d2)
          if s.name == "S2_compression" and s.direction == "up"][0]
    row = evaluate_state(df, st, 30.0, 240)
    assert row["n_observations"] == 0
    assert np.isnan(row["primary_outcome_rate"])

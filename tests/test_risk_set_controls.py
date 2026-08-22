"""Risk-set controls: no purge, no outcome filtering, and no future information."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forex_research.risk_set_controls import (
    BASE_MATCH_KEYS,
    OutcomeLeakError,
    assert_no_outcome_keys,
    balance_report,
    build_match_key,
    build_risk_set_pool,
    sample_random_context_controls,
    sample_risk_set_controls,
    standardised_difference,
)


def _context(n=600, start="2025-06-02 00:00", seed=0):
    t = pd.date_range(start, periods=n, freq="5min")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "decision_time": t,
        "session": np.where(t.hour < 8, "asia", np.where(t.hour < 13, "london", "new_york")),
        "hour_bucket": (t.hour // 4) * 4,
        "day_of_week": t.dayofweek,
        "volatility_bin": rng.integers(0, 4, n),
        "compression_bin": rng.integers(0, 4, n),
        # a future outcome that MUST NOT influence anything
        "expansion_up_30p_240m": rng.integers(0, 2, n).astype(float),
    })


def _events(ctx, k=15, offset=50, step=25):
    idx = np.arange(offset, offset + k * step, step)
    e = ctx.iloc[idx].copy().reset_index(drop=True)
    e["event_id"] = [f"evt_{i:03d}" for i in range(len(e))]
    e["canonical_decision_time"] = e["decision_time"]
    e["stratum"] = "up_30p_240m"
    return e


# ------------------------------------------------- the defining property
def test_controls_may_have_positive_future_outcomes():
    """A control that later expands is VALID and must be retained."""
    ctx = _context()
    ev = _events(ctx)
    pool = build_risk_set_pool(ctx, pd.Series(True, index=ctx.index),
                               ev["canonical_decision_time"].to_numpy(),
                               ctx["expansion_up_30p_240m"])
    assert pool.n_positive_outcome_retained > 0
    assert (pool.frame["expansion_up_30p_240m"] == 1).any()

    controls, _ = sample_risk_set_controls(ev, pool.frame, 3, seed=1)
    picked = ctx.set_index("decision_time").loc[
        pd.DatetimeIndex(controls["control_decision_time"])]
    assert (picked["expansion_up_30p_240m"] == 1).any(), (
        "risk-set controls must be allowed to later experience the outcome")


def test_no_temporal_purge_is_applied():
    """Rows immediately adjacent to an event stay in the pool."""
    ctx = _context()
    ev = _events(ctx, k=1, offset=100)
    pool = build_risk_set_pool(ctx, pd.Series(True, index=ctx.index),
                               ev["canonical_decision_time"].to_numpy())
    ev_t = ev["canonical_decision_time"].iloc[0]
    near = pool.frame[(pool.frame["decision_time"] > ev_t - pd.Timedelta(minutes=30))
                      & (pool.frame["decision_time"] < ev_t + pd.Timedelta(minutes=30))]
    assert len(near) > 0, "a +/-240m purge would have removed these"
    assert len(pool.frame) == len(ctx) - 1


def test_only_the_events_own_timestamp_is_excluded():
    ctx = _context()
    ev = _events(ctx, k=5, offset=40, step=50)
    pool = build_risk_set_pool(ctx, pd.Series(True, index=ctx.index),
                               ev["canonical_decision_time"].to_numpy())
    assert pool.n_excluded_identity == 5
    assert not pd.DatetimeIndex(pool.frame["decision_time"]).isin(
        pd.DatetimeIndex(ev["canonical_decision_time"])).any()


def test_eligibility_mask_is_respected():
    ctx = _context()
    elig = pd.Series([i % 3 == 0 for i in range(len(ctx))], index=ctx.index)
    pool = build_risk_set_pool(ctx, elig, np.array([], dtype="datetime64[ns]"))
    assert len(pool.frame) == int(elig.sum())


# ----------------------------------------------------- no outcome leakage
def test_matching_on_an_outcome_column_is_refused():
    with pytest.raises(OutcomeLeakError, match="future-derived"):
        assert_no_outcome_keys(("session", "expansion_up_30p_240m"))
    with pytest.raises(OutcomeLeakError):
        assert_no_outcome_keys(("fwd_pips_240m",))


def test_build_match_key_refuses_outcome_columns():
    ctx = _context()
    with pytest.raises(OutcomeLeakError):
        build_match_key(ctx, ("session", "expansion_up_30p_240m"))


def test_sampler_refuses_outcome_match_keys():
    ctx = _context()
    ev = _events(ctx)
    with pytest.raises(OutcomeLeakError):
        sample_risk_set_controls(ev, ctx, 2, seed=1,
                                 match_keys=("session", "expansion_up_30p_240m"))


def test_matching_only_uses_point_in_time_keys():
    assert BASE_MATCH_KEYS == ("session", "hour_bucket", "volatility_bin")
    assert_no_outcome_keys(BASE_MATCH_KEYS)


def test_shuffling_outcomes_does_not_change_the_draw():
    """Strongest check: the matcher cannot be reading the label at all."""
    ctx = _context()
    ev = _events(ctx)
    pool = build_risk_set_pool(ctx, pd.Series(True, index=ctx.index),
                               ev["canonical_decision_time"].to_numpy())
    a, _ = sample_risk_set_controls(ev, pool.frame, 3, seed=5)

    scrambled = pool.frame.copy()
    rng = np.random.default_rng(99)
    scrambled["expansion_up_30p_240m"] = rng.permutation(
        scrambled["expansion_up_30p_240m"].to_numpy())
    b, _ = sample_risk_set_controls(ev, scrambled, 3, seed=5)

    pd.testing.assert_series_equal(a["control_decision_time"], b["control_decision_time"])


# ------------------------------------------------------------ mechanics
def test_matched_controls_share_the_matching_cell():
    ctx = _context()
    ev = _events(ctx)
    controls, _ = sample_risk_set_controls(ev, ctx, 2, seed=3)
    lookup = ev.set_index("event_id")
    for c in controls.itertuples():
        e = lookup.loc[c.matched_event_id]
        assert c.matching_session == e["session"]
        assert c.matching_hour_bucket == e["hour_bucket"]
        assert c.matching_volatility_bin == e["volatility_bin"]


def test_fixed_seed_is_reproducible():
    ctx = _context()
    ev = _events(ctx)
    a, _ = sample_risk_set_controls(ev, ctx, 3, seed=11)
    b, _ = sample_risk_set_controls(ev, ctx, 3, seed=11)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_the_draw():
    ctx = _context()
    ev = _events(ctx)
    a, _ = sample_risk_set_controls(ev, ctx, 3, seed=11)
    b, _ = sample_risk_set_controls(ev, ctx, 3, seed=12)
    assert not a["control_decision_time"].equals(b["control_decision_time"])


def test_compression_matched_variant_adds_the_key():
    ctx = _context()
    ev = _events(ctx)
    controls, _ = sample_risk_set_controls(
        ev, ctx, 2, seed=4, control_type="risk_set_compression_matched",
        match_keys=BASE_MATCH_KEYS + ("compression_bin",))
    lookup = ev.set_index("event_id")
    for c in controls.itertuples():
        assert c.matching_compression_bin == lookup.loc[c.matched_event_id]["compression_bin"]


def test_random_context_controls_are_reproducible_and_stratified():
    ctx = _context()
    ev = _events(ctx, k=40, offset=20, step=12)
    a = sample_random_context_controls(ev, ctx, 200, seed=7)
    b = sample_random_context_controls(ev, ctx, 200, seed=7)
    pd.testing.assert_frame_equal(a, b)
    ev_share = ev["session"].value_counts(normalize=True)
    ct_share = a["matching_session"].value_counts(normalize=True)
    for s in ev_share.index:
        assert abs(ev_share[s] - ct_share.get(s, 0.0)) < 0.15


# ------------------------------------------------------------- balance
def test_standardised_difference_is_zero_for_identical_samples():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert standardised_difference(s, s) == pytest.approx(0.0)


def test_balance_report_flags_categorical_by_share_gap():
    ctx = _context()
    ev = _events(ctx)
    controls, _ = sample_risk_set_controls(ev, ctx, 3, seed=2)
    rep = balance_report(
        ev, controls, "risk_set_matched",
        ("session", "hour_bucket", "volatility_bin"),
        {"session": "session", "hour_bucket": "hour_bucket",
         "volatility_bin": "volatility_bin"},
        {"session": "matching_session", "hour_bucket": "matching_hour_bucket",
         "volatility_bin": "matching_volatility_bin"})
    sess = rep[rep["variable"] == "session"].iloc[0]
    # exact cell matching -> perfectly balanced, and must be reported as such
    assert sess["max_category_share_gap"] == pytest.approx(0.0)
    assert bool(sess["balanced"])
    assert sess["balance_criterion"] == "max_category_share_gap<0.05"
    num = rep[rep["variable"] == "volatility_bin"].iloc[0]
    assert num["balance_criterion"] == "abs_smd<0.1"
    assert bool(num["balanced"])


def test_empty_inputs_return_empty():
    c, r = sample_risk_set_controls(pd.DataFrame(), _context(), 3, seed=1)
    assert c.empty and r.empty
    assert sample_random_context_controls(pd.DataFrame(), _context(), 10, seed=1).empty

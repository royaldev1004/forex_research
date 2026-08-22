"""Control sampling: matching, purging, determinism and no future information."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import ROOT

from forex_research.control_sampling import (
    balance_report,
    build_control_pool,
    compression_bins,
    sample_matched_controls,
    sample_random_controls,
    volatility_bins,
)
from forex_research.event_definitions import EventSpec, load_step2_config

CONFIG = ROOT / "configs" / "forex_day1.yaml"


@pytest.fixture
def cfg2():
    return load_step2_config(CONFIG)


def _spec():
    return EventSpec("up", 20.0, 240)


def _context(n=800, start="2025-06-02 00:00"):
    t = pd.date_range(start, periods=n, freq="5min")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "decision_time": t,
        "session": np.where(t.hour < 8, "asia", np.where(t.hour < 13, "london", "new_york")),
        "hour_bucket": (t.hour // 4) * 4,
        "day_of_week": t.dayofweek,
        "volatility_bin": rng.integers(0, 4, n),
        "compression_bin": rng.integers(0, 4, n),
        "compression_percentile": rng.random(n),
        "compression_state": "neutral",
    })


def _events(ctx, k=20, offset=100, step=30):
    idx = np.arange(offset, offset + k * step, step)
    e = ctx.iloc[idx].copy().reset_index(drop=True)
    e["event_id"] = [f"evt_{i:04d}" for i in range(len(e))]
    e["canonical_decision_time"] = e["decision_time"]
    e["event_start_time"] = e["decision_time"]
    e["event_end_time"] = e["decision_time"] + pd.Timedelta(minutes=240)
    return e


# ------------------------------------------------------------- binning
def test_volatility_bins_are_quantiles():
    v = pd.Series(np.arange(100, dtype=float))
    b = volatility_bins(v, 4)
    assert b.nunique() == 4
    assert b.iloc[0] == 0 and b.iloc[-1] == 3


def test_volatility_bins_use_a_shared_reference_scale():
    ref = pd.Series(np.arange(100, dtype=float))
    small = pd.Series([10.0, 90.0])
    b = volatility_bins(small, 4, reference=ref)
    assert b.iloc[0] < b.iloc[1]


def test_volatility_bins_handle_all_nan():
    b = volatility_bins(pd.Series([np.nan, np.nan]), 4)
    assert (b == -1).all()


def test_compression_bins_are_fixed_edges():
    """Edges are 0.25/0.5/0.75 and right-closed, so 0.5 lands in bin 1."""
    b = compression_bins(pd.Series([0.0, 0.24, 0.5, 0.6, 0.99]))
    assert b.tolist() == [0, 0, 1, 2, 3]


def test_compression_bins_are_stable_across_populations():
    """Fixed edges mean a bin means the same thing for events and controls."""
    a = compression_bins(pd.Series([0.1, 0.9]))
    b = compression_bins(pd.Series([0.1, 0.2, 0.3, 0.9]))
    assert a.iloc[0] == b.iloc[0] and a.iloc[-1] == b.iloc[-1]


def test_compression_bins_handle_nan():
    assert compression_bins(pd.Series([np.nan, 0.5])).tolist() == [-1, 1]


# ---------------------------------------------------------------- pool
def test_pool_excludes_purged_rows(cfg2):
    ctx = _context()
    ev = _events(ctx, k=3, offset=200, step=200)
    eligible = pd.Series(True, index=ctx.index)
    non_event = pd.Series(True, index=ctx.index)
    pool = build_control_pool(ctx, eligible, non_event, ev, _spec(), cfg2)
    assert pool.n_purged > 0
    assert pool.n_after_purge == pool.n_before_purge - pool.n_purged
    assert len(pool.frame) == pool.n_after_purge


def test_no_control_sits_inside_an_event_window(cfg2):
    """The core temporal-leakage guarantee."""
    ctx = _context()
    ev = _events(ctx, k=5, offset=150, step=120)
    pool = build_control_pool(ctx, pd.Series(True, index=ctx.index),
                              pd.Series(True, index=ctx.index), ev, _spec(), cfg2)
    ctl_times = pd.DatetimeIndex(pool.frame["decision_time"])
    hz = pd.Timedelta(minutes=240)
    for e in ev.itertuples():
        # a control's forward window must not touch the event extent
        overlap = (ctl_times + hz >= e.event_start_time) & (ctl_times <= e.event_end_time)
        assert not overlap.any(), f"control overlaps event at {e.event_start_time}"


def test_pool_respects_eligibility(cfg2):
    ctx = _context()
    eligible = pd.Series([i % 2 == 0 for i in range(len(ctx))], index=ctx.index)
    pool = build_control_pool(ctx, eligible, pd.Series(True, index=ctx.index),
                              pd.DataFrame(), _spec(), cfg2)
    assert len(pool.frame) == int(eligible.sum())


# ------------------------------------------------------------ matching
def test_matched_controls_share_the_matching_cell(cfg2):
    ctx = _context()
    ev = _events(ctx)
    controls, report = sample_matched_controls(
        ev, ctx, _spec(), cfg2, "general_matched", use_compression=False, n_per_event=2)
    assert not controls.empty
    lookup = ev.set_index("event_id")
    for c in controls.itertuples():
        e = lookup.loc[c.matched_event_id]
        assert c.matching_session == e["session"]
        assert c.matching_hour_bucket == e["hour_bucket"]
        assert c.matching_volatility_bin == e["volatility_bin"]


def test_consolidation_matching_also_matches_compression(cfg2):
    ctx = _context()
    ev = _events(ctx)
    controls, _ = sample_matched_controls(
        ev, ctx, _spec(), cfg2, "consolidation_matched", use_compression=True, n_per_event=2)
    lookup = ev.set_index("event_id")
    for c in controls.itertuples():
        assert c.matching_compression_bin == lookup.loc[c.matched_event_id]["compression_bin"]


def test_matching_is_reproducible_with_a_fixed_seed(cfg2):
    ctx = _context()
    ev = _events(ctx)
    a, _ = sample_matched_controls(ev, ctx, _spec(), cfg2, "general_matched", False, 3)
    b, _ = sample_matched_controls(ev, ctx, _spec(), cfg2, "general_matched", False, 3)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_give_different_draws(cfg2):
    ctx = _context()
    ev = _events(ctx)
    a, _ = sample_matched_controls(ev, ctx, _spec(), cfg2, "general_matched", False, 3)
    b, _ = sample_matched_controls(ev, ctx, _spec(), cfg2, "general_matched", False, 3,
                                   seed_offset=99)
    assert not a["control_decision_time"].equals(b["control_decision_time"])


def test_control_ids_unique(cfg2):
    ctx = _context()
    controls, _ = sample_matched_controls(
        _events(ctx), ctx, _spec(), cfg2, "general_matched", False, 3)
    assert controls["control_id"].is_unique


def test_empty_inputs_return_empty(cfg2):
    c, r = sample_matched_controls(pd.DataFrame(), _context(), _spec(), cfg2, "x", False, 3)
    assert c.empty and r.empty


# -------------------------------------------------------------- random
def test_random_controls_are_reproducible(cfg2):
    ctx = _context()
    ev = _events(ctx)
    a = sample_random_controls(ev, ctx, _spec(), cfg2, 30)
    b = sample_random_controls(ev, ctx, _spec(), cfg2, 30)
    pd.testing.assert_frame_equal(a, b)


def test_random_controls_follow_the_event_session_mix(cfg2):
    ctx = _context()
    ev = _events(ctx, k=40, offset=50, step=15)
    rnd = sample_random_controls(ev, ctx, _spec(), cfg2, 200)
    ev_share = ev["session"].value_counts(normalize=True)
    rn_share = rnd["matching_session"].value_counts(normalize=True)
    for s in ev_share.index:
        assert abs(ev_share[s] - rn_share.get(s, 0.0)) < 0.15, f"session {s} badly skewed"


def test_random_controls_come_from_the_pool(cfg2):
    ctx = _context()
    rnd = sample_random_controls(_events(ctx), ctx, _spec(), cfg2, 50)
    assert set(rnd["control_decision_time"]).issubset(set(ctx["decision_time"]))


# ------------------------------------------------------------- balance
def test_balance_report_shape(cfg2):
    ctx = _context()
    ev = _events(ctx)
    controls, _ = sample_matched_controls(ev, ctx, _spec(), cfg2, "general_matched", False, 3)
    b = balance_report(ev.rename(columns={"session": "matching_session"}),
                       controls, "matching_session", "session")
    assert {"session", "event_share", "control_share", "abs_difference"} <= set(b.columns)
    assert abs(b["event_share"].sum() - 1.0) < 1e-6

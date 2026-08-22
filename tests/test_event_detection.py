"""Event detection, de-duplication and temporal purging."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import ROOT, make_config, synthetic_base

from forex_research.event_deduplication import (
    build_purge_intervals,
    collapse_by_separation,
    collapse_to_episodes,
    in_any_interval,
    overlap_report,
)
from forex_research.event_definitions import EventSpec, load_step2_config, session_of
from forex_research.event_detection import (
    COMPRESSED,
    add_time_context,
    classify_compression,
    context_label,
    eligibility_mask,
)

CONFIG = ROOT / "configs" / "forex_day1.yaml"


@pytest.fixture
def cfg2():
    return load_step2_config(CONFIG)


def _spec(direction="up", thr=20.0, hor=240):
    return EventSpec(direction, thr, hor)


def _candidates(times, ttt=30.0, up=10.0, dn=-5.0, compression="neutral"):
    return pd.DataFrame({
        "decision_time": pd.to_datetime(times),
        "time_to_threshold_minutes": ttt,
        "max_future_delta_pips": up,
        "min_future_delta_pips": dn,
        "fwd_pips": 1.0,
        "opposite_expansion": 0.0,
        "first_touch": 1.0,
        "compression_state": compression,
    })


# ----------------------------------------------------------- definitions
def test_event_spec_column_names():
    s = EventSpec("up", 20.0, 240)
    assert s.tag == "up_20p_240m"
    assert s.expansion_column == "expansion_up_20p_240m"
    assert s.time_to_column == "time_to_expansion_up_20p_240m"
    assert s.opposite_expansion_column == "expansion_down_20p_240m"


def test_session_boundaries():
    assert session_of(0) == "asia"
    assert session_of(9) == "london"
    assert session_of(14) == "london_ny_overlap"
    assert session_of(18) == "new_york"
    assert session_of(23) == "late"


def test_config_rejects_future_peeking_canonical_rule(tmp_path):
    import yaml
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["step2"]["canonical_rule"] = "best_entry_in_episode"
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(Exception, match="leak"):
        load_step2_config(p)


# ------------------------------------------------------- de-duplication
def test_overlapping_candidates_collapse_into_one_episode(cfg2):
    """Rows every 5 minutes across one move must become a single episode."""
    times = pd.date_range("2025-06-02 10:00", periods=12, freq="5min")
    ep = collapse_to_episodes(_candidates(times), _spec(hor=240), cfg2)
    assert len(ep) == 1
    assert ep["n_candidates_collapsed"].iloc[0] == 12
    assert ep["canonical_decision_time"].iloc[0] == times[0]


def test_canonical_timestamp_is_the_earliest_not_the_best(cfg2):
    """Canonical selection must not depend on the future path."""
    times = pd.date_range("2025-06-02 10:00", periods=6, freq="5min")
    cand = _candidates(times)
    cand["max_future_delta_pips"] = [10, 90, 20, 15, 12, 11]   # row 1 looks 'best'
    ep = collapse_to_episodes(cand, _spec(), cfg2)
    assert ep["canonical_decision_time"].iloc[0] == times[0]


def test_candidates_beyond_the_window_open_a_new_episode(cfg2):
    a = pd.date_range("2025-06-02 10:00", periods=3, freq="5min")
    b = pd.date_range("2025-06-02 18:00", periods=3, freq="5min")   # 8h later
    ep = collapse_to_episodes(_candidates(a.union(b)), _spec(hor=240), cfg2)
    assert len(ep) == 2


def test_opposite_directions_are_never_merged(cfg2):
    """Up and down are scanned separately, so they cannot collapse together."""
    times = pd.date_range("2025-06-02 10:00", periods=4, freq="5min")
    up = collapse_to_episodes(_candidates(times), _spec("up"), cfg2, prefix="evt")
    dn = collapse_to_episodes(_candidates(times), _spec("down"), cfg2, prefix="evt")
    assert len(up) == 1 and len(dn) == 1
    assert up["event_direction"].iloc[0] == "up"
    assert dn["event_direction"].iloc[0] == "down"
    assert set(up["event_id"]).isdisjoint(set(dn["event_id"]))


def test_episode_ids_are_unique(cfg2):
    times = pd.date_range("2025-06-02 00:00", periods=200, freq="30min")
    ep = collapse_to_episodes(_candidates(times), _spec(hor=60), cfg2)
    assert ep["event_id"].is_unique


def test_episodes_do_not_overlap(cfg2):
    times = pd.date_range("2025-06-02 00:00", periods=300, freq="10min")
    ep = collapse_to_episodes(_candidates(times), _spec(hor=120), cfg2)
    starts = pd.DatetimeIndex(ep["canonical_decision_time"])
    assert (starts.to_series().diff().dropna() >= pd.Timedelta(minutes=120)).all()


def test_empty_candidates_produce_empty_episodes(cfg2):
    ep = collapse_to_episodes(pd.DataFrame(), _spec(), cfg2)
    assert ep.empty


def test_collapse_by_separation_thins_dense_runs(cfg2):
    times = pd.date_range("2025-06-02 00:00", periods=60, freq="5min")   # 5 hours
    rows = pd.DataFrame({"decision_time": times, "max_future_delta_pips": 1.0,
                         "min_future_delta_pips": -1.0, "fwd_pips": 0.0})
    out = collapse_by_separation(rows, 240, "cons_fail", _spec())
    assert len(out) == 2                        # 5h at 4h spacing
    assert out["n_candidates_collapsed"].sum() == 60


def test_overlap_report_arithmetic():
    rep = overlap_report({"up_20p_60m": (1000, 300, 50)})
    r = rep.iloc[0]
    assert r["rows_suppressed"] == 250
    assert r["avg_rows_per_episode"] == 6.0


# --------------------------------------------------------------- purge
def test_purge_interval_covers_the_lookback_window():
    ep = pd.DataFrame({
        "event_start_time": [pd.Timestamp("2025-06-02 12:00")],
        "event_end_time": [pd.Timestamp("2025-06-02 16:00")],
    })
    iv = build_purge_intervals(ep, purge_minutes=60, horizon_minutes=240)
    # a control at t looks forward 240m, so exclusion must start 240+60 before
    assert iv[0, 0] == np.datetime64("2025-06-02 07:00")
    assert iv[0, 1] == np.datetime64("2025-06-02 17:00")


def test_control_inside_purge_window_is_detected():
    ep = pd.DataFrame({
        "event_start_time": [pd.Timestamp("2025-06-02 12:00")],
        "event_end_time": [pd.Timestamp("2025-06-02 16:00")],
    })
    iv = build_purge_intervals(ep, 60, 240)
    times = pd.to_datetime([
        "2025-06-02 06:00",   # before window
        "2025-06-02 08:00",   # inside (its forward window reaches the event)
        "2025-06-02 14:00",   # inside the event itself
        "2025-06-02 18:00",   # after
    ]).to_numpy()
    got = in_any_interval(times, iv)
    assert list(got) == [False, True, True, False]


def test_purge_handles_many_overlapping_intervals():
    starts = pd.date_range("2025-06-02 00:00", periods=50, freq="1h")
    ep = pd.DataFrame({"event_start_time": starts,
                       "event_end_time": starts + pd.Timedelta(hours=4)})
    iv = build_purge_intervals(ep, 30, 60)
    times = pd.date_range("2025-06-01 00:00", periods=200, freq="1h").to_numpy()
    got = in_any_interval(times, iv)
    # brute-force check
    expect = np.array([bool(((t >= iv[:, 0]) & (t < iv[:, 1])).any()) for t in times])
    np.testing.assert_array_equal(got, expect)


def test_no_purge_intervals_excludes_nothing():
    got = in_any_interval(pd.date_range("2025-01-01", periods=5).to_numpy(),
                          np.empty((0, 2), dtype="datetime64[ns]"))
    assert not got.any()


# ------------------------------------------------------------- context
def test_compression_uses_trailing_features_only(cfg2):
    feats = pd.DataFrame({
        "decision_time": pd.date_range("2025-06-02", periods=5, freq="5min"),
        cfg2.compression_primary_column: [0.05, 0.30, 0.50, 0.80, np.nan],
    })
    st = classify_compression(feats, cfg2)
    assert st.tolist() == [COMPRESSED, "neutral", "neutral", "expanding", "unknown"]


def test_missing_compression_column_fails_loudly(cfg2):
    with pytest.raises(KeyError, match="compression"):
        classify_compression(pd.DataFrame({"decision_time": []}), cfg2)


def test_time_context_columns(cfg2):
    df = pd.DataFrame({"decision_time": pd.to_datetime(
        ["2025-06-02 09:30", "2025-06-02 18:10"])})
    out = add_time_context(df, cfg2)
    assert out["session"].tolist() == ["london", "new_york"]
    assert out["hour"].tolist() == [9, 18]
    assert (out["hour_bucket"] % cfg2.hour_bucket_hours == 0).all()


def test_context_labels():
    assert context_label("up", True, True) == "major_up_after_compression"
    assert context_label("down", False, True) == "major_down_without_compression"
    assert context_label("up", True, False) == "compressed_no_major_expansion"


def test_eligibility_excludes_incomplete_and_low_quality(cfg2):
    n = 6
    outcomes = pd.DataFrame({
        "decision_time": pd.date_range("2025-06-02", periods=n, freq="5min"),
        "horizon_complete_240m": [1, 1, 0, 1, 1, 1],
        "bars_in_window_240m": [10, 0, 10, 10, 10, 10],
        "max_future_delta_pips_240m": [1.0, 1.0, 1.0, np.nan, 1.0, 1.0],
        "expansion_first_20p_240m": [0, 0, 0, 0, 0, 2],
    })
    alignment = pd.DataFrame({
        "feature_row_valid": [True] * 5 + [True],
        "source_quality_ok": [True, True, True, True, False, True],
    })
    ok, reasons = eligibility_mask(outcomes, alignment, _spec(), cfg2)
    assert reasons["market_closed_window"] == 1
    assert reasons["incomplete_horizon"] == 1
    assert reasons["outcome_nan"] == 1
    assert reasons["incomplete_htf_source_bar"] == 1
    assert ok.tolist() == [True, False, False, False, False, True]

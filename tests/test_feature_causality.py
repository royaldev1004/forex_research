"""Feature causality and label separation.

The central test: modify future price rows, rebuild everything, and require
that every feature row at or before the cut is bit-identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_config, synthetic_base

from forex_research.pipeline import build_feature_frame, prepare_dataset
from forex_research.outcomes import build_outcomes
from forex_research.validation import (
    ValidationError,
    check_dictionary,
    check_label_separation,
    run_causality_check,
)


@pytest.fixture(scope="module")
def small_cfg():
    return make_config()


@pytest.fixture(scope="module")
def bars():
    return synthetic_base(n=900)


def test_future_mutation_does_not_change_historical_features(small_cfg, bars):
    """The required test: corrupt the future, prove the past is unchanged."""
    cut_pos = int(len(bars) * 0.6)
    cut_time = bars["bar_close_time"].iloc[cut_pos]

    clean, _ = build_feature_frame(bars, small_cfg)

    corrupted = bars.copy()
    future = corrupted["bar_close_time"] > cut_time
    for col in ("open", "high", "low", "close"):
        corrupted.loc[future, col] = corrupted.loc[future, col] * 1.05 + 0.03
    dirty, _ = build_feature_frame(corrupted, small_cfg)

    past = clean["decision_time"] <= cut_time
    feat_cols = [c for c in clean.columns
                 if c not in ("symbol", "decision_time", "feature_row_valid")]
    assert feat_cols, "no feature columns were produced"

    a = clean.loc[past, feat_cols].to_numpy("float32")
    b = dirty.loc[past, feat_cols].to_numpy("float32")

    assert np.array_equal(np.isnan(a), np.isnan(b)), "NaN pattern changed before the cut"
    m = ~np.isnan(a)
    changed = [feat_cols[j] for j in range(len(feat_cols))
               if not np.array_equal(a[~np.isnan(a[:, j]), j], b[~np.isnan(a[:, j]), j])]
    assert not changed, f"{len(changed)} historical feature(s) changed, e.g. {changed[:5]}"
    assert np.array_equal(a[m], b[m])


def test_future_mutation_does_change_future_features(small_cfg, bars):
    """Guards against a vacuous pass of the test above."""
    cut_pos = int(len(bars) * 0.6)
    cut_time = bars["bar_close_time"].iloc[cut_pos]

    clean, _ = build_feature_frame(bars, small_cfg)
    corrupted = bars.copy()
    future = corrupted["bar_close_time"] > cut_time
    for col in ("open", "high", "low", "close"):
        corrupted.loc[future, col] = corrupted.loc[future, col] * 1.05 + 0.03
    dirty, _ = build_feature_frame(corrupted, small_cfg)

    fut = clean["decision_time"] > cut_time
    feat_cols = [c for c in clean.columns
                 if c not in ("symbol", "decision_time", "feature_row_valid")]
    a = clean.loc[fut, feat_cols].to_numpy("float32")
    b = dirty.loc[fut, feat_cols].to_numpy("float32")
    m = ~np.isnan(a) & ~np.isnan(b)
    assert m.any() and not np.array_equal(a[m], b[m])


def test_truncating_the_future_does_not_change_the_past(small_cfg, bars):
    """A second, independent causality argument: fewer future bars, same past."""
    cut = int(len(bars) * 0.7)
    full, _ = build_feature_frame(bars, small_cfg)
    short, _ = build_feature_frame(bars.iloc[:cut].reset_index(drop=True), small_cfg)

    cut_time = bars["bar_close_time"].iloc[cut - 1]
    feat_cols = [c for c in full.columns
                 if c not in ("symbol", "decision_time", "feature_row_valid")]

    # 4h/daily are absent from the small config, so compare on the common span
    # minus one bar of each higher timeframe to avoid the truncated tail bin.
    limit = cut_time - pd.Timedelta(hours=1)
    fa = full[full["decision_time"] <= limit]
    sa = short[short["decision_time"] <= limit]
    n = min(len(fa), len(sa))
    a = fa.iloc[:n][feat_cols].to_numpy("float32")
    b = sa.iloc[:n][feat_cols].to_numpy("float32")

    assert np.array_equal(np.isnan(a), np.isnan(b))
    m = ~np.isnan(a)
    assert np.array_equal(a[m], b[m])


def test_run_causality_check_helper(small_cfg, bars):
    results, detail = run_causality_check(bars, small_cfg, n_rows=len(bars), cut_fraction=0.6)
    by = {r.check: r for r in results}
    assert by["historical_features_unchanged_after_future_mutation"].status == "ok"
    assert by["future_features_did_change"].status == "ok"
    assert detail["historical_rows_identical"].iloc[0]


def test_causality_check_rejects_a_leaky_pipeline(small_cfg, bars, monkeypatch):
    """If a lookahead is introduced, the checker must fail rather than pass."""
    import forex_research.pipeline as pl

    real = pl.prepare_dataset

    def leaky(base, cfg):
        ds = real(base, cfg)
        # inject a lookahead: shift one timeframe's alignment one bar into the future
        tf = cfg.base_timeframe
        idx = ds.alignment.indices[tf]
        ds.alignment.indices[tf] = np.minimum(idx + 1, len(ds.timeframes[tf].frame) - 1)
        return ds

    monkeypatch.setattr(pl, "prepare_dataset", leaky)
    with pytest.raises(ValidationError, match="Causality violation"):
        run_causality_check(bars, small_cfg, n_rows=len(bars), cut_fraction=0.6)


def test_outcome_columns_never_appear_in_features(small_cfg, bars):
    features, ds = build_feature_frame(bars, small_cfg)
    outcomes, _ = build_outcomes(bars, small_cfg)

    feat_cols = set(features.columns) - {"symbol", "decision_time", "feature_row_valid"}
    out_cols = set(outcomes.columns) - {"symbol", "decision_time"}
    assert not (feat_cols & out_cols)
    check_label_separation(list(feat_cols), list(outcomes.columns))


def test_label_separation_detects_a_leak():
    with pytest.raises(ValidationError, match="leaked into the feature matrix"):
        check_label_separation(["a", "fwd_pips_60m"], ["fwd_pips_60m"])


def test_feature_dictionary_covers_every_column(small_cfg, bars):
    ds = prepare_dataset(bars, small_cfg)
    results = check_dictionary(ds.specs, ds.feature_columns)
    assert all(r.status in ("ok", "info") for r in results)


def test_no_feature_is_flagged_as_using_future_data(small_cfg, bars):
    ds = prepare_dataset(bars, small_cfg)
    assert not [s.feature_name for s in ds.specs if s.uses_future_data]


def test_every_outcome_is_flagged_as_using_future_data(small_cfg, bars):
    _, specs = build_outcomes(bars, small_cfg)
    assert specs and all(s.uses_future_data for s in specs)


def test_feature_builder_never_receives_outcome_columns(small_cfg, bars):
    """prepare_dataset must work on a frame that has no outcome columns at all."""
    allowed = {"bar_open_time", "bar_close_time", "bar_open_time_utc", "bar_close_time_utc",
               "open", "high", "low", "close", "tick_volume", "real_volume", "spread_points"}
    assert set(bars.columns) <= allowed
    ds = prepare_dataset(bars, small_cfg)
    assert ds.n_features > 0

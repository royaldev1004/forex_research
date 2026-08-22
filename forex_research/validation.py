from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .alignment import Alignment, assert_no_lookahead
from .config import ResearchConfig, timeframe_minutes
from .data_loader import epoch_seconds
from .feature_dictionary import FeatureSpec
from .logging_utils import get_logger
from .pipeline import build_feature_frame
from .timeframe_builder import TimeframeSeries

log = get_logger("validation")


class ValidationError(RuntimeError):
    pass


@dataclass
class ValidationResult:

    check: str
    scope: str
    value: Any
    status: str
    detail: str = ""


def check_timestamp_integrity(
    timeframes: dict[str, TimeframeSeries], cfg: ResearchConfig
) -> list[ValidationResult]:
    out: list[ValidationResult] = []
    for tf, ts in timeframes.items():
        df = ts.frame
        sorted_ok = bool(df["bar_close_time"].is_monotonic_increasing)
        out.append(ValidationResult("timestamps_sorted", tf, sorted_ok,
                                    "ok" if sorted_ok else "fail"))

        dups = int(df["bar_open_time"].duplicated().sum())
        out.append(ValidationResult("duplicate_bar_open_times", tf, dups,
                                    "ok" if dups == 0 else "fail"))

        expected = pd.Timedelta(minutes=ts.minutes)
        delta_ok = bool(((df["bar_close_time"] - df["bar_open_time"]) == expected).all())
        out.append(ValidationResult("bar_duration_matches_timeframe", tf, delta_ok,
                                    "ok" if delta_ok else "fail",
                                    f"bar_close_time - bar_open_time == {expected}"))

        secs = epoch_seconds(df["bar_open_time"])
        grid_ok = bool((secs % (ts.minutes * 60) == 0).all())
        out.append(ValidationResult("bars_on_timeframe_grid", tf, grid_ok,
                                    "ok" if grid_ok else "fail"))

        gaps = df["bar_open_time"].diff().dropna()
        min_gap = gaps.min() if len(gaps) else pd.Timedelta(0)
        out.append(ValidationResult("min_bar_spacing", tf, str(min_gap),
                                    "ok" if min_gap >= expected else "fail",
                                    "bars must never be closer together than one bar"))

        n_missing = int((df["n_base_bars"] < df["expected_base_bars"]).sum())
        out.append(ValidationResult("bars_with_missing_base_data", tf, n_missing,
                                    "ok" if n_missing == 0 else "warn",
                                    "holiday sessions and feed gaps; documented, not repaired"))
    return out


def check_label_separation(
    feature_columns: list[str], outcome_columns: list[str]
) -> list[ValidationResult]:
    fset, oset = set(feature_columns), set(outcome_columns)
    overlap = sorted(fset & oset)
    status = "ok" if not overlap else "fail"
    results = [ValidationResult("feature_outcome_column_overlap", "dataset", len(overlap),
                                status, ", ".join(overlap[:10]))]

    # Defence in depth: outcome names are recognisable by prefix.
    suspicious = sorted(
        c for c in fset
        if c.startswith(("fwd_", "mfe_", "mae_", "expansion_", "time_to_", "horizon_"))
    )
    results.append(ValidationResult(
        "feature_columns_with_outcome_like_names", "dataset", len(suspicious),
        "ok" if not suspicious else "fail", ", ".join(suspicious[:10])))

    if overlap or suspicious:
        raise ValidationError(
            f"Outcome columns leaked into the feature matrix: {(overlap + suspicious)[:10]}"
        )
    return results


def check_dictionary(specs: list[FeatureSpec], columns: list[str]) -> list[ValidationResult]:
    names = [s.feature_name for s in specs]
    leaking = [s.feature_name for s in specs if s.uses_future_data]
    missing = sorted(set(columns) - set(names))
    extra = sorted(set(names) - set(columns))

    results = [
        ValidationResult("features_documented", "dictionary", len(missing) == 0,
                         "ok" if not missing else "fail", ", ".join(missing[:10])),
        ValidationResult("dictionary_entries_without_column", "dictionary", len(extra),
                         "ok" if not extra else "fail", ", ".join(extra[:10])),
        ValidationResult("feature_specs_uses_future_data", "dictionary", len(leaking),
                         "ok" if not leaking else "fail", ", ".join(leaking[:10])),
        ValidationResult("duplicate_feature_names", "dictionary",
                         len(names) - len(set(names)),
                         "ok" if len(names) == len(set(names)) else "fail"),
    ]
    if missing or extra or leaking:
        raise ValidationError(
            "Feature dictionary is inconsistent with the feature matrix "
            f"(missing={len(missing)}, extra={len(extra)}, future_flagged={len(leaking)})."
        )
    return results


def check_alignment(alignment: Alignment, cfg: ResearchConfig) -> tuple[pd.DataFrame, list[ValidationResult]]:
    table = assert_no_lookahead(alignment)
    results = [ValidationResult("no_lookahead_assertion", "alignment", "passed", "ok",
                                "source_bar_close_time <= decision_time on every timeframe")]

    for row in table.itertuples():
        tf = row.timeframe
        limit = timeframe_minutes(tf)
        # A freshly closed bar has age 0; it can never be older than one full bar
        # of its own timeframe unless the feed had a gap.
        within = np.isnan(row.max_age_minutes) or row.max_age_minutes >= 0
        results.append(ValidationResult(
            "source_bar_age_non_negative", tf, round(float(row.min_age_minutes), 3),
            "ok" if within and (np.isnan(row.min_age_minutes) or row.min_age_minutes >= 0) else "fail",
            f"minimum age in minutes; must be >= 0 (one {tf} bar is {limit}m)"))
        results.append(ValidationResult(
            "source_bar_max_age_minutes", tf, round(float(row.max_age_minutes), 3), "info",
            "large values indicate weekend/holiday gaps, which are expected"))
    return table, results


def check_resampling(
    base: pd.DataFrame, timeframes: dict[str, TimeframeSeries], cfg: ResearchConfig
) -> list[ValidationResult]:
    out: list[ValidationResult] = []
    base_min = timeframe_minutes(cfg.base_timeframe)
    for tf, ts in timeframes.items():
        if ts.minutes == base_min:
            continue
        step = ts.minutes * 60
        key = (epoch_seconds(base["bar_open_time"]) // step) * step
        g = base.groupby(key.rename("bin"), sort=True).agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"))
        g.index = pd.to_datetime(g.index, unit="s")

        built = ts.frame.set_index("bar_open_time")[["open", "high", "low", "close"]]
        common = built.index.intersection(g.index)
        a = built.loc[common].to_numpy()
        b = g.loc[common].to_numpy()
        match = bool(np.array_equal(a, b, equal_nan=True))
        out.append(ValidationResult(
            "resampling_matches_independent_recompute", tf, match,
            "ok" if match else "fail",
            f"{len(common)} bins compared against a groupby-based recomputation"))
        if not match:
            raise ValidationError(f"{tf}: resampled OHLC does not match independent recomputation.")
    return out


def run_causality_check(
    base: pd.DataFrame, cfg: ResearchConfig, n_rows: int, cut_fraction: float = 0.6
) -> tuple[list[ValidationResult], pd.DataFrame]:
    slice_df = base.iloc[:n_rows].reset_index(drop=True).copy()
    if len(slice_df) < 100:
        raise ValidationError("Causality check needs at least 100 base bars.")

    cut_pos = int(len(slice_df) * cut_fraction)
    cut_time = slice_df["bar_close_time"].iloc[cut_pos]

    log.info("Causality check: %d bars, cut at %s (row %d)", len(slice_df), cut_time, cut_pos)

    clean, _ = build_feature_frame(slice_df, cfg)

    corrupted = slice_df.copy()
    future = corrupted["bar_close_time"] > cut_time
    bump = 0.05  # ~500 pips: far outside any plausible real move
    for col in ("open", "high", "low", "close"):
        corrupted.loc[future, col] = corrupted.loc[future, col] + bump
    # also scramble the ordering of the future path so it is not merely shifted
    rng = np.random.default_rng(cfg.random_seed)
    n_future = int(future.sum())
    if n_future > 1:
        noise = rng.normal(0.0, 0.002, n_future)
        corrupted.loc[future, "close"] = corrupted.loc[future, "close"] + noise
        corrupted.loc[future, "high"] = np.maximum(
            corrupted.loc[future, "high"], corrupted.loc[future, "close"])
        corrupted.loc[future, "low"] = np.minimum(
            corrupted.loc[future, "low"], corrupted.loc[future, "close"])

    dirty, _ = build_feature_frame(corrupted, cfg)

    past = clean["decision_time"] <= cut_time
    feat_cols = [c for c in clean.columns
                 if c not in ("symbol", "decision_time", "feature_row_valid")]

    a = clean.loc[past, feat_cols].to_numpy(dtype="float32")
    b = dirty.loc[past, feat_cols].to_numpy(dtype="float32")

    same_nan = np.array_equal(np.isnan(a), np.isnan(b))
    both = ~np.isnan(a) & ~np.isnan(b)
    identical = bool(same_nan and np.array_equal(a[both], b[both]))

    n_changed_cols = 0
    changed_examples: list[str] = []
    if not identical:
        for j, c in enumerate(feat_cols):
            ca, cb = a[:, j], b[:, j]
            if not np.array_equal(np.isnan(ca), np.isnan(cb)):
                n_changed_cols += 1
            else:
                m = ~np.isnan(ca)
                if not np.array_equal(ca[m], cb[m]):
                    n_changed_cols += 1
            if n_changed_cols and len(changed_examples) < 5 and c not in changed_examples:
                changed_examples.append(c)

    # sanity: the corruption must actually have changed *later* rows, otherwise
    # the test would pass trivially
    fut = clean["decision_time"] > cut_time
    af = clean.loc[fut, feat_cols].to_numpy(dtype="float32")
    bf = dirty.loc[fut, feat_cols].to_numpy(dtype="float32")
    m = ~np.isnan(af) & ~np.isnan(bf)
    future_changed = bool(m.any() and not np.array_equal(af[m], bf[m]))

    results = [
        ValidationResult("causality_rows_compared", "features", int(past.sum()), "info",
                         f"decision_time <= {cut_time}"),
        ValidationResult("causality_features_compared", "features", len(feat_cols), "info"),
        ValidationResult("historical_features_unchanged_after_future_mutation", "features",
                         identical, "ok" if identical else "fail",
                         "" if identical
                         else f"{n_changed_cols} column(s) changed, e.g. {changed_examples}"),
        ValidationResult("future_features_did_change", "features", future_changed,
                         "ok" if future_changed else "fail",
                         "confirms the mutation was actually applied, so the test is not vacuous"),
    ]

    if not identical:
        raise ValidationError(
            f"Causality violation: {n_changed_cols} historical feature column(s) changed when "
            f"future data was modified. Examples: {changed_examples}"
        )
    if not future_changed:
        raise ValidationError(
            "Causality check was vacuous: mutating future prices changed nothing at all."
        )

    detail = pd.DataFrame({
        "cut_time": [str(cut_time)],
        "rows_before_cut": [int(past.sum())],
        "rows_after_cut": [int(fut.sum())],
        "features_compared": [len(feat_cols)],
        "historical_rows_identical": [identical],
        "future_rows_changed": [future_changed],
    })
    log.info("Causality check passed: %d historical rows x %d features unchanged.",
             int(past.sum()), len(feat_cols))
    return results, detail


def check_missing_data_policy(
    ds_valid: np.ndarray, cfg: ResearchConfig
) -> list[ValidationResult]:
    n_valid = int(ds_valid.sum())
    n_total = int(len(ds_valid))
    ok = n_valid > 0
    results = [
        ValidationResult("valid_feature_rows", "dataset", n_valid, "ok" if ok else "fail",
                         f"{n_valid}/{n_total} rows have every timeframe warmed up"),
        ValidationResult("valid_row_fraction", "dataset",
                         round(n_valid / n_total, 4) if n_total else 0.0,
                         "ok" if ok else "fail"),
    ]
    if not ok:
        raise ValidationError(
            "No decision row satisfies the warm-up requirement on every timeframe. "
            f"Warm-up needs {cfg.warmup_bars()} bars per timeframe; the longest configured "
            f"moving-average period is {cfg.max_indicator_period()}. Either supply more "
            "history, shorten the period list, or drop the slowest timeframe."
        )
    return results


def to_frame(results: list[ValidationResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])[
        ["check", "scope", "value", "status", "detail"]
    ]

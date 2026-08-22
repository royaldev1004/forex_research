from __future__ import annotations

import numpy as np
import pandas as pd

from .event_definitions import EventSpec, Step2Config, session_of
from .logging_utils import get_logger

log = get_logger("event_detection")

COMPRESSED = "compressed"
NEUTRAL = "neutral"
EXPANDING = "expanding"


def add_time_context(df: pd.DataFrame, cfg2: Step2Config) -> pd.DataFrame:
    out = df.copy()
    dt = pd.DatetimeIndex(out["decision_time"])
    out["hour"] = dt.hour
    out["session"] = [session_of(h) for h in dt.hour]
    out["hour_bucket"] = (dt.hour // cfg2.hour_bucket_hours) * cfg2.hour_bucket_hours
    out["day_of_week"] = dt.dayofweek
    return out


def classify_compression(
    features: pd.DataFrame, cfg2: Step2Config
) -> pd.Series:
    col = cfg2.compression_primary_column
    if col not in features.columns:
        raise KeyError(
            f"Compression column {col!r} is not in the feature matrix. "
            f"Set step2.compression_primary_column to one of the available "
            f"'*__compression_percentile_*' columns."
        )
    pct = features[col]
    state = pd.Series(NEUTRAL, index=features.index, dtype="object")
    state[pct <= cfg2.compression_compressed_max_percentile] = COMPRESSED
    state[pct >= cfg2.compression_expanding_min_percentile] = EXPANDING
    state[pct.isna()] = "unknown"
    return state


def eligibility_mask(
    outcomes: pd.DataFrame,
    alignment: pd.DataFrame,
    spec: EventSpec,
    cfg2: Step2Config,
) -> tuple[pd.Series, dict[str, int]]:
    h = spec.horizon_minutes
    n = len(outcomes)
    reasons: dict[str, int] = {}

    ok = pd.Series(True, index=outcomes.index)

    valid = alignment["feature_row_valid"].to_numpy()
    reasons["feature_row_invalid"] = int((~valid).sum())
    ok &= valid

    if cfg2.require_complete_horizon:
        comp = outcomes[f"horizon_complete_{h}m"] == 1
        reasons["incomplete_horizon"] = int((ok & ~comp).sum())
        ok &= comp

    if cfg2.require_observed_bars:
        obs = outcomes[f"bars_in_window_{h}m"] > 0
        reasons["market_closed_window"] = int((ok & ~obs).sum())
        ok &= obs

    notna = outcomes[f"max_future_delta_pips_{h}m"].notna()
    reasons["outcome_nan"] = int((ok & ~notna).sum())
    ok &= notna

    if cfg2.require_source_quality:
        if "source_quality_ok" in alignment.columns:
            sq = alignment["source_quality_ok"].to_numpy().astype(bool)
            reasons["incomplete_htf_source_bar"] = int((ok & ~sq).sum())
            ok &= sq
        else:
            log.warning("source_quality_ok missing from alignment map; quality filter skipped.")
            reasons["incomplete_htf_source_bar"] = 0

    if cfg2.exclude_ambiguous_first_touch:
        amb = outcomes[f"expansion_first_{spec.threshold_pips:g}p_{h}m"] == 2
        reasons["ambiguous_first_touch"] = int((ok & amb).sum())
        ok &= ~amb

    reasons["eligible"] = int(ok.sum())
    reasons["total_rows"] = n
    return ok, reasons


def find_candidates(
    outcomes: pd.DataFrame,
    alignment: pd.DataFrame,
    features: pd.DataFrame,
    spec: EventSpec,
    cfg2: Step2Config,
) -> tuple[pd.DataFrame, dict[str, int]]:
    eligible, reasons = eligibility_mask(outcomes, alignment, spec, cfg2)
    hit = outcomes[spec.expansion_column] == 1
    mask = eligible & hit
    reasons["candidates"] = int(mask.sum())

    h = spec.horizon_minutes
    cand = pd.DataFrame({
        "decision_time": outcomes.loc[mask, "decision_time"].to_numpy(),
        "time_to_threshold_minutes": outcomes.loc[mask, spec.time_to_column].to_numpy(),
        "max_future_delta_pips": outcomes.loc[mask, f"max_future_delta_pips_{h}m"].to_numpy(),
        "min_future_delta_pips": outcomes.loc[mask, f"min_future_delta_pips_{h}m"].to_numpy(),
        "fwd_pips": outcomes.loc[mask, f"fwd_pips_{h}m"].to_numpy(),
        "opposite_expansion": outcomes.loc[mask, spec.opposite_expansion_column].to_numpy(),
        "first_touch": outcomes.loc[
            mask, f"expansion_first_{spec.threshold_pips:g}p_{h}m"].to_numpy(),
    }).sort_values("decision_time").reset_index(drop=True)

    comp = classify_compression(features, cfg2)
    comp_by_time = pd.Series(comp.to_numpy(), index=pd.DatetimeIndex(features["decision_time"]))
    cand["compression_state"] = comp_by_time.reindex(
        pd.DatetimeIndex(cand["decision_time"])).to_numpy()

    log.info("%-18s candidates: %6d (of %d eligible rows)",
             spec.tag, len(cand), reasons["eligible"])
    return cand, reasons


def find_compressed_rows(
    outcomes: pd.DataFrame,
    alignment: pd.DataFrame,
    features: pd.DataFrame,
    spec: EventSpec,
    cfg2: Step2Config,
    expanded: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    eligible, reasons = eligibility_mask(outcomes, alignment, spec, cfg2)
    comp = classify_compression(features, cfg2)
    is_comp = pd.Series(comp.to_numpy() == COMPRESSED, index=outcomes.index)

    thr, h = spec.threshold_pips, spec.horizon_minutes
    up = outcomes[f"expansion_up_{thr:g}p_{h}m"] == 1
    dn = outcomes[f"expansion_down_{thr:g}p_{h}m"] == 1

    if expanded:
        target = up if spec.direction == "up" else dn
    else:
        target = ~(up | dn)      # no expansion either way

    mask = eligible & is_comp & target
    reasons["compressed_eligible"] = int((eligible & is_comp).sum())
    reasons["candidates"] = int(mask.sum())

    out = pd.DataFrame({
        "decision_time": outcomes.loc[mask, "decision_time"].to_numpy(),
        "max_future_delta_pips": outcomes.loc[mask, f"max_future_delta_pips_{h}m"].to_numpy(),
        "min_future_delta_pips": outcomes.loc[mask, f"min_future_delta_pips_{h}m"].to_numpy(),
        "fwd_pips": outcomes.loc[mask, f"fwd_pips_{h}m"].to_numpy(),
        "compression_state": COMPRESSED,
    }).sort_values("decision_time").reset_index(drop=True)
    if expanded:
        out["time_to_threshold_minutes"] = outcomes.loc[mask, spec.time_to_column].to_numpy()
    return out, reasons


def context_label(direction: str, compressed: bool, expanded: bool) -> str:
    if not expanded:
        return "compressed_no_major_expansion"
    suffix = "after_compression" if compressed else "without_compression"
    return f"major_{direction}_{suffix}"

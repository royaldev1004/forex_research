from __future__ import annotations

import numpy as np
import pandas as pd

from .event_definitions import EventSpec, Step2Config
from .logging_utils import get_logger

log = get_logger("event_deduplication")


def collapse_to_episodes(
    candidates: pd.DataFrame,
    spec: EventSpec,
    cfg2: Step2Config,
    prefix: str = "evt",
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=[
            "event_id", "event_direction", "event_threshold_pips", "event_horizon_minutes",
            "event_start_time", "canonical_decision_time", "event_end_time",
            "event_max_up_pips", "event_max_down_pips", "event_duration_minutes",
            "n_candidates_collapsed", "compression_state",
        ])

    cand = candidates.sort_values("decision_time").reset_index(drop=True)
    times = pd.DatetimeIndex(cand["decision_time"])
    h = spec.horizon_minutes
    extra = pd.Timedelta(minutes=cfg2.extra_separation_minutes)

    episodes: list[dict] = []
    claimed_until: pd.Timestamp | None = None
    current: dict | None = None

    for i in range(len(cand)):
        t = times[i]
        if claimed_until is not None and t <= claimed_until:
            current["n_candidates_collapsed"] += 1
            current["_last_candidate_time"] = t
            # widen the observed move using rows inside the same episode
            current["event_max_up_pips"] = max(
                current["event_max_up_pips"], float(cand["max_future_delta_pips"].iloc[i]))
            current["event_max_down_pips"] = min(
                current["event_max_down_pips"], float(cand["min_future_delta_pips"].iloc[i]))
            continue

        if current is not None:
            episodes.append(current)

        row = cand.iloc[i]
        ttt = float(row.get("time_to_threshold_minutes", np.nan))
        if cfg2.suppression_rule == "to_threshold" and np.isfinite(ttt):
            span = pd.Timedelta(minutes=ttt)
        else:
            span = pd.Timedelta(minutes=h)

        current = {
            "event_direction": spec.direction,
            "event_threshold_pips": spec.threshold_pips,
            "event_horizon_minutes": h,
            "event_start_time": t,
            "canonical_decision_time": t,
            "event_end_time": t + span,
            "event_max_up_pips": float(row["max_future_delta_pips"]),
            "event_max_down_pips": float(row["min_future_delta_pips"]),
            "event_duration_minutes": ttt,
            "n_candidates_collapsed": 1,
            "compression_state": row.get("compression_state", "unknown"),
            "opposite_expansion": float(row.get("opposite_expansion", np.nan)),
            "first_touch": float(row.get("first_touch", np.nan)),
            "fwd_pips": float(row.get("fwd_pips", np.nan)),
            "_last_candidate_time": t,
        }
        claimed_until = t + span + extra

    if current is not None:
        episodes.append(current)

    ep = pd.DataFrame(episodes)
    ep["event_id"] = [f"{prefix}_{spec.tag}_{i:05d}" for i in range(len(ep))]
    ep = ep.drop(columns=["_last_candidate_time"])

    cols = ["event_id", "event_direction", "event_threshold_pips", "event_horizon_minutes",
            "event_start_time", "canonical_decision_time", "event_end_time",
            "event_max_up_pips", "event_max_down_pips", "event_duration_minutes",
            "n_candidates_collapsed", "compression_state"]
    rest = [c for c in ep.columns if c not in cols]
    ep = ep[cols + rest]

    log.info("%-18s episodes: %5d from %6d candidates (%.1f rows collapsed per episode)",
             spec.tag, len(ep), len(cand), len(cand) / max(1, len(ep)))
    return ep


def collapse_by_separation(
    rows: pd.DataFrame, separation_minutes: int, prefix: str, spec: EventSpec
) -> pd.DataFrame:
    if rows.empty:
        return rows.assign(event_id=pd.Series(dtype="object"),
                           n_candidates_collapsed=pd.Series(dtype="int64"))

    r = rows.sort_values("decision_time").reset_index(drop=True)
    times = pd.DatetimeIndex(r["decision_time"])
    gap = pd.Timedelta(minutes=separation_minutes)

    keep: list[int] = []
    counts: list[int] = []
    last: pd.Timestamp | None = None
    for i, t in enumerate(times):
        if last is None or t - last >= gap:
            keep.append(i)
            counts.append(1)
            last = t
        else:
            counts[-1] += 1

    out = r.iloc[keep].reset_index(drop=True)
    out["n_candidates_collapsed"] = counts
    out["event_id"] = [f"{prefix}_{spec.tag}_{i:05d}" for i in range(len(out))]
    out["canonical_decision_time"] = out["decision_time"]
    log.info("%-18s %s episodes: %5d from %6d rows", spec.tag, prefix, len(out), len(r))
    return out


def overlap_report(
    per_spec: dict[str, tuple[int, int, int]]
) -> pd.DataFrame:
    rows = []
    for tag, (eligible, candidates, episodes) in per_spec.items():
        rows.append({
            "stratum": tag,
            "eligible_rows": eligible,
            "raw_candidate_rows": candidates,
            "unique_episodes": episodes,
            "rows_suppressed": candidates - episodes,
            "avg_rows_per_episode": round(candidates / episodes, 2) if episodes else 0.0,
            "candidate_rate_pct": round(100 * candidates / eligible, 4) if eligible else 0.0,
        })
    return pd.DataFrame(rows)


def build_purge_intervals(
    episodes: pd.DataFrame, purge_minutes: int, horizon_minutes: int
) -> np.ndarray:
    if episodes.empty:
        return np.empty((0, 2), dtype="datetime64[ns]")
    start = pd.DatetimeIndex(episodes["event_start_time"]).to_numpy()
    end = pd.DatetimeIndex(episodes["event_end_time"]).to_numpy()
    pad = np.timedelta64(purge_minutes, "m")
    look = np.timedelta64(horizon_minutes, "m")
    lo = start - pad - look
    hi = end + pad
    return np.stack([lo, hi], axis=1)


def in_any_interval(times: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    if len(intervals) == 0 or len(times) == 0:
        return np.zeros(len(times), dtype=bool)
    order = np.argsort(intervals[:, 0])
    lo = intervals[order, 0]
    hi = intervals[order, 1]
    # running maximum of interval ends lets a single searchsorted decide membership
    hi_max = np.maximum.accumulate(hi)
    idx = np.searchsorted(lo, times, side="right") - 1
    inside = np.zeros(len(times), dtype=bool)
    ok = idx >= 0
    inside[ok] = times[ok] < hi_max[idx[ok]]
    return inside

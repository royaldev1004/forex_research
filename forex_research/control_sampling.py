from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .event_deduplication import build_purge_intervals, in_any_interval
from .event_definitions import EventSpec, Step2Config
from .logging_utils import get_logger

log = get_logger("control_sampling")


@dataclass
class ControlPool:

    frame: pd.DataFrame
    n_before_purge: int
    n_purged: int
    n_after_purge: int


def volatility_bins(
    values: pd.Series, n_bins: int, reference: pd.Series | None = None
) -> pd.Series:
    ref = reference if reference is not None else values
    ref = ref.dropna()
    if ref.empty:
        return pd.Series(np.full(len(values), -1), index=values.index, dtype="int64")
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.nanquantile(ref.to_numpy(), qs))
    if len(edges) < 2:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype="int64")
    edges[0] = -np.inf
    edges[-1] = np.inf
    binned = pd.cut(values, bins=edges, labels=False, include_lowest=True)
    return binned.fillna(-1).astype("int64")


def compression_bins(pct: pd.Series, n_bins: int = 4) -> pd.Series:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return pd.cut(pct, bins=edges, labels=False, include_lowest=True).fillna(-1).astype("int64")


def build_control_pool(
    context: pd.DataFrame,
    eligible: pd.Series,
    non_event: pd.Series,
    episodes: pd.DataFrame,
    spec: EventSpec,
    cfg2: Step2Config,
) -> ControlPool:
    base = eligible & non_event
    n_before = int(base.sum())

    intervals = build_purge_intervals(
        episodes, cfg2.control_purge_minutes, spec.horizon_minutes)
    times = pd.DatetimeIndex(context["decision_time"]).to_numpy()
    purged = in_any_interval(times, intervals)

    keep = base.to_numpy() & ~purged
    n_purged = int((base.to_numpy() & purged).sum())

    pool = context.loc[keep].copy()
    log.info(
        "%-18s control pool: %6d eligible non-event rows, %6d purged, %6d usable",
        spec.tag, n_before, n_purged, len(pool),
    )
    return ControlPool(frame=pool, n_before_purge=n_before,
                       n_purged=n_purged, n_after_purge=len(pool))


def _match_keys(df: pd.DataFrame, cfg2: Step2Config, use_compression: bool) -> pd.Series:
    parts: list[pd.Series] = []
    if cfg2.match_on_session:
        parts.append(df["session"].astype(str))
    if cfg2.match_on_hour_bucket:
        parts.append("hb" + df["hour_bucket"].astype(str))
    if cfg2.match_on_day_of_week:
        parts.append("dow" + df["day_of_week"].astype(str))
    parts.append("vol" + df["volatility_bin"].astype(str))
    if use_compression:
        parts.append("cmp" + df["compression_bin"].astype(str))
    if not parts:
        return pd.Series("all", index=df.index)
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    return key


def sample_matched_controls(
    events: pd.DataFrame,
    pool: pd.DataFrame,
    spec: EventSpec,
    cfg2: Step2Config,
    control_type: str,
    use_compression: bool,
    n_per_event: int,
    seed_offset: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty or pool.empty:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(cfg2.random_seed + seed_offset)

    pool = pool.copy()
    pool["match_key"] = _match_keys(pool, cfg2, use_compression)
    ev = events.copy()
    ev["match_key"] = _match_keys(ev, cfg2, use_compression)

    buckets: dict[str, np.ndarray] = {}
    for key, grp in pool.groupby("match_key", sort=False):
        arr = np.array(grp.index.to_numpy(), copy=True)   # index arrays are read-only
        rng.shuffle(arr)
        buckets[key] = arr
    cursor: dict[str, int] = {k: 0 for k in buckets}

    rows: list[dict] = []
    matched_counts: list[dict] = []

    for e in ev.itertuples():
        key = e.match_key
        avail = buckets.get(key)
        got = 0
        if avail is not None and len(avail):
            start = cursor[key]
            take = avail[start:start + n_per_event]
            if len(take) < n_per_event:          # wrap once, cell is small
                take = np.concatenate([take, avail[: n_per_event - len(take)]])
            cursor[key] = (start + n_per_event) % max(1, len(avail))
            for idx in take:
                r = pool.loc[idx]
                rows.append({
                    "control_id": f"ctl_{control_type}_{spec.tag}_{len(rows):06d}",
                    "control_type": control_type,
                    "matched_event_id": e.event_id,
                    "control_decision_time": r["decision_time"],
                    "stratum": spec.tag,
                    "event_direction": spec.direction,
                    "event_threshold_pips": spec.threshold_pips,
                    "event_horizon_minutes": spec.horizon_minutes,
                    "matching_session": r["session"],
                    "matching_hour_bucket": int(r["hour_bucket"]),
                    "matching_day_of_week": int(r["day_of_week"]),
                    "matching_volatility_bin": int(r["volatility_bin"]),
                    "matching_compression_bin": (
                        int(r["compression_bin"]) if use_compression else -1),
                    "matching_key": key,
                    "compression_state": r.get("compression_state", "unknown"),
                })
                got += 1
        matched_counts.append({
            "event_id": e.event_id, "matching_key": key,
            "controls_requested": n_per_event, "controls_found": got,
            "cell_supply": 0 if avail is None else len(avail),
        })

    controls = pd.DataFrame(rows)
    report = pd.DataFrame(matched_counts)
    if not report.empty:
        log.info("%-18s %-22s: %5d controls for %4d events (match rate %.1f%%)",
                 spec.tag, control_type, len(controls), len(ev),
                 100 * float((report["controls_found"] > 0).mean()))
    return controls, report


def sample_random_controls(
    events: pd.DataFrame,
    pool: pd.DataFrame,
    spec: EventSpec,
    cfg2: Step2Config,
    n_total: int,
) -> pd.DataFrame:
    if events.empty or pool.empty or n_total <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(cfg2.random_seed + 991)
    share = events["session"].value_counts(normalize=True)

    picks: list[int] = []
    for session, frac in share.items():
        want = int(round(frac * n_total))
        cell = pool.index[pool["session"] == session].to_numpy()
        if want <= 0 or len(cell) == 0:
            continue
        take = rng.choice(cell, size=min(want, len(cell)), replace=False)
        picks.extend(take.tolist())

    if not picks:
        return pd.DataFrame()

    sel = pool.loc[picks]
    return pd.DataFrame({
        "control_id": [f"ctl_random_{spec.tag}_{i:06d}" for i in range(len(sel))],
        "control_type": "random",
        "matched_event_id": "",
        "control_decision_time": sel["decision_time"].to_numpy(),
        "stratum": spec.tag,
        "event_direction": spec.direction,
        "event_threshold_pips": spec.threshold_pips,
        "event_horizon_minutes": spec.horizon_minutes,
        "matching_session": sel["session"].to_numpy(),
        "matching_hour_bucket": sel["hour_bucket"].to_numpy().astype(int),
        "matching_day_of_week": sel["day_of_week"].to_numpy().astype(int),
        "matching_volatility_bin": sel["volatility_bin"].to_numpy().astype(int),
        "matching_compression_bin": -1,
        "matching_key": "session_stratified",
        "compression_state": sel.get(
            "compression_state", pd.Series(["unknown"] * len(sel))).to_numpy(),
    })


def balance_report(
    events: pd.DataFrame, controls: pd.DataFrame, column: str, label: str
) -> pd.DataFrame:
    if events.empty or controls.empty:
        return pd.DataFrame()
    ev = events[column].value_counts(normalize=True).rename("event_share")
    ct = controls[column].value_counts(normalize=True).rename("control_share")
    out = pd.concat([ev, ct], axis=1).fillna(0.0).reset_index()
    out = out.rename(columns={out.columns[0]: label})
    out["abs_difference"] = (out["event_share"] - out["control_share"]).abs()
    out["event_share"] = out["event_share"].round(4)
    out["control_share"] = out["control_share"].round(4)
    out["abs_difference"] = out["abs_difference"].round(4)
    return out.sort_values(label).reset_index(drop=True)

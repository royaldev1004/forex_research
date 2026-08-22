from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .logging_utils import get_logger

log = get_logger("risk_set_controls")

#: Matching variables. All are observable at the decision timestamp.
BASE_MATCH_KEYS = ("session", "hour_bucket", "volatility_bin")

#: Columns the matcher is allowed to see. Anything outcome-shaped is rejected.
FORBIDDEN_PREFIXES = ("expansion_", "fwd_", "long_mfe", "long_mae", "short_mfe",
                      "short_mae", "max_future_delta", "min_future_delta",
                      "time_to_", "horizon_complete", "bars_in_window",
                      "max_abs_move", "future_range")


class OutcomeLeakError(RuntimeError):
    pass


def assert_no_outcome_keys(keys: tuple[str, ...]) -> None:
    bad = [k for k in keys if k.startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise OutcomeLeakError(
            f"Matching keys must be observable at the decision time; these are "
            f"future-derived: {bad}"
        )


def build_match_key(df: pd.DataFrame, keys: tuple[str, ...]) -> pd.Series:
    assert_no_outcome_keys(keys)
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError(f"Missing matching columns: {missing}")
    key = df[keys[0]].astype(str)
    for k in keys[1:]:
        key = key + "|" + k + "=" + df[k].astype(str)
    return key


@dataclass
class RiskSetPool:

    frame: pd.DataFrame
    n_eligible: int
    n_excluded_identity: int
    n_positive_outcome_retained: int


def build_risk_set_pool(
    context: pd.DataFrame,
    eligible: pd.Series,
    event_times: np.ndarray,
    outcome_flag: pd.Series | None = None,
) -> RiskSetPool:
    base = eligible.to_numpy().astype(bool)
    n_eligible = int(base.sum())

    is_event = np.asarray(pd.DatetimeIndex(context["decision_time"]).isin(
        pd.DatetimeIndex(event_times)))
    keep = base & ~is_event
    n_identity = int((base & is_event).sum())

    pool = context.loc[keep].copy()
    n_pos = 0
    if outcome_flag is not None:
        n_pos = int((outcome_flag.to_numpy()[keep] == 1).sum())

    log.info(
        "risk-set pool: %d eligible, %d removed as the event's own timestamp, "
        "%d usable (%d of which later expand and are DELIBERATELY retained)",
        n_eligible, n_identity, len(pool), n_pos,
    )
    return RiskSetPool(frame=pool, n_eligible=n_eligible,
                       n_excluded_identity=n_identity,
                       n_positive_outcome_retained=n_pos)


def sample_risk_set_controls(
    events: pd.DataFrame,
    pool: pd.DataFrame,
    n_per_event: int,
    seed: int,
    control_type: str = "risk_set_matched",
    match_keys: tuple[str, ...] = BASE_MATCH_KEYS,
    event_time_column: str = "canonical_decision_time",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert_no_outcome_keys(match_keys)
    if events.empty or pool.empty:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(seed)
    pool = pool.copy()
    pool["match_key"] = build_match_key(pool, match_keys)
    ev = events.copy()
    ev["match_key"] = build_match_key(ev, match_keys)

    buckets: dict[str, np.ndarray] = {}
    for key, grp in pool.groupby("match_key", sort=False):
        arr = np.array(grp.index.to_numpy(), copy=True)
        rng.shuffle(arr)
        buckets[key] = arr
    cursor = dict.fromkeys(buckets, 0)

    rows: list[dict] = []
    report: list[dict] = []

    for e in ev.itertuples():
        key = e.match_key
        avail = buckets.get(key)
        found = 0
        if avail is not None and len(avail):
            start = cursor[key]
            take = avail[start:start + n_per_event]
            if len(take) < n_per_event:
                take = np.concatenate([take, avail[: n_per_event - len(take)]])
            cursor[key] = (start + n_per_event) % max(1, len(avail))
            for idx in take:
                r = pool.loc[idx]
                rows.append({
                    "control_id": f"{control_type}_{len(rows):07d}",
                    "control_type": control_type,
                    "matched_event_id": getattr(e, "event_id", ""),
                    "control_decision_time": r["decision_time"],
                    "stratum": getattr(e, "stratum", ""),
                    "matching_session": r["session"],
                    "matching_hour_bucket": int(r["hour_bucket"]),
                    "matching_volatility_bin": int(r["volatility_bin"]),
                    "matching_compression_bin": int(r.get("compression_bin", -1)),
                    "matching_key": key,
                })
                found += 1
        report.append({
            "event_id": getattr(e, "event_id", ""),
            "matching_key": key,
            "controls_requested": n_per_event,
            "controls_found": found,
            "cell_supply": 0 if avail is None else len(avail),
        })

    controls = pd.DataFrame(rows)
    rep = pd.DataFrame(report)
    if not rep.empty:
        log.info("%s: %d controls for %d events (match rate %.1f%%)",
                 control_type, len(controls), len(ev),
                 100 * float((rep["controls_found"] > 0).mean()))
    return controls, rep


def sample_random_context_controls(
    events: pd.DataFrame,
    pool: pd.DataFrame,
    n_total: int,
    seed: int,
    match_keys: tuple[str, ...] = BASE_MATCH_KEYS,
) -> pd.DataFrame:
    assert_no_outcome_keys(match_keys)
    if events.empty or pool.empty or n_total <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    pool = pool.copy()
    pool["match_key"] = build_match_key(pool, match_keys)
    ev_key = build_match_key(events, match_keys)
    share = ev_key.value_counts(normalize=True)

    picks: list[int] = []
    for key, frac in share.items():
        cell = pool.index[pool["match_key"] == key].to_numpy()
        want = int(round(frac * n_total))
        if want <= 0 or len(cell) == 0:
            continue
        picks.extend(rng.choice(cell, size=min(want, len(cell)), replace=False).tolist())

    if not picks:
        return pd.DataFrame()
    sel = pool.loc[picks]
    return pd.DataFrame({
        "control_id": [f"random_context_{i:07d}" for i in range(len(sel))],
        "control_type": "random_context_matched",
        "matched_event_id": "",
        "control_decision_time": sel["decision_time"].to_numpy(),
        "matching_session": sel["session"].to_numpy(),
        "matching_hour_bucket": sel["hour_bucket"].to_numpy().astype(int),
        "matching_volatility_bin": sel["volatility_bin"].to_numpy().astype(int),
        "matching_compression_bin": sel.get(
            "compression_bin", pd.Series(-1, index=sel.index)).to_numpy().astype(int),
        "matching_key": sel["match_key"].to_numpy(),
    })


def standardised_difference(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna().astype("float64"), b.dropna().astype("float64")
    if a.empty or b.empty:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=0) + b.var(ddof=0)) / 2.0)
    if not np.isfinite(pooled) or pooled == 0:
        return 0.0 if np.isclose(a.mean(), b.mean()) else float("nan")
    return float((a.mean() - b.mean()) / pooled)


def balance_report(
    events: pd.DataFrame,
    controls: pd.DataFrame,
    control_type: str,
    variables: tuple[str, ...],
    event_cols: dict[str, str],
    control_cols: dict[str, str],
    stratum: str = "ALL",
) -> pd.DataFrame:
    rows: list[dict] = []
    for var in variables:
        ec, cc = event_cols.get(var), control_cols.get(var)
        if ec not in events.columns or cc not in controls.columns:
            continue
        e, c = events[ec], controls[cc]
        smd = standardised_difference(pd.to_numeric(e, errors="coerce"),
                                      pd.to_numeric(c, errors="coerce"))
        es = e.value_counts(normalize=True)
        cs = c.value_counts(normalize=True)
        cats = es.index.union(cs.index)
        max_gap = float((es.reindex(cats).fillna(0) - cs.reindex(cats).fillna(0)).abs().max())
        # Categorical variables (e.g. session) have no meaningful standardised
        # mean difference; judge them on the largest category share gap instead,
        # otherwise a perfectly balanced categorical reads as "unbalanced".
        is_categorical = not np.isfinite(smd)
        balanced = (max_gap < 0.05) if is_categorical else (abs(smd) < 0.1)
        rows.append({
            "stratum": stratum,
            "control_type": control_type,
            "variable": var,
            "event_n": int(e.notna().sum()),
            "control_n": int(c.notna().sum()),
            "standardised_difference": (round(smd, 4) if np.isfinite(smd) else np.nan),
            "max_category_share_gap": round(max_gap, 4),
            "balance_criterion": ("max_category_share_gap<0.05" if is_categorical
                                  else "abs_smd<0.1"),
            "balanced": bool(balanced),
        })
    return pd.DataFrame(rows)

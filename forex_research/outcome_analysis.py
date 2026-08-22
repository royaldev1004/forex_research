from __future__ import annotations

import numpy as np
import pandas as pd

from .event_definitions import Step3AConfig
from .logging_utils import get_logger

log = get_logger("outcome_analysis")


def parse_expansion_column(col: str) -> tuple[str, float, int] | None:
    if not col.startswith("expansion_"):
        return None
    parts = col.split("_")
    if len(parts) != 4 or parts[1] not in ("up", "down"):
        return None
    try:
        return parts[1], float(parts[2].rstrip("p")), int(parts[3].rstrip("m"))
    except ValueError:
        return None


def event_implies_outcome(
    ev_dir: str, ev_thr: float, ev_hor: int, outcome_col: str
) -> bool:
    parsed = parse_expansion_column(outcome_col)
    if parsed is None:
        return False
    d, t, hor = parsed
    return d == ev_dir and t <= ev_thr and hor >= ev_hor


def control_excludes_outcome(
    excl_thr: float, excl_hor: int, outcome_col: str
) -> bool:
    parsed = parse_expansion_column(outcome_col)
    if parsed is None:
        return False
    _d, t, hor = parsed
    return t >= excl_thr and hor <= excl_hor


def outcome_columns(horizon: int, thresholds: tuple[float, ...]) -> list[str]:
    cols = [
        f"fwd_pips_{horizon}m", f"fwd_log_return_{horizon}m",
        f"max_future_delta_pips_{horizon}m", f"min_future_delta_pips_{horizon}m",
        f"long_mfe_pips_{horizon}m", f"long_mae_pips_{horizon}m",
        f"short_mfe_pips_{horizon}m", f"short_mae_pips_{horizon}m",
        f"max_abs_move_pips_{horizon}m", f"future_range_pips_{horizon}m",
        f"time_to_max_high_minutes_{horizon}m", f"time_to_min_low_minutes_{horizon}m",
        f"horizon_complete_{horizon}m", f"bars_in_window_{horizon}m",
    ]
    for t in thresholds:
        cols += [
            f"expansion_up_{t:g}p_{horizon}m", f"expansion_down_{t:g}p_{horizon}m",
            f"expansion_first_{t:g}p_{horizon}m",
            f"time_to_expansion_up_{t:g}p_{horizon}m",
            f"time_to_expansion_down_{t:g}p_{horizon}m",
        ]
    return cols


def attach_outcomes(
    population: pd.DataFrame, time_column: str, outcomes: pd.DataFrame, cols: list[str]
) -> pd.DataFrame:
    keep = ["decision_time"] + [c for c in cols if c in outcomes.columns]
    sub = outcomes[keep].copy()
    out = population.merge(
        sub, left_on=time_column, right_on="decision_time", how="left",
        suffixes=("", "_outcome"))
    return out


def apply_exclusions(
    df: pd.DataFrame, horizon: int, population: str, stratum: str
) -> tuple[pd.DataFrame, list[dict]]:
    reasons: list[dict] = []
    n0 = len(df)
    ok = pd.Series(True, index=df.index)

    def drop(mask: pd.Series, reason: str) -> None:
        nonlocal ok
        n = int((ok & mask).sum())
        if n:
            reasons.append({"population": population, "stratum": stratum,
                            "horizon_minutes": horizon, "reason": reason, "n_excluded": n})
        ok &= ~mask

    hc = f"horizon_complete_{horizon}m"
    bw = f"bars_in_window_{horizon}m"
    mv = f"max_future_delta_pips_{horizon}m"

    if hc in df.columns:
        drop(df[hc] != 1, "incomplete_forward_horizon")
    if bw in df.columns:
        drop(df[bw].fillna(0) <= 0, "market_closed_entire_window")
    if mv in df.columns:
        drop(df[mv].isna(), "outcome_nan")

    kept = df.loc[ok]
    reasons.append({"population": population, "stratum": stratum,
                    "horizon_minutes": horizon, "reason": "retained", "n_excluded": len(kept)})
    if n0 != len(kept):
        log.info("%s / %s @%dm: %d -> %d rows after outcome exclusions",
                 population, stratum, horizon, n0, len(kept))
    return kept, reasons


def describe_population(
    df: pd.DataFrame, horizon: int, cfg3: Step3AConfig, population: str, stratum: str
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    metrics = {
        "fwd_pips": f"fwd_pips_{horizon}m",
        "max_abs_move_pips": f"max_abs_move_pips_{horizon}m",
        "long_mfe_pips": f"long_mfe_pips_{horizon}m",
        "long_mae_pips": f"long_mae_pips_{horizon}m",
        "short_mfe_pips": f"short_mfe_pips_{horizon}m",
        "short_mae_pips": f"short_mae_pips_{horizon}m",
        "future_range_pips": f"future_range_pips_{horizon}m",
    }
    for label, col in metrics.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        rec = {
            "population": population, "stratum": stratum, "horizon_minutes": horizon,
            "metric": label, "n": int(len(s)),
            "mean": round(float(s.mean()), 3),
            "median": round(float(s.median()), 3),
            "mean_abs": round(float(s.abs().mean()), 3),
            "std": round(float(s.std(ddof=0)), 3),
        }
        for q in cfg3.distribution_quantiles:
            rec[f"q{int(q * 100):02d}"] = round(float(s.quantile(q)), 3)
        rows.append(rec)

    for t in cfg3.report_thresholds_pips:
        for d in ("up", "down"):
            col = f"expansion_{d}_{t:g}p_{horizon}m"
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if s.empty:
                continue
            tt = df[f"time_to_expansion_{d}_{t:g}p_{horizon}m"].dropna()
            rows.append({
                "population": population, "stratum": stratum, "horizon_minutes": horizon,
                "metric": f"expansion_{d}_{t:g}p_rate", "n": int(len(s)),
                "mean": round(float(s.mean()), 5),
                "median": round(float(s.median()), 5),
                "mean_abs": round(float(s.mean()), 5), "std": round(float(s.std(ddof=0)), 5),
                "median_time_to_threshold_minutes": (
                    round(float(tt.median()), 1) if len(tt) else np.nan),
            })
    return pd.DataFrame(rows)


def rate(df: pd.DataFrame, col: str) -> tuple[int, float]:
    if col not in df.columns:
        return 0, float("nan")
    s = df[col].dropna()
    return int(len(s)), (float(s.mean()) if len(s) else float("nan"))


def compare_rates(
    events: pd.DataFrame,
    controls: pd.DataFrame,
    outcome_col: str,
    comparison: str,
    stratum: str,
    horizon: int,
    event_definition: tuple[str, float, int] | None,
    event_label: str,
    control_label: str,
    control_exclusion: tuple[float, int] | None = None,
    min_n: int = 30,
) -> dict:
    n_e, r_e = rate(events, outcome_col)
    n_c, r_c = rate(controls, outcome_col)
    diff = r_e - r_c if np.isfinite(r_e) and np.isfinite(r_c) else np.nan
    lift = (r_e / r_c - 1.0) if (np.isfinite(r_e) and np.isfinite(r_c) and r_c > 0) else np.nan

    ev_implied = bool(
        event_definition is not None
        and event_implies_outcome(*event_definition, outcome_col)
    )
    ct_implied = bool(
        control_exclusion is not None
        and control_excludes_outcome(*control_exclusion, outcome_col)
    )

    return {
        "comparison": comparison,
        "stratum": stratum,
        "horizon_minutes": horizon,
        "outcome": outcome_col,
        "event_population": event_label,
        "control_population": control_label,
        "event_n": n_e,
        "control_n": n_c,
        "event_rate": round(r_e, 5) if np.isfinite(r_e) else np.nan,
        "control_rate": round(r_c, 5) if np.isfinite(r_c) else np.nan,
        "absolute_difference_pp": round(100 * diff, 3) if np.isfinite(diff) else np.nan,
        "relative_lift_pct": round(100 * lift, 2) if np.isfinite(lift) else np.nan,
        "event_side_implied": ev_implied,
        "control_side_implied": ct_implied,
        "is_tautological": bool(ev_implied or ct_implied),
        "min_population_n": min(n_e, n_c),
        "interpretable": bool(min(n_e, n_c) >= min_n and not ev_implied and not ct_implied),
    }


def compare_magnitudes(
    events: pd.DataFrame,
    controls: pd.DataFrame,
    metric_col: str,
    comparison: str,
    stratum: str,
    horizon: int,
    event_label: str,
    control_label: str,
) -> dict:
    e = events[metric_col].dropna() if metric_col in events.columns else pd.Series(dtype=float)
    c = controls[metric_col].dropna() if metric_col in controls.columns else pd.Series(dtype=float)
    return {
        "comparison": comparison, "stratum": stratum, "horizon_minutes": horizon,
        "outcome": metric_col,
        "event_population": event_label, "control_population": control_label,
        "event_n": int(len(e)), "control_n": int(len(c)),
        "event_mean": round(float(e.mean()), 3) if len(e) else np.nan,
        "control_mean": round(float(c.mean()), 3) if len(c) else np.nan,
        "event_median": round(float(e.median()), 3) if len(e) else np.nan,
        "control_median": round(float(c.median()), 3) if len(c) else np.nan,
        "mean_difference": (round(float(e.mean() - c.mean()), 3)
                            if len(e) and len(c) else np.nan),
        "median_difference": (round(float(e.median() - c.median()), 3)
                              if len(e) and len(c) else np.nan),
        "min_population_n": min(len(e), len(c)),
        "interpretable": bool(min(len(e), len(c)) >= 30),
    }


def feature_family_summary(
    event_features: pd.DataFrame,
    control_features: pd.DataFrame,
    dictionary: pd.DataFrame,
    families: tuple[str, ...],
) -> pd.DataFrame:
    if event_features.empty or control_features.empty or dictionary.empty:
        return pd.DataFrame()

    fam_of = dict(zip(dictionary["feature_name"], dictionary["feature_family"]))
    shared = [c for c in event_features.columns
              if c in control_features.columns and c in fam_of
              and fam_of[c] in families]
    if not shared:
        return pd.DataFrame()

    e = event_features[shared].astype("float64")
    c = control_features[shared].astype("float64")

    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.sqrt((e.var(ddof=0) + c.var(ddof=0)) / 2.0)
        smd = (e.mean() - c.mean()) / pooled.replace(0, np.nan)

    df = pd.DataFrame({"feature_name": shared, "smd": smd.to_numpy()})
    df["feature_family"] = df["feature_name"].map(fam_of)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["smd"])
    if df.empty:
        return pd.DataFrame()

    out = df.groupby("feature_family").agg(
        n_features=("smd", "size"),
        mean_abs_smd=("smd", lambda s: round(float(s.abs().mean()), 4)),
        median_abs_smd=("smd", lambda s: round(float(s.abs().median()), 4)),
        max_abs_smd=("smd", lambda s: round(float(s.abs().max()), 4)),
        pct_features_abs_smd_over_0p2=(
            "smd", lambda s: round(100 * float((s.abs() > 0.2).mean()), 2)),
    ).reset_index().sort_values("mean_abs_smd", ascending=False).reset_index(drop=True)
    out["status"] = "EXPLORATORY - not a finding, not corrected for multiple testing"
    return out

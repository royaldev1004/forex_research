from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .logging_utils import get_logger

log = get_logger("benchmark")

OUTCOME_PREFIXES = ("expansion_", "fwd_", "long_mfe", "long_mae", "short_mfe",
                    "short_mae", "max_future_delta", "min_future_delta",
                    "time_to_", "horizon_complete", "bars_in_window",
                    "max_abs_move", "future_range")


class StateLeakError(RuntimeError):
    pass


@dataclass(frozen=True)
class BenchmarkState:

    name: str
    direction: str                       # "up" | "down"
    description: str
    source_columns: tuple[str, ...]
    predicate: Callable[[pd.DataFrame], pd.Series]

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        assert_state_is_point_in_time(self.name, self.source_columns)
        mask = self.predicate(df)
        return mask.fillna(False).astype(bool)


def assert_state_is_point_in_time(name: str, columns: tuple[str, ...]) -> None:
    bad = [c for c in columns if c.startswith(OUTCOME_PREFIXES)]
    if bad:
        raise StateLeakError(
            f"Benchmark state {name!r} would use future information: {bad}. "
            "States must be built only from Day 1 point-in-time features."
        )


def build_states(cfg2: dict) -> list[BenchmarkState]:
    comp = cfg2["compression_column"]
    xdir = cfg2["crossover_direction_column"]
    htf = cfg2["htf_alignment_column"]
    thr = float(cfg2["htf_alignment_min_abs_ordering"])

    states: list[BenchmarkState] = []
    for direction in ("up", "down"):
        bull = direction == "up"
        sgn = 1 if bull else -1
        word = "bullish" if bull else "bearish"

        def cross(df, _x=xdir, _s=sgn):
            return (df[_x] * _s) > 0

        def compressed(df, _c=comp):
            return df[_c] == 1

        def aligned(df, _h=htf, _s=sgn, _t=thr):
            return (df[_h] * _s) >= _t

        states += [
            BenchmarkState(
                "S1_crossover_activity", direction,
                f"Net {word} EMA crossover activity on the decision bar "
                f"({xdir} {'>' if bull else '<'} 0)",
                (xdir,), cross),
            BenchmarkState(
                "S2_compression", direction,
                f"Trailing compression state ({comp} == 1). Direction-independent, "
                "reported under both directions for comparability.",
                (comp,), compressed),
            BenchmarkState(
                "S3_compression_and_crossover", direction,
                f"Compressed AND net {word} crossover activity",
                (comp, xdir), lambda df, c=compressed, x=cross: c(df) & x(df)),
            BenchmarkState(
                "S4_compression_and_htf_alignment", direction,
                f"Compressed AND {cfg2['htf_alignment_timeframe']} EMA stack "
                f"ordered {word} (|{htf}| >= {thr})",
                (comp, htf), lambda df, c=compressed, a=aligned: c(df) & a(df)),
            BenchmarkState(
                "S5_compression_crossover_htf", direction,
                f"Compressed AND net {word} crossover AND "
                f"{cfg2['htf_alignment_timeframe']} stack ordered {word}",
                (comp, xdir, htf),
                lambda df, c=compressed, x=cross, a=aligned: c(df) & x(df) & a(df)),
        ]
    return states


def cell_key(df: pd.DataFrame) -> pd.Series:
    return (df["session"].astype(str)
            + "|h" + df["hour_bucket"].astype(str)
            + "|v" + df["volatility_bin"].astype(str))


def standardised_baseline(
    frame: pd.DataFrame, state_mask: pd.Series, outcome_col: str
) -> tuple[float, float, int]:
    cells = cell_key(frame)
    y = frame[outcome_col]
    ok = y.notna()

    state = state_mask & ok
    other = (~state_mask) & ok
    if not state.any() or not other.any():
        return float("nan"), float("nan"), 0

    w = cells[state].value_counts(normalize=True)
    r = y[other].groupby(cells[other]).mean()

    common = w.index.intersection(r.index)
    if len(common) == 0:
        return float("nan"), 0.0, 0

    covered = float(w.loc[common].sum())
    weights = w.loc[common] / covered
    return float((weights * r.loc[common]).sum()), covered, int(len(common))


def evaluate_state(
    frame: pd.DataFrame,
    state: BenchmarkState,
    threshold: float,
    horizon: int,
    label: str = "primary",
) -> dict:
    outcome_col = f"expansion_{state.direction}_{threshold:g}p_{horizon}m"
    mask = state.evaluate(frame)

    y = frame[outcome_col]
    sub = y[mask & y.notna()]
    n = int(len(sub))
    rate = float(sub.mean()) if n else float("nan")

    base, covered, n_cells = standardised_baseline(frame, mask, outcome_col)
    diff = rate - base if np.isfinite(rate) and np.isfinite(base) else np.nan
    lift = (rate / base - 1.0) if (np.isfinite(rate) and np.isfinite(base)
                                   and base > 0) else np.nan

    row = {
        "endpoint": label,
        "state_name": state.name,
        "direction": state.direction,
        "threshold_pips": threshold,
        "horizon_minutes": horizon,
        "n_observations": n,
        "primary_outcome_rate": round(rate, 5) if np.isfinite(rate) else np.nan,
        "matched_baseline_rate": round(base, 5) if np.isfinite(base) else np.nan,
        "absolute_difference_pp": round(100 * diff, 3) if np.isfinite(diff) else np.nan,
        "relative_lift_pct": round(100 * lift, 2) if np.isfinite(lift) else np.nan,
        "baseline_weight_covered": round(covered, 4) if np.isfinite(covered) else np.nan,
        "baseline_cells_used": n_cells,
        "state_share_of_eligible_pct": round(100 * float(mask.mean()), 3),
    }

    # magnitude context, direction-aware
    fwd = f"fwd_pips_{horizon}m"
    mfe = f"long_mfe_pips_{horizon}m" if state.direction == "up" else f"short_mfe_pips_{horizon}m"
    mae = f"long_mae_pips_{horizon}m" if state.direction == "up" else f"short_mae_pips_{horizon}m"
    for out_name, col in (("median_forward_delta_pips", fwd),
                          ("median_directional_mfe_pips", mfe),
                          ("median_adverse_excursion_pips", mae)):
        if col in frame.columns:
            s = frame.loc[mask, col].dropna()
            row[out_name] = round(float(s.median()), 3) if len(s) else np.nan
        else:
            row[out_name] = np.nan
    return row


def unconditional_row(
    frame: pd.DataFrame, direction: str, threshold: float, horizon: int, label: str
) -> dict:
    outcome_col = f"expansion_{direction}_{threshold:g}p_{horizon}m"
    y = frame[outcome_col].dropna()
    horizon_cols = {
        "median_forward_delta_pips": f"fwd_pips_{horizon}m",
        "median_directional_mfe_pips": (f"long_mfe_pips_{horizon}m" if direction == "up"
                                        else f"short_mfe_pips_{horizon}m"),
        "median_adverse_excursion_pips": (f"long_mae_pips_{horizon}m" if direction == "up"
                                          else f"short_mae_pips_{horizon}m"),
    }
    row = {
        "endpoint": label,
        "state_name": "B0_eligible_market",
        "direction": direction,
        "threshold_pips": threshold,
        "horizon_minutes": horizon,
        "n_observations": int(len(y)),
        "primary_outcome_rate": round(float(y.mean()), 5) if len(y) else np.nan,
        "matched_baseline_rate": np.nan,
        "absolute_difference_pp": np.nan,
        "relative_lift_pct": np.nan,
        "baseline_weight_covered": np.nan,
        "baseline_cells_used": 0,
        "state_share_of_eligible_pct": 100.0,
    }
    for k, col in horizon_cols.items():
        s = frame[col].dropna() if col in frame.columns else pd.Series(dtype=float)
        row[k] = round(float(s.median()), 3) if len(s) else np.nan
    return row


def control_row(
    frame: pd.DataFrame,
    control_times: np.ndarray,
    direction: str,
    threshold: float,
    horizon: int,
    label: str,
    state_name: str,
) -> dict:
    outcome_col = f"expansion_{direction}_{threshold:g}p_{horizon}m"
    sel = frame[pd.DatetimeIndex(frame["decision_time"]).isin(
        pd.DatetimeIndex(control_times))]
    y = sel[outcome_col].dropna()
    row = {
        "endpoint": label,
        "state_name": state_name,
        "direction": direction,
        "threshold_pips": threshold,
        "horizon_minutes": horizon,
        "n_observations": int(len(y)),
        "primary_outcome_rate": round(float(y.mean()), 5) if len(y) else np.nan,
        "matched_baseline_rate": np.nan,
        "absolute_difference_pp": np.nan,
        "relative_lift_pct": np.nan,
        "baseline_weight_covered": np.nan,
        "baseline_cells_used": 0,
        "state_share_of_eligible_pct": round(100 * len(sel) / max(1, len(frame)), 3),
    }
    for k, col in (("median_forward_delta_pips", f"fwd_pips_{horizon}m"),
                   ("median_directional_mfe_pips",
                    f"long_mfe_pips_{horizon}m" if direction == "up"
                    else f"short_mfe_pips_{horizon}m"),
                   ("median_adverse_excursion_pips",
                    f"long_mae_pips_{horizon}m" if direction == "up"
                    else f"short_mae_pips_{horizon}m")):
        s = sel[col].dropna() if col in sel.columns else pd.Series(dtype=float)
        row[k] = round(float(s.median()), 3) if len(s) else np.nan
    return row


def state_counts(frame: pd.DataFrame, states: list[BenchmarkState]) -> pd.DataFrame:
    rows = []
    for st in states:
        m = st.evaluate(frame)
        rows.append({
            "state_name": st.name,
            "direction": st.direction,
            "n_observations": int(m.sum()),
            "share_of_eligible_pct": round(100 * float(m.mean()), 3),
            "description": st.description,
            "source_columns": ",".join(st.source_columns),
        })
    return pd.DataFrame(rows)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ResearchConfig
from .logging_utils import get_logger

log = get_logger("crossover_analysis")

BULLISH = "bullish"
BEARISH = "bearish"


@dataclass(frozen=True)
class CrossoverResult:

    detail: pd.DataFrame
    counts: pd.DataFrame
    pair_counts: pd.DataFrame
    by_session: pd.DataFrame


def _pair_crossings(
    values: np.ndarray,
    labels_a: list[tuple[str, int]],
    labels_b: list[tuple[str, int]],
    left: np.ndarray,
    right: np.ndarray,
    bar_close: np.ndarray,
) -> pd.DataFrame:
    diff = (values[:, left] - values[:, right]).astype("float32")
    finite = np.isfinite(diff)
    sgn = np.where(finite, np.sign(diff), 0).astype("int8")
    prev_sgn = np.vstack([np.zeros((1, sgn.shape[1]), "int8"), sgn[:-1]])
    valid = finite & np.vstack([np.zeros((1, finite.shape[1]), bool), finite[:-1]])

    turned_up = valid & (sgn > 0) & (prev_sgn <= 0)
    turned_dn = valid & (sgn < 0) & (prev_sgn >= 0)

    rows: list[pd.DataFrame] = []
    for mat, direction in ((turned_up, BULLISH), (turned_dn, BEARISH)):
        bar_idx, pair_idx = np.nonzero(mat)
        if not len(bar_idx):
            continue
        rows.append(pd.DataFrame({
            "bar_close_time": bar_close[bar_idx],
            "indicator_a_type": [labels_a[p][0] for p in pair_idx],
            "indicator_a_period": [labels_a[p][1] for p in pair_idx],
            "indicator_b_type": [labels_b[p][0] for p in pair_idx],
            "indicator_b_period": [labels_b[p][1] for p in pair_idx],
            "direction": direction,
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def crossovers_for_timeframe(
    tf: str,
    indicator_values: pd.DataFrame,
    bar_close_time: pd.Series,
    cfg: ResearchConfig,
    families: tuple[str, ...],
) -> pd.DataFrame:
    bar_close = pd.DatetimeIndex(bar_close_time).to_numpy()
    out: list[pd.DataFrame] = []

    for family in families:
        if family in ("ema", "sma"):
            periods = list(cfg.ema_periods if family == "ema" else cfg.sma_periods)
            if len(periods) < 2:
                continue
            cols = [f"{family}_{p}" for p in periods]
            vals = indicator_values[cols].to_numpy(dtype="float64")
            ia, ib = np.triu_indices(len(periods), 1)
            labels = [(family, p) for p in periods]
            det = _pair_crossings(vals, [labels[i] for i in ia], [labels[j] for j in ib],
                                  ia, ib, bar_close)
        elif family == "ema_vs_sma":
            shared = [p for p in cfg.ema_periods if p in set(cfg.sma_periods)]
            if not shared:
                continue
            cols = [f"ema_{p}" for p in shared] + [f"sma_{p}" for p in shared]
            vals = indicator_values[cols].to_numpy(dtype="float64")
            k = len(shared)
            ia = np.arange(k)
            ib = np.arange(k, 2 * k)
            det = _pair_crossings(vals, [("ema", p) for p in shared],
                                  [("sma", p) for p in shared], ia, ib, bar_close)
        else:
            raise ValueError(f"Unknown crossover family: {family!r}")

        if not det.empty:
            det["timeframe"] = tf
            det["indicator_family"] = family
            out.append(det)

    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True)
    return res.sort_values(["bar_close_time", "indicator_family",
                            "indicator_a_period"]).reset_index(drop=True)


def group_episodes(
    timestamps: pd.Series, gap_bars: int, tf_minutes: int
) -> np.ndarray:
    if len(timestamps) == 0:
        return np.array([], dtype="int64")
    t = pd.DatetimeIndex(timestamps).to_numpy()
    gap = np.timedelta64(gap_bars * tf_minutes, "m")
    new = np.empty(len(t), dtype=bool)
    new[0] = True
    new[1:] = (t[1:] - t[:-1]) >= gap
    return np.cumsum(new) - 1


def summarise(
    detail: pd.DataFrame,
    tf_minutes: dict[str, int],
    gap_bars: int,
    eligible_times: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if detail.empty:
        return pd.DataFrame(), pd.DataFrame()

    det = detail
    if eligible_times is not None:
        det = det[det["bar_close_time"].isin(eligible_times)]
        if det.empty:
            return pd.DataFrame(), pd.DataFrame()

    rows: list[dict] = []
    for (tf, family, direction), grp in det.groupby(
        ["timeframe", "indicator_family", "direction"], sort=False
    ):
        uniq = grp["bar_close_time"].drop_duplicates().sort_values()
        ep = group_episodes(uniq, gap_bars, tf_minutes[tf])
        rows.append({
            "timeframe": tf,
            "indicator_family": family,
            "direction": direction,
            "n_pair_cross_events": int(len(grp)),
            "n_unique_timestamps_with_any_cross": int(len(uniq)),
            "n_unique_cross_episodes": int(ep.max() + 1) if len(ep) else 0,
            "mean_pairs_per_crossing_timestamp": round(len(grp) / max(1, len(uniq)), 2),
            "mean_timestamps_per_episode": round(
                len(uniq) / max(1, (ep.max() + 1) if len(ep) else 1), 2),
        })
    counts = pd.DataFrame(rows).sort_values(
        ["timeframe", "indicator_family", "direction"]).reset_index(drop=True)

    pair_counts = (det.groupby(
        ["timeframe", "indicator_family", "indicator_a_type", "indicator_a_period",
         "indicator_b_type", "indicator_b_period", "direction"], sort=False)
        .size().reset_index(name="n_cross_events")
        .sort_values(["timeframe", "indicator_family", "n_cross_events"],
                     ascending=[True, True, False]).reset_index(drop=True))

    return counts, pair_counts


def summarise_by_session(
    detail: pd.DataFrame, session_of_time: pd.Series
) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    sess = pd.Series(session_of_time.to_numpy(),
                     index=pd.DatetimeIndex(session_of_time.index))
    det = detail.copy()
    det["session"] = sess.reindex(pd.DatetimeIndex(det["bar_close_time"])).to_numpy()
    det = det[det["session"].notna()]
    if det.empty:
        return pd.DataFrame()
    uniq = det.drop_duplicates(["timeframe", "indicator_family", "direction",
                                "bar_close_time"])
    return (uniq.groupby(["timeframe", "indicator_family", "direction", "session"])
            .size().reset_index(name="n_unique_timestamps_with_any_cross")
            .sort_values(["timeframe", "indicator_family", "direction", "session"])
            .reset_index(drop=True))

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ResearchConfig
from .feature_dictionary import FeatureSpec
from .logging_utils import get_logger

log = get_logger("outcomes")


class RangeIndex:

    def __init__(self, values: np.ndarray, mode: str):
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.mode = mode
        self.values = np.asarray(values, dtype="float64")
        n = len(self.values)
        self.n = n
        levels = max(1, int(np.floor(np.log2(n))) + 1) if n else 1
        self.log_table = np.zeros(n + 1, dtype="int64")
        for i in range(2, n + 1):
            self.log_table[i] = self.log_table[i // 2] + 1
        self.arg = np.empty((levels, n), dtype="int64")
        if n:
            self.arg[0] = np.arange(n)
            better = np.greater if mode == "max" else np.less
            for k in range(1, levels):
                span = 1 << (k - 1)
                width = 1 << k
                m = n - width + 1
                if m <= 0:
                    self.arg[k, :] = self.arg[k - 1, :]
                    continue
                left = self.arg[k - 1, :m]
                right = self.arg[k - 1, span:span + m]
                pick = np.where(better(self.values[left], self.values[right]), left, right)
                self.arg[k, :m] = pick
                self.arg[k, m:] = self.arg[k - 1, m:]

    def query_arg(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        lo = np.asarray(lo, dtype="int64")
        hi = np.asarray(hi, dtype="int64")
        out = np.full(len(lo), -1, dtype="int64")
        ok = (lo <= hi) & (lo >= 0) & (hi < self.n)
        if not ok.any():
            return out
        l, h = lo[ok], hi[ok]
        k = self.log_table[h - l + 1]
        a = self.arg[k, l]
        b = self.arg[k, h - (1 << k) + 1]
        better = np.greater if self.mode == "max" else np.less
        out[ok] = np.where(better(self.values[a], self.values[b]), a, b)
        return out

    def query_value(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        arg = self.query_arg(lo, hi)
        out = np.full(len(arg), np.nan)
        ok = arg >= 0
        out[ok] = self.values[arg[ok]]
        return out


def _first_touch(
    table: RangeIndex, lo: np.ndarray, hi: np.ndarray, target: np.ndarray, direction: str
) -> np.ndarray:
    n = len(lo)
    result = np.full(n, -1, dtype="int64")
    valid = (lo <= hi) & (lo >= 0)
    if not valid.any():
        return result

    extreme = table.query_value(lo, hi)
    if direction == "up":
        reached = valid & np.isfinite(extreme) & (extreme >= target)
    else:
        reached = valid & np.isfinite(extreme) & (extreme <= target)
    if not reached.any():
        return result

    left = lo.copy()
    right = hi.copy()
    idx = np.where(reached)[0]
    l, r = left[idx], right[idx]
    tgt = target[idx]

    # invariant: the answer lies in [l, r]
    for _ in range(int(np.ceil(np.log2(max(2, table.n)))) + 2):
        active = l < r
        if not active.any():
            break
        mid = np.where(active, (l + r) // 2, l)
        ext = table.query_value(np.where(active, lo[idx], l), mid)
        if direction == "up":
            hit = active & np.isfinite(ext) & (ext >= tgt)
        else:
            hit = active & np.isfinite(ext) & (ext <= tgt)
        r = np.where(hit, mid, r)
        l = np.where(active & ~hit, mid + 1, l)

    result[idx] = l
    return result


def build_outcomes(
    base: pd.DataFrame, cfg: ResearchConfig
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    pip = cfg.pip_size
    n = len(base)
    close_t = pd.DatetimeIndex(base["bar_close_time"]).to_numpy()
    high = base["high"].to_numpy(dtype="float64")
    low = base["low"].to_numpy(dtype="float64")
    close = base["close"].to_numpy(dtype="float64")

    if cfg.outcome_entry_reference != "decision_bar_close":
        raise ValueError(
            f"Unsupported outcome_entry_reference: {cfg.outcome_entry_reference!r}"
        )
    entry = close  # reference price observable at the decision time itself

    hi_tbl = RangeIndex(high, "max")
    lo_tbl = RangeIndex(low, "min")

    out: dict[str, np.ndarray] = {}
    specs: list[FeatureSpec] = []

    def add(name: str, values: np.ndarray, family: str, description: str,
            units: str, lookback: int | None = None) -> None:
        out[name] = np.asarray(values, dtype="float64")
        specs.append(FeatureSpec(
            feature_name=name, feature_family=family, description=description, units=units,
            timeframe=cfg.base_timeframe, lookback=lookback, uses_future_data=True,
            source_columns="future high,low,close of the base timeframe"))

    last_close_t = close_t[-1] if n else None
    start = np.arange(n) + 1  # window opens at the *next* bar

    for h in cfg.forward_horizons_minutes:
        end_t = close_t + np.timedelta64(h, "m")
        stop = np.searchsorted(close_t, end_t, side="right") - 1
        complete = end_t <= last_close_t
        empty = start > stop

        n_bars = np.where(empty, 0, stop - start + 1).astype("float64")

        # --- forward return ------------------------------------------------
        last_px = np.full(n, np.nan)
        ok = ~empty
        last_px[ok] = close[stop[ok]]
        delta = last_px - entry

        add(f"fwd_price_change_{h}m", delta, "forward_return",
            f"Close {h} minutes ahead minus the decision-bar close", "price", h)
        add(f"fwd_pips_{h}m", delta / pip, "forward_return",
            f"Price change over the next {h} minutes", "pips", h)
        with np.errstate(divide="ignore", invalid="ignore"):
            add(f"fwd_log_return_{h}m", np.log(last_px / entry), "forward_return",
                f"Log return over the next {h} minutes", "log_return", h)

        # --- excursions ------------------------------------------------------
        arg_hi = hi_tbl.query_arg(start, stop)
        arg_lo = lo_tbl.query_arg(start, stop)
        max_hi = np.where(arg_hi >= 0, high[np.maximum(arg_hi, 0)], np.nan)
        min_lo = np.where(arg_lo >= 0, low[np.maximum(arg_lo, 0)], np.nan)

        # --- signed extrema: the raw, information-preserving measurement -----
        # These are deltas from the decision-bar close and may legitimately be
        # negative (price never traded back to the reference during the window).
        max_delta = (max_hi - entry) / pip
        min_delta = (min_lo - entry) / pip
        raw_note = (
            "Signed delta from the decision-bar close, NOT floored at zero. A positive "
            "min_future_delta (or negative max_future_delta) means price never traded back "
            "to the reference price during the window. This is the raw measurement; the "
            "conventional floored excursions are derived from it."
        )
        add(f"max_future_delta_pips_{h}m", max_delta, "future_extreme",
            (f"Highest high in the next {h} minutes minus the decision-bar close. "
             f"{raw_note}"), "pips", h)
        add(f"min_future_delta_pips_{h}m", min_delta, "future_extreme",
            (f"Lowest low in the next {h} minutes minus the decision-bar close. "
             f"{raw_note}"), "pips", h)

        # --- conventional excursions: floored at zero, never negative --------
        # Definitionally >= 0, matching the usual meaning of MFE/MAE. These are
        # still market-movement measurements: no entry rule, direction, spread,
        # slippage or commission is assumed, so they are NOT trade P&L.
        conv_note = (
            "Conventional excursion floored at zero, so it is never negative. Derived from "
            "the signed future extrema; no entry rule, direction, spread or commission is "
            "assumed, so this is a market-movement measurement and NOT trade profit or loss."
        )
        # np.maximum, not np.fmax: fmax ignores NaN and would turn an unobserved
        # window into a spurious 0.0, i.e. silently report "no movement" where the
        # truth is "no data". NaN must propagate.
        add(f"long_mfe_pips_{h}m", np.maximum(0.0, max_delta), "excursion",
            (f"Maximum favourable excursion for a hypothetical long over {h} minutes: "
             f"max(0, max_future_delta_pips_{h}m). {conv_note}"), "pips", h)
        add(f"long_mae_pips_{h}m", np.maximum(0.0, -min_delta), "excursion",
            (f"Maximum adverse excursion for a hypothetical long over {h} minutes: "
             f"max(0, -min_future_delta_pips_{h}m). {conv_note}"), "pips", h)
        add(f"short_mfe_pips_{h}m", np.maximum(0.0, -min_delta), "excursion",
            (f"Maximum favourable excursion for a hypothetical short over {h} minutes: "
             f"max(0, -min_future_delta_pips_{h}m). {conv_note}"), "pips", h)
        add(f"short_mae_pips_{h}m", np.maximum(0.0, max_delta), "excursion",
            (f"Maximum adverse excursion for a hypothetical short over {h} minutes: "
             f"max(0, max_future_delta_pips_{h}m). {conv_note}"), "pips", h)

        t_up = np.where(arg_hi >= 0,
                        (close_t[np.maximum(arg_hi, 0)] - close_t) / np.timedelta64(1, "m"), np.nan)
        t_dn = np.where(arg_lo >= 0,
                        (close_t[np.maximum(arg_lo, 0)] - close_t) / np.timedelta64(1, "m"), np.nan)
        add(f"time_to_max_high_minutes_{h}m", t_up, "excursion",
            f"Minutes from the decision time to the highest high within {h} minutes", "minutes", h)
        add(f"time_to_min_low_minutes_{h}m", t_dn, "excursion",
            f"Minutes from the decision time to the lowest low within {h} minutes", "minutes", h)

        add(f"max_abs_move_pips_{h}m", np.maximum(np.abs(max_delta), np.abs(min_delta)),
            "excursion",
            (f"Largest absolute distance travelled from the decision-bar close in either "
             f"direction within {h} minutes"), "pips", h)
        add(f"future_range_pips_{h}m", max_delta - min_delta, "future_extreme",
            (f"High-low range of the next {h} minutes in pips "
             f"(max_future_delta minus min_future_delta); always >= 0"), "pips", h)

        add(f"horizon_complete_{h}m", complete.astype("float64"), "outcome_quality",
            (f"1 when the dataset extends a full {h} minutes past the decision time, so the "
             f"window is not truncated by the end of the data. This does NOT imply the "
             f"window contains any bars: when the market was closed for the whole window "
             f"(weekend or holiday) this is 1 while bars_in_window_{h}m is 0 and every "
             f"outcome for that row is NaN. Filter on both columns."),
            "boolean", h)
        add(f"bars_in_window_{h}m", n_bars, "outcome_quality",
            (f"Base bars actually observed inside the {h}-minute window. Below the nominal "
             f"count across weekends, holidays and feed gaps; 0 means the market was shut "
             f"for the entire window and all outcomes for this row are NaN."), "count", h)

        # --- expansion labels ------------------------------------------------
        for thr in cfg.expansion_thresholds_pips:
            up_target = entry + thr * pip
            dn_target = entry - thr * pip
            i_up = _first_touch(hi_tbl, start, stop, up_target, "up")
            i_dn = _first_touch(lo_tbl, start, stop, dn_target, "down")
            tag = f"{thr:g}p_{h}m"

            add(f"expansion_up_{tag}", (i_up >= 0).astype("float64"), "expansion_label",
                (f"1 when price traded at least {thr:g} pips above the decision-bar close within "
                 f"{h} minutes. A market-movement label, not a trade result."), "boolean", h)
            add(f"expansion_down_{tag}", (i_dn >= 0).astype("float64"), "expansion_label",
                f"1 when price traded at least {thr:g} pips below the close within {h} minutes",
                "boolean", h)

            t_u = np.where(i_up >= 0,
                           (close_t[np.maximum(i_up, 0)] - close_t) / np.timedelta64(1, "m"), np.nan)
            t_d = np.where(i_dn >= 0,
                           (close_t[np.maximum(i_dn, 0)] - close_t) / np.timedelta64(1, "m"), np.nan)
            add(f"time_to_expansion_up_{tag}", t_u, "expansion_label",
                f"Minutes until the {thr:g}-pip upward level was first reached", "minutes", h)
            add(f"time_to_expansion_down_{tag}", t_d, "expansion_label",
                f"Minutes until the {thr:g}-pip downward level was first reached", "minutes", h)

            # 1 up first, -1 down first, 0 neither, 2 both inside the same bar
            first = np.zeros(n, dtype="float64")
            hu, hd = i_up >= 0, i_dn >= 0
            first = np.where(hu & ~hd, 1.0, first)
            first = np.where(hd & ~hu, -1.0, first)
            both = hu & hd
            first = np.where(both & (i_up < i_dn), 1.0, first)
            first = np.where(both & (i_dn < i_up), -1.0, first)
            first = np.where(both & (i_up == i_dn), 2.0, first)
            add(f"expansion_first_{tag}", first, "expansion_label",
                (f"Which {thr:g}-pip level was reached first within {h} minutes: 1 up, -1 down, "
                 "0 neither, 2 both inside the same 5-minute bar (unresolvable without tick data)"),
                "category", h)

    frame = pd.DataFrame(out)
    frame.insert(0, "decision_time", close_t)
    frame.insert(0, "symbol", cfg.symbol)

    log.info("Built %d outcome columns for %d decision rows.", len(out), n)
    return frame, specs


def outcome_summary(frame: pd.DataFrame, cfg: ResearchConfig, valid: np.ndarray) -> pd.DataFrame:
    rows = []
    sub = frame.loc[valid]
    for h in cfg.forward_horizons_minutes:
        comp = sub[f"horizon_complete_{h}m"]
        bars = sub[f"bars_in_window_{h}m"]
        lmfe = sub[f"long_mfe_pips_{h}m"]
        lmae = sub[f"long_mae_pips_{h}m"]
        rows.append({
            "kind": "forward_horizon", "horizon_minutes": h, "threshold_pips": "",
            "n_valid_rows": int(len(sub)),
            "n_complete_windows": int((comp == 1).sum()),
            "n_incomplete_windows": int((comp == 0).sum()),
            "n_empty_windows_market_closed": int(((comp == 1) & (bars == 0)).sum()),
            "n_non_null": int(sub[f"fwd_pips_{h}m"].notna().sum()),
            "n_long_mfe_available": int(lmfe.notna().sum()),
            "n_long_mae_available": int(lmae.notna().sum()),
            "n_conventional_excursions_negative": int(((lmfe < 0) | (lmae < 0)).sum()),
            "n_signed_max_delta_negative": int((sub[f"max_future_delta_pips_{h}m"] < 0).sum()),
            "n_signed_min_delta_positive": int((sub[f"min_future_delta_pips_{h}m"] > 0).sum()),
            "mean_abs_move_pips": round(float(sub[f"max_abs_move_pips_{h}m"].mean()), 3),
            "median_abs_move_pips": round(float(sub[f"max_abs_move_pips_{h}m"].median()), 3),
        })
        for thr in cfg.expansion_thresholds_pips:
            tag = f"{thr:g}p_{h}m"
            up = sub[f"expansion_up_{tag}"]
            dn = sub[f"expansion_down_{tag}"]
            first = sub[f"expansion_first_{tag}"]
            rows.append({
                "kind": "expansion_label", "horizon_minutes": h, "threshold_pips": thr,
                "n_valid_rows": int(len(sub)),
                "n_up_hit": int((up == 1).sum()),
                "n_down_hit": int((dn == 1).sum()),
                "n_either_hit": int(((up == 1) | (dn == 1)).sum()),
                "n_both_hit": int(((up == 1) & (dn == 1)).sum()),
                "n_same_bar_ambiguous": int((first == 2).sum()),
                "pct_up_hit": round(float((up == 1).mean()) * 100, 3),
                "pct_down_hit": round(float((dn == 1).mean()) * 100, 3),
            })
    return pd.DataFrame(rows)

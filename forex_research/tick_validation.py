from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .tick_stream import MS_PER_5MIN, MS_PER_DAY, M5Reconstruction

log = get_logger("tick_validation")

#: Conclusion vocabulary. Ordered weakest to strongest evidential claim.
VERDICTS = ("CONTRADICTED", "INCONCLUSIVE", "STRONGLY SUPPORTED", "CONFIRMED")


def ms_to_timestamp(ms: int) -> pd.Timestamp:
    return pd.Timestamp(int(ms), unit="ms")


def load_m5_reference(path: Path, point_size: float) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip().strip("<>").upper() for c in df.columns]
    open_time = pd.to_datetime(df["DATE"] + " " + df["TIME"],
                               format="%Y.%m.%d %H:%M:%S", errors="coerce")
    if open_time.isna().any():
        raise ValueError(f"{path.name}: {int(open_time.isna().sum())} timestamps unparseable.")

    scale = 1.0 / point_size
    out = pd.DataFrame({
        "bar_open_time": open_time,
        "epoch_ms": (open_time - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1),
    })
    for c in ("OPEN", "HIGH", "LOW", "CLOSE"):
        out[c.lower()] = np.rint(
            pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64") * scale
        ).astype("int64")
    out["spread_points"] = pd.to_numeric(df["SPREAD"], errors="coerce").to_numpy()
    return out


@dataclass
class M5ComparisonResult:
    label: str
    description: str
    summary: dict
    per_date: pd.DataFrame


def compare_m5(
    recon: M5Reconstruction,
    m5: pd.DataFrame,
    tolerance_points: int = 1,
) -> M5ComparisonResult:
    bins = (m5["epoch_ms"].to_numpy() // MS_PER_5MIN) - recon.base_bin
    inside = (bins >= 0) & (bins < recon.n_bins)
    have = np.zeros(len(m5), dtype=bool)
    have[inside] = recon.count[bins[inside]] > 0

    n_m5 = int(len(m5))
    n_overlap = int(have.sum())

    # Always emit the full key set, so a zero-overlap run (e.g. a bounded smoke
    # test over a prefix that predates the bar history) produces a well-formed
    # row rather than a partial dict that breaks downstream reporting.
    base: dict = {
        "interpretation": recon.label,
        "description": recon.description,
        "m5_bars_total": n_m5,
        "m5_bars_with_tick_coverage": n_overlap,
        "coverage_pct": round(100.0 * n_overlap / n_m5, 4) if n_m5 else float("nan"),
        "tick_count_median_per_bar": float("nan"),
        "full_ohlc_exact_match_pct": float("nan"),
        f"full_ohlc_within_{tolerance_points}pt_pct": float("nan"),
        "n_bars_mismatched_exact": 0,
        "max_abs_diff_points_any_field": 0,
        "note": "",
    }
    for k in ("open", "high", "low", "close"):
        base[f"{k}_exact_match_pct"] = float("nan")
        base[f"{k}_within_{tolerance_points}pt_pct"] = float("nan")
        base[f"{k}_median_abs_diff_points"] = float("nan")
        base[f"{k}_max_abs_diff_points"] = 0

    if n_overlap == 0:
        base["note"] = "No overlap between the reconstructed bins and the M5 export."
        empty = pd.DataFrame(columns=["date", "n_bars", "open_match_pct", "high_match_pct",
                                      "low_match_pct", "close_match_pct", "ohlc_match_pct",
                                      "median_abs_diff_points", "max_abs_diff_points",
                                      "interpretation"])
        log.warning("M5 reconstruction [%s]: no overlapping bars.", recon.label)
        return M5ComparisonResult(recon.label, recon.description, base, empty)

    idx = bins[have]
    got = {
        "open": recon.open_[idx], "high": recon.high[idx],
        "low": recon.low[idx], "close": recon.close[idx],
    }
    want = {k: m5[k].to_numpy()[have] for k in ("open", "high", "low", "close")}

    diffs = {k: np.abs(got[k] - want[k]) for k in got}
    exact = {k: (diffs[k] == 0) for k in got}
    within = {k: (diffs[k] <= tolerance_points) for k in got}
    all_exact = exact["open"] & exact["high"] & exact["low"] & exact["close"]
    all_within = within["open"] & within["high"] & within["low"] & within["close"]
    max_diff = np.maximum.reduce([diffs[k] for k in ("open", "high", "low", "close")])

    summary = dict(base)
    summary["tick_count_median_per_bar"] = float(np.median(recon.count[idx]))
    for k in ("open", "high", "low", "close"):
        summary[f"{k}_exact_match_pct"] = round(100.0 * float(exact[k].mean()), 4)
        summary[f"{k}_within_{tolerance_points}pt_pct"] = round(
            100.0 * float(within[k].mean()), 4)
        summary[f"{k}_median_abs_diff_points"] = float(np.median(diffs[k]))
        summary[f"{k}_max_abs_diff_points"] = int(diffs[k].max())
    summary["full_ohlc_exact_match_pct"] = round(100.0 * float(all_exact.mean()), 4)
    summary[f"full_ohlc_within_{tolerance_points}pt_pct"] = round(
        100.0 * float(all_within.mean()), 4)
    summary["n_bars_mismatched_exact"] = int((~all_exact).sum())
    summary["max_abs_diff_points_any_field"] = int(max_diff.max())

    dates = pd.DatetimeIndex(m5["bar_open_time"].to_numpy()[have]).normalize()
    per_date = pd.DataFrame({
        "date": dates,
        "open_ok": exact["open"], "high_ok": exact["high"],
        "low_ok": exact["low"], "close_ok": exact["close"],
        "ohlc_ok": all_exact, "ohlc_within_tol": all_within,
        "max_abs_diff_points": max_diff,
        "tick_count": recon.count[idx],
    }).groupby("date", as_index=False).agg(
        n_bars=("ohlc_ok", "size"),
        open_match_pct=("open_ok", lambda s: round(100.0 * s.mean(), 3)),
        high_match_pct=("high_ok", lambda s: round(100.0 * s.mean(), 3)),
        low_match_pct=("low_ok", lambda s: round(100.0 * s.mean(), 3)),
        close_match_pct=("close_ok", lambda s: round(100.0 * s.mean(), 3)),
        ohlc_match_pct=("ohlc_ok", lambda s: round(100.0 * s.mean(), 3)),
        ohlc_within_tol_pct=("ohlc_within_tol", lambda s: round(100.0 * s.mean(), 3)),
        median_abs_diff_points=("max_abs_diff_points", "median"),
        max_abs_diff_points=("max_abs_diff_points", "max"),
        median_ticks_per_bar=("tick_count", "median"),
    )
    per_date["interpretation"] = recon.label

    log.info("M5 reconstruction [%s]: %d overlapping bars, full-OHLC exact %.3f%%",
             recon.label, n_overlap, summary["full_ohlc_exact_match_pct"])
    return M5ComparisonResult(recon.label, recon.description, summary, per_date)


def m5_bars_missing_from_ticks(recon: M5Reconstruction, m5: pd.DataFrame) -> pd.DataFrame:
    bins = (m5["epoch_ms"].to_numpy() // MS_PER_5MIN) - recon.base_bin
    inside = (bins >= 0) & (bins < recon.n_bins)
    covered = np.zeros(len(m5), dtype=bool)
    covered[inside] = recon.count[bins[inside]] > 0
    missing = m5.loc[~covered, ["bar_open_time"]].copy()
    if missing.empty:
        return pd.DataFrame(columns=["date", "n_m5_bars_without_ticks",
                                     "first_bar", "last_bar"])
    missing["date"] = pd.DatetimeIndex(missing["bar_open_time"]).normalize()
    return missing.groupby("date", as_index=False).agg(
        n_m5_bars_without_ticks=("bar_open_time", "size"),
        first_bar=("bar_open_time", "min"),
        last_bar=("bar_open_time", "max"),
    )


# ---------------------------------------------------------------------------
# Timezone assessment
# ---------------------------------------------------------------------------

def weekly_session_table(
    week_first_ms: dict[int, int], week_last_ms: dict[int, int]
) -> pd.DataFrame:
    rows = []
    for wk in sorted(week_first_ms):
        f = ms_to_timestamp(week_first_ms[wk])
        l = ms_to_timestamp(week_last_ms[wk])
        rows.append({
            "week_start_date": (f.normalize() - pd.Timedelta(days=int(f.dayofweek))).date(),
            "first_tick": f, "first_weekday": f.day_name(),
            "first_hhmm": f.strftime("%H:%M:%S"),
            "last_tick": l, "last_weekday": l.day_name(),
            "last_hhmm": l.strftime("%H:%M:%S"),
        })
    return pd.DataFrame(rows)


def us_dst_transition_weeks(sessions: pd.DataFrame) -> pd.DataFrame:
    if sessions.empty:
        return sessions
    import zoneinfo

    tz = zoneinfo.ZoneInfo("America/New_York")
    years = sorted({pd.Timestamp(d).year for d in sessions["week_start_date"]})
    transitions: list[pd.Timestamp] = []
    for y in years:
        hours = pd.date_range(f"{y}-01-01", f"{y}-12-31 23:00", freq="h", tz="UTC")
        offs = np.asarray([t.astimezone(tz).utcoffset().total_seconds() for t in hours])
        change = np.flatnonzero(np.r_[False, offs[1:] != offs[:-1]])
        transitions += [hours[i].tz_localize(None) for i in change]

    out = sessions.copy()
    starts = pd.to_datetime(out["week_start_date"])
    flag = np.zeros(len(out), dtype=bool)
    for t in transitions:
        flag |= (starts <= t) & (t < starts + pd.Timedelta(days=7))
    out = out.loc[flag].copy()
    out["contains_us_dst_transition"] = True
    return out


def rollover_hour_evidence(hour_median_spread: dict[int, float]) -> dict:
    if not hour_median_spread:
        return {}
    hours = sorted(hour_median_spread)
    med = np.array([hour_median_spread[h] for h in hours], dtype=float)
    widest = int(hours[int(np.argmax(med))])
    typical = float(np.median(med))
    return {
        "widest_spread_hour_file_native": widest,
        "widest_hour_median_spread_points": float(med.max()),
        "typical_hour_median_spread_points": typical,
        "widening_ratio": round(float(med.max()) / typical, 2) if typical else float("nan"),
        "implied_new_york_hour": (widest - 7) % 24,
        "consistent_with_1700_ny_rollover": bool((widest - 7) % 24 == 17),
    }


def assess_timezone(
    sessions: pd.DataFrame,
    best_m5: M5ComparisonResult | None,
    has_authoritative_metadata: bool,
    hour_median_spread: dict[int, float] | None = None,
) -> tuple[str, str, dict]:
    stats: dict = {}
    if sessions.empty:
        return "INCONCLUSIVE", "No weekly session boundaries were observed.", stats

    open_wd = sessions["first_weekday"].value_counts()
    close_wd = sessions["last_weekday"].value_counts()
    n = len(sessions)
    monday_opens = int(open_wd.get("Monday", 0))
    friday_closes = int(close_wd.get("Friday", 0))
    sunday_opens = int(open_wd.get("Sunday", 0))

    first_hours = pd.to_datetime(sessions["first_hhmm"], format="%H:%M:%S").dt.hour
    last_hours = pd.to_datetime(sessions["last_hhmm"], format="%H:%M:%S").dt.hour
    stats.update({
        "weeks_observed": n,
        "weeks_opening_monday": monday_opens,
        "weeks_opening_sunday": sunday_opens,
        "weeks_closing_friday": friday_closes,
        "pct_open_monday": round(100.0 * monday_opens / n, 2),
        "pct_close_friday": round(100.0 * friday_closes / n, 2),
        "modal_open_hour": int(first_hours.mode().iloc[0]) if len(first_hours) else -1,
        "modal_close_hour": int(last_hours.mode().iloc[0]) if len(last_hours) else -1,
        "open_hour_distribution": first_hours.value_counts().sort_index().to_dict(),
        "close_hour_distribution": last_hours.value_counts().sort_index().to_dict(),
    })

    roll = rollover_hour_evidence(hour_median_spread or {})
    stats.update(roll)

    dst_weeks = us_dst_transition_weeks(sessions)
    stats["dst_transition_weeks_observed"] = int(len(dst_weeks))
    if not dst_weeks.empty:
        dh = pd.to_datetime(dst_weeks["first_hhmm"], format="%H:%M:%S").dt.hour
        stats["dst_week_open_hours"] = dh.value_counts().sort_index().to_dict()
        stats["dst_weeks_opening_monday"] = int(
            (dst_weeks["first_weekday"] == "Monday").sum())

    session_consistent = (
        stats["pct_open_monday"] >= 95.0
        and stats["pct_close_friday"] >= 95.0
        and stats["modal_open_hour"] == 0
        and stats["modal_close_hour"] >= 22
    )
    dst_stable = (
        dst_weeks.empty
        or stats.get("dst_weeks_opening_monday", 0) == len(dst_weeks)
    )
    m5_agrees = bool(
        best_m5 is not None
        and best_m5.label == "A_native"
        and best_m5.summary.get("full_ohlc_exact_match_pct", 0.0) >= 99.0
    )

    if has_authoritative_metadata:
        return "CONFIRMED", "The data carries authoritative timezone metadata.", stats

    if session_consistent and dst_stable and m5_agrees:
        verdict = "STRONGLY SUPPORTED"
        roll_line = ""
        if roll.get("consistent_with_1700_ny_rollover"):
            roll_line = (
                f"\n\nA third, mechanically independent check agrees. The daily FX "
                f"rollover widens spreads sharply at 17:00 New York at every retail "
                f"venue. In this file the widest median spread by a wide margin falls in "
                f"hour {roll['widest_spread_hour_file_native']:02d} "
                f"({roll['widest_hour_median_spread_points']:.0f} points versus "
                f"{roll['typical_hour_median_spread_points']:.0f} for a typical hour, a "
                f"{roll['widening_ratio']}x widening). Hour "
                f"{roll['widest_spread_hour_file_native']:02d} on this clock is "
                f"{roll['implied_new_york_hour']}:00 New York - exactly the rollover. "
                f"On a UTC clock the rollover would appear at hour 21 or 22 instead. "
                f"This evidence comes from spread behaviour rather than from timestamps, "
                f"so it does not share a failure mode with the other two tests."
            )
        reasoning = (
            f"Across {n} observed weeks, {stats['pct_open_monday']:.1f}% open on Monday "
            f"and {stats['pct_close_friday']:.1f}% close on Friday, with a modal open hour "
            f"of {stats['modal_open_hour']:02d} and a modal close hour of "
            f"{stats['modal_close_hour']:02d} in the file's own clock. That is exactly the "
            "Sunday 17:00 - Friday 17:00 New York FX week displaced by +7 hours. The "
            f"pattern holds through {stats['dst_transition_weeks_observed']} weeks "
            "containing a US DST transition, which is where a fixed-offset clock would "
            "break. Independently, the tick stream reproduces the existing M5 bid OHLC "
            "under the hypothesis that both files already share one clock, with no "
            "transformation applied."
            + roll_line +
            "\n\nThis is NOT elevated to CONFIRMED: all of this evidence comes from files "
            "supplied by the same client from the same platform, so it establishes that "
            "the exports are mutually consistent and that the clock behaves exactly as a "
            "New York + 7 server clock would. It is not documentation of what the "
            "broker's server was actually configured to. Confirmation requires the "
            "broker's stated server timezone and DST rule."
        )
    elif session_consistent and not m5_agrees:
        verdict = "INCONCLUSIVE"
        reasoning = (
            "Weekly session boundaries are consistent with New York + 7, but the M5 "
            "reconstruction did not corroborate it under any tested interpretation. "
            "The two lines of evidence disagree, so no conclusion is drawn."
        )
    elif not session_consistent:
        verdict = "CONTRADICTED" if n >= 20 else "INCONCLUSIVE"
        reasoning = (
            f"Weekly boundaries do not match the New York + 7 pattern: "
            f"{stats['pct_open_monday']:.1f}% Monday opens, {stats['pct_close_friday']:.1f}% "
            f"Friday closes, modal open hour {stats['modal_open_hour']:02d}, modal close "
            f"hour {stats['modal_close_hour']:02d}."
        )
    else:
        verdict = "INCONCLUSIVE"
        reasoning = "Evidence is mixed; see the session and reconstruction tables."
    return verdict, reasoning, stats


# ---------------------------------------------------------------------------
# Research overlap
# ---------------------------------------------------------------------------

def research_overlap(
    tick_start: pd.Timestamp,
    tick_end: pd.Timestamp,
    datasets: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    rows = []
    for name, (start, end) in datasets.items():
        ov_start = max(start, tick_start)
        ov_end = min(end, tick_end)
        has = ov_start <= ov_end
        total_days = (end - start).total_seconds() / 86400.0
        ov_days = (ov_end - ov_start).total_seconds() / 86400.0 if has else 0.0
        rows.append({
            "dataset": name,
            "existing_start": start, "existing_end": end,
            "tick_start": tick_start, "tick_end": tick_end,
            "overlap_start": ov_start if has else pd.NaT,
            "overlap_end": ov_end if has else pd.NaT,
            "overlap_days": round(ov_days, 3),
            "existing_span_days": round(total_days, 3),
            "pct_of_existing_covered": round(100.0 * ov_days / total_days, 3) if total_days else 0.0,
            "uncovered_head_days": round(max(0.0, (min(end, tick_start) - start)
                                             .total_seconds() / 86400.0), 3),
            "uncovered_tail_days": round(max(0.0, (end - max(start, tick_end))
                                             .total_seconds() / 86400.0), 3),
        })
    return pd.DataFrame(rows)

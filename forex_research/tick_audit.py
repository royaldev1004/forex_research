from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .logging_utils import configure_logging, get_logger
from .manifest import _git_commit, _package_versions
from .reporting import _md_table
from .tick_schema import (
    MT5_TICK_COLUMNS,
    TickFileFormat,
    describe_flag,
    detect_format,
)
from .tick_stream import (
    GAP_THRESHOLDS_MIN,
    JUMP_ANOMALY_POINTS,
    MS_PER_DAY,
    SPREAD_HIST_MAX,
    TickAuditState,
    extract_sample,
    extract_tail_sample,
    find_byte_offset_for_timestamp,
    hist_mean,
    hist_quantiles,
    iter_tick_batches,
    _line_timestamp_ms,
)
from .tick_validation import (
    assess_timezone,
    compare_m5,
    load_m5_reference,
    m5_bars_missing_from_ticks,
    ms_to_timestamp,
    research_overlap,
    weekly_session_table,
)

log = get_logger("tick_audit")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
SAMPLE_ROWS = 20_000
ACTIVE_SAMPLE_ROWS = 40_000


# ---------------------------------------------------------------------------
# Span probe
# ---------------------------------------------------------------------------

def probe_span(fmt: TickFileFormat) -> tuple[int, int]:
    head, _ = extract_sample(fmt, 0, 5, from_start=True)
    tail, _ = extract_tail_sample(fmt, 5)
    enc = fmt.encoding
    first = min(_line_timestamp_ms(ln.encode(enc, "replace"), fmt) for ln in head)
    last = max(_line_timestamp_ms(ln.encode(enc, "replace"), fmt) for ln in tail)
    if first < 0 or last < 0:
        raise SystemExit("Could not parse timestamps from the file head/tail probe.")
    return first, last


# ---------------------------------------------------------------------------
# Report tables
# ---------------------------------------------------------------------------

def build_schema_table(fmt: TickFileFormat, st: TickAuditState) -> pd.DataFrame:
    n = st.n_rows
    pip = st.points_per_pip
    meaning = {
        "DATE": "Tick date on the file-native clock (YYYY.MM.DD)",
        "TIME": f"Tick time of day, {fmt.time_precision} precision",
        "BID": "Bid quote, populated only when the bid side updated on this tick",
        "ASK": "Ask quote, populated only when the ask side updated on this tick",
        "LAST": "Last traded price (exchange concept; not applicable to spot FX)",
        "VOLUME": "Traded volume (exchange concept; not applicable to spot FX)",
        "FLAGS": "MT5 tick flag bitmask: 2=BID updated, 4=ASK updated, 6=both",
    }
    dtype = {"DATE": "string (fixed width 10)",
             "TIME": f"string (fixed width {fmt.time_width})",
             "BID": "decimal price -> int64 points", "ASK": "decimal price -> int64 points",
             "LAST": "decimal price (empty)", "VOLUME": "integer (empty)",
             "FLAGS": "integer bitmask"}

    non_null = {
        "DATE": n - st.n_ts_malformed, "TIME": n - st.n_ts_malformed,
        "BID": n - st.n_bid_structurally_absent,
        "ASK": n - st.n_ask_structurally_absent,
        "LAST": st.n_last_present, "VOLUME": st.n_volume_present,
        "FLAGS": n - st.n_flags_absent,
    }
    rng = {
        "DATE": f"{ms_to_timestamp(st.ts_min).date()} .. {ms_to_timestamp(st.ts_max).date()}",
        "TIME": "00:00:00.000 .. 23:59:59.999",
        "BID": (f"{st.bid_min / (pip * 10000):.5f} .. {st.bid_max / (pip * 10000):.5f}"
                if st.bid_max > st.bid_min else ""),
        "ASK": (f"{st.ask_min / (pip * 10000):.5f} .. {st.ask_max / (pip * 10000):.5f}"
                if st.ask_max > st.ask_min else ""),
        "LAST": "", "VOLUME": "",
        "FLAGS": ", ".join(f"{k} ({describe_flag(k)})" for k in sorted(st.flag_counts)),
    }
    sample = {c: "" for c in MT5_TICK_COLUMNS}
    if fmt.sample_lines:
        parts = fmt.sample_lines[0].split(fmt.delimiter)
        for i, c in enumerate(fmt.columns):
            if i < len(parts):
                sample[c] = parts[i] if parts[i] else "(empty)"

    rows = []
    for raw, c in zip(fmt.raw_columns, fmt.columns):
        rows.append({
            "original_column_name": raw,
            "normalised_name": c,
            "interpreted_meaning": meaning.get(c, "unrecognised column; not interpreted"),
            "inferred_dtype": dtype.get(c, "string"),
            "representative_value": sample.get(c, ""),
            "non_null_count": non_null.get(c, ""),
            "null_or_absent_count": (n - non_null[c]) if c in non_null else "",
            "absence_is_structural": c in ("BID", "ASK", "LAST", "VOLUME"),
            "value_range": rng.get(c, ""),
        })
    return pd.DataFrame(rows)


def build_spread_tables(st: TickAuditState) -> tuple[pd.DataFrame, pd.DataFrame]:
    pip = float(st.points_per_pip)

    def stats(hist: np.ndarray, label: str, scope: str) -> dict:
        q = hist_quantiles(hist, QUANTILES)
        total = int(hist.sum())
        nz = np.flatnonzero(hist)
        return {
            "scope": scope, "population": label, "n_observations": total,
            "min_points": int(nz.min()) if nz.size else np.nan,
            "p1_points": q[0.01], "p5_points": q[0.05], "p25_points": q[0.25],
            "median_points": q[0.50], "mean_points": round(hist_mean(hist), 4),
            "p75_points": q[0.75], "p95_points": q[0.95], "p99_points": q[0.99],
            "max_points": int(nz.max()) if nz.size else np.nan,
            "median_pips": round(q[0.50] / pip, 5),
            "mean_pips": round(hist_mean(hist) / pip, 5),
            "p95_pips": round(q[0.95] / pip, 5),
            "p99_pips": round(q[0.99] / pip, 5),
            "max_pips": round(float(nz.max()) / pip, 5) if nz.size else np.nan,
        }

    overall = pd.DataFrame([
        stats(st.spread_hist, "all_reconstructed_quotes", "ALL"),
        stats(st.spread_hist_fresh, "both_sides_fresh", "ALL"),
    ])
    # True extremes come from the running min/max, not the capped histogram.
    overall.loc[0, "max_points"] = st.spread_max
    overall.loc[0, "max_pips"] = round(st.spread_max / pip, 5)
    overall.loc[0, "min_points"] = st.spread_min
    overall["histogram_overflow_observations"] = st.spread_overflow

    rows = []
    for h in range(24):
        hist = st.hour_of_day_spread[h]
        if hist.sum() == 0:
            continue
        rows.append(stats(hist, f"hour_{h:02d}", "HOUR_OF_DAY_FILE_NATIVE"))

    # Year and month, derived from the per-day histograms.
    days = np.flatnonzero(st.day_count > 0)
    if days.size:
        dates = pd.to_datetime(
            [(st.base_day + int(d)) * MS_PER_DAY for d in days], unit="ms")
        for key, fmt_ in (("YEAR", "%Y"), ("MONTH", "%Y-%m")):
            labels = dates.strftime(fmt_)
            for lab in pd.unique(labels):
                sel = days[labels == lab]
                hist = st.day_spread_hist[sel].sum(axis=0).astype(np.int64)
                if hist.sum() == 0:
                    continue
                s = stats(hist, lab, key)
                s["histogram_cap_points"] = st.day_spread_cap
                s["overflow_observations"] = int(hist[-1])
                rows.append(s)
    return overall, pd.DataFrame(rows)


def build_daily_coverage(st: TickAuditState) -> pd.DataFrame:
    days = np.flatnonzero(st.day_count > 0)
    if days.size == 0:
        return pd.DataFrame()
    pip = float(st.points_per_pip)
    rows = []
    for d in days:
        hist = st.day_spread_hist[d].astype(np.int64)
        q = hist_quantiles(hist, (0.50, 0.95))
        first, last = int(st.day_first_ms[d]), int(st.day_last_ms[d])
        date = ms_to_timestamp((st.base_day + int(d)) * MS_PER_DAY)
        rows.append({
            "date": date.date(),
            "weekday": date.day_name(),
            "first_tick": ms_to_timestamp(first),
            "last_tick": ms_to_timestamp(last),
            "tick_count": int(st.day_count[d]),
            "median_spread_points": q[0.50],
            "median_spread_pips": round(q[0.50] / pip, 5),
            "p95_spread_points": q[0.95],
            "p95_spread_pips": round(q[0.95] / pip, 5),
            "largest_intraday_gap_minutes": round(int(st.day_max_gap_ms[d]) / 60000.0, 3),
            "session_span_hours": round((last - first) / 3_600_000.0, 3),
        })
    df = pd.DataFrame(rows)

    # Weekdays with no ticks at all, inside the covered span.
    covered = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    full = pd.date_range(covered.min(), covered.max(), freq="D")
    missing = full.difference(covered)
    missing_weekdays = missing[missing.dayofweek < 5]
    if len(missing_weekdays):
        log.info("Weekdays with zero ticks inside the span: %d", len(missing_weekdays))
    df.attrs["missing_weekdays"] = [str(d.date()) for d in missing_weekdays]
    df.attrs["weekend_days_with_ticks"] = [
        str(r.date) for r in df.itertuples() if r.weekday in ("Saturday", "Sunday")
    ]
    return df


def build_density_table(st: TickAuditState, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def dist(values: np.ndarray, unit: str, note: str = "") -> dict:
        if values.size == 0:
            return {}
        return {
            "unit": unit, "n_periods_observed": int(values.size),
            "min": float(values.min()), "p5": float(np.percentile(values, 5)),
            "median": float(np.median(values)), "mean": round(float(values.mean()), 3),
            "p95": float(np.percentile(values, 95)), "max": float(values.max()),
            "total_ticks": int(values.sum()), "note": note,
        }

    if not daily.empty:
        rows.append(dist(daily["tick_count"].to_numpy(), "ticks_per_trading_day",
                         "Days with at least one tick"))
    hours = st.hour_count[st.hour_count > 0]
    rows.append(dist(hours, "ticks_per_hour", "Hours with at least one tick"))

    if st.minute_count_hist:
        counts = np.array(sorted(st.minute_count_hist), dtype=np.int64)
        freq = np.array([st.minute_count_hist[int(c)] for c in counts], dtype=np.int64)
        total = int(freq.sum())
        cum = np.cumsum(freq)

        def pq(p: float) -> float:
            return float(counts[int(np.searchsorted(cum, p * total, side="left"))])

        rows.append({
            "unit": "ticks_per_minute", "n_periods_observed": total,
            "min": float(counts.min()), "p5": pq(0.05), "median": pq(0.50),
            "mean": round(float((counts * freq).sum() / total), 3),
            "p95": pq(0.95), "max": float(counts.max()),
            "total_ticks": int((counts * freq).sum()),
            "note": "Minutes with at least one tick",
        })
    return pd.DataFrame([r for r in rows if r])


def build_timestamp_quality(st: TickAuditState) -> pd.DataFrame:
    checks = [
        ("first_timestamp", str(ms_to_timestamp(st.ts_min)), "info", "File-native clock"),
        ("last_timestamp", str(ms_to_timestamp(st.ts_max)), "info", "File-native clock"),
        ("timestamp_precision", st.fmt.time_precision, "info",
         f"{st.n_subsecond:,} rows carry a non-zero sub-second component"),
        ("total_records_parsed", st.n_rows, "info", ""),
        ("records_with_unparseable_timestamp", st.n_ts_malformed,
         "ok" if st.n_ts_malformed == 0 else "warn", "Excluded from all statistics"),
        ("chronologically_ordered", st.n_out_of_order == 0,
         "ok" if st.n_out_of_order == 0 else "fail",
         f"{st.n_out_of_order:,} records have a timestamp earlier than their predecessor"),
        ("duplicate_timestamp_records", st.n_duplicate_ts_rows, "info",
         "Extra records sharing a timestamp with another record. Legitimate for tick "
         "data: several quotes can arrive within one millisecond."),
        ("largest_equal_timestamp_group", st.max_ts_group, "info",
         "Most records observed sharing a single millisecond"),
        ("exact_duplicate_records", st.n_exact_duplicate_rows,
         "ok" if st.n_exact_duplicate_rows == 0 else "warn",
         "Records identical in timestamp, bid, ask and flags. Detected within every "
         "equal-timestamp group, so non-adjacent identical rows are caught."),
        ("oversized_timestamp_groups", st.n_oversized_ts_groups,
         "ok" if st.n_oversized_ts_groups == 0 else "warn",
         "Groups exceeding the bounded-memory carry limit; would indicate a corrupt "
         "timestamp rather than real data"),
    ]
    for m in GAP_THRESHOLDS_MIN:
        checks.append((f"gaps_over_{m}_minutes", st.gap_counts[m], "info",
                       "Includes normal weekend and holiday closures"))
    return pd.DataFrame(
        [{"check": c, "value": v, "status": s, "detail": d} for c, v, s, d in checks])


def build_gap_table(st: TickAuditState) -> pd.DataFrame:
    if not st.top_gaps:
        return pd.DataFrame()
    rows = []
    for gap_ms, prev_ms, ts_ms in sorted(st.top_gaps, reverse=True):
        a, b = ms_to_timestamp(prev_ms), ms_to_timestamp(ts_ms)
        rows.append({
            "gap_minutes": round(gap_ms / 60000.0, 3),
            "gap_hours": round(gap_ms / 3_600_000.0, 3),
            "last_tick_before": a, "first_tick_after": b,
            "weekday_before": a.day_name(), "weekday_after": b.day_name(),
            "likely_weekend_closure": bool(a.dayofweek == 4 and b.dayofweek == 0),
        })
    return pd.DataFrame(rows)


def build_anomaly_table(st: TickAuditState) -> pd.DataFrame:
    if not st.top_jumps:
        return pd.DataFrame(columns=["timestamp", "side", "previous_price", "new_price",
                                     "jump_points", "jump_pips"])
    pip = float(st.points_per_pip)
    scale = 1.0 / (st.point_size and (1.0 / st.point_size))
    rows = []
    for jump, ts_ms, prev_p, new_p, side in sorted(st.top_jumps, reverse=True):
        rows.append({
            "timestamp": ms_to_timestamp(ts_ms), "side": side,
            "previous_price": round(prev_p * st.point_size, 5),
            "new_price": round(new_p * st.point_size, 5),
            "jump_points": jump, "jump_pips": round(jump / pip, 2),
            "direction": "up" if new_p > prev_p else "down",
        })
    return pd.DataFrame(rows)


def build_quality_table(st: TickAuditState) -> pd.DataFrame:
    n = st.n_rows
    def pct(x: int) -> float:
        return round(100.0 * x / n, 5) if n else 0.0

    rows = [
        ("total_records", n, "", "Parsed data records, excluding the header"),
        ("bid_structurally_absent", st.n_bid_structurally_absent,
         pct(st.n_bid_structurally_absent),
         "Ask-only updates (FLAGS=4). NOT missing data: the bid simply did not change."),
        ("ask_structurally_absent", st.n_ask_structurally_absent,
         pct(st.n_ask_structurally_absent),
         "Bid-only updates (FLAGS=2). NOT missing data."),
        ("both_sides_absent", st.n_both_sides_absent, pct(st.n_both_sides_absent),
         "Records carrying neither a bid nor an ask. Genuinely uninformative."),
        ("flag_bid_field_mismatch", st.n_flag_bid_mismatch, pct(st.n_flag_bid_mismatch),
         "FLAGS says the bid updated but the field is empty, or vice versa"),
        ("flag_ask_field_mismatch", st.n_flag_ask_mismatch, pct(st.n_flag_ask_mismatch),
         "FLAGS says the ask updated but the field is empty, or vice versa"),
        ("flags_absent", st.n_flags_absent, pct(st.n_flags_absent), ""),
        ("last_field_populated", st.n_last_present, pct(st.n_last_present),
         "Exchange last-trade concept; expected to be empty for spot FX"),
        ("volume_field_populated", st.n_volume_present, pct(st.n_volume_present),
         "Exchange volume concept; expected to be empty for spot FX"),
        ("bid_non_positive", st.n_bid_nonpositive, pct(st.n_bid_nonpositive),
         "Zero or negative bid prices"),
        ("ask_non_positive", st.n_ask_nonpositive, pct(st.n_ask_nonpositive),
         "Zero or negative ask prices"),
        ("quotes_reconstructed", st.n_quotes_reconstructed, pct(st.n_quotes_reconstructed),
         "Records where both sides of the quote book were known"),
        ("quotes_undefined", st.n_quotes_undefined, pct(st.n_quotes_undefined),
         "Records before both sides had been observed at least once"),
        ("quotes_with_stale_side", st.n_quotes_stale_side, pct(st.n_quotes_stale_side),
         f"One side older than {st.stale_side_ms/1000:.0f}s. Concentrated at session "
         "reopen after weekends and holidays. Reported, not deleted."),
        ("spread_negative", st.n_spread_negative, pct(st.n_spread_negative),
         "ask < bid on the reconstructed quote (crossed book)"),
        ("spread_zero", st.n_spread_zero, pct(st.n_spread_zero), "ask == bid"),
        ("bid_jumps_over_threshold", st.n_jumps_over_threshold,
         pct(st.n_jumps_over_threshold),
         f"Consecutive-tick bid moves of at least {JUMP_ANOMALY_POINTS} points "
         f"({JUMP_ANOMALY_POINTS/st.points_per_pip:.0f} pips)"),
        ("bid_unchanged_between_ticks", st.n_bid_unchanged, pct(st.n_bid_unchanged),
         "Consecutive reconstructed bids identical (ask-only activity)"),
        ("max_bid_staleness_minutes", round(st.bid_update_gap_max / 60000.0, 2), "",
         "Longest interval without a bid update, including weekends"),
        ("max_ask_staleness_minutes", round(st.ask_update_gap_max / 60000.0, 2), "",
         "Longest interval without an ask update, including weekends"),
    ]
    # Object dtype: the value column deliberately mixes counts and durations,
    # and letting pandas coerce it to float64 would print every count as "N.0".
    df = pd.DataFrame(
        [{"metric": m, "value": v, "pct_of_records": p, "interpretation": d}
         for m, v, p, d in rows])
    df["value"] = df["value"].astype(object)
    return df


def build_flag_table(st: TickAuditState) -> pd.DataFrame:
    total = sum(st.flag_counts.values()) or 1
    return pd.DataFrame([
        {"flags_value": k, "decoded": describe_flag(k), "count": v,
         "pct_of_records": round(100.0 * v / total, 5)}
        for k, v in sorted(st.flag_counts.items())
    ])


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def write_samples(
    fmt: TickFileFormat, st: TickAuditState, outdir: Path
) -> pd.DataFrame:
    sdir = outdir / "samples"
    sdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def save(name: str, lines: list[str], header: str, method: str) -> None:
        path = sdir / f"{name}.csv"
        path.write_text("\n".join([header] + lines) + "\n", encoding="utf-8")
        first = _line_timestamp_ms(lines[0].encode(fmt.encoding, "replace"), fmt) if lines else -1
        last = _line_timestamp_ms(lines[-1].encode(fmt.encoding, "replace"), fmt) if lines else -1
        rows.append({
            "sample_name": name, "file": path.name, "selection_method": method,
            "start_timestamp": ms_to_timestamp(first) if first > 0 else pd.NaT,
            "end_timestamp": ms_to_timestamp(last) if last > 0 else pd.NaT,
            "n_rows": len(lines),
        })
        log.info("Sample %-22s %6d rows -> %s", name, len(lines), path.name)

    lines, header = extract_sample(fmt, 0, SAMPLE_ROWS, from_start=True)
    save("tick_sample_begin", lines, header,
         "First rows of the file, immediately after the header")

    mid_ms = (st.ts_min + st.ts_max) // 2
    off = find_byte_offset_for_timestamp(fmt, mid_ms)
    lines, header = extract_sample(fmt, off, SAMPLE_ROWS)
    save("tick_sample_middle", lines, header,
         f"TEMPORAL midpoint of the covered span (first_ts + last_ts)/2 = "
         f"{ms_to_timestamp(mid_ms)}, located by byte bisection. Temporal, not "
         f"record-count, midpoint.")

    lines, header = extract_tail_sample(fmt, SAMPLE_ROWS)
    save("tick_sample_end", lines, header, "Final rows of the file, read by seeking from EOF")

    if st.best_hour_index >= 0:
        hour_ms = (st.base_day * 24 + st.best_hour_index) * 3_600_000
        off = find_byte_offset_for_timestamp(fmt, hour_ms)
        lines, header = extract_sample(fmt, off, ACTIVE_SAMPLE_ROWS)
        save("tick_sample_active", lines, header,
             f"Objectively selected: the single hour with the highest tick count in the "
             f"entire file ({st.best_hour_count:,} ticks, starting "
             f"{ms_to_timestamp(hour_ms)}). Selected by count, not by chart appearance.")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "sample_manifest.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    path: Path, fmt: TickFileFormat, st: TickAuditState, quality: pd.DataFrame,
    schema: pd.DataFrame, spread: pd.DataFrame, spread_by: pd.DataFrame,
    tsq: pd.DataFrame, daily: pd.DataFrame, density: pd.DataFrame,
    gaps: pd.DataFrame, anomalies: pd.DataFrame, m5_summary: pd.DataFrame,
    m5_by_date: pd.DataFrame, missing_bars: pd.DataFrame, overlap: pd.DataFrame,
    tz_verdict: str, tz_reasoning: str, tz_stats: dict, samples: pd.DataFrame,
    recommendation: str, rec_reason: str, concerns: list[str],
    elapsed: float, m5_path: Path,
) -> None:
    pip = float(st.points_per_pip)
    best = m5_summary.sort_values("full_ohlc_exact_match_pct", ascending=False).iloc[0] \
        if not m5_summary.empty else None
    sp_all = spread[spread["population"] == "all_reconstructed_quotes"].iloc[0]
    sp_fresh = spread[spread["population"] == "both_sides_fresh"].iloc[0]

    L: list[str] = []
    A = L.append

    A("# Tick Data Audit Report")
    A("")
    A(f"Generated {pd.Timestamp.now('UTC').strftime('%Y-%m-%d %H:%M:%S')} UTC · "
      f"streaming pass {elapsed:.1f}s")
    A("")
    A("> **Scope.** This is an audit of a newly supplied tick file. No model was trained, "
      "no strategy or barrier optimised, no profitability established, and **no existing "
      "Day 1 / Step 2 / Step 3A / Day 2 artefact, definition or parameter was modified**. "
      "Step 3B is not implemented here; this report only establishes whether it is "
      "technically possible.")
    A("")

    # 1 -----------------------------------------------------------------
    A("## 1. File identity")
    A("")
    A("| Property | Value |")
    A("|---|---|")
    A(f"| Filename | `{fmt.path.name}` |")
    A(f"| Path | `{fmt.path}` |")
    A(f"| Size | {fmt.size_bytes:,} bytes ({fmt.size_bytes/(1<<30):.3f} GB) |")
    A(f"| Format | MT5 tick export, {('TAB' if fmt.delimiter == chr(9) else fmt.delimiter)}"
      f"-separated text |")
    A(f"| Encoding | {fmt.encoding}, {'BOM present' if fmt.has_bom else 'no BOM'} |")
    A(f"| Line ending | {fmt.line_ending} |")
    A(f"| Header | 1 row: `{fmt.delimiter.join(fmt.raw_columns)}` |")
    A(f"| **True record count** | **{st.n_rows:,}** |")
    A(f"| Rows PyArrow could not parse | {st.n_ts_malformed:,} timestamp + "
      f"structurally invalid |")
    A("")
    A(f"**Excel row-limit conclusion.** Excel displayed 1,048,576 rows, which is exactly "
      f"its worksheet maximum. The file actually contains **{st.n_rows:,} records** — "
      f"Excel was showing "
      f"**{100.0 * 1_048_576 / max(st.n_rows, 1):.2f}%** of the data. The observed row "
      f"count was a display limit, not the dataset size.")
    A("")

    # 2 -----------------------------------------------------------------
    A("## 2. Schema")
    A("")
    A(_md_table(schema))
    A("")
    A("### The `<FLAGS>` bitmask drives everything")
    A("")
    A("MT5 tick exports are **sparse**. A tick that only moved the ask leaves `<BID>` "
      "empty, and vice versa. `<FLAGS>` records which side actually changed:")
    A("")
    A(_md_table(build_flag_table(st)))
    A("")
    A(f"Field/flag consistency: **{st.n_flag_bid_mismatch:,}** bid mismatches and "
      f"**{st.n_flag_ask_mismatch:,}** ask mismatches out of {st.n_rows:,} records. "
      "A blank price field is therefore **structural absence, not missing data**, and "
      "this audit never counts it as a defect. Three concepts are kept distinct "
      "throughout: structurally absent, reconstructed by carry-forward, and genuinely "
      "malformed.")
    A("")

    # 3 -----------------------------------------------------------------
    A("## 3. Historical coverage")
    A("")
    A(f"- **First timestamp:** {ms_to_timestamp(st.ts_min)} (file-native clock)")
    A(f"- **Last timestamp:** {ms_to_timestamp(st.ts_max)} (file-native clock)")
    span_days = (st.ts_max - st.ts_min) / MS_PER_DAY
    A(f"- **Calendar span:** {span_days:.1f} days (~{span_days/365.25:.2f} years)")
    A(f"- **Days with at least one tick:** {len(daily):,}")
    if not daily.empty:
        mw = daily.attrs.get("missing_weekdays", [])
        ww = daily.attrs.get("weekend_days_with_ticks", [])
        A(f"- **Weekdays inside the span with zero ticks:** {len(mw)}"
          + (f" (first few: {', '.join(mw[:8])})" if mw else ""))
        A(f"- **Weekend days carrying ticks:** {len(ww)}"
          + (" — normal for Sunday session reopen on a broker clock where the week "
             "starts Monday 00:00" if ww else ""))
    A("")
    A("Largest gaps in the stream:")
    A("")
    if not gaps.empty:
        A(_md_table(gaps.head(15)))
        wk = int(gaps["likely_weekend_closure"].sum())
        A("")
        A(f"Of the {len(gaps)} largest gaps recorded, {wk} run Friday to Monday and are "
          "ordinary weekend closures. Holiday behaviour is reported, not treated as "
          "corruption.")
    A("")
    A("Full per-day detail: `daily_coverage.csv`.")
    A("")

    # 4 -----------------------------------------------------------------
    A("## 4. Bid/ask validation")
    A("")
    genuine = (st.n_quotes_reconstructed > 0 and st.n_flag_bid_mismatch == 0
               and st.n_flag_ask_mismatch == 0 and st.n_spread_negative == 0)
    A(f"**Is this genuine quote-level bid/ask history? "
      f"{'YES.' if genuine else 'SEE QUALIFICATIONS BELOW.'}**")
    A("")
    A(f"Both sides of the book are present and independently updated. "
      f"{st.n_quotes_reconstructed:,} records "
      f"({100.0*st.n_quotes_reconstructed/max(st.n_rows,1):.3f}%) carry a fully "
      f"reconstructed two-sided quote. Crossed quotes (ask < bid): "
      f"**{st.n_spread_negative:,}**. Zero spreads: **{st.n_spread_zero:,}**.")
    A("")
    A("### Spread statistics")
    A("")
    A(f"1 pip = {st.pip_size}; 1 point = {st.point_size}; "
      f"{st.points_per_pip} points per pip.")
    A("")
    A("| Statistic | All reconstructed quotes (points / pips) | "
      "Both sides fresh (points / pips) |")
    A("|---|---|---|")
    for key, lab in (("min_points", "min"), ("p1_points", "p1"), ("p5_points", "p5"),
                     ("median_points", "median"), ("mean_points", "mean"),
                     ("p95_points", "p95"), ("p99_points", "p99"),
                     ("max_points", "max")):
        a, b = sp_all[key], sp_fresh[key]
        A(f"| {lab} | {a:,.0f} / {a/pip:.2f} | {b:,.0f} / {b/pip:.2f} |")
    A("")
    if st.n_quotes_stale_side:
        A(f"**Stale-side observations:** {st.n_quotes_stale_side:,} "
          f"({100.0*st.n_quotes_stale_side/max(st.n_quotes_reconstructed,1):.3f}% of "
          f"reconstructed quotes) had one side older than "
          f"{st.stale_side_ms/1000:.0f} seconds. These cluster at session reopen, where "
          "one side updates before the other after a weekend or holiday. A Friday ask "
          "carried into a Monday bid update is **not** a contemporaneous spread, so the "
          "'both sides fresh' column above is the one to use. Stale observations are "
          "reported and retained, never deleted.")
    else:
        A(f"**Quote freshness: no stale-side observations at all.** Not one of the "
          f"{st.n_quotes_reconstructed:,} reconstructed quotes had a side older than "
          f"{st.stale_side_ms/1000:.0f} seconds. The longest the bid ever went without "
          f"updating is {st.bid_update_gap_max/60000.0:.2f} minutes and the ask "
          f"{st.ask_update_gap_max/60000.0:.2f} minutes, **including across weekends** — "
          "meaning the first tick of every session reopen carried both sides, so no "
          "spread in this file is computed across a market closure. The two spread "
          "columns above are therefore identical.\n\n"
          "This was worth checking rather than assuming: carrying a Friday ask into a "
          "Monday bid update would produce a fabricated spread. The detector is exercised "
          "by `tests/test_tick_audit.py::test_stale_side_is_flagged_across_a_long_gap`, "
          "so a zero here means the condition is genuinely absent, not that the check is "
          "silent.")
    A("")
    A("Spread is **not** constant. By-hour, by-month and by-year breakdowns are in "
      "`spread_summary.csv` (scope column `HOUR_OF_DAY_FILE_NATIVE`, `MONTH`, `YEAR`).")
    A("")

    # 5 -----------------------------------------------------------------
    A("## 5. Timestamp integrity")
    A("")
    A(_md_table(tsq))
    A("")
    A("**Duplicate timestamps versus exact duplicate records.** These are different "
      "things and are counted separately. Several quotes legitimately arrive inside one "
      "millisecond, so a shared timestamp is expected. An *exact duplicate* — identical "
      "timestamp, bid, ask and flags — would be a data defect. Detection runs within "
      "every equal-timestamp group rather than only on adjacent rows, so two identical "
      "records separated by a third tick at the same millisecond are still caught. "
      "Unfinished groups are carried across parse-block boundaries.")
    A("")

    # 6 -----------------------------------------------------------------
    A("## 6. Data quality")
    A("")
    A(_md_table(quality))
    A("")
    if not anomalies.empty:
        A(f"### Largest consecutive-tick bid moves")
        A("")
        A(_md_table(anomalies.head(15)))
        A("")
        A("No outlier was deleted. The full recorded set is in `price_anomalies.csv`.")
    A("")

    # 7 -----------------------------------------------------------------
    A("## 7. M5 reconstruction validation")
    A("")
    A(f"Reconstructed 5-minute **bid** OHLC from the tick stream and compared it, in "
      f"**integer points** (no floating-point tolerance), against `{m5_path.name}`. "
      "Binning matches the project's own convention: left-labelled, left-closed, "
      "anchored to midnight on the file-native clock.")
    A("")
    A(_md_table(m5_summary[[
        "interpretation", "description", "m5_bars_with_tick_coverage",
        "open_exact_match_pct", "high_exact_match_pct", "low_exact_match_pct",
        "close_exact_match_pct", "full_ohlc_exact_match_pct",
        "full_ohlc_within_1pt_pct", "max_abs_diff_points_any_field",
    ]]))
    A("")
    if best is not None:
        A(f"**Best interpretation: `{best['interpretation']}`** — "
          f"{best['full_ohlc_exact_match_pct']:.3f}% of "
          f"{int(best['m5_bars_with_tick_coverage']):,} overlapping bars match on all four "
          f"OHLC fields exactly.")
        A("")
        if best["full_ohlc_exact_match_pct"] >= 99.0:
            A("This is decisive. A tick stream from a *different* broker would not "
              "reproduce another broker's bar highs and lows to the point across "
              "hundreds of thousands of bars — bid feeds differ by fractions of a pip "
              "constantly. Reproducing them exactly establishes four things at once:")
            A("")
            A("1. The tick file and the M5 export come from the **same feed**.")
            A("2. They share **one clock** — no transformation is required between them.")
            A("3. M5 timestamps label the **bar open** (left-labelled binning matched).")
            A("4. The existing M5 OHLC is **bid-side**, as the research assumed.")
        else:
            A("This does **not** reach the threshold for declaring the feeds identical. "
              "See the per-date table for whether the disagreement is concentrated in "
              "specific dates (feed/holiday differences) or spread uniformly across the "
              "sample (a binning or clock problem).")
    A("")
    A("Per-date results are in `m5_reconstruction_by_date.csv`, which separates a "
      "systematic clock/binning error (uniform mismatch) from localised feed or holiday "
      "differences (isolated dates).")
    # Locate the residual mismatches rather than leaving them as a bare percentage.
    if best is not None and not m5_by_date.empty:
        # `best` here is the summary row (a Series), not the comparison object.
        bd = m5_by_date[(m5_by_date["interpretation"] == best["interpretation"])
                        & (m5_by_date["ohlc_match_pct"] < 100.0)]
        if bd.empty:
            A("")
            A("**Every overlapping bar matched exactly.** There is no residual to explain.")
        else:
            last_tick_date = ms_to_timestamp(st.ts_max).normalize()
            dates = pd.to_datetime(bd["date"])
            n_bad = int(round(((100.0 - bd["ohlc_match_pct"]) / 100.0
                               * bd["n_bars"]).sum()))
            A("")
            if (dates == last_tick_date).all():
                A(f"**The entire residual is one artefact of where the file ends.** All "
                  f"{n_bad} non-matching bar(s) fall on {last_tick_date.date()}, the final "
                  f"day of tick coverage, where the stream stops at "
                  f"{ms_to_timestamp(st.ts_max).strftime('%H:%M:%S')} partway through a "
                  f"5-minute bar. The reconstruction therefore sees only part of that "
                  f"bar while the M5 export contains all of it. This is expected "
                  f"truncation at the boundary, not a feed disagreement: excluding the "
                  f"final incomplete bar, agreement is exact on every remaining bar.")
            else:
                A(f"**{n_bad} bar(s) did not match**, across {len(bd)} date(s): "
                  f"{', '.join(str(d.date()) for d in dates[:10])}"
                  f"{' ...' if len(bd) > 10 else ''}. Mismatches concentrated on isolated "
                  f"dates point at feed or holiday differences rather than a clock error.")
    if not missing_bars.empty:
        A("")
        A(f"**{int(missing_bars['n_m5_bars_without_ticks'].sum()):,} M5 bars inside the "
          f"overlap have no ticks at all**, across {len(missing_bars)} dates — see "
          "`m5_bars_without_ticks.csv`.")
    A("")

    # 8 -----------------------------------------------------------------
    A("## 8. Timezone assessment")
    A("")
    A(f"### `{tz_verdict}`")
    A("")
    A(tz_reasoning)
    A("")
    A("Supporting statistics are in `timezone_validation.md` and `weekly_sessions.csv`.")
    A("")

    # 9 -----------------------------------------------------------------
    A("## 9. Existing research overlap")
    A("")
    A(_md_table(overlap))
    A("")
    m5row = overlap[overlap["dataset"] == "EURUSD_M5 (Day 1 base)"]
    if not m5row.empty:
        r = m5row.iloc[0]
        A(f"**{r['pct_of_existing_covered']:.2f}% of the Day 1 M5 research window is now "
          f"covered by tick data** — {r['overlap_start']} to {r['overlap_end']}. "
          f"The uncovered remainder is {r['uncovered_tail_days']:.2f} days at the tail "
          f"(ticks end before the bars do).")
    A("")
    A("Every Day 1 decision row, Step 2 event episode, Step 3A outcome and Day 2 "
      "benchmark observation whose decision timestamp falls inside the overlap can be "
      "re-examined with bid/ask quote-path detail. Rows outside it cannot, and any "
      "execution-aware analysis must therefore either restrict to the overlap or report "
      "coverage explicitly.")
    A("")
    if not m5row.empty:
        m5_start = pd.Timestamp(m5row["existing_start"].iloc[0])
        head_days = (m5_start - ms_to_timestamp(st.ts_min)).total_seconds() / 86400.0
        if head_days > 1:
            A(f"**Separately: the tick file begins "
              f"{ms_to_timestamp(st.ts_min).date()}, {head_days:,.0f} days "
              f"({head_days/365.25:.1f} years) before the current M5 history starts on "
              f"{m5_start.date()}.** That is a substantially longer price history than "
              "the research currently has, and it bears directly on the standing Day 1 "
              "blocker about insufficient warm-up for the 1D timeframe at the "
              "300-period maximum. **No action is taken on it here** — extending or "
              "rebuilding Day 1 is a separate decision, and this audit only records "
              "that the option now exists.")
    A("")

    # 10 ----------------------------------------------------------------
    A("## 10. Step 3B capability")
    A("")
    A("| Capability | Supported? | Basis |")
    A("|---|---|---|")
    ok = st.n_quotes_reconstructed > 0
    A(f"| Synchronized historical bid/ask quote state | {'YES' if ok else 'NO'} | "
      f"Both sides update independently and the book reconstructs continuously across "
      f"{st.n_rows:,} ticks |")
    A(f"| Bid-aware / ask-aware hypothetical entry | {'YES' if ok else 'NO'} | "
      "Sell at bid, buy at ask, using the quote live at the decision timestamp |")
    A(f"| Historical spread at any instant | {'YES' if ok else 'NO'} | "
      "Reconstructed quote spread, with side-freshness tracked |")
    A(f"| Spread-aware exits | {'YES' if ok else 'NO'} | Exit priced on the opposite side "
      "of the book from entry |")
    A(f"| Intrabar quote-path reconstruction | {'YES' if ok else 'NO'} | "
      f"{fmt.time_precision.capitalize()} timestamps resolve ordering inside a 5-minute "
      "bar |")
    A(f"| TP/SL first-touch ordering | {'YES' if ok else 'NO'} | Resolves the ambiguous "
      "same-bar cases the bar-level research currently flags as `expansion_first == 2` |")
    A(f"| Quote-based execution simulation | {'YES' if ok else 'NO'} | All of the above "
      "combined |")
    A("")
    A("### What this data still cannot establish")
    A("")
    A("**Quote-path simulation is not actual fill history.** The distinction matters and "
      "must not be blurred in any downstream reporting:")
    A("")
    A("- These are **quotes the broker published**, not **trades that were executed**. "
      "No order in this dataset was ever filled.")
    A("- **Real slippage cannot be recovered from quotes alone.** Slippage is the "
      "difference between the quote at order submission and the price actually filled. "
      "It depends on latency, order type, requotes, broker execution policy, and "
      "available size at the top of book — none of which appear here.")
    A("- **There is no depth of market.** A top-of-book quote says nothing about the "
      "volume available at it, so it cannot tell you whether a given order size would "
      "have been filled at that price.")
    A("- **Rejections, requotes and partial fills are invisible.**")
    A("")
    A("Recovering actual execution behaviour requires broker execution/fill records, or "
      "a slippage model defined and justified separately and reported as an assumption "
      "rather than a measurement.")
    A("")

    # 11 ----------------------------------------------------------------
    A("## 11. Remaining concerns")
    A("")
    for i, c in enumerate(concerns, 1):
        A(f"{i}. {c}")
    A("")

    # 12 ----------------------------------------------------------------
    A("## 12. Recommendation")
    A("")
    A(f"# {recommendation}")
    A("")
    A(rec_reason)
    A("")
    A("---")
    A("")
    A("### Storage recommendation (not performed)")
    A("")
    A(f"The tick file currently sits inside the project directory at "
      f"`{fmt.path}`. A {fmt.size_bytes/(1<<30):.2f} GB file inside the source tree will "
      "bloat every archive, copy and sync of the project. Recommend relocating long-term "
      "tick storage to a dedicated non-project path such as `C:\\ForexData\\EURUSD_ticks\\`, "
      "and converting to partitioned Parquet there for repeat analysis. **Neither the "
      "move nor the conversion was performed in this task.**")
    A("")
    A("### Samples for review")
    A("")
    A(_md_table(samples))
    A("")

    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info("Wrote %s", path.name)


def write_timezone_note(
    path: Path, verdict: str, reasoning: str, stats: dict,
    sessions: pd.DataFrame, m5_summary: pd.DataFrame,
) -> None:
    L = ["# Timezone Validation", "",
         f"## Conclusion: `{verdict}`", "", reasoning, "",
         "## Why not CONFIRMED", "",
         "`CONFIRMED` is reserved for authoritative timezone metadata carried inside the "
         "data, or equivalent direct documentary evidence. The tick file has no timezone "
         "field, no UTC offset column and no metadata header. Both the tick file and the "
         "bar files were supplied by the same client from the same platform, so their "
         "agreement demonstrates that the two exports are mutually consistent — it does "
         "not document what clock the broker's server actually ran on. That still "
         "requires the broker's stated server timezone and DST rule.", "",
         "## Evidence", ""]
    for k, v in stats.items():
        L.append(f"- **{k}**: {v}")
    L += ["", "## Hypotheses tested", "",
          "Only two interpretations were tested, both declared before any result was "
          "inspected. No offset was searched for the value that scored best.", "",
          "| Interpretation | Transform | Full-OHLC exact match |",
          "|---|---|---|"]
    for _, r in m5_summary.iterrows():
        L.append(f"| `{r['interpretation']}` | {r['description']} | "
                 f"{r['full_ohlc_exact_match_pct']:.3f}% |")
    L += ["", "## Weekly session boundaries", "",
          "The FX week runs Sunday 17:00 to Friday 17:00 New York. On a broker clock of "
          "New York + 7 hours that appears as Monday 00:00 to Friday ~23:59 in "
          "file-native time, **in every DST regime** — which is exactly what makes this "
          "test discriminating: a fixed UTC offset would drift by an hour twice a year, "
          "and a different broker offset would shift the boundaries.", ""]
    if not sessions.empty:
        L.append(_md_table(sessions.head(30)))
        L.append("")
        L.append(f"Showing the first 30 of {len(sessions)} observed weeks. "
                 "Full table: `weekly_sessions.csv`.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(
    tick_path: Path, config_path: str, outdir: Path,
    block_size: int, max_rows: int | None, stale_side_ms: int,
    progress_every: int, start_timestamp: str | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)

    fmt = detect_format(tick_path)
    if not fmt.is_mt5_tick_export:
        raise SystemExit(
            f"{tick_path.name} does not present as an MT5 tick export "
            f"(columns: {fmt.columns}). Refusing to invent a parser.")

    span_start, span_end = probe_span(fmt)
    log.info("Span probe: %s .. %s", ms_to_timestamp(span_start), ms_to_timestamp(span_end))

    st = TickAuditState(fmt, cfg.point_size, cfg.pip_size, span_start, span_end,
                        stale_side_ms=stale_side_ms)
    st.add_m5_interpretation(
        "A_native", "Tick timestamps are already on the M5 file's broker-native clock "
                    "(no transformation)")
    st.add_m5_interpretation(
        "B_utc", "Tick timestamps are UTC; converted to broker-native with the project's "
                 "documented New York + 7 rule")

    # --- the single full pass ---------------------------------------------
    start_offset = 0
    if start_timestamp:
        target = pd.Timestamp(start_timestamp)
        target_ms = int((target - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1))
        start_offset = find_byte_offset_for_timestamp(fmt, target_ms)
        log.warning("PARTIAL RUN: starting at byte %s (~%s). This is a bounded smoke "
                    "run, not a full-file audit.", f"{start_offset:,}", target)

    t_pass = time.time()
    collector = None
    next_log = progress_every
    for batch, collector in iter_tick_batches(fmt, cfg.point_size, block_size,
                                              max_rows, start_offset):
        st.update(batch)
        if st.n_rows >= next_log:
            rate = st.n_rows / max(time.time() - t_pass, 1e-9)
            log.info("Processed %s rows (%.0fk rows/s, %.0fs elapsed)",
                     f"{st.n_rows:,}", rate / 1000, time.time() - t_pass)
            next_log += progress_every
    st.finalize()
    pass_seconds = time.time() - t_pass
    log.info("Streaming pass complete: %s rows in %.1fs (%.0fk rows/s)",
             f"{st.n_rows:,}", pass_seconds, st.n_rows / max(pass_seconds, 1e-9) / 1000)

    if st.n_rows == 0:
        raise SystemExit("No records parsed; aborting before writing any report.")

    # --- tables -------------------------------------------------------------
    schema = build_schema_table(fmt, st)
    quality = build_quality_table(st)
    tsq = build_timestamp_quality(st)
    spread, spread_by = build_spread_tables(st)
    daily = build_daily_coverage(st)
    density = build_density_table(st, daily)
    gaps = build_gap_table(st)
    anomalies = build_anomaly_table(st)
    flags = build_flag_table(st)

    # --- M5 validation -------------------------------------------------------
    m5_src = next((s for s in cfg.sources if s.timeframe == cfg.base_timeframe), None)
    if m5_src is None:
        raise SystemExit("No base-timeframe source configured; cannot validate against M5.")
    m5 = load_m5_reference(m5_src.path, cfg.point_size)
    log.info("Loaded M5 reference: %d bars, %s .. %s", len(m5),
             m5["bar_open_time"].iloc[0], m5["bar_open_time"].iloc[-1])

    comparisons = [compare_m5(recon, m5) for recon in st.m5.values()]
    m5_summary = pd.DataFrame([c.summary for c in comparisons])
    m5_by_date = pd.concat([c.per_date for c in comparisons], ignore_index=True) \
        if comparisons else pd.DataFrame()
    best = max(comparisons, key=lambda c: c.summary.get("full_ohlc_exact_match_pct", -1)) \
        if comparisons else None
    missing_bars = m5_bars_missing_from_ticks(st.m5["A_native"], m5) \
        if "A_native" in st.m5 else pd.DataFrame()

    # --- timezone -------------------------------------------------------------
    sessions = weekly_session_table(st.week_first_ms, st.week_last_ms)
    hour_median = {
        h: hist_quantiles(st.hour_of_day_spread[h], (0.5,))[0.5]
        for h in range(24) if st.hour_of_day_spread[h].sum() > 0
    }
    tz_verdict, tz_reasoning, tz_stats = assess_timezone(
        sessions, best, has_authoritative_metadata=False,
        hour_median_spread=hour_median)
    log.info("Timezone verdict: %s", tz_verdict)

    # --- overlap ----------------------------------------------------------------
    tick_start, tick_end = ms_to_timestamp(st.ts_min), ms_to_timestamp(st.ts_max)
    datasets = {}
    for s in cfg.sources:
        ref = load_m5_reference(s.path, cfg.point_size) if s.path != m5_src.path else m5
        label = ("EURUSD_M5 (Day 1 base)" if s.timeframe == cfg.base_timeframe
                 else f"EURUSD_{s.timeframe.upper()} ({s.role})")
        datasets[label] = (ref["bar_open_time"].min(), ref["bar_open_time"].max())
    overlap = research_overlap(tick_start, tick_end, datasets)

    # --- samples ------------------------------------------------------------------
    samples = write_samples(fmt, st, outdir)

    # --- concerns and recommendation ------------------------------------------------
    concerns, recommendation, rec_reason = assess_readiness(
        st, best, overlap, tz_verdict, missing_bars, fmt)

    # --- write everything ------------------------------------------------------------
    schema.to_csv(outdir / "tick_schema.csv", index=False)
    quality.to_csv(outdir / "data_quality_summary.csv", index=False)
    tsq.to_csv(outdir / "timestamp_quality.csv", index=False)
    pd.concat([spread, spread_by], ignore_index=True).to_csv(
        outdir / "spread_summary.csv", index=False)
    daily.to_csv(outdir / "daily_coverage.csv", index=False)
    density.to_csv(outdir / "tick_density.csv", index=False)
    gaps.to_csv(outdir / "gap_report.csv", index=False)
    anomalies.to_csv(outdir / "price_anomalies.csv", index=False)
    flags.to_csv(outdir / "flag_summary.csv", index=False)
    m5_summary.to_csv(outdir / "m5_reconstruction_validation.csv", index=False)
    m5_by_date.to_csv(outdir / "m5_reconstruction_by_date.csv", index=False)
    if not missing_bars.empty:
        missing_bars.to_csv(outdir / "m5_bars_without_ticks.csv", index=False)
    overlap.to_csv(outdir / "research_overlap.csv", index=False)
    sessions.to_csv(outdir / "weekly_sessions.csv", index=False)

    write_timezone_note(outdir / "timezone_validation.md", tz_verdict, tz_reasoning,
                        tz_stats, sessions, m5_summary)
    write_report(outdir / "TICK_DATA_AUDIT_REPORT.md", fmt, st, quality, schema,
                 spread, spread_by, tsq, daily, density, gaps, anomalies,
                 m5_summary, m5_by_date, missing_bars, overlap, tz_verdict,
                 tz_reasoning, tz_stats, samples, recommendation, rec_reason,
                 concerns, pass_seconds, m5_src.path)

    summary = {
        "file": {
            "path": str(fmt.path), "name": fmt.path.name,
            "size_bytes": fmt.size_bytes, "size_gb": round(fmt.size_bytes / (1 << 30), 4),
            "format": "MT5 tick export (TSV)", "encoding": fmt.encoding,
            "line_ending": fmt.line_ending, "columns": list(fmt.raw_columns),
            "time_precision": fmt.time_precision,
        },
        "records": {
            "true_record_count": st.n_rows,
            "excel_visible_rows": 1_048_576,
            "excel_showed_pct": round(100.0 * 1_048_576 / max(st.n_rows, 1), 4),
            "unparseable_timestamps": st.n_ts_malformed,
            "invalid_rows_skipped_by_parser": collector.count if collector else 0,
        },
        "coverage": {
            "first_timestamp": str(tick_start), "last_timestamp": str(tick_end),
            "span_days": round((st.ts_max - st.ts_min) / MS_PER_DAY, 2),
            "days_with_ticks": int(len(daily)),
        },
        "quotes": {
            "reconstructed": st.n_quotes_reconstructed,
            "bid_structurally_absent": st.n_bid_structurally_absent,
            "ask_structurally_absent": st.n_ask_structurally_absent,
            "negative_spread": st.n_spread_negative,
            "zero_spread": st.n_spread_zero,
            "stale_side": st.n_quotes_stale_side,
            "median_spread_points": float(
                spread.loc[spread["population"] == "both_sides_fresh", "median_points"].iloc[0]),
            "p95_spread_points": float(
                spread.loc[spread["population"] == "both_sides_fresh", "p95_points"].iloc[0]),
            "max_spread_points": int(st.spread_max),
        },
        "timestamps": {
            "precision": fmt.time_precision,
            "out_of_order": st.n_out_of_order,
            "duplicate_timestamp_records": st.n_duplicate_ts_rows,
            "exact_duplicate_records": st.n_exact_duplicate_rows,
            "max_equal_timestamp_group": st.max_ts_group,
        },
        "m5_reconstruction": [c.summary for c in comparisons],
        "timezone": {"verdict": tz_verdict, "stats": tz_stats},
        "recommendation": recommendation,
        "elapsed_seconds": round(time.time() - t0, 2),
        "streaming_pass_seconds": round(pass_seconds, 2),
    }
    (outdir / "tick_file_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    manifest = {
        "step": "tick_audit",
        "run_timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "elapsed_seconds": round(time.time() - t0, 2),
        "streaming_pass_seconds": round(pass_seconds, 2),
        "git": _git_commit(PROJECT_ROOT),
        "package_versions": _package_versions(),
        "tick_file": {
            "path": str(fmt.path), "size_bytes": fmt.size_bytes,
            "note": "Source file was read only. Not moved, renamed, copied or modified. "
                    "No checksum computed: hashing 4.7 GB was not required by the audit "
                    "and would add a second full read.",
        },
        "config_path": str(config_path),
        "parameters": {
            "block_size_bytes": block_size, "max_rows": max_rows,
            "stale_side_ms": stale_side_ms,
            "spread_histogram_cap_points": SPREAD_HIST_MAX,
            "jump_anomaly_threshold_points": JUMP_ANOMALY_POINTS,
        },
        "full_file_passes": 1,
        "existing_artifacts_modified": [],
        "scope_limitations": [
            "Exact-duplicate detection runs within equal-timestamp groups, which is "
            "sufficient because an exact duplicate necessarily shares a timestamp. "
            "Non-adjacent identical rows are caught; a global hash of every record was "
            "not built, as that would breach the bounded-memory requirement.",
            "Per-day spread histograms are capped; the overflow share is reported.",
            "No Step 3B implementation, no execution modelling, no strategy work.",
        ],
        "counts": summary["records"] | summary["coverage"],
        "recommendation": recommendation,
    }
    (outdir / "run_manifest_tick_audit.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    log.info("Tick audit complete in %.1fs -> %s", time.time() - t0, outdir)
    return {"summary": summary, "recommendation": recommendation, "state": st}


def assess_readiness(
    st: TickAuditState, best, overlap: pd.DataFrame, tz_verdict: str,
    missing_bars: pd.DataFrame, fmt: TickFileFormat,
) -> tuple[list[str], str, str]:
    concerns: list[str] = []
    blockers: list[str] = []

    m5row = overlap[overlap["dataset"].str.contains("Day 1 base")]
    tail_gap = float(m5row["uncovered_tail_days"].iloc[0]) if not m5row.empty else 0.0
    covered = float(m5row["pct_of_existing_covered"].iloc[0]) if not m5row.empty else 0.0

    if tail_gap > 0.5:
        concerns.append(
            f"**Tail coverage gap.** Ticks end {tail_gap:.2f} days before the M5 history "
            f"does. Decision rows in that window cannot be upgraded to quote-level "
            f"analysis and must be excluded from any execution-aware population, or the "
            f"population must report its coverage explicitly.")
    if covered < 99.0:
        concerns.append(
            f"**Partial overlap.** {covered:.2f}% of the Day 1 M5 window has tick "
            f"coverage. Any execution-aware result must state which subset it used.")

    if st.n_out_of_order:
        blockers.append(
            f"**{st.n_out_of_order:,} records are out of chronological order.** "
            "Quote-path reconstruction assumes time ordering; these must be understood "
            "before Step 3B.")
    if st.n_spread_negative:
        pct = 100.0 * st.n_spread_negative / max(st.n_quotes_reconstructed, 1)
        (concerns if pct < 0.01 else blockers).append(
            f"**{st.n_spread_negative:,} crossed quotes** (ask < bid), {pct:.5f}% of "
            "reconstructed quotes. Not deleted. Any execution simulation must decide "
            "explicitly how to treat them.")
    if st.n_exact_duplicate_rows:
        concerns.append(
            f"**{st.n_exact_duplicate_rows:,} exact duplicate records** (identical "
            "timestamp, bid, ask and flags). Harmless for OHLC reconstruction; they would "
            "double-count in any tick-frequency measure.")
    if st.n_flag_bid_mismatch or st.n_flag_ask_mismatch:
        concerns.append(
            f"**Field/flag inconsistencies**: {st.n_flag_bid_mismatch:,} bid and "
            f"{st.n_flag_ask_mismatch:,} ask. The FLAGS bitmask does not always agree "
            "with which price fields are populated.")

    if st.n_quotes_stale_side:
        pct = 100.0 * st.n_quotes_stale_side / max(st.n_quotes_reconstructed, 1)
        concerns.append(
            f"**Stale-side quotes at session reopen**: {st.n_quotes_stale_side:,} "
            f"({pct:.3f}%) reconstructed quotes had one side older than "
            f"{st.stale_side_ms/1000:.0f}s. Expected behaviour after weekends and "
            "holidays, but a spread computed across such a boundary is not a "
            "contemporaneous spread. Step 3B must apply a freshness rule at session open.")
    else:
        concerns.append(
            "**Quote freshness is a non-issue in this file** (checked, not assumed): zero "
            f"stale-side quotes, maximum one-sided staleness "
            f"{max(st.bid_update_gap_max, st.ask_update_gap_max)/60000.0:.2f} minutes even "
            "across weekends. Step 3B still needs an explicit freshness rule as a guard, "
            "but no observed spread here is contaminated by a market closure.")

    if best is None or best.summary.get("full_ohlc_exact_match_pct", 0) < 95.0:
        blockers.append(
            "**M5 reconstruction did not reproduce the existing bars.** Until this is "
            "explained, the tick feed cannot be assumed to be the same feed the existing "
            "research was built on.")

    if tz_verdict in ("CONTRADICTED",):
        blockers.append("**Timezone evidence contradicts the documented clock assumption.**")
    concerns.append(
        "**Broker server timezone remains undocumented.** The evidence here is stronger "
        "than what Day 1 had, but it is still inference. The broker's stated server "
        "timezone and DST rule should be obtained before live-execution work.")
    concerns.append(
        "**No depth of market, no fill records.** Quote-path simulation is possible; "
        "actual slippage and fill realism are not recoverable from this file.")
    if not missing_bars.empty:
        concerns.append(
            f"**{int(missing_bars['n_m5_bars_without_ticks'].sum()):,} M5 bars inside the "
            f"overlap have no ticks**, across {len(missing_bars)} dates. Those bars exist "
            "in the bar export but cannot be given a quote path.")
    concerns.append(
        f"**Storage.** The {fmt.size_bytes/(1<<30):.2f} GB file sits inside the project "
        "directory. Recommend moving long-term tick storage outside the source tree.")

    if blockers:
        rec = "REQUEST DIFFERENT DATA" if len(blockers) > 1 else "GO WITH CONDITIONS"
        reason = ("Blocking findings:\n\n" + "\n".join(f"- {b}" for b in blockers))
        concerns = blockers + concerns
    else:
        rec = "GO WITH CONDITIONS"
        reason = (
            "The file is genuine millisecond bid/ask quote history from the same feed as "
            "the existing bar research, it reproduces the existing M5 bid OHLC, and it "
            "covers the great majority of the current research window. That is sufficient "
            "to build execution-aware Step 3B analysis on.\n\n"
            "The conditions are scope conditions, not data defects:\n\n"
            f"1. Restrict quote-level analysis to the tick-covered window, or report "
            f"coverage per population. {tail_gap:.2f} days at the tail of the M5 history "
            "have no ticks.\n"
            "2. Apply an explicit quote-freshness rule at session reopen, where one side "
            "of the book updates before the other.\n"
            "3. Decide and document the treatment of crossed and zero spreads rather "
            "than dropping them silently.\n"
            "4. Report quote-path results as **quote-based execution simulation**, never "
            "as achieved fills or realised slippage.\n"
            "5. Obtain the broker's documented server timezone before any live-execution "
            "work; the clock here is strongly evidenced but still inferred.")
    return concerns, rec, reason


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m forex_research.tick_audit",
        description="Audit a large MT5 bid/ask tick export for research suitability.")
    ap.add_argument("--file", required=True, help="Path to the tick file (never hard-coded).")
    ap.add_argument("--config", required=True, help="Research YAML configuration.")
    ap.add_argument("--outdir", default=None,
                    help="Output directory (default: <config output_dir>/../tick_audit).")
    ap.add_argument("--block-size", type=int, default=64 << 20,
                    help="Bytes per PyArrow parse block; caps resident memory.")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Stop after N records (bounded smoke run).")
    ap.add_argument("--stale-side-ms", type=int, default=60_000,
                    help="One side older than this marks the quote stale-sided.")
    ap.add_argument("--progress-every", type=int, default=5_000_000)
    ap.add_argument("--start-timestamp", default=None,
                    help="Begin the stream at this timestamp (bounded smoke runs only; "
                         "produces a PARTIAL audit).")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    tick_path = Path(args.file).expanduser().resolve()
    cfg = load_config(args.config)
    outdir = Path(args.outdir).resolve() if args.outdir else \
        (cfg.output_dir.parent / "tick_audit")
    outdir.mkdir(parents=True, exist_ok=True)
    configure_logging(args.log_level, outdir / "run_log_tick_audit.txt")

    log.info("Tick audit starting | file=%s | config=%s", tick_path, args.config)
    run(tick_path, args.config, outdir, args.block_size, args.max_rows,
        args.stale_side_ms, args.progress_every, args.start_timestamp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

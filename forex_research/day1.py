from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import validation as val
from .config import ResearchConfig, load_config, timeframe_minutes
from .data_loader import load_all
from .data_quality import audit_all
from .event_quality import (
    annotate_source_quality,
    build_trading_calendar,
    propagate_quality_to_decisions,
    source_quality_report,
)
from .feature_dictionary import build_dictionary
from .logging_utils import configure_logging, get_logger
from .manifest import build_manifest, write_manifest
from .outcomes import build_outcomes, outcome_summary
from .pipeline import iter_feature_chunks, prepare_dataset
from .reporting import write_day1_report, write_validation_addendum
from .timing_audit import build_timing_audit, lag_distribution, write_timing_note

log = get_logger("day1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write_chunked_parquet(chunks, path: Path, compression: str = "snappy") -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    n_rows = 0
    try:
        for chunk in chunks:
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression=compression)
            writer.write_table(table)
            n_rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()
    return n_rows, path.stat().st_size if path.exists() else 0


def _human(n_bytes: int) -> str:
    x = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024 or unit == "GB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} GB"


def _artifact_size(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_file():
        return path.stat().st_size, 1
    total = 0
    count = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
            count += 1
    return total, count


def run(cfg: ResearchConfig, causality_rows: int, skip_full: bool = False) -> dict[str, Any]:
    t0 = time.time()
    timings: dict[str, float] = {}
    warnings: list[str] = []
    blockers: list[str] = []
    outdir = cfg.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []

    def record(name: str, path: Path, description: str, rows: int | None = None) -> None:
        total, n_files = _artifact_size(path)
        files.append({
            "file": name,
            "rows": f"{rows:,}" if rows is not None else "",
            "size": _human(total) if total else "missing",
            "bytes": total,
            "n_files": n_files,
            "description": description,
        })

    # ---------------------------------------------------------------- load
    step = time.time()
    loaded = load_all(cfg)
    base_series = loaded[cfg.base_timeframe]
    base = base_series.frame
    timings["load"] = time.time() - step

    # ------------------------------------------------------- data quality
    step = time.time()
    dq = audit_all(loaded, cfg)
    dq_path = outdir / "data_quality_report.csv"
    dq.to_csv(dq_path, index=False)
    record("data_quality_report.csv", dq_path,
           "Every structural check on the raw sources, including the timestamp-semantics proof",
           len(dq))
    timings["data_quality"] = time.time() - step

    dq_fail = dq[dq["status"] == "fail"]
    if not dq_fail.empty:
        raise val.ValidationError(
            "Data-quality failures block the run:\n"
            + dq_fail[["source", "check", "value", "detail"]].to_string(index=False)
        )

    # Standing blockers: things that require the client or the data provider to act.
    # These do not stop the Day 1 dataset from being built, but they bound what can
    # be concluded from it, so they are recorded explicitly rather than buried.
    blockers.append(
        "BID/ASK AND TICK DATA MISSING. The supplied files are MT5 bar exports with a single "
        "(bid-side) OHLC series plus a per-bar spread column in points. Without true bid/ask "
        "tick history: (a) execution cost, slippage and fill realism cannot be modelled; "
        "(b) intrabar path order cannot be resolved, so cases where price reaches both an "
        "upward and a downward threshold inside the same 5-minute bar are flagged ambiguous "
        "rather than decided. NEEDED FROM CLIENT: EUR/USD tick history with bid and ask."
    )
    blockers.append(
        "BROKER SERVER TIMEZONE NOT DOCUMENTED. The file clock was inferred as New York + 7 "
        "hours (UTC+3 under US DST, UTC+2 otherwise) from weekly session boundaries across "
        "three DST regimes. The evidence is strong and reproducible, but it is an inference. "
        "NEEDED FROM CLIENT: the broker's documented server timezone and DST rule, so 4h and "
        "daily bar boundaries can be confirmed rather than inferred."
    )
    warnings.append(
        "No bid/ask or tick data is available. The MT5 export carries a single bid-side OHLC "
        "series plus a per-bar spread column. Execution-cost and slippage analysis is therefore "
        "PENDING; Day 1 outcomes are market-movement labels only."
    )
    m1_first = loaded.get("1m").frame["bar_open_time"].min() if "1m" in loaded else None
    if m1_first is not None:
        warnings.append(
            f"M1 coverage starts {m1_first} while M5 starts {base['bar_open_time'].min()}. "
            "Intrabar path resolution finer than 5m is only possible on the M1-covered subset, "
            "so Day 1 outcomes are computed on 5m bars for every row to keep labels comparable."
        )

    # ------------------------------------------------- features & alignment
    step = time.time()
    ds = prepare_dataset(base, cfg)
    timings["prepare_features"] = time.time() - step

    coverage_rows = []
    for tf, ts in ds.timeframes.items():
        idx = ds.alignment.indices[tf]
        coverage_rows.append({
            "timeframe": tf,
            "minutes_per_bar": ts.minutes,
            "n_bars": len(ts.frame),
            "first_bar_open_native": str(ts.frame["bar_open_time"].iloc[0]),
            "last_bar_close_native": str(ts.frame["bar_close_time"].iloc[-1]),
            "warmup_bars_required": cfg.warmup_bars(),
            "n_bars_after_warmup": max(0, len(ts.frame) - cfg.warmup_bars()),
            "warmup_satisfied": len(ts.frame) > cfg.warmup_bars(),
            "n_decision_rows_with_closed_bar": int((idx >= 0).sum()),
            "n_decision_rows_warmed_up": int((idx >= cfg.warmup_bars()).sum()),
            "n_incomplete_tail_bins_dropped": ts.n_incomplete_dropped,
            "n_bins_with_missing_base_bars": ts.n_bins_with_missing_base_bars,
        })
        if len(ts.frame) <= cfg.warmup_bars():
            blockers.append(
                f"Timeframe {tf} has only {len(ts.frame)} bars but warm-up needs "
                f"{cfg.warmup_bars()}. It cannot contribute any valid feature row with the "
                f"configured period list (longest period {cfg.max_indicator_period()})."
            )

    if not cfg.include_daily:
        base_days = (base["bar_close_time"].max() - base["bar_open_time"].min()).days
        est_daily_bars = int(base_days * 5 / 7)
        blockers.append(
            f"DAILY TIMEFRAME DISABLED - INSUFFICIENT HISTORY. The authoritative period list "
            f"runs to {cfg.max_indicator_period()}, so a 1D series needs {cfg.warmup_bars()} "
            f"daily bars of warm-up. The supplied history spans about {base_days} calendar "
            f"days (~{est_daily_bars} trading days), which would leave almost no usable rows "
            f"and would additionally truncate every other timeframe to that same short "
            f"window. NEEDED FROM CLIENT: roughly 3+ years of EUR/USD history to include 1D "
            f"at these periods. Set include_daily: true in the config once available."
        )

    warmup_lost = int(len(base) - ds.alignment.valid.sum())
    blockers.append(
        f"WARM-UP COST OF THE 300-PERIOD LIST. Requiring {cfg.warmup_bars()} bars on every "
        f"timeframe consumes {warmup_lost:,} of {len(base):,} decision rows "
        f"({warmup_lost / len(base) * 100:.1f}%); the 4h timeframe is the binding constraint. "
        f"Valid rows start {ds.alignment.decision_time.iloc[int(np.flatnonzero(ds.alignment.valid)[0])]} "
        f"rather than at the start of the data. More history would recover these rows; this "
        f"is a consequence of the client's period list, not a defect."
    )
    coverage = pd.DataFrame(coverage_rows)
    cov_path = outdir / "timeframe_coverage.csv"
    coverage.to_csv(cov_path, index=False)
    record("timeframe_coverage.csv", cov_path, "Bars and usable rows per timeframe", len(coverage))

    # -------------------------------------------------- source-bar quality
    step = time.time()
    calendar = build_trading_calendar(base, cfg)
    annotated = annotate_source_quality(ds.timeframes, calendar, cfg)
    sq = source_quality_report(annotated, calendar)
    sq_path = outdir / "source_bar_quality.csv"
    sq.to_csv(sq_path, index=False)
    record("source_bar_quality.csv", sq_path,
           "Per-timeframe source-bar completeness, separating market closures from data gaps",
           len(sq))

    quality = propagate_quality_to_decisions(annotated, ds.alignment.indices, ds.n_rows)
    n_bad = int((~quality["source_quality_ok"]).sum())
    if n_bad:
        warnings.append(
            f"{n_bad:,} decision rows have at least one timeframe whose aligned source bar "
            "was built from incomplete base data (a feed gap during open market, not a "
            "market closure). They are flagged source_quality_ok=False in "
            "alignment_map.parquet so confirmatory analysis can exclude them."
        )
    timings["source_quality"] = time.time() - step

    # -------------------------------------------------- indicator timing audit
    step = time.time()
    timing = build_timing_audit(ds.timeframes, ds.alignment.decision_time, cfg)
    lagdist = lag_distribution(ds.timeframes, ds.alignment.decision_time, cfg)
    ta_path = outdir / "indicator_timing_audit.csv"
    pd.concat([timing.assign(record_kind="example"),
               lagdist.assign(record_kind="lag_distribution")],
              ignore_index=True).to_csv(ta_path, index=False)
    record("indicator_timing_audit.csv", ta_path,
           "Which raw candle feeds each indicator, per timeframe and source mode",
           len(timing) + len(lagdist))
    write_timing_note(outdir / "INDICATOR_TIMING_NOTE.md", cfg, timing, lagdist)
    record("INDICATOR_TIMING_NOTE.md", outdir / "INDICATOR_TIMING_NOTE.md",
           "Analysis and decision record for indicator timing semantics")

    if timing.empty or not bool(
        timing["source_closed_at_or_before_decision"].all()
    ):
        raise val.ValidationError(
            "Indicator timing audit found a contributing candle closing after the decision "
            "time. This is a lookahead and the run must not continue."
        )
    timings["timing_audit"] = time.time() - step

    # ------------------------------------------------------------ validation
    step = time.time()
    results: list[val.ValidationResult] = []
    results += val.check_timestamp_integrity(ds.timeframes, cfg)
    results += val.check_resampling(base, ds.timeframes, cfg)
    align_table, align_results = val.check_alignment(ds.alignment, cfg)
    results += align_results
    results += val.check_dictionary(ds.specs, ds.feature_columns)
    results += val.check_missing_data_policy(ds.alignment.valid, cfg)

    align_path = outdir / "alignment_validation.csv"
    align_table.to_csv(align_path, index=False)
    record("alignment_validation.csv", align_path,
           "Per-timeframe proof that source_bar_close_time <= decision_time", len(align_table))
    timings["validation_structural"] = time.time() - step

    # ------------------------------------------------------------- outcomes
    step = time.time()
    outcomes, outcome_specs = build_outcomes(base, cfg)
    results += val.check_label_separation(ds.feature_columns, list(outcomes.columns))
    timings["outcomes"] = time.time() - step

    incomplete = {
        h: int((outcomes.loc[ds.alignment.valid, f"horizon_complete_{h}m"] == 0).sum())
        for h in cfg.forward_horizons_minutes
    }
    if any(incomplete.values()):
        warnings.append(
            "Rows with an incomplete forward window at the end of the dataset: "
            + ", ".join(f"{h}m={n}" for h, n in incomplete.items())
            + ". These are flagged by horizon_complete_*m and must be excluded from any "
              "outcome analysis rather than treated as no-move rows."
        )

    empty = {
        h: int(((outcomes.loc[ds.alignment.valid, f"horizon_complete_{h}m"] == 1)
                & (outcomes.loc[ds.alignment.valid, f"bars_in_window_{h}m"] == 0)).sum())
        for h in cfg.forward_horizons_minutes
    }
    if any(empty.values()):
        warnings.append(
            "Rows where the market was closed for the entire forward window (holiday or "
            "weekend), so horizon_complete_*m is 1 but no bar was observed and every "
            "outcome is NaN: "
            + ", ".join(f"{h}m={n}" for h, n in empty.items())
            + ". Filter on bars_in_window_*m > 0 as well as horizon_complete_*m."
        )

    one_sided = {
        h: int((outcomes.loc[ds.alignment.valid, f"min_future_delta_pips_{h}m"] > 0).sum())
        for h in cfg.forward_horizons_minutes
    }
    if any(one_sided.values()):
        warnings.append(
            "Rows where price never traded back to the decision-bar close inside the window "
            "(one-sided move): "
            + ", ".join(f"{h}m={n}" for h, n in one_sided.items())
            + ". The signed max/min_future_delta_pips columns keep that information; the "
              "conventional long/short MFE and MAE columns are floored at zero and are never "
              "negative."
        )

    # ------------------------------------------------------- causality check
    step = time.time()
    n_causal = min(causality_rows, len(base))
    causal_results, causal_detail = val.run_causality_check(base, cfg, n_causal)
    results += causal_results
    causal_path = outdir / "causality_check.csv"
    causal_detail.to_csv(causal_path, index=False)
    record("causality_check.csv", causal_path,
           "Future-mutation test: historical feature rows unchanged", len(causal_detail))
    timings["causality_check"] = time.time() - step

    validation_df = val.to_frame(results)
    val_path = outdir / "validation_report.csv"
    validation_df.to_csv(val_path, index=False)
    record("validation_report.csv", val_path, "All automated quality-control results",
           len(validation_df))

    # ------------------------------------------------------------ artefacts
    step = time.time()
    alignment_map = pd.DataFrame({"decision_time": ds.alignment.decision_time.to_numpy()})
    alignment_map.insert(0, "symbol", cfg.symbol)
    for tf in ds.alignment.indices:
        alignment_map[f"{tf}__source_bar_index"] = ds.alignment.indices[tf]
        alignment_map[f"{tf}__source_bar_open_time"] = ds.alignment.source_open[tf]
        alignment_map[f"{tf}__source_bar_close_time"] = ds.alignment.source_close[tf]
        alignment_map[f"{tf}__source_bar_age_minutes"] = (
            alignment_map["decision_time"].to_numpy() - ds.alignment.source_close[tf]
        ) / np.timedelta64(1, "m")
    alignment_map["feature_row_valid"] = ds.alignment.valid
    for col in quality.columns:
        alignment_map[col] = quality[col].to_numpy()
    am_path = outdir / "alignment_map.parquet"
    alignment_map.to_parquet(am_path, index=False, compression="snappy")
    record("alignment_map.parquet", am_path,
           "decision_time -> source bar per timeframe; join key for the long-form panel",
           len(alignment_map))

    dictionary = build_dictionary(ds.specs)
    fd_path = outdir / "feature_dictionary.csv"
    dictionary.to_csv(fd_path, index=False)
    record("feature_dictionary.csv", fd_path,
           "One row per feature; uses_future_data is False for every entry", len(dictionary))

    outcome_dict = build_dictionary(outcome_specs)
    od_path = outdir / "outcome_dictionary.csv"
    outcome_dict.to_csv(od_path, index=False)
    record("outcome_dictionary.csv", od_path,
           "One row per outcome column; uses_future_data is True for every entry",
           len(outcome_dict))

    osum = outcome_summary(outcomes, cfg, ds.alignment.valid)
    os_path = outdir / "outcome_summary.csv"
    osum.to_csv(os_path, index=False)
    record("outcome_summary.csv", os_path, "Counts per forward horizon and expansion threshold",
           len(osum))

    # long-form indicator panel, one parquet per timeframe
    panel_dir = outdir / "indicator_panel_long"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_rows = 0
    for tf, tff in ds.tf_features.items():
        p = panel_dir / f"timeframe={tf}.parquet"
        tff.long_panel.to_parquet(p, index=False, compression="snappy")
        panel_rows += len(tff.long_panel)
        tff.long_panel = pd.DataFrame()  # release memory
    record("indicator_panel_long/", panel_dir,
           "Normalised long-form indicator state at each timeframe's native resolution",
           panel_rows)
    timings["artefacts"] = time.time() - step

    # ------------------------------------------------------ feature matrix
    step = time.time()
    rng = np.random.default_rng(cfg.random_seed)
    valid_idx = np.flatnonzero(ds.alignment.valid)
    sample_idx = np.sort(rng.choice(valid_idx, size=min(cfg.sample_rows, len(valid_idx)),
                                    replace=False)) if len(valid_idx) else np.array([], dtype=int)
    sample_set = set(sample_idx.tolist())

    sample_parts: list[pd.DataFrame] = []
    written_rows = 0

    def chunk_stream():
        nonlocal written_rows
        offset = 0
        for chunk in iter_feature_chunks(ds, cfg):
            local = [i - offset for i in sample_set if offset <= i < offset + len(chunk)]
            if local:
                sample_parts.append(chunk.iloc[sorted(local)].copy())
            offset += len(chunk)
            written_rows += len(chunk)
            if written_rows % (cfg.chunk_rows * 10) == 0:
                log.info("Feature matrix: %d/%d rows written", written_rows, ds.n_rows)
            yield chunk

    if cfg.write_full_datasets and not skip_full:
        feat_path = outdir / "features.parquet"
        rows, size = _write_chunked_parquet(chunk_stream(), feat_path)
        record("features.parquet", feat_path,
               f"Complete point-in-time feature matrix ({ds.n_features:,} features)", rows)
        log.info("features.parquet: %d rows x %d features (%s)", rows, ds.n_features, _human(size))
    else:
        for _ in chunk_stream():
            pass
        warnings.append("write_full_datasets is disabled; only the feature sample was written.")

    feature_sample = (pd.concat(sample_parts, ignore_index=True)
                      if sample_parts else pd.DataFrame(columns=["symbol", "decision_time"]))
    fs_path = outdir / "feature_sample.parquet"
    feature_sample.to_parquet(fs_path, index=False, compression="snappy")
    record("feature_sample.parquet", fs_path,
           f"Random sample of {len(feature_sample):,} valid feature rows for quick inspection",
           len(feature_sample))
    timings["feature_matrix"] = time.time() - step

    step = time.time()
    if cfg.write_full_datasets and not skip_full:
        oc_path = outdir / "outcomes.parquet"
        outcomes.to_parquet(oc_path, index=False, compression="snappy")
        record("outcomes.parquet", oc_path,
               "Complete future-outcome table, joined to features on symbol + decision_time",
               len(outcomes))

    osample = outcomes.iloc[sample_idx] if len(sample_idx) else outcomes.iloc[:0]
    osp_path = outdir / "outcomes_sample.parquet"
    osample.to_parquet(osp_path, index=False, compression="snappy")
    record("outcomes_sample.parquet", osp_path,
           "Outcome rows matching feature_sample.parquet", len(osample))
    timings["outcomes_write"] = time.time() - step

    # --------------------------------------------------------------- report
    counts = {
        "n_decision_rows": int(ds.n_rows),
        "n_valid_rows": int(ds.alignment.valid.sum()),
        "n_features": int(ds.n_features),
        "n_outcome_columns": int(len(outcomes.columns) - 2),
        "n_timeframes": len(cfg.active_timeframes),
        "n_indicators_per_timeframe": len(cfg.indicator_specs()),
        "n_long_panel_rows": int(panel_rows),
        "first_valid_decision_time": str(
            ds.alignment.decision_time.iloc[int(valid_idx[0])]) if len(valid_idx) else "",
        "last_valid_decision_time": str(
            ds.alignment.decision_time.iloc[int(valid_idx[-1])]) if len(valid_idx) else "",
    }
    outputs = {"output_dir": str(outdir), "files": files}

    write_day1_report(
        outdir / "DAY1_REPORT.md", cfg, dq, coverage, align_table, validation_df,
        ds.specs, outcome_specs, osum, counts, warnings, blockers, outputs,
    )
    record("DAY1_REPORT.md", outdir / "DAY1_REPORT.md", "Human-readable Day 1 report")

    write_validation_addendum(
        outdir / "DAY1_VALIDATION_ADDENDUM.md", cfg, validation_df, lagdist, sq,
        counts, n_bad, files,
    )
    record("DAY1_VALIDATION_ADDENDUM.md", outdir / "DAY1_VALIDATION_ADDENDUM.md",
           "Audit record of the post-Day-1 review findings and fixes")

    timings["total"] = time.time() - t0
    manifest = build_manifest(
        cfg=cfg,
        sources=[{
            "path": str(s.path), "timeframe": s.timeframe, "role": s.role,
            "sha256": s.sha256, "rows": len(s.frame),
            "first_bar_open_native": str(s.frame["bar_open_time"].iloc[0]),
            "last_bar_close_native": str(s.frame["bar_close_time"].iloc[-1]),
            "first_bar_open_utc": str(s.frame["bar_open_time_utc"].iloc[0]),
            "last_bar_close_utc": str(s.frame["bar_close_time_utc"].iloc[-1]),
        } for s in loaded.values()],
        date_range={
            "base_first_bar_open_native": str(base["bar_open_time"].iloc[0]),
            "base_last_bar_close_native": str(base["bar_close_time"].iloc[-1]),
            "first_valid_decision_time": counts["first_valid_decision_time"],
            "last_valid_decision_time": counts["last_valid_decision_time"],
        },
        counts=counts, warnings=warnings, blockers=blockers, outputs=outputs,
        timings={k: round(v, 2) for k, v in timings.items()}, project_root=PROJECT_ROOT,
    )
    write_manifest(manifest, outdir / "run_manifest.json")
    record("run_manifest.json", outdir / "run_manifest.json",
           "Config, source hashes, versions, counts, warnings")

    log.info("Day 1 complete in %.1fs. Output: %s", timings["total"], outdir)
    log.info("Features: %d x %d valid rows | Outcomes: %d columns",
             counts["n_features"], counts["n_valid_rows"], counts["n_outcome_columns"])
    return {"counts": counts, "warnings": warnings, "blockers": blockers, "timings": timings}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m forex_research.day1",
        description="Build the Day 1 point-in-time multi-timeframe EUR/USD research dataset.",
    )
    ap.add_argument("--config", required=True, help="Path to the research YAML configuration.")
    ap.add_argument("--outdir", default=None, help="Override the configured output directory.")
    ap.add_argument("--causality-rows", type=int, default=30000,
                    help="Base bars used for the end-to-end future-mutation causality check.")
    ap.add_argument("--skip-full-datasets", action="store_true",
                    help="Write only samples and reports (faster smoke run).")
    ap.add_argument("--log-level", default=None)
    args = ap.parse_args(argv)

    overrides: dict[str, Any] = {}
    if args.outdir:
        overrides["output_dir"] = args.outdir
    if args.log_level:
        overrides["log_level"] = args.log_level

    cfg = load_config(args.config, overrides or None)
    configure_logging(cfg.log_level, cfg.output_dir / "run_log.txt")
    log.info("Configuration: %s", args.config)
    log.info("Timeframes=%s | EMA periods=%s | SMA periods=%s | basis=%s",
             cfg.active_timeframes, cfg.ema_periods, cfg.sma_periods, cfg.price_basis)

    run(cfg, causality_rows=args.causality_rows, skip_full=args.skip_full_datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

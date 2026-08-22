from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .benchmark import (
    build_states,
    cell_key,
    control_row,
    evaluate_state,
    state_counts,
    unconditional_row,
)
from .config import ResearchConfig, load_config, timeframe_minutes
from .control_sampling import compression_bins, volatility_bins
from .crossover_analysis import (
    crossovers_for_timeframe,
    summarise,
    summarise_by_session,
)
from .data_loader import load_all
from .event_definitions import load_step2_config, session_of
from .logging_utils import configure_logging, get_logger
from .manifest import _git_commit, _package_versions
from .pipeline import prepare_dataset
from .reporting_day2 import (
    write_benchmark_plan,
    write_benchmark_report,
    write_checkpoint,
    write_control_design_addendum,
)
from .risk_set_controls import (
    BASE_MATCH_KEYS,
    balance_report,
    build_risk_set_pool,
    sample_random_context_controls,
    sample_risk_set_controls,
)

log = get_logger("day2_checkpoint")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_day2_config(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg2 = raw.get("day2_checkpoint")
    if not cfg2:
        raise SystemExit("Config is missing the 'day2_checkpoint' section.")
    cfg2 = dict(cfg2)
    cfg2["output_dir"] = (Path(path).parent / cfg2.get(
        "output_dir", "../output/day2_checkpoint")).resolve()
    return cfg2


def build_analysis_frame(
    cfg: ResearchConfig, d2: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, int]]:
    d1 = cfg.output_dir
    needed = ["features.parquet", "outcomes.parquet", "alignment_map.parquet"]
    for n in needed:
        if not (d1 / n).exists():
            raise SystemExit(
                f"Missing Day 1 artefact {d1/n}. Run:\n"
                "    python -m forex_research.day1 --config configs/forex_day1.yaml")

    state_cols = sorted({d2["compression_column"], d2["crossover_direction_column"],
                         d2["htf_alignment_column"], "5m__atr_pips",
                         f"{d2['state_timeframe']}__compression_percentile_12"})
    feats = pd.read_parquet(d1 / "features.parquet",
                            columns=["decision_time", "feature_row_valid", *state_cols])
    align = pd.read_parquet(d1 / "alignment_map.parquet",
                            columns=["decision_time", "source_quality_ok"])
    outcomes = pd.read_parquet(d1 / "outcomes.parquet")

    df = feats.merge(align, on="decision_time").merge(outcomes, on="decision_time")

    h = int(d2["primary_horizon_minutes"])
    counts = {"all_rows": len(df)}
    elig = (df["feature_row_valid"].astype(bool)
            & df["source_quality_ok"].astype(bool)
            & (df[f"horizon_complete_{h}m"] == 1)
            & (df[f"bars_in_window_{h}m"] > 0))
    counts["excluded_invalid_feature_row"] = int((~df["feature_row_valid"].astype(bool)).sum())
    counts["excluded_source_quality"] = int((~df["source_quality_ok"].astype(bool)).sum())
    counts["excluded_incomplete_horizon"] = int((df[f"horizon_complete_{h}m"] != 1).sum())
    counts["excluded_market_closed"] = int((df[f"bars_in_window_{h}m"] <= 0).sum())
    counts["eligible_rows"] = int(elig.sum())

    df = df.loc[elig].reset_index(drop=True)

    dt = pd.DatetimeIndex(df["decision_time"])
    df["session"] = [session_of(x) for x in dt.hour]
    df["hour_bucket"] = (dt.hour // int(d2["hour_bucket_hours"])) * int(d2["hour_bucket_hours"])
    df["day_of_week"] = dt.dayofweek
    df["volatility_bin"] = volatility_bins(df["5m__atr_pips"], int(d2["volatility_bins"]))
    df["compression_bin"] = compression_bins(
        df[f"{d2['state_timeframe']}__compression_percentile_12"])

    log.info("Analysis frame: %d eligible rows of %d", len(df), counts["all_rows"])
    return df, counts


def run(cfg: ResearchConfig, d2: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    outdir = Path(d2["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- PLAN
    # Written BEFORE any result is computed, so the analysis choices are on
    # record rather than settled after seeing the numbers.
    write_benchmark_plan(outdir / "DAY2_BENCHMARK_PLAN.md", cfg, d2)
    log.info("Benchmark plan frozen before computing results.")

    frame, elig_counts = build_analysis_frame(cfg, d2)

    # -------------------------------------------------------- PART A: crossovers
    step = time.time()
    base = load_all(cfg)[cfg.base_timeframe].frame
    ds = prepare_dataset(base, cfg)
    families = tuple(d2["crossover_indicator_families"])
    tf_minutes = {tf: timeframe_minutes(tf) for tf in d2["crossover_timeframes"]}

    details = []
    for tf in d2["crossover_timeframes"]:
        tff = ds.tf_features[tf]
        det = crossovers_for_timeframe(
            tf, tff.indicator_values, tff.bar_close_time, cfg, families)
        if not det.empty:
            details.append(det)
    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()

    eligible_times = pd.DatetimeIndex(frame["decision_time"])
    counts_x, pair_counts = summarise(
        detail, tf_minutes, int(d2["crossover_episode_gap_bars"]), eligible_times)
    sess = pd.Series(frame["session"].to_numpy(),
                     index=pd.DatetimeIndex(frame["decision_time"]))
    by_session = summarise_by_session(
        detail[detail["bar_close_time"].isin(eligible_times)] if not detail.empty
        else detail, sess)

    detail.to_parquet(outdir / "crossover_event_detail.parquet",
                      index=False, compression="snappy")
    counts_x.to_csv(outdir / "crossover_counts.csv", index=False)
    pair_counts.to_csv(outdir / "crossover_pair_counts.csv", index=False)
    if not by_session.empty:
        by_session.to_csv(outdir / "crossover_counts_by_session.csv", index=False)
    log.info("Crossovers: %d pair-level rows in %.1fs", len(detail), time.time() - step)

    # ------------------------------------------------- PART B: risk-set controls
    step = time.time()
    s2cfg = load_step2_config_path(cfg)
    events = pd.read_parquet(s2cfg / "events.parquet")
    h = int(d2["primary_horizon_minutes"])
    thr = float(d2["primary_threshold_pips"])
    primary_stratum = f"up_{thr:g}p_{h}m"

    ev_ctx = events.merge(
        frame[["decision_time", "session", "hour_bucket", "day_of_week",
               "volatility_bin", "compression_bin"]],
        left_on="canonical_decision_time", right_on="decision_time", how="inner",
        suffixes=("", "_ctx"))
    ev_primary = ev_ctx[ev_ctx["stratum"] == primary_stratum]

    outcome_flag = frame[f"expansion_up_{thr:g}p_{h}m"]
    pool = build_risk_set_pool(
        frame, pd.Series(True, index=frame.index),
        ev_primary["canonical_decision_time"].to_numpy(), outcome_flag)

    rs_controls, rs_report = sample_risk_set_controls(
        ev_primary, pool.frame, int(d2["risk_set_controls_per_event"]),
        int(d2["random_seed"]), "risk_set_matched", BASE_MATCH_KEYS)

    rs_comp_controls, rs_comp_report = sample_risk_set_controls(
        ev_primary, pool.frame, int(d2["risk_set_controls_per_event"]),
        int(d2["random_seed"]) + 31, "risk_set_compression_matched",
        BASE_MATCH_KEYS + ("compression_bin",))

    rnd_ctx = sample_random_context_controls(
        ev_primary, pool.frame, int(d2["random_context_controls"]),
        int(d2["random_seed"]) + 77)

    all_rs = pd.concat([c for c in (rs_controls, rs_comp_controls) if not c.empty],
                       ignore_index=True) if (not rs_controls.empty or
                                              not rs_comp_controls.empty) else pd.DataFrame()
    if not all_rs.empty:
        all_rs.to_parquet(outdir / "risk_set_controls.parquet",
                          index=False, compression="snappy")
    if not rnd_ctx.empty:
        rnd_ctx.to_parquet(outdir / "random_context_controls.parquet",
                           index=False, compression="snappy")

    match_rep = pd.concat(
        [r.assign(control_type=t) for r, t in
         ((rs_report, "risk_set_matched"), (rs_comp_report, "risk_set_compression_matched"))
         if not r.empty], ignore_index=True) if not rs_report.empty else pd.DataFrame()
    if not match_rep.empty:
        match_rep.to_csv(outdir / "risk_set_matching_report.csv", index=False)

    # balance, globally and within the primary stratum's sessions
    ecols = {"session": "session", "hour_bucket": "hour_bucket",
             "volatility_bin": "volatility_bin", "compression_bin": "compression_bin",
             "day_of_week": "day_of_week"}
    ccols = {"session": "matching_session", "hour_bucket": "matching_hour_bucket",
             "volatility_bin": "matching_volatility_bin",
             "compression_bin": "matching_compression_bin",
             "day_of_week": "matching_hour_bucket"}
    bal_parts = []
    for ctl, name in ((rs_controls, "risk_set_matched"),
                      (rs_comp_controls, "risk_set_compression_matched"),
                      (rnd_ctx, "random_context_matched")):
        if ctl.empty:
            continue
        bal_parts.append(balance_report(
            ev_primary, ctl, name,
            ("session", "hour_bucket", "volatility_bin", "compression_bin"),
            ecols, ccols, stratum="ALL"))
        for s in sorted(ev_primary["session"].dropna().unique()):
            e_s = ev_primary[ev_primary["session"] == s]
            c_s = ctl[ctl["matching_session"] == s]
            if len(e_s) >= 20 and len(c_s) >= 20:
                bal_parts.append(balance_report(
                    e_s, c_s, name, ("hour_bucket", "volatility_bin", "compression_bin"),
                    ecols, ccols, stratum=f"session={s}"))
    balance = pd.concat(bal_parts, ignore_index=True) if bal_parts else pd.DataFrame()
    if not balance.empty:
        balance.to_csv(outdir / "risk_set_balance_report.csv", index=False)

    write_control_design_addendum(
        outdir / "CONTROL_DESIGN_ADDENDUM.md", cfg, d2, pool, rs_controls,
        rs_comp_controls, rnd_ctx, match_rep, balance, primary_stratum)
    log.info("Risk-set controls built in %.1fs", time.time() - step)

    # ------------------------------------------------------- PART C: benchmark
    step = time.time()
    states = build_states(d2)
    counts_states = state_counts(frame, states)
    counts_states.to_csv(outdir / "benchmark_state_counts.csv", index=False)

    primary_rows: list[dict] = []
    for direction in ("up", "down"):
        primary_rows.append(unconditional_row(frame, direction, thr, h, "primary"))
        if not rnd_ctx.empty:
            primary_rows.append(control_row(
                frame, rnd_ctx["control_decision_time"].to_numpy(), direction, thr, h,
                "primary", "B1_context_matched_random"))
        if not rs_controls.empty:
            primary_rows.append(control_row(
                frame, rs_controls["control_decision_time"].to_numpy(), direction, thr, h,
                "primary", "B1b_risk_set_matched_control"))
        for st in [s for s in states if s.direction == direction]:
            primary_rows.append(evaluate_state(frame, st, thr, h, "primary"))
    primary = pd.DataFrame(primary_rows)
    primary.to_csv(outdir / "benchmark_primary.csv", index=False)

    secondary_rows: list[dict] = []
    for ep in d2["secondary_endpoints"]:
        t2, h2 = float(ep["threshold_pips"]), int(ep["horizon_minutes"])
        sub = frame[(frame[f"horizon_complete_{h2}m"] == 1)
                    & (frame[f"bars_in_window_{h2}m"] > 0)]
        for direction in ("up", "down"):
            secondary_rows.append(unconditional_row(
                sub, direction, t2, h2, f"secondary_{t2:g}p_{h2}m"))
            for st in [s for s in states if s.direction == direction]:
                secondary_rows.append(evaluate_state(
                    sub, st, t2, h2, f"secondary_{t2:g}p_{h2}m"))
    secondary = pd.DataFrame(secondary_rows)
    secondary.to_csv(outdir / "benchmark_secondary.csv", index=False)

    if not balance.empty:
        balance.to_csv(outdir / "benchmark_balance.csv", index=False)

    write_benchmark_report(outdir / "DAY2_BENCHMARK_REPORT.md", cfg, d2,
                           primary, secondary, counts_states, balance, elig_counts)
    log.info("Benchmark computed in %.1fs", time.time() - step)

    # --------------------------------------------------------- PART D: checkpoint
    cons = pd.read_parquet(s2cfg / "consolidation_events.parquet")
    s2_controls = pd.read_parquet(s2cfg / "event_controls.parquet")
    overlap = pd.read_csv(s2cfg / "event_overlap_report.csv")

    write_checkpoint(
        outdir / "JONATHAN_2DAY_CHECKPOINT.md", cfg, d2, frame, elig_counts,
        counts_x, events, cons, s2_controls, overlap, rs_controls, rs_comp_controls,
        rnd_ctx, primary, secondary, balance, primary_stratum)

    # ------------------------------------------------------------- manifest
    elapsed = time.time() - t0
    d1 = cfg.output_dir
    manifest = {
        "step": "day2_checkpoint",
        "run_timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "git": _git_commit(PROJECT_ROOT),
        "package_versions": _package_versions(),
        "source_data_hashes": {s.path.name: _sha256(s.path) for s in cfg.sources},
        "day1_artifact_hashes": {n: _sha256(d1 / n) for n in
                                 ["features.parquet", "outcomes.parquet",
                                  "alignment_map.parquet"]},
        "step2_artifact_hashes": {"events.parquet": _sha256(s2cfg / "events.parquet")},
        "research_config": cfg.to_dict(),
        "day2_config": {k: (str(v) if isinstance(v, Path) else v) for k, v in d2.items()},
        "random_seed": int(d2["random_seed"]),
        "primary_endpoint": f"{thr:g}p directional expansion within {h}m",
        "secondary_endpoints": [f"{e['threshold_pips']:g}p/{e['horizon_minutes']}m"
                                for e in d2["secondary_endpoints"]],
        "control_definitions": {
            "episode_isolation": "Step 2 controls: purged +/-240m around episodes; "
                                 "preserved unchanged, not used as the predictive baseline.",
            "risk_set_matched": "No purge, no outcome filter; only the event's own "
                                "timestamp excluded. Matched on session/hour/volatility.",
            "risk_set_compression_matched": "As above plus compression bin.",
            "random_context_matched": "Stratified on session/hour/volatility.",
            "random_session_only": "Preserved in Step 2 outputs as control_type='random'.",
        },
        "benchmark_states": counts_states.to_dict(orient="records"),
        "counts": {
            "eligible_rows": elig_counts["eligible_rows"],
            "crossover_pair_events": int(len(detail)),
            "event_episodes": int(len(events)),
            "consolidation_failure_episodes": int(len(cons)),
            "step2_controls": int(len(s2_controls)),
            "risk_set_matched_controls": int(len(rs_controls)),
            "risk_set_compression_matched_controls": int(len(rs_comp_controls)),
            "random_context_controls": int(len(rnd_ctx)),
        },
        "exclusions": elig_counts,
        "warnings": [
            "Descriptive checkpoint. No model trained, no strategy or barrier optimised, "
            "no profitability established.",
            "Rows are autocorrelated and outcome windows overlap; no p-values or "
            "confidence intervals are reported.",
            "Step 2 purged controls are retained as episode-isolation controls and are "
            "deliberately NOT used as the predictive baseline.",
        ],
    }
    (outdir / "run_manifest_day2_checkpoint.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    log.info("Day 2 checkpoint complete in %.1fs -> %s", elapsed, outdir)
    return {"primary": primary, "secondary": secondary, "crossovers": counts_x,
            "manifest": manifest}


def load_step2_config_path(cfg: ResearchConfig) -> Path:
    p = cfg.output_dir.parent / "step2"
    if not (p / "events.parquet").exists():
        raise SystemExit(
            f"Missing Step 2 artefacts in {p}. Run:\n"
            "    python -m forex_research.step2 --config configs/forex_day1.yaml")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m forex_research.day2_checkpoint",
        description="Produce the two-day client checkpoint from existing artefacts.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    d2 = load_day2_config(args.config)
    if args.outdir:
        d2["output_dir"] = Path(args.outdir)

    configure_logging(args.log_level, Path(d2["output_dir"]) / "run_log_day2.txt")
    run(cfg, d2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ResearchConfig, load_config
from .control_sampling import (
    balance_report,
    build_control_pool,
    compression_bins,
    sample_matched_controls,
    sample_random_controls,
    volatility_bins,
)
from .event_deduplication import (
    build_purge_intervals,
    collapse_by_separation,
    collapse_to_episodes,
    in_any_interval,
    overlap_report,
)
from .event_definitions import EventSpec, Step2Config, load_step2_config
from .event_detection import (
    COMPRESSED,
    add_time_context,
    classify_compression,
    context_label,
    eligibility_mask,
    find_candidates,
    find_compressed_rows,
)
from .logging_utils import configure_logging, get_logger
from .manifest import _git_commit, _package_versions
from .reporting_step2 import write_step2_report

log = get_logger("step2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Columns needed from the Day 1 feature matrix to build context and matching keys.
CONTEXT_FEATURE_COLUMNS = ("decision_time",)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_day1(cfg: ResearchConfig, cfg2: Step2Config) -> dict[str, Any]:
    d1 = cfg.output_dir
    needed = ["features.parquet", "outcomes.parquet", "alignment_map.parquet"]
    for n in needed:
        if not (d1 / n).exists():
            raise SystemExit(
                f"Missing Day 1 artefact {d1 / n}.\nRun Day 1 first:\n"
                "    python -m forex_research.day1 --config configs/forex_day1.yaml"
            )

    outcomes = pd.read_parquet(d1 / "outcomes.parquet")
    alignment = pd.read_parquet(d1 / "alignment_map.parquet")

    feat_cols = [cfg2.compression_primary_column, cfg2.volatility_column]
    context = pd.read_parquet(
        d1 / "features.parquet", columns=["decision_time", *feat_cols])

    for frame, name in ((outcomes, "outcomes"), (alignment, "alignment_map"),
                        (context, "features")):
        if not frame["decision_time"].is_monotonic_increasing:
            raise SystemExit(f"{name} decision_time is not sorted; cannot align by position.")
    if not (len(outcomes) == len(alignment) == len(context)):
        raise SystemExit(
            f"Day 1 artefacts disagree on row count: outcomes={len(outcomes)}, "
            f"alignment={len(alignment)}, features={len(context)}."
        )
    if not (outcomes["decision_time"].to_numpy() == context["decision_time"].to_numpy()).all():
        raise SystemExit("outcomes and features decision_time vectors differ.")

    log.info("Loaded Day 1: %d rows, %d outcome columns", len(outcomes), outcomes.shape[1])
    return {
        "outcomes": outcomes.reset_index(drop=True),
        "alignment": alignment.reset_index(drop=True),
        "context": context.reset_index(drop=True),
        "hashes": {n: _sha256(d1 / n) for n in needed},
        "day1_dir": d1,
    }


def build_context(day1: dict, cfg2: Step2Config) -> pd.DataFrame:
    ctx = day1["context"].copy()
    ctx = add_time_context(ctx, cfg2)
    ctx["volatility"] = ctx[cfg2.volatility_column]
    ctx["volatility_bin"] = volatility_bins(ctx["volatility"], cfg2.volatility_bins)
    ctx["compression_percentile"] = ctx[cfg2.compression_primary_column]
    ctx["compression_bin"] = compression_bins(ctx["compression_percentile"])
    ctx["compression_state"] = classify_compression(day1["context"], cfg2).to_numpy()
    return ctx


def attach_feature_rows(
    decision_times: np.ndarray, day1_dir: Path, chunk: int = 20000
) -> pd.DataFrame:
    if len(decision_times) == 0:
        return pd.DataFrame()
    wanted = pd.DatetimeIndex(pd.unique(decision_times)).sort_values()
    parts: list[pd.DataFrame] = []
    for i in range(0, len(wanted), chunk):
        sl = wanted[i:i + chunk]
        parts.append(pd.read_parquet(
            day1_dir / "features.parquet",
            filters=[("decision_time", "in", sl.to_pydatetime().tolist())],
        ))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out.sort_values("decision_time").reset_index(drop=True)


def run(cfg: ResearchConfig, cfg2: Step2Config) -> dict[str, Any]:
    t0 = time.time()
    outdir = cfg2.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    day1 = load_day1(cfg, cfg2)
    outcomes, alignment = day1["outcomes"], day1["alignment"]
    ctx = build_context(day1, cfg2)

    all_events: list[pd.DataFrame] = []
    all_consolidation: list[pd.DataFrame] = []
    all_controls: list[pd.DataFrame] = []
    all_match_reports: list[pd.DataFrame] = []
    overlap: dict[str, tuple[int, int, int]] = {}
    exclusions: list[dict] = []
    purge_rows: list[dict] = []

    for spec in cfg2.event_specs():
        # ---------------------------------------------------- events
        cand, reasons = find_candidates(outcomes, alignment, day1["context"], spec, cfg2)
        cand = cand.merge(
            ctx[["decision_time", "session", "hour_bucket", "day_of_week",
                 "volatility_bin", "compression_bin", "compression_percentile"]],
            on="decision_time", how="left")
        episodes = collapse_to_episodes(cand, spec, cfg2, prefix="evt")

        if not episodes.empty:
            episodes = episodes.merge(
                ctx[["decision_time", "session", "hour_bucket", "day_of_week",
                     "volatility_bin", "compression_bin", "compression_percentile"]],
                left_on="canonical_decision_time", right_on="decision_time", how="left"
            ).drop(columns=["decision_time"])
            episodes["stratum"] = spec.tag
            episodes["event_context"] = [
                context_label(spec.direction, c == COMPRESSED, True)
                for c in episodes["compression_state"]
            ]
            episodes["source_quality_ok"] = True
            all_events.append(episodes)

        overlap[spec.tag] = (reasons["eligible"], reasons["candidates"], len(episodes))
        exclusions.append({"stratum": spec.tag, "population": "events", **reasons})

        # ------------------------------------- failed consolidation episodes
        failed_rows, freasons = find_compressed_rows(
            outcomes, alignment, day1["context"], spec, cfg2, expanded=False)
        failed = collapse_by_separation(
            failed_rows, cfg2.consolidation_episode_separation_minutes, "cons_fail", spec)
        if not failed.empty:
            failed = failed.merge(
                ctx[["decision_time", "session", "hour_bucket", "day_of_week",
                     "volatility_bin", "compression_bin", "compression_percentile"]],
                on="decision_time", how="left")
            failed["stratum"] = spec.tag
            failed["event_direction"] = spec.direction
            failed["event_threshold_pips"] = spec.threshold_pips
            failed["event_horizon_minutes"] = spec.horizon_minutes
            failed["compression_state"] = COMPRESSED
            failed["event_context"] = "compressed_no_major_expansion"
            all_consolidation.append(failed)
        exclusions.append({"stratum": spec.tag, "population": "compressed_no_expansion",
                           **freasons})

        # ---------------------------------------------------- controls
        eligible, _ = eligibility_mask(outcomes, alignment, spec, cfg2)

        # Control A / random draw from ORDINARY market states. Filtering expanders
        # out of the pool would make the comparator degenerate: its rate for the
        # very outcome under study would be 0 by construction, and every nested
        # outcome with it. Temporal purging (below) is what stops a control
        # sharing the event's move; outcome filtering is not needed for that and
        # is off by default.
        if cfg2.outcome_filter_general_controls:
            non_event = (outcomes[f"expansion_up_{spec.threshold_pips:g}p_"
                                  f"{spec.horizon_minutes}m"] != 1) & \
                        (outcomes[f"expansion_down_{spec.threshold_pips:g}p_"
                                  f"{spec.horizon_minutes}m"] != 1)
        else:
            non_event = pd.Series(True, index=outcomes.index)

        pool = build_control_pool(ctx, eligible, non_event, episodes, spec, cfg2)
        purge_rows.append({
            "stratum": spec.tag,
            "eligible_non_event_rows": pool.n_before_purge,
            "rows_purged_overlapping_events": pool.n_purged,
            "usable_control_rows": pool.n_after_purge,
            "purge_minutes": cfg2.control_purge_minutes,
            "purge_loss_pct": round(
                100 * pool.n_purged / pool.n_before_purge, 3) if pool.n_before_purge else 0.0,
        })

        if not episodes.empty and not pool.frame.empty:
            gen, rep = sample_matched_controls(
                episodes, pool.frame, spec, cfg2, "general_matched",
                use_compression=False, n_per_event=cfg2.controls_per_event)
            if not gen.empty:
                all_controls.append(gen)
            if not rep.empty:
                all_match_reports.append(rep.assign(stratum=spec.tag,
                                                    control_type="general_matched"))

            rnd = sample_random_controls(
                episodes, pool.frame, spec, cfg2,
                n_total=cfg2.random_controls_per_event * len(episodes))
            if not rnd.empty:
                all_controls.append(rnd)

        # consolidation-matched controls: compressed events vs compressed non-expanders
        comp_events = episodes[episodes["compression_state"] == COMPRESSED] \
            if not episodes.empty else pd.DataFrame()
        if not comp_events.empty and not failed.empty:
            fpool = failed.copy()
            intervals = build_purge_intervals(
                episodes, cfg2.control_purge_minutes, spec.horizon_minutes)
            inside = in_any_interval(
                pd.DatetimeIndex(fpool["decision_time"]).to_numpy(), intervals)
            fpool = fpool.loc[~inside]
            purge_rows[-1]["consolidation_controls_purged"] = int(inside.sum())
            purge_rows[-1]["consolidation_controls_usable"] = len(fpool)
            if not fpool.empty:
                cm, rep2 = sample_matched_controls(
                    comp_events, fpool, spec, cfg2, "consolidation_matched",
                    use_compression=True, n_per_event=cfg2.controls_per_event,
                    seed_offset=17)
                if not cm.empty:
                    all_controls.append(cm)
                if not rep2.empty:
                    all_match_reports.append(
                        rep2.assign(stratum=spec.tag, control_type="consolidation_matched"))

    # ------------------------------------------------------------- assemble
    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    consolidation = (pd.concat(all_consolidation, ignore_index=True)
                     if all_consolidation else pd.DataFrame())
    controls = pd.concat(all_controls, ignore_index=True) if all_controls else pd.DataFrame()
    match_report = (pd.concat(all_match_reports, ignore_index=True)
                    if all_match_reports else pd.DataFrame())

    for frame, name in ((events, "events"), (consolidation, "consolidation_events"),
                        (controls, "event_controls")):
        if not frame.empty:
            frame.to_parquet(outdir / f"{name}.parquet", index=False, compression="snappy")

    ov = overlap_report(overlap)
    ov.to_csv(outdir / "event_overlap_report.csv", index=False)
    pd.DataFrame(purge_rows).to_csv(outdir / "control_purge_report.csv", index=False)
    pd.DataFrame(exclusions).to_csv(outdir / "exclusions_report.csv", index=False)

    # -------------------------------------------------------- feature matrices
    if not events.empty:
        ev_feats = attach_feature_rows(
            events["canonical_decision_time"].to_numpy(), day1["day1_dir"])
        ev_feats.to_parquet(outdir / "event_feature_matrix.parquet",
                            index=False, compression="snappy")
    else:
        ev_feats = pd.DataFrame()

    if not controls.empty:
        ct_feats = attach_feature_rows(
            controls["control_decision_time"].to_numpy(), day1["day1_dir"])
        ct_feats.to_parquet(outdir / "control_feature_matrix.parquet",
                            index=False, compression="snappy")
    else:
        ct_feats = pd.DataFrame()

    if not consolidation.empty:
        cons_feats = attach_feature_rows(
            consolidation["canonical_decision_time"].to_numpy(), day1["day1_dir"])
        cons_feats.to_parquet(outdir / "consolidation_feature_matrix.parquet",
                              index=False, compression="snappy")

    # ------------------------------------------------------------- counts
    counts = _event_counts(events, consolidation, controls)
    counts.to_csv(outdir / "event_counts.csv", index=False)

    balances: list[pd.DataFrame] = []
    if not events.empty and not controls.empty:
        for ctype in controls["control_type"].unique():
            sub = controls[controls["control_type"] == ctype]
            for col, label in (("matching_session", "session"),
                               ("matching_volatility_bin", "volatility_bin"),
                               ("matching_compression_bin", "compression_bin")):
                b = balance_report(
                    events.rename(columns={"session": "matching_session",
                                           "volatility_bin": "matching_volatility_bin",
                                           "compression_bin": "matching_compression_bin"}),
                    sub, col, label)
                if not b.empty:
                    balances.append(b.assign(control_type=ctype, variable=label))
    bal = pd.concat(balances, ignore_index=True) if balances else pd.DataFrame()
    if not match_report.empty:
        match_report.to_csv(outdir / "control_matching_report.csv", index=False)
    if not bal.empty:
        bal.to_csv(outdir / "control_balance_report.csv", index=False)

    elapsed = time.time() - t0
    write_step2_report(
        outdir / "STEP2_REPORT.md", cfg, cfg2, events, consolidation, controls,
        ov, counts, pd.DataFrame(purge_rows), match_report, bal,
        pd.DataFrame(exclusions), elapsed,
    )

    manifest = {
        "step": "step2",
        "run_timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "git": _git_commit(PROJECT_ROOT),
        "package_versions": _package_versions(),
        "day1_artifact_hashes": day1["hashes"],
        "day1_output_dir": str(day1["day1_dir"]),
        "research_config": cfg.to_dict(),
        "step2_config": cfg2.to_dict(),
        "event_definitions": [
            {"tag": s.tag, "direction": s.direction, "threshold_pips": s.threshold_pips,
             "horizon_minutes": s.horizon_minutes} for s in cfg2.event_specs()
        ],
        "random_seed": cfg2.random_seed,
        "purge_minutes": cfg2.control_purge_minutes,
        "matching": {
            "session": cfg2.match_on_session, "hour_bucket": cfg2.match_on_hour_bucket,
            "day_of_week": cfg2.match_on_day_of_week,
            "volatility_column": cfg2.volatility_column,
            "volatility_bins": cfg2.volatility_bins,
        },
        "counts": {
            "event_episodes": int(len(events)),
            "consolidation_failure_episodes": int(len(consolidation)),
            "controls": int(len(controls)),
            "control_types": (controls["control_type"].value_counts().to_dict()
                              if not controls.empty else {}),
            "event_feature_rows": int(len(ev_feats)),
            "control_feature_rows": int(len(ct_feats)),
        },
        "warnings": _warnings(events, consolidation, controls, match_report),
    }
    import json
    (outdir / "run_manifest_step2.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    log.info("Step 2 complete in %.1fs: %d event episodes, %d consolidation-failure "
             "episodes, %d controls", elapsed, len(events), len(consolidation), len(controls))
    return {"events": events, "consolidation": consolidation, "controls": controls,
            "manifest": manifest}


def _event_counts(events, consolidation, controls) -> pd.DataFrame:
    rows: list[dict] = []
    if not events.empty:
        g = events.groupby(
            ["event_direction", "event_threshold_pips", "event_horizon_minutes",
             "event_context"], dropna=False).size().reset_index(name="n_episodes")
        g["population"] = "event"
        rows.append(g)
        s = events.groupby(
            ["event_direction", "event_threshold_pips", "event_horizon_minutes",
             "session"], dropna=False).size().reset_index(name="n_episodes")
        s["population"] = "event_by_session"
        s = s.rename(columns={"session": "event_context"})
        rows.append(s)
    if not consolidation.empty:
        c = consolidation.groupby(
            ["event_direction", "event_threshold_pips", "event_horizon_minutes",
             "event_context"], dropna=False).size().reset_index(name="n_episodes")
        c["population"] = "consolidation_failure"
        rows.append(c)
    if not controls.empty:
        k = controls.groupby(
            ["event_direction", "event_threshold_pips", "event_horizon_minutes",
             "control_type"], dropna=False).size().reset_index(name="n_episodes")
        k = k.rename(columns={"control_type": "event_context"})
        k["population"] = "control"
        rows.append(k)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _warnings(events, consolidation, controls, match_report) -> list[str]:
    w: list[str] = []
    if events.empty:
        w.append("No event episodes were produced.")
    if consolidation.empty:
        w.append("No failed-consolidation episodes were produced; the consolidation-matched "
                 "comparison cannot be formed.")
    if not match_report.empty:
        rate = float((match_report["controls_found"] > 0).mean())
        if rate < 0.95:
            w.append(f"Only {rate*100:.1f}% of events found at least one matched control; "
                     "sparse matching cells reduce the usable comparison set.")
    small = []
    if not events.empty:
        g = events.groupby("stratum").size()
        small = [s for s, n in g.items() if n < 30]
    if small:
        w.append(f"Strata with fewer than 30 episodes (descriptive only, too small to "
                 f"interpret): {', '.join(sorted(small)[:12])}")
    w.append("No model has been trained, no strategy optimised and no profitability "
             "established. These are event/control construction outputs only.")
    return w


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m forex_research.step2",
        description="Build the focused event and control dataset from the Day 1 outputs.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg2 = load_step2_config(args.config)
    if args.outdir:
        cfg2 = type(cfg2)(**{**cfg2.__dict__, "output_dir": Path(args.outdir)})

    configure_logging(args.log_level, cfg2.output_dir / "run_log_step2.txt")
    log.info("Step 2 config: %d strata, purge=%dm, controls/event=%d, seed=%d",
             len(cfg2.event_specs()), cfg2.control_purge_minutes,
             cfg2.controls_per_event, cfg2.random_seed)
    run(cfg, cfg2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

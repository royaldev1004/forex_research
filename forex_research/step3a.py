from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ResearchConfig, load_config
from .event_definitions import Step2Config, Step3AConfig, load_step2_config, load_step3a_config
from .logging_utils import configure_logging, get_logger
from .manifest import _git_commit, _package_versions
from .outcome_analysis import (
    apply_exclusions,
    attach_outcomes,
    control_excludes_outcome,
    compare_magnitudes,
    compare_rates,
    describe_population,
    feature_family_summary,
    outcome_columns,
)
from .reporting_step3a import write_step3a_report

log = get_logger("step3a")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path, what: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"Missing {what}: {path}\nRun Step 2 first:\n"
            "    python -m forex_research.step2 --config configs/forex_day1.yaml"
        )
    return pd.read_parquet(path)


def run(cfg: ResearchConfig, cfg2: Step2Config, cfg3: Step3AConfig) -> dict[str, Any]:
    t0 = time.time()
    outdir = cfg3.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    s2 = cfg2.output_dir
    events = _load(s2 / "events.parquet", "Step 2 events")
    controls = _load(s2 / "event_controls.parquet", "Step 2 controls")
    consolidation = _load(s2 / "consolidation_events.parquet", "Step 2 consolidation events")
    outcomes = pd.read_parquet(cfg.output_dir / "outcomes.parquet")

    all_cols: list[str] = []
    for h in cfg3.report_horizons_minutes:
        all_cols += outcome_columns(h, cfg3.report_thresholds_pips)
    all_cols = sorted(set(all_cols))

    ev_out = attach_outcomes(events, "canonical_decision_time", outcomes, all_cols)
    ct_out = attach_outcomes(controls, "control_decision_time", outcomes, all_cols)
    cons_out = attach_outcomes(consolidation, "canonical_decision_time", outcomes, all_cols)

    ev_out.to_parquet(outdir / "event_outcomes.parquet", index=False, compression="snappy")
    ct_out.to_parquet(outdir / "control_outcomes.parquet", index=False, compression="snappy")
    cons_out.to_parquet(outdir / "consolidation_outcomes.parquet",
                        index=False, compression="snappy")

    exclusions: list[dict] = []
    comparisons: list[dict] = []
    magnitudes: list[dict] = []
    distributions: list[pd.DataFrame] = []

    # unconditional base rate over every eligible decision row, per horizon
    base_rates: dict[tuple[int, str], float] = {}
    for h in cfg3.report_horizons_minutes:
        pool = outcomes[(outcomes[f"horizon_complete_{h}m"] == 1)
                        & (outcomes[f"bars_in_window_{h}m"] > 0)]
        for t in cfg3.report_thresholds_pips:
            for d in ("up", "down"):
                col = f"expansion_{d}_{t:g}p_{h}m"
                base_rates[(h, col)] = float(pool[col].dropna().mean())

    for spec in cfg2.event_specs():
        tag, h = spec.tag, spec.horizon_minutes
        # (direction, threshold, horizon) drives the logical-implication checks,
        # which must respect BOTH horizon nesting and threshold nesting.
        ev_def = (spec.direction, spec.threshold_pips, h)
        # general/random controls are ordinary market states unless the pool was
        # explicitly outcome-filtered, in which case their zero rates are implied.
        gen_excl = ((spec.threshold_pips, h)
                    if cfg2.outcome_filter_general_controls else None)
        # consolidation controls are non-expanders by definition, always.
        cons_excl = (spec.threshold_pips, h)

        ev = ev_out[ev_out["stratum"] == tag]
        ev, r1 = apply_exclusions(ev, h, "event", tag)
        exclusions += r1

        gm = ct_out[(ct_out["stratum"] == tag) & (ct_out["control_type"] == "general_matched")]
        gm, r2 = apply_exclusions(gm, h, "general_matched_control", tag)
        exclusions += r2

        rnd = ct_out[(ct_out["stratum"] == tag) & (ct_out["control_type"] == "random")]
        rnd, r3 = apply_exclusions(rnd, h, "random_control", tag)
        exclusions += r3

        cm = ct_out[(ct_out["stratum"] == tag)
                    & (ct_out["control_type"] == "consolidation_matched")]
        cm, r4 = apply_exclusions(cm, h, "consolidation_matched_control", tag)
        exclusions += r4

        cons = cons_out[cons_out["stratum"] == tag]
        cons, r5 = apply_exclusions(cons, h, "compressed_no_expansion", tag)
        exclusions += r5

        ev_comp = ev[ev["compression_state"] == "compressed"]

        for pop, name in ((ev, "event"), (gm, "general_matched_control"),
                          (rnd, "random_control"), (cons, "compressed_no_expansion"),
                          (ev_comp, "event_after_compression")):
            d = describe_population(pop, h, cfg3, name, tag)
            if not d.empty:
                distributions.append(d)

        # ---- Comparisons 1 & 2: events vs general matched controls -------
        comp_name = ("C1_major_up_vs_matched_control" if spec.direction == "up"
                     else "C2_major_down_vs_matched_control")
        for hh in cfg3.report_horizons_minutes:
            for t in cfg3.report_thresholds_pips:
                for d in ("up", "down"):
                    col = f"expansion_{d}_{t:g}p_{hh}m"
                    if col not in ev.columns:
                        continue
                    comparisons.append(compare_rates(
                        ev, gm, col, comp_name, tag, hh, ev_def,
                        "event_episode", "general_matched_control",
                        control_exclusion=gen_excl))

        for metric in (f"max_abs_move_pips_{h}m", f"long_mfe_pips_{h}m",
                       f"long_mae_pips_{h}m", f"future_range_pips_{h}m"):
            magnitudes.append(compare_magnitudes(
                ev, gm, metric, comp_name, tag, h,
                "event_episode", "general_matched_control"))

        # ---- Comparisons 3 & 4: compression -> expansion vs compression -> none
        comp_name2 = ("C3_compression_up_vs_compression_no_expansion" if spec.direction == "up"
                      else "C4_compression_down_vs_compression_no_expansion")
        control_pop = cm if not cm.empty else cons
        control_name = ("consolidation_matched_control" if not cm.empty
                        else "compressed_no_expansion")
        for hh in cfg3.report_horizons_minutes:
            for t in cfg3.report_thresholds_pips:
                for d in ("up", "down"):
                    col = f"expansion_{d}_{t:g}p_{hh}m"
                    if col not in ev_comp.columns:
                        continue
                    comparisons.append(compare_rates(
                        ev_comp, control_pop, col, comp_name2, tag, hh, ev_def,
                        "event_after_compression", control_name,
                        control_exclusion=cons_excl))

        for metric in (f"max_abs_move_pips_{h}m", f"future_range_pips_{h}m"):
            magnitudes.append(compare_magnitudes(
                ev_comp, control_pop, metric, comp_name2, tag, h,
                "event_after_compression", control_name))

        # ---- Comparison 5: controls vs the unconditional market rate ------
        for t in cfg3.report_thresholds_pips:
            for d in ("up", "down"):
                col = f"expansion_{d}_{t:g}p_{h}m"
                if col not in gm.columns:
                    continue
                n_c, r_c = (len(gm[col].dropna()), float(gm[col].dropna().mean())) \
                    if len(gm) else (0, np.nan)
                n_r, r_r = (len(rnd[col].dropna()), float(rnd[col].dropna().mean())) \
                    if len(rnd) else (0, np.nan)
                base = base_rates.get((h, col), np.nan)
                implied = (gen_excl is not None
                           and control_excludes_outcome(*gen_excl, col))
                for label, n, r in (("general_matched_control", n_c, r_c),
                                    ("random_control", n_r, r_r)):
                    diff = r - base if np.isfinite(r) and np.isfinite(base) else np.nan
                    comparisons.append({
                        "comparison": "C5_control_vs_unconditional_market",
                        "stratum": tag, "horizon_minutes": h, "outcome": col,
                        "event_population": label,
                        "control_population": "unconditional_all_eligible_rows",
                        "event_n": n, "control_n": int(
                            (outcomes[f"horizon_complete_{h}m"] == 1).sum()),
                        "event_rate": round(r, 5) if np.isfinite(r) else np.nan,
                        "control_rate": round(base, 5) if np.isfinite(base) else np.nan,
                        "absolute_difference_pp": (round(100 * diff, 3)
                                                   if np.isfinite(diff) else np.nan),
                        "relative_lift_pct": (round(100 * (r / base - 1), 2)
                                              if np.isfinite(r) and np.isfinite(base)
                                              and base > 0 else np.nan),
                        "event_side_implied": bool(implied),
                        "control_side_implied": False,
                        "is_tautological": bool(implied),
                        "min_population_n": n,
                        "interpretable": bool(n >= 30 and not implied),
                    })

    comp_df = pd.DataFrame(comparisons)
    mag_df = pd.DataFrame(magnitudes)
    dist_df = pd.concat(distributions, ignore_index=True) if distributions else pd.DataFrame()
    exc_df = pd.DataFrame(exclusions)

    comp_df.to_csv(outdir / "outcome_comparison_summary.csv", index=False)
    mag_df.to_csv(outdir / "outcome_magnitude_summary.csv", index=False)
    dist_df.to_csv(outdir / "outcome_distribution_summary.csv", index=False)
    exc_df.to_csv(outdir / "exclusions_report.csv", index=False)

    # optional breakdowns
    if not ev_out.empty:
        for by, name in (("session", "event_outcomes_by_session"),
                         ("event_horizon_minutes", "event_outcomes_by_horizon"),
                         ("event_threshold_pips", "event_outcomes_by_threshold"),
                         ("compression_state", "event_outcomes_by_compression_state")):
            if by not in ev_out.columns:
                continue
            g = ev_out.groupby(by).agg(
                n=("event_id", "count"),
                mean_max_abs_move_240m=(f"max_abs_move_pips_240m", "mean"),
                median_max_abs_move_240m=(f"max_abs_move_pips_240m", "median"),
            ).reset_index().round(3)
            g.to_csv(outdir / f"{name}.csv", index=False)

    # exploratory family-level view
    fam = pd.DataFrame()
    fd_path = cfg.output_dir / "feature_dictionary.csv"
    ef_path = cfg2.output_dir / "event_feature_matrix.parquet"
    cf_path = cfg2.output_dir / "control_feature_matrix.parquet"
    if fd_path.exists() and ef_path.exists() and cf_path.exists():
        fam = feature_family_summary(
            pd.read_parquet(ef_path), pd.read_parquet(cf_path),
            pd.read_csv(fd_path), cfg3.exploratory_feature_families)
        if not fam.empty:
            fam.to_csv(outdir / "exploratory_feature_family_summary.csv", index=False)

    elapsed = time.time() - t0
    write_step3a_report(outdir / "STEP3A_REPORT.md", cfg, cfg2, cfg3,
                        comp_df, mag_df, dist_df, exc_df, fam,
                        events, controls, consolidation, elapsed)

    manifest = {
        "step": "step3a",
        "run_timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "git": _git_commit(PROJECT_ROOT),
        "package_versions": _package_versions(),
        "step2_output_dir": str(cfg2.output_dir),
        "day1_output_dir": str(cfg.output_dir),
        "step3a_config": cfg3.to_dict(),
        "counts": {
            "event_rows": int(len(ev_out)),
            "control_rows": int(len(ct_out)),
            "consolidation_rows": int(len(cons_out)),
            "comparisons": int(len(comp_df)),
            "interpretable_comparisons": int(comp_df["interpretable"].sum())
            if not comp_df.empty else 0,
            "tautological_comparisons": int(comp_df["is_tautological"].sum())
            if not comp_df.empty else 0,
        },
        "warnings": [
            "Descriptive only. No model trained, no strategy optimised, no take-profit or "
            "stop-loss searched, no profitability established.",
            "5-minute observations are autocorrelated and event windows overlap, so rows are "
            "not independent. No p-values or confidence intervals are reported.",
            "Comparisons flagged is_tautological measure the outcome that defined the event "
            "and are circular by construction; they are retained only as a build check.",
            "Execution economics remain blocked on bid/ask tick data.",
        ],
    }
    (outdir / "run_manifest_step3a.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    log.info("Step 3A complete in %.1fs: %d comparisons (%d interpretable), %d exclusion rows",
             elapsed, len(comp_df), manifest["counts"]["interpretable_comparisons"], len(exc_df))
    return {"comparisons": comp_df, "magnitudes": mag_df, "distributions": dist_df,
            "exclusions": exc_df, "manifest": manifest}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m forex_research.step3a",
        description="Attach and describe future outcomes for the Step 2 populations.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg2 = load_step2_config(args.config)
    cfg3 = load_step3a_config(args.config)
    if args.outdir:
        cfg3 = type(cfg3)(**{**cfg3.__dict__, "output_dir": Path(args.outdir)})

    configure_logging(args.log_level, cfg3.output_dir / "run_log_step3a.txt")
    run(cfg, cfg2, cfg3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

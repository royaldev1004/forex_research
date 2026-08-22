#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "output" / "day1"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 70)


def _need(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(
            f"Missing {path}.\nRun the pipeline first:\n"
            "    python -m forex_research.day1 --config configs/forex_day1.yaml"
        )
    return path


def cmd_summary(args) -> None:
    import json

    m = json.loads(_need(OUT / "run_manifest.json").read_text(encoding="utf-8"))
    c = m["counts"]
    print("=" * 78)
    print("DAY 1 DATASET SUMMARY")
    print("=" * 78)
    print(f"  Run at (UTC)        : {m['run_timestamp_utc']}")
    print(f"  Symbol / base TF    : {m['config']['symbol']} / {m['config']['base_timeframe']}")
    print(f"  Timeframes          : {', '.join(m['config']['timeframes'])}"
          f"   (daily included: {m['config']['include_daily']})")
    print(f"  Price basis         : {m['config']['price_basis']}")
    print(f"  Indicators per TF   : {c['n_indicators_per_timeframe']} "
          f"({len(m['config']['ema_periods'])} EMA + {len(m['config']['sma_periods'])} SMA)")
    print()
    print(f"  Decision rows       : {c['n_decision_rows']:,}")
    print(f"  Valid rows          : {c['n_valid_rows']:,}  "
          f"({c['n_valid_rows']/c['n_decision_rows']*100:.1f}% after warm-up)")
    print(f"  Features            : {c['n_features']:,}")
    print(f"  Outcome columns     : {c['n_outcome_columns']}")
    print(f"  Long-panel rows     : {c['n_long_panel_rows']:,}")
    print(f"  Valid date range    : {c['first_valid_decision_time']} -> {c['last_valid_decision_time']}")
    print(f"  Runtime             : {m['timings_seconds']['total']}s")

    print("\n--- feature families " + "-" * 57)
    fd = pd.read_csv(_need(OUT / "feature_dictionary.csv"))
    print(fd["feature_family"].value_counts().to_string())

    print("\n--- validation " + "-" * 63)
    v = pd.read_csv(_need(OUT / "validation_report.csv"))
    print(f"  {int((v.status == 'ok').sum())} ok, "
          f"{int((v.status == 'warn').sum())} warn, "
          f"{int((v.status == 'fail').sum())} FAIL")
    key = v[v.check.isin([
        "no_lookahead_assertion",
        "historical_features_unchanged_after_future_mutation",
        "future_features_did_change",
        "feature_outcome_column_overlap",
    ])]
    for r in key.itertuples():
        print(f"  [{r.status:^4}] {r.check} = {r.value}")

    if m["blockers"]:
        print("\n--- blockers " + "-" * 65)
        for b in m["blockers"]:
            print(f"  * {b[:150]}{'...' if len(b) > 150 else ''}")


def cmd_dict(args) -> None:
    which = OUT / ("outcome_dictionary.csv" if args.outcomes else "feature_dictionary.csv")
    fd = pd.read_csv(_need(which))
    if args.search:
        fd = fd[fd.feature_name.str.contains(args.search, case=False, na=False)]
    if args.family:
        fd = fd[fd.feature_family == args.family]
    if args.timeframe:
        fd = fd[fd.timeframe.astype(str) == args.timeframe]
    print(f"{len(fd)} matching entries in {which.name}\n")
    cols = ["feature_name", "feature_family", "timeframe", "units", "description"]
    print(fd[cols].head(args.n).to_string(index=False))


def cmd_columns(args) -> None:
    import pyarrow.parquet as pq

    schema = pq.read_schema(_need(OUT / args.file))
    names = [n for n in schema.names
             if not args.search or args.search.lower() in n.lower()]
    print(f"{len(names)} matching columns in {args.file}\n")
    for n in names[: args.n]:
        print(" ", n)
    if len(names) > args.n:
        print(f"  ... and {len(names) - args.n} more (raise --n to see them)")


def cmd_outcomes(args) -> None:
    s = pd.read_csv(_need(OUT / "outcome_summary.csv"))
    print("--- forward horizons " + "-" * 56)
    fh = s[s.kind == "forward_horizon"]
    print(fh[["horizon_minutes", "n_valid_rows", "n_complete_windows",
              "n_empty_windows_market_closed", "median_abs_move_pips",
              "mean_abs_move_pips"]].to_string(index=False))
    print("\n--- expansion labels (market movement, NOT trade results) " + "-" * 19)
    ex = s[s.kind == "expansion_label"]
    print(ex[["horizon_minutes", "threshold_pips", "n_up_hit", "n_down_hit",
              "n_same_bar_ambiguous", "pct_up_hit", "pct_down_hit"]].to_string(index=False))


def cmd_row(args) -> None:
    ts = pd.Timestamp(args.timestamp)
    feats = _need(OUT / "features.parquet")

    df = pd.read_parquet(feats, filters=[("decision_time", "==", ts)])
    if df.empty:
        near = pd.read_parquet(feats, columns=["decision_time"])
        i = (near.decision_time - ts).abs().idxmin()
        raise SystemExit(
            f"No decision row at {ts}.\nNearest available: {near.decision_time.iloc[i]}"
        )

    r = df.iloc[0]
    print("=" * 78)
    print(f"MULTI-TIMEFRAME STATE AT {ts}")
    print(f"row valid (all timeframes warmed up): {bool(r.feature_row_valid)}")
    print("=" * 78)

    am = pd.read_parquet(OUT / "alignment_map.parquet",
                         filters=[("decision_time", "==", ts)]).iloc[0]

    tfs = ["5m", "15m", "1h", "4h"]
    print("\n--- which bar each timeframe contributed (proof of no lookahead) " + "-" * 12)
    for tf in tfs:
        print(f"  {tf:>3}: bar closed {am[f'{tf}__source_bar_close_time']}  "
              f"({am[f'{tf}__source_bar_age_minutes']:.0f} min old)  <= decision time")

    def g(col):
        return r[col] if col in r.index and pd.notna(r[col]) else float("nan")

    print("\n--- per-timeframe structure " + "-" * 49)
    hdr = (f"  {'TF':>3} {'EMA order':>10} {'ribbon w':>9} {'entangl':>8} "
           f"{'ATR pips':>9} {'compressed':>11} {'px vs EMA300':>13}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for tf in tfs:
        print(f"  {tf:>3} {g(f'{tf}__ema_ordering_score'):>10.3f} "
              f"{g(f'{tf}__combined_ribbon_width_atr'):>9.2f} "
              f"{g(f'{tf}__ema_entanglement_fraction'):>8.2f} "
              f"{g(f'{tf}__atr_pips'):>9.2f} "
              f"{g(f'{tf}__is_compressed_6'):>11.0f} "
              f"{g(f'{tf}__ema_300_price_dist_pips'):>13.1f}")
    print("\n  EMA order: +1 = fully bullish stack (short above long), -1 = fully bearish")
    print("  compressed: 1 = trailing range in its lowest quartile vs its own history")

    print("\n--- cross-timeframe agreement " + "-" * 47)
    print(f"  timeframes bullish : {g('mtf__n_timeframes_bullish'):.0f} / {len(tfs)}")
    print(f"  timeframes bearish : {g('mtf__n_timeframes_bearish'):.0f} / {len(tfs)}")
    print(f"  timeframes compressed: {g('mtf__n_timeframes_compressed'):.0f} / {len(tfs)}")
    print(f"  disagreement score : {g('mtf__direction_disagreement'):.3f}  (0 = all agree)")
    for a, b in [("5m", "15m"), ("5m", "1h"), ("5m", "4h"), ("15m", "1h"),
                 ("15m", "4h"), ("1h", "4h")]:
        print(f"  {a:>3} vs {b:<3} trend agreement: {g(f'x_{a}_{b}__trend_agreement'):+.0f}"
              f"   ribbon width ratio: {g(f'x_{a}_{b}__ribbon_width_ratio_atr'):.2f}")

    oc = pd.read_parquet(OUT / "outcomes.parquet", filters=[("decision_time", "==", ts)])
    if not oc.empty:
        o = oc.iloc[0]
        print("\n--- what happened NEXT (label, not a trade result) " + "-" * 26)
        print(f"  {'horizon':>8} {'move pips':>10} {'MFE long':>9} {'MAE long':>9} {'bars':>6}")
        for h in (15, 30, 60, 120, 240):
            print(f"  {str(h)+'m':>8} {o.get(f'fwd_pips_{h}m', float('nan')):>10.1f} "
                  f"{o.get(f'mfe_long_pips_{h}m', float('nan')):>9.1f} "
                  f"{o.get(f'mae_long_pips_{h}m', float('nan')):>9.1f} "
                  f"{o.get(f'bars_in_window_{h}m', float('nan')):>6.0f}")


def cmd_export_sample(args) -> None:
    f = pd.read_parquet(_need(OUT / "feature_sample.parquet"))
    o = pd.read_parquet(_need(OUT / "outcomes_sample.parquet"))
    n = args.n
    keep = ["symbol", "decision_time", "feature_row_valid"] + [
        c for c in f.columns
        if any(k in c for k in ("ordering_score", "combined_ribbon_width_atr",
                                "atr_pips", "is_compressed_6", "entanglement",
                                "trend_agreement", "mtf__"))
    ]
    fp = OUT / "feature_sample_readable.csv"
    op = OUT / "outcomes_sample_readable.csv"
    f[keep].head(n).to_csv(fp, index=False)
    o.head(n).to_csv(op, index=False)
    print(f"Wrote {fp}  ({n} rows x {len(keep)} readable columns)")
    print(f"Wrote {op}  ({n} rows x {len(o.columns)} columns)")
    print("\nBoth open directly in Excel.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Quick terminal view of the Day 1 output artefacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("summary", help="headline counts, families and validation status"
                   ).set_defaults(func=cmd_summary)

    d = sub.add_parser("dict", help="search the feature / outcome dictionary")
    d.add_argument("--search", help="substring of the feature name")
    d.add_argument("--family", help="exact feature family")
    d.add_argument("--timeframe", help="exact timeframe label")
    d.add_argument("--outcomes", action="store_true", help="search outcomes instead")
    d.add_argument("--n", type=int, default=30)
    d.set_defaults(func=cmd_dict)

    c = sub.add_parser("columns", help="list column names in a parquet file")
    c.add_argument("--file", default="features.parquet")
    c.add_argument("--search")
    c.add_argument("--n", type=int, default=40)
    c.set_defaults(func=cmd_columns)

    sub.add_parser("outcomes", help="forward-horizon and expansion-label counts"
                   ).set_defaults(func=cmd_outcomes)

    r = sub.add_parser("row", help="full multi-timeframe state at one timestamp")
    r.add_argument("timestamp", help='e.g. "2026-06-16 15:25" (file-native clock)')
    r.set_defaults(func=cmd_row)

    e = sub.add_parser("export-sample", help="write Excel-openable CSV slices")
    e.add_argument("--n", type=int, default=500)
    e.set_defaults(func=cmd_export_sample)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

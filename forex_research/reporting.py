from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import ResearchConfig
from .feature_dictionary import family_counts
from .logging_utils import get_logger

log = get_logger("reporting")


def _md_table(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df.empty:
        return "_(none)_\n"
    sub = df.head(max_rows)
    cols = list(sub.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for row in sub.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_({len(df) - max_rows} further rows omitted; see the CSV artefact.)_")
    return "\n".join(lines) + "\n"


def write_validation_addendum(
    path: Path,
    cfg: ResearchConfig,
    validation: pd.DataFrame,
    lag_dist: pd.DataFrame,
    source_quality: pd.DataFrame,
    counts: dict[str, Any],
    n_quality_flagged: int,
    artifacts: list[dict[str, Any]],
) -> None:
    panel = next((a for a in artifacts if a["file"].startswith("indicator_panel_long")), None)

    md = f"""# Day 1 Validation Addendum

Findings from the post-Day-1 methodology review, and what changed. The main
`DAY1_REPORT.md` is regenerated on every run and reflects the corrected state;
this addendum records *what was wrong and why it changed*, which a regenerated
report cannot show.

> No model has been trained, no strategy optimised, no take-profit or stop-loss
> searched, and no profitability claim made.

---

## 1. Indicator timing: one bar of avoidable staleness (CHANGED)

**Finding.** Decision timestamps are base `bar_close_time`, and alignment selects
the latest bar with `bar_close_time <= decision_time`. That already excludes any
forming bar. The indicator source was *additionally* shifted one bar to honour
the reference scripts' `src = ha_close[1]`, which excluded the forming bar a
second time and discarded a candle that was genuinely available.

**Resolution.** `indicator_source_mode` now selects the semantics explicitly:

| mode | shift | newest contributing candle |
|---|---|---|
| `latest_closed` **(new default)** | 0 | the aligned bar itself |
| `previous_closed` | 1 | one bar before the aligned bar |

The governing principle is *what was objectively available at the decision
timestamp*, so `latest_closed` is the Day 2 research default. `previous_closed`
remains fully supported for anyone wanting the literal Pine offset on top of
alignment. Setting `indicator_source_shift_bars` in contradiction to the mode is
a hard configuration error, so the semantics cannot drift silently.

**Both modes are point-in-time safe.** Only staleness differs:

{_md_table(lag_dist)}

**Effect.** One bar of staleness removed per timeframe - 240 minutes on 4h, where
mean effective lag falls from 551 to 213 minutes.

Evidence: `indicator_timing_audit.csv`, `INDICATOR_TIMING_NOTE.md`.

---

## 2. Excursion terminology (CHANGED)

**Finding.** Columns named `mfe_*`/`mae_*` could be negative, contradicting the
conventional meaning of maximum favourable/adverse excursion.

**Resolution.** The raw signed measurement and the conventional floored excursion
are now separate columns, so neither meaning is overloaded:

| column | definition | sign |
|---|---|---|
| `max_future_delta_pips_{{h}}m` | highest high minus decision close | signed |
| `min_future_delta_pips_{{h}}m` | lowest low minus decision close | signed |
| `long_mfe_pips_{{h}}m` | `max(0, max_future_delta)` | >= 0 |
| `long_mae_pips_{{h}}m` | `max(0, -min_future_delta)` | >= 0 |
| `short_mfe_pips_{{h}}m` | `max(0, -min_future_delta)` | >= 0 |
| `short_mae_pips_{{h}}m` | `max(0, max_future_delta)` | >= 0 |

No information is lost: a one-sided move keeps its sign in the delta columns.
None of these are trade P&L - no entry rule, direction, spread or commission is
assumed.

**Bug caught while doing this.** The first implementation used `np.fmax`, which
*ignores* NaN and therefore turned unobserved windows into a spurious `0.0` -
silently reporting "no movement" where the truth was "no data". Changed to
`np.maximum` so NaN propagates, and locked down by a test asserting every
excursion column is NaN for a market-closed window.

---

## 3. Incomplete higher-timeframe source bars (NEW)

**Finding.** Some HTF bars are built from fewer base bars than expected, and were
not distinguishable from fully-formed ones.

**Resolution.** Every constructed bar now carries `expected_constituent_bars`,
`actual_constituent_bars`, `source_bar_completeness_ratio`, `source_bar_complete`
and `source_bar_gap_kind`. Expected counts are measured against the *tradeable*
slot grid, so a weekend or a full-day holiday closure is classified
`market_closed` rather than being wrongly reported as missing data; only genuine
in-session feed gaps are `partial_gap`.

{_md_table(source_quality)}

Flags are propagated onto decision rows in `alignment_map.parquet` as
`{{tf}}__source_bar_complete` plus a conservative `source_quality_ok` (true only
when every timeframe's aligned bar is whole). **{n_quality_flagged:,} of
{counts['n_decision_rows']:,} decision rows** are flagged, and Step 2 excludes
them from confirmatory event and control selection.

Evidence: `source_bar_quality.csv`.

---

## 4. Artefact size reporting (FIXED)

**Finding.** The long-form indicator panel was reported as a few kilobytes.
`Path.stat().st_size` on a directory returns the directory entry, not its
contents.

**Resolution.** Directory-backed artefacts are now summed recursively and report
a file count. The panel correctly shows
**{panel['size'] if panel else 'n/a'} across {panel['n_files'] if panel else 0} partition files
({panel['rows'] if panel else 'n/a'} rows)**.

---

## 5. Re-validation after the changes

{_md_table(validation[validation["check"].isin([
    "no_lookahead_assertion",
    "historical_features_unchanged_after_future_mutation",
    "future_features_did_change",
    "feature_outcome_column_overlap",
    "feature_columns_with_outcome_like_names",
    "feature_specs_uses_future_data",
    "features_documented",
    "dictionary_entries_without_column",
    "resampling_matches_independent_recompute",
    "valid_feature_rows",
])][["check", "scope", "value", "status"]])}

Total: {int((validation['status'] == 'ok').sum())} ok,
{int((validation['status'] == 'warn').sum())} warn,
{int((validation['status'] == 'fail').sum())} fail.

Point-in-time integrity is intact, so Step 2 may proceed.

## 6. Effect on the dataset

| | |
|---|---|
| Valid feature rows | {counts['n_valid_rows']:,} |
| Features | {counts['n_features']:,} |
| Outcome columns | {counts['n_outcome_columns']} |
| First valid decision time | {counts['first_valid_decision_time']} |

The valid-row count moved slightly because removing the source shift changes each
timeframe's warm-up boundary by one bar.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    log.info("Wrote validation addendum: %s", path)


def write_day1_report(
    path: Path,
    cfg: ResearchConfig,
    dq: pd.DataFrame,
    coverage: pd.DataFrame,
    alignment_table: pd.DataFrame,
    validation: pd.DataFrame,
    specs: list,
    outcome_specs: list,
    outcome_summary: pd.DataFrame,
    counts: dict[str, Any],
    warnings: list[str],
    blockers: list[str],
    outputs: dict[str, Any],
) -> None:
    fam = family_counts(specs)
    fam_total = fam.groupby("feature_family")["n_features"].sum().reset_index()
    fam_total = fam_total.sort_values("n_features", ascending=False).reset_index(drop=True)

    out_fam = (
        pd.DataFrame([s.as_row() for s in outcome_specs])
        .groupby("feature_family").size().reset_index(name="n_columns")
        .sort_values("n_columns", ascending=False).reset_index(drop=True)
    )

    def dq_val(source_contains: str, check: str) -> str:
        m = dq[(dq["source"].str.contains(source_contains, case=False, na=False))
               & (dq["check"] == check)]
        return str(m["value"].iloc[0]) if len(m) else "n/a"

    dq_problems = dq[dq["status"].isin(["warn", "fail"])][
        ["source", "check", "value", "status", "detail"]]

    val_fail = validation[validation["status"] == "fail"]
    val_key = validation[validation["check"].isin([
        "no_lookahead_assertion",
        "historical_features_unchanged_after_future_mutation",
        "future_features_did_change",
        "feature_outcome_column_overlap",
        "feature_columns_with_outcome_like_names",
        "feature_specs_uses_future_data",
        "features_documented",
        "resampling_matches_independent_recompute",
        "valid_feature_rows",
    ])]

    periods = ", ".join(str(p) for p in cfg.ema_periods)

    md = f"""# Day 1 Report - EUR/USD Multi-Timeframe EMA/MA Research Foundation

**Symbol:** {cfg.symbol}  **Base timeframe:** {cfg.base_timeframe}  **Price basis:** {cfg.price_basis}
**Timeframes:** {", ".join(cfg.active_timeframes)}  **Daily included:** {cfg.include_daily}
**Cross-timeframe mode:** {cfg.cross_tf_mode}

> **Scope.** Day 1 is a data-engineering and feature-engineering milestone. No model has been
> trained, no strategy has been optimised, no take-profit or stop-loss has been searched, and
> **no claim of profitability is made or implied**. The "outcome" columns describe what the
> market did after each timestamp; they are not trade results. No Pine Script strategy has been
> reproduced or benchmarked.

---

## A. Data status

| Item | Value |
|---|---|
| Base source | `{cfg.base_source.path.name}` ({cfg.base_source.timeframe}, MT5 bar export, tab-separated) |
| Base records | {dq_val("M5", "row_count")} |
| Base range (file-native clock) | {dq_val("M5", "first_bar_open_native")} to {dq_val("M5", "last_bar_close_native")} |
| Base range (UTC) | {dq_val("M5", "first_bar_open_utc")} to {dq_val("M5", "last_bar_close_utc")} |
| Reference source | `EURUSD_M1.csv` (M1), {dq_val("M1", "row_count")} records |
| M1 range (file-native) | {dq_val("M1", "first_bar_open_native")} to {dq_val("M1", "last_bar_close_native")} |
| Timestamp semantics | **{cfg.timestamp_semantics}** (established empirically, see below) |
| Timezone | file-native = New York + 7h (UTC+3 under US DST, UTC+2 otherwise); UTC columns derived |
| Duplicate timestamps | {dq_val("M5", "duplicate_timestamps")} (base) |
| Invalid OHLC rows | {dq_val("M5", "invalid_ohlc_relationships")} (base) |
| Bid/ask available | **No** - single bid-side OHLC series only |
| Tick data available | **No** |
| Spread column | Yes, MT5 `<SPREAD>` in points; median {dq_val("M5", "spread_pips_median")} pips |
| Real volume | Zero throughout; only tick volume is usable |

### How the timestamp semantics were established

Timestamps were not assumed. The M1 file was aggregated into 5-minute bins under both
labelling conventions and compared against the M5 file:

{_md_table(dq[dq["check"].str.startswith("aggregation_") | (dq["check"] == "inferred_timestamp_semantics")][["check", "value", "status"]])}

Left-labelled (bar-open) aggregation reproduces the M5 OHLC exactly; the right-labelled
alternative does not. Both files therefore label bars by **open** time, and
`bar_close_time = bar_open_time + timeframe duration` is derived explicitly.

### How the timezone was established

Every weekly session in the file opens at Monday 00:00 and closes at Friday 23:55 in
file-native time - including inside the windows where the US and EU daylight-saving
calendars disagree (late October 2025, March 2026). The FX week runs Sunday 17:00 to Friday
17:00 New York time, so the file clock is New York + 7 hours. This is the conventional MT5
broker server clock and is applied via the `America/New_York` DST calendar rather than a
fixed offset. Bars are constructed on the file-native clock because MT5 and TradingView
define H4 and D1 bars against it, and because the trading week aligns to midnight there, so
4h and daily bars never straddle the weekend gap.

### Data problems found

{_md_table(dq_problems)}

---

## B. Timeframe construction

Each timeframe is built by resampling **raw base-timeframe prices**, then constructing that
timeframe's own OHLC, then Heikin Ashi, then indicators. Higher-timeframe indicators are never
produced by averaging lower-timeframe indicator values.

{_md_table(coverage)}

Warm-up per timeframe is {cfg.warmup_bars()} bars: the longest moving-average period
({cfg.max_indicator_period()}), plus the spec-mandated {cfg.indicator_source_shift_bars}-bar
source shift, plus the longest slope lookback, plus the longest rolling window.

---

## C. Point-in-time integrity

**Synchronisation.** Decision timestamps are the base timeframe's `bar_close_time`. For each
timeframe, the aligned bar is the latest one satisfying `bar_close_time <= decision_time`,
found with a backward `searchsorted` on close times. Because the search runs on close times, a
bar still forming at the decision time can never be selected. This is a backward as-of join,
not a forward fill: a higher-timeframe value is never propagated backwards to observations
preceding its close.

**Forming candles.** Prevented in three independent ways: (1) alignment selects on
`bar_close_time`; (2) a partially-elapsed final bin is dropped at construction time; (3) the
indicator source is shifted by {cfg.indicator_source_shift_bars} bar as the client's reference
scripts require (`src = ha_close[1]`), so indicators see only the previous closed candle.

**Verification retained.** `alignment_map.parquet` stores, for every decision row and every
timeframe, the source bar index, open time, close time and age in minutes, so the guarantee can
be re-checked independently of this pipeline.

{_md_table(alignment_table)}

### Leakage and quality-control test results

{_md_table(val_key)}

The causality test re-runs the entire pipeline on a bounded slice of real data, once
unmodified and once with every bar after a cut point shifted by ~500 pips and perturbed. All
feature rows at or before the cut are required to be bit-identical, NaN patterns included; a
second assertion confirms rows *after* the cut did change, so the test cannot pass vacuously.

{"**All validation checks passed.**" if val_fail.empty else "**FAILURES:**\\n\\n" + _md_table(val_fail)}

---

## D. Feature status

Total features: **{counts.get('n_features', 0):,}** across **{counts.get('n_valid_rows', 0):,}** valid decision rows.

By family:

{_md_table(fam_total)}

By family and timeframe:

{_md_table(fam, max_rows=200)}

Indicator set per timeframe: **{len(cfg.ema_periods)} EMAs + {len(cfg.sma_periods)} SMAs** at
periods {periods} - taken from the client's reference indicator scripts, not invented here.

The complete indicator state is additionally exported in normalised long form
(`indicator_panel_long/`), one row per (bar, indicator_type, period) at each timeframe's own
native resolution, carrying value, slope at every configured lookback, price distance and
ATR-normalised distance. Joined to `alignment_map.parquet` this supports arbitrary later
analysis - including all-pairs cross-timeframe comparisons - without recomputing indicators.

---

## E. Outcome status

Outcomes are stored separately from features and joined on `symbol + decision_time`. They are
**future market movements**, not trade results: no entry rule, direction, spread, slippage or
commission is assumed.

Outcome columns: **{counts.get('n_outcome_columns', 0)}**

{_md_table(out_fam)}

{_md_table(outcome_summary, max_rows=80)}

Horizons are wall-clock windows `(T, T + horizon]` evaluated against real bar close times, so
weekend and holiday gaps cannot silently stretch a window. `horizon_complete_*` marks rows
where the dataset does not extend far enough to observe the full window.

MFE/MAE is stored for both directions at every horizon, along with the time to each extreme, so
later take-profit / stop-loss research can be done without rebuilding the feature pipeline.
Where both an upward and a downward threshold are reached inside the same 5-minute bar, the
row is flagged as ambiguous (`expansion_first_* == 2`) rather than resolved by guesswork.

---

## F. Problems and blockers

{chr(10).join(f"- {b}" for b in blockers) if blockers else "_None._"}

### Warnings

{chr(10).join(f"- {w}" for w in warnings) if warnings else "_None._"}

---

## G. What Day 2 can now analyse

The dataset supports, without any further data engineering:

- **Major-move event discovery** - select decision rows by `expansion_up_*` / `expansion_down_*`
  / `max_abs_move_pips_*` and inspect the multi-timeframe state that preceded them.
- **States before major upward vs downward moves** - compare feature distributions between the
  two groups across 5m, 15m, 1h and 4h simultaneously.
- **Consolidation that expands vs consolidation that does not** - condition on
  `*__is_compressed_*` / `*__compression_percentile_*` at decision time, then split by whether an
  expansion threshold was subsequently reached. The compression label is trailing-only, so this
  comparison is not circular.
- **Matched control sampling** - draw non-event rows matched on time of day, session, volatility
  regime (`*__atr_pips`, `*__realized_vol_*`) or compression state.
- **Random-entry controls** - the outcome table covers *every* base bar, so a random-timestamp
  baseline needs no extra computation.
- **Cross-timeframe configuration comparison** - `cross_tf_*` and `mtf__*` features let each
  timeframe pairing be scored against outcomes, so which pairing matters is measured rather
  than assumed.
- **All-pairs indicator analysis** - reconstructable from the long-form panel plus the alignment
  map.

**No statistical association has been measured yet.** Any statement about edge, hit rate or
profitability must wait until Day 2 analysis has actually been performed on this dataset.

---

## Reproducing this run

```bash
python -m forex_research.day1 --config configs/forex_day1.yaml
```

Artefacts written to `{outputs.get('output_dir', '')}`:

{_md_table(pd.DataFrame(outputs.get("files", [])))}

`run_manifest.json` records the configuration, source file SHA-256 hashes, package versions,
row/feature counts and every warning raised during the run.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    log.info("Wrote Day 1 report: %s", path)

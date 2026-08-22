from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ResearchConfig
from .event_definitions import Step2Config
from .logging_utils import get_logger

log = get_logger("reporting_step2")


def _tbl(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df is None or df.empty:
        return "_(none)_\n"
    sub = df.head(max_rows)
    cols = list(sub.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in sub.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_({len(df) - max_rows} further rows omitted; see the CSV artefact.)_")
    return "\n".join(lines) + "\n"


def write_step2_report(
    path: Path,
    cfg: ResearchConfig,
    cfg2: Step2Config,
    events: pd.DataFrame,
    consolidation: pd.DataFrame,
    controls: pd.DataFrame,
    overlap: pd.DataFrame,
    counts: pd.DataFrame,
    purge: pd.DataFrame,
    match_report: pd.DataFrame,
    balance: pd.DataFrame,
    exclusions: pd.DataFrame,
    elapsed: float,
) -> None:
    by_dir = (events.groupby(["event_direction", "event_threshold_pips",
                              "event_horizon_minutes"]).size()
              .reset_index(name="n_episodes") if not events.empty else pd.DataFrame())

    by_ctx = (events.groupby(["event_direction", "event_context"]).size()
              .reset_index(name="n_episodes") if not events.empty else pd.DataFrame())

    by_sess = (events.groupby(["event_direction", "session"]).size()
               .reset_index(name="n_episodes") if not events.empty else pd.DataFrame())

    cons_by = (consolidation.groupby(["event_threshold_pips", "event_horizon_minutes"]).size()
               .reset_index(name="n_episodes") if not consolidation.empty else pd.DataFrame())

    ctl_by = (controls.groupby(["control_type", "event_horizon_minutes"]).size()
              .reset_index(name="n_controls") if not controls.empty else pd.DataFrame())

    match_rate = ""
    if not match_report.empty:
        r = match_report.groupby(["control_type"]).agg(
            events=("event_id", "nunique"),
            with_control=("controls_found", lambda s: int((s > 0).sum())),
            mean_found=("controls_found", "mean"),
        ).reset_index()
        r["match_rate_pct"] = (100 * r["with_control"] / r["events"]).round(2)
        r["mean_found"] = r["mean_found"].round(2)
        match_rate = _tbl(r)

    excl_ev = exclusions[exclusions["population"] == "events"] if not exclusions.empty \
        else pd.DataFrame()
    excl_cols = [c for c in ["stratum", "total_rows", "feature_row_invalid",
                             "incomplete_horizon", "market_closed_window", "outcome_nan",
                             "incomplete_htf_source_bar", "eligible", "candidates"]
                 if not excl_ev.empty and c in excl_ev.columns]

    md = f"""# Step 2 Report - Focused Event and Control Dataset

**Symbol:** {cfg.symbol}  **Base timeframe:** {cfg.base_timeframe}
**Indicator timing:** `{cfg.indicator_source_mode}`  **Runtime:** {elapsed:.1f}s

> **No predictive model has been trained. No trading profitability has been
> established. No take-profit or stop-loss has been optimised.** These are
> event and control *construction* outputs: populations to be analysed later,
> not results.

---

## A. Event definitions

An **event episode** is a period during which the market expanded by at least a
threshold, in a direction, within a horizon - counted once, not once per
overlapping 5-minute row.

| Parameter | Value |
|---|---|
| Directions | {", ".join(cfg2.directions)} |
| Thresholds (pips) | {", ".join(f"{t:g}" for t in cfg2.thresholds_pips)} |
| Horizons (minutes) | {", ".join(str(h) for h in cfg2.horizons_minutes)} |
| Canonical timestamp rule | `{cfg2.canonical_rule}` |
| Episode suppression rule | `{cfg2.suppression_rule}` |
| Extra separation | {cfg2.extra_separation_minutes} min |
| Control purge | {cfg2.control_purge_minutes} min |
| Controls per event | {cfg2.controls_per_event} matched + {cfg2.random_controls_per_event} random |
| Random seed | {cfg2.random_seed} |

These thresholds and horizons are **research strata**, declared in config before
outcomes were inspected. They were not selected by looking at which produced the
most attractive result.

### Episode de-duplication

Per stratum (direction x threshold x horizon), scanning strictly forward:

1. Sort candidate timestamps ascending.
2. The first unclaimed candidate opens an episode and becomes its **canonical
   decision timestamp**.
3. Later candidates falling inside the suppression window are absorbed into that
   episode.
4. The next unclaimed candidate opens the next episode.

The canonical timestamp is always the **earliest qualifying** row. Picking the
"best looking" row inside an episode by consulting the future path would leak
outcome information into event selection, so it is not offered.

Up and down episodes are detected in **separate scans**, so an up-move and a
down-move cannot merge into one event.

### Temporal purge

A control at time `t` looks forward `horizon` minutes. To stop a control sharing
the very move it is controlling for, any `t` whose forward window overlaps an
episode extent - widened by {cfg2.control_purge_minutes} minutes on both sides -
is excluded from the control pool.

### Compression context

Read from `{cfg2.compression_primary_column}`, a Day 1 trailing-only percentile
that never sees the future:

| State | Rule |
|---|---|
| compressed | percentile <= {cfg2.compression_compressed_max_percentile} |
| neutral | in between |
| expanding | percentile >= {cfg2.compression_expanding_min_percentile} |

Consolidation is **not** defined retrospectively from the fact that a breakout
happened. Continuous percentile values are preserved alongside these flags.

---

## B. Event counts

By direction, threshold and horizon:

{_tbl(by_dir)}

By compression context:

{_tbl(by_ctx)}

By session:

{_tbl(by_sess)}

---

## C. Overlap handling

This is the core de-duplication result: how many raw labelled rows collapsed into
how many genuine episodes.

{_tbl(overlap)}

A high `avg_rows_per_episode` is expected and is precisely the reason this step
exists - without it, one multi-hour move would have been counted many times over.

---

## D. Consolidation events

Compression that expanded, versus compression that did not:

{_tbl(cons_by)}

Failed-consolidation episodes require **no expansion in either direction** at the
stratum's threshold and horizon. A row that fell 30 pips is not a failed
consolidation merely because it did not rise. They are additionally thinned to a
minimum spacing of {cfg2.consolidation_episode_separation_minutes} minutes so a
single quiet afternoon cannot dominate the negative sample.

---

## E. Controls

{_tbl(ctl_by)}

Matching success:

{match_rate or "_(no matched controls)_"}

Purge losses:

{_tbl(purge)}

Balance between events and controls on the matching variables:

{_tbl(balance, max_rows=40)}

**Control A (`general_matched`)** matches session, hour bucket and trailing
volatility bin, but deliberately **not** compression - it asks whether event
pre-states differ from ordinary market states under similar conditions.

**Control B (`consolidation_matched`)** matches compression bin as well, drawing
from compressed rows that did not expand - it asks what separates consolidation
that breaks out from consolidation that does not.

**Random controls** are fixed-seed draws stratified on session, so the null is
not trivially different in time-of-day composition.

---

## F. Data-quality exclusions

Rows removed before any event label was consulted:

{_tbl(excl_ev[excl_cols] if excl_cols else pd.DataFrame(), max_rows=40)}

Exclusion reasons: incomplete forward horizon at the end of the dataset; windows
where the market was closed throughout; NaN outcomes; and decision rows whose
aligned higher-timeframe source bar was built from incomplete base data
(`source_quality_ok=False`, from the Day 1 addendum).

---

## G. No claims

* No predictive model has been trained.
* No trading profitability has been established or suggested.
* No entry, take-profit or stop-loss has been optimised.
* Execution economics remain blocked on bid/ask tick data.
* These outputs are event and control **populations**, not findings.

Descriptive outcome comparisons are produced separately in Step 3A; inferential
testing, multiple-testing control and walk-forward validation belong to Step 4
and beyond.

---

## Reproducing

```bash
python -m forex_research.day1  --config configs/forex_day1.yaml
python -m forex_research.step2 --config configs/forex_day1.yaml
```

`run_manifest_step2.json` records the Day 1 artefact hashes, event definitions,
seed, purge and matching parameters, and all counts.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    log.info("Wrote Step 2 report: %s", path)

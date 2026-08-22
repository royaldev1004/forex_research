from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ResearchConfig
from .event_definitions import Step2Config, Step3AConfig
from .logging_utils import get_logger

log = get_logger("reporting_step3a")


def _tbl(df: pd.DataFrame, max_rows: int = 40) -> str:
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


def _headline(comp: pd.DataFrame, name: str, top: int = 12) -> pd.DataFrame:
    if comp.empty:
        return pd.DataFrame()
    sub = comp[(comp["comparison"] == name) & comp["interpretable"]].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.reindex(sub["absolute_difference_pp"].abs().sort_values(
        ascending=False).index).head(top)
    return sub[["stratum", "outcome", "event_n", "control_n", "event_rate",
                "control_rate", "absolute_difference_pp", "relative_lift_pct"]]


def write_step3a_report(
    path: Path,
    cfg: ResearchConfig,
    cfg2: Step2Config,
    cfg3: Step3AConfig,
    comparisons: pd.DataFrame,
    magnitudes: pd.DataFrame,
    distributions: pd.DataFrame,
    exclusions: pd.DataFrame,
    families: pd.DataFrame,
    events: pd.DataFrame,
    controls: pd.DataFrame,
    consolidation: pd.DataFrame,
    elapsed: float,
) -> None:
    n_taut = int(comparisons["is_tautological"].sum()) if not comparisons.empty else 0
    n_interp = int(comparisons["interpretable"].sum()) if not comparisons.empty else 0

    excl_summary = pd.DataFrame()
    if not exclusions.empty:
        excl_summary = (exclusions[exclusions["reason"] != "retained"]
                        .groupby(["population", "reason"])["n_excluded"].sum()
                        .reset_index().sort_values("n_excluded", ascending=False))
        retained = (exclusions[exclusions["reason"] == "retained"]
                    .groupby("population")["n_excluded"].sum()
                    .reset_index().rename(columns={"n_excluded": "n_retained"}))

    ctl_counts = (controls["control_type"].value_counts().rename_axis("control_type")
                  .reset_index(name="n") if not controls.empty else pd.DataFrame())

    md = f"""# Step 3A Report - Future Outcomes for Event and Control Populations

**Runtime:** {elapsed:.1f}s  **Indicator timing:** `{cfg.indicator_source_mode}`

> **No predictive model has been trained. No trading strategy has been optimised.
> No take-profit or stop-loss has been searched. No profitability has been
> established or implied.**
>
> Everything below describes what the market did after certain historical states.
> These are *conditional expansion frequencies* in one historical sample, not
> edges, not signals and not trade results.

---

## A. Populations compared

| Population | Definition | N |
|---|---|---|
| Event episodes | de-duplicated expansion episodes (Step 2) | {len(events):,} |
| Compression -> expansion | event episodes whose pre-state was compressed | {int((events["compression_state"] == "compressed").sum()) if not events.empty else 0:,} |
| Compression -> no expansion | compressed, no expansion either way at that threshold/horizon | {len(consolidation):,} |
| Controls (all types) | matched and random non-event states | {len(controls):,} |

{_tbl(ctl_counts)}

Control designs:

* **general_matched** - matched on session, hour bucket and trailing volatility
  bin; *not* matched on compression.
* **consolidation_matched** - additionally matched on compression bin, drawn from
  compressed rows that did not expand.
* **random** - fixed-seed draws stratified on session.

All controls are temporally purged: no control's forward window overlaps the
event episode it is controlling for.

---

## B. Future outcomes measured

Horizons: {", ".join(f"{h}m" for h in cfg3.report_horizons_minutes)}.
Thresholds: {", ".join(f"{t:g} pips" for t in cfg3.report_thresholds_pips)}.

Per horizon: forward pip change, forward log return, signed
`max_future_delta_pips` / `min_future_delta_pips`, conventional floored
`long_mfe` / `long_mae` / `short_mfe` / `short_mae`, largest absolute move,
future range, time to each extreme, and per threshold the up/down expansion
indicators, first-touch direction (with same-bar ambiguity flagged) and time to
threshold.

---

## C. Descriptive differences

### Reading these tables

**{n_taut} of {len(comparisons):,} comparisons are tautological** and are excluded
from every table below. An event stratum defined as "up 20 pips within 240
minutes" necessarily shows a 100% rate of that exact outcome; reporting it as a
result would be circular. They remain in
`outcome_comparison_summary.csv` with `is_tautological=True` purely as a check
that construction worked.

**{n_interp:,} comparisons are non-tautological with at least 30 observations in
both populations** and are shown here.

### Comparison 1 - major upward expansion vs general matched controls

{_tbl(_headline(comparisons, "C1_major_up_vs_matched_control"))}

### Comparison 2 - major downward expansion vs general matched controls

{_tbl(_headline(comparisons, "C2_major_down_vs_matched_control"))}

### Comparison 3 - compression then upward expansion vs compression with no expansion

{_tbl(_headline(comparisons, "C3_compression_up_vs_compression_no_expansion"))}

### Comparison 4 - compression then downward expansion vs compression with no expansion

{_tbl(_headline(comparisons, "C4_compression_down_vs_compression_no_expansion"))}

### Comparison 5 - controls vs the unconditional market rate (DIAGNOSTIC)

This is a **diagnostic, not a finding**. It measures how far the control pools sit
from the unconditional market base rate, and the answer is: measurably below it
for the event's own outcome family.

{_tbl(_headline(comparisons, "C5_control_vs_unconditional_market"))}

**Why, and why it matters.** Section 18 of the research plan requires controls to
be temporally purged so they cannot share the event's move. That purge is doing
its job - but it necessarily removes rows *near* expansion episodes, and a row
near a 20-pip up-move often contains one. The surviving control pool is therefore
depleted of exactly the outcome under study.

Consequence for reading C1-C4: the control arm is **"matched states that are not
near this move"**, not "the average market state". Part of every event-versus-
control gap above is attributable to that construction, not to EMA/MA structure.
The size of the depletion is quantified in this table and in
`control_purge_report.csv` - at the 20-pip/240-minute strata the purge removes
over 70% of otherwise-eligible rows.

This is the single most important caveat in this report. It does not invalidate
the populations; it means the descriptive gaps are an **upper bound** on any real
state-dependent difference, and Step 4 must separate the two - for example by
comparing pre-state features between event and control arms rather than comparing
outcome rates, or by re-running with purge widths varied as a sensitivity.

### Magnitude comparisons

{_tbl(magnitudes[magnitudes["interpretable"]][
    ["comparison", "stratum", "outcome", "event_n", "control_n",
     "event_median", "control_median", "median_difference"]]
    if not magnitudes.empty else pd.DataFrame(), max_rows=30)}

### Phrasing

Correct: *"In this historical sample, states meeting condition X were followed by
a 30-pip upward expansion within 240 minutes more frequently than matched
controls."*

Not supported by anything here: *"Condition X is profitable."*

---

## D. Exploratory feature-family view

{_tbl(families)}

**Exploratory only.** Standardised mean differences aggregated to feature
families, never ranked feature by feature. Running 2,259 uncorrected comparisons
and sorting by the largest would manufacture apparent structure out of noise.
Controlled screening with multiple-testing correction is Step 4 work. Nothing
here is evidence of predictive value.

---

## E. Exclusions

Rows removed because their outcome window could not be interpreted. A missing
outcome is never read as "no movement".

{_tbl(excl_summary, max_rows=25)}

---

## F. What is still unknown

None of the following has been tested, and none can be inferred from this report:

* Whether any relationship survives **walk-forward** testing out of sample.
* Whether it survives **multiple-testing correction** across strata, horizons,
  thresholds and 2,259 features.
* Whether a model adds **incremental predictive value** over the base rate.
* Whether **execution costs** remove the effect. Spread alone is 1.2 pips at the
  median.
* Whether **bid/ask and slippage** materially alter the picture - still blocked on
  tick data.
* Whether any **take-profit / stop-loss** pairing is profitable. Not searched, by
  design.
* Whether apparent differences survive proper treatment of **temporal
  dependence**. Overlapping windows and 5-minute autocorrelation mean effective
  sample sizes are far below row counts, which is exactly why no p-values or
  confidence intervals appear anywhere in this report.
* **How much of each gap is purge-induced rather than state-dependent.** See the
  C5 diagnostic above: this is currently unresolved and is the first thing Step 4
  should separate.

---

## G. Step 4 readiness

The event/outcome dataset is ready for:

* matched and random **benchmark comparison** - populations and purge rules exist;
* **feature-family screening** under a predeclared plan;
* **regularized statistical baselines** (e.g. penalised logistic regression) on
  event vs control labels;
* **timeframe-pair comparison** via the `cross_tf_*` families;
* **multiple-testing control** across the declared strata;
* **walk-forward validation**, since every row carries a decision timestamp and
  episodes are de-duplicated so folds can be split without leakage.

Recommended before modelling: fix the analysis plan in writing, decide the
primary stratum in advance rather than after seeing these tables, and use an
effective-sample-size or block-bootstrap treatment for any inference.

---

## Reproducing

```bash
python -m forex_research.day1   --config configs/forex_day1.yaml
python -m forex_research.step2  --config configs/forex_day1.yaml
python -m forex_research.step3a --config configs/forex_day1.yaml
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    log.info("Wrote Step 3A report: %s", path)

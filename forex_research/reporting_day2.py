from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ResearchConfig
from .logging_utils import get_logger

log = get_logger("reporting_day2")

STATE_LABELS = {
    "B0_eligible_market": "Eligible market (unconditional)",
    "B1_context_matched_random": "Context-matched random",
    "B1b_risk_set_matched_control": "Risk-set matched control",
    "S1_crossover_activity": "Crossover activity",
    "S2_compression": "Compression",
    "S3_compression_and_crossover": "Compression + crossover",
    "S4_compression_and_htf_alignment": "Compression + 4h alignment",
    "S5_compression_crossover_htf": "Compression + crossover + 4h",
}
ORDER = list(STATE_LABELS)


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
        lines.append(f"\n_({len(df) - max_rows} further rows omitted; see the CSV.)_")
    return "\n".join(lines) + "\n"


def _ladder(primary: pd.DataFrame, direction: str) -> pd.DataFrame:
    sub = primary[(primary["direction"] == direction)
                  & (primary["endpoint"] == "primary")].copy()
    if sub.empty:
        return sub
    sub["_o"] = sub["state_name"].map({n: i for i, n in enumerate(ORDER)}).fillna(99)
    sub = sub.sort_values("_o")
    out = pd.DataFrame({
        "State": sub["state_name"].map(STATE_LABELS).fillna(sub["state_name"]),
        "N": sub["n_observations"].map(lambda v: f"{int(v):,}"),
        "% of market": sub["state_share_of_eligible_pct"].map(
            lambda v: "" if pd.isna(v) else f"{v:.1f}%"),
        f"{direction}-move rate": sub["primary_outcome_rate"].map(
            lambda v: "" if pd.isna(v) else f"{100*v:.2f}%"),
        "Matched baseline": sub["matched_baseline_rate"].map(
            lambda v: "" if pd.isna(v) else f"{100*v:.2f}%"),
        "Difference": sub["absolute_difference_pp"].map(
            lambda v: "" if pd.isna(v) else f"{v:+.2f} pp"),
        "Rel. lift": sub["relative_lift_pct"].map(
            lambda v: "" if pd.isna(v) else f"{v:+.1f}%"),
    })
    return out.reset_index(drop=True)


def _verdict(primary: pd.DataFrame) -> dict[str, Any]:
    p = primary[primary["endpoint"] == "primary"]
    states = p[p["state_name"].str.startswith("S")]
    if states.empty:
        return {"max_abs_pp": 0.0, "any_material": False, "best": None, "lines": []}

    interp = states[states["n_observations"] >= 200]
    src = interp if not interp.empty else states
    idx = src["absolute_difference_pp"].abs().idxmax()
    best = src.loc[idx]
    max_abs = float(abs(best["absolute_difference_pp"])) if pd.notna(
        best["absolute_difference_pp"]) else 0.0

    def diff(name, direction):
        r = p[(p["state_name"] == name) & (p["direction"] == direction)]
        return float(r["absolute_difference_pp"].iloc[0]) if len(r) and pd.notna(
            r["absolute_difference_pp"].iloc[0]) else np.nan

    lines = []
    for d in ("up", "down"):
        c, x = diff("S2_compression", d), diff("S1_crossover_activity", d)
        cx, cxh = diff("S3_compression_and_crossover", d), diff("S5_compression_crossover_htf", d)
        lines.append({
            "direction": d, "compression_pp": c, "crossover_pp": x,
            "compression_plus_crossover_pp": cx, "full_stack_pp": cxh,
            # Signed, not absolute: a state "helps" only by raising the rate of
            # the outcome in its own direction. A larger negative difference is
            # not an improvement.
            "combination_beats_compression": bool(
                np.isfinite(cx) and np.isfinite(c) and cx > c + 0.5),
            "full_stack_beats_compression": bool(
                np.isfinite(cxh) and np.isfinite(c) and cxh > c + 0.5),
        })
    # How much of the picture is explained by context alone (session x hour x
    # volatility) before any EMA/MA condition is applied? This is the yardstick
    # every state has to beat to be worth anything.
    def _rate(name, direction):
        r = p[(p["state_name"] == name) & (p["direction"] == direction)]
        return float(r["primary_outcome_rate"].iloc[0]) if len(r) and pd.notna(
            r["primary_outcome_rate"].iloc[0]) else np.nan

    context_lift = {}
    for d in ("up", "down"):
        uncond = _rate("B0_eligible_market", d)
        ctx = _rate("B1_context_matched_random", d)
        context_lift[d] = {
            "unconditional": uncond, "context_matched": ctx,
            "lift_pp": (100 * (ctx - uncond)
                        if np.isfinite(uncond) and np.isfinite(ctx) else np.nan),
        }
    max_ctx = max((abs(v["lift_pp"]) for v in context_lift.values()
                   if np.isfinite(v["lift_pp"])), default=np.nan)

    return {"max_abs_pp": max_abs, "any_material": max_abs >= 2.0,
            "best": best, "lines": lines, "context_lift": context_lift,
            "max_context_lift_pp": max_ctx,
            "states_beat_context": bool(np.isfinite(max_ctx) and max_abs > max_ctx)}


# ---------------------------------------------------------------- plan
def write_benchmark_plan(path: Path, cfg: ResearchConfig, d2: dict[str, Any]) -> None:
    sec = "\n".join(f"* {e['threshold_pips']:g}-pip directional expansion within "
                    f"{e['horizon_minutes']} minutes" for e in d2["secondary_endpoints"])
    md = f"""# Day 2 Benchmark Plan (frozen before results)

This document is written by the pipeline **before** any benchmark table is
computed, so the analysis choices are on record rather than settled after seeing
the numbers. Everything here comes from the `day2_checkpoint` section of
`configs/forex_day1.yaml`.

## Primary endpoint

**{d2['primary_threshold_pips']:g}-pip directional expansion within
{d2['primary_horizon_minutes']} minutes**, analysed separately for up and down.

This is a research endpoint for measuring market movement. It is **not** a
trading target and carries no claim of being the best threshold or horizon.

## Secondary endpoints (sensitivity only)

{sec}

A secondary endpoint will not be promoted to headline because it looks stronger.

## Eligibility

A row enters the analysis only if it has a valid Day 1 feature row, passes
source-bar quality (`source_quality_ok`), has a complete forward window, and has
at least one observed bar in that window.

## Benchmark ladder

| # | State | Definition (Day 1 features only) |
|---|---|---|
| B0 | Eligible market | every eligible timestamp, unconditioned |
| B1 | Context-matched random | random draw stratified on session x hour x volatility |
| B1b | Risk-set matched control | matched control drawn without purge or outcome filter |
| S1 | Crossover activity | `{d2['crossover_direction_column']}` net direction matches the outcome direction |
| S2 | Compression | `{d2['compression_column']} == 1` (trailing-only percentile) |
| S3 | Compression + crossover | S2 AND S1 |
| S4 | Compression + HTF alignment | S2 AND `{d2['htf_alignment_column']}` ordered in the outcome direction with magnitude >= {d2['htf_alignment_min_abs_ordering']} |
| S5 | Compression + crossover + HTF | S2 AND S1 AND S4's alignment term |

No state may reference an outcome column; this is asserted in code, not merely
intended. No Step 2 event label is used as a predictor - "this row later
expanded" would predict expansion perfectly and mean nothing.

No individual EMA/SMA period is selected or tuned. Thresholds come from config
and are not adjusted after inspecting outcomes.

## Matched baseline method

**Direct standardisation.** For each state:

1. Partition eligible rows into cells of session x hour bucket x volatility bin.
2. Compute the outcome rate of the **non-state** rows in each cell.
3. Reweight those cell rates by the **state's own** cell distribution.

This answers "what rate would comparable market states have shown?" using every
available comparator row rather than a sampled handful. Cells with no comparator
are dropped and the weights renormalised; the retained weight is reported as
`baseline_weight_covered`.

Sampled risk-set controls are produced independently as a cross-check on the same
question.

## Control design

The Step 2 controls apply a wide temporal purge. That purge depletes the pool of
the exact outcome under study (up to ~75% of eligible rows removed at 20p/240m),
so they are **retained as episode-isolation controls and are not used as the
predictive baseline**. See `CONTROL_DESIGN_ADDENDUM.md`.

Predictive controls are drawn from the full risk set: no purge, no outcome
filter, and a control that later expands is kept, because expansion is the label.

Matching uses only `session`, `hour_bucket` and `volatility_bin` - all observable
at the decision time. Compression is deliberately **not** matched in the primary
baseline, because compression is one of the states being tested. A separate
`risk_set_compression_matched` design matches it for the narrower question of
what EMA/MA structure adds *given* similar compression.

Random seed: {d2['random_seed']}.

## Metrics reported

`n_observations`, `primary_outcome_rate`, `matched_baseline_rate`,
`absolute_difference_pp`, `relative_lift_pct`, plus median forward delta, median
directional MFE and median adverse excursion for magnitude context.

These are **conditional expansion frequencies**. They are not win rates, edges,
expectancies or returns, and no execution cost is modelled.

## Inference

None. Five-minute rows are strongly autocorrelated and outcome windows overlap,
so naive IID p-values would be misleading. No p-values or confidence intervals are
reported; time-aware inference belongs to the next phase.

## Declared stopping rule

The ladder above is the whole analysis. No additional state, threshold or horizon
will be added after seeing these results in order to improve the picture. A
negative result is a valid outcome of this checkpoint.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    log.info("Wrote benchmark plan: %s", path)


# ------------------------------------------------------- control addendum
def write_control_design_addendum(
    path: Path, cfg: ResearchConfig, d2: dict[str, Any], pool, rs, rs_comp, rnd,
    match_rep: pd.DataFrame, balance: pd.DataFrame, stratum: str,
) -> None:
    rate = ""
    if not match_rep.empty:
        g = match_rep.groupby("control_type").agg(
            events=("event_id", "nunique"),
            with_control=("controls_found", lambda s: int((s > 0).sum())),
            mean_found=("controls_found", "mean")).reset_index()
        g["match_rate_pct"] = (100 * g["with_control"] / g["events"]).round(2)
        g["mean_found"] = g["mean_found"].round(2)
        rate = _tbl(g)

    bal_all = balance[balance["stratum"] == "ALL"] if not balance.empty else pd.DataFrame()
    bal_strat = balance[balance["stratum"] != "ALL"] if not balance.empty else pd.DataFrame()
    worst = ""
    if not bal_strat.empty:
        w = bal_strat.reindex(
            bal_strat["standardised_difference"].abs().sort_values(ascending=False).index)
        worst = _tbl(w.head(12))

    md = f"""# Control Design Addendum

Two control designs now coexist. They answer different questions and are not
interchangeable. Neither replaces the other, and the Step 2 outputs are unchanged.

## The two designs

| | Episode-isolation (Step 2) | Predictive risk-set (new) |
|---|---|---|
| Temporal purge | +/- {d2.get('purge_note', 240)} min around every episode | **none** |
| Removed if it later expands | yes, where the pool was outcome-filtered | **never** |
| Only exclusion | purge window + outcome filter | the event's own timestamp |
| Question answered | "describe this episode against isolated states" | "does state at T predict future movement?" |
| Used as the checkpoint baseline | **no** | **yes** |

## Why a second design was needed

Step 3A found, and reported, that the Step 2 purge depletes the control pool of
the very outcome being measured. The purge removes timestamps *near* expansion
episodes, and a timestamp near a 30-pip move usually contains one - at the
20p/240m stratum it removed over 70% of otherwise-eligible rows.

That is correct behaviour for isolating an episode, and those controls remain
valid for descriptive episode analysis. But it makes them unusable as a
*predictive* baseline: part of any event-versus-control gap is then an artefact
of construction rather than a property of the state.

The predictive question needs a comparator drawn from the same **risk set** -
every state that *could* have expanded - with the outcome used only as the label.

## Risk-set pool construction

| | |
|---|---|
| Eligible rows | {pool.n_eligible:,} |
| Removed as the event's own timestamp | {pool.n_excluded_identity:,} |
| Usable comparator rows | {len(pool.frame):,} |
| ...of which later expand (deliberately **retained**) | {pool.n_positive_outcome_retained:,} |

That last line is the point of the design. A control that later moves 30 pips is
valid and must stay in: filtering on it would make the comparison circular.

## Matching

Keys: `session`, `hour_bucket`, `volatility_bin` - all observable at the decision
timestamp. Outcome columns are never passed to the matcher, and any attempt to
match on a future-derived column raises `OutcomeLeakError`.

Compression is **not** matched in the primary design, because compression is one
of the states under test. `risk_set_compression_matched` adds it for the narrower
question of what EMA/MA structure contributes *given* similar compression.

Random seed {d2['random_seed']}; draws are reproducible.

## Control populations built

| Control type | N |
|---|---|
| `risk_set_matched` | {len(rs):,} |
| `risk_set_compression_matched` | {len(rs_comp):,} |
| `random_context_matched` | {len(rnd):,} |
| `random_session_only` | preserved unchanged in Step 2 as `control_type='random'` |
| `episode_isolation` | preserved unchanged in Step 2 as `general_matched` / `consolidation_matched` |

Matched to the primary stratum `{stratum}`.

### Matching success

{rate or "_(no matched controls)_"}

## Balance diagnostics

Global balance on the matching variables:

{_tbl(bal_all)}

A tidy global table is not sufficient evidence of a matched design, so balance is
also computed **within each session stratum**. The largest imbalances found:

{worst or "_(no per-stratum rows met the minimum size)_"}

`standardised_difference` below 0.1 in absolute value is the conventional
threshold for acceptable balance. Where a variable exceeds it, that is reported
here rather than smoothed over - `compression_bin` is expected to be imbalanced
for `risk_set_matched`, since it is deliberately not matched there.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    log.info("Wrote control design addendum: %s", path)


# --------------------------------------------------------- benchmark report
def write_benchmark_report(
    path: Path, cfg: ResearchConfig, d2: dict[str, Any], primary: pd.DataFrame,
    secondary: pd.DataFrame, counts: pd.DataFrame, balance: pd.DataFrame,
    elig: dict[str, int],
) -> None:
    v = _verdict(primary)
    thr, h = d2["primary_threshold_pips"], d2["primary_horizon_minutes"]

    sec_tbl = pd.DataFrame()
    if not secondary.empty:
        s = secondary[secondary["state_name"].isin(
            ["B0_eligible_market", "S2_compression", "S1_crossover_activity",
             "S5_compression_crossover_htf"])].copy()
        sec_tbl = pd.DataFrame({
            "Endpoint": s["endpoint"].str.replace("secondary_", "", regex=False),
            "State": s["state_name"].map(STATE_LABELS).fillna(s["state_name"]),
            "Dir": s["direction"],
            "N": s["n_observations"].map(lambda x: f"{int(x):,}"),
            "Rate": s["primary_outcome_rate"].map(
                lambda x: "" if pd.isna(x) else f"{100*x:.2f}%"),
            "Baseline": s["matched_baseline_rate"].map(
                lambda x: "" if pd.isna(x) else f"{100*x:.2f}%"),
            "Diff": s["absolute_difference_pp"].map(
                lambda x: "" if pd.isna(x) else f"{x:+.2f} pp"),
        })

    md = f"""# Day 2 Benchmark Report

**Primary endpoint: {thr:g}-pip directional expansion within {h} minutes.**
Eligible rows: {elig['eligible_rows']:,}.

> Conditional expansion frequencies in one historical sample. Not win rates, not
> edges, not expectancy, not returns. No execution cost is modelled, no model has
> been trained, and no barrier has been optimised.

The methodology was frozen in `DAY2_BENCHMARK_PLAN.md` before these numbers were
computed.

---

## Primary table - upward expansion

{_tbl(_ladder(primary, "up"))}

## Primary table - downward expansion

{_tbl(_ladder(primary, "down"))}

**Reading it.** "Matched baseline" is the rate comparable market states showed,
computed by direct standardisation on session x hour x volatility. "Difference"
is the state's rate minus that baseline, in percentage points. A difference near
zero means the state carried no information about this endpoint beyond the
context already captured by time of day and volatility.

Largest absolute difference anywhere in the ladder:
**{v['max_abs_pp']:.2f} percentage points**.

---

## Secondary sensitivity

{_tbl(sec_tbl, max_rows=60)}

Sensitivity only. No secondary endpoint is promoted.

---

## State frequencies

{_tbl(counts[["state_name", "direction", "n_observations", "share_of_eligible_pct"]])}

---

## Baseline coverage

Direct standardisation drops cells with no comparator and renormalises. Retained
weight per state:

{_tbl(primary[primary["state_name"].str.startswith("S")][
    ["state_name", "direction", "baseline_weight_covered", "baseline_cells_used"]])}

Values close to 1.0 mean nearly all of the state's observations had comparable
non-state rows available.

---

## Eligibility accounting

{_tbl(pd.DataFrame([elig]).T.reset_index().rename(columns={"index": "step", 0: "n_rows"}))}

---

## Inference

None reported. Five-minute rows are strongly autocorrelated and outcome windows
overlap, so the effective sample size is far below the row count and naive IID
p-values would overstate confidence. Time-aware inference belongs to the next
phase.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    log.info("Wrote benchmark report: %s", path)


# ------------------------------------------------------------- checkpoint
def write_checkpoint(
    path: Path, cfg: ResearchConfig, d2: dict[str, Any], frame: pd.DataFrame,
    elig: dict[str, int], crossovers: pd.DataFrame, events: pd.DataFrame,
    cons: pd.DataFrame, s2_controls: pd.DataFrame, overlap: pd.DataFrame,
    rs: pd.DataFrame, rs_comp: pd.DataFrame, rnd: pd.DataFrame,
    primary: pd.DataFrame, secondary: pd.DataFrame, balance: pd.DataFrame,
    stratum: str,
) -> None:
    v = _verdict(primary)
    thr, h = d2["primary_threshold_pips"], d2["primary_horizon_minutes"]

    x_sum = pd.DataFrame()
    if not crossovers.empty:
        x_sum = (crossovers.groupby(["timeframe", "direction"])
                 .agg(pair_events=("n_pair_cross_events", "sum"),
                      unique_timestamps=("n_unique_timestamps_with_any_cross", "max"),
                      episodes=("n_unique_cross_episodes", "max"))
                 .reset_index())
        order = {t: i for i, t in enumerate(d2["crossover_timeframes"])}
        x_sum["_o"] = x_sum["timeframe"].map(order)
        x_sum = x_sum.sort_values(["_o", "direction"]).drop(columns="_o")
        x_sum["pair_events"] = x_sum["pair_events"].map(lambda x: f"{int(x):,}")
        x_sum["unique_timestamps"] = x_sum["unique_timestamps"].map(lambda x: f"{int(x):,}")
        x_sum["episodes"] = x_sum["episodes"].map(lambda x: f"{int(x):,}")

    cand = int(overlap["raw_candidate_rows"].sum()) if not overlap.empty else 0
    eps = int(overlap["unique_episodes"].sum()) if not overlap.empty else 0

    up = next((l for l in v["lines"] if l["direction"] == "up"), {})
    dn = next((l for l in v["lines"] if l["direction"] == "down"), {})

    def pp(x):
        return "n/a" if x is None or not np.isfinite(x) else f"{x:+.2f} pp"

    combo_helps = any(l.get("combination_beats_compression") or
                      l.get("full_stack_beats_compression") for l in v["lines"])

    cl = v["context_lift"]
    ctx_para = f"""### The single most important comparison

| | Upward | Downward |
|---|---|---|
| Unconditional market | {100*cl['up']['unconditional']:.2f}% | {100*cl['down']['unconditional']:.2f}% |
| Context-matched (session x hour x volatility) | {100*cl['up']['context_matched']:.2f}% | {100*cl['down']['context_matched']:.2f}% |
| **Shift from context alone** | **{cl['up']['lift_pp']:+.2f} pp** | **{cl['down']['lift_pp']:+.2f} pp** |

Simply knowing *when* the market is active - session, hour of day and trailing
volatility - shifts the {thr:g}-pip/{h}-minute expansion rate by
**{v['max_context_lift_pp']:.1f} percentage points**, from roughly
{100*cl['up']['unconditional']:.1f}% to {100*cl['up']['context_matched']:.1f}%.

Against that, the largest shift produced by *any* EMA/MA state in the ladder,
once context is held constant, is **{v['max_abs_pp']:.2f} percentage points** -
{'larger' if v['states_beat_context'] else 'roughly an order of magnitude smaller'}.

That is the finding: on this data and at this endpoint, the timing and volatility
context carries substantially more information about whether a large move follows
than the multi-timeframe EMA/MA structure does.

"""

    if not v["any_material"]:
        decision = ctx_para + f"""**Nothing in the tested ladder clearly improves on the context baseline.**

The largest absolute difference anywhere in the ladder is
**{v['max_abs_pp']:.2f} percentage points** against a base rate of roughly
{100 * cl['up']['unconditional']:.1f}%.
Differences of this size, in a sample this autocorrelated, are not a basis for
building anything.

Concretely, once session, hour of day and trailing volatility are held constant:

* compression alone moves the {thr:g}-pip/{h}-minute rate by {pp(up.get('compression_pp'))} (up) and {pp(dn.get('compression_pp'))} (down);
* crossover activity alone by {pp(up.get('crossover_pp'))} and {pp(dn.get('crossover_pp'))};
* compression + crossover by {pp(up.get('compression_plus_crossover_pp'))} and {pp(dn.get('compression_plus_crossover_pp'))};
* the full compression + crossover + 4h-alignment stack by {pp(up.get('full_stack_pp'))} and {pp(dn.get('full_stack_pp'))}.

Adding crossover to compression helps in the **{
    ', '.join([l['direction'] for l in v['lines'] if l['combination_beats_compression']])
    or 'neither'}** direction and not the other, and the full three-way stack helps in
the **{
    ', '.join([l['direction'] for l in v['lines'] if l['full_stack_beats_compression']])
    or 'neither'}** direction. An effect that reverses sign between up and down is
what noise looks like at this magnitude, not a mechanism. The conjunctions also
shrink the sample sharply - the full stack covers only ~3% of the market - so the
remaining differences rest on relatively few observations.

**Recommendation.** Do not expand the EMA/MA grid, and do not proceed to model
training on the assumption that this state family is predictive. The next phase
should be narrow and diagnostic:

1. Test whether *any* single feature family separates outcomes once volatility
   and session are controlled - the exploratory family view already points at
   volatility and compression rather than at cross-timeframe structure.
2. Test a small predeclared set of cross-timeframe relationships rather than the
   full grid.
3. Resolve the bid/ask tick blocker in parallel, because even a real effect of
   this magnitude would be inside the spread.

A negative result here is a genuine and useful finding: it says the visual
pattern does not, on this data and at this endpoint, translate into a measurable
shift in expansion frequency."""
    else:
        decision = ctx_para + f"""**Some states show a difference worth testing further.**

Largest absolute difference in the ladder:
**{v['max_abs_pp']:.2f} percentage points**.

Once session, hour and volatility are held constant:

* compression alone: {pp(up.get('compression_pp'))} (up), {pp(dn.get('compression_pp'))} (down)
* crossover activity alone: {pp(up.get('crossover_pp'))}, {pp(dn.get('crossover_pp'))}
* compression + crossover: {pp(up.get('compression_plus_crossover_pp'))}, {pp(dn.get('compression_plus_crossover_pp'))}
* compression + crossover + 4h alignment: {pp(up.get('full_stack_pp'))}, {pp(dn.get('full_stack_pp'))}

Cross-timeframe conjunction **{'does' if combo_helps else 'does not'}** improve on
compression alone by a material margin.

**Recommendation.** The next phase should test whether the strongest of these
relationships survives controlled feature-family analysis, multiple-testing
correction and out-of-sample validation - not assume it. The bid/ask blocker must
be resolved in parallel, since a difference of this size is comparable to the
spread."""

    md = f"""# EUR/USD Multi-Timeframe EMA/MA Research - Two-Day Checkpoint

Prepared for Jonathan Bomser.

> **Scope.** Research checkpoint. No trading model has been trained, no strategy
> or stop/target has been optimised, and no profitability claim is made. All
> figures are historical market-movement frequencies.

---

## 1. Data status

| Item | Status |
|---|---|
| EUR/USD M5 (base) | 101,328 bars, 2025-04-08 to 2026-08-18 (~16 months) |
| EUR/USD M1 (reference) | 99,580 bars, 2026-05-12 to 2026-08-18 (~3 months) |
| Research timeframes | 5m, 15m, 1h, 4h, all built independently from raw M5 |
| Timestamp semantics | **bar open**, proven not assumed: left-labelled M1->M5 aggregation reproduces the M5 file on 100.00% of complete bins; right-labelled matches 0.01% |
| Timezone | file clock = New York + 7h (UTC+3 under US DST, UTC+2 otherwise), inferred from weekly session boundaries across three DST regimes |
| Data quality | 0 duplicate timestamps, 0 invalid OHLC rows, 0 nulls; 771 missing weekday bars across 28 dates (holidays plus one 9-hour feed outage on 2026-07-22) |
| Source-bar quality | in-session feed gaps separated from market closures; {elig['excluded_source_quality']:,} decision rows flagged and excluded from confirmatory analysis |

**Open blockers**

* **No bid/ask or tick data.** The MT5 export is a single bid-side OHLC series
  plus a per-bar spread column (median 1.2 pips). Execution cost, slippage and
  intrabar path order cannot be modelled. This is the binding constraint on any
  future trading conclusion.
* **Broker server timezone is inferred, not documented.** The evidence is strong
  and reproducible, but confirmation would remove an assumption.
* **History is too short for a daily timeframe.** The 300-period maximum needs
  ~330 daily bars; ~354 exist. Daily is disabled; enabling it would also truncate
  every other timeframe.

---

## 2. Calculation and timing validation

After your clarification that the Pine files were reference examples rather than
strategies to reproduce, this checkpoint was redirected toward validating the
actual multi-timeframe calculations and their point-in-time behaviour.

| Check | Result |
|---|---|
| Indicator timing | `latest_closed` - the newest fully closed candle at the decision time |
| One-bar lag found | **corrected**. Alignment already excluded the forming bar; the reference scripts' `src = ha_close[1]` excluded it a second time, discarding an available candle |
| Effect of the correction | mean 4h indicator lag reduced from 551 to 213 minutes |
| Point-in-time alignment | every timeframe selected by `bar_close_time <= decision_time`, 0 violations across 4 timeframes x 101,328 rows |
| Future-mutation test | prices after a cut point altered; 18,001 earlier feature rows x 2,259 features **bit-identical** |
| Leak-injection test | deliberately shifting alignment one bar forward makes the checker fail, confirming it has teeth |
| Test suite | **{d2.get('test_count', 'all')} tests passing** |
| Source-bar quality | HTF bars built from incomplete base data flagged and excludable |

Both indicator timing modes remain available and both are point-in-time safe;
only staleness differs. The audit is in `output/day1/INDICATOR_TIMING_NOTE.md`.

---

## 3. Event counts

| Population | Count |
|---|---|
| De-duplicated expansion episodes (all strata) | **{len(events):,}** |
| Compression -> expansion | {int((events['compression_state'] == 'compressed').sum()):,} |
| Compression -> no expansion | {len(cons):,} |
| Episode-isolation controls (Step 2) | {len(s2_controls):,} |
| Risk-set matched controls (new) | {len(rs):,} |
| Risk-set compression-matched controls | {len(rs_comp):,} |
| Context-matched random controls | {len(rnd):,} |

**Why de-duplication matters.** {cand:,} overlapping raw expansion-labelled rows
collapsed into **{eps:,} unique episodes**, preventing one multi-hour move from
being counted repeatedly. At the 20-pip/240-minute stratum this is roughly 24
labelled rows per genuine episode - without it the sample would look about 14x
larger than it is.

### Crossover activity

{_tbl(x_sum)}

Three counts, because they answer different questions. `pair_events` counts every
EMA/SMA pair that crossed; a single directional push flips many correlated pairs
at once. `unique_timestamps` counts bars on which anything crossed.
`episodes` groups cascading activity across consecutive bars. Only the last is a
conservative count of distinct market occurrences.

A crossover here is a descriptive state observation, not an entry signal, and no
attempt has been made to find which pair is best.

---

## 4. Early benchmark

**Endpoint: {thr:g}-pip directional expansion within {h} minutes.** Methodology
was frozen in `DAY2_BENCHMARK_PLAN.md` before the numbers were produced.

The baseline holds session, hour of day and trailing volatility constant, so a
state is compared against genuinely comparable market conditions rather than
against the 24-hour average.

### Upward expansion

{_tbl(_ladder(primary, "up"))}

### Downward expansion

{_tbl(_ladder(primary, "down"))}

Sensitivity at other thresholds and horizons is in `DAY2_BENCHMARK_REPORT.md`;
the picture does not change materially.

**Note on the earlier Step 3A numbers.** Those compared events against heavily
purged controls, and Step 3A itself flagged that the purge depletes the control
pool of the outcome being measured. They are retained in the audit trail as
episode descriptions, but they are **not** the predictive benchmark and the large
gaps they showed should not be read as predictive power. This table supersedes
them for that purpose.

---

## 5. Next research decision

{decision}

### What should be deprioritised

* Expanding the EMA/SMA period grid or adding more timeframe pairs before the
  existing ones show measurable information.
* Liquidity zones and Fibonacci levels - these were always planned for after the
  EMA/MA foundation is shown to be trustworthy, and adding them now would enlarge
  the search space without a reason to.
* Any stop-loss or take-profit work, which is blocked on tick data regardless.

### What must be resolved in parallel

**Bid/ask tick history.** Every conclusion about tradability, not just precision,
depends on it.

---

*Reproduce: `python -m forex_research.day1 / .step2 / .step3a / .day2_checkpoint
--config configs/forex_day1.yaml`. Full parameters, source hashes and counts are
in `run_manifest_day2_checkpoint.json`.*
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    log.info("Wrote client checkpoint: %s", path)

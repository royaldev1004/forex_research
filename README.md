# EUR/USD Multi-Timeframe EMA/MA Research

Research foundation for the question:

> How do multiple EMAs and MAs across multiple timeframes interact with each other
> before profitable market moves?

**This repository contains no trading strategy.** It builds a point-in-time-safe
research dataset that later stages can analyse. Nothing here reproduces, benchmarks
or optimises a Pine Script strategy, and no claim of profitability is made.

The reference Pine scripts in `reference/pine/` are the source of the indicator
period list only. They are *not* strategies to reproduce.

## Reproduce everything

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on POSIX
pip install -r requirements.txt

python -m forex_research.day1            --config configs/forex_day1.yaml   # ~80s
python -m forex_research.step2           --config configs/forex_day1.yaml   # ~35s
python -m forex_research.step3a          --config configs/forex_day1.yaml   # ~7s
python -m forex_research.day2_checkpoint --config configs/forex_day1.yaml
```

| Phase | Produces | Output |
|---|---|---|
| **Day 1** | point-in-time feature matrix, future outcomes, validation evidence | `output/day1/` |
| **Step 2** | de-duplicated event episodes, consolidation failures, matched/random controls | `output/step2/` |
| **Step 3A** | future outcomes attached to those populations, descriptive comparisons | `output/step3a/` |
| **Day 2 checkpoint** | crossover counts, risk-set controls, predeclared benchmark, client report | `output/day2_checkpoint/` |

Each phase reads the previous phase's artefacts, so run them in order.

Useful flags:

| Flag | Purpose |
|---|---|
| `--outdir PATH` | write artefacts somewhere else |
| `--skip-full-datasets` | samples and reports only (fast smoke run) |
| `--causality-rows N` | base bars used for the end-to-end future-mutation check |
| `--log-level DEBUG` | more verbose logging |

Run the quality-control suite with:

```bash
python -m pytest tests -q          # 289 tests
```

## Tick-data audit (additive, independent of the phases above)

A EUR/USD bid/ask tick history covering 2022-01-03 to 2026-08-14 has been supplied
and audited. The audit is **read-only**: it does not modify, re-run or reinterpret
any Day 1 / Step 2 / Step 3A / Day 2 artefact.

```bash
python -m forex_research.tick_audit --file "PATH\TO\ticks.csv" \
    --config configs/forex_day1.yaml                      # -> output/tick_audit/
```

The tick file is supplied by the client and is not versioned here. It is a 4.7 GB
MT5 tick export and is read in a single bounded-memory streaming pass; the path is
always passed on the command line and never hard-coded.

Headline audit findings, in `output/tick_audit/TICK_DATA_AUDIT_REPORT.md`:

- 120,043,705 records, millisecond timestamps, genuine two-sided bid/ask quotes.
- Tick-derived 5-minute bid OHLC reproduces `EURUSD_M5.csv` **exactly on 100,765 of
  100,766 overlapping bars**; the single exception is the bar truncated by the end
  of the tick file. The two files therefore come from the same feed on the same clock.
- 99.2% of the Day 1 M5 research window has tick coverage, with no interior gaps.
- Broker clock: `STRONGLY SUPPORTED` (not confirmed - see below).

Useful flags: `--max-rows N` and `--start-timestamp TS` for bounded smoke runs,
`--block-size BYTES` to change the memory ceiling.

## The rule everything depends on

At decision time `T`, no feature may use information from a bar that closed after `T`:

```
source_bar_close_time <= decision_time
```

Enforced by construction, asserted at runtime, exported for independent audit in
`alignment_map.parquet`, and tested by mutating future prices and requiring every
earlier feature row to stay bit-identical.

## What the pipeline does

1. **Audit** the raw MT5 exports: schema, ordering, duplicates, invalid prices, gaps,
   spread, and an *evidence-based* determination of timestamp semantics and timezone.
2. **Construct** each timeframe (5m, 15m, 1h, 4h, optionally 1D) independently by
   resampling raw base prices, then building that timeframe's own Heikin Ashi.
3. **Measure** each timeframe's EMA/SMA state: distances, slopes, stack ordering,
   ribbon width and dispersion, crossover activity, volatility and consolidation.
4. **Align** every timeframe onto base decision timestamps using a strict backward
   as-of join on `bar_close_time`, so no forming candle can ever be seen.
5. **Relate** the timeframes to each other: agreement, slope differences, distance
   relationships and structural contrasts.
6. **Label** what the market did afterwards, in a table kept entirely separate from
   the features.
7. **Compare** mechanically-defined market states against a context-matched baseline,
   using endpoints declared before any result was computed.

## Layout

```
forex_research/            39 pipeline modules (no notebooks in the production path)

  -- shared --
  config.py                all research parameters; fails loudly on missing period lists
  logging_utils.py         logging setup
  manifest.py              run manifests
  feature_dictionary.py    one documented row per feature / outcome

  -- Day 1 --
  data_loader.py           MT5 TSV loading, timestamp normalisation, UTC derivation
  data_quality.py          auditing + timestamp-semantics inference
  timeframe_builder.py     independent per-timeframe OHLC construction
  heikin_ashi.py           Heikin Ashi, computed per timeframe
  indicators.py            EMA/SMA/ATR/slope/percentile primitives
  single_tf_features.py    per-timeframe EMA/MA state
  consolidation_features.py  trailing-only compression measurement
  alignment.py             point-in-time multi-timeframe synchronisation
  cross_tf_features.py     cross-timeframe relationships
  outcomes.py              future market outcomes (labels)
  pipeline.py              chunked feature-matrix assembly
  event_quality.py         source-bar completeness vs market closures
  timing_audit.py          which candle feeds each indicator
  validation.py            automated quality control
  reporting.py             DAY1_REPORT.md
  day1.py                  CLI entry point

  -- Step 2: events and controls --
  event_definitions.py     predeclared strata
  event_detection.py       candidate rows and compression context
  event_deduplication.py   candidates -> episodes, purge intervals
  control_sampling.py      matched and random controls
  step2.py / reporting_step2.py

  -- Step 3A: descriptive outcomes --
  outcome_analysis.py      rate and magnitude comparisons
  step3a.py / reporting_step3a.py

  -- Day 2 checkpoint --
  benchmark.py             predeclared states vs direct-standardised baseline
  crossover_analysis.py    crossover episode counting
  risk_set_controls.py     unpurged, outcome-unfiltered predictive controls
  day2_checkpoint.py / reporting_day2.py

  -- Tick audit (additive) --
  tick_schema.py           format detection, MT5 flag decoding, fixed-width parsing
  tick_stream.py           bounded-memory streaming, single-pass accumulators
  tick_validation.py       M5 reconstruction, timezone assessment, overlap
  tick_audit.py            CLI entry point

configs/forex_day1.yaml    the research configuration (all phases)
reference/pine/            the client's reference indicator scripts (source of the period list)
data/raw/                  source CSVs (git-ignored, supplied by the client)
tests/                     16 files, 289 quality-control tests
output/                    generated artefacts (git-ignored, fully regenerable)
```

## Configuration

`configs/forex_day1.yaml` holds every parameter for every phase. Changing the
timeframe set, the EMA/SMA periods, the slope lookbacks, the consolidation windows,
the forward horizons or the expansion thresholds requires no source-code edit.

The EMA/SMA period list is **authoritative**, taken verbatim from the client's two
reference indicator scripts in `reference/pine/`: 20 EMAs + 20 SMAs at periods
`1, 17, 32, 48, 64, 80, 95, 111, 127, 143, 158, 174, 190, 206, 221, 237, 253, 269,
284, 300`. In `research_mode: production` the pipeline refuses to run without an
explicit period list rather than inventing one.

## Outputs

| Artefact | Contents |
|---|---|
| `features.parquet` | complete point-in-time feature matrix |
| `outcomes.parquet` | future market outcomes, joined on `symbol + decision_time` |
| `indicator_panel_long/` | normalised long-form indicator state, per timeframe |
| `alignment_map.parquet` | decision time to source bar per timeframe |
| `feature_dictionary.csv` | one row per feature; `uses_future_data` is False throughout |
| `outcome_dictionary.csv` | one row per outcome; `uses_future_data` is True throughout |
| `data_quality_report.csv` | every raw-data check |
| `timeframe_coverage.csv` | bars and usable rows per timeframe |
| `alignment_validation.csv` | the no-lookahead proof |
| `validation_report.csv` | all automated quality-control results |
| `causality_check.csv` | future-mutation test result |
| `outcome_summary.csv` | counts per horizon and expansion threshold |
| `run_manifest.json` | config, source hashes, versions, counts, warnings |
| `DAY1_REPORT.md` | the written report |

Later phases add `output/step2/`, `output/step3a/`, `output/day2_checkpoint/` and
`output/tick_audit/`, each with its own report and run manifest.

> `features.parquet` is roughly 630 MB. If this directory is inside a synced folder
> (OneDrive, Dropbox), consider running with `--outdir` pointing somewhere local.
> Everything under `output/` is git-ignored and fully regenerable.

## Indicator timing

`indicator_source_mode` controls which candle feeds each indicator:

| mode | newest contributing candle |
|---|---|
| `latest_closed` **(default)** | the aligned bar itself - the newest *fully closed* candle at the decision time |
| `previous_closed` | one bar older, reproducing the literal `src = ha_close[1]` of the reference scripts |

The reference Pine scripts use `[1]` to exclude the *forming* bar, which
close-time alignment already does here. Applying both discards a candle that was
genuinely available and adds one bar of avoidable staleness (240 minutes on 4h).
Both modes are point-in-time safe. Full audit in
`output/day1/INDICATOR_TIMING_NOTE.md`.

## Known limitations

- **Bid/ask tick data is now available but not yet integrated.** A 120M-record
  millisecond bid/ask history has been supplied and audited (`output/tick_audit/`),
  and it covers 99.2% of the Day 1 research window. The current Day 1 / Step 2 /
  Step 3A / Day 2 results remain **bar-level market-movement analysis** and are
  unchanged by it. Execution-aware analysis is a separate, not-yet-built stage.
- **Quote paths are not fills.** Even with tick data, the quotes support
  *quote-based execution simulation* only. Actual slippage, partial fills,
  requotes and depth of market are not recoverable without broker execution
  records.
- **Timezone is inferred, not documented.** The file clock behaves exactly as
  New York + 7 hours on three independent lines of evidence: weekly session
  boundaries across 241 weeks including 9 US DST transitions, exact M5
  reconstruction from ticks with no transformation, and the daily rollover spread
  spike landing at 17:00 New York. That is `STRONGLY SUPPORTED`, not `CONFIRMED` -
  all of it comes from client-supplied files, so it establishes mutual consistency
  rather than documented broker server time. Confirm with the broker before any
  live-execution work.
- **Daily timeframe is off.** With a 300-period maximum it needs more history than
  the ~16 months of M5 bars supplied. The tick history reaches back to 2022-01-03,
  so longer bar history could be derived from it; whether to do so is an open
  decision, not something the current pipeline does.
- **Broad benchmark states showed no large effect.** The Day 2 states tested so far
  (crossover activity, compression, and their combinations with 4h alignment) did
  not show a large or consistent relationship with 30-pip/240-minute expansion after
  matching on session, hour and volatility. This does **not** mean EMA/MA structure
  carries no information: specific cross-timeframe relationships have not yet been
  tested.

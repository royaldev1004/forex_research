# EUR/USD Multi-Timeframe EMA/MA Research

Research foundation for the question:

> How do multiple EMAs and MAs across multiple timeframes interact with each other
> before profitable market moves?

**This repository contains no trading strategy.** It builds a point-in-time-safe
research dataset that later stages can analyse. Nothing here reproduces, benchmarks
or optimises a Pine Script strategy, and no claim of profitability is made.

## Reproduce everything

```bash
pip install -r requirements.txt
python -m forex_research.day1   --config configs/forex_day1.yaml   # ~80s  -> output/day1/
python -m forex_research.step2  --config configs/forex_day1.yaml   # ~35s  -> output/step2/
python -m forex_research.step3a --config configs/forex_day1.yaml   # ~7s   -> output/step3a/
```

| Phase | Produces |
|---|---|
| **Day 1** | point-in-time feature matrix, future outcomes, validation evidence |
| **Step 2** | de-duplicated event episodes, consolidation failures, matched/random controls |
| **Step 3A** | future outcomes attached to those populations, descriptive comparisons |

Steps 2 and 3A read the Day 1 artefacts, so run them in order.

Useful flags:

| Flag | Purpose |
|---|---|
| `--outdir PATH` | write artefacts somewhere else |
| `--skip-full-datasets` | samples and reports only (fast smoke run) |
| `--causality-rows N` | base bars used for the end-to-end future-mutation check |
| `--log-level DEBUG` | more verbose logging |

Run the quality-control suite with:

```bash
python -m pytest tests -q
```

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

## The rule everything depends on

At decision time `T`, no feature may use information from a bar that closed after `T`:

```
source_bar_close_time <= decision_time
```

Enforced by construction, asserted at runtime, exported for independent audit in
`alignment_map.parquet`, and tested by mutating future prices and requiring every
earlier feature row to stay bit-identical.

## Layout

```
forex_research/          pipeline modules (no notebooks in the production path)
  config.py              all research parameters; fails loudly on missing period lists
  data_loader.py         MT5 TSV loading, timestamp normalisation, UTC derivation
  data_quality.py        auditing + timestamp-semantics inference
  timeframe_builder.py   independent per-timeframe OHLC construction
  heikin_ashi.py         Heikin Ashi, computed per timeframe
  indicators.py          EMA/SMA/ATR/slope/percentile primitives
  single_tf_features.py  per-timeframe EMA/MA state
  consolidation_features.py  trailing-only compression measurement
  alignment.py           point-in-time multi-timeframe synchronisation
  cross_tf_features.py   cross-timeframe relationships
  outcomes.py            future market outcomes (labels)
  pipeline.py            chunked feature-matrix assembly
  validation.py          automated quality control
  reporting.py           DAY1_REPORT.md
  manifest.py            run_manifest.json
  day1.py                CLI entry point
configs/forex_day1.yaml  the research configuration
reference/pine/          the client's reference indicator scripts (source of the period list)
data/raw/                source CSVs (git-ignored, supplied by the client)
tests/                   quality-control suite
output/day1/             generated artefacts (git-ignored)
```

## Configuration

`configs/forex_day1.yaml` holds every parameter. Changing the timeframe set, the
EMA/SMA periods, the slope lookbacks, the consolidation windows, the forward
horizons or the expansion thresholds requires no source-code edit.

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

- **No bid/ask or tick data.** The MT5 export is a single bid-side OHLC series plus a
  per-bar spread column. Execution-cost, slippage and sub-5-minute path analysis are
  pending on better data.
- **Timezone is inferred, not documented.** The evidence is strong and reproducible
  (see `DAY1_REPORT.md`), but should be confirmed with the broker before live work.
- **Daily timeframe is off.** With a 300-period maximum it needs more history than the
  ~16 months supplied.

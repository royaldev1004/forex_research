from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

PriceBasis = Literal["standard", "heikin_ashi"]
CrossTfMode = Literal["same_period", "selected_pairs", "all_pairs"]
ResearchMode = Literal["production", "smoke"]
IndicatorSourceMode = Literal["latest_closed", "previous_closed"]

#: How many bars of extra lag each indicator source mode introduces.
#:
#: ``latest_closed`` (0) - the indicator at a timeframe bar includes that bar's
#: own close. Combined with close-time alignment this means the newest *fully
#: closed* candle at the decision time contributes. This is the research default:
#: it is exactly "what was objectively available at the decision timestamp".
#:
#: ``previous_closed`` (1) - the indicator additionally drops the aligned bar, so
#: the newest contributing candle is one bar older. This reproduces the literal
#: ``src = ha_close[1]`` of the reference Pine scripts *as written*, but note that
#: in Pine that offset exists to exclude the forming bar, a job already done here
#: by close-time alignment. Selecting it therefore double-counts the protection
#: and adds one whole bar of avoidable staleness. See INDICATOR_TIMING_NOTE.md.
INDICATOR_SOURCE_SHIFT: dict[str, int] = {"latest_closed": 0, "previous_closed": 1}

#: Timeframe label -> minutes. Extend here to support more timeframes.
TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "1D": 1440,
}


class ConfigError(ValueError):
    pass


def timeframe_minutes(tf: str) -> int:
    try:
        return TIMEFRAME_MINUTES[tf]
    except KeyError as exc:  # pragma: no cover - guarded by validation
        raise ConfigError(
            f"Unknown timeframe {tf!r}. Known timeframes: {sorted(TIMEFRAME_MINUTES)}"
        ) from exc


@dataclass(frozen=True)
class DataSource:

    path: Path
    timeframe: str
    role: str  # "base" | "reference"
    format: str = "mt5_tsv"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True)
class ResearchConfig:

    # --- identity -------------------------------------------------------
    symbol: str
    pip_size: float
    point_size: float

    # --- data -----------------------------------------------------------
    sources: tuple[DataSource, ...]
    source_timezone_mode: str
    source_timezone_note: str
    timestamp_semantics: str  # "bar_open" | "bar_close"

    # --- timeframes -----------------------------------------------------
    base_timeframe: str
    timeframes: tuple[str, ...]
    include_daily: bool

    # --- indicators -----------------------------------------------------
    ema_periods: tuple[int, ...]
    sma_periods: tuple[int, ...]
    price_basis: PriceBasis
    indicator_source_mode: IndicatorSourceMode
    indicator_source_shift_bars: int
    atr_period: int
    atr_price_basis: PriceBasis

    # --- feature parameters ---------------------------------------------
    slope_lookbacks: tuple[int, ...]
    primary_slope_lookback: int
    consolidation_windows: tuple[int, ...]
    crossover_windows: tuple[int, ...]
    dispersion_change_lookback: int
    entanglement_atr_fraction: float
    compression_percentile_lookback: int
    compression_percentile_threshold: float
    trend_ordering_threshold: float

    # --- cross timeframe --------------------------------------------------
    cross_tf_mode: CrossTfMode
    cross_tf_selected_pairs: tuple[tuple[str, str], ...]
    cross_tf_per_indicator_features: tuple[str, ...]
    wide_per_indicator_features: tuple[str, ...]

    # --- outcomes ---------------------------------------------------------
    forward_horizons_minutes: tuple[int, ...]
    expansion_thresholds_pips: tuple[float, ...]
    outcome_price_basis: PriceBasis
    outcome_entry_reference: str

    # --- run --------------------------------------------------------------
    research_mode: ResearchMode
    output_dir: Path
    write_full_datasets: bool
    chunk_rows: int
    sample_rows: int
    random_seed: int
    log_level: str

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.pip_size <= 0 or self.point_size <= 0:
            raise ConfigError("pip_size and point_size must be positive.")

        if self.base_timeframe not in self.timeframes:
            raise ConfigError(
                f"base_timeframe {self.base_timeframe!r} must appear in timeframes "
                f"{list(self.timeframes)!r}."
            )

        for tf in self.timeframes:
            timeframe_minutes(tf)

        base_min = timeframe_minutes(self.base_timeframe)
        for tf in self.timeframes:
            m = timeframe_minutes(tf)
            if m < base_min:
                raise ConfigError(
                    f"Timeframe {tf!r} is shorter than the base timeframe "
                    f"{self.base_timeframe!r}; higher-resolution series cannot be "
                    "derived by resampling."
                )
            if m % base_min != 0:
                raise ConfigError(
                    f"Timeframe {tf!r} ({m}m) is not an integer multiple of the base "
                    f"timeframe {self.base_timeframe!r} ({base_min}m); it cannot be "
                    "constructed without ambiguity."
                )

        if self.timestamp_semantics not in ("bar_open", "bar_close"):
            raise ConfigError("timestamp_semantics must be 'bar_open' or 'bar_close'.")

        if self.price_basis not in ("standard", "heikin_ashi"):
            raise ConfigError("price_basis must be 'standard' or 'heikin_ashi'.")

        if self.research_mode == "production":
            if not self.ema_periods and not self.sma_periods:
                raise ConfigError(
                    "research_mode='production' requires a non-empty ema_periods and/or "
                    "sma_periods list. The authoritative period list must come from the "
                    "client's reference indicator scripts, never from a default invented "
                    "by this pipeline. Set research_mode='smoke' only for non-research "
                    "smoke tests."
                )

        for name, periods in (("ema_periods", self.ema_periods), ("sma_periods", self.sma_periods)):
            if any(p < 1 for p in periods):
                raise ConfigError(f"{name} must contain positive integers only.")
            if len(set(periods)) != len(periods):
                raise ConfigError(f"{name} contains duplicate periods: {periods!r}")
            if list(periods) != sorted(periods):
                raise ConfigError(f"{name} must be sorted ascending: {periods!r}")

        if self.indicator_source_mode not in INDICATOR_SOURCE_SHIFT:
            raise ConfigError(
                f"indicator_source_mode must be one of {sorted(INDICATOR_SOURCE_SHIFT)}; "
                f"got {self.indicator_source_mode!r}."
            )

        if self.indicator_source_shift_bars < 0:
            raise ConfigError("indicator_source_shift_bars must be >= 0.")

        expected = INDICATOR_SOURCE_SHIFT[self.indicator_source_mode]
        if self.indicator_source_shift_bars != expected:
            raise ConfigError(
                f"indicator_source_shift_bars={self.indicator_source_shift_bars} contradicts "
                f"indicator_source_mode={self.indicator_source_mode!r} (which implies "
                f"{expected}). Indicator timing is a research-semantics decision, so it must "
                "not be set two ways at once. Change the mode, or drop the explicit shift."
            )

        if self.primary_slope_lookback not in self.slope_lookbacks:
            raise ConfigError(
                f"primary_slope_lookback {self.primary_slope_lookback} must be one of "
                f"slope_lookbacks {list(self.slope_lookbacks)}."
            )

        if not (0.0 < self.compression_percentile_threshold < 1.0):
            raise ConfigError("compression_percentile_threshold must be in (0, 1).")

        if not (0.0 <= self.trend_ordering_threshold <= 1.0):
            raise ConfigError("trend_ordering_threshold must be in [0, 1].")

        if not self.consolidation_windows:
            raise ConfigError("consolidation_windows must not be empty.")
        if not self.crossover_windows:
            raise ConfigError("crossover_windows must not be empty.")
        if not self.slope_lookbacks:
            raise ConfigError("slope_lookbacks must not be empty.")

        if self.cross_tf_mode not in ("same_period", "selected_pairs", "all_pairs"):
            raise ConfigError(
                "cross_tf_mode must be 'same_period', 'selected_pairs' or 'all_pairs'."
            )

        if self.cross_tf_mode == "selected_pairs" and not self.cross_tf_selected_pairs:
            raise ConfigError(
                "cross_tf_mode='selected_pairs' requires cross_tf_selected_pairs."
            )

        base_min = timeframe_minutes(self.base_timeframe)
        for h in self.forward_horizons_minutes:
            if h <= 0:
                raise ConfigError("forward_horizons_minutes must be positive.")
            if h % base_min != 0:
                raise ConfigError(
                    f"Forward horizon {h}m is not a multiple of the base timeframe "
                    f"({base_min}m)."
                )

        if any(t <= 0 for t in self.expansion_thresholds_pips):
            raise ConfigError("expansion_thresholds_pips must be positive.")

        if not self.sources:
            raise ConfigError("At least one data source must be configured.")

        base_sources = [s for s in self.sources if s.role == "base"]
        if len(base_sources) != 1:
            raise ConfigError(
                f"Exactly one source must have role='base'; found {len(base_sources)}."
            )
        if base_sources[0].timeframe != self.base_timeframe:
            raise ConfigError(
                f"The base source timeframe {base_sources[0].timeframe!r} does not match "
                f"base_timeframe {self.base_timeframe!r}."
            )

    # ------------------------------------------------------------------
    @property
    def base_source(self) -> DataSource:
        return next(s for s in self.sources if s.role == "base")

    @property
    def active_timeframes(self) -> tuple[str, ...]:
        return tuple(
            tf for tf in self.timeframes if self.include_daily or timeframe_minutes(tf) < 1440
        )

    @property
    def higher_timeframes(self) -> tuple[str, ...]:
        base_min = timeframe_minutes(self.base_timeframe)
        return tuple(tf for tf in self.active_timeframes if timeframe_minutes(tf) > base_min)

    def indicator_specs(self) -> list[tuple[str, int]]:
        specs = [("ema", p) for p in self.ema_periods]
        specs += [("sma", p) for p in self.sma_periods]
        return specs

    def timeframe_pairs(self) -> list[tuple[str, str]]:
        if self.cross_tf_mode == "selected_pairs":
            return [tuple(p) for p in self.cross_tf_selected_pairs]  # type: ignore[misc]
        tfs = list(self.active_timeframes)
        tfs.sort(key=timeframe_minutes)
        return [
            (a, b)
            for i, a in enumerate(tfs)
            for b in tfs[i + 1 :]
        ]

    def max_indicator_period(self) -> int:
        allp = list(self.ema_periods) + list(self.sma_periods)
        return max(allp) if allp else 0

    def warmup_bars(self) -> int:
        rolling = max(
            list(self.consolidation_windows)
            + list(self.crossover_windows)
            + [self.dispersion_change_lookback, self.atr_period]
        )
        return (
            self.max_indicator_period()
            + self.indicator_source_shift_bars
            + max(self.slope_lookbacks)
            + rolling
        )

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["sources"] = [
            {"path": str(s.path), "timeframe": s.timeframe, "role": s.role, "format": s.format}
            for s in self.sources
        ]
        d["output_dir"] = str(self.output_dir)
        d["cross_tf_selected_pairs"] = [list(p) for p in self.cross_tf_selected_pairs]
        return d


def _req(raw: dict[str, Any], key: str) -> Any:
    if key not in raw:
        raise ConfigError(f"Missing required configuration key: {key!r}")
    return raw[key]


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ResearchConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} did not parse into a mapping.")
    if overrides:
        raw = {**raw, **overrides}

    cfg_dir = path.parent

    def _resolve(p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (cfg_dir / p).resolve()

    sources = tuple(
        DataSource(
            path=_resolve(s["path"]),
            timeframe=s["timeframe"],
            role=s.get("role", "reference"),
            format=s.get("format", "mt5_tsv"),
        )
        for s in _req(raw, "sources")
    )

    slope_lookbacks = tuple(int(x) for x in raw.get("slope_lookbacks", [1, 3, 5]))

    # Indicator timing: the mode is authoritative and the shift is derived from it.
    # An explicit shift is accepted only when it agrees with the mode, so the
    # semantics can never be changed silently by editing one of the two.
    _indicator_mode = raw.get("indicator_source_mode", "latest_closed")
    if _indicator_mode not in INDICATOR_SOURCE_SHIFT:
        raise ConfigError(
            f"indicator_source_mode must be one of {sorted(INDICATOR_SOURCE_SHIFT)}; "
            f"got {_indicator_mode!r}."
        )
    _indicator_shift = int(
        raw.get("indicator_source_shift_bars", INDICATOR_SOURCE_SHIFT[_indicator_mode])
    )

    return ResearchConfig(
        symbol=_req(raw, "symbol"),
        pip_size=float(raw.get("pip_size", 0.0001)),
        point_size=float(raw.get("point_size", 0.00001)),
        sources=sources,
        source_timezone_mode=raw.get("source_timezone_mode", "new_york_plus_7"),
        source_timezone_note=raw.get("source_timezone_note", ""),
        timestamp_semantics=raw.get("timestamp_semantics", "bar_open"),
        base_timeframe=_req(raw, "base_timeframe"),
        timeframes=tuple(_req(raw, "timeframes")),
        include_daily=bool(raw.get("include_daily", False)),
        ema_periods=tuple(int(x) for x in raw.get("ema_periods") or []),
        sma_periods=tuple(int(x) for x in raw.get("sma_periods") or []),
        price_basis=raw.get("price_basis", "heikin_ashi"),
        indicator_source_mode=_indicator_mode,
        indicator_source_shift_bars=_indicator_shift,
        atr_period=int(raw.get("atr_period", 14)),
        atr_price_basis=raw.get("atr_price_basis", "standard"),
        slope_lookbacks=slope_lookbacks,
        primary_slope_lookback=int(raw.get("primary_slope_lookback", slope_lookbacks[-1])),
        consolidation_windows=tuple(int(x) for x in raw.get("consolidation_windows", [6, 12, 24])),
        crossover_windows=tuple(int(x) for x in raw.get("crossover_windows", [6, 24])),
        dispersion_change_lookback=int(raw.get("dispersion_change_lookback", 6)),
        entanglement_atr_fraction=float(raw.get("entanglement_atr_fraction", 0.10)),
        compression_percentile_lookback=int(raw.get("compression_percentile_lookback", 1000)),
        compression_percentile_threshold=float(raw.get("compression_percentile_threshold", 0.25)),
        trend_ordering_threshold=float(raw.get("trend_ordering_threshold", 0.5)),
        cross_tf_mode=raw.get("cross_tf_mode", "same_period"),
        cross_tf_selected_pairs=tuple(
            tuple(p) for p in raw.get("cross_tf_selected_pairs", []) or []
        ),
        cross_tf_per_indicator_features=tuple(
            raw.get(
                "cross_tf_per_indicator_features",
                ["value_dist_pips", "value_dist_atr", "slope_diff_atr"],
            )
        ),
        wide_per_indicator_features=tuple(
            raw.get("wide_per_indicator_features", ["dist_pips", "dist_atr", "slope_atr"])
        ),
        forward_horizons_minutes=tuple(
            int(x) for x in raw.get("forward_horizons_minutes", [15, 30, 60, 120, 240])
        ),
        expansion_thresholds_pips=tuple(
            float(x) for x in raw.get("expansion_thresholds_pips", [20, 30, 50])
        ),
        outcome_price_basis=raw.get("outcome_price_basis", "standard"),
        outcome_entry_reference=raw.get("outcome_entry_reference", "decision_bar_close"),
        research_mode=raw.get("research_mode", "production"),
        output_dir=_resolve(raw.get("output_dir", "../output/day1")),
        write_full_datasets=bool(raw.get("write_full_datasets", True)),
        chunk_rows=int(raw.get("chunk_rows", 20000)),
        sample_rows=int(raw.get("sample_rows", 5000)),
        random_seed=int(raw.get("random_seed", 20260821)),
        log_level=raw.get("log_level", "INFO"),
    )

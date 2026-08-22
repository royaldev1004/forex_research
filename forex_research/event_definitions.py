from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from .config import ConfigError

CanonicalRule = Literal["earliest_qualifying"]
SuppressionRule = Literal["horizon", "to_threshold"]

#: Session boundaries in the feed's native clock (New York + 7h). The FX day on
#: this feed starts at 00:00, which is 17:00 New York.
SESSIONS: tuple[tuple[str, int, int], ...] = (
    ("asia", 0, 8),
    ("london", 8, 13),
    ("london_ny_overlap", 13, 17),
    ("new_york", 17, 22),
    ("late", 22, 24),
)


@dataclass(frozen=True)
class EventSpec:

    direction: str            # "up" | "down"
    threshold_pips: float
    horizon_minutes: int

    @property
    def tag(self) -> str:
        return f"{self.direction}_{self.threshold_pips:g}p_{self.horizon_minutes}m"

    @property
    def expansion_column(self) -> str:
        return f"expansion_{self.direction}_{self.threshold_pips:g}p_{self.horizon_minutes}m"

    @property
    def time_to_column(self) -> str:
        return (f"time_to_expansion_{self.direction}_"
                f"{self.threshold_pips:g}p_{self.horizon_minutes}m")

    @property
    def opposite_expansion_column(self) -> str:
        other = "down" if self.direction == "up" else "up"
        return f"expansion_{other}_{self.threshold_pips:g}p_{self.horizon_minutes}m"


@dataclass(frozen=True)
class Step2Config:

    directions: tuple[str, ...]
    thresholds_pips: tuple[float, ...]
    horizons_minutes: tuple[int, ...]

    canonical_rule: CanonicalRule
    suppression_rule: SuppressionRule
    extra_separation_minutes: int

    require_source_quality: bool
    require_complete_horizon: bool
    require_observed_bars: bool
    exclude_ambiguous_first_touch: bool

    compression_primary_column: str
    compression_compressed_max_percentile: float
    compression_expanding_min_percentile: float
    compression_min_consecutive_bars: int
    consolidation_episode_separation_minutes: int

    controls_per_event: int
    control_purge_minutes: int
    outcome_filter_general_controls: bool
    match_on_session: bool
    match_on_hour_bucket: bool
    match_on_day_of_week: bool
    volatility_column: str
    volatility_bins: int
    hour_bucket_hours: int
    random_controls_per_event: int
    random_seed: int

    output_dir: Path

    def event_specs(self) -> list[EventSpec]:
        return [
            EventSpec(d, t, h)
            for h in self.horizons_minutes
            for t in self.thresholds_pips
            for d in self.directions
        ]

    def to_dict(self) -> dict[str, Any]:
        import dataclasses

        d = dataclasses.asdict(self)
        d["output_dir"] = str(self.output_dir)
        return d


@dataclass(frozen=True)
class Step3AConfig:

    report_horizons_minutes: tuple[int, ...]
    report_thresholds_pips: tuple[float, ...]
    distribution_quantiles: tuple[float, ...]
    output_dir: Path
    exploratory_feature_families: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        import dataclasses

        d = dataclasses.asdict(self)
        d["output_dir"] = str(self.output_dir)
        return d


def session_of(hour: int) -> str:
    for name, lo, hi in SESSIONS:
        if lo <= hour < hi:
            return name
    return "late"


def _resolve(cfg_dir: Path, p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (cfg_dir / p).resolve()


def load_step2_config(path: str | Path) -> Step2Config:
    path = Path(path)
    raw_all = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = raw_all.get("step2") or {}

    directions = tuple(raw.get("directions", ["up", "down"]))
    for d in directions:
        if d not in ("up", "down"):
            raise ConfigError(f"step2.directions must contain only 'up'/'down'; got {d!r}")

    thresholds = tuple(float(x) for x in raw.get("thresholds_pips", [20, 30, 50]))
    horizons = tuple(int(x) for x in raw.get("horizons_minutes", [60, 120, 240]))
    if not thresholds or not horizons:
        raise ConfigError("step2 requires at least one threshold and one horizon.")

    canonical = raw.get("canonical_rule", "earliest_qualifying")
    if canonical != "earliest_qualifying":
        raise ConfigError(
            f"Unsupported step2.canonical_rule {canonical!r}. Only 'earliest_qualifying' is "
            "implemented: any rule that inspects the future path to pick a 'better' timestamp "
            "inside an episode would leak outcome information into event selection."
        )

    suppression = raw.get("suppression_rule", "horizon")
    if suppression not in ("horizon", "to_threshold"):
        raise ConfigError("step2.suppression_rule must be 'horizon' or 'to_threshold'.")

    lo_pct = float(raw.get("compression_compressed_max_percentile", 0.25))
    hi_pct = float(raw.get("compression_expanding_min_percentile", 0.75))
    if not (0.0 < lo_pct < hi_pct < 1.0):
        raise ConfigError(
            "step2 compression percentiles must satisfy 0 < compressed_max < expanding_min < 1."
        )

    return Step2Config(
        directions=directions,
        thresholds_pips=thresholds,
        horizons_minutes=horizons,
        canonical_rule=canonical,
        suppression_rule=suppression,
        extra_separation_minutes=int(raw.get("extra_separation_minutes", 0)),
        require_source_quality=bool(raw.get("require_source_quality", True)),
        require_complete_horizon=bool(raw.get("require_complete_horizon", True)),
        require_observed_bars=bool(raw.get("require_observed_bars", True)),
        exclude_ambiguous_first_touch=bool(raw.get("exclude_ambiguous_first_touch", False)),
        compression_primary_column=raw.get(
            "compression_primary_column", "5m__compression_percentile_12"),
        compression_compressed_max_percentile=lo_pct,
        compression_expanding_min_percentile=hi_pct,
        compression_min_consecutive_bars=int(raw.get("compression_min_consecutive_bars", 1)),
        consolidation_episode_separation_minutes=int(
            raw.get("consolidation_episode_separation_minutes", 240)),
        controls_per_event=int(raw.get("controls_per_event", 3)),
        control_purge_minutes=int(raw.get("control_purge_minutes", 240)),
        outcome_filter_general_controls=bool(
            raw.get("outcome_filter_general_controls", False)),
        match_on_session=bool(raw.get("match_on_session", True)),
        match_on_hour_bucket=bool(raw.get("match_on_hour_bucket", True)),
        match_on_day_of_week=bool(raw.get("match_on_day_of_week", False)),
        volatility_column=raw.get("volatility_column", "5m__atr_pips"),
        volatility_bins=int(raw.get("volatility_bins", 4)),
        hour_bucket_hours=int(raw.get("hour_bucket_hours", 4)),
        random_controls_per_event=int(raw.get("random_controls_per_event", 3)),
        random_seed=int(raw.get("random_seed", 20260821)),
        output_dir=_resolve(path.parent, raw.get("output_dir", "../output/step2")),
    )


def load_step3a_config(path: str | Path) -> Step3AConfig:
    path = Path(path)
    raw_all = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = raw_all.get("step3a") or {}
    return Step3AConfig(
        report_horizons_minutes=tuple(int(x) for x in raw.get("report_horizons_minutes",
                                                              [60, 120, 240])),
        report_thresholds_pips=tuple(float(x) for x in raw.get("report_thresholds_pips",
                                                               [20, 30, 50])),
        distribution_quantiles=tuple(float(x) for x in raw.get("distribution_quantiles",
                                                               [0.1, 0.25, 0.5, 0.75, 0.9])),
        exploratory_feature_families=tuple(raw.get("exploratory_feature_families", [
            "consolidation", "ribbon_structure", "indicator_slope", "price_to_htf",
            "cross_tf_slope", "cross_tf_distance", "cross_tf_direction",
            "cross_tf_structure", "crossover", "ordering", "volatility",
        ])),
        output_dir=_resolve(path.parent, raw.get("output_dir", "../output/step3a")),
    )

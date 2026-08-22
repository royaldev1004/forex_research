from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:

    feature_name: str
    feature_family: str
    description: str
    units: str
    timeframe: str = ""
    indicator_type: str = ""
    period: int | None = None
    lookback: int | None = None
    normalized: bool = False
    uses_future_data: bool = False
    source_columns: str = ""

    def as_row(self) -> dict:
        return {
            "feature_name": self.feature_name,
            "feature_family": self.feature_family,
            "timeframe": self.timeframe,
            "indicator_type": self.indicator_type,
            "period": self.period if self.period is not None else "",
            "lookback": self.lookback if self.lookback is not None else "",
            "description": self.description,
            "units": self.units,
            "normalized": self.normalized,
            "uses_future_data": self.uses_future_data,
            "source_columns": self.source_columns,
        }


class DuplicateFeatureError(ValueError):
    pass


@dataclass
class FeatureSet:

    columns: dict[str, pd.Series] = field(default_factory=dict)
    specs: list[FeatureSpec] = field(default_factory=list)

    def add(self, series: pd.Series, spec: FeatureSpec) -> None:
        if spec.feature_name in self.columns:
            raise DuplicateFeatureError(f"Feature already registered: {spec.feature_name}")
        self.columns[spec.feature_name] = series
        self.specs.append(spec)

    def extend(self, other: "FeatureSet") -> None:
        for spec in other.specs:
            self.add(other.columns[spec.feature_name], spec)

    def to_frame(self, index=None, dtype: str = "float32") -> pd.DataFrame:
        if not self.columns:
            return pd.DataFrame(index=index)
        df = pd.DataFrame(self.columns, index=index)
        return df.astype(dtype)

    @property
    def names(self) -> list[str]:
        return [s.feature_name for s in self.specs]

    def __len__(self) -> int:
        return len(self.specs)


def build_dictionary(specs: list[FeatureSpec]) -> pd.DataFrame:
    return pd.DataFrame([s.as_row() for s in specs])


def family_counts(specs: list[FeatureSpec]) -> pd.DataFrame:
    df = build_dictionary(specs)
    if df.empty:
        return pd.DataFrame(columns=["feature_family", "timeframe", "n_features"])
    out = (
        df.groupby(["feature_family", "timeframe"], dropna=False)
        .size()
        .reset_index(name="n_features")
        .sort_values(["feature_family", "timeframe"])
        .reset_index(drop=True)
    )
    return out

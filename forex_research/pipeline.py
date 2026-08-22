from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .alignment import Alignment, align_timeframes, take
from .config import ResearchConfig
from .consolidation_features import build_consolidation_features
from .cross_tf_features import ChunkState, CrossTfBuilder
from .feature_dictionary import FeatureSpec
from .logging_utils import get_logger
from .single_tf_features import TimeframeFeatures, build_single_tf_features
from .timeframe_builder import TimeframeSeries, build_all_timeframes

log = get_logger("pipeline")


@dataclass
class BuiltDataset:

    timeframes: dict[str, TimeframeSeries]
    tf_features: dict[str, TimeframeFeatures]
    alignment: Alignment
    specs: list[FeatureSpec]
    feature_columns: list[str]
    cross_builder: CrossTfBuilder
    native_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    native_columns: dict[str, list[str]] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.alignment)

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)


def prepare_dataset(base: pd.DataFrame, cfg: ResearchConfig) -> BuiltDataset:
    timeframes = build_all_timeframes(base, cfg)

    tf_features: dict[str, TimeframeFeatures] = {}
    native_arrays: dict[str, np.ndarray] = {}
    native_columns: dict[str, list[str]] = {}
    specs: list[FeatureSpec] = []

    for tf, ts in timeframes.items():
        tff = build_single_tf_features(tf, ts.frame, cfg)
        cons = build_consolidation_features(tf, ts.frame, tff.atr, cfg)
        tff.features.extend(cons)

        frame = tff.features.to_frame(index=ts.frame.index, dtype="float32")
        native_arrays[tf] = frame.to_numpy(dtype="float32")
        native_columns[tf] = list(frame.columns)
        specs.extend(tff.features.specs)
        tf_features[tf] = tff
        log.info("%-3s native feature block: %d columns x %d bars",
                 tf, frame.shape[1], frame.shape[0])

    base_tf = cfg.base_timeframe
    alignment = align_timeframes(
        decision_time=timeframes[base_tf].frame["bar_close_time"],
        timeframe_closes={tf: ts.frame["bar_close_time"] for tf, ts in timeframes.items()},
        cfg=cfg,
    )

    indicator_keys = [f"{k}_{p}" for k, p in cfg.indicator_specs()]
    cross = CrossTfBuilder(cfg, indicator_keys)
    specs.extend(cross.specs)

    feature_columns: list[str] = []
    for tf in cfg.active_timeframes:
        feature_columns.extend(native_columns[tf])
    feature_columns.extend(s.feature_name for s in cross.specs)

    dup = {c for c in feature_columns if feature_columns.count(c) > 1}
    if dup:
        raise RuntimeError(f"Duplicate feature column names: {sorted(dup)[:10]}")

    log.info("Feature matrix plan: %d features x %d decision rows",
             len(feature_columns), len(alignment))
    return BuiltDataset(
        timeframes=timeframes,
        tf_features=tf_features,
        alignment=alignment,
        specs=specs,
        feature_columns=feature_columns,
        cross_builder=cross,
        native_arrays=native_arrays,
        native_columns=native_columns,
    )


def iter_feature_chunks(
    ds: BuiltDataset, cfg: ResearchConfig, chunk_rows: int | None = None
) -> Iterator[pd.DataFrame]:
    chunk_rows = chunk_rows or cfg.chunk_rows
    n = ds.n_rows
    base_tf = cfg.base_timeframe
    indicator_keys = [f"{k}_{p}" for k, p in cfg.indicator_specs()]

    ind_np = {tf: ds.tf_features[tf].indicator_values[indicator_keys].to_numpy("float64")
              for tf in cfg.active_timeframes}
    slope_np = {tf: ds.tf_features[tf].indicator_slopes_atr[indicator_keys].to_numpy("float64")
                for tf in cfg.active_timeframes}
    atr_np = {tf: ds.tf_features[tf].atr.to_numpy("float64") for tf in cfg.active_timeframes}
    price_np = {tf: ds.tf_features[tf].price.to_numpy("float64") for tf in cfg.active_timeframes}

    decision_time = ds.alignment.decision_time.to_numpy()

    for lo in range(0, n, chunk_rows):
        hi = min(lo + chunk_rows, n)
        m = hi - lo
        cols: dict[str, np.ndarray] = {}
        named: dict[str, np.ndarray] = {}

        gathered_values: dict[str, np.ndarray] = {}
        gathered_slopes: dict[str, np.ndarray] = {}
        gathered_atr: dict[str, np.ndarray] = {}

        for tf in cfg.active_timeframes:
            idx = ds.alignment.indices[tf][lo:hi]
            ok = idx >= 0
            safe = np.where(ok, idx, 0)

            block = ds.native_arrays[tf][safe]
            block[~ok] = np.nan
            for j, name in enumerate(ds.native_columns[tf]):
                col = block[:, j].astype("float64")
                cols[name] = col
                named[name] = col

            v = ind_np[tf][safe].astype("float64")
            v[~ok] = np.nan
            gathered_values[tf] = v

            s = slope_np[tf][safe].astype("float64")
            s[~ok] = np.nan
            gathered_slopes[tf] = s

            a = np.where(ok, atr_np[tf][safe], np.nan)
            gathered_atr[tf] = a

        base_idx = ds.alignment.indices[base_tf][lo:hi]
        base_price = take(price_np[base_tf], base_idx)

        state = ChunkState(
            n_rows=m,
            values=gathered_values,
            slopes_atr=gathered_slopes,
            atr=gathered_atr,
            named=named,
            base_price=base_price,
        )
        cols.update(ds.cross_builder.build(state))

        chunk = pd.DataFrame(
            {name: cols[name] for name in ds.feature_columns}
        ).astype("float32")
        chunk.insert(0, "feature_row_valid", ds.alignment.valid[lo:hi])
        chunk.insert(0, "decision_time", decision_time[lo:hi])
        chunk.insert(0, "symbol", cfg.symbol)
        yield chunk


def build_feature_frame(
    base: pd.DataFrame, cfg: ResearchConfig
) -> tuple[pd.DataFrame, BuiltDataset]:
    ds = prepare_dataset(base, cfg)
    frame = pd.concat(list(iter_feature_chunks(ds, cfg)), ignore_index=True)
    return frame, ds

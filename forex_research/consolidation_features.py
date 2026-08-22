from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .config import ResearchConfig
from .feature_dictionary import FeatureSet, FeatureSpec
from .logging_utils import get_logger

log = get_logger("consolidation_features")


def _col(tf: str, name: str) -> str:
    return f"{tf}__{name}"


def build_consolidation_features(
    tf: str, bars: pd.DataFrame, atr: pd.Series, cfg: ResearchConfig
) -> FeatureSet:
    index = bars.index
    pip = cfg.pip_size
    high, low, close = bars["high"], bars["low"], bars["close"]
    atr_v = atr.to_numpy(dtype="float64")
    fs = FeatureSet()

    for w in cfg.consolidation_windows:
        roll_hi = high.rolling(w, min_periods=w).max()
        roll_lo = low.rolling(w, min_periods=w).min()
        rng = roll_hi - roll_lo

        fs.add(rng / pip, FeatureSpec(
            _col(tf, f"range_pips_{w}"), "consolidation", timeframe=tf, lookback=w,
            description=f"Trailing {w}-bar high-low range",
            units="pips", source_columns="high,low"))

        rng_atr = pd.Series(ind.safe_divide(rng.to_numpy(dtype="float64"), atr_v), index=index)
        fs.add(rng_atr, FeatureSpec(
            _col(tf, f"range_atr_{w}"), "consolidation", timeframe=tf, lookback=w,
            description=f"Trailing {w}-bar range normalised by ATR (low values = compression)",
            units="atr_multiples", normalized=True, source_columns="high,low,atr"))

        fs.add(close.rolling(w, min_periods=w).std(ddof=0) / pip, FeatureSpec(
            _col(tf, f"close_std_pips_{w}"), "consolidation", timeframe=tf, lookback=w,
            description=f"Standard deviation of closes over the trailing {w} bars",
            units="pips", source_columns="close"))

        # --- location inside the trailing range --------------------------
        fs.add(pd.Series(ind.safe_divide((close - roll_lo).to_numpy(dtype="float64"),
                                         rng.to_numpy(dtype="float64")), index=index),
               FeatureSpec(
                   _col(tf, f"position_in_range_{w}"), "consolidation", timeframe=tf, lookback=w,
                   description=(f"Where the close sits in the trailing {w}-bar range: "
                                "0 at the range low, 1 at the range high"),
                   units="fraction", normalized=True, source_columns="close,high,low"))
        fs.add((roll_hi - close) / pip, FeatureSpec(
            _col(tf, f"dist_to_range_high_pips_{w}"), "consolidation", timeframe=tf, lookback=w,
            description=f"Distance from the close up to the trailing {w}-bar high",
            units="pips", source_columns="close,high"))
        fs.add((close - roll_lo) / pip, FeatureSpec(
            _col(tf, f"dist_to_range_low_pips_{w}"), "consolidation", timeframe=tf, lookback=w,
            description=f"Distance from the close down to the trailing {w}-bar low",
            units="pips", source_columns="close,low"))
        fs.add(pd.Series(ind.safe_divide((roll_hi - close).to_numpy(dtype="float64"), atr_v),
                         index=index), FeatureSpec(
            _col(tf, f"dist_to_range_high_atr_{w}"), "consolidation", timeframe=tf, lookback=w,
            description=f"ATR-normalised distance to the trailing {w}-bar high",
            units="atr_multiples", normalized=True, source_columns="close,high,atr"))
        fs.add(pd.Series(ind.safe_divide((close - roll_lo).to_numpy(dtype="float64"), atr_v),
                         index=index), FeatureSpec(
            _col(tf, f"dist_to_range_low_atr_{w}"), "consolidation", timeframe=tf, lookback=w,
            description=f"ATR-normalised distance to the trailing {w}-bar low",
            units="atr_multiples", normalized=True, source_columns="close,low,atr"))

        # --- breakouts of the *prior* trailing range ---------------------
        prior_hi = roll_hi.shift(1)
        prior_lo = roll_lo.shift(1)
        broke_up = (close > prior_hi).astype("float64").where(prior_hi.notna())
        broke_dn = (close < prior_lo).astype("float64").where(prior_lo.notna())
        fs.add(broke_up.rolling(w, min_periods=w).sum(), FeatureSpec(
            _col(tf, f"breakout_up_count_{w}"), "consolidation", timeframe=tf, lookback=w,
            description=(f"Closes above the previous {w}-bar high, counted over the trailing "
                         f"{w} bars"),
            units="count", source_columns="close,high"))
        fs.add(broke_dn.rolling(w, min_periods=w).sum(), FeatureSpec(
            _col(tf, f"breakout_down_count_{w}"), "consolidation", timeframe=tf, lookback=w,
            description=(f"Closes below the previous {w}-bar low, counted over the trailing "
                         f"{w} bars"),
            units="count", source_columns="close,low"))

        # --- point-in-time compression percentile ------------------------
        pct = ind.rolling_percentile(rng_atr, cfg.compression_percentile_lookback)
        fs.add(pct, FeatureSpec(
            _col(tf, f"compression_percentile_{w}"), "consolidation", timeframe=tf, lookback=w,
            description=(f"Percentile rank of the current {w}-bar ATR-normalised range within "
                         f"its own trailing {cfg.compression_percentile_lookback}-bar history. "
                         "Low values mean unusually compressed. Trailing-only; no future data."),
            units="percentile_0_to_1", normalized=True,
            source_columns=f"{tf}__range_atr_{w}"))

        compressed = (pct <= cfg.compression_percentile_threshold).astype("float64").where(pct.notna())
        fs.add(compressed, FeatureSpec(
            _col(tf, f"is_compressed_{w}"), "consolidation", timeframe=tf, lookback=w,
            description=(f"1 when the trailing compression percentile is at or below "
                         f"{cfg.compression_percentile_threshold}. The threshold is applied to a "
                         "trailing percentile, so a bar's classification never changes later."),
            units="boolean", source_columns=f"{tf}__compression_percentile_{w}"))
        fs.add(ind.consecutive_true(compressed), FeatureSpec(
            _col(tf, f"consecutive_compressed_bars_{w}"), "consolidation", timeframe=tf, lookback=w,
            description=f"Consecutive bars ending here classified as compressed at window {w}",
            units="bars", source_columns=f"{tf}__is_compressed_{w}"))

        # expansion state is simply the opposite tail of the same percentile
        fs.add((pct >= (1.0 - cfg.compression_percentile_threshold)).astype("float64")
               .where(pct.notna()), FeatureSpec(
            _col(tf, f"is_expanded_{w}"), "consolidation", timeframe=tf, lookback=w,
            description=(f"1 when the trailing compression percentile is at or above "
                         f"{1.0 - cfg.compression_percentile_threshold} (unusually wide range)"),
            units="boolean", source_columns=f"{tf}__compression_percentile_{w}"))

        fs.add(rng_atr - rng_atr.shift(w), FeatureSpec(
            _col(tf, f"range_atr_change_{w}"), "consolidation", timeframe=tf, lookback=w,
            description=f"Change in the ATR-normalised {w}-bar range over {w} bars",
            units="atr_multiples", normalized=True, source_columns=f"{tf}__range_atr_{w}"))

    log.info("%-3s consolidation features: %d", tf, len(fs))
    return fs

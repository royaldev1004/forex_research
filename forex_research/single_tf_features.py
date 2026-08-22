from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind
from .config import ResearchConfig
from .feature_dictionary import FeatureSet, FeatureSpec
from .heikin_ashi import ohlc_columns, price_series
from .logging_utils import get_logger

log = get_logger("single_tf_features")


@dataclass
class TimeframeFeatures:

    timeframe: str
    bar_close_time: pd.Series
    bar_open_time: pd.Series
    features: FeatureSet
    indicator_values: pd.DataFrame          # columns: "ema_17", "sma_300", ...
    indicator_slopes_atr: pd.DataFrame      # ATR-normalised primary-lookback slopes
    atr: pd.Series
    price: pd.Series                        # configured price basis close
    long_panel: pd.DataFrame = field(default_factory=pd.DataFrame)


def _col(tf: str, name: str) -> str:
    return f"{tf}__{name}"


def _pairwise_matrix(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = values.shape[1]
    ia, ib = np.triu_indices(k, 1)
    diff = (values[:, ia] - values[:, ib]).astype("float32")
    return diff, ia, ib


def _stack_features(
    fs: FeatureSet,
    tf: str,
    kind: str,
    values: np.ndarray,
    periods: list[int],
    atr: pd.Series,
    price: pd.Series,
    cfg: ResearchConfig,
    index: pd.Index,
) -> None:
    n, k = values.shape
    pip = cfg.pip_size
    atr_v = atr.to_numpy(dtype="float64")
    src = f"{kind}_" + ",".join(str(p) for p in periods)

    if k >= 2:
        diff, _, _ = _pairwise_matrix(values)
        finite = np.isfinite(diff)
        n_pairs = finite.sum(axis=1).astype("float64")
        with np.errstate(invalid="ignore"):
            bull = np.where(finite, diff > 0, False).sum(axis=1) / np.where(n_pairs > 0, n_pairs, np.nan)
            bear = np.where(finite, diff < 0, False).sum(axis=1) / np.where(n_pairs > 0, n_pairs, np.nan)

        fs.add(pd.Series(bull, index=index), FeatureSpec(
            _col(tf, f"{kind}_bullish_pair_fraction"), "ordering", timeframe=tf, indicator_type=kind,
            description=f"Fraction of {kind.upper()} pairs where the shorter period is above the longer",
            units="fraction", normalized=True, source_columns=src))
        fs.add(pd.Series(bear, index=index), FeatureSpec(
            _col(tf, f"{kind}_bearish_pair_fraction"), "ordering", timeframe=tf, indicator_type=kind,
            description=f"Fraction of {kind.upper()} pairs where the shorter period is below the longer",
            units="fraction", normalized=True, source_columns=src))
        fs.add(pd.Series(bull - bear, index=index), FeatureSpec(
            _col(tf, f"{kind}_ordering_score"), "ordering", timeframe=tf, indicator_type=kind,
            description=f"{kind.upper()} stack ordering score: +1 fully bullish, -1 fully bearish",
            units="score_-1_to_1", normalized=True, source_columns=src))

        # --- crossovers -------------------------------------------------
        # NaN maps to 0; `valid` already excludes those positions, and casting
        # NaN straight to int8 is undefined.
        sgn = np.where(finite, np.sign(diff), 0).astype("int8")
        sgn_prev = np.vstack([np.full((1, sgn.shape[1]), 0, dtype="int8"), sgn[:-1]])
        valid = finite & np.vstack([np.zeros((1, sgn.shape[1]), bool), finite[:-1]])
        turned_up = valid & (sgn > 0) & (sgn_prev <= 0)
        turned_dn = valid & (sgn < 0) & (sgn_prev >= 0)
        n_cross = (turned_up | turned_dn).sum(axis=1).astype("float64")
        net_dir = turned_up.sum(axis=1).astype("float64") - turned_dn.sum(axis=1).astype("float64")

        cross_s = pd.Series(n_cross, index=index)
        fs.add(cross_s, FeatureSpec(
            _col(tf, f"{kind}_cross_events"), "crossover", timeframe=tf, indicator_type=kind,
            description=f"Number of {kind.upper()} pairs that changed order on this bar",
            units="count", source_columns=src))
        fs.add(pd.Series(net_dir, index=index), FeatureSpec(
            _col(tf, f"{kind}_cross_net_direction"), "crossover", timeframe=tf, indicator_type=kind,
            description="Pairs turning bullish minus pairs turning bearish on this bar",
            units="count", source_columns=src))
        fs.add((cross_s > 0).astype("float64"), FeatureSpec(
            _col(tf, f"{kind}_new_cross_flag"), "crossover", timeframe=tf, indicator_type=kind,
            description=f"1 when at least one {kind.upper()} pair crossed on this bar",
            units="boolean", source_columns=src))
        fs.add(ind.bars_since(cross_s > 0), FeatureSpec(
            _col(tf, f"{kind}_bars_since_cross"), "crossover", timeframe=tf, indicator_type=kind,
            description=f"Bars since the last {kind.upper()} crossing (0 on the crossing bar)",
            units="bars", source_columns=src))

        for w in cfg.crossover_windows:
            rolled = cross_s.rolling(w, min_periods=w).sum()
            fs.add(rolled, FeatureSpec(
                _col(tf, f"{kind}_cross_count_{w}"), "crossover", timeframe=tf, indicator_type=kind,
                lookback=w, description=f"{kind.upper()} crossings in the trailing {w} bars",
                units="count", source_columns=src))
            fs.add(rolled / float(w), FeatureSpec(
                _col(tf, f"{kind}_cross_density_{w}"), "crossover", timeframe=tf, indicator_type=kind,
                lookback=w, description=f"{kind.upper()} crossings per bar over the trailing {w} bars",
                units="count_per_bar", normalized=True, source_columns=src))

        # entanglement: fraction of pairs closer together than a fraction of ATR
        thresh = cfg.entanglement_atr_fraction * atr_v[:, None]
        with np.errstate(invalid="ignore"):
            ent = np.where(finite, np.abs(diff) < thresh, False).sum(axis=1) / np.where(
                n_pairs > 0, n_pairs, np.nan)
        fs.add(pd.Series(ent, index=index), FeatureSpec(
            _col(tf, f"{kind}_entanglement_fraction"), "crossover", timeframe=tf, indicator_type=kind,
            description=(f"Fraction of {kind.upper()} pairs separated by less than "
                         f"{cfg.entanglement_atr_fraction} ATR (structure is entangled/compressed)"),
            units="fraction", normalized=True, source_columns=f"{src},atr"))
        del diff, sgn, sgn_prev, valid, turned_up, turned_dn

    # --- ribbon extents ---------------------------------------------------
    with np.errstate(invalid="ignore"), ind.quiet_warmup_warnings():
        rmin = np.nanmin(values, axis=1) if k else np.full(n, np.nan)
        rmax = np.nanmax(values, axis=1) if k else np.full(n, np.nan)
        rmean = np.nanmean(values, axis=1) if k else np.full(n, np.nan)
        rstd = np.nanstd(values, axis=1, ddof=0) if k else np.full(n, np.nan)
    allnan = ~np.isfinite(values).any(axis=1) if k else np.ones(n, bool)
    rmin[allnan] = np.nan
    rmax[allnan] = np.nan

    width = rmax - rmin
    fs.add(pd.Series(rmin, index=index), FeatureSpec(
        _col(tf, f"{kind}_ribbon_min"), "ribbon_structure", timeframe=tf, indicator_type=kind,
        description=f"Lowest {kind.upper()} value in the ribbon", units="price", source_columns=src))
    fs.add(pd.Series(rmax, index=index), FeatureSpec(
        _col(tf, f"{kind}_ribbon_max"), "ribbon_structure", timeframe=tf, indicator_type=kind,
        description=f"Highest {kind.upper()} value in the ribbon", units="price", source_columns=src))
    fs.add(pd.Series(width / pip, index=index), FeatureSpec(
        _col(tf, f"{kind}_ribbon_width_pips"), "ribbon_structure", timeframe=tf, indicator_type=kind,
        description=f"{kind.upper()} ribbon width (max minus min)", units="pips", source_columns=src))
    fs.add(pd.Series(ind.safe_divide(width, atr_v), index=index), FeatureSpec(
        _col(tf, f"{kind}_ribbon_width_atr"), "ribbon_structure", timeframe=tf, indicator_type=kind,
        description=f"{kind.upper()} ribbon width normalised by ATR", units="atr_multiples",
        normalized=True, source_columns=f"{src},atr"))
    fs.add(pd.Series(rmean, index=index), FeatureSpec(
        _col(tf, f"{kind}_ribbon_mean"), "ribbon_structure", timeframe=tf, indicator_type=kind,
        description=f"Mean of all {kind.upper()} values", units="price", source_columns=src))
    fs.add(pd.Series(rstd / pip, index=index), FeatureSpec(
        _col(tf, f"{kind}_ribbon_dispersion_pips"), "ribbon_structure", timeframe=tf, indicator_type=kind,
        description=f"Standard deviation across {kind.upper()} values", units="pips",
        source_columns=src))
    disp_atr = pd.Series(ind.safe_divide(rstd, atr_v), index=index)
    fs.add(disp_atr, FeatureSpec(
        _col(tf, f"{kind}_ribbon_dispersion_atr"), "ribbon_structure", timeframe=tf, indicator_type=kind,
        description=f"{kind.upper()} dispersion normalised by ATR", units="atr_multiples",
        normalized=True, source_columns=f"{src},atr"))

    lb = cfg.dispersion_change_lookback
    fs.add(disp_atr - disp_atr.shift(lb), FeatureSpec(
        _col(tf, f"{kind}_dispersion_change_{lb}"), "ribbon_structure", timeframe=tf, indicator_type=kind,
        lookback=lb, description=f"Change in ATR-normalised {kind.upper()} dispersion over {lb} bars",
        units="atr_multiples", normalized=True, source_columns=src))
    width_atr = pd.Series(ind.safe_divide(width, atr_v), index=index)
    fs.add((width_atr - width_atr.shift(lb)) / float(lb), FeatureSpec(
        _col(tf, f"{kind}_expansion_rate_{lb}"), "ribbon_structure", timeframe=tf, indicator_type=kind,
        lookback=lb,
        description=(f"Per-bar expansion (positive) or contraction (negative) rate of the "
                     f"{kind.upper()} ribbon width over {lb} bars"),
        units="atr_multiples_per_bar", normalized=True, source_columns=src))

    # --- price position relative to the ribbon ---------------------------
    px = price.to_numpy(dtype="float64")
    with np.errstate(invalid="ignore"):
        n_below = (values < px[:, None]).sum(axis=1).astype("float64") if k else np.full(n, np.nan)
        frac_below = n_below / float(k) if k else np.full(n, np.nan)
        pos = ind.safe_divide(px - rmin, width)
    fs.add(pd.Series(frac_below, index=index), FeatureSpec(
        _col(tf, f"{kind}_fraction_below_price"), "ordering", timeframe=tf, indicator_type=kind,
        description=f"Fraction of {kind.upper()} values below the current price",
        units="fraction", normalized=True, source_columns=f"{src},price"))
    fs.add(pd.Series(pos, index=index), FeatureSpec(
        _col(tf, f"{kind}_price_position_in_ribbon"), "ribbon_structure", timeframe=tf, indicator_type=kind,
        description=("Where price sits inside the ribbon: 0 at ribbon min, 1 at ribbon max, "
                     "outside [0,1] when price is beyond the ribbon"),
        units="fraction", normalized=True, source_columns=f"{src},price"))


def build_single_tf_features(
    tf: str, bars: pd.DataFrame, cfg: ResearchConfig
) -> TimeframeFeatures:
    index = bars.index
    n = len(bars)
    pip = cfg.pip_size

    # --- ATR on real (standard) ranges, independent of the indicator basis
    a_o, a_h, a_l, a_c = ohlc_columns(cfg.atr_price_basis)
    atr = ind.atr(bars[a_h], bars[a_l], bars[a_c], cfg.atr_period)

    price = price_series(bars, cfg.price_basis).astype("float64")

    # --- spec-mandated indicator source: the previous *closed* candle -----
    src = price.shift(cfg.indicator_source_shift_bars)

    fs = FeatureSet()
    specs_src = f"{cfg.price_basis}_close(shift={cfg.indicator_source_shift_bars})"

    # --- raw indicator values --------------------------------------------
    values: dict[str, pd.Series] = {}
    for kind, period in cfg.indicator_specs():
        fn = ind.ema if kind == "ema" else ind.sma
        values[f"{kind}_{period}"] = fn(src, period)
    indicator_values = pd.DataFrame(values, index=index)

    atr_v = atr.to_numpy(dtype="float64")
    px_v = price.to_numpy(dtype="float64")

    # --- per-indicator distance and slope features ------------------------
    slopes: dict[int, dict[str, pd.Series]] = {lb: {} for lb in cfg.slope_lookbacks}
    slopes_atr_primary: dict[str, pd.Series] = {}
    wanted = set(cfg.wide_per_indicator_features)

    for kind, period in cfg.indicator_specs():
        key = f"{kind}_{period}"
        v = indicator_values[key]
        v_v = v.to_numpy(dtype="float64")
        dist = px_v - v_v

        if "dist_pips" in wanted:
            fs.add(pd.Series(dist / pip, index=index), FeatureSpec(
                _col(tf, f"{key}_price_dist_pips"), "indicator_distance", timeframe=tf,
                indicator_type=kind, period=period,
                description=f"Price minus {kind.upper()}({period})", units="pips",
                source_columns=f"price,{key}"))
        if "dist_atr" in wanted:
            fs.add(pd.Series(ind.safe_divide(dist, atr_v), index=index), FeatureSpec(
                _col(tf, f"{key}_price_dist_atr"), "indicator_distance", timeframe=tf,
                indicator_type=kind, period=period,
                description=f"Price minus {kind.upper()}({period}), normalised by ATR",
                units="atr_multiples", normalized=True, source_columns=f"price,{key},atr"))

        for lb in cfg.slope_lookbacks:
            sl = ind.slope(v, lb)
            slopes[lb][key] = sl
            sl_atr = pd.Series(ind.safe_divide(sl.to_numpy(dtype="float64"), atr_v), index=index)
            if lb == cfg.primary_slope_lookback:
                slopes_atr_primary[key] = sl_atr
            if "slope_atr" in wanted:
                fs.add(sl_atr, FeatureSpec(
                    _col(tf, f"{key}_slope_atr_{lb}"), "indicator_slope", timeframe=tf,
                    indicator_type=kind, period=period, lookback=lb,
                    description=(f"Per-bar rate of change of {kind.upper()}({period}) over {lb} "
                                 "bars, normalised by ATR"),
                    units="atr_multiples_per_bar", normalized=True,
                    source_columns=f"{key},atr"))
            if "slope_pips" in wanted:
                fs.add(sl / pip, FeatureSpec(
                    _col(tf, f"{key}_slope_pips_{lb}"), "indicator_slope", timeframe=tf,
                    indicator_type=kind, period=period, lookback=lb,
                    description=f"Per-bar rate of change of {kind.upper()}({period}) over {lb} bars",
                    units="pips_per_bar", source_columns=key))
            if "slope_raw" in wanted:
                fs.add(sl, FeatureSpec(
                    _col(tf, f"{key}_slope_raw_{lb}"), "indicator_slope", timeframe=tf,
                    indicator_type=kind, period=period, lookback=lb,
                    description=f"Absolute per-bar slope of {kind.upper()}({period}) over {lb} bars",
                    units="price_per_bar", source_columns=key))
            if "slope_sign" in wanted:
                fs.add(ind.sign(sl), FeatureSpec(
                    _col(tf, f"{key}_slope_sign_{lb}"), "indicator_slope", timeframe=tf,
                    indicator_type=kind, period=period, lookback=lb,
                    description=f"Sign of the {lb}-bar slope of {kind.upper()}({period})",
                    units="sign", source_columns=key))

    # --- family-level structure ------------------------------------------
    for kind, periods in (("ema", list(cfg.ema_periods)), ("sma", list(cfg.sma_periods))):
        if not periods:
            continue
        mat = indicator_values[[f"{kind}_{p}" for p in periods]].to_numpy(dtype="float64")
        _stack_features(fs, tf, kind, mat, periods, atr, price, cfg, index)

    # --- combined EMA+SMA ribbon -----------------------------------------
    all_cols = [f"{k}_{p}" for k, p in cfg.indicator_specs()]
    if len(all_cols) >= 2:
        mat = indicator_values[all_cols].to_numpy(dtype="float64")
        with np.errstate(invalid="ignore"), ind.quiet_warmup_warnings():
            cmin, cmax = np.nanmin(mat, axis=1), np.nanmax(mat, axis=1)
            cstd = np.nanstd(mat, axis=1, ddof=0)
        allnan = ~np.isfinite(mat).any(axis=1)
        cmin[allnan] = np.nan
        cmax[allnan] = np.nan
        cwidth = cmax - cmin
        srcs = "all_ema_sma"
        fs.add(pd.Series(cwidth / pip, index=index), FeatureSpec(
            _col(tf, "combined_ribbon_width_pips"), "ribbon_structure", timeframe=tf,
            indicator_type="combined",
            description="Width of the combined EMA+SMA ribbon", units="pips", source_columns=srcs))
        cwidth_atr = pd.Series(ind.safe_divide(cwidth, atr_v), index=index)
        fs.add(cwidth_atr, FeatureSpec(
            _col(tf, "combined_ribbon_width_atr"), "ribbon_structure", timeframe=tf,
            indicator_type="combined",
            description="Combined EMA+SMA ribbon width normalised by ATR",
            units="atr_multiples", normalized=True, source_columns=f"{srcs},atr"))
        fs.add(pd.Series(ind.safe_divide(cstd, atr_v), index=index), FeatureSpec(
            _col(tf, "combined_ribbon_dispersion_atr"), "ribbon_structure", timeframe=tf,
            indicator_type="combined",
            description="Dispersion across all EMA and SMA values, normalised by ATR",
            units="atr_multiples", normalized=True, source_columns=f"{srcs},atr"))
        lb = cfg.dispersion_change_lookback
        fs.add((cwidth_atr - cwidth_atr.shift(lb)) / float(lb), FeatureSpec(
            _col(tf, f"combined_expansion_rate_{lb}"), "ribbon_structure", timeframe=tf,
            indicator_type="combined", lookback=lb,
            description=f"Per-bar expansion/contraction rate of the combined ribbon over {lb} bars",
            units="atr_multiples_per_bar", normalized=True, source_columns=srcs))

    # --- EMA vs SMA same-period crossings ---------------------------------
    shared = [p for p in cfg.ema_periods if p in set(cfg.sma_periods)]
    if shared:
        e = indicator_values[[f"ema_{p}" for p in shared]].to_numpy(dtype="float64")
        s = indicator_values[[f"sma_{p}" for p in shared]].to_numpy(dtype="float64")
        d = e - s
        fin = np.isfinite(d)
        npair = fin.sum(axis=1).astype("float64")
        sgn = np.where(fin, np.sign(d), 0).astype("int8")
        sgn_prev = np.vstack([np.zeros((1, sgn.shape[1]), "int8"), sgn[:-1]])
        valid = fin & np.vstack([np.zeros((1, fin.shape[1]), bool), fin[:-1]])
        crossed = valid & (sgn != sgn_prev) & (sgn != 0) & (sgn_prev != 0)
        cs = pd.Series(crossed.sum(axis=1).astype("float64"), index=index)
        srcs = "ema_vs_sma_same_period"
        fs.add(cs, FeatureSpec(
            _col(tf, "ema_sma_cross_events"), "crossover", timeframe=tf, indicator_type="ema_vs_sma",
            description="Same-period EMA/SMA pairs that crossed on this bar",
            units="count", source_columns=srcs))
        fs.add(ind.bars_since(cs > 0), FeatureSpec(
            _col(tf, "ema_sma_bars_since_cross"), "crossover", timeframe=tf, indicator_type="ema_vs_sma",
            description="Bars since the last same-period EMA/SMA crossing",
            units="bars", source_columns=srcs))
        with np.errstate(invalid="ignore"):
            above = np.where(fin, d > 0, False).sum(axis=1) / np.where(npair > 0, npair, np.nan)
        fs.add(pd.Series(above, index=index), FeatureSpec(
            _col(tf, "ema_above_sma_fraction"), "ordering", timeframe=tf, indicator_type="ema_vs_sma",
            description="Fraction of periods where the EMA is above the same-period SMA",
            units="fraction", normalized=True, source_columns=srcs))
        for w in cfg.crossover_windows:
            fs.add(cs.rolling(w, min_periods=w).sum(), FeatureSpec(
                _col(tf, f"ema_sma_cross_count_{w}"), "crossover", timeframe=tf,
                indicator_type="ema_vs_sma", lookback=w,
                description=f"Same-period EMA/SMA crossings in the trailing {w} bars",
                units="count", source_columns=srcs))

    # --- volatility --------------------------------------------------------
    fs.add(atr / pip, FeatureSpec(
        _col(tf, "atr_pips"), "volatility", timeframe=tf, lookback=cfg.atr_period,
        description=f"Wilder ATR({cfg.atr_period}) on {cfg.atr_price_basis} ranges",
        units="pips", source_columns="high,low,close"))
    fs.add(atr / atr.shift(cfg.atr_period) - 1.0, FeatureSpec(
        _col(tf, f"atr_change_{cfg.atr_period}"), "volatility", timeframe=tf, lookback=cfg.atr_period,
        description=f"Relative change in ATR over {cfg.atr_period} bars",
        units="ratio", normalized=True, source_columns="atr"))

    std_c = bars["close"].astype("float64")
    for w in cfg.consolidation_windows:
        rv = ind.realized_volatility(std_c, w)
        fs.add(rv, FeatureSpec(
            _col(tf, f"realized_vol_{w}"), "volatility", timeframe=tf, lookback=w,
            description=f"Standard deviation of log returns over the trailing {w} bars",
            units="log_return_sd", normalized=True, source_columns="close"))
        fs.add(rv / rv.shift(w) - 1.0, FeatureSpec(
            _col(tf, f"realized_vol_change_{w}"), "volatility", timeframe=tf, lookback=w,
            description=f"Relative change in {w}-bar realised volatility",
            units="ratio", normalized=True, source_columns="close"))

    # --- execution context --------------------------------------------------
    pts_per_pip = cfg.pip_size / cfg.point_size
    fs.add(bars["spread_points_last"].astype("float64") / pts_per_pip, FeatureSpec(
        _col(tf, "spread_pips_last"), "execution_context", timeframe=tf,
        description=("Broker spread reported on the last base bar of this timeframe bar. "
                     "Known at decision time; retained for later execution-cost research."),
        units="pips", source_columns="spread_points"))
    fs.add(bars["base_bar_coverage"].astype("float64"), FeatureSpec(
        _col(tf, "base_bar_coverage"), "execution_context", timeframe=tf,
        description="Fraction of expected base bars actually present in this timeframe bar",
        units="fraction", normalized=True, source_columns="n_base_bars"))

    log.info("%-3s single-timeframe features: %d", tf, len(fs))

    return TimeframeFeatures(
        timeframe=tf,
        bar_close_time=bars["bar_close_time"],
        bar_open_time=bars["bar_open_time"],
        features=fs,
        indicator_values=indicator_values,
        indicator_slopes_atr=pd.DataFrame(slopes_atr_primary, index=index),
        atr=atr,
        price=price,
        long_panel=_build_long_panel(tf, bars, indicator_values, slopes, atr, price, cfg),
    )


def _build_long_panel(
    tf: str,
    bars: pd.DataFrame,
    indicator_values: pd.DataFrame,
    slopes: dict[int, dict[str, pd.Series]],
    atr: pd.Series,
    price: pd.Series,
    cfg: ResearchConfig,
) -> pd.DataFrame:
    pip = cfg.pip_size
    atr_v = atr.to_numpy(dtype="float64")
    px_v = price.to_numpy(dtype="float64")
    n = len(bars)

    # rank of each indicator value within its own family, per bar
    ranks: dict[str, np.ndarray] = {}
    for kind, periods in (("ema", list(cfg.ema_periods)), ("sma", list(cfg.sma_periods))):
        if not periods:
            continue
        mat = indicator_values[[f"{kind}_{p}" for p in periods]].to_numpy(dtype="float64")
        r = np.argsort(np.argsort(np.nan_to_num(mat, nan=-np.inf), axis=1), axis=1).astype("float32")
        r[~np.isfinite(mat)] = np.nan
        for j, p in enumerate(periods):
            ranks[f"{kind}_{p}"] = r[:, j]

    blocks = []
    for kind, period in cfg.indicator_specs():
        key = f"{kind}_{period}"
        v = indicator_values[key].to_numpy(dtype="float64")
        dist = px_v - v
        block = {
            "bar_open_time": bars["bar_open_time"].to_numpy(),
            "bar_close_time": bars["bar_close_time"].to_numpy(),
            "timeframe": np.full(n, tf),
            "indicator_type": np.full(n, kind),
            "period": np.full(n, period, dtype="int32"),
            "value": v.astype("float32"),
            "price": px_v.astype("float32"),
            "atr": atr_v.astype("float32"),
            "price_distance": dist.astype("float32"),
            "price_distance_pips": (dist / pip).astype("float32"),
            "normalized_price_distance": ind.safe_divide(dist, atr_v).astype("float32"),
            "rank_in_stack": ranks.get(key, np.full(n, np.nan, "float32")),
        }
        for lb in cfg.slope_lookbacks:
            sl = slopes[lb][key].to_numpy(dtype="float64")
            block[f"slope_pips_{lb}"] = (sl / pip).astype("float32")
            block[f"slope_atr_{lb}"] = ind.safe_divide(sl, atr_v).astype("float32")
        blocks.append(pd.DataFrame(block))

    panel = pd.concat(blocks, ignore_index=True)
    panel["timeframe"] = panel["timeframe"].astype("category")
    panel["indicator_type"] = panel["indicator_type"].astype("category")
    return panel.sort_values(["bar_close_time", "indicator_type", "period"]).reset_index(drop=True)

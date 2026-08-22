from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ResearchConfig
from .feature_dictionary import FeatureSpec
from .indicators import quiet_warmup_warnings, safe_divide
from .logging_utils import get_logger

log = get_logger("cross_tf_features")


@dataclass
class ChunkState:

    n_rows: int
    values: dict[str, np.ndarray]       # tf -> (n_rows, n_indicators)
    slopes_atr: dict[str, np.ndarray]   # tf -> (n_rows, n_indicators)
    atr: dict[str, np.ndarray]          # tf -> (n_rows,)
    named: dict[str, np.ndarray]        # "{tf}__{feature}" -> (n_rows,)
    base_price: np.ndarray              # (n_rows,) base-timeframe price at decision time


def _sign(a: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(a), np.sign(a), np.nan)


class CrossTfBuilder:

    def __init__(self, cfg: ResearchConfig, indicator_keys: list[str]):
        self.cfg = cfg
        self.indicator_keys = indicator_keys
        self.key_index = {k: i for i, k in enumerate(indicator_keys)}
        self.pairs = cfg.timeframe_pairs()
        self.specs: list[FeatureSpec] = []
        self._plan: list[tuple] = []
        self._build_plan()

    # ------------------------------------------------------------------
    def _add(self, spec: FeatureSpec, op: tuple) -> None:
        self.specs.append(spec)
        self._plan.append(op)

    def _build_plan(self) -> None:
        cfg = self.cfg
        specs = cfg.indicator_specs()
        lb = cfg.primary_slope_lookback
        wanted = set(cfg.cross_tf_per_indicator_features)

        # --- same-period / all-pairs indicator comparisons ---------------
        for lo_tf, hi_tf in self.pairs:
            if cfg.cross_tf_mode == "all_pairs":
                combos = [
                    ((k1, p1), (k2, p2))
                    for (k1, p1) in specs
                    for (k2, p2) in specs
                ]
            else:
                combos = [((k, p), (k, p)) for (k, p) in specs]

            for (k1, p1), (k2, p2) in combos:
                a, b = f"{k1}_{p1}", f"{k2}_{p2}"
                tag = f"{a}_vs_{b}" if (a != b) else a
                pair_tag = f"x_{lo_tf}_{hi_tf}"

                if "value_dist_pips" in wanted:
                    self._add(
                        FeatureSpec(
                            f"{pair_tag}__{tag}_value_dist_pips", "cross_tf_distance",
                            timeframe=f"{lo_tf}|{hi_tf}", indicator_type=k1, period=p1,
                            description=(f"{lo_tf} {k1.upper()}({p1}) minus {hi_tf} "
                                         f"{k2.upper()}({p2})"),
                            units="pips",
                            source_columns=f"{lo_tf}:{a},{hi_tf}:{b}"),
                        ("val_dist_pips", lo_tf, hi_tf, a, b),
                    )
                if "value_dist_atr" in wanted:
                    self._add(
                        FeatureSpec(
                            f"{pair_tag}__{tag}_value_dist_atr", "cross_tf_distance",
                            timeframe=f"{lo_tf}|{hi_tf}", indicator_type=k1, period=p1,
                            description=(f"{lo_tf} {k1.upper()}({p1}) minus {hi_tf} "
                                         f"{k2.upper()}({p2}), normalised by {lo_tf} ATR"),
                            units="atr_multiples", normalized=True,
                            source_columns=f"{lo_tf}:{a},{hi_tf}:{b},{lo_tf}:atr"),
                        ("val_dist_atr", lo_tf, hi_tf, a, b),
                    )
                if "slope_diff_atr" in wanted:
                    self._add(
                        FeatureSpec(
                            f"{pair_tag}__{tag}_slope_diff_atr", "cross_tf_slope",
                            timeframe=f"{lo_tf}|{hi_tf}", indicator_type=k1, period=p1,
                            lookback=lb,
                            description=(f"{lo_tf} minus {hi_tf} ATR-normalised {lb}-bar slope "
                                         f"of {k1.upper()}({p1}) vs {k2.upper()}({p2}). Slopes are "
                                         "per-bar rates of change on each timeframe's own bars."),
                            units="atr_multiples_per_bar", normalized=True,
                            source_columns=f"{lo_tf}:{a},{hi_tf}:{b}"),
                        ("slope_diff_atr", lo_tf, hi_tf, a, b),
                    )
                if "slope_sign_agree" in wanted:
                    self._add(
                        FeatureSpec(
                            f"{pair_tag}__{tag}_slope_sign_agree", "cross_tf_slope",
                            timeframe=f"{lo_tf}|{hi_tf}", indicator_type=k1, period=p1,
                            lookback=lb,
                            description=(f"1 when the {lo_tf} and {hi_tf} slopes of "
                                         f"{k1.upper()}({p1}) share a sign, else 0"),
                            units="boolean",
                            source_columns=f"{lo_tf}:{a},{hi_tf}:{b}"),
                        ("slope_sign_agree", lo_tf, hi_tf, a, b),
                    )
                if "slope_ratio" in wanted:
                    self._add(
                        FeatureSpec(
                            f"{pair_tag}__{tag}_slope_ratio", "cross_tf_slope",
                            timeframe=f"{lo_tf}|{hi_tf}", indicator_type=k1, period=p1,
                            lookback=lb,
                            description=(f"{lo_tf} slope divided by {hi_tf} slope; NaN when the "
                                         "denominator is zero so the ratio stays defined"),
                            units="ratio", normalized=True,
                            source_columns=f"{lo_tf}:{a},{hi_tf}:{b}"),
                        ("slope_ratio", lo_tf, hi_tf, a, b),
                    )

        # --- current price vs each higher-timeframe indicator -------------
        for hi_tf in self.cfg.higher_timeframes:
            for k, p in specs:
                key = f"{k}_{p}"
                self._add(
                    FeatureSpec(
                        f"p2h_{hi_tf}__{key}_dist_pips", "price_to_htf",
                        timeframe=hi_tf, indicator_type=k, period=p,
                        description=(f"Current base-timeframe price minus the {hi_tf} "
                                     f"{k.upper()}({p}) value that was in force at the decision time"),
                        units="pips", source_columns=f"base_price,{hi_tf}:{key}"),
                    ("p2h_pips", hi_tf, key),
                )
                self._add(
                    FeatureSpec(
                        f"p2h_{hi_tf}__{key}_dist_atr", "price_to_htf",
                        timeframe=hi_tf, indicator_type=k, period=p,
                        description=(f"Current base-timeframe price minus the {hi_tf} "
                                     f"{k.upper()}({p}), normalised by {hi_tf} ATR"),
                        units="atr_multiples", normalized=True,
                        source_columns=f"base_price,{hi_tf}:{key},{hi_tf}:atr"),
                    ("p2h_atr", hi_tf, key),
                )

        # --- direction / alignment ----------------------------------------
        tfs = list(cfg.active_timeframes)
        for lo_tf, hi_tf in self.pairs:
            self._add(
                FeatureSpec(
                    f"x_{lo_tf}_{hi_tf}__trend_agreement", "cross_tf_direction",
                    timeframe=f"{lo_tf}|{hi_tf}",
                    description=(f"1 when the {lo_tf} and {hi_tf} EMA stacks are ordered in the "
                                 "same direction, -1 when opposed, 0 when either is flat"),
                    units="score_-1_to_1", normalized=True,
                    source_columns=f"{lo_tf}__ema_ordering_score,{hi_tf}__ema_ordering_score"),
                ("trend_agree", lo_tf, hi_tf),
            )
            self._add(
                FeatureSpec(
                    f"x_{lo_tf}_{hi_tf}__ordering_score_diff", "cross_tf_direction",
                    timeframe=f"{lo_tf}|{hi_tf}",
                    description=f"{lo_tf} EMA ordering score minus the {hi_tf} EMA ordering score",
                    units="score", normalized=True,
                    source_columns=f"{lo_tf}__ema_ordering_score,{hi_tf}__ema_ordering_score"),
                ("ordering_diff", lo_tf, hi_tf),
            )
            self._add(
                FeatureSpec(
                    f"x_{lo_tf}_{hi_tf}__ribbon_width_ratio_atr", "cross_tf_structure",
                    timeframe=f"{lo_tf}|{hi_tf}",
                    description=(f"{lo_tf} combined ribbon width divided by the {hi_tf} combined "
                                 "ribbon width, both in ATR units"),
                    units="ratio", normalized=True,
                    source_columns=f"{lo_tf}__combined_ribbon_width_atr,"
                                   f"{hi_tf}__combined_ribbon_width_atr"),
                ("width_ratio", lo_tf, hi_tf),
            )
            self._add(
                FeatureSpec(
                    f"x_{lo_tf}_{hi_tf}__cross_density_ratio", "cross_tf_structure",
                    timeframe=f"{lo_tf}|{hi_tf}",
                    description=(f"{lo_tf} EMA crossing density divided by the {hi_tf} crossing "
                                 "density: high values mean the lower timeframe is churning while "
                                 "the higher timeframe is stable"),
                    units="ratio", normalized=True,
                    source_columns=f"{lo_tf}__ema_cross_density_*,{hi_tf}__ema_cross_density_*"),
                ("density_ratio", lo_tf, hi_tf, cfg.crossover_windows[-1]),
            )
            self._add(
                FeatureSpec(
                    f"x_{lo_tf}_{hi_tf}__entanglement_diff", "cross_tf_structure",
                    timeframe=f"{lo_tf}|{hi_tf}",
                    description=(f"{lo_tf} minus {hi_tf} EMA entanglement fraction: positive means "
                                 "the lower timeframe structure is the more compressed of the two"),
                    units="fraction", normalized=True,
                    source_columns=f"{lo_tf}__ema_entanglement_fraction,"
                                   f"{hi_tf}__ema_entanglement_fraction"),
                ("entangle_diff", lo_tf, hi_tf),
            )

            w = cfg.consolidation_windows[0]
            self._add(
                FeatureSpec(
                    f"x_{lo_tf}_{hi_tf}__ltf_compressed_htf_trending", "cross_tf_structure",
                    timeframe=f"{lo_tf}|{hi_tf}", lookback=w,
                    description=(f"1 when {lo_tf} is compressed (trailing percentile) while the "
                                 f"{hi_tf} EMA stack is strongly ordered "
                                 f"(|ordering score| >= {cfg.trend_ordering_threshold})"),
                    units="boolean",
                    source_columns=f"{lo_tf}__is_compressed_{w},{hi_tf}__ema_ordering_score"),
                ("ltf_comp_htf_trend", lo_tf, hi_tf, w),
            )
            self._add(
                FeatureSpec(
                    f"x_{lo_tf}_{hi_tf}__ltf_expanding_htf_compressed", "cross_tf_structure",
                    timeframe=f"{lo_tf}|{hi_tf}", lookback=w,
                    description=f"1 when {lo_tf} is in an expanded range state while {hi_tf} is compressed",
                    units="boolean",
                    source_columns=f"{lo_tf}__is_expanded_{w},{hi_tf}__is_compressed_{w}"),
                ("ltf_exp_htf_comp", lo_tf, hi_tf, w),
            )
            self._add(
                FeatureSpec(
                    f"x_{lo_tf}_{hi_tf}__both_compressed", "cross_tf_structure",
                    timeframe=f"{lo_tf}|{hi_tf}", lookback=w,
                    description=f"1 when both {lo_tf} and {hi_tf} are in a compressed range state",
                    units="boolean",
                    source_columns=f"{lo_tf}__is_compressed_{w},{hi_tf}__is_compressed_{w}"),
                ("both_comp", lo_tf, hi_tf, w),
            )
            self._add(
                FeatureSpec(
                    f"x_{lo_tf}_{hi_tf}__ltf_bull_htf_bear", "cross_tf_structure",
                    timeframe=f"{lo_tf}|{hi_tf}",
                    description=(f"1 when the {lo_tf} EMA stack is bullish-ordered while the "
                                 f"{hi_tf} stack is bearish-ordered (counter-trend lower timeframe)"),
                    units="boolean",
                    source_columns=f"{lo_tf}__ema_ordering_score,{hi_tf}__ema_ordering_score"),
                ("ltf_bull_htf_bear", lo_tf, hi_tf),
            )

        # --- whole-stack summaries across all timeframes -------------------
        self._add(
            FeatureSpec("mtf__n_timeframes_bullish", "cross_tf_direction",
                        timeframe="all",
                        description="Number of active timeframes whose EMA stack is bullish-ordered",
                        units="count",
                        source_columns=",".join(f"{t}__ema_ordering_score" for t in tfs)),
            ("n_bull", tfs),
        )
        self._add(
            FeatureSpec("mtf__n_timeframes_bearish", "cross_tf_direction",
                        timeframe="all",
                        description="Number of active timeframes whose EMA stack is bearish-ordered",
                        units="count",
                        source_columns=",".join(f"{t}__ema_ordering_score" for t in tfs)),
            ("n_bear", tfs),
        )
        self._add(
            FeatureSpec("mtf__direction_disagreement", "cross_tf_direction",
                        timeframe="all",
                        description=("Dispersion of EMA ordering scores across timeframes: 0 means "
                                     "every timeframe agrees, larger means they conflict"),
                        units="score_sd", normalized=True,
                        source_columns=",".join(f"{t}__ema_ordering_score" for t in tfs)),
            ("disagree", tfs),
        )
        self._add(
            FeatureSpec("mtf__mean_ordering_score", "cross_tf_direction",
                        timeframe="all",
                        description="Mean EMA ordering score across all active timeframes",
                        units="score_-1_to_1", normalized=True,
                        source_columns=",".join(f"{t}__ema_ordering_score" for t in tfs)),
            ("mean_order", tfs),
        )
        w0 = cfg.consolidation_windows[0]
        self._add(
            FeatureSpec("mtf__n_timeframes_compressed", "cross_tf_structure",
                        timeframe="all", lookback=w0,
                        description="Number of active timeframes currently in a compressed range state",
                        units="count",
                        source_columns=",".join(f"{t}__is_compressed_{w0}" for t in tfs)),
            ("n_comp", tfs, w0),
        )

        log.info("Cross-timeframe feature plan: %d features over %d timeframe pairs (mode=%s)",
                 len(self.specs), len(self.pairs), cfg.cross_tf_mode)

    # ------------------------------------------------------------------
    def build(self, st: ChunkState) -> dict[str, np.ndarray]:
        cfg = self.cfg
        pip = cfg.pip_size
        ki = self.key_index
        out: dict[str, np.ndarray] = {}

        def named(tf: str, feat: str) -> np.ndarray:
            return st.named.get(f"{tf}__{feat}", np.full(st.n_rows, np.nan))

        for spec, op in zip(self.specs, self._plan):
            kind = op[0]

            if kind in ("val_dist_pips", "val_dist_atr", "slope_diff_atr",
                        "slope_sign_agree", "slope_ratio"):
                _, lo_tf, hi_tf, a, b = op
                if kind.startswith("val"):
                    d = st.values[lo_tf][:, ki[a]] - st.values[hi_tf][:, ki[b]]
                    res = d / pip if kind == "val_dist_pips" else safe_divide(d, st.atr[lo_tf])
                else:
                    sa = st.slopes_atr[lo_tf][:, ki[a]]
                    sb = st.slopes_atr[hi_tf][:, ki[b]]
                    if kind == "slope_diff_atr":
                        res = sa - sb
                    elif kind == "slope_sign_agree":
                        res = np.where(
                            np.isfinite(sa) & np.isfinite(sb),
                            (np.sign(sa) == np.sign(sb)).astype("float64"), np.nan)
                    else:
                        res = safe_divide(sa, sb)

            elif kind == "p2h_pips":
                _, hi_tf, key = op
                res = (st.base_price - st.values[hi_tf][:, ki[key]]) / pip
            elif kind == "p2h_atr":
                _, hi_tf, key = op
                res = safe_divide(st.base_price - st.values[hi_tf][:, ki[key]], st.atr[hi_tf])

            elif kind == "trend_agree":
                _, lo_tf, hi_tf = op
                a = _sign(named(lo_tf, "ema_ordering_score"))
                b = _sign(named(hi_tf, "ema_ordering_score"))
                res = a * b
            elif kind == "ordering_diff":
                _, lo_tf, hi_tf = op
                res = named(lo_tf, "ema_ordering_score") - named(hi_tf, "ema_ordering_score")
            elif kind == "width_ratio":
                _, lo_tf, hi_tf = op
                res = safe_divide(named(lo_tf, "combined_ribbon_width_atr"),
                                  named(hi_tf, "combined_ribbon_width_atr"))
            elif kind == "density_ratio":
                _, lo_tf, hi_tf, w = op
                res = safe_divide(named(lo_tf, f"ema_cross_density_{w}"),
                                  named(hi_tf, f"ema_cross_density_{w}"))
            elif kind == "entangle_diff":
                _, lo_tf, hi_tf = op
                res = named(lo_tf, "ema_entanglement_fraction") - named(hi_tf, "ema_entanglement_fraction")
            elif kind == "ltf_comp_htf_trend":
                _, lo_tf, hi_tf, w = op
                c = named(lo_tf, f"is_compressed_{w}")
                o = named(hi_tf, "ema_ordering_score")
                res = np.where(np.isfinite(c) & np.isfinite(o),
                               ((c > 0.5) & (np.abs(o) >= cfg.trend_ordering_threshold)
                                ).astype("float64"), np.nan)
            elif kind == "ltf_exp_htf_comp":
                _, lo_tf, hi_tf, w = op
                e = named(lo_tf, f"is_expanded_{w}")
                c = named(hi_tf, f"is_compressed_{w}")
                res = np.where(np.isfinite(e) & np.isfinite(c),
                               ((e > 0.5) & (c > 0.5)).astype("float64"), np.nan)
            elif kind == "both_comp":
                _, lo_tf, hi_tf, w = op
                a = named(lo_tf, f"is_compressed_{w}")
                b = named(hi_tf, f"is_compressed_{w}")
                res = np.where(np.isfinite(a) & np.isfinite(b),
                               ((a > 0.5) & (b > 0.5)).astype("float64"), np.nan)
            elif kind == "ltf_bull_htf_bear":
                _, lo_tf, hi_tf = op
                a = named(lo_tf, "ema_ordering_score")
                b = named(hi_tf, "ema_ordering_score")
                res = np.where(np.isfinite(a) & np.isfinite(b),
                               ((a > 0) & (b < 0)).astype("float64"), np.nan)

            elif kind in ("n_bull", "n_bear", "disagree", "mean_order"):
                tfs = op[1]
                mat = np.column_stack([named(t, "ema_ordering_score") for t in tfs])
                fin = np.isfinite(mat)
                if kind == "n_bull":
                    res = np.where(fin, mat > 0, False).sum(axis=1).astype("float64")
                elif kind == "n_bear":
                    res = np.where(fin, mat < 0, False).sum(axis=1).astype("float64")
                elif kind == "mean_order":
                    with np.errstate(invalid="ignore"), quiet_warmup_warnings():
                        res = np.nanmean(mat, axis=1)
                else:
                    with np.errstate(invalid="ignore"), quiet_warmup_warnings():
                        res = np.nanstd(mat, axis=1, ddof=0)
                res = np.where(fin.any(axis=1), res, np.nan)
            elif kind == "n_comp":
                tfs, w = op[1], op[2]
                mat = np.column_stack([named(t, f"is_compressed_{w}") for t in tfs])
                fin = np.isfinite(mat)
                res = np.where(fin, mat > 0.5, False).sum(axis=1).astype("float64")
                res = np.where(fin.any(axis=1), res, np.nan)
            else:  # pragma: no cover - plan/build must stay in sync
                raise RuntimeError(f"Unhandled cross-timeframe op: {kind}")

            out[spec.feature_name] = np.asarray(res, dtype="float64")

        return out

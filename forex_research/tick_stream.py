from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

from .logging_utils import get_logger
from .tick_schema import (
    TICK_FLAG_ASK,
    TICK_FLAG_BID,
    TickFileFormat,
    parse_datetime_ms,
    parse_int_column,
    parse_price_points,
)

log = get_logger("tick_stream")

MS_PER_DAY = 86_400_000
MS_PER_5MIN = 300_000

#: Spread histogram ceiling, in points. Values above land in an overflow bucket.
SPREAD_HIST_MAX = 2_000

#: Gap thresholds reported as counts, in minutes.
GAP_THRESHOLDS_MIN = (1, 5, 30, 120, 720, 2880)

#: Tick-to-tick jump above this many points is recorded as a price anomaly.
JUMP_ANOMALY_POINTS = 500          # 50 pips between consecutive ticks

#: Largest equal-timestamp group carried across a batch boundary before the
#: group is treated as evidence of a corrupt timestamp rather than real data.
MAX_TIMESTAMP_GROUP = 200_000

_TOP_N = 200


# ---------------------------------------------------------------------------
# Broker-clock conversion for the "ticks are UTC" hypothesis
# ---------------------------------------------------------------------------

def build_broker_offset_table() -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    hours = pd.date_range("2020-01-01", "2030-01-01", freq="h", tz="UTC")
    ny_offsets = np.asarray(
        hours.tz_convert("America/New_York").map(lambda t: t.utcoffset().total_seconds()),
        dtype=np.int64,
    )
    native_offset_s = ny_offsets + 7 * 3600          # NY wall + 7h
    # Unit-agnostic: pandas 3 defaults to microsecond resolution, older versions
    # to nanosecond, so dividing by a Timedelta is the portable conversion.
    epoch_ms = ((hours.tz_localize(None) - pd.Timestamp(0))
                // pd.Timedelta(milliseconds=1)).to_numpy().astype(np.int64)

    change = np.r_[True, native_offset_s[1:] != native_offset_s[:-1]]
    return epoch_ms[change], (native_offset_s[change] * 1000).astype(np.int64)


def utc_ms_to_broker_native_ms(
    epoch_ms: np.ndarray, transitions: np.ndarray, offsets_ms: np.ndarray
) -> np.ndarray:
    idx = np.searchsorted(transitions, epoch_ms, side="right") - 1
    idx = np.clip(idx, 0, len(offsets_ms) - 1)
    return epoch_ms + offsets_ms[idx]


# ---------------------------------------------------------------------------
# Batch iteration
# ---------------------------------------------------------------------------

@dataclass
class TickBatch:

    epoch_ms: np.ndarray          # int64, INT64_MIN where unparseable
    ts_ok: np.ndarray             # bool
    bid_pts: np.ndarray           # int64, -1 where structurally absent
    bid_present: np.ndarray       # bool
    ask_pts: np.ndarray
    ask_present: np.ndarray
    flags: np.ndarray             # int64, -1 absent
    flags_present: np.ndarray
    last_present: np.ndarray
    volume_present: np.ndarray
    n_rows: int


class InvalidRowCollector:

    def __init__(self, keep: int = 50) -> None:
        self.count = 0
        self.examples: list[str] = []
        self._keep = keep

    def __call__(self, row) -> str:
        self.count += 1
        if len(self.examples) < self._keep:
            self.examples.append(
                f"line~{row.number} expected={row.expected_columns} "
                f"actual={row.actual_columns}: {str(row.text)[:200]}"
            )
        return "skip"


def iter_tick_batches(
    fmt: TickFileFormat,
    point_size: float,
    block_size: int = 64 << 20,
    max_rows: int | None = None,
    start_offset: int = 0,
) -> Iterator[tuple[TickBatch, InvalidRowCollector]]:
    collector = InvalidRowCollector()
    read_opts = pacsv.ReadOptions(block_size=block_size, use_threads=True)
    if start_offset:
        read_opts.column_names = list(fmt.raw_columns)
    parse_opts = pacsv.ParseOptions(
        delimiter=fmt.delimiter, invalid_row_handler=collector)
    convert_opts = pacsv.ConvertOptions(
        column_types={c: pa.string() for c in fmt.raw_columns},
        strings_can_be_null=True,
    )

    name = {n: raw for n, raw in zip(fmt.columns, fmt.raw_columns)}
    emitted = 0

    handle = fmt.path.open("rb")
    if start_offset:
        handle.seek(start_offset)  # already a line start; nothing to discard

    with handle, pacsv.open_csv(handle, read_options=read_opts,
                                parse_options=parse_opts,
                                convert_options=convert_opts) as reader:
        for rb in reader:
            n = rb.num_rows
            if n == 0:
                continue
            if max_rows is not None and emitted + n > max_rows:
                rb = rb.slice(0, max_rows - emitted)
                n = rb.num_rows
                if n == 0:
                    break

            epoch_ms, ts_ok = parse_datetime_ms(
                rb.column(name["DATE"]), rb.column(name["TIME"]), fmt.time_width)
            bid_pts, bid_present = parse_price_points(rb.column(name["BID"]), point_size)
            ask_pts, ask_present = parse_price_points(rb.column(name["ASK"]), point_size)

            if "FLAGS" in name:
                flags, flags_present = parse_int_column(rb.column(name["FLAGS"]))
            else:
                flags = np.full(n, -1, dtype=np.int64)
                flags_present = np.zeros(n, dtype=bool)

            if "LAST" in name:
                _, last_present = parse_price_points(rb.column(name["LAST"]), point_size)
            else:
                last_present = np.zeros(n, dtype=bool)
            if "VOLUME" in name:
                _, vol_present = parse_int_column(rb.column(name["VOLUME"]))
            else:
                vol_present = np.zeros(n, dtype=bool)

            emitted += n
            yield TickBatch(
                epoch_ms=epoch_ms, ts_ok=ts_ok,
                bid_pts=bid_pts, bid_present=bid_present,
                ask_pts=ask_pts, ask_present=ask_present,
                flags=flags, flags_present=flags_present,
                last_present=last_present, volume_present=vol_present,
                n_rows=n,
            ), collector

            if max_rows is not None and emitted >= max_rows:
                break


# ---------------------------------------------------------------------------
# Single-pass accumulator
# ---------------------------------------------------------------------------

def _ffill(values: np.ndarray, present: np.ndarray, carry: int) -> np.ndarray:
    idx = np.where(present, np.arange(len(values)), -1)
    np.maximum.accumulate(idx, out=idx)
    out = np.where(idx >= 0, values[np.maximum(idx, 0)], carry)
    return out


def _run_groups(sorted_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = sorted_vals.size
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    starts = np.flatnonzero(np.r_[True, sorted_vals[1:] != sorted_vals[:-1]])
    counts = np.diff(np.r_[starts, n])
    return sorted_vals[starts], starts, counts


def _is_sorted(a: np.ndarray) -> bool:
    return a.size < 2 or bool(np.all(a[1:] >= a[:-1]))


@dataclass
class M5Reconstruction:

    label: str
    description: str
    base_bin: int
    n_bins: int
    open_: np.ndarray = field(init=False)
    high: np.ndarray = field(init=False)
    low: np.ndarray = field(init=False)
    close: np.ndarray = field(init=False)
    count: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.open_ = np.zeros(self.n_bins, dtype=np.int64)
        self.high = np.full(self.n_bins, np.iinfo(np.int64).min, dtype=np.int64)
        self.low = np.full(self.n_bins, np.iinfo(np.int64).max, dtype=np.int64)
        self.close = np.zeros(self.n_bins, dtype=np.int64)
        self.count = np.zeros(self.n_bins, dtype=np.int64)

    def update(self, bin_abs: np.ndarray, bid: np.ndarray) -> None:
        rel = bin_abs - self.base_bin
        keep = (rel >= 0) & (rel < self.n_bins)
        if not keep.any():
            return
        rel = rel[keep]
        bid = bid[keep]

        if not _is_sorted(rel):
            order = np.argsort(rel, kind="stable")
            rel, bid = rel[order], bid[order]

        uniq, first_idx, counts = _run_groups(rel)
        grp_open = bid[first_idx]
        grp_max = np.maximum.reduceat(bid, first_idx)
        grp_min = np.minimum.reduceat(bid, first_idx)
        last_idx = np.r_[first_idx[1:] - 1, len(bid) - 1]
        grp_close = bid[last_idx]

        fresh = self.count[uniq] == 0
        if fresh.any():
            self.open_[uniq[fresh]] = grp_open[fresh]
        self.high[uniq] = np.maximum(self.high[uniq], grp_max)
        self.low[uniq] = np.minimum(self.low[uniq], grp_min)
        self.close[uniq] = grp_close
        self.count[uniq] += counts


class TickAuditState:

    def __init__(
        self,
        fmt: TickFileFormat,
        point_size: float,
        pip_size: float,
        span_start_ms: int,
        span_end_ms: int,
        stale_side_ms: int = 60_000,
    ) -> None:
        self.fmt = fmt
        self.point_size = point_size
        self.pip_size = pip_size
        self.points_per_pip = int(round(pip_size / point_size))
        self.stale_side_ms = stale_side_ms

        # --- counts -------------------------------------------------------
        self.n_rows = 0
        self.n_ts_malformed = 0
        self.n_bid_structurally_absent = 0
        self.n_ask_structurally_absent = 0
        self.n_both_sides_absent = 0
        self.n_last_present = 0
        self.n_volume_present = 0
        self.flag_counts: dict[int, int] = {}
        self.n_flag_bid_mismatch = 0
        self.n_flag_ask_mismatch = 0
        self.n_flags_absent = 0

        # --- price validity ----------------------------------------------
        self.n_bid_nonpositive = 0
        self.n_ask_nonpositive = 0
        self.bid_min = np.iinfo(np.int64).max
        self.bid_max = np.iinfo(np.int64).min
        self.ask_min = np.iinfo(np.int64).max
        self.ask_max = np.iinfo(np.int64).min

        # --- quote book (carried across batches) --------------------------
        self.last_bid = -1
        self.last_ask = -1
        self.last_bid_ts = -1
        self.last_ask_ts = -1

        # --- spread --------------------------------------------------------
        self.spread_hist = np.zeros(SPREAD_HIST_MAX + 2, dtype=np.int64)
        self.spread_hist_fresh = np.zeros(SPREAD_HIST_MAX + 2, dtype=np.int64)
        self.spread_overflow = 0
        self.n_spread_negative = 0
        self.n_spread_zero = 0
        self.n_quotes_reconstructed = 0
        self.n_quotes_stale_side = 0
        self.n_quotes_undefined = 0
        self.spread_sum = 0
        self.spread_min = np.iinfo(np.int64).max
        self.spread_max = np.iinfo(np.int64).min

        # --- timestamps ----------------------------------------------------
        self.ts_min = np.iinfo(np.int64).max
        self.ts_max = np.iinfo(np.int64).min
        self.prev_ts = -1
        self.n_out_of_order = 0
        self.n_duplicate_ts_rows = 0
        self.n_exact_duplicate_rows = 0
        self.max_ts_group = 0
        self.n_oversized_ts_groups = 0
        self.top_gaps: list[tuple[int, int, int]] = []
        self.gap_counts = {m: 0 for m in GAP_THRESHOLDS_MIN}
        self.n_subsecond = 0

        # --- carry for equal-timestamp group spanning a batch boundary ----
        self._carry: dict[str, np.ndarray] | None = None

        # --- calendar-indexed coverage -------------------------------------
        self.span_start_ms = span_start_ms
        self.base_day = span_start_ms // MS_PER_DAY
        self.n_days = int((span_end_ms - span_start_ms) // MS_PER_DAY) + 3
        self.day_count = np.zeros(self.n_days, dtype=np.int64)
        self.day_first_ms = np.full(self.n_days, -1, dtype=np.int64)
        self.day_last_ms = np.full(self.n_days, -1, dtype=np.int64)
        self.day_max_gap_ms = np.zeros(self.n_days, dtype=np.int64)
        self.day_spread_sum = np.zeros(self.n_days, dtype=np.int64)
        self.day_spread_n = np.zeros(self.n_days, dtype=np.int64)
        #: Per-day spread histogram. Capped; the final bin is an overflow bucket
        #: whose share is reported rather than silently folded into the tail.
        self.day_spread_cap = 512
        self.day_spread_hist = np.zeros((self.n_days, self.day_spread_cap + 2),
                                        dtype=np.int32)

        self.hour_count = np.zeros(self.n_days * 24, dtype=np.int64)
        self.hour_of_day_spread = np.zeros((24, SPREAD_HIST_MAX + 2), dtype=np.int64)

        # UTC -> file-native transition table for the "ticks are UTC" hypothesis.
        self._tz_transitions, self._tz_offsets_ms = build_broker_offset_table()

        # --- tick density via run-length over the sorted stream ------------
        self._cur_minute = -1
        self._cur_minute_n = 0
        self.minute_count_hist: dict[int, int] = {}

        # --- price anomalies -----------------------------------------------
        self.top_jumps: list[tuple[int, int, int, int, str]] = []
        self.n_jumps_over_threshold = 0
        self.bid_update_gap_max = 0
        self.ask_update_gap_max = 0
        self.n_bid_unchanged = 0

        # --- weekly session boundaries (timezone evidence) -----------------
        self.week_first_ms: dict[int, int] = {}
        self.week_last_ms: dict[int, int] = {}

        # --- byte/row landmarks for sampling -------------------------------
        self.best_hour_index = -1
        self.best_hour_count = 0

        self.m5: dict[str, M5Reconstruction] = {}

    # -- registration ------------------------------------------------------
    def add_m5_interpretation(self, label: str, description: str) -> None:
        base_bin = self.span_start_ms // MS_PER_5MIN - 288
        n_bins = self.n_days * 288 + 576
        self.m5[label] = M5Reconstruction(label, description, base_bin, n_bins)

    # -- the hot path ------------------------------------------------------
    def update(self, b: TickBatch) -> None:
        n = b.n_rows
        self.n_rows += n

        bad_ts = ~b.ts_ok
        if bad_ts.any():
            self.n_ts_malformed += int(bad_ts.sum())

        ok = b.ts_ok
        if not ok.any():
            return
        ts = b.epoch_ms[ok]
        bid_p = b.bid_pts[ok]
        bid_ok = b.bid_present[ok]
        ask_p = b.ask_pts[ok]
        ask_ok = b.ask_present[ok]
        flags = b.flags[ok]
        flags_ok = b.flags_present[ok]

        # --- field presence accounting -------------------------------------
        self.n_bid_structurally_absent += int((~bid_ok).sum())
        self.n_ask_structurally_absent += int((~ask_ok).sum())
        self.n_both_sides_absent += int((~bid_ok & ~ask_ok).sum())
        self.n_last_present += int(b.last_present[ok].sum())
        self.n_volume_present += int(b.volume_present[ok].sum())
        self.n_flags_absent += int((~flags_ok).sum())

        if flags_ok.any():
            fv = flags[flags_ok]
            # Flags are a small bitmask, so a bincount beats a sort-based unique.
            bc = np.bincount(np.clip(fv, 0, 255))
            for v in np.flatnonzero(bc):
                self.flag_counts[int(v)] = self.flag_counts.get(int(v), 0) + int(bc[v])
            expect_bid = (flags & TICK_FLAG_BID) != 0
            expect_ask = (flags & TICK_FLAG_ASK) != 0
            self.n_flag_bid_mismatch += int((flags_ok & (expect_bid != bid_ok)).sum())
            self.n_flag_ask_mismatch += int((flags_ok & (expect_ask != ask_ok)).sum())

        # --- price validity -------------------------------------------------
        if bid_ok.any():
            v = bid_p[bid_ok]
            self.n_bid_nonpositive += int((v <= 0).sum())
            self.bid_min = min(self.bid_min, int(v.min()))
            self.bid_max = max(self.bid_max, int(v.max()))
        if ask_ok.any():
            v = ask_p[ask_ok]
            self.n_ask_nonpositive += int((v <= 0).sum())
            self.ask_min = min(self.ask_min, int(v.min()))
            self.ask_max = max(self.ask_max, int(v.max()))

        # --- quote book reconstruction, carrying state across batches -------
        sync_bid = _ffill(bid_p, bid_ok, self.last_bid)
        sync_ask = _ffill(ask_p, ask_ok, self.last_ask)
        bid_ts = _ffill(ts, bid_ok, self.last_bid_ts)
        ask_ts = _ffill(ts, ask_ok, self.last_ask_ts)

        have_bid = sync_bid > 0
        have_ask = sync_ask > 0
        have_both = have_bid & have_ask
        self.n_quotes_undefined += int((~have_both).sum())

        if have_both.any():
            sp = (sync_ask - sync_bid)[have_both]
            bid_age = (ts - bid_ts)[have_both]
            ask_age = (ts - ask_ts)[have_both]
            fresh = (bid_age <= self.stale_side_ms) & (ask_age <= self.stale_side_ms)

            self.n_quotes_reconstructed += int(have_both.sum())
            self.n_quotes_stale_side += int((~fresh).sum())
            self.n_spread_negative += int((sp < 0).sum())
            self.n_spread_zero += int((sp == 0).sum())
            self.spread_sum += int(sp.sum())
            self.spread_min = min(self.spread_min, int(sp.min()))
            self.spread_max = max(self.spread_max, int(sp.max()))

            clipped = np.clip(sp, 0, SPREAD_HIST_MAX + 1)
            self.spread_hist += np.bincount(clipped, minlength=SPREAD_HIST_MAX + 2)
            self.spread_overflow += int((sp > SPREAD_HIST_MAX).sum())
            if fresh.any():
                self.spread_hist_fresh += np.bincount(
                    clipped[fresh], minlength=SPREAD_HIST_MAX + 2)

            self.bid_update_gap_max = max(self.bid_update_gap_max, int(bid_age.max()))
            self.ask_update_gap_max = max(self.ask_update_gap_max, int(ask_age.max()))

            # By hour of day. Yearly and monthly breakdowns are derived from the
            # per-day histograms at report time rather than accumulated here.
            # A flat bincount replaces np.add.at, which is unbuffered and an
            # order of magnitude slower on arrays this size.
            ts_both = ts[have_both]
            hod = ((ts_both // 3_600_000) % 24).astype(np.int64)
            width = SPREAD_HIST_MAX + 2
            flat = hod * width + clipped
            self.hour_of_day_spread += np.bincount(
                flat, minlength=24 * width).reshape(24, width)

        # --- timestamps: ordering, duplicates, gaps -------------------------
        self.ts_min = min(self.ts_min, int(ts.min()))
        self.ts_max = max(self.ts_max, int(ts.max()))
        self.n_subsecond += int((ts % 1000 != 0).sum())

        prev = np.r_[self.prev_ts if self.prev_ts >= 0 else ts[0], ts[:-1]]
        delta = ts - prev
        self.n_out_of_order += int((delta < 0).sum())

        pos = delta > 0
        if pos.any():
            for m in GAP_THRESHOLDS_MIN:
                self.gap_counts[m] += int((delta >= m * 60_000).sum())
            big = np.flatnonzero(delta >= GAP_THRESHOLDS_MIN[0] * 60_000)
            for i in big:
                item = (int(delta[i]), int(prev[i]), int(ts[i]))
                if len(self.top_gaps) < _TOP_N:
                    heapq.heappush(self.top_gaps, item)
                elif item[0] > self.top_gaps[0][0]:
                    heapq.heapreplace(self.top_gaps, item)

        self._update_duplicates(ts, sync_bid, sync_ask, bid_p, ask_p,
                                bid_ok, ask_ok, flags)

        # --- price jumps -----------------------------------------------------
        if have_bid.any():
            prev_bid = np.r_[self.last_bid if self.last_bid > 0 else sync_bid[0],
                             sync_bid[:-1]]
            jump = np.abs(sync_bid - prev_bid)
            valid_jump = have_bid & (prev_bid > 0)
            self.n_bid_unchanged += int((valid_jump & (jump == 0)).sum())
            big = np.flatnonzero(valid_jump & (jump >= JUMP_ANOMALY_POINTS))
            self.n_jumps_over_threshold += len(big)
            for i in big:
                item = (int(jump[i]), int(ts[i]), int(prev_bid[i]), int(sync_bid[i]), "bid")
                if len(self.top_jumps) < _TOP_N:
                    heapq.heappush(self.top_jumps, item)
                elif item[0] > self.top_jumps[0][0]:
                    heapq.heapreplace(self.top_jumps, item)

        # --- calendar coverage ------------------------------------------------
        days = (ts // MS_PER_DAY - self.base_day).astype(np.int64)
        in_range = (days >= 0) & (days < self.n_days)
        d = days[in_range]
        if d.size:
            self.day_count += np.bincount(d, minlength=self.n_days)
            tsr = ts[in_range]
            dl = np.maximum(delta[in_range], 0)

            if _is_sorted(d):
                # Fast path. Timestamps ascending means the first row of each
                # day-group is its minimum and the last is its maximum, so group
                # boundaries give both without any scatter-reduction.
                ud, starts, _ = _run_groups(d)
                ends = np.r_[starts[1:] - 1, d.size - 1]
                first_here, last_here = tsr[starts], tsr[ends]
                unset = self.day_first_ms[ud] < 0
                if unset.any():
                    self.day_first_ms[ud[unset]] = first_here[unset]
                self.day_first_ms[ud] = np.minimum(self.day_first_ms[ud], first_here)
                self.day_last_ms[ud] = np.maximum(self.day_last_ms[ud], last_here)
                self.day_max_gap_ms[ud] = np.maximum(
                    self.day_max_gap_ms[ud], np.maximum.reduceat(dl, starts))
            else:
                # Out-of-order timestamps: fall back to correct-but-slower
                # scatter reductions rather than trusting the ordering.
                unset = self.day_first_ms[d] < 0
                if unset.any():
                    self.day_first_ms[d[unset]] = tsr[unset]
                np.minimum.at(self.day_first_ms, d, tsr)
                np.maximum.at(self.day_last_ms, d, tsr)
                np.maximum.at(self.day_max_gap_ms, d, dl)

            hours = ((ts[in_range] // 3_600_000) - self.base_day * 24).astype(np.int64)
            hok = (hours >= 0) & (hours < len(self.hour_count))
            if hok.any():
                self.hour_count += np.bincount(hours[hok], minlength=len(self.hour_count))

            if have_both.any():
                dboth = (ts[have_both] // MS_PER_DAY - self.base_day).astype(np.int64)
                dok = (dboth >= 0) & (dboth < self.n_days)
                if dok.any():
                    spd = sp[dok]
                    dd = dboth[dok]
                    self.day_spread_sum += np.bincount(
                        dd, weights=spd, minlength=self.n_days).astype(np.int64)
                    self.day_spread_n += np.bincount(dd, minlength=self.n_days)
                    w = self.day_spread_cap + 2
                    flat = dd * w + np.clip(spd, 0, self.day_spread_cap + 1)
                    self.day_spread_hist += np.bincount(
                        flat, minlength=self.n_days * w
                    ).reshape(self.n_days, w).astype(np.int32)

        # Weekly session boundaries are derived from the per-day arrays in
        # finalize() rather than accumulated here: the weekly first/last tick is
        # just the min/max over that week's daily first/last, so computing it
        # per batch would repeat work on every one of ~80 blocks.

        # --- ticks per minute, run-length over the sorted stream --------------
        self._update_minute_density(ts)

        # --- M5 reconstruction, one fold per timestamp interpretation ----------
        # Only two hypotheses are tested, both declared before any result was
        # seen: the tick clock already IS the bar clock, or the tick clock is
        # UTC and needs the project's documented broker-clock conversion. No
        # offset is searched for the value that happens to score best.
        for label, recon in self.m5.items():
            if label == "A_native":
                native = ts
            elif label == "B_utc":
                native = utc_ms_to_broker_native_ms(
                    ts, self._tz_transitions, self._tz_offsets_ms)
            else:
                raise ValueError(f"Unregistered M5 interpretation: {label!r}")
            recon.update(native // MS_PER_5MIN, sync_bid)

        # --- carry the quote book forward --------------------------------------
        if have_bid.any():
            self.last_bid = int(sync_bid[-1])
            self.last_bid_ts = int(bid_ts[-1])
        if have_ask.any():
            self.last_ask = int(sync_ask[-1])
            self.last_ask_ts = int(ask_ts[-1])
        self.prev_ts = int(ts[-1])

    # -- duplicate detection within equal-timestamp groups --------------------
    def _update_duplicates(
        self, ts, sync_bid, sync_ask, bid_p, ask_p, bid_ok, ask_ok, flags
    ) -> None:
        key = np.stack([
            ts,
            np.where(bid_ok, bid_p, -1),
            np.where(ask_ok, ask_p, -1),
            flags,
        ])

        if self._carry is not None:
            key = np.concatenate([self._carry["key"], key], axis=1)
        ts_all = key[0]

        if ts_all.size == 0:
            return

        # Split off the trailing, possibly-unfinished timestamp group.
        last_ts = ts_all[-1]
        cut = int(np.searchsorted(ts_all, last_ts, side="left")) \
            if np.all(np.diff(ts_all) >= 0) else ts_all.size
        if ts_all.size - cut > MAX_TIMESTAMP_GROUP:
            self.n_oversized_ts_groups += 1
            cut = ts_all.size
        done, carry = key[:, :cut], key[:, cut:]
        self._carry = {"key": carry} if carry.shape[1] else None

        self._count_group_duplicates(done)

    def _count_group_duplicates(self, key: np.ndarray) -> None:
        if key.shape[1] == 0:
            return
        ts = key[0]
        if _is_sorted(ts):
            uniq, starts, counts = _run_groups(ts)
        else:
            uniq, counts = np.unique(ts, return_counts=True)
            starts = None
        if counts.size:
            self.max_ts_group = max(self.max_ts_group, int(counts.max()))
        multi = counts > 1
        if not multi.any():
            return
        self.n_duplicate_ts_rows += int((counts[multi] - 1).sum())

        if starts is not None:
            # Expand only the multi-row groups, without an O(n log n) isin.
            sel = np.concatenate([np.arange(s, s + c)
                                  for s, c in zip(starts[multi], counts[multi])])
            sub = key[:, sel]
        else:
            sub = key[:, np.isin(ts, uniq[multi])]
        order = np.lexsort(tuple(sub[::-1]))
        s = sub[:, order]
        if s.shape[1] > 1:
            same = np.all(s[:, 1:] == s[:, :-1], axis=0)
            self.n_exact_duplicate_rows += int(same.sum())

    def _update_minute_density(self, ts: np.ndarray) -> None:
        minutes = ts // 60_000
        if _is_sorted(minutes):
            uniq, _, counts = _run_groups(minutes)
        else:
            uniq, counts = np.unique(minutes, return_counts=True)
        for m, c in zip(uniq.tolist(), counts.tolist()):
            if m == self._cur_minute:
                self._cur_minute_n += c
            else:
                if self._cur_minute >= 0:
                    k = self._cur_minute_n
                    self.minute_count_hist[k] = self.minute_count_hist.get(k, 0) + 1
                self._cur_minute = m
                self._cur_minute_n = c

    def finalize(self) -> None:
        if self._carry is not None:
            self._count_group_duplicates(self._carry["key"])
            self._carry = None
        if self._cur_minute >= 0:
            k = self._cur_minute_n
            self.minute_count_hist[k] = self.minute_count_hist.get(k, 0) + 1
            self._cur_minute = -1
        if self.hour_count.size:
            i = int(np.argmax(self.hour_count))
            self.best_hour_index = i
            self.best_hour_count = int(self.hour_count[i])

        # Weekly first/last tick, from the per-day extremes. Day index 0 is
        # self.base_day; the +3 aligns the ISO week because 1970-01-01 was a
        # Thursday, so week boundaries land on Monday.
        seen = np.flatnonzero(self.day_count > 0)
        if seen.size:
            abs_days = self.base_day + seen
            weeks = (abs_days + 3) // 7
            order = np.argsort(weeks, kind="stable")
            w_sorted = weeks[order]
            first_sorted = self.day_first_ms[seen][order]
            last_sorted = self.day_last_ms[seen][order]
            uniq, starts, counts = _run_groups(w_sorted)
            ends = np.r_[starts[1:] - 1, w_sorted.size - 1]
            for w, s, e in zip(uniq.tolist(), starts.tolist(), ends.tolist()):
                self.week_first_ms[w] = int(first_sorted[s:e + 1].min())
                self.week_last_ms[w] = int(last_sorted[s:e + 1].max())


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------

def hist_quantiles(hist: np.ndarray, qs: tuple[float, ...]) -> dict[float, float]:
    total = int(hist.sum())
    if total == 0:
        return {q: float("nan") for q in qs}
    cum = np.cumsum(hist)
    out: dict[float, float] = {}
    for q in qs:
        target = q * total
        idx = int(np.searchsorted(cum, target, side="left"))
        out[q] = float(min(idx, len(hist) - 1))
    return out


def hist_mean(hist: np.ndarray) -> float:
    total = int(hist.sum())
    if total == 0:
        return float("nan")
    values = np.arange(len(hist), dtype=np.float64)
    return float((values * hist).sum() / total)


# ---------------------------------------------------------------------------
# Targeted sampling by byte bisection (no second full pass)
# ---------------------------------------------------------------------------

def _read_line_at(fh, offset: int, eol: bytes, probe: int = 1 << 16) -> tuple[int, bytes]:
    fh.seek(max(0, offset))
    buf = fh.read(probe)
    if not buf:
        return -1, b""
    nl = buf.find(eol)
    if nl < 0:
        return -1, b""
    start = offset + nl + len(eol)
    rest = buf[nl + len(eol):]
    nl2 = rest.find(eol)
    while nl2 < 0:
        more = fh.read(probe)
        if not more:
            return start, rest
        rest += more
        nl2 = rest.find(eol)
    return start, rest[:nl2]


def _line_timestamp_ms(line: bytes, fmt: TickFileFormat) -> int:
    parts = line.split(fmt.delimiter.encode())
    if len(parts) < 2:
        return -1
    d, t = parts[0], parts[1]
    try:
        y, mo, da = int(d[0:4]), int(d[5:7]), int(d[8:10])
        hh, mi, ss = int(t[0:2]), int(t[3:5]), int(t[6:8])
        ms = int(t[9:12]) if len(t) >= 12 else 0
    except (ValueError, IndexError):
        return -1
    from .tick_schema import _days_from_civil
    days = int(_days_from_civil(np.array([y]), np.array([mo]), np.array([da]))[0])
    return days * MS_PER_DAY + hh * 3_600_000 + mi * 60_000 + ss * 1000 + ms


def find_byte_offset_for_timestamp(fmt: TickFileFormat, target_ms: int) -> int:
    eol = {"CRLF": b"\r\n", "LF": b"\n", "CR": b"\r"}[fmt.line_ending]
    window = 1 << 13
    lo, hi = 0, fmt.size_bytes
    with fmt.path.open("rb") as fh:
        for _ in range(64):
            if hi - lo < window:
                break
            mid = (lo + hi) // 2
            start, line = _read_line_at(fh, mid, eol)
            if start < 0:
                hi = mid
                continue
            ts = _line_timestamp_ms(line, fmt)
            if ts < 0:
                hi = mid
                continue
            if ts < target_ms:
                lo = start
            else:
                hi = mid

        # Refine: bisection lands within `window` bytes. Walk forward from there
        # to the exact first line at or after the target, so the returned offset
        # is precise rather than "somewhere shortly before".
        fh.seek(lo)
        if lo == 0:
            fh.readline()                     # skip the header
        offset = fh.tell()
        while True:
            raw = fh.readline()
            if not raw:
                return offset
            ts = _line_timestamp_ms(raw.rstrip(eol), fmt)
            if ts >= target_ms:
                return offset
            offset = fh.tell()


def extract_sample(
    fmt: TickFileFormat, start_offset: int, n_rows: int, from_start: bool = False,
    offset_is_line_start: bool = True,
) -> tuple[list[str], str]:
    eol = {"CRLF": b"\r\n", "LF": b"\n", "CR": b"\r"}[fmt.line_ending]
    header = fmt.delimiter.join(fmt.raw_columns)
    lines: list[str] = []
    with fmt.path.open("rb") as fh:
        if from_start:
            fh.readline()                     # skip the header
        else:
            fh.seek(start_offset)
            if not offset_is_line_start:
                fh.readline()                 # discard a partial line
        for _ in range(n_rows):
            raw = fh.readline()
            if not raw:
                break
            lines.append(raw.rstrip(eol).decode(fmt.encoding, "replace"))
    return lines, header


def extract_tail_sample(fmt: TickFileFormat, n_rows: int) -> tuple[list[str], str]:
    eol = {"CRLF": b"\r\n", "LF": b"\n", "CR": b"\r"}[fmt.line_ending]
    header = fmt.delimiter.join(fmt.raw_columns)
    approx = int(fmt.mean_row_bytes * n_rows * 1.4) + (1 << 20)
    start = max(0, fmt.size_bytes - approx)
    with fmt.path.open("rb") as fh:
        fh.seek(start)
        if start:
            fh.readline()
        data = fh.read()
    lines = [ln.decode(fmt.encoding, "replace")
             for ln in data.split(eol) if ln]
    return lines[-n_rows:], header

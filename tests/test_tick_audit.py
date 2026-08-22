"""Quality control for the additive tick-audit modules.

Every test builds its own tiny synthetic MT5 tick file so expected values can be
computed by hand. **No test touches the multi-gigabyte client file**, and no test
depends on any Day 1 / Step 2 / Step 3A / Day 2 artefact existing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forex_research.tick_schema import (  # noqa: E402
    TICK_FLAG_ASK,
    TICK_FLAG_BID,
    TickFormatError,
    describe_flag,
    detect_format,
    parse_datetime_ms,
    parse_int_column,
    parse_price_points,
)
from forex_research.tick_stream import (  # noqa: E402
    MS_PER_5MIN,
    TickAuditState,
    _ffill,
    _run_groups,
    build_broker_offset_table,
    extract_sample,
    extract_tail_sample,
    find_byte_offset_for_timestamp,
    hist_mean,
    hist_quantiles,
    iter_tick_batches,
    utc_ms_to_broker_native_ms,
)
from forex_research.tick_validation import (  # noqa: E402
    compare_m5,
    load_m5_reference,
    research_overlap,
    weekly_session_table,
)

HEADER = "<DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>\t<FLAGS>"
POINT = 0.00001
PIP = 0.0001


def write_ticks(path: Path, rows: list[str], header: str = HEADER,
                eol: str = "\r\n") -> Path:
    path.write_bytes((eol.join([header] + rows) + eol).encode("ascii"))
    return path


def tick(date: str, time: str, bid: str = "", ask: str = "", flags: int = 6) -> str:
    return f"{date}\t{time}\t{bid}\t{ask}\t\t\t{flags}"


def ms(ts: str) -> int:
    return int((pd.Timestamp(ts) - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1))


def build_state(path: Path, first: str, last: str, **kw) -> TickAuditState:
    fmt = detect_format(path)
    st = TickAuditState(fmt, POINT, PIP, ms(first), ms(last), **kw)
    st.add_m5_interpretation("A_native", "native")
    for batch, _ in iter_tick_batches(fmt, POINT, block_size=1 << 16):
        st.update(batch)
    st.finalize()
    return st


# ---------------------------------------------------------------------------
# Format and schema detection
# ---------------------------------------------------------------------------

def test_detects_mt5_tick_export(tmp_path):
    p = write_ticks(tmp_path / "t.csv", [
        tick("2022.01.03", "00:00:00.916", "1.13684", "1.13819"),
        tick("2022.01.03", "00:00:01.166", "1.13644", "1.13794"),
        tick("2022.01.03", "00:00:02.818", "1.13650", "1.13800"),
    ])
    fmt = detect_format(p)
    assert fmt.is_mt5_tick_export
    assert fmt.delimiter == "\t"
    assert fmt.line_ending == "CRLF"
    assert fmt.encoding == "ascii"
    assert not fmt.has_bom
    assert fmt.columns == ("DATE", "TIME", "BID", "ASK", "LAST", "VOLUME", "FLAGS")
    assert fmt.raw_columns[0] == "<DATE>"          # original names preserved
    assert fmt.time_precision == "milliseconds"
    assert fmt.price_decimals == 5


def test_detects_comma_delimiter_and_lf(tmp_path):
    header = "<DATE>,<TIME>,<BID>,<ASK>,<LAST>,<VOLUME>,<FLAGS>"
    rows = [f"2022.01.03,00:00:0{i}.100,1.1368{i},1.1381{i},,,6" for i in range(4)]
    fmt = detect_format(write_ticks(tmp_path / "c.csv", rows, header, eol="\n"))
    assert fmt.delimiter == ","
    assert fmt.line_ending == "LF"


def test_detects_second_precision(tmp_path):
    rows = [tick("2022.01.03", f"00:00:0{i}", "1.13684", "1.13819") for i in range(4)]
    fmt = detect_format(write_ticks(tmp_path / "s.csv", rows))
    assert fmt.time_precision == "seconds"
    assert fmt.time_width == 8


def test_binary_file_is_refused_not_guessed(tmp_path):
    p = tmp_path / "bin.dat"
    p.write_bytes(b"\x00\x01\x02\x03" * 500 + b"\r\n" * 5)
    with pytest.raises(TickFormatError, match="BINARY"):
        detect_format(p)


def test_missing_header_is_refused(tmp_path):
    rows = [tick("2022.01.03", "00:00:00.916", "1.13684", "1.13819")] * 4
    p = write_ticks(tmp_path / "nh.csv", rows, header=rows[0])
    with pytest.raises(TickFormatError, match="header"):
        detect_format(p)


def test_estimated_row_count_is_in_the_right_order(tmp_path):
    rows = [tick("2022.01.03", f"00:00:{i:02d}.100", "1.13684", "1.13819")
            for i in range(60)]
    fmt = detect_format(write_ticks(tmp_path / "e.csv", rows))
    assert 50 <= fmt.estimated_rows <= 70


# ---------------------------------------------------------------------------
# Field parsing
# ---------------------------------------------------------------------------

def test_parse_datetime_matches_pandas():
    dates = pa.array(["2022.01.03", "2024.02.29", "2026.08.14", "2025.12.31"])
    times = pa.array(["00:00:00.916", "12:34:56.789", "11:52:42.546", "23:59:59.999"])
    got, ok = parse_datetime_ms(dates, times, 12)
    assert ok.all()
    want = [ms("2022-01-03 00:00:00.916"), ms("2024-02-29 12:34:56.789"),
            ms("2026-08-14 11:52:42.546"), ms("2025-12-31 23:59:59.999")]
    assert got.tolist() == want


def test_parse_datetime_flags_malformed_without_raising():
    dates = pa.array(["2022.01.03", "2022-01-03", "2022.13.03", None])
    times = pa.array(["00:00:00.916", "00:00:00.916", "00:00:00.916", "00:00:00.916"])
    _, ok = parse_datetime_ms(dates, times, 12)
    assert ok.tolist() == [True, False, False, False]


def test_parse_price_to_integer_points_is_exact():
    pts, present = parse_price_points(pa.array(["1.13684", "1.13819", None, "0.99999"]),
                                      POINT)
    assert pts.tolist() == [113684, 113819, -1, 99999]
    assert present.tolist() == [True, True, False, True]


def test_structurally_absent_price_is_not_a_parse_failure():
    """A blank BID on an ask-only tick must report absent, not malformed."""
    pts, present = parse_price_points(pa.array([None, "1.13800"]), POINT)
    assert present.tolist() == [False, True]
    assert pts[0] == -1


def test_parse_int_column_handles_nulls():
    vals, present = parse_int_column(pa.array(["6", "2", None, "4"]))
    assert vals.tolist() == [6, 2, -1, 4]
    assert present.tolist() == [True, True, False, True]


def test_describe_flag_decodes_bitmask():
    assert describe_flag(2) == "TICK_FLAG_BID"
    assert describe_flag(4) == "TICK_FLAG_ASK"
    assert describe_flag(6) == "TICK_FLAG_BID|TICK_FLAG_ASK"


# ---------------------------------------------------------------------------
# Chunked streaming
# ---------------------------------------------------------------------------

def test_chunked_parsing_counts_every_row(tmp_path):
    rows = [tick("2022.01.03", f"00:{i//60:02d}:{i%60:02d}.100", "1.13684", "1.13819")
            for i in range(500)]
    fmt = detect_format(write_ticks(tmp_path / "chunk.csv", rows))
    total = sum(b.n_rows for b, _ in iter_tick_batches(fmt, POINT, block_size=4096))
    assert total == 500


def test_streaming_never_materialises_the_whole_file(tmp_path, monkeypatch):
    """The audit path must not call a whole-file reader anywhere."""
    import pyarrow.csv as pacsv

    rows = [tick("2022.01.03", f"00:00:{i%60:02d}.{i%1000:03d}", "1.13684", "1.13819")
            for i in range(300)]
    fmt = detect_format(write_ticks(tmp_path / "nofull.csv", rows))

    def explode(*a, **k):                      # pragma: no cover - must not run
        raise AssertionError("read_csv loads the entire file into memory")

    monkeypatch.setattr(pacsv, "read_csv", explode)
    monkeypatch.setattr(pd, "read_csv", explode)
    assert sum(b.n_rows for b, _ in iter_tick_batches(fmt, POINT, block_size=2048)) == 300


def test_max_rows_bounds_the_stream(tmp_path):
    rows = [tick("2022.01.03", f"00:00:{i%60:02d}.{i%1000:03d}", "1.13684", "1.13819")
            for i in range(400)]
    fmt = detect_format(write_ticks(tmp_path / "cap.csv", rows))
    total = sum(b.n_rows for b, _ in iter_tick_batches(fmt, POINT, block_size=2048,
                                                       max_rows=150))
    assert total == 150


def test_malformed_rows_are_skipped_and_counted(tmp_path):
    rows = [tick("2022.01.03", "00:00:00.100", "1.13684", "1.13819"),
            "2022.01.03\t00:00:01.100\tbroken",            # too few fields
            tick("2022.01.03", "00:00:02.100", "1.13690", "1.13820")]
    fmt = detect_format(write_ticks(tmp_path / "bad.csv", rows))
    collector = None
    n = 0
    for b, collector in iter_tick_batches(fmt, POINT, block_size=1 << 16):
        n += b.n_rows
    assert n == 2
    assert collector.count == 1
    assert collector.examples


# ---------------------------------------------------------------------------
# Quote-book reconstruction and freshness
# ---------------------------------------------------------------------------

def test_ffill_carries_previous_value():
    vals = np.array([-1, 5, -1, -1, 9], dtype=np.int64)
    present = np.array([False, True, False, False, True])
    assert _ffill(vals, present, carry=3).tolist() == [3, 5, 5, 5, 9]


def test_sparse_sides_reconstruct_the_quote_book(tmp_path):
    """FLAGS 2/4/6: absent sides must be carried, not treated as missing."""
    rows = [
        tick("2025.06.02", "00:00:00.000", "1.10000", "1.10015", flags=6),
        tick("2025.06.02", "00:00:01.000", "", "1.10016", flags=TICK_FLAG_ASK),
        tick("2025.06.02", "00:00:02.000", "1.10002", "", flags=TICK_FLAG_BID),
        tick("2025.06.02", "00:00:03.000", "1.10003", "1.10018", flags=6),
    ]
    st = build_state(write_ticks(tmp_path / "sparse.csv", rows),
                     "2025-06-02", "2025-06-03")
    assert st.n_rows == 4
    assert st.n_bid_structurally_absent == 1
    assert st.n_ask_structurally_absent == 1
    assert st.n_flag_bid_mismatch == 0 and st.n_flag_ask_mismatch == 0
    # All four rows have a two-sided quote once carry-forward is applied.
    assert st.n_quotes_reconstructed == 4
    assert st.n_spread_negative == 0


def test_quote_book_state_survives_a_chunk_boundary(tmp_path):
    """The last bid/ask of one block must seed the next."""
    rows = [tick("2025.06.02", "00:00:00.000", "1.10000", "1.10015", flags=6)]
    rows += [tick("2025.06.02", f"00:00:{i:02d}.000", "", f"1.100{15+i%5:02d}",
                  flags=TICK_FLAG_ASK) for i in range(1, 60)]
    p = write_ticks(tmp_path / "carry.csv", rows)

    one_block = build_state(p, "2025-06-02", "2025-06-03")
    fmt = detect_format(p)
    st = TickAuditState(fmt, POINT, PIP, ms("2025-06-02"), ms("2025-06-03"))
    st.add_m5_interpretation("A_native", "native")
    n_batches = 0
    for batch, _ in iter_tick_batches(fmt, POINT, block_size=512):
        st.update(batch)
        n_batches += 1
    st.finalize()

    assert n_batches > 1, "test needs multiple blocks to be meaningful"
    # Every row still has a bid, because the carry crossed the block boundary.
    assert st.n_quotes_reconstructed == one_block.n_quotes_reconstructed == len(rows)
    assert st.n_quotes_undefined == 0


def test_stale_side_is_flagged_across_a_long_gap(tmp_path):
    """A Friday ask carried into a Monday bid update is not a live spread."""
    rows = [
        tick("2025.06.06", "23:59:00.000", "1.10000", "1.10015", flags=6),
        tick("2025.06.09", "00:00:00.000", "1.10500", "", flags=TICK_FLAG_BID),
        tick("2025.06.09", "00:00:00.500", "1.10501", "1.10515", flags=6),
    ]
    st = build_state(write_ticks(tmp_path / "stale.csv", rows),
                     "2025-06-06", "2025-06-10", stale_side_ms=60_000)
    assert st.n_quotes_reconstructed == 3
    # The Monday bid-only tick pairs a fresh bid with a two-day-old ask.
    assert st.n_quotes_stale_side == 1
    assert st.spread_hist_fresh.sum() == 2
    assert st.spread_hist.sum() == 3


def test_negative_spread_is_detected_not_repaired(tmp_path):
    rows = [
        tick("2025.06.02", "00:00:00.000", "1.10000", "1.10015", flags=6),
        tick("2025.06.02", "00:00:01.000", "1.10020", "1.10010", flags=6),
        tick("2025.06.02", "00:00:02.000", "1.10000", "1.10000", flags=6),
    ]
    st = build_state(write_ticks(tmp_path / "neg.csv", rows), "2025-06-02", "2025-06-03")
    assert st.n_spread_negative == 1
    assert st.n_spread_zero == 1
    assert st.spread_min == -10


def test_spread_statistics_are_exact_from_the_histogram():
    hist = np.zeros(50, dtype=np.int64)
    hist[[10, 20, 30]] = [50, 30, 20]          # 100 observations
    q = hist_quantiles(hist, (0.5, 0.95))
    assert q[0.5] == 10
    assert q[0.95] == 30
    assert hist_mean(hist) == pytest.approx((10 * 50 + 20 * 30 + 30 * 20) / 100)


# ---------------------------------------------------------------------------
# Timestamps: ordering and duplicates
# ---------------------------------------------------------------------------

def test_duplicate_timestamps_are_not_exact_duplicates(tmp_path):
    """Several distinct quotes inside one millisecond are legitimate."""
    rows = [
        tick("2025.06.02", "00:00:00.100", "1.10000", "1.10015", flags=6),
        tick("2025.06.02", "00:00:00.100", "1.10001", "1.10016", flags=6),
        tick("2025.06.02", "00:00:00.100", "1.10002", "1.10017", flags=6),
        tick("2025.06.02", "00:00:01.100", "1.10003", "1.10018", flags=6),
    ]
    st = build_state(write_ticks(tmp_path / "dupts.csv", rows), "2025-06-02", "2025-06-03")
    assert st.n_duplicate_ts_rows == 2          # 3 rows share one timestamp
    assert st.max_ts_group == 3
    assert st.n_exact_duplicate_rows == 0       # all three differ in price


def test_exact_duplicates_are_caught_even_when_not_adjacent(tmp_path):
    """Two identical rows separated by a third tick at the same millisecond."""
    rows = [
        tick("2025.06.02", "00:00:00.100", "1.10000", "1.10015", flags=6),
        tick("2025.06.02", "00:00:00.100", "1.10009", "1.10019", flags=6),
        tick("2025.06.02", "00:00:00.100", "1.10000", "1.10015", flags=6),   # duplicate
        tick("2025.06.02", "00:00:01.100", "1.10003", "1.10018", flags=6),
    ]
    st = build_state(write_ticks(tmp_path / "dupexact.csv", rows),
                     "2025-06-02", "2025-06-03")
    assert st.n_exact_duplicate_rows == 1
    assert st.n_duplicate_ts_rows == 2


def test_timestamp_group_is_carried_across_a_chunk_boundary(tmp_path):
    """An equal-timestamp group split by a block boundary must still be checked."""
    rows = [tick("2025.06.02", "00:00:00.100", "1.10000", "1.10015", flags=6)
            for _ in range(80)]
    rows.append(tick("2025.06.02", "00:00:05.100", "1.10009", "1.10019", flags=6))
    p = write_ticks(tmp_path / "carrygrp.csv", rows)
    fmt = detect_format(p)

    st = TickAuditState(fmt, POINT, PIP, ms("2025-06-02"), ms("2025-06-03"))
    st.add_m5_interpretation("A_native", "native")
    n_batches = 0
    for batch, _ in iter_tick_batches(fmt, POINT, block_size=512):
        st.update(batch)
        n_batches += 1
    st.finalize()

    assert n_batches > 1, "test needs multiple blocks to be meaningful"
    assert st.n_exact_duplicate_rows == 79
    assert st.max_ts_group == 80


def test_out_of_order_records_are_counted(tmp_path):
    rows = [
        tick("2025.06.02", "00:00:02.000", "1.10000", "1.10015", flags=6),
        tick("2025.06.02", "00:00:01.000", "1.10001", "1.10016", flags=6),   # backwards
        tick("2025.06.02", "00:00:03.000", "1.10002", "1.10017", flags=6),
    ]
    st = build_state(write_ticks(tmp_path / "ooo.csv", rows), "2025-06-02", "2025-06-03")
    assert st.n_out_of_order == 1


def test_run_groups_matches_numpy_unique_on_sorted_input():
    a = np.array([1, 1, 2, 5, 5, 5, 9], dtype=np.int64)
    uniq, starts, counts = _run_groups(a)
    u2, i2, c2 = np.unique(a, return_index=True, return_counts=True)
    assert uniq.tolist() == u2.tolist()
    assert starts.tolist() == i2.tolist()
    assert counts.tolist() == c2.tolist()


# ---------------------------------------------------------------------------
# M5 aggregation from ticks
# ---------------------------------------------------------------------------

def test_m5_aggregation_from_ticks_bins_and_ohlc(tmp_path):
    """Hand-computed OHLC across two 5-minute bins."""
    rows = [
        tick("2025.06.02", "00:00:10.000", "1.10000", "1.10015", flags=6),   # bin 1 open
        tick("2025.06.02", "00:01:00.000", "1.10050", "1.10065", flags=6),   # bin 1 high
        tick("2025.06.02", "00:02:00.000", "1.09900", "1.09915", flags=6),   # bin 1 low
        tick("2025.06.02", "00:04:59.000", "1.10010", "1.10025", flags=6),   # bin 1 close
        tick("2025.06.02", "00:05:00.000", "1.10100", "1.10115", flags=6),   # bin 2 open
        tick("2025.06.02", "00:09:00.000", "1.10200", "1.10215", flags=6),   # bin 2 close
    ]
    st = build_state(write_ticks(tmp_path / "m5.csv", rows), "2025-06-02", "2025-06-03")
    recon = st.m5["A_native"]

    b1 = ms("2025-06-02 00:00:00") // MS_PER_5MIN - recon.base_bin
    b2 = ms("2025-06-02 00:05:00") // MS_PER_5MIN - recon.base_bin
    assert recon.count[b1] == 4 and recon.count[b2] == 2
    assert (recon.open_[b1], recon.high[b1], recon.low[b1], recon.close[b1]) == \
           (110000, 110050, 109900, 110010)
    assert (recon.open_[b2], recon.high[b2], recon.low[b2], recon.close[b2]) == \
           (110100, 110200, 110100, 110200)


def test_m5_bin_alignment_is_left_labelled_and_left_closed(tmp_path):
    """A tick exactly on a bin boundary belongs to the NEW bin."""
    rows = [
        tick("2025.06.02", "00:04:59.999", "1.10000", "1.10015", flags=6),
        tick("2025.06.02", "00:05:00.000", "1.10500", "1.10515", flags=6),
    ]
    st = build_state(write_ticks(tmp_path / "edge.csv", rows), "2025-06-02", "2025-06-03")
    recon = st.m5["A_native"]
    b1 = ms("2025-06-02 00:00:00") // MS_PER_5MIN - recon.base_bin
    b2 = ms("2025-06-02 00:05:00") // MS_PER_5MIN - recon.base_bin
    assert recon.count[b1] == 1 and recon.count[b2] == 1
    assert recon.close[b1] == 110000
    assert recon.open_[b2] == 110500


def test_m5_bin_state_survives_a_chunk_boundary(tmp_path):
    """A bin split across blocks must keep its true open, high, low and close."""
    rows = [tick("2025.06.02", f"00:00:{i:02d}.000", f"1.10{i:03d}", f"1.11{i:03d}",
                 flags=6) for i in range(60)]
    p = write_ticks(tmp_path / "binsplit.csv", rows)
    fmt = detect_format(p)
    st = TickAuditState(fmt, POINT, PIP, ms("2025-06-02"), ms("2025-06-03"))
    st.add_m5_interpretation("A_native", "native")
    n_batches = 0
    for batch, _ in iter_tick_batches(fmt, POINT, block_size=512):
        st.update(batch)
        n_batches += 1
    st.finalize()

    assert n_batches > 1
    recon = st.m5["A_native"]
    b = ms("2025-06-02 00:00:00") // MS_PER_5MIN - recon.base_bin
    assert recon.count[b] == 60
    assert recon.open_[b] == 110000            # first tick of the bin
    assert recon.close[b] == 110059            # last tick of the bin
    assert recon.low[b] == 110000 and recon.high[b] == 110059


def test_m5_reconstruction_uses_carried_bid_on_ask_only_open(tmp_path):
    """If a bin opens with an ask-only tick, the bar open is the carried bid."""
    rows = [
        tick("2025.06.02", "00:04:00.000", "1.10000", "1.10015", flags=6),
        tick("2025.06.02", "00:05:00.000", "", "1.10020", flags=TICK_FLAG_ASK),
        tick("2025.06.02", "00:06:00.000", "1.10030", "1.10045", flags=6),
    ]
    st = build_state(write_ticks(tmp_path / "askopen.csv", rows),
                     "2025-06-02", "2025-06-03")
    recon = st.m5["A_native"]
    b = ms("2025-06-02 00:05:00") // MS_PER_5MIN - recon.base_bin
    assert recon.open_[b] == 110000            # carried, not skipped or zero


def test_compare_m5_reports_exact_and_tolerance_matches(tmp_path):
    rows = [
        tick("2025.06.02", "00:00:10.000", "1.10000", "1.10015", flags=6),
        tick("2025.06.02", "00:01:00.000", "1.10050", "1.10065", flags=6),
        tick("2025.06.02", "00:02:00.000", "1.09900", "1.09915", flags=6),
        tick("2025.06.02", "00:04:59.000", "1.10010", "1.10025", flags=6),
    ]
    st = build_state(write_ticks(tmp_path / "cmp.csv", rows), "2025-06-02", "2025-06-03")

    m5 = pd.DataFrame({
        "bar_open_time": [pd.Timestamp("2025-06-02 00:00:00")],
        "epoch_ms": [ms("2025-06-02 00:00:00")],
        "open": [110000], "high": [110050], "low": [109900], "close": [110010],
    })
    res = compare_m5(st.m5["A_native"], m5)
    assert res.summary["m5_bars_with_tick_coverage"] == 1
    assert res.summary["full_ohlc_exact_match_pct"] == 100.0
    assert len(res.per_date) == 1

    m5_off = m5.copy()
    m5_off["close"] = [110011]                 # one point out
    res2 = compare_m5(st.m5["A_native"], m5_off)
    assert res2.summary["full_ohlc_exact_match_pct"] == 0.0
    assert res2.summary["full_ohlc_within_1pt_pct"] == 100.0


def test_compare_m5_with_no_overlap_returns_full_key_set(tmp_path):
    rows = [tick("2025.06.02", "00:00:10.000", "1.10000", "1.10015", flags=6)]
    st = build_state(write_ticks(tmp_path / "noov.csv", rows), "2025-06-02", "2025-06-03")
    m5 = pd.DataFrame({
        "bar_open_time": [pd.Timestamp("2030-01-01")],
        "epoch_ms": [ms("2030-01-01")],
        "open": [1], "high": [1], "low": [1], "close": [1],
    })
    res = compare_m5(st.m5["A_native"], m5)
    assert res.summary["m5_bars_with_tick_coverage"] == 0
    assert "full_ohlc_exact_match_pct" in res.summary
    assert np.isnan(res.summary["full_ohlc_exact_match_pct"])


def test_load_m5_reference_parses_to_points(tmp_path):
    p = tmp_path / "m5.csv"
    p.write_text(
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        "2025.06.02\t00:00:00\t1.10000\t1.10050\t1.09900\t1.10010\t50\t0\t12\n",
        encoding="ascii")
    df = load_m5_reference(p, POINT)
    assert df["open"].iloc[0] == 110000
    assert df["low"].iloc[0] == 109900
    assert df["bar_open_time"].iloc[0] == pd.Timestamp("2025-06-02 00:00:00")


# ---------------------------------------------------------------------------
# Timezone transformation
# ---------------------------------------------------------------------------

def test_broker_offset_table_is_plus_2_or_plus_3():
    trans, offs = build_broker_offset_table()
    assert trans.size == offs.size > 0
    hours = np.unique(offs // 3_600_000)
    assert set(hours.tolist()) == {2, 3}


def test_utc_to_broker_native_matches_the_project_rule():
    """UTC+3 under US DST, UTC+2 otherwise - the documented New York + 7 clock."""
    from forex_research.data_loader import to_utc

    trans, offs = build_broker_offset_table()
    for utc_str, expected_offset_h in (("2025-01-15 12:00:00", 2),    # EST
                                       ("2025-07-15 12:00:00", 3)):   # EDT
        got = utc_ms_to_broker_native_ms(np.array([ms(utc_str)]), trans, offs)[0]
        assert got - ms(utc_str) == expected_offset_h * 3_600_000

        # And it inverts the project's own native -> UTC conversion.
        native = pd.Series([pd.Timestamp(got, unit="ms")])
        assert to_utc(native, "new_york_plus_7").iloc[0] == \
            pd.Timestamp(utc_str, tz="UTC")


def test_rollover_hour_evidence_identifies_1700_new_york():
    """Widest spread at file-native 00:00 implies 17:00 New York on a NY+7 clock."""
    from forex_research.tick_validation import rollover_hour_evidence

    medians = {h: 18.0 for h in range(24)}
    medians[0] = 37.0                          # the rollover widening
    ev = rollover_hour_evidence(medians)
    assert ev["widest_spread_hour_file_native"] == 0
    assert ev["implied_new_york_hour"] == 17
    assert ev["consistent_with_1700_ny_rollover"] is True
    assert ev["widening_ratio"] == pytest.approx(37.0 / 18.0, abs=0.01)


def test_rollover_hour_evidence_rejects_a_utc_clock():
    """If the file clock were UTC, rollover would land at 21:00, not 00:00."""
    from forex_research.tick_validation import rollover_hour_evidence

    medians = {h: 18.0 for h in range(24)}
    medians[21] = 37.0
    ev = rollover_hour_evidence(medians)
    assert ev["widest_spread_hour_file_native"] == 21
    assert ev["implied_new_york_hour"] == 14
    assert ev["consistent_with_1700_ny_rollover"] is False


def test_rollover_evidence_is_empty_without_data():
    from forex_research.tick_validation import rollover_hour_evidence
    assert rollover_hour_evidence({}) == {}


def test_weekly_session_table_reports_boundaries():
    first = {2900: ms("2025-06-02 00:00:05")}
    last = {2900: ms("2025-06-06 23:59:58")}
    df = weekly_session_table(first, last)
    assert df["first_weekday"].iloc[0] == "Monday"
    assert df["last_weekday"].iloc[0] == "Friday"


# ---------------------------------------------------------------------------
# Coverage, density, overlap
# ---------------------------------------------------------------------------

def test_daily_and_hourly_coverage_counts(tmp_path):
    rows = [tick("2025.06.02", f"0{h}:00:0{i}.000", "1.10000", "1.10015", flags=6)
            for h in range(3) for i in range(5)]
    rows += [tick("2025.06.03", "01:00:00.000", "1.10000", "1.10015", flags=6)]
    st = build_state(write_ticks(tmp_path / "cov.csv", rows), "2025-06-02", "2025-06-05")
    d0 = ms("2025-06-02") // 86_400_000 - st.base_day
    d1 = ms("2025-06-03") // 86_400_000 - st.base_day
    assert st.day_count[d0] == 15
    assert st.day_count[d1] == 1
    assert st.day_first_ms[d0] == ms("2025-06-02 00:00:00")
    assert st.day_last_ms[d0] == ms("2025-06-02 02:00:04")


def test_highest_activity_hour_is_selected_by_count(tmp_path):
    rows = [tick("2025.06.02", f"00:00:{i:02d}.000", "1.10000", "1.10015", flags=6)
            for i in range(5)]
    rows += [tick("2025.06.02", f"03:00:{i:02d}.000", "1.10000", "1.10015", flags=6)
             for i in range(40)]
    st = build_state(write_ticks(tmp_path / "busy.csv", rows), "2025-06-02", "2025-06-03")
    assert st.best_hour_count == 40
    busiest = pd.Timestamp((st.base_day * 24 + st.best_hour_index) * 3_600_000, unit="ms")
    assert busiest.hour == 3


def test_research_overlap_computes_ranges():
    df = research_overlap(
        pd.Timestamp("2022-01-03"), pd.Timestamp("2026-08-14"),
        {"M5": (pd.Timestamp("2025-04-08"), pd.Timestamp("2026-08-18"))})
    r = df.iloc[0]
    assert r["overlap_start"] == pd.Timestamp("2025-04-08")
    assert r["overlap_end"] == pd.Timestamp("2026-08-14")
    assert r["uncovered_tail_days"] == pytest.approx(4.0)
    assert r["uncovered_head_days"] == 0.0
    assert 98.0 < r["pct_of_existing_covered"] < 100.0


def test_research_overlap_handles_disjoint_ranges():
    df = research_overlap(
        pd.Timestamp("2022-01-03"), pd.Timestamp("2022-06-01"),
        {"M5": (pd.Timestamp("2025-04-08"), pd.Timestamp("2026-08-18"))})
    assert pd.isna(df.iloc[0]["overlap_start"])
    assert df.iloc[0]["overlap_days"] == 0.0


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_byte_bisection_finds_a_timestamp(tmp_path):
    rows = [tick("2025.06.02", f"{i//3600:02d}:{(i//60)%60:02d}:{i%60:02d}.000",
                 "1.10000", "1.10015", flags=6) for i in range(0, 20000)]
    p = write_ticks(tmp_path / "seek.csv", rows)
    fmt = detect_format(p)
    target = ms("2025-06-02 02:00:00")
    off = find_byte_offset_for_timestamp(fmt, target)
    lines, _ = extract_sample(fmt, off, 50)
    stamps = [pd.Timestamp(f"{l.split(chr(9))[0].replace('.', '-')} "
                           f"{l.split(chr(9))[1]}") for l in lines]
    # The offset is an exact line start: the first row read IS the target row.
    assert stamps[0] == pd.Timestamp("2025-06-02 02:00:00")
    assert stamps == sorted(stamps)


def test_byte_bisection_before_first_and_after_last(tmp_path):
    rows = [tick("2025.06.02", f"01:{i//60:02d}:{i%60:02d}.000", "1.10000", "1.10015")
            for i in range(600)]
    fmt = detect_format(write_ticks(tmp_path / "edges.csv", rows))

    off = find_byte_offset_for_timestamp(fmt, ms("2020-01-01"))
    lines, _ = extract_sample(fmt, off, 1)
    assert lines[0] == rows[0]                 # clamps to the first data row

    off = find_byte_offset_for_timestamp(fmt, ms("2030-01-01"))
    lines, _ = extract_sample(fmt, off, 1)
    assert lines == []                         # past the end, nothing to read


def test_sample_extraction_preserves_original_columns(tmp_path):
    rows = [tick("2025.06.02", f"00:00:{i:02d}.000", "1.10000", "", flags=TICK_FLAG_BID)
            for i in range(30)]
    fmt = detect_format(write_ticks(tmp_path / "samp.csv", rows))
    lines, header = extract_sample(fmt, 0, 10, from_start=True)
    assert header == HEADER
    assert len(lines) == 10
    assert lines[0] == rows[0]                 # byte-identical, nothing rewritten
    assert lines[0].split("\t")[3] == ""       # structurally absent ask preserved


def test_tail_sample_returns_the_final_rows(tmp_path):
    rows = [tick("2025.06.02", f"00:00:{i:02d}.000", "1.10000", "1.10015", flags=6)
            for i in range(50)]
    fmt = detect_format(write_ticks(tmp_path / "tail.csv", rows))
    lines, _ = extract_tail_sample(fmt, 5)
    assert len(lines) == 5
    assert lines[-1] == rows[-1]


# ---------------------------------------------------------------------------
# The audit must not disturb existing research
# ---------------------------------------------------------------------------

def test_tick_modules_do_not_import_the_research_pipeline():
    """The audit must not be able to re-run or mutate Day 1 / Step 2 / 3A / Day 2."""
    import forex_research.tick_audit as ta
    import forex_research.tick_stream as ts
    import forex_research.tick_validation as tv

    forbidden = {"pipeline", "outcomes", "validation", "step2", "step3a",
                 "day1", "day2_checkpoint", "benchmark", "single_tf_features",
                 "cross_tf_features", "event_detection", "alignment"}
    for mod in (ta, ts, tv):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        for name in forbidden:
            assert f"from .{name} import" not in src, f"{mod.__name__} imports {name}"
            assert f"import forex_research.{name}" not in src

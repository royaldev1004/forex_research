from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa

from .logging_utils import get_logger

log = get_logger("tick_schema")

# --- MT5 tick flag bitmask (MqlTick.flags) --------------------------------
TICK_FLAG_BID = 2
TICK_FLAG_ASK = 4
TICK_FLAG_LAST = 8
TICK_FLAG_VOLUME = 16
TICK_FLAG_BUY = 32
TICK_FLAG_SELL = 64

FLAG_NAMES: dict[int, str] = {
    TICK_FLAG_BID: "TICK_FLAG_BID",
    TICK_FLAG_ASK: "TICK_FLAG_ASK",
    TICK_FLAG_LAST: "TICK_FLAG_LAST",
    TICK_FLAG_VOLUME: "TICK_FLAG_VOLUME",
    TICK_FLAG_BUY: "TICK_FLAG_BUY",
    TICK_FLAG_SELL: "TICK_FLAG_SELL",
}

#: Column names expected in an MT5 tick export, after ``<>`` stripping.
MT5_TICK_COLUMNS = ["DATE", "TIME", "BID", "ASK", "LAST", "VOLUME", "FLAGS"]

#: Fixed widths of the MT5 date/time text fields.
DATE_WIDTH = 10          # YYYY.MM.DD
TIME_WIDTH_MS = 12       # HH:MM:SS.mmm
TIME_WIDTH_SEC = 8       # HH:MM:SS

_DATE_RE = re.compile(rb"^\d{4}\.\d{2}\.\d{2}$")
_TIME_MS_RE = re.compile(rb"^\d{2}:\d{2}:\d{2}\.\d{3}$")
_TIME_SEC_RE = re.compile(rb"^\d{2}:\d{2}:\d{2}$")


class TickFormatError(RuntimeError):
    pass


@dataclass(frozen=True)
class TickFileFormat:

    path: Path
    size_bytes: int
    encoding: str
    has_bom: bool
    delimiter: str
    line_ending: str
    has_header: bool
    raw_columns: tuple[str, ...]          # original names, e.g. "<DATE>"
    columns: tuple[str, ...]              # normalised, e.g. "DATE"
    n_columns: int
    time_precision: str                   # "milliseconds" | "seconds"
    time_width: int
    decimal_separator: str
    price_decimals: int
    sample_lines: tuple[str, ...]
    mean_row_bytes: float
    estimated_rows: int
    is_mt5_tick_export: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    def column_index(self, name: str) -> int:
        return self.columns.index(name)


def _detect_line_ending(prefix: bytes) -> str:
    if b"\r\n" in prefix:
        return "CRLF"
    if b"\n" in prefix:
        return "LF"
    if b"\r" in prefix:
        return "CR"
    raise TickFormatError("No line terminator found in the file prefix.")


def _detect_delimiter(header_line: bytes) -> str:
    counts = {"\t": header_line.count(b"\t"),
              ",": header_line.count(b","),
              ";": header_line.count(b";")}
    best = max(counts, key=lambda k: counts[k])
    if counts[best] == 0:
        raise TickFormatError(
            f"No tab, comma or semicolon found in the header line: {header_line[:200]!r}")
    return best


def detect_format(path: Path, prefix_bytes: int = 1 << 16) -> TickFileFormat:
    path = Path(path)
    if not path.exists():
        raise TickFormatError(f"Tick file not found: {path}")
    size = path.stat().st_size

    with path.open("rb") as fh:
        prefix = fh.read(prefix_bytes)
    if not prefix:
        raise TickFormatError(f"Tick file is empty: {path}")

    notes: list[str] = []

    # --- encoding / BOM ---------------------------------------------------
    has_bom = prefix.startswith(b"\xef\xbb\xbf")
    body = prefix[3:] if has_bom else prefix
    if b"\x00" in body[:4096]:
        raise TickFormatError(
            "Null bytes found in the prefix: the file appears to be BINARY, not text. "
            "Refusing to invent a parser. Report the signature and request the format."
        )
    try:
        body.decode("ascii")
        encoding = "ascii"
    except UnicodeDecodeError:
        try:
            body.decode("utf-8")
            encoding = "utf-8"
            notes.append("Non-ASCII UTF-8 bytes present in the prefix.")
        except UnicodeDecodeError:
            encoding = "latin-1"
            notes.append("Prefix is not valid UTF-8; falling back to latin-1.")

    line_ending = _detect_line_ending(body)
    eol = {"CRLF": b"\r\n", "LF": b"\n", "CR": b"\r"}[line_ending]
    lines = body.split(eol)
    if len(lines) < 3:
        raise TickFormatError("Fewer than three lines in the prefix; cannot infer format.")

    header_line = lines[0]
    delimiter = _detect_delimiter(header_line)
    delim_b = delimiter.encode()

    raw_cols = tuple(c.decode(encoding, "replace") for c in header_line.split(delim_b))
    cols = tuple(c.strip().strip("<>").upper() for c in raw_cols)
    has_header = any(c in MT5_TICK_COLUMNS for c in cols)
    if not has_header:
        raise TickFormatError(
            f"First line does not look like a header: {raw_cols}. "
            "Refusing to assume a column order."
        )

    # --- data-line shape --------------------------------------------------
    data_lines = [ln for ln in lines[1:] if ln]
    if not data_lines:
        raise TickFormatError("No data lines in the prefix.")
    # The fixed-size prefix read can slice the final line in half. Drop it only
    # when the read actually hit the cap; if it reached EOF, that line is whole
    # and discarding it would break detection on a small file.
    hit_cap = len(prefix) >= prefix_bytes
    if hit_cap and len(data_lines) > 1:
        data_lines = data_lines[:-1]

    widths = {len(ln.split(delim_b)) for ln in data_lines}
    if len(widths) != 1:
        notes.append(f"Inconsistent field counts in the prefix: {sorted(widths)}")
    n_columns = len(cols)

    first_fields = data_lines[0].split(delim_b)
    if len(first_fields) != n_columns:
        notes.append(
            f"Header declares {n_columns} columns but the first data row has "
            f"{len(first_fields)}.")

    # --- timestamp precision ---------------------------------------------
    d_idx = cols.index("DATE") if "DATE" in cols else 0
    t_idx = cols.index("TIME") if "TIME" in cols else 1
    date_tok = first_fields[d_idx]
    time_tok = first_fields[t_idx]
    if not _DATE_RE.match(date_tok):
        raise TickFormatError(f"DATE field is not YYYY.MM.DD: {date_tok!r}")
    if _TIME_MS_RE.match(time_tok):
        time_precision, time_width = "milliseconds", TIME_WIDTH_MS
    elif _TIME_SEC_RE.match(time_tok):
        time_precision, time_width = "seconds", TIME_WIDTH_SEC
    else:
        raise TickFormatError(f"TIME field is neither HH:MM:SS.mmm nor HH:MM:SS: {time_tok!r}")

    # --- price formatting -------------------------------------------------
    price_decimals = 0
    for name in ("BID", "ASK"):
        if name not in cols:
            continue
        i = cols.index(name)
        for ln in data_lines[:200]:
            parts = ln.split(delim_b)
            if i >= len(parts):
                continue          # short/ragged line; counted as malformed later
            tok = parts[i]
            if tok and b"." in tok:
                price_decimals = max(price_decimals, len(tok.split(b".")[1]))
    if b"," in first_fields[cols.index("BID")] if "BID" in cols else False:
        raise TickFormatError("Comma decimal separator detected in BID; not supported.")

    mean_row = float(np.mean([len(ln) + len(eol) for ln in data_lines]))
    header_bytes = len(header_line) + len(eol) + (3 if has_bom else 0)
    estimated_rows = int((size - header_bytes) / mean_row) if mean_row else 0

    is_mt5 = all(c in cols for c in ("DATE", "TIME", "BID", "ASK"))
    if is_mt5 and "FLAGS" not in cols:
        notes.append("No <FLAGS> column: sparse bid/ask updates cannot be cross-checked "
                     "against the MT5 flag bitmask.")

    fmt = TickFileFormat(
        path=path, size_bytes=size, encoding=encoding, has_bom=has_bom,
        delimiter=delimiter, line_ending=line_ending, has_header=has_header,
        raw_columns=raw_cols, columns=cols, n_columns=n_columns,
        time_precision=time_precision, time_width=time_width,
        decimal_separator=".", price_decimals=price_decimals,
        sample_lines=tuple(ln.decode(encoding, "replace") for ln in data_lines[:5]),
        mean_row_bytes=mean_row, estimated_rows=estimated_rows,
        is_mt5_tick_export=is_mt5, notes=tuple(notes),
    )
    log.info("Detected %s: %s-delimited, %s, %d columns, %s timestamps, ~%s rows estimated",
             path.name, {"\t": "TAB", ",": "COMMA", ";": "SEMICOLON"}[delimiter],
             line_ending, n_columns, time_precision, f"{estimated_rows:,}")
    return fmt


# ---------------------------------------------------------------------------
# Fast fixed-width parsing over Arrow string buffers
# ---------------------------------------------------------------------------

def _string_matrix(arr: pa.Array, width: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
        if isinstance(arr, pa.ChunkedArray):        # empty chunked array
            arr = arr.chunk(0) if arr.num_chunks else pa.array([], type=pa.string())

    n = len(arr)
    if n == 0:
        return np.zeros((0, width), dtype=np.uint8), np.zeros(0, dtype=bool)

    buffers = arr.buffers()
    validity, offsets_buf, data_buf = buffers[0], buffers[1], buffers[2]
    off_dtype = np.int64 if pa.types.is_large_string(arr.type) else np.int32
    offsets = np.frombuffer(offsets_buf, dtype=off_dtype, count=n + 1,
                            offset=arr.offset * np.dtype(off_dtype).itemsize)
    lengths = np.diff(offsets)

    ok = lengths == width
    if validity is not None and arr.null_count:
        valid_bits = np.unpackbits(
            np.frombuffer(validity, dtype=np.uint8), bitorder="little")
        ok &= valid_bits[arr.offset:arr.offset + n].astype(bool)

    data = np.frombuffer(data_buf, dtype=np.uint8)
    mat = np.zeros((n, width), dtype=np.uint8)
    if ok.any():
        starts = offsets[:-1][ok].astype(np.int64)
        idx = starts[:, None] + np.arange(width, dtype=np.int64)[None, :]
        mat[ok] = data[idx]
    return mat, ok


def _digits(mat: np.ndarray, start: int, count: int) -> np.ndarray:
    out = np.zeros(mat.shape[0], dtype=np.int64)
    for k in range(count):
        out = out * 10 + (mat[:, start + k].astype(np.int64) - 48)
    return out


def _days_from_civil(y: np.ndarray, m: np.ndarray, d: np.ndarray) -> np.ndarray:
    y = y - (m <= 2)
    era = np.where(y >= 0, y, y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + np.where(m > 2, -3, 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def parse_datetime_ms(
    date_arr: pa.Array, time_arr: pa.Array, time_width: int
) -> tuple[np.ndarray, np.ndarray]:
    dmat, dok = _string_matrix(date_arr, DATE_WIDTH)
    tmat, tok = _string_matrix(time_arr, time_width)
    ok = dok & tok

    # Structural separators must be exactly where the format says they are.
    if len(dmat):
        ok &= (dmat[:, 4] == ord(".")) & (dmat[:, 7] == ord("."))
        ok &= (tmat[:, 2] == ord(":")) & (tmat[:, 5] == ord(":"))
        if time_width == TIME_WIDTH_MS:
            ok &= tmat[:, 8] == ord(".")

        digit_cols_d = [0, 1, 2, 3, 5, 6, 8, 9]
        for c in digit_cols_d:
            ok &= (dmat[:, c] >= 48) & (dmat[:, c] <= 57)
        digit_cols_t = [0, 1, 3, 4, 6, 7] + ([9, 10, 11] if time_width == TIME_WIDTH_MS else [])
        for c in digit_cols_t:
            ok &= (tmat[:, c] >= 48) & (tmat[:, c] <= 57)

    year = _digits(dmat, 0, 4)
    month = _digits(dmat, 5, 2)
    day = _digits(dmat, 8, 2)
    hour = _digits(tmat, 0, 2)
    minute = _digits(tmat, 3, 2)
    second = _digits(tmat, 6, 2)
    milli = _digits(tmat, 9, 3) if time_width == TIME_WIDTH_MS else np.zeros_like(second)

    ok &= (month >= 1) & (month <= 12) & (day >= 1) & (day <= 31)
    ok &= (hour <= 23) & (minute <= 59) & (second <= 60) & (milli <= 999)

    days = _days_from_civil(year, month, day)
    epoch_ms = (days * 86_400_000 + hour * 3_600_000 + minute * 60_000
                + second * 1_000 + milli)
    epoch_ms = np.where(ok, epoch_ms, np.iinfo(np.int64).min)
    return epoch_ms, ok


def parse_price_points(arr: pa.Array, point_size: float) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.compute as pc

    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    try:
        as_f = pc.cast(arr, pa.float64())
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        # Rare path: at least one value is not a valid decimal. Fall back to a
        # per-value coercion so the bad rows are counted, not fatal.
        import pandas as pd
        vals = pd.to_numeric(pd.Series(arr.to_pylist(), dtype="object"),
                             errors="coerce").to_numpy(dtype="float64")
    else:
        vals = as_f.to_numpy(zero_copy_only=False).astype("float64", copy=False)

    present = np.isfinite(vals)
    scale = 1.0 / point_size
    pts = np.full(n, -1, dtype=np.int64)
    if present.any():
        pts[present] = np.rint(vals[present] * scale).astype(np.int64)
    return pts, present


def parse_int_column(arr: pa.Array) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.compute as pc

    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    try:
        vals = pc.cast(arr, pa.float64()).to_numpy(zero_copy_only=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        import pandas as pd
        vals = pd.to_numeric(pd.Series(arr.to_pylist(), dtype="object"),
                             errors="coerce").to_numpy(dtype="float64")

    # Go through float64 and mask before casting: a null becomes NaN, and
    # casting NaN straight to int64 is undefined behaviour, not a -1.
    as_f = np.asarray(vals, dtype="float64")
    present = np.isfinite(as_f)
    out = np.full(n, -1, dtype=np.int64)
    if present.any():
        out[present] = as_f[present].astype(np.int64)
    return out, present


def describe_flag(value: int) -> str:
    if value < 0:
        return "absent"
    parts = [name for bit, name in FLAG_NAMES.items() if value & bit]
    return "|".join(parts) if parts else f"unknown({value})"

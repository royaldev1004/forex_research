from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataSource, ResearchConfig, timeframe_minutes
from .logging_utils import get_logger

log = get_logger("data_loader")

MT5_COLUMNS = ["DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "TICKVOL", "VOL", "SPREAD"]

#: Canonical column names produced by :func:`load_source`.
CANONICAL_COLUMNS = [
    "bar_open_time",
    "bar_close_time",
    "bar_open_time_utc",
    "bar_close_time_utc",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "real_volume",
    "spread_points",
]

#: Fixed offset between the New York wall clock and the file-native clock.
NEW_YORK_OFFSET_HOURS = 7


class DataLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedSeries:

    frame: pd.DataFrame
    timeframe: str
    role: str
    path: Path
    sha256: str
    raw_row_count: int


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def to_utc(naive_native: pd.Series, mode: str = "new_york_plus_7") -> pd.Series:
    if mode == "utc":
        return naive_native.dt.tz_localize("UTC")
    if mode != "new_york_plus_7":
        raise DataLoadError(f"Unsupported source_timezone_mode: {mode!r}")
    ny_wall = naive_native - pd.Timedelta(hours=NEW_YORK_OFFSET_HOURS)
    localized = ny_wall.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    )
    return localized.dt.tz_convert("UTC")


def load_mt5_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataLoadError(f"Data file not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip().strip("<>").upper() for c in df.columns]
    missing = [c for c in MT5_COLUMNS if c not in df.columns]
    if missing:
        raise DataLoadError(
            f"{path.name}: missing expected MT5 columns {missing}. Found {list(df.columns)}."
        )
    return df


def load_source(source: DataSource, cfg: ResearchConfig) -> LoadedSeries:
    if source.format != "mt5_tsv":
        raise DataLoadError(f"Unsupported source format: {source.format!r}")

    raw = load_mt5_tsv(source.path)
    raw_rows = len(raw)

    open_time = pd.to_datetime(
        raw["DATE"] + " " + raw["TIME"], format="%Y.%m.%d %H:%M:%S", errors="coerce"
    )
    if open_time.isna().any():
        n = int(open_time.isna().sum())
        raise DataLoadError(f"{source.path.name}: {n} timestamps failed to parse.")

    tf_minutes = timeframe_minutes(source.timeframe)
    bar_delta = pd.Timedelta(minutes=tf_minutes)

    if cfg.timestamp_semantics == "bar_open":
        bar_open = open_time
        bar_close = open_time + bar_delta
    else:
        bar_close = open_time
        bar_open = open_time - bar_delta

    out = pd.DataFrame(
        {
            "bar_open_time": bar_open,
            "bar_close_time": bar_close,
            "open": pd.to_numeric(raw["OPEN"], errors="coerce"),
            "high": pd.to_numeric(raw["HIGH"], errors="coerce"),
            "low": pd.to_numeric(raw["LOW"], errors="coerce"),
            "close": pd.to_numeric(raw["CLOSE"], errors="coerce"),
            "tick_volume": pd.to_numeric(raw["TICKVOL"], errors="coerce"),
            "real_volume": pd.to_numeric(raw["VOL"], errors="coerce"),
            "spread_points": pd.to_numeric(raw["SPREAD"], errors="coerce"),
        }
    )

    out = out.sort_values("bar_open_time", kind="mergesort").reset_index(drop=True)
    out["bar_open_time_utc"] = to_utc(out["bar_open_time"], cfg.source_timezone_mode)
    out["bar_close_time_utc"] = to_utc(out["bar_close_time"], cfg.source_timezone_mode)
    out = out[CANONICAL_COLUMNS]

    log.info(
        "Loaded %s (%s): %d rows, %s -> %s (file-native clock)",
        source.path.name,
        source.timeframe,
        len(out),
        out["bar_open_time"].iloc[0],
        out["bar_open_time"].iloc[-1],
    )
    return LoadedSeries(
        frame=out,
        timeframe=source.timeframe,
        role=source.role,
        path=source.path,
        sha256=_sha256(source.path),
        raw_row_count=raw_rows,
    )


def load_all(cfg: ResearchConfig) -> dict[str, LoadedSeries]:
    loaded: dict[str, LoadedSeries] = {}
    for src in cfg.sources:
        series = load_source(src, cfg)
        if series.timeframe in loaded:
            raise DataLoadError(f"Duplicate source timeframe: {series.timeframe!r}")
        loaded[series.timeframe] = series
    return loaded


def add_pip_columns(df: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    out = df.copy()
    points_per_pip = cfg.pip_size / cfg.point_size
    out["spread_pips"] = out["spread_points"].astype("float64") / points_per_pip
    return out


def price_to_pips(delta: np.ndarray | pd.Series, cfg: ResearchConfig):
    return delta / cfg.pip_size


def epoch_seconds(times: pd.Series) -> pd.Series:
    return (times - pd.Timestamp(0)) // pd.Timedelta(seconds=1)

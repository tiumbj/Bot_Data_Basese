# ==================================================================================================
# FILE: build_parquet_tf_dataset_cache.py
# PATH: C:\Data\Bot\Local_LLM\gold_research\jobs\build_parquet_tf_dataset_cache.py
# VERSION: v1.0.0
#
# CHANGELOG:
# - v1.0.0
#   1) Create central CSV -> Parquet dataset cache builder for all locked research timeframes
#   2) Normalize OHLC schema across all timeframes
#   3) Validate monotonic time ordering, nulls, duplicates, and row counts
#   4) Write per-timeframe Parquet + dataset summary JSON + conversion manifest JSONL
#   5) Prepare data layer for full intelligent vectorized backtest engine
#
# DESIGN RATIONALE:
# - Full research speed cannot rely on per-job CSV parsing.
# - Parquet is the correct storage layer for repeated large-scale scans.
# - This file is the first data-layer foundation of the v2 fast research engine.
#
# OUTPUT:
# - C:\Data\Bot\central_market_data\parquet\XAUUSD_<TF>.parquet
# - C:\Data\Bot\central_market_data\parquet\parquet_cache_summary.json
# - C:\Data\Bot\central_market_data\parquet\parquet_cache_manifest.jsonl
#
# NOTES:
# - Schema is normalized to:
#     time, open, high, low, close, volume
# - time is written as datetime-compatible timestamp column
# - volume is always present; missing source volume becomes 0.0
# ==================================================================================================

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

VERSION = "v1.0.0"
SYMBOL = "XAUUSD"

# --------------------------------------------------------------------------------------------------
# LOCKED RESEARCH TIMEFRAMES
# --------------------------------------------------------------------------------------------------
RESEARCH_TIMEFRAMES: List[str] = [
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M10",
    "M15",
    "M30",
    "H1",
    "H4",
    "D1",
]

# --------------------------------------------------------------------------------------------------
# INPUT / OUTPUT PATHS
# --------------------------------------------------------------------------------------------------
CSV_DATASET_MAP: Dict[str, str] = {
    "M1": r"C:\Data\Bot\central_market_data\tf\XAUUSD_M1.csv",
    "M2": r"C:\Data\Bot\central_market_data\tf\XAUUSD_M2.csv",
    "M3": r"C:\Data\Bot\central_market_data\tf\XAUUSD_M3.csv",
    "M4": r"C:\Data\Bot\central_market_data\tf\XAUUSD_M4.csv",
    "M5": r"C:\Data\Bot\central_market_data\tf\XAUUSD_M5.csv",
    "M6": r"C:\Data\Bot\central_market_data\tf\XAUUSD_M6.csv",
    "M10": r"C:\Data\Bot\central_market_data\tf\XAUUSD_M10.csv",
    "M15": r"C:\Data\Bot\central_market_data\tf\XAUUSD_M15.csv",
    "M30": r"C:\Data\Bot\central_market_data\tf\XAUUSD_M30.csv",
    "H1": r"C:\Data\Bot\central_market_data\tf\XAUUSD_H1.csv",
    "H4": r"C:\Data\Bot\central_market_data\tf\XAUUSD_H4.csv",
    "D1": r"C:\Data\Bot\central_market_data\tf\XAUUSD_D1.csv",
}

OUTDIR = Path(r"C:\Data\Bot\central_market_data\parquet")
SUMMARY_PATH = OUTDIR / "parquet_cache_summary.json"
MANIFEST_PATH = OUTDIR / "parquet_cache_manifest.jsonl"

# --------------------------------------------------------------------------------------------------
# DATA MODELS
# --------------------------------------------------------------------------------------------------
@dataclass
class ConversionRecord:
    version: str
    symbol: str
    timeframe: str
    source_csv: str
    output_parquet: str
    row_count: int
    duplicate_time_count: int
    null_row_count_before_clean: int
    removed_null_row_count: int
    removed_duplicate_time_count: int
    first_time: Optional[str]
    last_time: Optional[str]
    file_size_bytes: int
    generated_at_utc: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# --------------------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def find_first_existing(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        found = lower_map.get(candidate.lower())
        if found is not None:
            return found
    return None


def detect_schema(columns: List[str]) -> Dict[str, Optional[str]]:
    return {
        "time": find_first_existing(columns, ["time", "datetime", "date", "timestamp", "ts"]),
        "open": find_first_existing(columns, ["open", "o"]),
        "high": find_first_existing(columns, ["high", "h"]),
        "low": find_first_existing(columns, ["low", "l"]),
        "close": find_first_existing(columns, ["close", "c"]),
        "volume": find_first_existing(columns, ["tick_volume", "volume", "vol", "real_volume"]),
    }


def validate_detected_schema(detected: Dict[str, Optional[str]], csv_path: Path) -> None:
    required = ["time", "open", "high", "low", "close"]
    missing = [key for key in required if not detected.get(key)]
    if missing:
        raise RuntimeError(f"Missing required columns in {csv_path}: {missing}")


def load_csv_as_polars(csv_path: Path) -> pl.DataFrame:
    if not csv_path.exists():
        raise RuntimeError(f"CSV file not found: {csv_path}")

    df = pl.read_csv(csv_path, infer_schema_length=10000, ignore_errors=False)
    if df.height == 0:
        raise RuntimeError(f"CSV file is empty: {csv_path}")
    return df


def normalize_ohlc(df: pl.DataFrame, csv_path: Path) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    detected = detect_schema(df.columns)
    validate_detected_schema(detected, csv_path)

    time_col = detected["time"]
    open_col = detected["open"]
    high_col = detected["high"]
    low_col = detected["low"]
    close_col = detected["close"]
    volume_col = detected["volume"]

    assert time_col is not None
    assert open_col is not None
    assert high_col is not None
    assert low_col is not None
    assert close_col is not None

    selected_exprs = [
        pl.col(time_col).alias("time_raw"),
        pl.col(open_col).alias("open_raw"),
        pl.col(high_col).alias("high_raw"),
        pl.col(low_col).alias("low_raw"),
        pl.col(close_col).alias("close_raw"),
    ]

    if volume_col is not None:
        selected_exprs.append(pl.col(volume_col).alias("volume_raw"))

    work = df.select(selected_exprs)

    if "volume_raw" not in work.columns:
        work = work.with_columns(pl.lit(0.0).alias("volume_raw"))

    work = work.with_columns(
        [
            pl.col("time_raw").cast(pl.Utf8, strict=False).alias("time_text"),
            pl.col("open_raw").cast(pl.Float64, strict=False).alias("open"),
            pl.col("high_raw").cast(pl.Float64, strict=False).alias("high"),
            pl.col("low_raw").cast(pl.Float64, strict=False).alias("low"),
            pl.col("close_raw").cast(pl.Float64, strict=False).alias("close"),
            pl.col("volume_raw").cast(pl.Float64, strict=False).fill_null(0.0).alias("volume"),
        ]
    )

    work = work.with_columns(
        [
            pl.coalesce(
                [
                    pl.col("time_text").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False),
                    pl.col("time_text").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S", strict=False),
                    pl.col("time_text").str.strptime(pl.Datetime, format="%Y.%m.%d %H:%M:%S", strict=False),
                    pl.col("time_text").str.strptime(pl.Datetime, format="%d/%m/%Y %H:%M:%S", strict=False),
                    pl.col("time_text").str.strptime(pl.Datetime, strict=False),
                ]
            ).alias("time")
        ]
    )

    null_row_count_before_clean = work.filter(
        pl.any_horizontal(
            [
                pl.col("time").is_null(),
                pl.col("open").is_null(),
                pl.col("high").is_null(),
                pl.col("low").is_null(),
                pl.col("close").is_null(),
            ]
        )
    ).height

    cleaned = (
        work
        .drop(["time_raw", "open_raw", "high_raw", "low_raw", "close_raw", "volume_raw", "time_text"])
        .drop_nulls(["time", "open", "high", "low", "close"])
        .sort("time")
    )

    duplicate_time_count = cleaned.group_by("time").count().filter(pl.col("count") > 1).height
    rows_before_dedup = cleaned.height

    cleaned = cleaned.unique(subset=["time"], keep="first").sort("time")
    removed_duplicate_time_count = rows_before_dedup - cleaned.height

    if cleaned.height < 300:
        raise RuntimeError(f"Too few rows after normalization in {csv_path}: {cleaned.height}")

    cleaned = cleaned.select(["time", "open", "high", "low", "close", "volume"])

    stats = {
        "null_row_count_before_clean": int(null_row_count_before_clean),
        "removed_null_row_count": int(null_row_count_before_clean),
        "duplicate_time_count": int(duplicate_time_count),
        "removed_duplicate_time_count": int(removed_duplicate_time_count),
        "row_count": int(cleaned.height),
        "first_time": cleaned.select(pl.col("time").min()).item(),
        "last_time": cleaned.select(pl.col("time").max()).item(),
    }
    return cleaned, stats


def validate_ohlc_integrity(df: pl.DataFrame, csv_path: Path) -> Dict[str, Any]:
    if df.height == 0:
        raise RuntimeError(f"Normalized dataframe is empty: {csv_path}")

    invalid_price_rows = df.filter(
        (pl.col("high") < pl.col("low"))
        | (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0)
    ).height
    if invalid_price_rows > 0:
        raise RuntimeError(f"Invalid OHLC rows found in {csv_path}: {invalid_price_rows}")

    descending_time_rows = df.with_columns(pl.col("time").diff().alias("time_diff")).filter(
        pl.col("time_diff").is_not_null() & (pl.col("time_diff") <= pl.duration(seconds=0))
    ).height
    if descending_time_rows > 0:
        raise RuntimeError(f"Non-increasing timestamps found in {csv_path}: {descending_time_rows}")

    duplicate_time_rows = df.group_by("time").count().filter(pl.col("count") > 1).height
    if duplicate_time_rows > 0:
        raise RuntimeError(f"Duplicate timestamps remain in {csv_path}: {duplicate_time_rows}")

    return {
        "invalid_price_rows": int(invalid_price_rows),
        "descending_time_rows": int(descending_time_rows),
        "duplicate_time_rows_after_clean": int(duplicate_time_rows),
    }


def parquet_path_for_tf(timeframe: str) -> Path:
    return OUTDIR / f"{SYMBOL}_{timeframe}.parquet"


def convert_one_timeframe(timeframe: str) -> ConversionRecord:
    csv_path = Path(CSV_DATASET_MAP[timeframe])
    parquet_path = parquet_path_for_tf(timeframe)

    raw_df = load_csv_as_polars(csv_path)
    normalized_df, norm_stats = normalize_ohlc(raw_df, csv_path)
    integrity_stats = validate_ohlc_integrity(normalized_df, csv_path)

    normalized_df.write_parquet(parquet_path, compression="zstd")

    file_size_bytes = parquet_path.stat().st_size if parquet_path.exists() else 0
    first_time = norm_stats["first_time"]
    last_time = norm_stats["last_time"]

    if isinstance(first_time, datetime):
        first_time_text = first_time.isoformat()
    else:
        first_time_text = str(first_time) if first_time is not None else None

    if isinstance(last_time, datetime):
        last_time_text = last_time.isoformat()
    else:
        last_time_text = str(last_time) if last_time is not None else None

    if integrity_stats["duplicate_time_rows_after_clean"] != 0:
        raise RuntimeError(f"Unexpected duplicate rows after clean in {csv_path}")

    return ConversionRecord(
        version=VERSION,
        symbol=SYMBOL,
        timeframe=timeframe,
        source_csv=str(csv_path),
        output_parquet=str(parquet_path),
        row_count=int(norm_stats["row_count"]),
        duplicate_time_count=int(norm_stats["duplicate_time_count"]),
        null_row_count_before_clean=int(norm_stats["null_row_count_before_clean"]),
        removed_null_row_count=int(norm_stats["removed_null_row_count"]),
        removed_duplicate_time_count=int(norm_stats["removed_duplicate_time_count"]),
        first_time=first_time_text,
        last_time=last_time_text,
        file_size_bytes=int(file_size_bytes),
        generated_at_utc=utc_now_iso(),
    )


def build_summary(records: List[ConversionRecord]) -> Dict[str, Any]:
    total_rows = sum(x.row_count for x in records)
    total_file_size_bytes = sum(x.file_size_bytes for x in records)

    rows_by_tf = {x.timeframe: x.row_count for x in records}
    files_by_tf = {x.timeframe: x.output_parquet for x in records}
    sizes_by_tf = {x.timeframe: x.file_size_bytes for x in records}

    summary = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "symbol": SYMBOL,
        "timeframes": RESEARCH_TIMEFRAMES,
        "output_dir": str(OUTDIR),
        "total_timeframes": len(records),
        "total_rows": int(total_rows),
        "total_file_size_bytes": int(total_file_size_bytes),
        "rows_by_timeframe": rows_by_tf,
        "parquet_files_by_timeframe": files_by_tf,
        "file_sizes_by_timeframe": sizes_by_tf,
        "schema": ["time", "open", "high", "low", "close", "volume"],
        "compression": "zstd",
    }
    return summary


def reset_manifest(path: Path) -> None:
    if path.exists():
        path.unlink()


def main() -> None:
    ensure_dir(OUTDIR)
    reset_manifest(MANIFEST_PATH)

    records: List[ConversionRecord] = []

    print("=" * 120)
    print(f"[START] version={VERSION}")
    print(f"[START] symbol={SYMBOL}")
    print(f"[START] output_dir={OUTDIR}")
    print(f"[START] total_timeframes={len(RESEARCH_TIMEFRAMES)}")
    print("=" * 120)

    for timeframe in RESEARCH_TIMEFRAMES:
        record = convert_one_timeframe(timeframe)
        records.append(record)
        append_jsonl(MANIFEST_PATH, asdict(record))

        print(
            f"[DONE] timeframe={record.timeframe} "
            f"rows={record.row_count} "
            f"first_time={record.first_time} "
            f"last_time={record.last_time} "
            f"parquet={record.output_parquet}"
        )

    summary = build_summary(records)
    write_json(SUMMARY_PATH, summary)

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] manifest={MANIFEST_PATH}")
    print(f"[DONE] summary={SUMMARY_PATH}")
    print(f"[DONE] total_timeframes={summary['total_timeframes']}")
    print(f"[DONE] total_rows={summary['total_rows']}")
    print(f"[DONE] total_file_size_bytes={summary['total_file_size_bytes']}")
    print("=" * 120)


if __name__ == "__main__":
    main()
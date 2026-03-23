# ==================================================================================================
# FILE: build_base_feature_cache.py
# PATH: C:\Data\Bot\Local_LLM\gold_research\jobs\build_base_feature_cache.py
# VERSION: v1.0.2
#
# CHANGELOG:
# - v1.0.2
#   1) Rewrite whole file for production-fast base feature cache generation
#   2) Remove catastrophic rolling quantile loop bottleneck from old version
#   3) Use Numba JIT for EMA / ATR / ADX / RSI core indicator calculations
#   4) Use Polars native rolling_max / rolling_min for swing structure
#   5) Keep output columns aligned with downstream runner contract
#   6) Add per-timeframe progress logs so the job no longer appears frozen
#
# DESIGN RATIONALE:
# - Base features that are reused across many strategies must be cached once.
# - Strategy-specific features should be computed only inside the shard that needs them.
# - This file is the reusable feature layer feeding the fast ranking engine.
#
# OUTPUT:
# - C:\Data\Bot\central_feature_cache\XAUUSD_<TF>_base_features.parquet
# - C:\Data\Bot\central_feature_cache\base_feature_cache_manifest.jsonl
# - C:\Data\Bot\central_feature_cache\base_feature_cache_summary.json
# ==================================================================================================

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
from numba import njit

VERSION = "v1.0.2"
SYMBOL = "XAUUSD"

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

PARQUET_INDIR = Path(r"C:\Data\Bot\central_market_data\parquet")
OUTDIR = Path(r"C:\Data\Bot\central_feature_cache")
MANIFEST_PATH = OUTDIR / "base_feature_cache_manifest.jsonl"
SUMMARY_PATH = OUTDIR / "base_feature_cache_summary.json"

INPUT_COLUMNS = ["time", "open", "high", "low", "close", "volume"]

OUTPUT_COLUMNS = [
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ema_9",
    "ema_20",
    "ema_50",
    "ema_200",
    "atr_14",
    "atr_pct",
    "atr_pct_sma_200",
    "adx_14",
    "rsi_14",
    "return_1",
    "range",
    "body",
    "upper_wick",
    "lower_wick",
    "bull_stack",
    "bear_stack",
    "vol_bucket",
    "trend_bucket",
    "price_location_bucket",
    "bar_index",
    "swing_high_5",
    "swing_low_5",
    "swing_high_10",
    "swing_low_10",
]


@dataclass
class FeatureCacheRecord:
    version: str
    symbol: str
    timeframe: str
    input_parquet: str
    output_parquet: str
    row_count: int
    column_count: int
    first_time: Optional[str]
    last_time: Optional[str]
    file_size_bytes: int
    build_sec: float
    generated_at_utc: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dt_to_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def input_parquet_path(timeframe: str) -> Path:
    return PARQUET_INDIR / f"{SYMBOL}_{timeframe}.parquet"


def output_feature_path(timeframe: str) -> Path:
    return OUTDIR / f"{SYMBOL}_{timeframe}_base_features.parquet"


def validate_input_schema(df: pl.DataFrame, source_path: Path) -> None:
    missing = set(INPUT_COLUMNS) - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in {source_path}: {sorted(missing)}")

    if df.height < 300:
        raise RuntimeError(f"Too few rows in {source_path}: {df.height}")

    duplicate_time_rows = df.group_by("time").len().filter(pl.col("len") > 1).height
    if duplicate_time_rows > 0:
        raise RuntimeError(f"Duplicate timestamps found in {source_path}: {duplicate_time_rows}")


@njit(cache=True)
def ema_numba(values: np.ndarray, period: int) -> np.ndarray:
    n = values.size
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out

    alpha = 2.0 / (period + 1.0)
    out[0] = values[0]
    for i in range(1, n):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


@njit(cache=True)
def sma_numba(values: np.ndarray, period: int) -> np.ndarray:
    n = values.size
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out

    running_sum = 0.0
    for i in range(n):
        running_sum += values[i]
        if i >= period:
            running_sum -= values[i - period]

        count = i + 1 if i + 1 < period else period
        out[i] = running_sum / count

    return out


@njit(cache=True)
def atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = close.size
    tr = np.empty(n, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)

    if n == 0:
        return out

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    out[0] = tr[0]
    for i in range(1, n):
        out[i] = ((out[i - 1] * (period - 1)) + tr[i]) / period

    return out


@njit(cache=True)
def rsi_numba(close: np.ndarray, period: int) -> np.ndarray:
    n = close.size
    out = np.empty(n, dtype=np.float64)

    if n == 0:
        return out

    out[0] = 50.0
    if n == 1:
        return out

    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        delta = close[i] - close[i - 1]
        if delta > 0.0:
            gains[i] = delta
        else:
            losses[i] = -delta

    warmup = period if period < n else n - 1
    avg_gain = 0.0
    avg_loss = 0.0

    for i in range(1, warmup + 1):
        avg_gain += gains[i]
        avg_loss += losses[i]

    if warmup > 0:
        avg_gain /= warmup
        avg_loss /= warmup

    upto = period if period < n else n
    for i in range(1, upto):
        out[i] = 50.0

    if n <= period:
        return out

    if avg_loss == 0.0:
        out[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, n):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

        if avg_loss == 0.0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))

    return out


@njit(cache=True)
def adx_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = close.size
    out = np.zeros(n, dtype=np.float64)

    if n < 2:
        return out

    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        if up_move > down_move and up_move > 0.0:
            plus_dm[i] = up_move

        if down_move > up_move and down_move > 0.0:
            minus_dm[i] = down_move

        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    atr_smooth = np.zeros(n, dtype=np.float64)
    plus_smooth = np.zeros(n, dtype=np.float64)
    minus_smooth = np.zeros(n, dtype=np.float64)

    start = period if period < n else n - 1
    tr_seed = 0.0
    plus_seed = 0.0
    minus_seed = 0.0

    for i in range(1, start + 1):
        tr_seed += tr[i]
        plus_seed += plus_dm[i]
        minus_seed += minus_dm[i]

    if start < n:
        atr_smooth[start] = tr_seed
        plus_smooth[start] = plus_seed
        minus_smooth[start] = minus_seed

    dx = np.zeros(n, dtype=np.float64)
    for i in range(start, n):
        if i > start:
            atr_smooth[i] = atr_smooth[i - 1] - (atr_smooth[i - 1] / period) + tr[i]
            plus_smooth[i] = plus_smooth[i - 1] - (plus_smooth[i - 1] / period) + plus_dm[i]
            minus_smooth[i] = minus_smooth[i - 1] - (minus_smooth[i - 1] / period) + minus_dm[i]

        if atr_smooth[i] > 0.0:
            plus_di = 100.0 * (plus_smooth[i] / atr_smooth[i])
            minus_di = 100.0 * (minus_smooth[i] / atr_smooth[i])
            denom = plus_di + minus_di
            if denom > 0.0:
                dx[i] = 100.0 * abs(plus_di - minus_di) / denom

    adx_start = (period * 2) - 1
    if adx_start < n:
        dx_sum = 0.0
        count = 0
        for i in range(period, adx_start + 1):
            dx_sum += dx[i]
            count += 1

        if count > 0:
            out[adx_start] = dx_sum / count

        for i in range(adx_start + 1, n):
            out[i] = ((out[i - 1] * (period - 1)) + dx[i]) / period

    return out


def warmup_numba() -> None:
    sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    ema_numba(sample, 3)
    sma_numba(sample, 3)
    atr_numba(sample, sample, sample, 3)
    rsi_numba(sample, 3)
    adx_numba(sample, sample, sample, 3)


def build_feature_frame(df: pl.DataFrame) -> pl.DataFrame:
    open_np = df["open"].to_numpy().astype(np.float64, copy=False)
    high_np = df["high"].to_numpy().astype(np.float64, copy=False)
    low_np = df["low"].to_numpy().astype(np.float64, copy=False)
    close_np = df["close"].to_numpy().astype(np.float64, copy=False)

    ema_9 = ema_numba(close_np, 9)
    ema_20 = ema_numba(close_np, 20)
    ema_50 = ema_numba(close_np, 50)
    ema_200 = ema_numba(close_np, 200)

    atr_14 = atr_numba(high_np, low_np, close_np, 14)
    atr_pct = np.where(close_np != 0.0, (atr_14 / close_np) * 100.0, 0.0)
    atr_pct_sma_200 = sma_numba(atr_pct, 200)

    adx_14 = adx_numba(high_np, low_np, close_np, 14)
    rsi_14 = rsi_numba(close_np, 14)

    prev_close = np.empty_like(close_np)
    prev_close[0] = close_np[0]
    prev_close[1:] = close_np[:-1]
    return_1 = np.where(prev_close != 0.0, (close_np - prev_close) / prev_close, 0.0)
    return_1[0] = 0.0

    range_np = high_np - low_np
    body_np = np.abs(close_np - open_np)
    upper_wick_np = high_np - np.maximum(open_np, close_np)
    lower_wick_np = np.minimum(open_np, close_np) - low_np

    bull_stack = (ema_9 > ema_20) & (ema_20 > ema_50) & (ema_50 > ema_200)
    bear_stack = (ema_9 < ema_20) & (ema_20 < ema_50) & (ema_50 < ema_200)

    vol_ratio = np.where(atr_pct_sma_200 > 1e-12, atr_pct / atr_pct_sma_200, 1.0)
    vol_bucket = np.full(close_np.size, "MID_VOL", dtype=object)
    vol_bucket[vol_ratio < 0.85] = "LOW_VOL"
    vol_bucket[vol_ratio >= 1.15] = "HIGH_VOL"

    trend_bucket = np.full(close_np.size, "WEAK_TREND", dtype=object)
    trend_bucket[adx_14 >= 20.0] = "MID_TREND"
    trend_bucket[adx_14 >= 25.0] = "STRONG_TREND"

    near_band = np.maximum(atr_14 * 0.15, close_np * 0.00015)
    price_location_bucket = np.full(close_np.size, "NEAR_EMA_STACK", dtype=object)
    price_location_bucket[close_np > (ema_50 + near_band)] = "ABOVE_EMA_STACK"
    price_location_bucket[close_np < (ema_50 - near_band)] = "BELOW_EMA_STACK"

    feature_df = (
        df.select(INPUT_COLUMNS)
        .with_columns(
            [
                pl.Series("ema_9", ema_9),
                pl.Series("ema_20", ema_20),
                pl.Series("ema_50", ema_50),
                pl.Series("ema_200", ema_200),
                pl.Series("atr_14", atr_14),
                pl.Series("atr_pct", atr_pct),
                pl.Series("atr_pct_sma_200", atr_pct_sma_200),
                pl.Series("adx_14", adx_14),
                pl.Series("rsi_14", rsi_14),
                pl.Series("return_1", return_1),
                pl.Series("range", range_np),
                pl.Series("body", body_np),
                pl.Series("upper_wick", upper_wick_np),
                pl.Series("lower_wick", lower_wick_np),
                pl.Series("bull_stack", bull_stack),
                pl.Series("bear_stack", bear_stack),
                pl.Series("vol_bucket", vol_bucket.tolist()),
                pl.Series("trend_bucket", trend_bucket.tolist()),
                pl.Series("price_location_bucket", price_location_bucket.tolist()),
                pl.Series("bar_index", np.arange(df.height, dtype=np.int64)),
            ]
        )
        .with_columns(
            [
                pl.col("high").shift(1).rolling_max(window_size=5).alias("swing_high_5"),
                pl.col("low").shift(1).rolling_min(window_size=5).alias("swing_low_5"),
                pl.col("high").shift(1).rolling_max(window_size=10).alias("swing_high_10"),
                pl.col("low").shift(1).rolling_min(window_size=10).alias("swing_low_10"),
            ]
        )
        .select(OUTPUT_COLUMNS)
    )

    return feature_df


def validate_feature_frame(df: pl.DataFrame, source_path: Path) -> None:
    missing = set(OUTPUT_COLUMNS) - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing feature columns for {source_path}: {sorted(missing)}")

    if df.height == 0:
        raise RuntimeError(f"Feature dataframe is empty: {source_path}")

    if int(df["bar_index"][0]) != 0 or int(df["bar_index"][-1]) != df.height - 1:
        raise RuntimeError(f"Invalid bar_index sequence in {source_path}")

    duplicate_time_rows = df.group_by("time").len().filter(pl.col("len") > 1).height
    if duplicate_time_rows > 0:
        raise RuntimeError(f"Duplicate timestamps remain after feature build in {source_path}: {duplicate_time_rows}")


def build_one_timeframe(timeframe: str) -> FeatureCacheRecord:
    started_at = time.perf_counter()

    input_path = input_parquet_path(timeframe)
    output_path = output_feature_path(timeframe)

    if not input_path.exists():
        raise RuntimeError(f"Input parquet not found for timeframe {timeframe}: {input_path}")

    print(f"[LOAD] timeframe={timeframe} input={input_path}")
    df = pl.read_parquet(input_path, columns=INPUT_COLUMNS)
    validate_input_schema(df, input_path)
    print(f"[LOAD-DONE] timeframe={timeframe} rows={df.height} columns={len(df.columns)}")

    feature_df = build_feature_frame(df)
    validate_feature_frame(feature_df, input_path)

    print(f"[WRITE] timeframe={timeframe} output={output_path}")
    feature_df.write_parquet(output_path, compression="zstd")

    build_sec = time.perf_counter() - started_at
    first_time = feature_df.select(pl.col("time").min()).item()
    last_time = feature_df.select(pl.col("time").max()).item()
    file_size_bytes = output_path.stat().st_size if output_path.exists() else 0

    return FeatureCacheRecord(
        version=VERSION,
        symbol=SYMBOL,
        timeframe=timeframe,
        input_parquet=str(input_path),
        output_parquet=str(output_path),
        row_count=int(feature_df.height),
        column_count=int(len(feature_df.columns)),
        first_time=dt_to_text(first_time),
        last_time=dt_to_text(last_time),
        file_size_bytes=int(file_size_bytes),
        build_sec=round(build_sec, 4),
        generated_at_utc=utc_now_iso(),
    )


def build_summary(records: List[FeatureCacheRecord]) -> Dict[str, Any]:
    return {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "symbol": SYMBOL,
        "input_dir": str(PARQUET_INDIR),
        "output_dir": str(OUTDIR),
        "total_timeframes": len(records),
        "timeframes": [x.timeframe for x in records],
        "total_rows": int(sum(x.row_count for x in records)),
        "total_file_size_bytes": int(sum(x.file_size_bytes for x in records)),
        "feature_columns": OUTPUT_COLUMNS,
        "files_by_timeframe": {x.timeframe: x.output_parquet for x in records},
        "rows_by_timeframe": {x.timeframe: x.row_count for x in records},
        "columns_by_timeframe": {x.timeframe: x.column_count for x in records},
        "build_sec_by_timeframe": {x.timeframe: x.build_sec for x in records},
    }


def main() -> None:
    ensure_dir(OUTDIR)
    reset_file(MANIFEST_PATH)

    warmup_numba()

    records: List[FeatureCacheRecord] = []

    print("=" * 120)
    print(f"[START] version={VERSION}")
    print(f"[START] symbol={SYMBOL}")
    print(f"[START] input_dir={PARQUET_INDIR}")
    print(f"[START] output_dir={OUTDIR}")
    print(f"[START] total_timeframes={len(RESEARCH_TIMEFRAMES)}")
    print("=" * 120)

    total_started = time.perf_counter()

    for idx, timeframe in enumerate(RESEARCH_TIMEFRAMES, start=1):
        print(f"[TF-START] {idx}/{len(RESEARCH_TIMEFRAMES)} timeframe={timeframe}")

        record = build_one_timeframe(timeframe)
        records.append(record)
        append_jsonl(MANIFEST_PATH, asdict(record))

        print(
            f"[DONE] timeframe={record.timeframe} "
            f"rows={record.row_count} "
            f"columns={record.column_count} "
            f"build_sec={record.build_sec} "
            f"first_time={record.first_time} "
            f"last_time={record.last_time} "
            f"feature_parquet={record.output_parquet}"
        )

    summary = build_summary(records)
    write_json(SUMMARY_PATH, summary)

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] manifest={MANIFEST_PATH}")
    print(f"[DONE] summary={SUMMARY_PATH}")
    print(f"[DONE] total_timeframes={summary['total_timeframes']}")
    print(f"[DONE] total_rows={summary['total_rows']}")
    print(f"[DONE] total_elapsed_sec={round(time.perf_counter() - total_started, 4)}")
    print("=" * 120)


if __name__ == "__main__":
    main()
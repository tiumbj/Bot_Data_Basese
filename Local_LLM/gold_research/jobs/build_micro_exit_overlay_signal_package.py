# ============================================================
# ชื่อโค้ด: build_micro_exit_overlay_signal_package.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\build_micro_exit_overlay_signal_package.py
# คำสั่งรัน: python C:\Data\Bot\Local_LLM\gold_research\jobs\build_micro_exit_overlay_signal_package.py
# เวอร์ชัน: v1.0.0
# ============================================================

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

VERSION = "v1.0.0"
CANONICAL_SYMBOL = "XAUUSD"

ROOT = Path(r"C:\Data\Bot")
MARKET_DATA_ROOT = ROOT / "central_market_data" / "tf"
STRATEGY_REGISTRY_ROOT = ROOT / "central_strategy_registry"
CANDIDATES_ROOT = STRATEGY_REGISTRY_ROOT / "candidates"

BASE_STRATEGY_ID = "locked_ms_bos_choch_pullback_atr_adx_ema"
BASE_VERSION = "v1.1.0"
MICRO_EXIT_STRATEGY_ID = "locked_ms_bos_choch_pullback_atr_adx_ema_micro_exit"

RESEARCH_BACKTEST_TIMEFRAMES = [
    "M1", "M2", "M3", "M4", "M5", "M6",
    "M10", "M15", "M30", "H1", "H4", "D1",
]

PRICE_COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "timestamp": ["timestamp", "datetime", "date_time", "time", "Time", "DateTime", "DATE_TIME"],
    "open": ["open", "Open", "o", "O"],
    "high": ["high", "High", "h", "H"],
    "low": ["low", "Low", "l", "L"],
    "close": ["close", "Close", "c", "C"],
    "tick_volume": ["tick_volume", "TickVolume", "tickvol", "volume", "Volume"],
    "spread": ["spread", "Spread"],
    "real_volume": ["real_volume", "RealVolume", "realvol"],
}

SIGNAL_USECOLS = [
    "timestamp", "symbol", "timeframe",
    "long_entry", "short_entry", "exit_long", "exit_short", "sl_price", "tp_price",
]


@dataclass
class StrategyPackage:
    strategy_id: str
    version: str
    package_root: Path
    signals_file: Path
    strategy_spec_file: Path
    approved_parameters_file: Path
    validation_report_file: Path

    @classmethod
    def build(cls, strategy_id: str, version: str) -> "StrategyPackage":
        package_root = CANDIDATES_ROOT / strategy_id / version
        return cls(
            strategy_id=strategy_id,
            version=version,
            package_root=package_root,
            signals_file=package_root / "signals.csv",
            strategy_spec_file=package_root / "strategy_spec.json",
            approved_parameters_file=package_root / "approved_parameters.json",
            validation_report_file=package_root / "validation_report.json",
        )


def find_first_existing_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    lower_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def normalize_price_dataframe(raw_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    columns = list(raw_df.columns)
    out = pd.DataFrame()

    ts_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["timestamp"])
    open_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["open"])
    high_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["high"])
    low_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["low"])
    close_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["close"])

    if ts_col is None:
        date_col = find_first_existing_column(columns, ["date", "Date", "DATE"])
        time_col = find_first_existing_column(columns, ["time", "Time", "TIME"])
        if date_col is not None and time_col is not None:
            out["timestamp"] = pd.to_datetime(
                raw_df[date_col].astype(str).str.strip() + " " + raw_df[time_col].astype(str).str.strip(),
                errors="coerce",
            )
        else:
            raise RuntimeError(f"ไฟล์ราคาคอลัมน์ไม่ครบ TF={timeframe} | missing=['timestamp']")
    else:
        out["timestamp"] = pd.to_datetime(raw_df[ts_col], errors="coerce")

    missing = []
    if open_col is None:
        missing.append("open")
    if high_col is None:
        missing.append("high")
    if low_col is None:
        missing.append("low")
    if close_col is None:
        missing.append("close")

    if missing:
        raise RuntimeError(f"ไฟล์ราคาคอลัมน์ไม่ครบ TF={timeframe} | missing={missing}")

    out["open"] = pd.to_numeric(raw_df[open_col], errors="coerce")
    out["high"] = pd.to_numeric(raw_df[high_col], errors="coerce")
    out["low"] = pd.to_numeric(raw_df[low_col], errors="coerce")
    out["close"] = pd.to_numeric(raw_df[close_col], errors="coerce")

    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return out


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    hl = df["high"] - df["low"]
    hc = (df["high"] - prev_close).abs()
    lc = (df["low"] - prev_close).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def load_price_data(timeframe: str) -> pd.DataFrame:
    file_path = MARKET_DATA_ROOT / f"{CANONICAL_SYMBOL}_{timeframe}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูลกลาง | tf={timeframe} file={file_path}")
    raw_df = pd.read_csv(file_path)
    return normalize_price_dataframe(raw_df, timeframe)


def load_base_signals_for_timeframe(timeframe: str) -> pd.DataFrame:
    file_path = (
        CANDIDATES_ROOT
        / BASE_STRATEGY_ID
        / BASE_VERSION
        / "signals.csv"
    )
    if not file_path.exists():
        raise FileNotFoundError(f"ไม่พบ base signals | file={file_path}")

    chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(file_path, usecols=SIGNAL_USECOLS, chunksize=250000):
        chunk = chunk.loc[chunk["timeframe"].astype(str) == timeframe].copy()
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        raise RuntimeError(f"ไม่พบ base signals สำหรับ TF={timeframe}")

    df = pd.concat(chunks, axis=0, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ["long_entry", "short_entry", "exit_long", "exit_short"]:
        df[col] = df[col].fillna(False).astype(bool)
    df["sl_price"] = pd.to_numeric(df["sl_price"], errors="coerce")
    df["tp_price"] = pd.to_numeric(df["tp_price"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return df


def compute_micro_exit_overlay(price_df: pd.DataFrame) -> pd.DataFrame:
    work = price_df.copy()
    work["ema5"] = ema(work["close"], 5)
    work["ema10"] = ema(work["close"], 10)
    work["atr14"] = atr(work, 14)

    work["prev_low"] = work["low"].shift(1)
    work["prev_high"] = work["high"].shift(1)
    work["prev_close"] = work["close"].shift(1)
    work["rolling_low_3"] = work["low"].rolling(3, min_periods=3).min().shift(1)
    work["rolling_high_3"] = work["high"].rolling(3, min_periods=3).max().shift(1)

    bearish_reversal = (
        (work["close"] < work["open"])
        & (work["close"] < work["ema5"])
        & (work["close"] < work["prev_low"])
    )

    bullish_reversal = (
        (work["close"] > work["open"])
        & (work["close"] > work["ema5"])
        & (work["close"] > work["prev_high"])
    )

    long_micro_break = (
        work["rolling_low_3"].notna()
        & (work["close"] < work["rolling_low_3"])
    )

    short_micro_break = (
        work["rolling_high_3"].notna()
        & (work["close"] > work["rolling_high_3"])
    )

    long_momentum_fade = (
        (work["close"] < work["ema5"])
        & (work["ema5"] < work["ema10"])
        & (work["close"] < work["prev_close"])
    )

    short_momentum_fade = (
        (work["close"] > work["ema5"])
        & (work["ema5"] > work["ema10"])
        & (work["close"] > work["prev_close"])
    )

    long_atr_stall = (
        work["atr14"].notna()
        & ((work["high"] - work["low"]) < (work["atr14"] * 0.35))
        & (work["close"] < work["ema5"])
    )

    short_atr_stall = (
        work["atr14"].notna()
        & ((work["high"] - work["low"]) < (work["atr14"] * 0.35))
        & (work["close"] > work["ema5"])
    )

    overlay = pd.DataFrame(index=work.index)
    overlay["timestamp"] = work["timestamp"]
    overlay["micro_exit_long"] = (bearish_reversal | long_micro_break | long_momentum_fade | long_atr_stall).fillna(False)
    overlay["micro_exit_short"] = (bullish_reversal | short_micro_break | short_momentum_fade | short_atr_stall).fillna(False)
    return overlay


def build_overlay_signals_for_timeframe(timeframe: str) -> tuple[pd.DataFrame, dict]:
    price_df = load_price_data(timeframe)
    base_signals = load_base_signals_for_timeframe(timeframe)
    overlay = compute_micro_exit_overlay(price_df)

    merged = price_df[["timestamp"]].merge(
        base_signals,
        on="timestamp",
        how="left",
    ).merge(
        overlay,
        on="timestamp",
        how="left",
    )

    for col in ["long_entry", "short_entry", "exit_long", "exit_short", "micro_exit_long", "micro_exit_short"]:
        merged[col] = merged[col].fillna(False).astype(bool)

    merged["symbol"] = merged["symbol"].fillna(CANONICAL_SYMBOL)
    merged["timeframe"] = merged["timeframe"].fillna(timeframe)
    merged["sl_price"] = pd.to_numeric(merged["sl_price"], errors="coerce")
    merged["tp_price"] = pd.to_numeric(merged["tp_price"], errors="coerce")

    merged["exit_long"] = merged["exit_long"] | merged["micro_exit_long"]
    merged["exit_short"] = merged["exit_short"] | merged["micro_exit_short"]

    out = merged[
        [
            "timestamp",
            "symbol",
            "timeframe",
            "long_entry",
            "short_entry",
            "exit_long",
            "exit_short",
            "sl_price",
            "tp_price",
        ]
    ].copy()

    summary = {
        "timeframe": timeframe,
        "rows": int(len(out)),
        "long_entries": int(out["long_entry"].sum()),
        "short_entries": int(out["short_entry"].sum()),
        "base_exit_long_before_overlay": int(base_signals["exit_long"].sum()),
        "base_exit_short_before_overlay": int(base_signals["exit_short"].sum()),
        "micro_exit_long_added": int(overlay["micro_exit_long"].sum()),
        "micro_exit_short_added": int(overlay["micro_exit_short"].sum()),
        "exit_long_after_overlay": int(out["exit_long"].sum()),
        "exit_short_after_overlay": int(out["exit_short"].sum()),
        "first_timestamp": out["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S") if not out.empty else "",
        "last_timestamp": out["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S") if not out.empty else "",
    }
    return out, summary


def build_strategy_spec() -> dict:
    return {
        "strategy_id": MICRO_EXIT_STRATEGY_ID,
        "version": VERSION,
        "status": "candidate",
        "canonical_symbol": CANONICAL_SYMBOL,
        "research_backtest_timeframes": RESEARCH_BACKTEST_TIMEFRAMES,
        "deployment_entry_timeframes": ["M1", "M5", "M10", "M15", "H1", "H4"],
        "strategy_variant": "base_plus_micro_exit",
        "base_strategy_id": BASE_STRATEGY_ID,
        "base_strategy_version": BASE_VERSION,
        "concept": "Base locked market structure strategy + mandatory micro exit overlay",
        "entry_rules": [
            "คง entry เดิมจาก base strategy",
        ],
        "exit_rules": [
            "คง base exit เดิม",
            "เพิ่ม micro exit เมื่อเกิด micro structure break / momentum fade / ATR stall ระดับสั้น",
        ],
        "filters": [
            "micro exit ใช้เป็น overlay เท่านั้น",
            "ไม่แก้ entry และไม่แก้ SL/TP เดิม เพื่อวัดผลกระทบของ exit layer ให้ชัด",
        ],
    }


def build_approved_parameters() -> dict:
    return {
        "strategy_id": MICRO_EXIT_STRATEGY_ID,
        "version": VERSION,
        "strategy_variant": "base_plus_micro_exit",
        "base_strategy_id": BASE_STRATEGY_ID,
        "base_strategy_version": BASE_VERSION,
        "parameters": {
            "ema_micro_fast": 5,
            "ema_micro_slow": 10,
            "atr_length": 14,
            "atr_stall_threshold": 0.35,
            "micro_break_lookback": 3,
        },
        "timeframes": RESEARCH_BACKTEST_TIMEFRAMES,
        "note": "overlay นี้เพิ่มเฉพาะ micro exit โดยคง entry/sl/tp เดิมทั้งหมด",
    }


def build_validation_report(signal_summary: List[dict]) -> dict:
    return {
        "strategy_id": MICRO_EXIT_STRATEGY_ID,
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "candidate_built_not_backtested_yet",
        "strategy_variant": "base_plus_micro_exit",
        "base_strategy_id": BASE_STRATEGY_ID,
        "base_strategy_version": BASE_VERSION,
        "signal_summary": signal_summary,
        "totals": {
            "long_entries": int(sum(item["long_entries"] for item in signal_summary)),
            "short_entries": int(sum(item["short_entries"] for item in signal_summary)),
            "micro_exit_long_added": int(sum(item["micro_exit_long_added"] for item in signal_summary)),
            "micro_exit_short_added": int(sum(item["micro_exit_short_added"] for item in signal_summary)),
        },
    }


def main() -> None:
    package = StrategyPackage.build(strategy_id=MICRO_EXIT_STRATEGY_ID, version=VERSION)
    package.package_root.mkdir(parents=True, exist_ok=True)

    if package.signals_file.exists():
        package.signals_file.unlink()

    signal_summary: List[dict] = []
    first_write = True

    for tf in RESEARCH_BACKTEST_TIMEFRAMES:
        tf_df, summary = build_overlay_signals_for_timeframe(tf)
        out_df = tf_df.copy()
        out_df["timestamp"] = pd.to_datetime(out_df["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        out_df.to_csv(package.signals_file, mode="w" if first_write else "a", header=first_write, index=False)
        first_write = False

        signal_summary.append(summary)

        print(
            f"[DONE] tf={tf} rows={summary['rows']} long_entries={summary['long_entries']} "
            f"short_entries={summary['short_entries']} micro_exit_long_added={summary['micro_exit_long_added']} "
            f"micro_exit_short_added={summary['micro_exit_short_added']}"
        )

    package.strategy_spec_file.write_text(
        json.dumps(build_strategy_spec(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    package.approved_parameters_file.write_text(
        json.dumps(build_approved_parameters(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    package.validation_report_file.write_text(
        json.dumps(build_validation_report(signal_summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[DONE] micro exit strategy package built | strategy_id={MICRO_EXIT_STRATEGY_ID} version={VERSION}")
    print(f"[PACKAGE] {package.package_root}")
    print(f"[SIGNALS] {package.signals_file}")
    print(f"[SPEC] {package.strategy_spec_file}")


if __name__ == "__main__":
    main()

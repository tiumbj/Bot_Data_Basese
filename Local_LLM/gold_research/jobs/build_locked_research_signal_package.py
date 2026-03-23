# ============================================================
# ชื่อโค้ด: build_locked_research_signal_package.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\build_locked_research_signal_package.py
# คำสั่งรัน: python C:\Data\Bot\Local_LLM\gold_research\jobs\build_locked_research_signal_package.py
# เวอร์ชัน: v1.1.0
# ============================================================

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

VERSION = "v1.1.0"
CANONICAL_SYMBOL = "XAUUSD"

ROOT = Path(r"C:\Data\Bot")
MARKET_DATA_ROOT = ROOT / "central_market_data" / "tf"
STRATEGY_REGISTRY_ROOT = ROOT / "central_strategy_registry"
CANDIDATES_ROOT = STRATEGY_REGISTRY_ROOT / "candidates"

RESEARCH_BACKTEST_TIMEFRAMES = [
    "M1", "M2", "M3", "M4", "M5", "M6",
    "M10", "M15", "M30", "H1", "H4", "D1",
]

DEPLOYMENT_ENTRY_TIMEFRAMES = ["M1", "M5", "M10", "M15", "H1", "H4"]

TIMEFRAME_TO_MINUTES: Dict[str, int] = {
    "M1": 1,
    "M2": 2,
    "M3": 3,
    "M4": 4,
    "M5": 5,
    "M6": 6,
    "M10": 10,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}

PRICE_REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close"]
SIGNAL_OUTPUT_COLUMNS = [
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


def adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    atr_rma = tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / atr_rma
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / atr_rma

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    out = pd.DataFrame(index=df.index)
    out["adx14"] = adx_line.fillna(0.0)
    out["di_plus"] = plus_di.fillna(0.0)
    out["di_minus"] = minus_di.fillna(0.0)
    return out


def compute_swings(df: pd.DataFrame, lookback: int = 2) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["swing_high"] = np.nan
    out["swing_low"] = np.nan

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    for i in range(lookback, len(df) - lookback):
        left_high = highs[i - lookback:i]
        right_high = highs[i + 1:i + 1 + lookback]
        left_low = lows[i - lookback:i]
        right_low = lows[i + 1:i + 1 + lookback]

        if highs[i] > left_high.max() and highs[i] >= right_high.max():
            out.iloc[i, out.columns.get_loc("swing_high")] = highs[i]

        if lows[i] < left_low.min() and lows[i] <= right_low.min():
            out.iloc[i, out.columns.get_loc("swing_low")] = lows[i]

    out["last_swing_high"] = out["swing_high"].ffill()
    out["last_swing_low"] = out["swing_low"].ffill()
    return out


def build_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    swings = compute_swings(df, lookback=2)
    out = pd.DataFrame(index=df.index)
    out["last_swing_high"] = swings["last_swing_high"]
    out["last_swing_low"] = swings["last_swing_low"]

    out["bos_up"] = False
    out["bos_down"] = False
    out["choch_up"] = False
    out["choch_down"] = False
    out["bias"] = "NEUTRAL"

    current_bias = "NEUTRAL"

    for i in range(1, len(df)):
        close_i = float(df["close"].iloc[i])
        prev_swing_high = out["last_swing_high"].iloc[i - 1]
        prev_swing_low = out["last_swing_low"].iloc[i - 1]

        bos_up = pd.notna(prev_swing_high) and close_i > float(prev_swing_high)
        bos_down = pd.notna(prev_swing_low) and close_i < float(prev_swing_low)

        choch_up = bos_up and current_bias == "BEARISH"
        choch_down = bos_down and current_bias == "BULLISH"

        out.at[df.index[i], "bos_up"] = bool(bos_up)
        out.at[df.index[i], "bos_down"] = bool(bos_down)
        out.at[df.index[i], "choch_up"] = bool(choch_up)
        out.at[df.index[i], "choch_down"] = bool(choch_down)

        if bos_up:
            current_bias = "BULLISH"
        elif bos_down:
            current_bias = "BEARISH"

        out.at[df.index[i], "bias"] = current_bias

    out["bias"] = out["bias"].replace("", "NEUTRAL").ffill().fillna("NEUTRAL")
    return out


def normalize_price_columns(df: pd.DataFrame, timeframe: str, file_path: Path) -> pd.DataFrame:
    original_columns = list(df.columns)
    lower_map = {str(col).strip().lower(): col for col in df.columns}

    normalized = df.copy()

    timestamp_candidates = [
        "timestamp",
        "datetime",
        "date",
        "time",
        "date_time",
        "date/time",
        "datetime_utc",
    ]

    chosen_timestamp = None
    for candidate in timestamp_candidates:
        if candidate in lower_map:
            chosen_timestamp = lower_map[candidate]
            break

    if chosen_timestamp is None:
        if "date" in lower_map and "time" in lower_map:
            normalized["timestamp"] = (
                normalized[lower_map["date"]].astype(str).str.strip()
                + " "
                + normalized[lower_map["time"]].astype(str).str.strip()
            )
            chosen_timestamp = "timestamp"
        else:
            raise RuntimeError(
                f"ไฟล์ข้อมูลไม่พบคอลัมน์เวลา | tf={timeframe} file={file_path} columns={original_columns}"
            )
    else:
        normalized = normalized.rename(columns={chosen_timestamp: "timestamp"})

    price_aliases = {
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c"],
    }

    for target, aliases in price_aliases.items():
        found = None
        current_map = {str(col).strip().lower(): col for col in normalized.columns}
        for alias in aliases:
            if alias in current_map:
                found = current_map[alias]
                break
        if found is None:
            raise RuntimeError(
                f"ไฟล์ข้อมูลไม่ครบคอลัมน์ราคา | tf={timeframe} missing={target} file={file_path} columns={original_columns}"
            )
        if found != target:
            normalized = normalized.rename(columns={found: target})

    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")
    for col in ["open", "high", "low", "close"]:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    normalized = normalized.dropna(subset=PRICE_REQUIRED_COLUMNS).copy()
    normalized = normalized.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    if normalized.empty:
        raise RuntimeError(f"ไฟล์ข้อมูลว่างหลัง normalize | tf={timeframe} file={file_path}")

    return normalized[PRICE_REQUIRED_COLUMNS].copy()


def generate_signals_for_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    work = df.copy()

    if len(work) < 300:
        raise RuntimeError(f"ข้อมูลน้อยเกินไปสำหรับสร้าง signals | tf={timeframe} rows={len(work)}")

    work["ema20"] = ema(work["close"], 20)
    work["ema50"] = ema(work["close"], 50)
    work["atr14"] = atr(work, 14)

    adx_df = adx(work, 14)
    work["adx14"] = adx_df["adx14"]
    work["di_plus"] = adx_df["di_plus"]
    work["di_minus"] = adx_df["di_minus"]

    ms = build_market_structure(work)
    work["last_swing_high"] = ms["last_swing_high"]
    work["last_swing_low"] = ms["last_swing_low"]
    work["bos_up"] = ms["bos_up"]
    work["bos_down"] = ms["bos_down"]
    work["choch_up"] = ms["choch_up"]
    work["choch_down"] = ms["choch_down"]
    work["bias"] = ms["bias"]

    work["bullish_pullback_zone"] = (
        (work["bias"] == "BULLISH")
        & (work["low"] <= work["ema20"])
        & (work["close"] >= work["ema20"])
        & (work["ema20"] > work["ema50"])
    )

    work["bearish_pullback_zone"] = (
        (work["bias"] == "BEARISH")
        & (work["high"] >= work["ema20"])
        & (work["close"] <= work["ema20"])
        & (work["ema20"] < work["ema50"])
    )

    work["bull_filter_ok"] = (
        (work["atr14"] > 0)
        & (work["adx14"] >= 20.0)
        & (work["ema20"] > work["ema50"])
        & (work["di_plus"] > work["di_minus"])
    )

    work["bear_filter_ok"] = (
        (work["atr14"] > 0)
        & (work["adx14"] >= 20.0)
        & (work["ema20"] < work["ema50"])
        & (work["di_minus"] > work["di_plus"])
    )

    work["long_entry"] = (
        work["bullish_pullback_zone"]
        & work["bull_filter_ok"]
        & (~work["bos_down"])
    )

    work["short_entry"] = (
        work["bearish_pullback_zone"]
        & work["bear_filter_ok"]
        & (~work["bos_up"])
    )

    work["exit_long"] = (
        work["choch_down"]
        | (work["close"] < work["ema50"])
    )

    work["exit_short"] = (
        work["choch_up"]
        | (work["close"] > work["ema50"])
    )

    work["sl_price"] = np.nan
    work["tp_price"] = np.nan

    long_mask = work["long_entry"] & work["last_swing_low"].notna() & work["atr14"].notna()
    short_mask = work["short_entry"] & work["last_swing_high"].notna() & work["atr14"].notna()

    work.loc[long_mask, "sl_price"] = work.loc[long_mask, "last_swing_low"] - work.loc[long_mask, "atr14"] * 0.25
    long_risk = work["close"] - work["sl_price"]
    work.loc[long_mask, "tp_price"] = work.loc[long_mask, "close"] + (long_risk.loc[long_mask] * 2.0)

    work.loc[short_mask, "sl_price"] = work.loc[short_mask, "last_swing_high"] + work.loc[short_mask, "atr14"] * 0.25
    short_risk = work["sl_price"] - work["close"]
    work.loc[short_mask, "tp_price"] = work.loc[short_mask, "close"] - (short_risk.loc[short_mask] * 2.0)

    out = work[[
        "timestamp",
        "long_entry",
        "short_entry",
        "exit_long",
        "exit_short",
        "sl_price",
        "tp_price",
    ]].copy()
    out.insert(1, "symbol", CANONICAL_SYMBOL)
    out.insert(2, "timeframe", timeframe)

    for col in ["long_entry", "short_entry", "exit_long", "exit_short"]:
        out[col] = out[col].fillna(False).astype(bool)

    out["sl_price"] = pd.to_numeric(out["sl_price"], errors="coerce")
    out["tp_price"] = pd.to_numeric(out["tp_price"], errors="coerce")

    ready_mask = (
        work["ema20"].notna()
        & work["ema50"].notna()
        & work["atr14"].notna()
        & work["adx14"].notna()
    )
    out = out.loc[ready_mask].reset_index(drop=True)
    return out[SIGNAL_OUTPUT_COLUMNS].copy()


def load_price_data(timeframe: str) -> pd.DataFrame:
    file_path = MARKET_DATA_ROOT / f"{CANONICAL_SYMBOL}_{timeframe}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูลกลาง | tf={timeframe} file={file_path}")

    df = pd.read_csv(file_path)
    normalized = normalize_price_columns(df=df, timeframe=timeframe, file_path=file_path)
    return normalized


def build_strategy_spec(strategy_id: str, version: str) -> dict:
    return {
        "strategy_id": strategy_id,
        "version": version,
        "status": "candidate",
        "canonical_symbol": CANONICAL_SYMBOL,
        "research_backtest_timeframes": RESEARCH_BACKTEST_TIMEFRAMES,
        "deployment_entry_timeframes": DEPLOYMENT_ENTRY_TIMEFRAMES,
        "concept": "Locked market structure + BOS/CHOCH + swing + pullback zone + ATR/ADX + EMA20/EMA50 filter-only",
        "entry_rules": [
            "bias ต้องเป็น BULLISH หรือ BEARISH จาก BOS/CHOCH",
            "ต้องเกิด pullback zone ที่ EMA20",
            "ADX14 >= 20",
            "EMA20/EMA50 ใช้เป็น filter only",
            "DI direction ต้องสอดคล้องกับฝั่งที่เข้า",
        ],
        "exit_rules": [
            "ออกเมื่อเกิด CHOCH กลับทิศ",
            "หรือ close ตัดกลับ EMA50",
            "หรือโดน SL/TP ที่ engine",
        ],
        "filters": [
            "ATR14 > 0",
            "ADX14 >= 20",
            "EMA20 > EMA50 สำหรับ long",
            "EMA20 < EMA50 สำหรับ short",
        ],
    }


def build_approved_parameters(strategy_id: str, version: str) -> dict:
    return {
        "strategy_id": strategy_id,
        "version": version,
        "parameters": {
            "swing_lookback": 2,
            "atr_length": 14,
            "adx_length": 14,
            "adx_threshold": 20.0,
            "ema_fast": 20,
            "ema_slow": 50,
            "tp_rr": 2.0,
            "sl_atr_buffer": 0.25,
        },
        "timeframes": RESEARCH_BACKTEST_TIMEFRAMES,
        "note": "ชุดนี้เป็น candidate package ตัวแรกสำหรับเริ่มสะสมผล backtest กลาง",
    }


def build_validation_report(strategy_id: str, version: str, signal_summary: List[dict], schema_summary: List[dict]) -> dict:
    total_long = int(sum(item["long_entries"] for item in signal_summary))
    total_short = int(sum(item["short_entries"] for item in signal_summary))
    return {
        "strategy_id": strategy_id,
        "version": version,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "candidate_built_not_backtested_yet",
        "schema_summary": schema_summary,
        "signal_summary": signal_summary,
        "totals": {
            "long_entries": total_long,
            "short_entries": total_short,
            "all_entries": total_long + total_short,
        },
    }


def summarize_schema(df: pd.DataFrame, timeframe: str) -> dict:
    return {
        "timeframe": timeframe,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "first_timestamp": df["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S") if not df.empty else "",
        "last_timestamp": df["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S") if not df.empty else "",
    }


def main() -> None:
    strategy_id = "locked_ms_bos_choch_pullback_atr_adx_ema"
    package = StrategyPackage.build(strategy_id=strategy_id, version=VERSION)
    package.package_root.mkdir(parents=True, exist_ok=True)

    all_signals: List[pd.DataFrame] = []
    signal_summary: List[dict] = []
    schema_summary: List[dict] = []

    for tf in RESEARCH_BACKTEST_TIMEFRAMES:
        price_df = load_price_data(tf)
        schema_summary.append(summarize_schema(price_df, tf))

        signal_df = generate_signals_for_timeframe(price_df, tf)
        all_signals.append(signal_df)

        signal_summary.append(
            {
                "timeframe": tf,
                "rows": int(len(signal_df)),
                "long_entries": int(signal_df["long_entry"].sum()),
                "short_entries": int(signal_df["short_entry"].sum()),
                "first_timestamp": signal_df["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S") if not signal_df.empty else "",
                "last_timestamp": signal_df["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S") if not signal_df.empty else "",
            }
        )

        print(
            f"[DONE] tf={tf} rows={len(signal_df)} "
            f"long_entries={int(signal_df['long_entry'].sum())} "
            f"short_entries={int(signal_df['short_entry'].sum())}"
        )

    combined = pd.concat(all_signals, axis=0, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined = combined.dropna(subset=["timestamp"]).sort_values(["timeframe", "timestamp"]).reset_index(drop=True)
    combined["timestamp"] = combined["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    combined.to_csv(package.signals_file, index=False)

    package.strategy_spec_file.write_text(
        json.dumps(build_strategy_spec(strategy_id, VERSION), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    package.approved_parameters_file.write_text(
        json.dumps(build_approved_parameters(strategy_id, VERSION), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    package.validation_report_file.write_text(
        json.dumps(build_validation_report(strategy_id, VERSION, signal_summary, schema_summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[DONE] strategy package built | strategy_id={strategy_id} version={VERSION}")
    print(f"[PACKAGE] {package.package_root}")
    print(f"[SIGNALS] {package.signals_file}")
    print(f"[SPEC] {package.strategy_spec_file}")


if __name__ == "__main__":
    main()

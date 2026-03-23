# ============================================================
# ชื่อโค้ด: build_micro_exit_v2_component_packages.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\build_micro_exit_v2_component_packages.py
# คำสั่งรัน: python C:\Data\Bot\Local_LLM\gold_research\jobs\build_micro_exit_v2_component_packages.py
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
BASE_STRATEGY_VERSION = "v1.1.0"

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
}

MICRO_VARIANTS = [
    {
        "suffix": "micro_exit_v2_fast_invalidation",
        "label": "Fast invalidation",
        "logic_key": "fast_invalidation",
    },
    {
        "suffix": "micro_exit_v2_momentum_fade",
        "label": "Momentum fade",
        "logic_key": "momentum_fade",
    },
    {
        "suffix": "micro_exit_v2_short_ema_weakness",
        "label": "Short EMA weakness",
        "logic_key": "short_ema_weakness",
    },
    {
        "suffix": "micro_exit_v2_time_stop",
        "label": "Time stop",
        "logic_key": "time_stop",
    },
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


def load_price_data(timeframe: str) -> pd.DataFrame:
    file_path = MARKET_DATA_ROOT / f"{CANONICAL_SYMBOL}_{timeframe}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูลกลาง | tf={timeframe} file={file_path}")
    raw_df = pd.read_csv(file_path)
    return normalize_price_dataframe(raw_df, timeframe)


def load_base_signals() -> pd.DataFrame:
    file_path = CANDIDATES_ROOT / BASE_STRATEGY_ID / BASE_STRATEGY_VERSION / "signals.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"ไม่พบ base signals | file={file_path}")

    df = pd.read_csv(file_path)
    required = ["timestamp", "symbol", "timeframe", "long_entry", "short_entry", "exit_long", "exit_short", "sl_price", "tp_price"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"base signals คอลัมน์ไม่ครบ | missing={missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values(["timeframe", "timestamp"]).reset_index(drop=True)

    for col in ["long_entry", "short_entry", "exit_long", "exit_short"]:
        df[col] = df[col].fillna(False).astype(bool)

    df["sl_price"] = pd.to_numeric(df["sl_price"], errors="coerce")
    df["tp_price"] = pd.to_numeric(df["tp_price"], errors="coerce")
    return df


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


def enrich_price_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["ema5"] = ema(work["close"], 5)
    work["ema10"] = ema(work["close"], 10)
    work["ema20"] = ema(work["close"], 20)
    work["atr14"] = atr(work, 14)
    adx_df = adx(work, 14)
    work["adx14"] = adx_df["adx14"]
    work["di_plus"] = adx_df["di_plus"]
    work["di_minus"] = adx_df["di_minus"]
    work["body"] = (work["close"] - work["open"]).abs()
    work["range"] = (work["high"] - work["low"]).replace(0, np.nan)
    work["body_ratio"] = (work["body"] / work["range"]).fillna(0.0)
    work["bullish_close"] = work["close"] > work["open"]
    work["bearish_close"] = work["close"] < work["open"]
    work["lower_low"] = work["low"] < work["low"].shift(1)
    work["higher_high"] = work["high"] > work["high"].shift(1)
    return work


def apply_fast_invalidation(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    micro_long = ((out["close"] < out["low"].shift(1)) | (out["bearish_close"] & out["lower_low"]))
    micro_short = ((out["close"] > out["high"].shift(1)) | (out["bullish_close"] & out["higher_high"]))
    out["exit_long"] = out["exit_long"] | micro_long.fillna(False)
    out["exit_short"] = out["exit_short"] | micro_short.fillna(False)
    return out


def apply_momentum_fade(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    adx_falling = out["adx14"] < out["adx14"].shift(1)
    stall = out["atr14"] < out["atr14"].rolling(5, min_periods=1).mean()
    weak_body = out["body_ratio"] < 0.25
    micro_long = adx_falling & stall & weak_body & (out["di_plus"] < out["di_plus"].shift(1))
    micro_short = adx_falling & stall & weak_body & (out["di_minus"] < out["di_minus"].shift(1))
    out["exit_long"] = out["exit_long"] | micro_long.fillna(False)
    out["exit_short"] = out["exit_short"] | micro_short.fillna(False)
    return out


def apply_short_ema_weakness(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    micro_long = (out["ema5"] < out["ema10"]) | (out["close"] < out["ema5"])
    micro_short = (out["ema5"] > out["ema10"]) | (out["close"] > out["ema5"])
    out["exit_long"] = out["exit_long"] | micro_long.fillna(False)
    out["exit_short"] = out["exit_short"] | micro_short.fillna(False)
    return out


def apply_time_stop(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    long_counter = np.zeros(len(out), dtype=int)
    short_counter = np.zeros(len(out), dtype=int)
    in_long = False
    in_short = False
    lc = 0
    sc = 0

    for i in range(len(out)):
        if bool(out["long_entry"].iloc[i]):
            in_long = True
            in_short = False
            lc = 0
            sc = 0
        elif bool(out["short_entry"].iloc[i]):
            in_short = True
            in_long = False
            lc = 0
            sc = 0

        if in_long:
            lc += 1
        if in_short:
            sc += 1

        if bool(out["exit_long"].iloc[i]):
            in_long = False
            lc = 0
        if bool(out["exit_short"].iloc[i]):
            in_short = False
            sc = 0

        long_counter[i] = lc
        short_counter[i] = sc

    out["bars_in_long"] = long_counter
    out["bars_in_short"] = short_counter
    out["exit_long"] = out["exit_long"] | (out["bars_in_long"] >= 8)
    out["exit_short"] = out["exit_short"] | (out["bars_in_short"] >= 8)
    return out


def overlay_variant(base_tf_signals: pd.DataFrame, price_df: pd.DataFrame, logic_key: str) -> pd.DataFrame:
    merged = price_df.merge(
        base_tf_signals[["timestamp", "symbol", "timeframe", "long_entry", "short_entry", "exit_long", "exit_short", "sl_price", "tp_price"]],
        on="timestamp",
        how="left",
    )

    for col in ["long_entry", "short_entry", "exit_long", "exit_short"]:
        merged[col] = merged[col].fillna(False).astype(bool)

    merged["symbol"] = merged["symbol"].fillna(CANONICAL_SYMBOL)
    merged["timeframe"] = merged["timeframe"].fillna("")
    merged["sl_price"] = pd.to_numeric(merged["sl_price"], errors="coerce")
    merged["tp_price"] = pd.to_numeric(merged["tp_price"], errors="coerce")
    merged = enrich_price_features(merged)

    if logic_key == "fast_invalidation":
        overlaid = apply_fast_invalidation(merged)
    elif logic_key == "momentum_fade":
        overlaid = apply_momentum_fade(merged)
    elif logic_key == "short_ema_weakness":
        overlaid = apply_short_ema_weakness(merged)
    elif logic_key == "time_stop":
        overlaid = apply_time_stop(merged)
    else:
        raise RuntimeError(f"logic_key ไม่รองรับ: {logic_key}")

    return overlaid[["timestamp", "symbol", "timeframe", "long_entry", "short_entry", "exit_long", "exit_short", "sl_price", "tp_price"]].copy()


def build_strategy_spec(strategy_id: str, logic_label: str) -> dict:
    return {
        "strategy_id": strategy_id,
        "version": VERSION,
        "status": "candidate",
        "canonical_symbol": CANONICAL_SYMBOL,
        "research_backtest_timeframes": RESEARCH_BACKTEST_TIMEFRAMES,
        "deployment_entry_timeframes": ["M1", "M5", "M10", "M15", "H1", "H4"],
        "concept": f"Base locked market structure strategy + micro exit v2 component overlay: {logic_label}",
        "parent_strategy": {
            "strategy_id": BASE_STRATEGY_ID,
            "version": BASE_STRATEGY_VERSION,
        },
        "entry_rules": ["ใช้ entry เดิมจาก base package ตรง ๆ"],
        "exit_rules": ["ใช้ exit เดิมจาก base package", f"เพิ่ม micro exit component: {logic_label}"],
        "filters": ["ไม่แก้ entry filters เดิม", "ทดสอบเฉพาะผลกระทบของ exit layer"],
    }


def build_approved_parameters(logic_key: str) -> dict:
    payload = {
        "strategy_id": f"{BASE_STRATEGY_ID}_{logic_key}",
        "version": VERSION,
        "parameters": {
            "mode": "base_plus_micro_exit_v2_component",
            "parent_strategy_id": BASE_STRATEGY_ID,
            "parent_strategy_version": BASE_STRATEGY_VERSION,
            "logic_key": logic_key,
        },
        "timeframes": RESEARCH_BACKTEST_TIMEFRAMES,
    }
    if logic_key == "fast_invalidation":
        payload["parameters"].update({"close_break_prev_bar": True, "opposite_candle_structure_break": True})
    elif logic_key == "momentum_fade":
        payload["parameters"].update({"adx_falling_required": True, "atr_stall_window": 5, "body_ratio_threshold": 0.25})
    elif logic_key == "short_ema_weakness":
        payload["parameters"].update({"ema_fast": 5, "ema_slow": 10, "close_vs_ema5_required": True})
    elif logic_key == "time_stop":
        payload["parameters"].update({"bars_in_trade_cap": 8})
    return payload


def build_validation_report(strategy_id: str, logic_label: str, summaries: List[dict]) -> dict:
    return {
        "strategy_id": strategy_id,
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "candidate_built_not_backtested_yet",
        "parent_strategy": {"strategy_id": BASE_STRATEGY_ID, "version": BASE_STRATEGY_VERSION},
        "micro_exit_component": logic_label,
        "signal_summary": summaries,
    }


def main() -> None:
    base_signals = load_base_signals()

    for variant in MICRO_VARIANTS:
        strategy_id = f"{BASE_STRATEGY_ID}_{variant['suffix']}"
        package = StrategyPackage.build(strategy_id=strategy_id, version=VERSION)
        package.package_root.mkdir(parents=True, exist_ok=True)

        variant_outputs: List[pd.DataFrame] = []
        summaries: List[dict] = []

        print(f"[START] strategy_id={strategy_id} logic={variant['logic_key']}", flush=True)

        for tf in RESEARCH_BACKTEST_TIMEFRAMES:
            price_df = load_price_data(tf)
            tf_signals = base_signals.loc[base_signals["timeframe"].astype(str) == tf].copy()
            if tf_signals.empty:
                raise RuntimeError(f"base signals ไม่มี timeframe={tf}")

            out_df = overlay_variant(tf_signals, price_df, variant["logic_key"])
            out_df["symbol"] = CANONICAL_SYMBOL
            out_df["timeframe"] = tf

            base_exit_long = int(tf_signals["exit_long"].sum())
            base_exit_short = int(tf_signals["exit_short"].sum())
            new_exit_long = int(out_df["exit_long"].sum())
            new_exit_short = int(out_df["exit_short"].sum())

            summaries.append(
                {
                    "timeframe": tf,
                    "rows": int(len(out_df)),
                    "base_exit_long": base_exit_long,
                    "base_exit_short": base_exit_short,
                    "new_exit_long": new_exit_long,
                    "new_exit_short": new_exit_short,
                    "micro_exit_long_added": int(max(new_exit_long - base_exit_long, 0)),
                    "micro_exit_short_added": int(max(new_exit_short - base_exit_short, 0)),
                    "first_timestamp": out_df["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S") if not out_df.empty else "",
                    "last_timestamp": out_df["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S") if not out_df.empty else "",
                }
            )

            variant_outputs.append(out_df)
            print(
                f"[DONE] strategy_id={strategy_id} tf={tf} "
                f"micro_exit_long_added={max(new_exit_long - base_exit_long, 0)} "
                f"micro_exit_short_added={max(new_exit_short - base_exit_short, 0)}",
                flush=True,
            )

        combined = pd.concat(variant_outputs, axis=0, ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
        combined = combined.dropna(subset=["timestamp"]).sort_values(["timeframe", "timestamp"]).reset_index(drop=True)
        combined["timestamp"] = combined["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        combined.to_csv(package.signals_file, index=False)

        package.strategy_spec_file.write_text(json.dumps(build_strategy_spec(strategy_id, variant["label"]), indent=2, ensure_ascii=False), encoding="utf-8")
        package.approved_parameters_file.write_text(json.dumps(build_approved_parameters(variant["logic_key"]), indent=2, ensure_ascii=False), encoding="utf-8")
        package.validation_report_file.write_text(json.dumps(build_validation_report(strategy_id, variant["label"], summaries), indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"[PACKAGE] {package.package_root}", flush=True)
        print(f"[SIGNALS] {package.signals_file}", flush=True)

    print("[DONE] build_micro_exit_v2_component_packages completed", flush=True)


if __name__ == "__main__":
    main()

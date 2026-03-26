# ============================================================
# ชื่อโค้ด: run_single_research_job.py
# เวอร์ชัน: v1.0.1
# เป้าหมาย: รัน backtest 1 งานจาก job spec (JSON) โดยรองรับ schema เก่าและ schema ใหม่แบบ backward-compatible
# changelog:
# - v1.0.1
#   - เพิ่ม normalize_job_payload(job) เพื่อ validate/normalize และ derive result_key
#   - รองรับ feature_cache_path เป็น input หลัก (schema ใหม่) และ fallback ไป ohlc_csv (schema เก่า)
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\run_single_research_job.py
# คำสั่งรัน: python C:\Data\Bot\Local_LLM\gold_research\jobs\run_single_research_job.py --job <job.json> --result-root <dir>
# ============================================================

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

VERSION = "v1.0.1"


# --------------------------------------------------------------------------------------------------
# DATA MODELS
# --------------------------------------------------------------------------------------------------
@dataclass
class Position:
    side: str
    entry_index: int
    entry_time: str
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_reason: str
    mfe: float = 0.0
    mae: float = 0.0
    bars_held: int = 0


@dataclass
class ClosedTrade:
    side: str
    entry_index: int
    exit_index: int
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    pnl: float
    pnl_r: float
    bars_held: int
    mfe: float
    mae: float
    exit_reason: str
    entry_reason: str
    timeframe: str
    strategy_family: str
    entry_logic: str
    micro_exit: str
    regime_filter: str
    cooldown_bars: int
    side_policy: str
    volatility_filter: str
    trend_strength_filter: str


# --------------------------------------------------------------------------------------------------
# TIMEFRAME HELPERS
# --------------------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------------------
# GENERAL HELPERS
# --------------------------------------------------------------------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return float(value)
    except Exception:
        return default


def stable_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def timeframe_minutes(tf: str) -> int:
    if tf not in TIMEFRAME_TO_MINUTES:
        raise RuntimeError(f"Unsupported timeframe: {tf}")
    return TIMEFRAME_TO_MINUTES[tf]


# --------------------------------------------------------------------------------------------------
# DATA LOADING / NORMALIZATION
# --------------------------------------------------------------------------------------------------
def detect_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    low_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in low_map:
            return low_map[cand.lower()]
    return None


def load_ohlc_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"OHLC CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"OHLC CSV is empty: {path}")

    cols = list(df.columns)

    time_col = detect_column(cols, ["time", "datetime", "date", "timestamp", "ts"])
    open_col = detect_column(cols, ["open", "o"])
    high_col = detect_column(cols, ["high", "h"])
    low_col = detect_column(cols, ["low", "l"])
    close_col = detect_column(cols, ["close", "c"])
    volume_col = detect_column(cols, ["tick_volume", "volume", "vol", "real_volume"])

    missing = []
    if time_col is None:
        missing.append("time")
    if open_col is None:
        missing.append("open")
    if high_col is None:
        missing.append("high")
    if low_col is None:
        missing.append("low")
    if close_col is None:
        missing.append("close")
    if missing:
        raise RuntimeError(f"Missing required OHLC columns in {path}: {missing}")

    out = pd.DataFrame(
        {
            "time": pd.to_datetime(df[time_col], errors="coerce"),
            "open": pd.to_numeric(df[open_col], errors="coerce"),
            "high": pd.to_numeric(df[high_col], errors="coerce"),
            "low": pd.to_numeric(df[low_col], errors="coerce"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
        }
    )

    if volume_col is not None:
        out["volume"] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0.0)
    else:
        out["volume"] = 0.0

    out = out.dropna(subset=["time", "open", "high", "low", "close"]).copy()
    out = out.sort_values("time").reset_index(drop=True)
    out["time"] = out["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    if len(out) < 300:
        raise RuntimeError(f"Not enough bars after normalization: {len(out)} rows")

    return out


def load_ohlc_from_feature_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"feature_cache_path not found: {path}")

    cols = ["time", "open", "high", "low", "close", "volume"]
    try:
        df = pd.read_parquet(path, columns=cols).copy()
    except Exception:
        df = pd.read_parquet(path).copy()

    if df.empty:
        raise RuntimeError(f"feature cache is empty: {path}")

    found_time = detect_column(list(df.columns), ["time", "datetime", "date", "timestamp", "ts"])
    open_col = detect_column(list(df.columns), ["open", "o"])
    high_col = detect_column(list(df.columns), ["high", "h"])
    low_col = detect_column(list(df.columns), ["low", "l"])
    close_col = detect_column(list(df.columns), ["close", "c"])
    volume_col = detect_column(list(df.columns), ["tick_volume", "volume", "vol", "real_volume"])

    missing = []
    if found_time is None:
        missing.append("time")
    if open_col is None:
        missing.append("open")
    if high_col is None:
        missing.append("high")
    if low_col is None:
        missing.append("low")
    if close_col is None:
        missing.append("close")
    if missing:
        raise RuntimeError(f"Feature cache missing required OHLC columns at {path}: {missing}")

    out = pd.DataFrame(
        {
            "time": pd.to_datetime(df[found_time], errors="coerce"),
            "open": pd.to_numeric(df[open_col], errors="coerce"),
            "high": pd.to_numeric(df[high_col], errors="coerce"),
            "low": pd.to_numeric(df[low_col], errors="coerce"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
        }
    )

    if volume_col is not None:
        out["volume"] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0.0)
    else:
        out["volume"] = 0.0

    out = out.dropna(subset=["time", "open", "high", "low", "close"]).copy()
    out = out.sort_values("time").reset_index(drop=True)
    out["time"] = out["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    if len(out) < 300:
        raise RuntimeError(f"Not enough bars after normalization: {len(out)} rows")

    return out


def normalize_job_payload(job: dict) -> dict:
    if not isinstance(job, dict):
        raise RuntimeError("Job payload must be a JSON object")

    normalized: Dict[str, Any] = dict(job)
    for key in list(normalized.keys()):
        if isinstance(normalized.get(key), str):
            normalized[key] = normalized[key].strip()

    job_id = clean_str(normalized.get("job_id"))
    if not job_id:
        raise RuntimeError("Job payload missing required key: job_id")

    timeframe = clean_str(normalized.get("timeframe"))
    strategy_family = clean_str(normalized.get("strategy_family"))
    entry_logic = clean_str(normalized.get("entry_logic"))
    micro_exit = clean_str(normalized.get("micro_exit"))
    regime_filter = clean_str(normalized.get("regime_filter"))
    side_policy = clean_str(normalized.get("side_policy"))
    volatility_filter = clean_str(normalized.get("volatility_filter"))
    trend_strength_filter = clean_str(normalized.get("trend_strength_filter"))

    missing = []
    if not timeframe:
        missing.append("timeframe")
    if not strategy_family:
        missing.append("strategy_family")
    if not entry_logic:
        missing.append("entry_logic")
    if not micro_exit:
        missing.append("micro_exit")
    if not regime_filter:
        missing.append("regime_filter")
    if clean_str(normalized.get("cooldown_bars")) == "":
        missing.append("cooldown_bars")
    if not side_policy:
        missing.append("side_policy")
    if not volatility_filter:
        missing.append("volatility_filter")
    if not trend_strength_filter:
        missing.append("trend_strength_filter")
    if missing:
        raise RuntimeError(f"Job payload job_id={job_id} missing required keys: {sorted(missing)}")

    feature_cache_path = clean_str(normalized.get("feature_cache_path"))
    ohlc_csv = clean_str(normalized.get("ohlc_csv"))
    if not (feature_cache_path or ohlc_csv):
        raise RuntimeError(
            f"Job payload job_id={job_id} missing data reference: require one of ['feature_cache_path', 'ohlc_csv']"
        )

    derived_result_key = (
        clean_str(normalized.get("result_key"))
        or clean_str(normalized.get("parameter_fingerprint"))
        or job_id
    )
    normalized["job_id"] = job_id
    normalized["timeframe"] = timeframe
    normalized["strategy_family"] = strategy_family
    normalized["entry_logic"] = entry_logic
    normalized["micro_exit"] = micro_exit
    normalized["regime_filter"] = regime_filter
    normalized["side_policy"] = side_policy
    normalized["volatility_filter"] = volatility_filter
    normalized["trend_strength_filter"] = trend_strength_filter
    normalized["result_key"] = derived_result_key
    normalized["feature_cache_path"] = feature_cache_path
    normalized["htf_feature_cache_path"] = clean_str(normalized.get("htf_feature_cache_path"))
    normalized["htf_context_timeframe"] = clean_str(normalized.get("htf_context_timeframe"))
    normalized["parameter_fingerprint"] = clean_str(normalized.get("parameter_fingerprint"))
    normalized["one_axis_group_key"] = clean_str(normalized.get("one_axis_group_key"))
    normalized["rank_priority"] = clean_str(normalized.get("rank_priority"))
    normalized["status"] = clean_str(normalized.get("status"))
    normalized["ohlc_csv"] = ohlc_csv
    if clean_str(normalized.get("symbol")) == "":
        normalized["symbol"] = "XAUUSD"
    normalized["cooldown_bars"] = int(float(normalized.get("cooldown_bars")))
    return normalized


# --------------------------------------------------------------------------------------------------
# INDICATORS
# --------------------------------------------------------------------------------------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = pd.Series(tr, index=df.index)

    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = pd.Series(tr, index=df.index)

    atr_smoothed = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_dm_smoothed = pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()
    minus_dm_smoothed = pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()

    plus_di = 100.0 * plus_dm_smoothed / atr_smoothed.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm_smoothed / atr_smoothed.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_series = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return adx_series.fillna(0.0)


def rolling_swing_high(series: pd.Series, lookback: int = 5) -> pd.Series:
    return series.shift(1).rolling(lookback).max()


def rolling_swing_low(series: pd.Series, lookback: int = 5) -> pd.Series:
    return series.shift(1).rolling(lookback).min()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ema_9"] = ema(out["close"], 9)
    out["ema_20"] = ema(out["close"], 20)
    out["ema_50"] = ema(out["close"], 50)
    out["ema_200"] = ema(out["close"], 200)

    out["atr_14"] = atr(out, 14)
    out["atr_pct"] = 100.0 * out["atr_14"] / out["close"].replace(0.0, np.nan)
    out["adx_14"] = adx(out, 14)
    out["rsi_14"] = rsi(out["close"], 14)

    out["swing_high_5"] = rolling_swing_high(out["high"], 5)
    out["swing_low_5"] = rolling_swing_low(out["low"], 5)
    out["swing_high_10"] = rolling_swing_high(out["high"], 10)
    out["swing_low_10"] = rolling_swing_low(out["low"], 10)

    out["return_1"] = out["close"].pct_change().fillna(0.0)
    out["range"] = out["high"] - out["low"]
    out["body"] = (out["close"] - out["open"]).abs()
    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]

    out["bull_stack"] = (
        (out["ema_9"] > out["ema_20"])
        & (out["ema_20"] > out["ema_50"])
        & (out["ema_50"] > out["ema_200"])
    )
    out["bear_stack"] = (
        (out["ema_9"] < out["ema_20"])
        & (out["ema_20"] < out["ema_50"])
        & (out["ema_50"] < out["ema_200"])
    )

    atr_q1 = out["atr_pct"].rolling(200).quantile(0.33)
    atr_q2 = out["atr_pct"].rolling(200).quantile(0.66)

    out["vol_bucket"] = np.where(
        out["atr_pct"] >= atr_q2,
        "HIGH_VOL",
        np.where(out["atr_pct"] >= atr_q1, "MID_VOL", "LOW_VOL"),
    )

    out["trend_bucket"] = np.where(
        out["adx_14"] >= 25,
        "STRONG_TREND",
        np.where(out["adx_14"] >= 20, "MID_TREND", "WEAK_TREND"),
    )

    out["price_location_bucket"] = np.where(
        out["close"] > out["ema_50"],
        "ABOVE_EMA_STACK",
        np.where(out["close"] < out["ema_50"], "BELOW_EMA_STACK", "NEAR_EMA_STACK"),
    )

    out["bar_index"] = np.arange(len(out))

    return out


# --------------------------------------------------------------------------------------------------
# REGIME / FILTERS
# --------------------------------------------------------------------------------------------------
def regime_allows_long(row: pd.Series, regime_filter: str) -> bool:
    if regime_filter == "trend_only":
        return bool(row["bull_stack"] and row["adx_14"] >= 20)
    if regime_filter == "trend_or_neutral":
        return bool((row["bull_stack"] and row["adx_14"] >= 18) or (row["close"] > row["ema_50"]))
    if regime_filter == "volatility_gated":
        return bool(row["bull_stack"] and row["vol_bucket"] in {"MID_VOL", "HIGH_VOL"})
    if regime_filter == "always_on":
        return True
    return True


def regime_allows_short(row: pd.Series, regime_filter: str) -> bool:
    if regime_filter == "trend_only":
        return bool(row["bear_stack"] and row["adx_14"] >= 20)
    if regime_filter == "trend_or_neutral":
        return bool((row["bear_stack"] and row["adx_14"] >= 18) or (row["close"] < row["ema_50"]))
    if regime_filter == "volatility_gated":
        return bool(row["bear_stack"] and row["vol_bucket"] in {"MID_VOL", "HIGH_VOL"})
    if regime_filter == "always_on":
        return True
    return True


def volatility_filter_allows(row: pd.Series, variant: str) -> bool:
    if variant == "none":
        return True
    if variant == "atr_mid_high_only":
        return row["vol_bucket"] in {"MID_VOL", "HIGH_VOL"}
    if variant == "atr_high_only":
        return row["vol_bucket"] == "HIGH_VOL"
    return True


def trend_strength_filter_allows(row: pd.Series, variant: str) -> bool:
    if variant == "none":
        return True
    if variant == "adx20_plus":
        return row["adx_14"] >= 20
    if variant == "adx25_plus":
        return row["adx_14"] >= 25
    return True


# --------------------------------------------------------------------------------------------------
# ENTRY LOGIC
# --------------------------------------------------------------------------------------------------
def detect_long_signal(row: pd.Series, prev_row: pd.Series, strategy_family: str, entry_logic: str) -> Tuple[bool, str]:
    c = row["close"]
    o = row["open"]
    h = row["high"]
    l = row["low"]

    if entry_logic == "bos_choch_atr_adx_ema":
        cond = (
            c > row["swing_high_5"]
            and row["bull_stack"]
            and row["adx_14"] >= 20
            and row["atr_pct"] > 0
        )
        return bool(cond), "long_bos_choch_atr_adx_ema"

    if entry_logic == "bos_choch_ema_reclaim":
        cond = (
            prev_row["close"] <= prev_row["ema_20"]
            and c > row["ema_20"]
            and row["bull_stack"]
            and c > row["swing_high_5"]
        )
        return bool(cond), "long_bos_choch_ema_reclaim"

    if entry_logic == "pullback_to_ema_stack":
        cond = (
            row["bull_stack"]
            and l <= row["ema_20"]
            and c > row["ema_20"]
            and c > o
        )
        return bool(cond), "long_pullback_to_ema_stack"

    if entry_logic == "liquidity_sweep_reclaim":
        cond = (
            l < row["swing_low_5"]
            and c > row["ema_20"]
            and c > o
            and row["lower_wick"] > row["body"] * 0.7
        )
        return bool(cond), "long_liquidity_sweep_reclaim"

    if entry_logic == "breakout_retest_impulse":
        cond = (
            prev_row["close"] > prev_row["swing_high_5"]
            and l <= row["swing_high_5"]
            and c > row["swing_high_5"]
            and c > o
            and row["range"] > row["atr_14"] * 0.8
        )
        return bool(cond), "long_breakout_retest_impulse"

    return False, ""


def detect_short_signal(row: pd.Series, prev_row: pd.Series, strategy_family: str, entry_logic: str) -> Tuple[bool, str]:
    c = row["close"]
    o = row["open"]
    h = row["high"]
    l = row["low"]

    if entry_logic == "bos_choch_atr_adx_ema":
        cond = (
            c < row["swing_low_5"]
            and row["bear_stack"]
            and row["adx_14"] >= 20
            and row["atr_pct"] > 0
        )
        return bool(cond), "short_bos_choch_atr_adx_ema"

    if entry_logic == "bos_choch_ema_reclaim":
        cond = (
            prev_row["close"] >= prev_row["ema_20"]
            and c < row["ema_20"]
            and row["bear_stack"]
            and c < row["swing_low_5"]
        )
        return bool(cond), "short_bos_choch_ema_reclaim"

    if entry_logic == "pullback_to_ema_stack":
        cond = (
            row["bear_stack"]
            and h >= row["ema_20"]
            and c < row["ema_20"]
            and c < o
        )
        return bool(cond), "short_pullback_to_ema_stack"

    if entry_logic == "liquidity_sweep_reclaim":
        cond = (
            h > row["swing_high_5"]
            and c < row["ema_20"]
            and c < o
            and row["upper_wick"] > row["body"] * 0.7
        )
        return bool(cond), "short_liquidity_sweep_reclaim"

    if entry_logic == "breakout_retest_impulse":
        cond = (
            prev_row["close"] < prev_row["swing_low_5"]
            and h >= row["swing_low_5"]
            and c < row["swing_low_5"]
            and c < o
            and row["range"] > row["atr_14"] * 0.8
        )
        return bool(cond), "short_breakout_retest_impulse"

    return False, ""


def strategy_family_allows(strategy_family: str, row: pd.Series, side: str) -> bool:
    if strategy_family == "pullback_deep":
        if side == "LONG":
            return bool(row["close"] >= row["ema_20"] and row["low"] <= row["ema_50"])
        return bool(row["close"] <= row["ema_20"] and row["high"] >= row["ema_50"])

    if strategy_family == "pullback_shallow":
        if side == "LONG":
            return bool(row["close"] >= row["ema_20"] and row["low"] <= row["ema_20"])
        return bool(row["close"] <= row["ema_20"] and row["high"] >= row["ema_20"])

    if strategy_family == "trend_continuation":
        if side == "LONG":
            return bool(row["bull_stack"] and row["close"] > row["swing_high_5"])
        return bool(row["bear_stack"] and row["close"] < row["swing_low_5"])

    if strategy_family == "range_reversal":
        if side == "LONG":
            return bool(row["rsi_14"] < 40 and row["close"] > row["open"])
        return bool(row["rsi_14"] > 60 and row["close"] < row["open"])

    if strategy_family == "breakout_expansion":
        if side == "LONG":
            return bool(row["close"] > row["swing_high_10"] and row["range"] > row["atr_14"])
        return bool(row["close"] < row["swing_low_10"] and row["range"] > row["atr_14"])

    return True


# --------------------------------------------------------------------------------------------------
# STOP / TARGET / MICRO EXIT
# --------------------------------------------------------------------------------------------------
def base_stop_and_target(row: pd.Series, side: str, strategy_family: str) -> Tuple[float, float]:
    atr_v = max(float(row["atr_14"]), 1e-8)
    c = float(row["close"])

    if strategy_family == "pullback_deep":
        sl_mult = 1.40
        tp_mult = 2.80
    elif strategy_family == "pullback_shallow":
        sl_mult = 1.00
        tp_mult = 2.10
    elif strategy_family == "trend_continuation":
        sl_mult = 1.20
        tp_mult = 2.60
    elif strategy_family == "range_reversal":
        sl_mult = 1.00
        tp_mult = 1.80
    elif strategy_family == "breakout_expansion":
        sl_mult = 1.50
        tp_mult = 3.20
    else:
        sl_mult = 1.20
        tp_mult = 2.40

    if side == "LONG":
        stop_loss = c - atr_v * sl_mult
        take_profit = c + atr_v * tp_mult
    else:
        stop_loss = c + atr_v * sl_mult
        take_profit = c - atr_v * tp_mult

    return float(stop_loss), float(take_profit)


def should_exit_micro(
    position: Position,
    row: pd.Series,
    prev_row: pd.Series,
    micro_exit: str,
) -> Tuple[bool, str]:
    c = float(row["close"])

    if micro_exit == "fast_invalidation":
        if position.side == "LONG":
            cond = c < row["ema_20"] or c < prev_row["low"]
            return bool(cond), "micro_exit_fast_invalidation"
        cond = c > row["ema_20"] or c > prev_row["high"]
        return bool(cond), "micro_exit_fast_invalidation"

    if micro_exit == "momentum_fade":
        if position.side == "LONG":
            cond = (c < prev_row["close"]) and (row["rsi_14"] < prev_row["rsi_14"])
            return bool(cond), "micro_exit_momentum_fade"
        cond = (c > prev_row["close"]) and (row["rsi_14"] > prev_row["rsi_14"])
        return bool(cond), "micro_exit_momentum_fade"

    if micro_exit == "structure_trail":
        if position.side == "LONG":
            cond = c < row["swing_low_5"]
            return bool(cond), "micro_exit_structure_trail"
        cond = c > row["swing_high_5"]
        return bool(cond), "micro_exit_structure_trail"

    if micro_exit == "time_stop_compact":
        cond = position.bars_held >= 8
        return bool(cond), "micro_exit_time_stop_compact"

    return False, ""


# --------------------------------------------------------------------------------------------------
# BACKTEST ENGINE
# --------------------------------------------------------------------------------------------------
def side_allowed(side_policy: str, side: str) -> bool:
    if side_policy == "BIDIRECTIONAL":
        return True
    if side_policy == "LONG_ONLY" and side == "LONG":
        return True
    if side_policy == "SHORT_ONLY" and side == "SHORT":
        return True
    return False


def apply_cost_model(entry_price: float, exit_price: float, side: str, atr_value: float) -> Tuple[float, float]:
    # Simple research-grade cost approximation:
    # - spread/slippage combined as a fraction of ATR
    cost = max(atr_value * 0.05, entry_price * 0.00005)

    if side == "LONG":
        adj_entry = entry_price + cost * 0.5
        adj_exit = exit_price - cost * 0.5
    else:
        adj_entry = entry_price - cost * 0.5
        adj_exit = exit_price + cost * 0.5

    return float(adj_entry), float(adj_exit)


def close_position(
    position: Position,
    row: pd.Series,
    exit_price: float,
    exit_reason: str,
    job: Dict[str, Any],
) -> ClosedTrade:
    atr_v = max(float(row["atr_14"]), 1e-8)

    adj_entry, adj_exit = apply_cost_model(position.entry_price, exit_price, position.side, atr_v)

    if position.side == "LONG":
        pnl = adj_exit - adj_entry
        risk_unit = max(adj_entry - position.stop_loss, 1e-8)
        pnl_r = pnl / risk_unit
    else:
        pnl = adj_entry - adj_exit
        risk_unit = max(position.stop_loss - adj_entry, 1e-8)
        pnl_r = pnl / risk_unit

    return ClosedTrade(
        side=position.side,
        entry_index=position.entry_index,
        exit_index=int(row["bar_index"]),
        entry_time=position.entry_time,
        exit_time=str(row["time"]),
        entry_price=float(adj_entry),
        exit_price=float(adj_exit),
        stop_loss=float(position.stop_loss),
        take_profit=float(position.take_profit),
        pnl=float(pnl),
        pnl_r=float(pnl_r),
        bars_held=int(position.bars_held),
        mfe=float(position.mfe),
        mae=float(position.mae),
        exit_reason=exit_reason,
        entry_reason=position.entry_reason,
        timeframe=str(job["timeframe"]),
        strategy_family=str(job["strategy_family"]),
        entry_logic=str(job["entry_logic"]),
        micro_exit=str(job["micro_exit"]),
        regime_filter=str(job["regime_filter"]),
        cooldown_bars=int(job["cooldown_bars"]),
        side_policy=str(job["side_policy"]),
        volatility_filter=str(job["volatility_filter"]),
        trend_strength_filter=str(job["trend_strength_filter"]),
    )


def run_backtest(df: pd.DataFrame, job: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    trades: List[ClosedTrade] = []
    equity_records: List[Dict[str, Any]] = []

    position: Optional[Position] = None
    cooldown_remaining = 0
    equity = 0.0
    max_equity = 0.0

    diagnostics = {
        "signals_total": 0,
        "signals_long": 0,
        "signals_short": 0,
        "signals_blocked_side_policy": 0,
        "signals_blocked_regime": 0,
        "signals_blocked_volatility": 0,
        "signals_blocked_trend_strength": 0,
        "signals_blocked_cooldown": 0,
        "signals_blocked_existing_position": 0,
        "stop_loss_count": 0,
        "take_profit_count": 0,
        "micro_exit_count": 0,
        "force_close_end_count": 0,
        "cooldown_skipped_count": 0,
    }

    start_idx = 220  # enough buffer for EMA200 + rolling features

    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        if cooldown_remaining > 0 and position is None:
            cooldown_remaining -= 1

        if position is not None:
            position.bars_held += 1

            if position.side == "LONG":
                current_mfe = float(row["high"]) - position.entry_price
                current_mae = position.entry_price - float(row["low"])
            else:
                current_mfe = position.entry_price - float(row["low"])
                current_mae = float(row["high"]) - position.entry_price

            position.mfe = max(position.mfe, current_mfe)
            position.mae = max(position.mae, current_mae)

            hit_sl = False
            hit_tp = False
            exit_price = None
            exit_reason = ""

            if position.side == "LONG":
                if float(row["low"]) <= position.stop_loss:
                    hit_sl = True
                    exit_price = position.stop_loss
                    exit_reason = "stop_loss"
                elif float(row["high"]) >= position.take_profit:
                    hit_tp = True
                    exit_price = position.take_profit
                    exit_reason = "take_profit"
            else:
                if float(row["high"]) >= position.stop_loss:
                    hit_sl = True
                    exit_price = position.stop_loss
                    exit_reason = "stop_loss"
                elif float(row["low"]) <= position.take_profit:
                    hit_tp = True
                    exit_price = position.take_profit
                    exit_reason = "take_profit"

            if hit_sl or hit_tp:
                trade = close_position(position, row, float(exit_price), exit_reason, job)
                trades.append(trade)
                equity += trade.pnl
                max_equity = max(max_equity, equity)

                if exit_reason == "stop_loss":
                    diagnostics["stop_loss_count"] += 1
                elif exit_reason == "take_profit":
                    diagnostics["take_profit_count"] += 1

                cooldown_remaining = int(job["cooldown_bars"])
                position = None
            else:
                should_exit, micro_reason = should_exit_micro(
                    position=position,
                    row=row,
                    prev_row=prev_row,
                    micro_exit=str(job["micro_exit"]),
                )
                if should_exit:
                    trade = close_position(position, row, float(row["close"]), micro_reason, job)
                    trades.append(trade)
                    equity += trade.pnl
                    max_equity = max(max_equity, equity)

                    diagnostics["micro_exit_count"] += 1
                    cooldown_remaining = int(job["cooldown_bars"])
                    position = None

        if position is None:
            long_signal_raw, long_reason = detect_long_signal(
                row=row,
                prev_row=prev_row,
                strategy_family=str(job["strategy_family"]),
                entry_logic=str(job["entry_logic"]),
            )
            short_signal_raw, short_reason = detect_short_signal(
                row=row,
                prev_row=prev_row,
                strategy_family=str(job["strategy_family"]),
                entry_logic=str(job["entry_logic"]),
            )

            long_allowed = False
            short_allowed = False

            if long_signal_raw:
                diagnostics["signals_total"] += 1
                diagnostics["signals_long"] += 1
                long_allowed = True

                if not side_allowed(str(job["side_policy"]), "LONG"):
                    long_allowed = False
                    diagnostics["signals_blocked_side_policy"] += 1

                if long_allowed and not regime_allows_long(row, str(job["regime_filter"])):
                    long_allowed = False
                    diagnostics["signals_blocked_regime"] += 1

                if long_allowed and not volatility_filter_allows(row, str(job["volatility_filter"])):
                    long_allowed = False
                    diagnostics["signals_blocked_volatility"] += 1

                if long_allowed and not trend_strength_filter_allows(row, str(job["trend_strength_filter"])):
                    long_allowed = False
                    diagnostics["signals_blocked_trend_strength"] += 1

                if long_allowed and not strategy_family_allows(str(job["strategy_family"]), row, "LONG"):
                    long_allowed = False

                if long_allowed and cooldown_remaining > 0:
                    long_allowed = False
                    diagnostics["signals_blocked_cooldown"] += 1
                    diagnostics["cooldown_skipped_count"] += 1

                signals.append(
                    {
                        "time": str(row["time"]),
                        "bar_index": int(row["bar_index"]),
                        "side": "LONG",
                        "raw_signal": bool(long_signal_raw),
                        "final_signal": bool(long_allowed),
                        "entry_reason": long_reason,
                        "close": float(row["close"]),
                        "adx_14": float(row["adx_14"]),
                        "atr_pct": float(row["atr_pct"]),
                        "vol_bucket": str(row["vol_bucket"]),
                        "trend_bucket": str(row["trend_bucket"]),
                        "price_location_bucket": str(row["price_location_bucket"]),
                    }
                )

            if short_signal_raw:
                diagnostics["signals_total"] += 1
                diagnostics["signals_short"] += 1
                short_allowed = True

                if not side_allowed(str(job["side_policy"]), "SHORT"):
                    short_allowed = False
                    diagnostics["signals_blocked_side_policy"] += 1

                if short_allowed and not regime_allows_short(row, str(job["regime_filter"])):
                    short_allowed = False
                    diagnostics["signals_blocked_regime"] += 1

                if short_allowed and not volatility_filter_allows(row, str(job["volatility_filter"])):
                    short_allowed = False
                    diagnostics["signals_blocked_volatility"] += 1

                if short_allowed and not trend_strength_filter_allows(row, str(job["trend_strength_filter"])):
                    short_allowed = False
                    diagnostics["signals_blocked_trend_strength"] += 1

                if short_allowed and not strategy_family_allows(str(job["strategy_family"]), row, "SHORT"):
                    short_allowed = False

                if short_allowed and cooldown_remaining > 0:
                    short_allowed = False
                    diagnostics["signals_blocked_cooldown"] += 1
                    diagnostics["cooldown_skipped_count"] += 1

                signals.append(
                    {
                        "time": str(row["time"]),
                        "bar_index": int(row["bar_index"]),
                        "side": "SHORT",
                        "raw_signal": bool(short_signal_raw),
                        "final_signal": bool(short_allowed),
                        "entry_reason": short_reason,
                        "close": float(row["close"]),
                        "adx_14": float(row["adx_14"]),
                        "atr_pct": float(row["atr_pct"]),
                        "vol_bucket": str(row["vol_bucket"]),
                        "trend_bucket": str(row["trend_bucket"]),
                        "price_location_bucket": str(row["price_location_bucket"]),
                    }
                )

            if long_allowed:
                stop_loss, take_profit = base_stop_and_target(row, "LONG", str(job["strategy_family"]))
                position = Position(
                    side="LONG",
                    entry_index=int(row["bar_index"]),
                    entry_time=str(row["time"]),
                    entry_price=float(row["close"]),
                    stop_loss=float(stop_loss),
                    take_profit=float(take_profit),
                    entry_reason=long_reason,
                )
            elif short_allowed:
                stop_loss, take_profit = base_stop_and_target(row, "SHORT", str(job["strategy_family"]))
                position = Position(
                    side="SHORT",
                    entry_index=int(row["bar_index"]),
                    entry_time=str(row["time"]),
                    entry_price=float(row["close"]),
                    stop_loss=float(stop_loss),
                    take_profit=float(take_profit),
                    entry_reason=short_reason,
                )

        equity_records.append(
            {
                "time": str(row["time"]),
                "bar_index": int(row["bar_index"]),
                "equity": float(equity),
                "position_side": "" if position is None else position.side,
                "close": float(row["close"]),
            }
        )

    if position is not None:
        last_row = df.iloc[-1]
        trade = close_position(position, last_row, float(last_row["close"]), "force_close_end", job)
        trades.append(trade)
        diagnostics["force_close_end_count"] += 1
        equity += trade.pnl

    signals_df = pd.DataFrame(signals)
    trades_df = pd.DataFrame([asdict(x) for x in trades])
    equity_df = pd.DataFrame(equity_records)

    return signals_df, trades_df, equity_df, diagnostics


# --------------------------------------------------------------------------------------------------
# METRICS
# --------------------------------------------------------------------------------------------------
def compute_max_consecutive_losses(trades_df: pd.DataFrame) -> int:
    if trades_df.empty:
        return 0
    streak = 0
    max_streak = 0
    for pnl in trades_df["pnl"].tolist():
        if pnl < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return int(max_streak)


def compute_equity_drawdown(trades_df: pd.DataFrame) -> Tuple[float, float]:
    if trades_df.empty:
        return 0.0, 0.0
    equity = trades_df["pnl"].cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_drawdown = float(drawdown.min())
    max_drawdown_abs = abs(max_drawdown)
    return max_drawdown, max_drawdown_abs


def compute_session_from_time(time_text: str) -> str:
    hour = pd.to_datetime(time_text).hour
    if 0 <= hour < 7:
        return "ASIA"
    if 7 <= hour < 13:
        return "LONDON"
    if 13 <= hour < 21:
        return "NEWYORK"
    return "LATE_US"


def summarize_counts(df: pd.DataFrame, col: str) -> Dict[str, int]:
    if df.empty or col not in df.columns:
        return {}
    series = df[col].astype(str)
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}


def build_metrics(
    *,
    df: pd.DataFrame,
    signals_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    diagnostics: Dict[str, Any],
    job: Dict[str, Any],
) -> Dict[str, Any]:
    if trades_df.empty:
        metrics = {
            "version": VERSION,
            "generated_at_utc": utc_now_iso(),
            "job_id": str(job["job_id"]),
            "result_key": str(job["result_key"]),
            "trade_count": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "pnl_sum": 0.0,
            "avg_pnl": 0.0,
            "avg_pnl_r": 0.0,
            "payoff_ratio": 0.0,
            "profit_factor": 0.0,
            "max_consecutive_losses": 0,
            "max_drawdown": 0.0,
            "max_drawdown_abs": 0.0,
            "avg_bars_held": 0.0,
            "avg_mfe": 0.0,
            "avg_mae": 0.0,
            "signal_count": int(len(signals_df)),
            "final_signal_count": int(signals_df["final_signal"].sum()) if not signals_df.empty else 0,
            "score": 0.0,
            "diagnostics": diagnostics,
            "splits": {},
        }
        return metrics

    pnl = trades_df["pnl"]
    wins_df = trades_df[trades_df["pnl"] > 0]
    losses_df = trades_df[trades_df["pnl"] < 0]

    wins = int(len(wins_df))
    losses = int(len(losses_df))
    trade_count = int(len(trades_df))
    pnl_sum = float(pnl.sum())
    avg_pnl = float(pnl.mean())
    avg_pnl_r = float(trades_df["pnl_r"].mean())

    avg_win = float(wins_df["pnl"].mean()) if wins > 0 else 0.0
    avg_loss_abs = abs(float(losses_df["pnl"].mean())) if losses > 0 else 0.0
    payoff_ratio = stable_div(avg_win, avg_loss_abs, 0.0)

    gross_profit = float(wins_df["pnl"].sum()) if wins > 0 else 0.0
    gross_loss_abs = abs(float(losses_df["pnl"].sum())) if losses > 0 else 0.0
    profit_factor = stable_div(gross_profit, gross_loss_abs, 0.0)

    win_rate_pct = 100.0 * wins / trade_count if trade_count > 0 else 0.0
    max_consecutive_losses = compute_max_consecutive_losses(trades_df)
    max_drawdown, max_drawdown_abs = compute_equity_drawdown(trades_df)

    avg_bars_held = float(trades_df["bars_held"].mean())
    avg_mfe = float(trades_df["mfe"].mean())
    avg_mae = float(trades_df["mae"].mean())

    score = (
        pnl_sum
        + (payoff_ratio * 40.0)
        + (win_rate_pct * 1.5)
        - (max_consecutive_losses * 10.0)
        - (max_drawdown_abs * 0.8)
    )

    trades_df = trades_df.copy()
    trades_df["session"] = trades_df["entry_time"].map(compute_session_from_time)

    splits = {
        "exit_reason": summarize_counts(trades_df, "exit_reason"),
        "side": summarize_counts(trades_df, "side"),
        "session": summarize_counts(trades_df, "session"),
    }

    if not signals_df.empty:
        final_signals = signals_df[signals_df["final_signal"] == True].copy()
        splits["signal_side"] = summarize_counts(final_signals, "side")
        splits["signal_vol_bucket"] = summarize_counts(final_signals, "vol_bucket")
        splits["signal_trend_bucket"] = summarize_counts(final_signals, "trend_bucket")
        splits["signal_price_location_bucket"] = summarize_counts(final_signals, "price_location_bucket")
    else:
        final_signals = pd.DataFrame()

    metrics = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "job_id": str(job["job_id"]),
        "result_key": str(job["result_key"]),
        "trade_count": trade_count,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate_pct, 4),
        "pnl_sum": round(pnl_sum, 6),
        "avg_pnl": round(avg_pnl, 6),
        "avg_pnl_r": round(avg_pnl_r, 6),
        "payoff_ratio": round(payoff_ratio, 6),
        "profit_factor": round(profit_factor, 6),
        "max_consecutive_losses": int(max_consecutive_losses),
        "max_drawdown": round(max_drawdown, 6),
        "max_drawdown_abs": round(max_drawdown_abs, 6),
        "avg_bars_held": round(avg_bars_held, 4),
        "avg_mfe": round(avg_mfe, 6),
        "avg_mae": round(avg_mae, 6),
        "signal_count": int(len(signals_df)),
        "final_signal_count": int(final_signals.shape[0]),
        "score": round(score, 6),
        "diagnostics": diagnostics,
        "splits": splits,
    }
    return metrics


# --------------------------------------------------------------------------------------------------
# BUILD SUMMARY
# --------------------------------------------------------------------------------------------------
def build_summary_payload(
    job: Dict[str, Any],
    ohlc_path: Optional[Path],
    feature_cache_path: Optional[Path],
    htf_feature_cache_path: Optional[Path],
    input_mode: str,
    result_dir: Path,
    df: pd.DataFrame,
    signals_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "job_id": str(job["job_id"]),
        "result_key": str(job["result_key"]),
        "timeframe": str(job["timeframe"]),
        "symbol": str(job["symbol"]),
        "strategy_family": str(job["strategy_family"]),
        "entry_logic": str(job["entry_logic"]),
        "micro_exit": str(job["micro_exit"]),
        "regime_filter": str(job["regime_filter"]),
        "cooldown_bars": int(job["cooldown_bars"]),
        "side_policy": str(job["side_policy"]),
        "volatility_filter": str(job["volatility_filter"]),
        "trend_strength_filter": str(job["trend_strength_filter"]),
        "input_mode": str(input_mode),
        "feature_cache_path": str(feature_cache_path) if feature_cache_path is not None else "",
        "htf_feature_cache_path": str(htf_feature_cache_path) if htf_feature_cache_path is not None else "",
        "ohlc_csv": str(ohlc_path) if ohlc_path is not None else "",
        "result_dir": str(result_dir),
        "input_bar_count": int(len(df)),
        "signal_count": int(len(signals_df)),
        "trade_count": int(len(trades_df)),
        "pnl_sum": metrics["pnl_sum"],
        "payoff_ratio": metrics["payoff_ratio"],
        "win_rate_pct": metrics["win_rate_pct"],
        "max_consecutive_losses": metrics["max_consecutive_losses"],
        "score": metrics["score"],
    }


# --------------------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one intelligent research backtest job.")
    parser.add_argument("--job", required=True, help="Path to one job spec JSON")
    parser.add_argument("--result-root", required=True, help="Root directory for per-job results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    job_path = Path(args.job)
    result_root = Path(args.result_root)

    if not job_path.exists():
        raise RuntimeError(f"Job file not found: {job_path}")

    job = normalize_job_payload(read_json(job_path))
    result_key = str(job["result_key"])
    result_dir = result_root / result_key

    feature_cache_path = Path(job["feature_cache_path"]) if clean_str(job.get("feature_cache_path")) else None
    htf_feature_cache_path = Path(job["htf_feature_cache_path"]) if clean_str(job.get("htf_feature_cache_path")) else None
    ohlc_path = Path(job["ohlc_csv"]) if clean_str(job.get("ohlc_csv")) else None

    if feature_cache_path is not None:
        input_mode = "feature_cache"
    else:
        input_mode = "ohlc_csv"

    ensure_dir(result_root)
    ensure_dir(result_dir)

    used_job_path = result_dir / "used_job.json"
    build_summary_path = result_dir / "build_summary.json"
    metrics_path = result_dir / "metrics.json"
    diagnostics_path = result_dir / "diagnostics.json"
    signals_path = result_dir / "signals.csv"
    trades_path = result_dir / "trades.csv"
    equity_curve_path = result_dir / "equity_curve.csv"

    write_json(used_job_path, job)

    if input_mode == "feature_cache":
        df = load_ohlc_from_feature_cache(feature_cache_path)
    else:
        df = load_ohlc_csv(ohlc_path)
    df = compute_features(df)

    signals_df, trades_df, equity_df, diagnostics = run_backtest(df, job)
    metrics = build_metrics(
        df=df,
        signals_df=signals_df,
        trades_df=trades_df,
        equity_df=equity_df,
        diagnostics=diagnostics,
        job=job,
    )
    build_summary = build_summary_payload(
        job=job,
        ohlc_path=ohlc_path,
        feature_cache_path=feature_cache_path,
        htf_feature_cache_path=htf_feature_cache_path,
        input_mode=input_mode,
        result_dir=result_dir,
        df=df,
        signals_df=signals_df,
        trades_df=trades_df,
        metrics=metrics,
    )

    if signals_df.empty:
        signals_df = pd.DataFrame(
            columns=[
                "time",
                "bar_index",
                "side",
                "raw_signal",
                "final_signal",
                "entry_reason",
                "close",
                "adx_14",
                "atr_pct",
                "vol_bucket",
                "trend_bucket",
                "price_location_bucket",
            ]
        )

    if trades_df.empty:
        trades_df = pd.DataFrame(
            columns=[
                "side",
                "entry_index",
                "exit_index",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "stop_loss",
                "take_profit",
                "pnl",
                "pnl_r",
                "bars_held",
                "mfe",
                "mae",
                "exit_reason",
                "entry_reason",
                "timeframe",
                "strategy_family",
                "entry_logic",
                "micro_exit",
                "regime_filter",
                "cooldown_bars",
                "side_policy",
                "volatility_filter",
                "trend_strength_filter",
            ]
        )

    signals_df.to_csv(signals_path, index=False, encoding="utf-8")
    trades_df.to_csv(trades_path, index=False, encoding="utf-8")
    equity_df.to_csv(equity_curve_path, index=False, encoding="utf-8")

    write_json(metrics_path, metrics)
    write_json(diagnostics_path, diagnostics)
    write_json(build_summary_path, build_summary)

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] job_id={job['job_id']}")
    print(f"[DONE] result_key={job['result_key']}")
    print(f"[DONE] result_dir={result_dir}")
    print(f"[DONE] input_bar_count={len(df)}")
    print(f"[DONE] signal_count={len(signals_df)}")
    print(f"[DONE] trade_count={len(trades_df)}")
    print(f"[DONE] pnl_sum={metrics['pnl_sum']}")
    print(f"[DONE] payoff_ratio={metrics['payoff_ratio']}")
    print(f"[DONE] win_rate_pct={metrics['win_rate_pct']}")
    print(f"[DONE] max_consecutive_losses={metrics['max_consecutive_losses']}")
    print(f"[DONE] score={metrics['score']}")
    print("=" * 120)


if __name__ == "__main__":
    main()

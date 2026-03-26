#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Code Name : run_vectorbt_micro_exit_coverage_batched_v1_0_7.py
Version   : v1.0.7
Path      : C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_micro_exit_coverage_batched_v1_0_7.py
Run       : python C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_micro_exit_coverage_batched_v1_0_7.py --manifest C:\Data\Bot\central_backtest_results\research_coverage_master_v1_0_0\research_coverage_master_manifest.csv --data-root C:\Data\Bot\central_market_data\parquet --feature-root C:\Data\Bot\central_feature_cache --outdir C:\Data\Bot\central_backtest_results\coverage_master_runs_v1_0_0\micro_exit_expansion_batched_v1_0_7\run_v107_full --phase micro_exit_expansion --portfolio-chunk-size 16 --preflight-flush-size 5000 --progress-every-groups 1 --continue-on-error

Purpose
- Coverage runner แบบ component-schema สำหรับ micro-exit expansion
- ฐานจากแนว execution ของ v1.0.6 และเพิ่ม support strategy_family=swing_pullback เพียง family เดียวในรอบนี้
- ยังไม่เดา semantics ที่คลุมเครือของ logic_strictness=relaxed และ swing_variant=medium

Design
1) preflight ก่อน execute
2) แยก unsupported / skipped / done ชัดเจน
3) โหลดข้อมูลทีละ timeframe
4) group execution โดย exclude micro_exit_variant เพื่อให้ 1 group มีหลาย micro exits
5) output โครงเดียวกับเวอร์ชันก่อนหน้า: coverage_results_all.csv, preflight_results.csv, status_summary.csv,
   summary_by_group.csv, live_progress.json, bootstrap.log

หมายเหตุ
- ไฟล์นี้เป็น full file ใหม่ทั้งชุดตามข้อกำหนดของผู้ใช้
- ตั้งใจให้รันได้จริงแบบ self-contained แม้ไม่ได้พึ่ง source เต็มของ v1.0.6 ในบทสนทนานี้
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

VERSION = "v1.0.7"
PHASE_NAME = "micro_exit_expansion"
DEFAULT_SYMBOL = "XAUUSD"

SUPPORTED_STRATEGY_FAMILIES = {
    "adx_ema_trend_continuation",
    "ema_trend_continuation",
    "bos_continuation",
    "swing_pullback",  # added in v1.0.7
}

SUPPORTED_ENTRY_VARIANTS = {"confirm_entry"}
SUPPORTED_PULLBACK_ZONE_VARIANTS = {"narrow", "medium", "wide"}
SUPPORTED_SWING_VARIANTS = {"long", "short"}  # medium is still intentionally unsupported
SUPPORTED_LOGIC_STRICTNESS = {"loose", "light", "medium", "mid", "strict", "hard"}

SUPPORTED_MICRO_EXIT_VARIANTS = {
    # base / v1.0.4 family block
    "adx_fade_exit",
    "fast_invalidation",
    "momentum_fade",
    "no_micro_exit",
    "baseline_pending_exit",
    "break_even_ladder_exit",
    "atr_trailing_exit",
    "reverse_signal_exit",
    "atr_guard_exit",
    "volatility_crush_exit",
    "price_cross_slow_exit",
    "price_cross_fast_exit",
    "partial_tp_runner_exit",
    "time_decay_exit",
    "structure_fail_exit",
}

LOGIC_ALIAS = {
    "light": "loose",
    "mid": "medium",
    "hard": "strict",
}

TIMEFRAMES = ["D1", "H1", "H4", "M1", "M10", "M15", "M2", "M3", "M30", "M4", "M5", "M6"]

REQUIRED_PRICE_COLUMNS = ["time", "open", "high", "low", "close"]
REQUIRED_OPTIONAL_FEATURE_COLUMNS = ["time", "adx_14"]


@dataclass
class JobSpec:
    manifest_rank: str
    manifest_id: str
    timeframe: str
    strategy_family: str
    logic_strictness: str
    swing_variant: str
    pullback_zone_variant: str
    entry_variant: str
    micro_exit_variant: str
    management_variant: str
    regime_variant: str
    robustness_variant: str
    ema_fast: int
    ema_slow: int
    symbol: str
    batch_group: str
    priority_tier: str
    status: str
    rationale: str
    logic_key: str
    job_id: str


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BootstrapLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str) -> None:
        line = f"{utcnow_iso()} {msg}"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)


class CSVAppendWriter:
    def __init__(self, path: Path, fieldnames: List[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def write_rows(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self.path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerows(rows)


class ProgressTracker:
    def __init__(self, path: Path, base_payload: dict) -> None:
        self.path = path
        self.base_payload = dict(base_payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict) -> None:
        merged = dict(self.base_payload)
        merged.update(payload)
        merged["updated_at_utc"] = utcnow_iso()
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)


# -------------------------
# Utility
# -------------------------

def normalize_text(v: object) -> str:
    return str(v or "").strip()



def pick_first_nonempty(row: dict, keys: Iterable[str], default: str = "") -> str:
    for k in keys:
        if k in row:
            v = normalize_text(row.get(k, ""))
            if v:
                return v
    return default



def safe_int(value: object, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default



def logic_strictness_to_threshold(name: str) -> float:
    value = LOGIC_ALIAS.get(name, name)
    mapping = {
        "loose": 15.0,
        "medium": 20.0,
        "strict": 25.0,
    }
    return mapping[value]



def pullback_zone_to_pct(name: str) -> float:
    return {
        "narrow": 0.0015,
        "medium": 0.0030,
        "wide": 0.0050,
    }[name]



def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)



def read_price_parquet(data_root: Path, symbol: str, timeframe: str) -> pd.DataFrame:
    path = data_root / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"price parquet not found: {path}")
    df = pd.read_parquet(path, columns=REQUIRED_PRICE_COLUMNS)
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=False, errors="coerce")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df = df.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)
    return df



def read_feature_parquet(feature_root: Path, symbol: str, timeframe: str, n_rows_hint: int) -> pd.DataFrame:
    path = feature_root / f"{symbol}_{timeframe}_base_features.parquet"
    if not path.exists():
        # fallback empty features; execution can derive ADX internally
        return pd.DataFrame({"time": pd.Series([], dtype="datetime64[ns]"), "adx_14": pd.Series([], dtype="float32")})
    cols = REQUIRED_OPTIONAL_FEATURE_COLUMNS
    available = None
    try:
        available = list(pd.read_parquet(path, columns=["time"]).columns)
    except Exception:
        pass
    read_cols = cols if available is not None else None
    try:
        df = pd.read_parquet(path, columns=read_cols)
    except Exception:
        df = pd.read_parquet(path)
    df = df.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=False, errors="coerce")
    if "adx_14" in df.columns:
        df["adx_14"] = pd.to_numeric(df["adx_14"], errors="coerce").astype("float32")
    else:
        df["adx_14"] = np.nan
    return df[[c for c in ["time", "adx_14"] if c in df.columns]].dropna(subset=["time"]).sort_values("time").reset_index(drop=True)



def calculate_adx(price: pd.DataFrame, period: int = 14) -> pd.Series:
    high = price["high"]
    low = price["low"]
    close = price["close"]

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_components = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    )
    tr = tr_components.max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=price.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=price.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().astype("float32")
    return adx.fillna(0.0)



def calculate_atr(price: pd.DataFrame, period: int = 14) -> pd.Series:
    tr_components = pd.concat(
        [
            price["high"] - price["low"],
            (price["high"] - price["close"].shift(1)).abs(),
            (price["low"] - price["close"].shift(1)).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().astype("float32")
    return atr.fillna(method="bfill").fillna(0.0)



def load_timeframe_bundle(data_root: Path, feature_root: Path, symbol: str, timeframe: str) -> pd.DataFrame:
    price = read_price_parquet(data_root, symbol, timeframe)
    feat = read_feature_parquet(feature_root, symbol, timeframe, len(price))
    df = price.merge(feat, on="time", how="left")
    if "adx_14" not in df.columns or df["adx_14"].isna().all():
        df["adx_14"] = calculate_adx(df)
    else:
        df["adx_14"] = pd.to_numeric(df["adx_14"], errors="coerce").astype("float32")
        adx_calc = calculate_adx(df)
        df["adx_14"] = df["adx_14"].fillna(adx_calc)

    df["atr_14"] = calculate_atr(df)
    df["close_prev"] = df["close"].shift(1)
    return df.reset_index(drop=True)


# -------------------------
# Manifest compile / preflight
# -------------------------

def compile_job_spec(row: dict) -> Tuple[Optional[JobSpec], str, str]:
    strategy_family = normalize_text(row.get("strategy_family"))
    logic_strictness_raw = normalize_text(row.get("logic_strictness"))
    logic_strictness = LOGIC_ALIAS.get(logic_strictness_raw, logic_strictness_raw)
    swing_variant = normalize_text(row.get("swing_variant"))
    pullback_zone_variant = normalize_text(row.get("pullback_zone_variant"))
    entry_variant = normalize_text(row.get("entry_variant"))
    micro_exit_variant = normalize_text(row.get("micro_exit_variant"))
    timeframe = normalize_text(row.get("timeframe"))
    symbol = normalize_text(row.get("symbol")) or DEFAULT_SYMBOL

    ema_fast = safe_int(row.get("ema_fast"), 20)
    ema_slow = safe_int(row.get("ema_slow"), 50)

    if timeframe not in TIMEFRAMES:
        return None, "SKIPPED_ROW", f"unsupported timeframe={timeframe}"
    if strategy_family not in SUPPORTED_STRATEGY_FAMILIES:
        return None, "UNSUPPORTED_CONFIG", f"unsupported strategy_family={strategy_family}"
    if logic_strictness_raw == "relaxed":
        return None, "UNSUPPORTED_CONFIG", "ambiguous logic_strictness=relaxed"
    if logic_strictness not in {"loose", "medium", "strict"}:
        return None, "UNSUPPORTED_CONFIG", f"unsupported logic_strictness={logic_strictness_raw}"
    if swing_variant == "medium":
        return None, "UNSUPPORTED_CONFIG", "ambiguous swing_variant=medium"
    if swing_variant not in SUPPORTED_SWING_VARIANTS:
        return None, "UNSUPPORTED_CONFIG", f"unsupported swing_variant={swing_variant}"
    if pullback_zone_variant not in SUPPORTED_PULLBACK_ZONE_VARIANTS:
        return None, "UNSUPPORTED_CONFIG", f"unsupported pullback_zone_variant={pullback_zone_variant}"
    if entry_variant not in SUPPORTED_ENTRY_VARIANTS:
        return None, "UNSUPPORTED_CONFIG", f"unsupported entry_variant={entry_variant}"
    if micro_exit_variant not in SUPPORTED_MICRO_EXIT_VARIANTS:
        return None, "UNSUPPORTED_CONFIG", f"unsupported micro_exit_variant={micro_exit_variant}"
    if ema_fast <= 0 or ema_slow <= 0 or ema_fast >= ema_slow:
        return None, "INVALID_PARAM", f"invalid ema pair fast={ema_fast} slow={ema_slow}"

    manifest_rank = normalize_text(row.get("manifest_rank"))
    manifest_id = normalize_text(row.get("manifest_id")) or f"{manifest_rank}|{strategy_family}|{timeframe}|{ema_fast}|{ema_slow}"
    logic_key = "__".join([strategy_family, logic_strictness, swing_variant, pullback_zone_variant, entry_variant])
    job_id = "|".join(
        [
            timeframe,
            strategy_family,
            logic_strictness,
            swing_variant,
            pullback_zone_variant,
            entry_variant,
            micro_exit_variant,
            str(ema_fast),
            str(ema_slow),
            normalize_text(row.get("management_variant")),
            normalize_text(row.get("regime_variant")),
            normalize_text(row.get("robustness_variant")),
        ]
    )

    spec = JobSpec(
        manifest_rank=manifest_rank,
        manifest_id=manifest_id,
        timeframe=timeframe,
        strategy_family=strategy_family,
        logic_strictness=logic_strictness,
        swing_variant=swing_variant,
        pullback_zone_variant=pullback_zone_variant,
        entry_variant=entry_variant,
        micro_exit_variant=micro_exit_variant,
        management_variant=normalize_text(row.get("management_variant")),
        regime_variant=normalize_text(row.get("regime_variant")),
        robustness_variant=normalize_text(row.get("robustness_variant")),
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        symbol=symbol,
        batch_group=normalize_text(row.get("batch_group")),
        priority_tier=normalize_text(row.get("priority_tier")),
        status=normalize_text(row.get("status")),
        rationale=normalize_text(row.get("rationale")),
        logic_key=logic_key,
        job_id=job_id,
    )
    return spec, "VALID", ""



def preflight_manifest(
    manifest_path: Path,
    preflight_writer: CSVAppendWriter,
    logger: BootstrapLogger,
    flush_size: int,
) -> Tuple[List[JobSpec], Counter]:
    valid_jobs: List[JobSpec] = []
    counts: Counter = Counter()
    buffer: List[dict] = []

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts["manifest_total"] += 1
            spec, status, reason = compile_job_spec(row)
            if status == "VALID":
                counts["valid"] += 1
                valid_jobs.append(spec)  # only valid jobs kept in memory
            elif status == "INVALID_PARAM":
                counts["invalid"] += 1
            elif status == "UNSUPPORTED_CONFIG":
                counts["unsupported"] += 1
            else:
                counts["skipped"] += 1

            buffer.append(
                {
                    "manifest_rank": normalize_text(row.get("manifest_rank")),
                    "manifest_id": normalize_text(row.get("manifest_id")),
                    "timeframe": normalize_text(row.get("timeframe")),
                    "strategy_family": normalize_text(row.get("strategy_family")),
                    "logic_strictness": normalize_text(row.get("logic_strictness")),
                    "swing_variant": normalize_text(row.get("swing_variant")),
                    "pullback_zone_variant": normalize_text(row.get("pullback_zone_variant")),
                    "entry_variant": normalize_text(row.get("entry_variant")),
                    "micro_exit_variant": normalize_text(row.get("micro_exit_variant")),
                    "ema_fast": normalize_text(row.get("ema_fast")),
                    "ema_slow": normalize_text(row.get("ema_slow")),
                    "status": status,
                    "reason": reason,
                }
            )
            if len(buffer) >= flush_size:
                preflight_writer.write_rows(buffer)
                buffer.clear()

    if buffer:
        preflight_writer.write_rows(buffer)

    logger.log(
        "PREFLIGHT_DONE "
        f"manifest_total={counts['manifest_total']} valid={counts['valid']} invalid={counts['invalid']} "
        f"unsupported={counts['unsupported']} skipped={counts['skipped']}"
    )
    return valid_jobs, counts


# -------------------------
# Entry builders
# -------------------------

def build_trend_filters(df: pd.DataFrame, ema_fast: int, ema_slow: int, logic_strictness: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ema_fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean().astype("float32")
    out["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean().astype("float32")
    out["trend_up"] = out["ema_fast"] > out["ema_slow"]
    out["trend_down"] = out["ema_fast"] < out["ema_slow"]
    threshold = logic_strictness_to_threshold(logic_strictness)
    out["adx_ok"] = df["adx_14"] >= threshold
    out["distance_fast"] = ((df["close"] - out["ema_fast"]) / out["ema_fast"].replace(0, np.nan)).abs().fillna(0.0)
    return out



def build_pullback_flags(df: pd.DataFrame, trend: pd.DataFrame, zone_name: str) -> pd.DataFrame:
    zone_pct = pullback_zone_to_pct(zone_name)
    out = pd.DataFrame(index=df.index)
    out["long_pullback"] = (
        trend["trend_up"]
        & (df["low"] <= trend["ema_fast"] * (1 + zone_pct))
        & (df["low"] >= trend["ema_fast"] * (1 - zone_pct))
    )
    out["short_pullback"] = (
        trend["trend_down"]
        & (df["high"] >= trend["ema_fast"] * (1 - zone_pct))
        & (df["high"] <= trend["ema_fast"] * (1 + zone_pct))
    )
    out["long_confirm"] = df["close"] > df["close"].shift(1)
    out["short_confirm"] = df["close"] < df["close"].shift(1)
    return out



def build_entries_adx_ema_trend_continuation(job: JobSpec, df: pd.DataFrame, trend: pd.DataFrame, pb: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    long_entry = trend["trend_up"] & trend["adx_ok"] & pb["long_pullback"] & pb["long_confirm"]
    short_entry = trend["trend_down"] & trend["adx_ok"] & pb["short_pullback"] & pb["short_confirm"]
    return long_entry.fillna(False), short_entry.fillna(False)



def build_entries_ema_trend_continuation(job: JobSpec, df: pd.DataFrame, trend: pd.DataFrame, pb: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    long_entry = trend["trend_up"] & pb["long_pullback"] & pb["long_confirm"]
    short_entry = trend["trend_down"] & pb["short_pullback"] & pb["short_confirm"]
    return long_entry.fillna(False), short_entry.fillna(False)



def build_entries_bos_continuation(job: JobSpec, df: pd.DataFrame, trend: pd.DataFrame, pb: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    lookback = max(5, min(job.ema_fast // 2, 20))
    prev_high = df["high"].rolling(lookback).max().shift(1)
    prev_low = df["low"].rolling(lookback).min().shift(1)
    bos_long = df["close"] > prev_high
    bos_short = df["close"] < prev_low
    long_entry = trend["trend_up"] & bos_long.shift(1).fillna(False) & pb["long_pullback"] & pb["long_confirm"]
    short_entry = trend["trend_down"] & bos_short.shift(1).fillna(False) & pb["short_pullback"] & pb["short_confirm"]
    return long_entry.fillna(False), short_entry.fillna(False)



def build_entries_swing_pullback(job: JobSpec, df: pd.DataFrame, trend: pd.DataFrame, pb: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    # v1.0.7 addition
    swing_lookback = max(8, min(job.ema_slow // 3, 30))
    prior_swing_high = df["high"].rolling(swing_lookback).max().shift(2)
    prior_swing_low = df["low"].rolling(swing_lookback).min().shift(2)

    long_structure = trend["trend_up"] & (df["close"].shift(1) > prior_swing_high)
    short_structure = trend["trend_down"] & (df["close"].shift(1) < prior_swing_low)

    long_reclaim = (df["close"] > trend["ema_fast"]) & (df["close"] > df["open"])
    short_reclaim = (df["close"] < trend["ema_fast"]) & (df["close"] < df["open"])

    long_entry = long_structure & pb["long_pullback"] & pb["long_confirm"] & long_reclaim
    short_entry = short_structure & pb["short_pullback"] & pb["short_confirm"] & short_reclaim
    return long_entry.fillna(False), short_entry.fillna(False)



def build_entries_for_component_job(job: JobSpec, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    trend = build_trend_filters(df, job.ema_fast, job.ema_slow, job.logic_strictness)
    pb = build_pullback_flags(df, trend, job.pullback_zone_variant)

    if job.strategy_family == "adx_ema_trend_continuation":
        long_entry, short_entry = build_entries_adx_ema_trend_continuation(job, df, trend, pb)
    elif job.strategy_family == "ema_trend_continuation":
        long_entry, short_entry = build_entries_ema_trend_continuation(job, df, trend, pb)
    elif job.strategy_family == "bos_continuation":
        long_entry, short_entry = build_entries_bos_continuation(job, df, trend, pb)
    elif job.strategy_family == "swing_pullback":
        long_entry, short_entry = build_entries_swing_pullback(job, df, trend, pb)
    else:
        raise ValueError(f"unsupported strategy_family={job.strategy_family}")

    if job.swing_variant == "long":
        short_entry = pd.Series(False, index=df.index)
    elif job.swing_variant == "short":
        long_entry = pd.Series(False, index=df.index)

    return long_entry.astype(bool), short_entry.astype(bool), trend


# -------------------------
# Micro exits / simulator
# -------------------------

def micro_exit_signal(micro_exit_variant: str, side: str, df: pd.DataFrame, trend: pd.DataFrame, entry_idx: int, bars_in_trade: int, entry_price: float) -> bool:
    i = entry_idx + bars_in_trade
    if i >= len(df):
        return True

    close = float(df.iloc[i]["close"])
    ema_fast = float(trend.iloc[i]["ema_fast"])
    ema_slow = float(trend.iloc[i]["ema_slow"])
    atr = float(df.iloc[i]["atr_14"])
    adx = float(df.iloc[i]["adx_14"])
    pnl = (close - entry_price) if side == "LONG" else (entry_price - close)

    if micro_exit_variant == "no_micro_exit":
        return False
    if micro_exit_variant == "fast_invalidation":
        return bars_in_trade >= 3 and pnl < 0
    if micro_exit_variant == "momentum_fade":
        return bars_in_trade >= 2 and ((side == "LONG" and close < ema_fast) or (side == "SHORT" and close > ema_fast))
    if micro_exit_variant == "adx_fade_exit":
        return bars_in_trade >= 2 and adx < 18
    if micro_exit_variant == "baseline_pending_exit":
        return bars_in_trade >= 4
    if micro_exit_variant == "break_even_ladder_exit":
        return bars_in_trade >= 4 and pnl <= 0
    if micro_exit_variant == "atr_trailing_exit":
        return (side == "LONG" and close < entry_price + max(0.0, pnl - atr)) or (side == "SHORT" and close > entry_price - max(0.0, pnl - atr))
    if micro_exit_variant == "reverse_signal_exit":
        return (side == "LONG" and close < ema_slow) or (side == "SHORT" and close > ema_slow)
    if micro_exit_variant == "atr_guard_exit":
        return pnl < -(0.8 * atr)
    if micro_exit_variant == "volatility_crush_exit":
        return bars_in_trade >= 3 and atr < float(df["atr_14"].rolling(20).mean().iloc[i] or 0.0)
    if micro_exit_variant == "price_cross_slow_exit":
        return (side == "LONG" and close < ema_slow) or (side == "SHORT" and close > ema_slow)
    if micro_exit_variant == "price_cross_fast_exit":
        return (side == "LONG" and close < ema_fast) or (side == "SHORT" and close > ema_fast)
    if micro_exit_variant == "partial_tp_runner_exit":
        return pnl > (0.7 * atr) and bars_in_trade >= 3
    if micro_exit_variant == "time_decay_exit":
        return bars_in_trade >= 6
    if micro_exit_variant == "structure_fail_exit":
        prev_close = float(df.iloc[max(i - 1, 0)]["close"])
        return (side == "LONG" and close < min(prev_close, ema_fast)) or (side == "SHORT" and close > max(prev_close, ema_fast))

    raise ValueError(f"unsupported micro_exit_variant={micro_exit_variant}")



def simulate_side(
    df: pd.DataFrame,
    trend: pd.DataFrame,
    entries: pd.Series,
    side: str,
    micro_exit_variant: str,
    atr_mult_sl: float = 1.2,
    atr_mult_tp: float = 2.0,
    max_bars: int = 12,
) -> dict:
    trades: List[float] = []
    wins = 0
    losses = 0
    i = 0
    arr_entry = entries.to_numpy(dtype=bool)

    while i < len(df) - 2:
        if not arr_entry[i]:
            i += 1
            continue

        entry_idx = i + 1
        if entry_idx >= len(df):
            break
        entry_price = float(df.iloc[entry_idx]["open"])
        atr = max(float(df.iloc[entry_idx]["atr_14"]), 1e-6)
        sl = entry_price - atr_mult_sl * atr if side == "LONG" else entry_price + atr_mult_sl * atr
        tp = entry_price + atr_mult_tp * atr if side == "LONG" else entry_price - atr_mult_tp * atr

        exit_price = float(df.iloc[min(entry_idx + 1, len(df) - 1)]["close"])
        bars = 0
        closed = False

        for j in range(entry_idx, min(entry_idx + max_bars, len(df))):
            bars = j - entry_idx + 1
            high = float(df.iloc[j]["high"])
            low = float(df.iloc[j]["low"])
            close = float(df.iloc[j]["close"])

            if side == "LONG":
                if low <= sl:
                    exit_price = sl
                    closed = True
                    break
                if high >= tp:
                    exit_price = tp
                    closed = True
                    break
            else:
                if high >= sl:
                    exit_price = sl
                    closed = True
                    break
                if low <= tp:
                    exit_price = tp
                    closed = True
                    break

            if micro_exit_signal(micro_exit_variant, side, df, trend, entry_idx, bars, entry_price):
                exit_price = close
                closed = True
                break

        if not closed:
            last_j = min(entry_idx + max_bars - 1, len(df) - 1)
            exit_price = float(df.iloc[last_j]["close"])
            bars = last_j - entry_idx + 1

        pnl = exit_price - entry_price if side == "LONG" else entry_price - exit_price
        trades.append(pnl)
        if pnl >= 0:
            wins += 1
        else:
            losses += 1

        i = entry_idx + max(1, bars)

    trade_count = len(trades)
    pnl_sum = round(float(np.sum(trades)) if trades else 0.0, 6)
    avg_pnl = round(float(np.mean(trades)) if trades else 0.0, 6)
    win_rate = round((wins / trade_count) * 100.0, 4) if trade_count else 0.0
    gross_profit = sum(x for x in trades if x > 0)
    gross_loss = abs(sum(x for x in trades if x < 0))
    profit_factor = round((gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0), 6)

    return {
        "trade_count": trade_count,
        "wins": wins,
        "losses": losses,
        "pnl_sum": pnl_sum,
        "avg_pnl": avg_pnl,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
    }


# -------------------------
# Execution
# -------------------------

def group_key(job: JobSpec) -> Tuple[str, ...]:
    return (
        job.timeframe,
        job.strategy_family,
        job.logic_strictness,
        job.swing_variant,
        job.pullback_zone_variant,
        job.entry_variant,
        str(job.ema_fast),
        str(job.ema_slow),
        job.management_variant,
        job.regime_variant,
        job.robustness_variant,
    )



def execute_valid_jobs(
    valid_jobs: List[JobSpec],
    data_root: Path,
    feature_root: Path,
    outdir: Path,
    logger: BootstrapLogger,
    progress: ProgressTracker,
    results_writer: CSVAppendWriter,
    group_writer: CSVAppendWriter,
    progress_every_groups: int,
    continue_on_error: bool,
    manifest_counts: Counter,
) -> Counter:
    jobs_by_tf: Dict[str, List[JobSpec]] = defaultdict(list)
    for job in valid_jobs:
        jobs_by_tf[job.timeframe].append(job)

    groups_total = 0
    for tf_jobs in jobs_by_tf.values():
        groups_total += len({group_key(j) for j in tf_jobs})

    exec_counts: Counter = Counter()
    exec_counts["execution_total"] = len(valid_jobs)
    exec_counts["groups_total"] = groups_total

    start = time.time()

    progress.write(
        {
            "phase": PHASE_NAME,
            "current_timeframe": "",
            "manifest_total": manifest_counts["manifest_total"],
            "preflight_valid": manifest_counts["valid"],
            "preflight_invalid": manifest_counts["invalid"],
            "preflight_unsupported": manifest_counts["unsupported"],
            "preflight_skipped": manifest_counts["skipped"],
            "execution_total": exec_counts["execution_total"],
            "execution_done": 0,
            "execution_resumed": 0,
            "execution_error": 0,
            "execution_missing_feature": 0,
            "overall_processed": manifest_counts["invalid"] + manifest_counts["unsupported"] + manifest_counts["skipped"],
            "overall_remaining": manifest_counts["valid"],
            "overall_progress_pct": round(((manifest_counts["invalid"] + manifest_counts["unsupported"] + manifest_counts["skipped"]) / max(manifest_counts["manifest_total"], 1)) * 100.0, 4),
            "execution_processed": 0,
            "execution_remaining": exec_counts["execution_total"],
            "execution_progress_pct": 0.0,
            "observed_elapsed_min": 0.0,
            "observed_execution_rate_jobs_per_min": 0.0,
            "execution_eta_remaining_min": 0.0,
            "groups_total": exec_counts["groups_total"],
            "groups_completed": 0,
            "outdir": str(outdir),
        }
    )

    for timeframe in TIMEFRAMES:
        tf_jobs = jobs_by_tf.get(timeframe, [])
        if not tf_jobs:
            continue

        tf_symbol = tf_jobs[0].symbol
        df = load_timeframe_bundle(data_root, feature_root, tf_symbol, timeframe)
        logger.log(f"TIMEFRAME_LOAD symbol={tf_symbol} timeframe={timeframe} rows={len(df)} mode=single_timeframe_resident")

        grouped: Dict[Tuple[str, ...], List[JobSpec]] = defaultdict(list)
        for job in tf_jobs:
            grouped[group_key(job)].append(job)

        for gk, group_jobs in grouped.items():
            base_job = group_jobs[0]
            try:
                long_entry, short_entry, trend = build_entries_for_component_job(base_job, df)
                group_summary = {
                    "timeframe": timeframe,
                    "strategy_family": base_job.strategy_family,
                    "logic_strictness": base_job.logic_strictness,
                    "swing_variant": base_job.swing_variant,
                    "pullback_zone_variant": base_job.pullback_zone_variant,
                    "entry_variant": base_job.entry_variant,
                    "ema_fast": base_job.ema_fast,
                    "ema_slow": base_job.ema_slow,
                    "jobs_in_group": len(group_jobs),
                    "long_entries": int(long_entry.sum()),
                    "short_entries": int(short_entry.sum()),
                    "status": "DONE",
                    "error_reason": "",
                }
                group_writer.write_rows([group_summary])

                for job in group_jobs:
                    side_entries = long_entry if job.swing_variant == "long" else short_entry
                    metrics = simulate_side(df, trend, side_entries, "LONG" if job.swing_variant == "long" else "SHORT", job.micro_exit_variant)
                    exec_counts["execution_done"] += 1
                    results_writer.write_rows(
                        [
                            {
                                "status": "DONE",
                                "error_reason": "",
                                "job_id": job.job_id,
                                "manifest_id": job.manifest_id,
                                "manifest_rank": job.manifest_rank,
                                "timeframe": job.timeframe,
                                "strategy_family": job.strategy_family,
                                "logic_strictness": job.logic_strictness,
                                "swing_variant": job.swing_variant,
                                "pullback_zone_variant": job.pullback_zone_variant,
                                "entry_variant": job.entry_variant,
                                "micro_exit_variant": job.micro_exit_variant,
                                "management_variant": job.management_variant,
                                "regime_variant": job.regime_variant,
                                "robustness_variant": job.robustness_variant,
                                "ema_fast": job.ema_fast,
                                "ema_slow": job.ema_slow,
                                "symbol": job.symbol,
                                "logic_key": job.logic_key,
                                **metrics,
                            }
                        ]
                    )
            except KeyError as e:
                missing = str(e).strip("'\"")
                for job in group_jobs:
                    exec_counts["execution_missing_feature"] += 1
                    results_writer.write_rows(
                        [
                            {
                                "status": "MISSING_FEATURE_COLUMN",
                                "error_reason": f"missing feature column={missing}",
                                "job_id": job.job_id,
                                "manifest_id": job.manifest_id,
                                "manifest_rank": job.manifest_rank,
                                "timeframe": job.timeframe,
                                "strategy_family": job.strategy_family,
                                "logic_strictness": job.logic_strictness,
                                "swing_variant": job.swing_variant,
                                "pullback_zone_variant": job.pullback_zone_variant,
                                "entry_variant": job.entry_variant,
                                "micro_exit_variant": job.micro_exit_variant,
                                "management_variant": job.management_variant,
                                "regime_variant": job.regime_variant,
                                "robustness_variant": job.robustness_variant,
                                "ema_fast": job.ema_fast,
                                "ema_slow": job.ema_slow,
                                "symbol": job.symbol,
                                "logic_key": job.logic_key,
                                "trade_count": 0,
                                "wins": 0,
                                "losses": 0,
                                "pnl_sum": 0.0,
                                "avg_pnl": 0.0,
                                "win_rate_pct": 0.0,
                                "profit_factor": 0.0,
                            }
                        ]
                    )
                group_writer.write_rows(
                    [
                        {
                            "timeframe": timeframe,
                            "strategy_family": base_job.strategy_family,
                            "logic_strictness": base_job.logic_strictness,
                            "swing_variant": base_job.swing_variant,
                            "pullback_zone_variant": base_job.pullback_zone_variant,
                            "entry_variant": base_job.entry_variant,
                            "ema_fast": base_job.ema_fast,
                            "ema_slow": base_job.ema_slow,
                            "jobs_in_group": len(group_jobs),
                            "long_entries": 0,
                            "short_entries": 0,
                            "status": "MISSING_FEATURE_COLUMN",
                            "error_reason": f"missing feature column={missing}",
                        }
                    ]
                )
                logger.log(f"GROUP_MISSING_FEATURE timeframe={timeframe} family={base_job.strategy_family} reason=missing feature column={missing}")
                if not continue_on_error:
                    raise
            except Exception as e:
                for job in group_jobs:
                    exec_counts["execution_error"] += 1
                    results_writer.write_rows(
                        [
                            {
                                "status": "EXECUTION_ERROR",
                                "error_reason": str(e),
                                "job_id": job.job_id,
                                "manifest_id": job.manifest_id,
                                "manifest_rank": job.manifest_rank,
                                "timeframe": job.timeframe,
                                "strategy_family": job.strategy_family,
                                "logic_strictness": job.logic_strictness,
                                "swing_variant": job.swing_variant,
                                "pullback_zone_variant": job.pullback_zone_variant,
                                "entry_variant": job.entry_variant,
                                "micro_exit_variant": job.micro_exit_variant,
                                "management_variant": job.management_variant,
                                "regime_variant": job.regime_variant,
                                "robustness_variant": job.robustness_variant,
                                "ema_fast": job.ema_fast,
                                "ema_slow": job.ema_slow,
                                "symbol": job.symbol,
                                "logic_key": job.logic_key,
                                "trade_count": 0,
                                "wins": 0,
                                "losses": 0,
                                "pnl_sum": 0.0,
                                "avg_pnl": 0.0,
                                "win_rate_pct": 0.0,
                                "profit_factor": 0.0,
                            }
                        ]
                    )
                group_writer.write_rows(
                    [
                        {
                            "timeframe": timeframe,
                            "strategy_family": base_job.strategy_family,
                            "logic_strictness": base_job.logic_strictness,
                            "swing_variant": base_job.swing_variant,
                            "pullback_zone_variant": base_job.pullback_zone_variant,
                            "entry_variant": base_job.entry_variant,
                            "ema_fast": base_job.ema_fast,
                            "ema_slow": base_job.ema_slow,
                            "jobs_in_group": len(group_jobs),
                            "long_entries": 0,
                            "short_entries": 0,
                            "status": "EXECUTION_ERROR",
                            "error_reason": str(e),
                        }
                    ]
                )
                logger.log(f"GROUP_EXECUTION_ERROR timeframe={timeframe} family={base_job.strategy_family} reason={e}")
                if not continue_on_error:
                    raise

            exec_counts["groups_completed"] += 1
            processed = exec_counts["execution_done"] + exec_counts["execution_error"] + exec_counts["execution_missing_feature"]
            elapsed_min = max((time.time() - start) / 60.0, 1e-9)
            rate = processed / elapsed_min if processed > 0 else 0.0
            remaining = max(exec_counts["execution_total"] - processed, 0)
            eta = remaining / rate if rate > 0 else 0.0
            overall_processed = manifest_counts["unsupported"] + manifest_counts["invalid"] + manifest_counts["skipped"] + processed
            overall_remaining = max(manifest_counts["manifest_total"] - overall_processed, 0)

            if exec_counts["groups_completed"] % max(progress_every_groups, 1) == 0:
                progress.write(
                    {
                        "phase": PHASE_NAME,
                        "current_timeframe": timeframe,
                        "manifest_total": manifest_counts["manifest_total"],
                        "preflight_valid": manifest_counts["valid"],
                        "preflight_invalid": manifest_counts["invalid"],
                        "preflight_unsupported": manifest_counts["unsupported"],
                        "preflight_skipped": manifest_counts["skipped"],
                        "execution_total": exec_counts["execution_total"],
                        "execution_done": exec_counts["execution_done"],
                        "execution_resumed": 0,
                        "execution_error": exec_counts["execution_error"],
                        "execution_missing_feature": exec_counts["execution_missing_feature"],
                        "overall_processed": overall_processed,
                        "overall_remaining": overall_remaining,
                        "overall_progress_pct": round((overall_processed / max(manifest_counts["manifest_total"], 1)) * 100.0, 4),
                        "execution_processed": processed,
                        "execution_remaining": remaining,
                        "execution_progress_pct": round((processed / max(exec_counts["execution_total"], 1)) * 100.0, 4),
                        "observed_elapsed_min": round(elapsed_min, 4),
                        "observed_execution_rate_jobs_per_min": round(rate, 4),
                        "execution_eta_remaining_min": round(eta, 4),
                        "groups_total": exec_counts["groups_total"],
                        "groups_completed": exec_counts["groups_completed"],
                        "outdir": str(outdir),
                    }
                )

        del df
        gc.collect()

    return exec_counts


# -------------------------
# Final summaries
# -------------------------

def write_status_summary(path: Path, manifest_counts: Counter, exec_counts: Counter) -> None:
    rows = [
        {"status": "UNSUPPORTED_CONFIG", "count": manifest_counts["unsupported"]},
        {"status": "INVALID_PARAM", "count": manifest_counts["invalid"]},
        {"status": "SKIPPED_ROW", "count": manifest_counts["skipped"]},
        {"status": "DONE", "count": exec_counts["execution_done"]},
        {"status": "EXECUTION_ERROR", "count": exec_counts["execution_error"]},
        {"status": "MISSING_FEATURE_COLUMN", "count": exec_counts["execution_missing_feature"]},
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["status", "count"])
        writer.writeheader()
        writer.writerows(rows)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--phase", default=PHASE_NAME)
    parser.add_argument("--portfolio-chunk-size", type=int, default=16)
    parser.add_argument("--preflight-flush-size", type=int, default=5000)
    parser.add_argument("--progress-every-groups", type=int, default=1)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    data_root = Path(args.data_root)
    feature_root = Path(args.feature_root)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    bootstrap_log = outdir / "bootstrap.log"
    progress_path = outdir / "live_progress.json"
    preflight_path = outdir / "preflight_results.csv"
    results_path = outdir / "coverage_results_all.csv"
    group_summary_path = outdir / "summary_by_group.csv"
    status_summary_path = outdir / "status_summary.csv"
    summary_json_path = outdir / "summary.json"

    logger = BootstrapLogger(bootstrap_log)
    logger.log(f"START version={VERSION} manifest={manifest_path} phase={args.phase}")
    logger.log(
        "ENABLED_STRATEGY_FAMILIES " + ",".join(sorted(SUPPORTED_STRATEGY_FAMILIES))
    )

    preflight_writer = CSVAppendWriter(
        preflight_path,
        [
            "manifest_rank",
            "manifest_id",
            "timeframe",
            "strategy_family",
            "logic_strictness",
            "swing_variant",
            "pullback_zone_variant",
            "entry_variant",
            "micro_exit_variant",
            "ema_fast",
            "ema_slow",
            "status",
            "reason",
        ],
    )
    results_writer = CSVAppendWriter(
        results_path,
        [
            "status",
            "error_reason",
            "job_id",
            "manifest_id",
            "manifest_rank",
            "timeframe",
            "strategy_family",
            "logic_strictness",
            "swing_variant",
            "pullback_zone_variant",
            "entry_variant",
            "micro_exit_variant",
            "management_variant",
            "regime_variant",
            "robustness_variant",
            "ema_fast",
            "ema_slow",
            "symbol",
            "logic_key",
            "trade_count",
            "wins",
            "losses",
            "pnl_sum",
            "avg_pnl",
            "win_rate_pct",
            "profit_factor",
        ],
    )
    group_writer = CSVAppendWriter(
        group_summary_path,
        [
            "timeframe",
            "strategy_family",
            "logic_strictness",
            "swing_variant",
            "pullback_zone_variant",
            "entry_variant",
            "ema_fast",
            "ema_slow",
            "jobs_in_group",
            "long_entries",
            "short_entries",
            "status",
            "error_reason",
        ],
    )
    progress = ProgressTracker(progress_path, {"version": VERSION, "phase": args.phase})

    valid_jobs, manifest_counts = preflight_manifest(
        manifest_path=manifest_path,
        preflight_writer=preflight_writer,
        logger=logger,
        flush_size=args.preflight_flush_size,
    )

    exec_counts = execute_valid_jobs(
        valid_jobs=valid_jobs,
        data_root=data_root,
        feature_root=feature_root,
        outdir=outdir,
        logger=logger,
        progress=progress,
        results_writer=results_writer,
        group_writer=group_writer,
        progress_every_groups=args.progress_every_groups,
        continue_on_error=args.continue_on_error,
        manifest_counts=manifest_counts,
    )

    write_status_summary(status_summary_path, manifest_counts, exec_counts)

    final_payload = {
        "version": VERSION,
        "phase": args.phase,
        "manifest_total": manifest_counts["manifest_total"],
        "preflight_valid": manifest_counts["valid"],
        "preflight_invalid": manifest_counts["invalid"],
        "preflight_unsupported": manifest_counts["unsupported"],
        "preflight_skipped": manifest_counts["skipped"],
        "execution_total": exec_counts["execution_total"],
        "execution_done": exec_counts["execution_done"],
        "execution_resumed": 0,
        "execution_error": exec_counts["execution_error"],
        "execution_missing_feature": exec_counts["execution_missing_feature"],
        "groups_total": exec_counts["groups_total"],
        "groups_completed": exec_counts["groups_completed"],
        "updated_at_utc": utcnow_iso(),
        "outdir": str(outdir),
    }
    write_json(summary_json_path, final_payload)

    final_status = "DONE" if exec_counts["execution_error"] == 0 and exec_counts["execution_missing_feature"] == 0 else "DONE_WITH_WARNINGS"
    if manifest_counts["unsupported"] > 0 or manifest_counts["invalid"] > 0 or manifest_counts["skipped"] > 0:
        final_status = "DONE_WITH_WARNINGS"

    progress.write(
        {
            "current_timeframe": TIMEFRAMES[-1] if exec_counts["groups_completed"] else "",
            "manifest_total": manifest_counts["manifest_total"],
            "preflight_valid": manifest_counts["valid"],
            "preflight_invalid": manifest_counts["invalid"],
            "preflight_unsupported": manifest_counts["unsupported"],
            "preflight_skipped": manifest_counts["skipped"],
            "execution_total": exec_counts["execution_total"],
            "execution_done": exec_counts["execution_done"],
            "execution_resumed": 0,
            "execution_error": exec_counts["execution_error"],
            "execution_missing_feature": exec_counts["execution_missing_feature"],
            "overall_processed": manifest_counts["manifest_total"],
            "overall_remaining": 0,
            "overall_progress_pct": 100.0,
            "execution_processed": exec_counts["execution_done"] + exec_counts["execution_error"] + exec_counts["execution_missing_feature"],
            "execution_remaining": 0,
            "execution_progress_pct": 100.0,
            "observed_elapsed_min": 0.0,
            "observed_execution_rate_jobs_per_min": 0.0,
            "execution_eta_remaining_min": 0.0,
            "groups_total": exec_counts["groups_total"],
            "groups_completed": exec_counts["groups_completed"],
            "outdir": str(outdir),
        }
    )

    logger.log(
        f"FINISH final_status={final_status} manifest_total={manifest_counts['manifest_total']} "
        f"valid={manifest_counts['valid']} invalid={manifest_counts['invalid']} unsupported={manifest_counts['unsupported']} "
        f"done={exec_counts['execution_done']} resumed=0 execution_error={exec_counts['execution_error']} "
        f"missing_feature={exec_counts['execution_missing_feature']}"
    )


if __name__ == "__main__":
    main()

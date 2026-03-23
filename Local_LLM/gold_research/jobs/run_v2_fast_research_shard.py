# ==================================================================================================
# FILE: run_v2_fast_research_shard.py
# PATH: C:\Data\Bot\Local_LLM\gold_research\jobs\run_v2_fast_research_shard.py
# VERSION: v1.0.0
#
# CHANGELOG:
# - v1.0.0
#   1) Create production-style shard runner for discovery manifest
#   2) Read jobs incrementally from JSONL manifest using start/end offsets
#   3) Reuse central base feature cache instead of recalculating indicators
#   4) Group jobs by feature_cache_path to reduce repeated parquet reads
#   5) Run deterministic vectorized-prep + single-position event backtest
#   6) Support resume by skipping completed job_ids from existing results file
#   7) Write results jsonl + state jsonl + summary json per shard
#
# DESIGN:
# - This file is for fast discovery ranking, not final production validation.
# - Input manifest must be the streaming v1.0.2 manifest generated earlier.
# - This runner keeps one trade open at a time per job for deterministic comparison.
# - Entries use next-bar open to avoid same-bar lookahead entry bias.
# ==================================================================================================

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import polars as pl

VERSION = "v1.0.0"

RESULTS_FILENAME = "results.jsonl"
STATE_FILENAME = "state.jsonl"
SUMMARY_FILENAME = "summary.json"

LOAD_COLUMNS = [
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
class JobResult:
    version: str
    job_id: str
    timeframe: str
    feature_cache_path: str
    strategy_family: str
    entry_logic: str
    micro_exit: str
    regime_filter: str
    cooldown_bars: int
    side_policy: str
    volatility_filter: str
    trend_strength_filter: str
    total_trades: int
    wins: int
    losses: int
    win_rate_pct: float
    pnl_sum_r: float
    avg_pnl_r: float
    payoff_ratio: float
    profit_factor: float
    max_consecutive_losses: int
    avg_hold_bars: float
    long_trades: int
    short_trades: int
    skipped_reason: str
    build_sec: float
    generated_at_utc: str


@dataclass
class JobState:
    version: str
    job_id: str
    status: str
    message: str
    generated_at_utc: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(value: float, digits: int = 6) -> float:
    if value is None or isinstance(value, str):
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return round(float(value), digits)


def read_completed_job_ids(results_path: Path) -> set[str]:
    completed: set[str] = set()
    if not results_path.exists():
        return completed

    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                job_id = row.get("job_id")
                if isinstance(job_id, str) and job_id:
                    completed.add(job_id)
            except Exception:
                continue
    return completed


def iter_manifest_slice(manifest_path: Path, start: int, end: Optional[int]) -> Iterable[Dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start:
                continue
            if end is not None and idx >= end:
                break

            line = line.strip()
            if not line:
                continue

            yield json.loads(line)


def load_jobs_grouped_by_feature_path(
    manifest_path: Path,
    start: int,
    end: Optional[int],
    completed_job_ids: set[str],
    skip_completed: bool,
) -> Tuple[Dict[str, List[Dict[str, Any]]], int, int]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    selected_count = 0
    skipped_completed_count = 0

    for job in iter_manifest_slice(manifest_path, start, end):
        selected_count += 1
        job_id = job["job_id"]

        if skip_completed and job_id in completed_job_ids:
            skipped_completed_count += 1
            continue

        feature_cache_path = job["feature_cache_path"]
        grouped.setdefault(feature_cache_path, []).append(job)

    return grouped, selected_count, skipped_completed_count


def load_feature_frame(feature_cache_path: str) -> pl.DataFrame:
    return pl.read_parquet(feature_cache_path, columns=LOAD_COLUMNS)


def get_np(df: pl.DataFrame, col: str, dtype: Any = np.float64) -> np.ndarray:
    return df[col].to_numpy().astype(dtype, copy=False)


def get_np_bool(df: pl.DataFrame, col: str) -> np.ndarray:
    return df[col].to_numpy().astype(np.bool_, copy=False)


def get_np_str(df: pl.DataFrame, col: str) -> np.ndarray:
    return df[col].to_numpy()


def build_arrays(df: pl.DataFrame) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {
        "time": df["time"].to_list(),
        "open": get_np(df, "open"),
        "high": get_np(df, "high"),
        "low": get_np(df, "low"),
        "close": get_np(df, "close"),
        "ema_9": get_np(df, "ema_9"),
        "ema_20": get_np(df, "ema_20"),
        "ema_50": get_np(df, "ema_50"),
        "ema_200": get_np(df, "ema_200"),
        "atr_14": get_np(df, "atr_14"),
        "adx_14": get_np(df, "adx_14"),
        "rsi_14": get_np(df, "rsi_14"),
        "return_1": get_np(df, "return_1"),
        "range": get_np(df, "range"),
        "body": get_np(df, "body"),
        "upper_wick": get_np(df, "upper_wick"),
        "lower_wick": get_np(df, "lower_wick"),
        "swing_high_5": get_np(df, "swing_high_5"),
        "swing_low_5": get_np(df, "swing_low_5"),
        "swing_high_10": get_np(df, "swing_high_10"),
        "swing_low_10": get_np(df, "swing_low_10"),
        "bull_stack": get_np_bool(df, "bull_stack"),
        "bear_stack": get_np_bool(df, "bear_stack"),
        "vol_bucket": get_np_str(df, "vol_bucket"),
        "trend_bucket": get_np_str(df, "trend_bucket"),
        "price_location_bucket": get_np_str(df, "price_location_bucket"),
    }
    return arrays


def build_regime_mask(job: Dict[str, Any], a: Dict[str, np.ndarray]) -> np.ndarray:
    regime = job["regime_filter"]
    bull = a["bull_stack"]
    bear = a["bear_stack"]
    adx = a["adx_14"]

    if regime == "trend_only":
        return bull | bear
    if regime == "trend_or_neutral":
        return (bull | bear) | (adx >= 15.0)
    if regime == "range_only":
        return (~bull) & (~bear) & (adx < 20.0)
    return np.ones_like(bull, dtype=np.bool_)


def build_volatility_mask(job: Dict[str, Any], a: Dict[str, np.ndarray]) -> np.ndarray:
    vol_filter = job["volatility_filter"]
    vol_bucket = a["vol_bucket"]

    if vol_filter == "mid_high_vol":
        return (vol_bucket == "MID_VOL") | (vol_bucket == "HIGH_VOL")
    if vol_filter == "high_vol_only":
        return vol_bucket == "HIGH_VOL"
    return np.ones_like(vol_bucket, dtype=np.bool_)


def build_trend_strength_mask(job: Dict[str, Any], a: Dict[str, np.ndarray]) -> np.ndarray:
    trend_filter = job["trend_strength_filter"]
    trend_bucket = a["trend_bucket"]

    if trend_filter == "mid_trend_plus":
        return (trend_bucket == "MID_TREND") | (trend_bucket == "STRONG_TREND")
    if trend_filter == "strong_trend_only":
        return trend_bucket == "STRONG_TREND"
    return np.ones_like(trend_bucket, dtype=np.bool_)


def build_strategy_base_signals(job: Dict[str, Any], a: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    close = a["close"]
    low = a["low"]
    high = a["high"]
    ema_9 = a["ema_9"]
    ema_20 = a["ema_20"]
    ema_50 = a["ema_50"]
    ema_200 = a["ema_200"]
    rsi = a["rsi_14"]
    adx = a["adx_14"]
    ret = a["return_1"]
    bull = a["bull_stack"]
    bear = a["bear_stack"]
    swing_high_5 = a["swing_high_5"]
    swing_low_5 = a["swing_low_5"]
    swing_high_10 = a["swing_high_10"]
    swing_low_10 = a["swing_low_10"]

    strategy = job["strategy_family"]

    if strategy == "pullback_deep":
        long_sig = bull & (close > ema_50) & (close < ema_20)
        short_sig = bear & (close < ema_50) & (close > ema_20)
        return long_sig, short_sig

    if strategy == "pullback_shallow":
        long_sig = bull & (close >= ema_20) & (low <= ema_20)
        short_sig = bear & (close <= ema_20) & (high >= ema_20)
        return long_sig, short_sig

    if strategy == "trend_continuation":
        long_sig = bull & (close > ema_9) & (ret > 0.0) & (adx >= 18.0)
        short_sig = bear & (close < ema_9) & (ret < 0.0) & (adx >= 18.0)
        return long_sig, short_sig

    if strategy == "range_reversal":
        long_sig = (~bull) & (~bear) & (rsi <= 35.0) & (low <= swing_low_5)
        short_sig = (~bull) & (~bear) & (rsi >= 65.0) & (high >= swing_high_5)
        return long_sig, short_sig

    if strategy == "breakout_expansion":
        long_sig = (close > swing_high_10) & (adx >= 20.0)
        short_sig = (close < swing_low_10) & (adx >= 20.0)
        return long_sig, short_sig

    return np.zeros_like(close, dtype=np.bool_), np.zeros_like(close, dtype=np.bool_)


def build_entry_logic_filter(job: Dict[str, Any], a: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    close = a["close"]
    low = a["low"]
    high = a["high"]
    ema_20 = a["ema_20"]
    adx = a["adx_14"]
    swing_high_5 = a["swing_high_5"]
    swing_low_5 = a["swing_low_5"]

    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    logic = job["entry_logic"]

    if logic == "bos_choch_atr_adx_ema":
        long_mask = (adx >= 20.0) & (close > ema_20)
        short_mask = (adx >= 20.0) & (close < ema_20)
        return long_mask, short_mask

    if logic == "bos_choch_ema_reclaim":
        long_mask = (prev_close <= ema_20) & (close > ema_20)
        short_mask = (prev_close >= ema_20) & (close < ema_20)
        return long_mask, short_mask

    if logic == "pullback_to_ema_stack":
        long_mask = (low <= ema_20) & (close >= ema_20)
        short_mask = (high >= ema_20) & (close <= ema_20)
        return long_mask, short_mask

    if logic == "liquidity_sweep_reclaim":
        long_mask = (low < swing_low_5) & (close > swing_low_5)
        short_mask = (high > swing_high_5) & (close < swing_high_5)
        return long_mask, short_mask

    if logic == "breakout_retest":
        long_mask = (close > swing_high_5) & (low <= swing_high_5)
        short_mask = (close < swing_low_5) & (high >= swing_low_5)
        return long_mask, short_mask

    return np.ones_like(close, dtype=np.bool_), np.ones_like(close, dtype=np.bool_)


def apply_side_policy(
    job: Dict[str, Any],
    long_entries: np.ndarray,
    short_entries: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    side_policy = job["side_policy"]
    if side_policy == "long_only":
        return long_entries, np.zeros_like(short_entries, dtype=np.bool_)
    if side_policy == "short_only":
        return np.zeros_like(long_entries, dtype=np.bool_), short_entries
    return long_entries, short_entries


def build_entry_signals(job: Dict[str, Any], a: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    base_long, base_short = build_strategy_base_signals(job, a)
    logic_long, logic_short = build_entry_logic_filter(job, a)
    regime_mask = build_regime_mask(job, a)
    vol_mask = build_volatility_mask(job, a)
    trend_mask = build_trend_strength_mask(job, a)

    common_mask = regime_mask & vol_mask & trend_mask

    long_entries = base_long & logic_long & common_mask
    short_entries = base_short & logic_short & common_mask

    long_entries, short_entries = apply_side_policy(job, long_entries, short_entries)

    long_entries[0] = False
    short_entries[0] = False
    long_entries[-1] = False
    short_entries[-1] = False

    return long_entries, short_entries


def micro_exit_config(micro_exit: str) -> Dict[str, float]:
    if micro_exit == "micro_exit_v2_fast_invalidation":
        return {
            "stop_atr_mult": 0.80,
            "take_atr_mult": 1.20,
            "max_hold_bars": 8,
            "fade_ema": 9.0,
        }
    if micro_exit == "micro_exit_v2_momentum_fade":
        return {
            "stop_atr_mult": 1.00,
            "take_atr_mult": 1.80,
            "max_hold_bars": 12,
            "fade_ema": 9.0,
        }
    if micro_exit == "micro_exit_v2_structure_trail":
        return {
            "stop_atr_mult": 1.20,
            "take_atr_mult": 2.20,
            "max_hold_bars": 20,
            "fade_ema": 20.0,
        }
    return {
        "stop_atr_mult": 0.90,
        "take_atr_mult": 1.40,
        "max_hold_bars": 10,
        "fade_ema": 20.0,
    }


def pick_fade_line(a: Dict[str, np.ndarray], fade_ema: float) -> np.ndarray:
    if fade_ema <= 9.0:
        return a["ema_9"]
    return a["ema_20"]


def simulate_one_job(job: Dict[str, Any], a: Dict[str, np.ndarray]) -> Dict[str, Any]:
    long_entries, short_entries = build_entry_signals(job, a)

    open_ = a["open"]
    high = a["high"]
    low = a["low"]
    close = a["close"]
    atr = a["atr_14"]
    fade_line = pick_fade_line(a, micro_exit_config(job["micro_exit"])["fade_ema"])

    cfg = micro_exit_config(job["micro_exit"])
    stop_mult = cfg["stop_atr_mult"]
    take_mult = cfg["take_atr_mult"]
    max_hold_bars = int(cfg["max_hold_bars"])
    cooldown_bars = int(job["cooldown_bars"])

    n = len(close)
    pnl_r_list: List[float] = []
    hold_bars_list: List[int] = []
    side_list: List[str] = []

    i = 1
    cooldown_until = -1

    while i < n - 1:
        if i <= cooldown_until:
            i += 1
            continue

        if not long_entries[i] and not short_entries[i]:
            i += 1
            continue

        side = 1 if long_entries[i] else -1
        entry_idx = i + 1
        if entry_idx >= n:
            break

        entry_price = open_[entry_idx]
        risk_unit = max(atr[i] * stop_mult, 1e-9)
        stop_price = entry_price - risk_unit if side == 1 else entry_price + risk_unit
        take_price = entry_price + (risk_unit * (take_mult / stop_mult)) if side == 1 else entry_price - (risk_unit * (take_mult / stop_mult))

        exit_idx = None
        pnl_r = 0.0

        for j in range(entry_idx, min(n, entry_idx + max_hold_bars + 1)):
            bar_high = high[j]
            bar_low = low[j]
            bar_close = close[j]

            if side == 1:
                stop_hit = bar_low <= stop_price
                take_hit = bar_high >= take_price
                fade_hit = bar_close < fade_line[j]

                if stop_hit and take_hit:
                    exit_idx = j
                    pnl_r = -1.0
                    break
                if stop_hit:
                    exit_idx = j
                    pnl_r = -1.0
                    break
                if take_hit:
                    exit_idx = j
                    pnl_r = (take_price - entry_price) / risk_unit
                    break
                if fade_hit and j > entry_idx:
                    exit_idx = j
                    pnl_r = (bar_close - entry_price) / risk_unit
                    break
            else:
                stop_hit = bar_high >= stop_price
                take_hit = bar_low <= take_price
                fade_hit = bar_close > fade_line[j]

                if stop_hit and take_hit:
                    exit_idx = j
                    pnl_r = -1.0
                    break
                if stop_hit:
                    exit_idx = j
                    pnl_r = -1.0
                    break
                if take_hit:
                    exit_idx = j
                    pnl_r = (entry_price - take_price) / risk_unit
                    break
                if fade_hit and j > entry_idx:
                    exit_idx = j
                    pnl_r = (entry_price - bar_close) / risk_unit
                    break

        if exit_idx is None:
            exit_idx = min(n - 1, entry_idx + max_hold_bars)
            if side == 1:
                pnl_r = (close[exit_idx] - entry_price) / risk_unit
            else:
                pnl_r = (entry_price - close[exit_idx]) / risk_unit

        pnl_r_list.append(float(pnl_r))
        hold_bars_list.append(int(max(1, exit_idx - entry_idx + 1)))
        side_list.append("LONG" if side == 1 else "SHORT")

        cooldown_until = exit_idx + cooldown_bars
        i = exit_idx + 1

    total_trades = len(pnl_r_list)
    if total_trades == 0:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "pnl_sum_r": 0.0,
            "avg_pnl_r": 0.0,
            "payoff_ratio": 0.0,
            "profit_factor": 0.0,
            "max_consecutive_losses": 0,
            "avg_hold_bars": 0.0,
            "long_trades": 0,
            "short_trades": 0,
        }

    pnl_arr = np.array(pnl_r_list, dtype=np.float64)
    wins = int(np.sum(pnl_arr > 0.0))
    losses = int(np.sum(pnl_arr <= 0.0))
    pnl_sum_r = float(np.sum(pnl_arr))
    avg_pnl_r = float(np.mean(pnl_arr))
    win_rate_pct = float((wins / total_trades) * 100.0)

    pos = pnl_arr[pnl_arr > 0.0]
    neg = pnl_arr[pnl_arr <= 0.0]

    avg_win = float(np.mean(pos)) if pos.size > 0 else 0.0
    avg_loss_abs = float(abs(np.mean(neg))) if neg.size > 0 else 0.0
    payoff_ratio = float(avg_win / avg_loss_abs) if avg_loss_abs > 0 else 0.0
    profit_factor = float(np.sum(pos) / abs(np.sum(neg))) if neg.size > 0 and abs(np.sum(neg)) > 0 else 0.0

    max_consecutive_losses = 0
    current_losses = 0
    for x in pnl_arr:
        if x <= 0.0:
            current_losses += 1
            if current_losses > max_consecutive_losses:
                max_consecutive_losses = current_losses
        else:
            current_losses = 0

    avg_hold_bars = float(np.mean(np.array(hold_bars_list, dtype=np.float64)))
    long_trades = int(sum(1 for s in side_list if s == "LONG"))
    short_trades = int(sum(1 for s in side_list if s == "SHORT"))

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": safe_float(win_rate_pct),
        "pnl_sum_r": safe_float(pnl_sum_r),
        "avg_pnl_r": safe_float(avg_pnl_r),
        "payoff_ratio": safe_float(payoff_ratio),
        "profit_factor": safe_float(profit_factor),
        "max_consecutive_losses": max_consecutive_losses,
        "avg_hold_bars": safe_float(avg_hold_bars),
        "long_trades": long_trades,
        "short_trades": short_trades,
    }


def run_one_job(job: Dict[str, Any], a: Dict[str, np.ndarray]) -> JobResult:
    started = time.perf_counter()
    metrics = simulate_one_job(job, a)
    build_sec = time.perf_counter() - started

    skipped_reason = ""
    if metrics["total_trades"] == 0:
        skipped_reason = "NO_TRADES"

    return JobResult(
        version=VERSION,
        job_id=job["job_id"],
        timeframe=job["timeframe"],
        feature_cache_path=job["feature_cache_path"],
        strategy_family=job["strategy_family"],
        entry_logic=job["entry_logic"],
        micro_exit=job["micro_exit"],
        regime_filter=job["regime_filter"],
        cooldown_bars=int(job["cooldown_bars"]),
        side_policy=job["side_policy"],
        volatility_filter=job["volatility_filter"],
        trend_strength_filter=job["trend_strength_filter"],
        total_trades=int(metrics["total_trades"]),
        wins=int(metrics["wins"]),
        losses=int(metrics["losses"]),
        win_rate_pct=safe_float(metrics["win_rate_pct"]),
        pnl_sum_r=safe_float(metrics["pnl_sum_r"]),
        avg_pnl_r=safe_float(metrics["avg_pnl_r"]),
        payoff_ratio=safe_float(metrics["payoff_ratio"]),
        profit_factor=safe_float(metrics["profit_factor"]),
        max_consecutive_losses=int(metrics["max_consecutive_losses"]),
        avg_hold_bars=safe_float(metrics["avg_hold_bars"]),
        long_trades=int(metrics["long_trades"]),
        short_trades=int(metrics["short_trades"]),
        skipped_reason=skipped_reason,
        build_sec=safe_float(build_sec),
        generated_at_utc=utc_now_iso(),
    )


def build_summary(
    manifest_path: Path,
    outdir: Path,
    start: int,
    end: Optional[int],
    selected_count: int,
    skipped_completed_count: int,
    results_count: int,
    no_trade_count: int,
    error_count: int,
    total_elapsed_sec: float,
) -> Dict[str, Any]:
    return {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "manifest_path": str(manifest_path),
        "outdir": str(outdir),
        "start": start,
        "end": end,
        "selected_jobs": selected_count,
        "skipped_completed_jobs": skipped_completed_count,
        "result_rows_written": results_count,
        "no_trade_jobs": no_trade_count,
        "error_jobs": error_count,
        "total_elapsed_sec": round(total_elapsed_sec, 4),
        "results_path": str(outdir / RESULTS_FILENAME),
        "state_path": str(outdir / STATE_FILENAME),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fast research shard from discovery manifest.")
    parser.add_argument("--manifest", required=True, help="Path to research_job_manifest_full_discovery.jsonl")
    parser.add_argument("--outdir", required=True, help="Output directory for this shard")
    parser.add_argument("--start", type=int, required=True, help="Inclusive start line index in manifest")
    parser.add_argument("--end", type=int, default=None, help="Exclusive end line index in manifest")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N jobs")
    parser.add_argument("--skip-completed", action="store_true", default=True, help="Skip job_ids already in results.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    outdir = Path(args.outdir)
    results_path = outdir / RESULTS_FILENAME
    state_path = outdir / STATE_FILENAME
    summary_path = outdir / SUMMARY_FILENAME

    if not manifest_path.exists():
        raise RuntimeError(f"Manifest not found: {manifest_path}")

    ensure_dir(outdir)

    started = time.perf_counter()
    completed_job_ids = read_completed_job_ids(results_path)

    grouped_jobs, selected_count, skipped_completed_count = load_jobs_grouped_by_feature_path(
        manifest_path=manifest_path,
        start=args.start,
        end=args.end,
        completed_job_ids=completed_job_ids,
        skip_completed=args.skip_completed,
    )

    total_run_jobs = sum(len(v) for v in grouped_jobs.values())
    print("=" * 120)
    print(f"[START] version={VERSION}")
    print(f"[START] manifest={manifest_path}")
    print(f"[START] outdir={outdir}")
    print(f"[START] start={args.start}")
    print(f"[START] end={args.end}")
    print(f"[START] selected_jobs={selected_count}")
    print(f"[START] skipped_completed_jobs={skipped_completed_count}")
    print(f"[START] run_jobs={total_run_jobs}")
    print(f"[START] feature_cache_groups={len(grouped_jobs)}")
    print("=" * 120)

    results_count = 0
    no_trade_count = 0
    error_count = 0
    processed_count = 0

    for feature_idx, (feature_cache_path, jobs) in enumerate(grouped_jobs.items(), start=1):
        print(f"[LOAD] {feature_idx}/{len(grouped_jobs)} feature_cache={feature_cache_path} jobs={len(jobs)}")
        df = load_feature_frame(feature_cache_path)
        arrays = build_arrays(df)
        print(f"[LOAD-DONE] rows={df.height} columns={len(df.columns)}")

        for job in jobs:
            processed_count += 1
            try:
                result = run_one_job(job, arrays)
                append_jsonl(results_path, asdict(result))

                if result.skipped_reason == "NO_TRADES":
                    no_trade_count += 1

                state = JobState(
                    version=VERSION,
                    job_id=result.job_id,
                    status="DONE",
                    message=result.skipped_reason if result.skipped_reason else "OK",
                    generated_at_utc=utc_now_iso(),
                )
                append_jsonl(state_path, asdict(state))
                results_count += 1

            except Exception as e:
                error_count += 1
                state = JobState(
                    version=VERSION,
                    job_id=job.get("job_id", "UNKNOWN"),
                    status="ERROR",
                    message=str(e),
                    generated_at_utc=utc_now_iso(),
                )
                append_jsonl(state_path, asdict(state))

            if processed_count % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                rate = processed_count / elapsed if elapsed > 0 else 0.0
                print(
                    f"[PROGRESS] processed={processed_count}/{total_run_jobs} "
                    f"elapsed_sec={elapsed:.2f} jobs_per_sec={rate:.2f}"
                )

    total_elapsed_sec = time.perf_counter() - started
    summary = build_summary(
        manifest_path=manifest_path,
        outdir=outdir,
        start=args.start,
        end=args.end,
        selected_count=selected_count,
        skipped_completed_count=skipped_completed_count,
        results_count=results_count,
        no_trade_count=no_trade_count,
        error_count=error_count,
        total_elapsed_sec=total_elapsed_sec,
    )
    write_json(summary_path, summary)

    print("=" * 120)
    print(f"[DONE] results={results_path}")
    print(f"[DONE] state={state_path}")
    print(f"[DONE] summary={summary_path}")
    print(f"[DONE] processed_jobs={processed_count}")
    print(f"[DONE] results_count={results_count}")
    print(f"[DONE] no_trade_jobs={no_trade_count}")
    print(f"[DONE] error_jobs={error_count}")
    print(f"[DONE] total_elapsed_sec={total_elapsed_sec:.2f}")
    print("=" * 120)


if __name__ == "__main__":
    main()
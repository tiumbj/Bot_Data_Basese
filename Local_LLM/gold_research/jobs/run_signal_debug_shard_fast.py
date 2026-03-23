# ==================================================================================================
# FILE: run_signal_debug_shard_fast.py
# PATH: C:\Data\Bot\Local_LLM\gold_research\jobs\run_signal_debug_shard_fast.py
# VERSION: v2.0.0
#
# CHANGELOG:
# - v2.0.0
#   1) New fast shard runner using NumPy precomputed masks
#   2) Load feature cache once, reuse arrays for all jobs
#   3) Eliminate per-job DataFrame filtering bottleneck
#   4) Keep output contract compatible with summarize_signal_debug.py
#   5) Add progress logging for throughput measurement
#
# PURPOSE:
# - Fast signal-debug screening for many jobs
# - Identify where signals die before expensive VectorBT backtests
#
# OUTPUT CONTRACT:
# - job_debug_rows.jsonl
# - layer_failure_counts.json
# - stage_summary.json
# - run_summary.json
# ==================================================================================================

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import polars as pl

VERSION = "v2.0.0"
DEFAULT_PROGRESS_EVERY = 100

OUT_JOB_DEBUG_ROWS = "job_debug_rows.jsonl"
OUT_LAYER_FAILURE_COUNTS = "layer_failure_counts.json"
OUT_STAGE_SUMMARY = "stage_summary.json"
OUT_RUN_SUMMARY = "run_summary.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")


def reset_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def normalize_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text


def normalize_key(value: Any, default: str = "") -> str:
    text = normalize_text(value, default=default)
    if not text:
        return default
    return text.upper().replace("-", "_").replace(" ", "_")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def first_present(row: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


@dataclass
class JobSpec:
    job_id: str
    timeframe: str
    strategy_family: str
    entry_logic: str
    micro_exit: str
    regime_filter: str
    cooldown_bars: int
    side_policy: str
    volatility_filter: str
    trend_strength_filter: str


class FeatureArrays:
    def __init__(self, df: pl.DataFrame) -> None:
        self.row_count = df.height

        self.time = df["time"].to_list()
        self.open = df["open"].to_numpy().astype(np.float64, copy=False)
        self.high = df["high"].to_numpy().astype(np.float64, copy=False)
        self.low = df["low"].to_numpy().astype(np.float64, copy=False)
        self.close = df["close"].to_numpy().astype(np.float64, copy=False)

        self.ema_9 = df["ema_9"].to_numpy().astype(np.float64, copy=False)
        self.ema_20 = df["ema_20"].to_numpy().astype(np.float64, copy=False)
        self.ema_50 = df["ema_50"].to_numpy().astype(np.float64, copy=False)
        self.ema_200 = df["ema_200"].to_numpy().astype(np.float64, copy=False)

        self.atr_14 = df["atr_14"].to_numpy().astype(np.float64, copy=False)
        self.adx_14 = df["adx_14"].to_numpy().astype(np.float64, copy=False)
        self.rsi_14 = df["rsi_14"].to_numpy().astype(np.float64, copy=False)
        self.return_1 = df["return_1"].to_numpy().astype(np.float64, copy=False)

        self.bull_stack = df["bull_stack"].to_numpy().astype(bool, copy=False)
        self.bear_stack = df["bear_stack"].to_numpy().astype(bool, copy=False)

        self.vol_bucket = np.array(df["vol_bucket"].to_list(), dtype=object)
        self.trend_bucket = np.array(df["trend_bucket"].to_list(), dtype=object)
        self.price_location_bucket = np.array(df["price_location_bucket"].to_list(), dtype=object)

        self.swing_high_5 = df["swing_high_5"].to_numpy().astype(np.float64, copy=False)
        self.swing_low_5 = df["swing_low_5"].to_numpy().astype(np.float64, copy=False)
        self.swing_high_10 = df["swing_high_10"].to_numpy().astype(np.float64, copy=False)
        self.swing_low_10 = df["swing_low_10"].to_numpy().astype(np.float64, copy=False)

        self.base_true = np.ones(self.row_count, dtype=bool)

        self.mask_mid_vol = self.vol_bucket == "MID_VOL"
        self.mask_low_vol = self.vol_bucket == "LOW_VOL"
        self.mask_high_vol = self.vol_bucket == "HIGH_VOL"

        self.mask_strong_trend = self.trend_bucket == "STRONG_TREND"
        self.mask_mid_trend = self.trend_bucket == "MID_TREND"
        self.mask_weak_trend = self.trend_bucket == "WEAK_TREND"

        self.mask_above_ema = self.price_location_bucket == "ABOVE_EMA_STACK"
        self.mask_below_ema = self.price_location_bucket == "BELOW_EMA_STACK"
        self.mask_near_ema = self.price_location_bucket == "NEAR_EMA_STACK"

        self.long_pullback_deep = self._build_long_pullback_deep()
        self.short_pullback_deep = self._build_short_pullback_deep()
        self.long_ema_tight = self.bull_stack & self.mask_near_ema
        self.short_ema_tight = self.bear_stack & self.mask_near_ema
        self.long_swing_lkb3 = self._build_long_swing()
        self.short_swing_lkb3 = self._build_short_swing()
        self.long_bos_strict = self._build_long_bos()
        self.short_bos_strict = self._build_short_bos()
        self.long_default = self.bull_stack & (self.mask_above_ema | self.mask_near_ema)
        self.short_default = self.bear_stack & (self.mask_below_ema | self.mask_near_ema)

    def _build_long_pullback_deep(self) -> np.ndarray:
        near = self.mask_near_ema | (
            self.close <= (self.ema_20 + self.atr_14 * 0.20)
        )
        momentum_ok = self.return_1 > -0.01
        return self.bull_stack & near & momentum_ok

    def _build_short_pullback_deep(self) -> np.ndarray:
        near = self.mask_near_ema | (
            self.close >= (self.ema_20 - self.atr_14 * 0.20)
        )
        momentum_ok = self.return_1 < 0.01
        return self.bear_stack & near & momentum_ok

    def _build_long_swing(self) -> np.ndarray:
        valid = ~np.isnan(self.swing_high_5)
        return self.bull_stack & valid & (self.close > self.swing_high_5)

    def _build_short_swing(self) -> np.ndarray:
        valid = ~np.isnan(self.swing_low_5)
        return self.bear_stack & valid & (self.close < self.swing_low_5)

    def _build_long_bos(self) -> np.ndarray:
        valid = ~np.isnan(self.swing_high_10)
        return self.bull_stack & valid & (self.close > self.swing_high_10) & (self.adx_14 >= 25.0)

    def _build_short_bos(self) -> np.ndarray:
        valid = ~np.isnan(self.swing_low_10)
        return self.bear_stack & valid & (self.close < self.swing_low_10) & (self.adx_14 >= 25.0)


def load_jobs(manifest_path: Path) -> List[JobSpec]:
    if not manifest_path.exists():
        raise RuntimeError(f"Manifest not found: {manifest_path}")

    jobs: List[JobSpec] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            raw = json.loads(text)

            job_id = normalize_text(
                first_present(raw, ["job_id", "id", "name", "strategy_id"], default=f"job_{idx:06d}")
            )
            timeframe = normalize_key(first_present(raw, ["timeframe", "tf"], default="M1"), default="M1")
            strategy_family = normalize_key(
                first_present(raw, ["strategy_family", "strategy", "strategy_id"], default="UNKNOWN"),
                default="UNKNOWN",
            )
            entry_logic = normalize_key(
                first_present(raw, ["entry_logic", "entry", "entry_type"], default="DEFAULT"),
                default="DEFAULT",
            )
            micro_exit = normalize_key(
                first_present(raw, ["micro_exit", "exit_logic", "exit"], default="NONE"),
                default="NONE",
            )
            regime_filter = normalize_key(
                first_present(raw, ["regime_filter", "regime", "market_regime"], default="ALL"),
                default="ALL",
            )
            cooldown_bars = safe_int(
                first_present(raw, ["cooldown_bars", "cooldown", "entry_cooldown_bars"], default=0),
                default=0,
            )
            side_policy = normalize_key(
                first_present(raw, ["side_policy", "side", "trade_side"], default="BOTH"),
                default="BOTH",
            )
            volatility_filter = normalize_key(
                first_present(raw, ["volatility_filter", "vol_filter", "vol_bucket"], default="ALL"),
                default="ALL",
            )
            trend_strength_filter = normalize_key(
                first_present(raw, ["trend_strength_filter", "trend_filter", "trend_bucket"], default="ALL"),
                default="ALL",
            )

            jobs.append(
                JobSpec(
                    job_id=job_id,
                    timeframe=timeframe,
                    strategy_family=strategy_family,
                    entry_logic=entry_logic,
                    micro_exit=micro_exit,
                    regime_filter=regime_filter,
                    cooldown_bars=cooldown_bars,
                    side_policy=side_policy,
                    volatility_filter=volatility_filter,
                    trend_strength_filter=trend_strength_filter,
                )
            )
    if not jobs:
        raise RuntimeError(f"No jobs loaded from manifest: {manifest_path}")
    return jobs


def load_feature_cache(feature_cache_path: Path) -> FeatureArrays:
    if not feature_cache_path.exists():
        raise RuntimeError(f"Feature cache not found: {feature_cache_path}")

    required_cols = [
        "time", "open", "high", "low", "close",
        "ema_9", "ema_20", "ema_50", "ema_200",
        "atr_14", "adx_14", "rsi_14", "return_1",
        "bull_stack", "bear_stack",
        "vol_bucket", "trend_bucket", "price_location_bucket",
        "swing_high_5", "swing_low_5", "swing_high_10", "swing_low_10",
    ]
    df = pl.read_parquet(feature_cache_path, columns=required_cols)
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise RuntimeError(f"Feature cache missing columns: {sorted(missing)}")
    return FeatureArrays(df)


def count_true(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def apply_cooldown(mask: np.ndarray, cooldown_bars: int) -> np.ndarray:
    if cooldown_bars <= 0:
        return mask.copy()

    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.zeros(mask.shape[0], dtype=bool)

    keep_idx = []
    last_kept = -10**18
    for i in idx:
        if i - last_kept > cooldown_bars:
            keep_idx.append(i)
            last_kept = i

    out = np.zeros(mask.shape[0], dtype=bool)
    out[np.array(keep_idx, dtype=np.int64)] = True
    return out


def regime_mask(arr: FeatureArrays, side: str, regime_filter: str) -> np.ndarray:
    key = normalize_key(regime_filter, default="ALL")

    if key in ("ALL", "", "NONE"):
        return arr.base_true
    if key == "BULL_TREND":
        return arr.bull_stack & arr.mask_above_ema
    if key == "BEAR_TREND":
        return arr.bear_stack & arr.mask_below_ema
    if key == "WEAK_BULL":
        return arr.bull_stack & (arr.mask_near_ema | arr.mask_above_ema)
    if key == "WEAK_BEAR":
        return arr.bear_stack & (arr.mask_near_ema | arr.mask_below_ema)
    if key == "ABOVE_EMA_STACK":
        return arr.mask_above_ema
    if key == "BELOW_EMA_STACK":
        return arr.mask_below_ema
    if key == "NEAR_EMA_STACK":
        return arr.mask_near_ema
    if key == "MID_VOL":
        return arr.mask_mid_vol
    if key == "HIGH_VOL":
        return arr.mask_high_vol
    if key == "LOW_VOL":
        return arr.mask_low_vol
    if key == "STRONG_TREND":
        return arr.mask_strong_trend
    if key == "MID_TREND":
        return arr.mask_mid_trend
    if key == "WEAK_TREND":
        return arr.mask_weak_trend

    # fallback: do not kill unknown regime names
    return arr.base_true


def volatility_mask(arr: FeatureArrays, volatility_filter: str) -> np.ndarray:
    key = normalize_key(volatility_filter, default="ALL")
    if key in ("ALL", "", "NONE"):
        return arr.base_true
    if key == "MID_VOL":
        return arr.mask_mid_vol
    if key == "HIGH_VOL":
        return arr.mask_high_vol
    if key == "LOW_VOL":
        return arr.mask_low_vol
    if key == "NOT_LOW_VOL":
        return arr.mask_mid_vol | arr.mask_high_vol
    return arr.base_true


def trend_strength_mask(arr: FeatureArrays, trend_strength_filter: str) -> np.ndarray:
    key = normalize_key(trend_strength_filter, default="ALL")
    if key in ("ALL", "", "NONE"):
        return arr.base_true
    if key == "STRONG_TREND":
        return arr.mask_strong_trend
    if key == "MID_TREND":
        return arr.mask_mid_trend
    if key == "WEAK_TREND":
        return arr.mask_weak_trend
    if key == "MID_OR_STRONG":
        return arr.mask_mid_trend | arr.mask_strong_trend
    if key == "TREND_20":
        return arr.adx_14 >= 20.0
    if key == "TREND_25":
        return arr.adx_14 >= 25.0
    return arr.base_true


def entry_mask(arr: FeatureArrays, entry_logic: str, side: str) -> np.ndarray:
    key = normalize_key(entry_logic, default="DEFAULT")

    if side == "LONG":
        if "PULLBACK_DEEP" in key:
            return arr.long_pullback_deep
        if "EMA_TIGHT" in key:
            return arr.long_ema_tight
        if "SWING_LKB3" in key:
            return arr.long_swing_lkb3
        if "BOS_STRICT" in key:
            return arr.long_bos_strict
        return arr.long_default

    if side == "SHORT":
        if "PULLBACK_DEEP" in key:
            return arr.short_pullback_deep
        if "EMA_TIGHT" in key:
            return arr.short_ema_tight
        if "SWING_LKB3" in key:
            return arr.short_swing_lkb3
        if "BOS_STRICT" in key:
            return arr.short_bos_strict
        return arr.short_default

    raise RuntimeError(f"Unsupported side for entry mask: {side}")


def micro_exit_survival_factor(micro_exit: str) -> float:
    key = normalize_key(micro_exit, default="NONE")
    if key in ("NONE", ""):
        return 1.00
    if "FAST_INVALIDATION" in key:
        return 0.82
    if "MOMENTUM_FADE" in key:
        return 0.88
    if "MICRO_EXIT" in key:
        return 0.90
    return 0.92


def side_list(side_policy: str) -> List[str]:
    key = normalize_key(side_policy, default="BOTH")
    if key in ("LONG_ONLY", "LONG", "BUY_ONLY", "BUY"):
        return ["LONG"]
    if key in ("SHORT_ONLY", "SHORT", "SELL_ONLY", "SELL"):
        return ["SHORT"]
    return ["LONG", "SHORT"]


def evaluate_one_job(job: JobSpec, arr: FeatureArrays) -> Dict[str, Any]:
    pre_side_total = 0
    per_side_final_masks: List[np.ndarray] = []

    for side in side_list(job.side_policy):
        m_regime = regime_mask(arr, side, job.regime_filter)
        m_vol = volatility_mask(arr, job.volatility_filter)
        m_trend = trend_strength_mask(arr, job.trend_strength_filter)
        m_entry = entry_mask(arr, job.entry_logic, side)

        m_pre_side = m_regime & m_vol & m_trend & m_entry
        pre_side_total += count_true(m_pre_side)

        if count_true(m_pre_side) > 0:
            per_side_final_masks.append(m_pre_side)

    if pre_side_total <= 0:
        return {
            "job_id": job.job_id,
            "timeframe": job.timeframe,
            "strategy_family": job.strategy_family,
            "entry_logic": job.entry_logic,
            "micro_exit": job.micro_exit,
            "regime_filter": job.regime_filter,
            "cooldown_bars": job.cooldown_bars,
            "side_policy": job.side_policy,
            "volatility_filter": job.volatility_filter,
            "trend_strength_filter": job.trend_strength_filter,
            "pre_side_total_count": 0,
            "final_total_count": 0,
            "executable_total_count": 0,
            "estimated_trade_count": 0,
            "kill_stage": "FILTER_INTERSECTION",
            "kill_reason": "No rows survived combined filters before cooldown",
        }

    merged_mask = np.zeros(arr.row_count, dtype=bool)
    for m in per_side_final_masks:
        merged_mask |= m

    final_total_count = count_true(merged_mask)
    if final_total_count <= 0:
        return {
            "job_id": job.job_id,
            "timeframe": job.timeframe,
            "strategy_family": job.strategy_family,
            "entry_logic": job.entry_logic,
            "micro_exit": job.micro_exit,
            "regime_filter": job.regime_filter,
            "cooldown_bars": job.cooldown_bars,
            "side_policy": job.side_policy,
            "volatility_filter": job.volatility_filter,
            "trend_strength_filter": job.trend_strength_filter,
            "pre_side_total_count": pre_side_total,
            "final_total_count": 0,
            "executable_total_count": 0,
            "estimated_trade_count": 0,
            "kill_stage": "FINAL_SIGNAL",
            "kill_reason": "No merged signal rows survived",
        }

    executable_mask = apply_cooldown(merged_mask, job.cooldown_bars)
    executable_total_count = count_true(executable_mask)

    if executable_total_count <= 0:
        return {
            "job_id": job.job_id,
            "timeframe": job.timeframe,
            "strategy_family": job.strategy_family,
            "entry_logic": job.entry_logic,
            "micro_exit": job.micro_exit,
            "regime_filter": job.regime_filter,
            "cooldown_bars": job.cooldown_bars,
            "side_policy": job.side_policy,
            "volatility_filter": job.volatility_filter,
            "trend_strength_filter": job.trend_strength_filter,
            "pre_side_total_count": pre_side_total,
            "final_total_count": final_total_count,
            "executable_total_count": 0,
            "estimated_trade_count": 0,
            "kill_stage": "COOLDOWN",
            "kill_reason": "Cooldown removed all executable rows",
        }

    survival_factor = micro_exit_survival_factor(job.micro_exit)
    estimated_trade_count = int(math.floor(executable_total_count * survival_factor))

    if estimated_trade_count <= 0:
        return {
            "job_id": job.job_id,
            "timeframe": job.timeframe,
            "strategy_family": job.strategy_family,
            "entry_logic": job.entry_logic,
            "micro_exit": job.micro_exit,
            "regime_filter": job.regime_filter,
            "cooldown_bars": job.cooldown_bars,
            "side_policy": job.side_policy,
            "volatility_filter": job.volatility_filter,
            "trend_strength_filter": job.trend_strength_filter,
            "pre_side_total_count": pre_side_total,
            "final_total_count": final_total_count,
            "executable_total_count": executable_total_count,
            "estimated_trade_count": 0,
            "kill_stage": "EXECUTABLE_ZERO",
            "kill_reason": "Micro-exit survival factor reduced executable trades to zero",
        }

    return {
        "job_id": job.job_id,
        "timeframe": job.timeframe,
        "strategy_family": job.strategy_family,
        "entry_logic": job.entry_logic,
        "micro_exit": job.micro_exit,
        "regime_filter": job.regime_filter,
        "cooldown_bars": job.cooldown_bars,
        "side_policy": job.side_policy,
        "volatility_filter": job.volatility_filter,
        "trend_strength_filter": job.trend_strength_filter,
        "pre_side_total_count": pre_side_total,
        "final_total_count": final_total_count,
        "executable_total_count": executable_total_count,
        "estimated_trade_count": estimated_trade_count,
        "kill_stage": "SURVIVED",
        "kill_reason": "Executable signal rows remain after all filters",
    }


def summarize_rows(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, Any]]:
    kill_counter = Counter()
    survived_jobs = 0

    pre_side_nonzero = 0
    final_nonzero = 0
    executable_nonzero = 0

    for row in rows:
        kill_counter[row["kill_stage"]] += 1
        if row["estimated_trade_count"] > 0:
            survived_jobs += 1
        if row["pre_side_total_count"] > 0:
            pre_side_nonzero += 1
        if row["final_total_count"] > 0:
            final_nonzero += 1
        if row["executable_total_count"] > 0:
            executable_nonzero += 1

    stage_summary = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "total_jobs": len(rows),
        "stage_nonzero_jobs": {
            "pre_side_total_count": pre_side_nonzero,
            "final_total_count": final_nonzero,
            "executable_total_count": executable_nonzero,
            "estimated_trade_count": survived_jobs,
        },
        "kill_stage_counts": dict(kill_counter),
    }
    return dict(kill_counter), stage_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast signal debug shard runner using NumPy masks.")
    parser.add_argument("--manifest", required=True, help="JSONL manifest for jobs")
    parser.add_argument("--feature-cache", required=True, help="Base feature cache parquet for one timeframe")
    parser.add_argument("--outdir", required=True, help="Output directory for debug artifacts")
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY, help="Progress print interval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    feature_cache_path = Path(args.feature_cache)
    outdir = Path(args.outdir)
    progress_every = max(1, int(args.progress_every))

    ensure_dir(outdir)

    out_job_rows = outdir / OUT_JOB_DEBUG_ROWS
    out_layer_failure = outdir / OUT_LAYER_FAILURE_COUNTS
    out_stage_summary = outdir / OUT_STAGE_SUMMARY
    out_run_summary = outdir / OUT_RUN_SUMMARY

    reset_file(out_job_rows)

    jobs = load_jobs(manifest_path)
    arr = load_feature_cache(feature_cache_path)

    total_jobs = len(jobs)
    print(f"[LOAD] 1/1 feature_cache={feature_cache_path} jobs={total_jobs}")
    print(f"[LOAD-DONE] rows={arr.row_count} columns=precomputed")

    started = time.perf_counter()
    results: List[Dict[str, Any]] = []

    for idx, job in enumerate(jobs, start=1):
        row = evaluate_one_job(job, arr)
        results.append(row)
        append_jsonl(out_job_rows, row)

        if idx % progress_every == 0 or idx == total_jobs:
            elapsed = time.perf_counter() - started
            jobs_per_sec = idx / elapsed if elapsed > 0 else 0.0
            print(
                f"[PROGRESS] processed={idx}/{total_jobs} "
                f"elapsed_sec={elapsed:.2f} jobs_per_sec={jobs_per_sec:.2f}"
            )

    layer_failure_counts, stage_summary = summarize_rows(results)

    survived_jobs = int(sum(1 for row in results if row["estimated_trade_count"] > 0))
    top_kill_stage = "SURVIVED"
    filtered_failures = {k: v for k, v in layer_failure_counts.items() if k != "SURVIVED"}
    if filtered_failures:
        top_kill_stage = max(filtered_failures.items(), key=lambda x: x[1])[0]

    run_summary = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "manifest_path": str(manifest_path),
        "feature_cache_path": str(feature_cache_path),
        "outdir": str(outdir),
        "total_jobs": total_jobs,
        "survived_jobs": survived_jobs,
        "top_kill_stage": top_kill_stage,
        "elapsed_sec": round(time.perf_counter() - started, 4),
        "jobs_per_sec": round(total_jobs / max(time.perf_counter() - started, 1e-9), 4),
    }

    write_json(out_layer_failure, layer_failure_counts)
    write_json(out_stage_summary, stage_summary)
    write_json(out_run_summary, run_summary)

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] job_debug_rows={out_job_rows}")
    print(f"[DONE] layer_failure_counts={out_layer_failure}")
    print(f"[DONE] stage_summary={out_stage_summary}")
    print(f"[DONE] run_summary={out_run_summary}")
    print(f"[DONE] total_jobs={total_jobs}")
    print(f"[DONE] survived_jobs={survived_jobs}")
    print(f"[DONE] top_kill_stage={top_kill_stage}")
    print("=" * 120)


if __name__ == "__main__":
    main()
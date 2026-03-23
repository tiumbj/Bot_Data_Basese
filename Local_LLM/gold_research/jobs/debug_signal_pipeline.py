# ==================================================================================================
# FILE: debug_signal_pipeline.py
# PATH: C:\Data\Bot\Local_LLM\gold_research\jobs\debug_signal_pipeline.py
# VERSION: v1.1.0
#
# CHANGELOG:
# - v1.1.0
#   1) Rewrite debugger to be manifest-aware for actual enum names used in discovery manifest
#   2) Support actual enum values from manifest summary:
#      side_policies: LONG_ONLY / SHORT_ONLY / BIDIRECTIONAL
#      volatility: none / atr_mid_high_only / atr_high_only
#      trend_strength: none / adx20_plus / adx25_plus
#      regime: trend_only / trend_or_neutral / volatility_gated / always_on
#      entry logic: includes breakout_retest_impulse
#   3) Add relaxed intersection diagnostic lanes:
#      strict_intersection, no_trend_strength, no_volatility, no_regime, entry_only
#   4) Add seed recommendation flags for combinations that still have executable entries in relaxed lanes
#   5) Keep outputs compatible with summarize step
# ==================================================================================================

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

VERSION = "v1.1.0"

JOB_DEBUG_FILENAME = "job_debug_rows.jsonl"
LEADERBOARD_FILENAME = "job_debug_leaderboard.csv"
STAGE_SUMMARY_FILENAME = "stage_summary.json"
LAYER_FAILURE_FILENAME = "layer_failure_counts.json"
RUN_SUMMARY_FILENAME = "run_summary.json"

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
class JobDebugRow:
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
    bars_total: int
    base_long_count: int
    base_short_count: int
    base_total_count: int
    entry_logic_long_count: int
    entry_logic_short_count: int
    entry_logic_total_count: int
    regime_pass_count: int
    volatility_pass_count: int
    trend_strength_pass_count: int
    common_mask_pass_count: int
    pre_side_long_count: int
    pre_side_short_count: int
    pre_side_total_count: int
    final_long_count: int
    final_short_count: int
    final_total_count: int
    executable_long_count: int
    executable_short_count: int
    executable_total_count: int
    estimated_trade_count: int
    relaxed_no_trend_exec_count: int
    relaxed_no_vol_exec_count: int
    relaxed_no_regime_exec_count: int
    relaxed_entry_only_exec_count: int
    seed_candidate_flag: bool
    survival_base_to_logic_pct: float
    survival_logic_to_common_pct: float
    survival_common_to_final_pct: float
    survival_final_to_executable_pct: float
    kill_stage: str
    kill_reason: str
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


def safe_pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 4)


def read_manifest_slice(manifest_path: Path, start: int, end: Optional[int]) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start:
                continue
            if end is not None and idx >= end:
                break
            line = line.strip()
            if not line:
                continue
            jobs.append(json.loads(line))
    return jobs


def group_jobs_by_feature_path(jobs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for job in jobs:
        grouped.setdefault(job["feature_cache_path"], []).append(job)
    return grouped


def load_feature_frame(feature_cache_path: str) -> pl.DataFrame:
    return pl.read_parquet(feature_cache_path, columns=LOAD_COLUMNS)


def get_np(df: pl.DataFrame, col: str, dtype: Any = np.float64) -> np.ndarray:
    return df[col].to_numpy().astype(dtype, copy=False)


def get_np_bool(df: pl.DataFrame, col: str) -> np.ndarray:
    return df[col].to_numpy().astype(np.bool_, copy=False)


def get_np_str(df: pl.DataFrame, col: str) -> np.ndarray:
    return df[col].to_numpy()


def build_arrays(df: pl.DataFrame) -> Dict[str, np.ndarray]:
    return {
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


def count_true(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def build_regime_mask(job: Dict[str, Any], a: Dict[str, np.ndarray]) -> np.ndarray:
    regime = str(job["regime_filter"])
    bull = a["bull_stack"]
    bear = a["bear_stack"]
    adx = a["adx_14"]
    vol_bucket = a["vol_bucket"]

    if regime == "trend_only":
        return bull | bear
    if regime == "trend_or_neutral":
        return (bull | bear) | (adx >= 15.0)
    if regime == "volatility_gated":
        return (vol_bucket == "MID_VOL") | (vol_bucket == "HIGH_VOL")
    if regime == "always_on":
        return np.ones_like(bull, dtype=np.bool_)
    return np.ones_like(bull, dtype=np.bool_)


def build_volatility_mask(job: Dict[str, Any], a: Dict[str, np.ndarray]) -> np.ndarray:
    vol_filter = str(job["volatility_filter"])
    vol_bucket = a["vol_bucket"]

    if vol_filter == "none":
        return np.ones_like(vol_bucket, dtype=np.bool_)
    if vol_filter == "atr_mid_high_only":
        return (vol_bucket == "MID_VOL") | (vol_bucket == "HIGH_VOL")
    if vol_filter == "atr_high_only":
        return vol_bucket == "HIGH_VOL"
    return np.ones_like(vol_bucket, dtype=np.bool_)


def build_trend_strength_mask(job: Dict[str, Any], a: Dict[str, np.ndarray]) -> np.ndarray:
    trend_filter = str(job["trend_strength_filter"])
    adx = a["adx_14"]

    if trend_filter == "none":
        return np.ones_like(adx, dtype=np.bool_)
    if trend_filter == "adx20_plus":
        return adx >= 20.0
    if trend_filter == "adx25_plus":
        return adx >= 25.0
    return np.ones_like(adx, dtype=np.bool_)


def build_strategy_base_signals(job: Dict[str, Any], a: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    close = a["close"]
    low = a["low"]
    high = a["high"]
    ema_9 = a["ema_9"]
    ema_20 = a["ema_20"]
    ema_50 = a["ema_50"]
    rsi = a["rsi_14"]
    adx = a["adx_14"]
    ret = a["return_1"]
    bull = a["bull_stack"]
    bear = a["bear_stack"]
    swing_high_5 = a["swing_high_5"]
    swing_low_5 = a["swing_low_5"]
    swing_high_10 = a["swing_high_10"]
    swing_low_10 = a["swing_low_10"]

    strategy = str(job["strategy_family"])

    if strategy == "pullback_deep":
        long_sig = bull & (close > ema_50) & (low <= ema_20)
        short_sig = bear & (close < ema_50) & (high >= ema_20)
        return long_sig, short_sig

    if strategy == "pullback_shallow":
        long_sig = bull & (close >= ema_20) & (low <= ema_9)
        short_sig = bear & (close <= ema_20) & (high >= ema_9)
        return long_sig, short_sig

    if strategy == "trend_continuation":
        long_sig = bull & (close > ema_9) & (ret > 0.0)
        short_sig = bear & (close < ema_9) & (ret < 0.0)
        return long_sig, short_sig

    if strategy == "range_reversal":
        long_sig = (~bull) & (~bear) & (rsi <= 38.0) & (low <= swing_low_5)
        short_sig = (~bull) & (~bear) & (rsi >= 62.0) & (high >= swing_high_5)
        return long_sig, short_sig

    if strategy == "breakout_expansion":
        long_sig = (close > swing_high_10)
        short_sig = (close < swing_low_10)
        return long_sig, short_sig

    return np.zeros_like(close, dtype=np.bool_), np.zeros_like(close, dtype=np.bool_)


def build_entry_logic_filter(job: Dict[str, Any], a: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    close = a["close"]
    low = a["low"]
    high = a["high"]
    ema_9 = a["ema_9"]
    ema_20 = a["ema_20"]
    adx = a["adx_14"]
    swing_high_5 = a["swing_high_5"]
    swing_low_5 = a["swing_low_5"]

    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    logic = str(job["entry_logic"])

    if logic == "bos_choch_atr_adx_ema":
        long_mask = (adx >= 18.0) & (close > ema_20)
        short_mask = (adx >= 18.0) & (close < ema_20)
        return long_mask, short_mask

    if logic == "bos_choch_ema_reclaim":
        long_mask = (prev_close <= ema_20) & (close > ema_20)
        short_mask = (prev_close >= ema_20) & (close < ema_20)
        return long_mask, short_mask

    if logic == "pullback_to_ema_stack":
        long_mask = (low <= ema_20) & (close >= ema_9)
        short_mask = (high >= ema_20) & (close <= ema_9)
        return long_mask, short_mask

    if logic == "liquidity_sweep_reclaim":
        long_mask = (low < swing_low_5) & (close >= swing_low_5)
        short_mask = (high > swing_high_5) & (close <= swing_high_5)
        return long_mask, short_mask

    if logic in ("breakout_retest", "breakout_retest_impulse"):
        long_mask = (close > swing_high_5) | ((low <= swing_high_5) & (close > ema_9))
        short_mask = (close < swing_low_5) | ((high >= swing_low_5) & (close < ema_9))
        return long_mask, short_mask

    return np.ones_like(close, dtype=np.bool_), np.ones_like(close, dtype=np.bool_)


def apply_side_policy(job: Dict[str, Any], long_entries: np.ndarray, short_entries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    side_policy = str(job["side_policy"])
    if side_policy == "LONG_ONLY":
        return long_entries, np.zeros_like(short_entries, dtype=np.bool_)
    if side_policy == "SHORT_ONLY":
        return np.zeros_like(long_entries, dtype=np.bool_), short_entries
    return long_entries, short_entries


def build_executable_signals(final_long: np.ndarray, final_short: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    executable_long = final_long.copy()
    executable_short = final_short.copy()

    if executable_long.size > 0:
        executable_long[0] = False
        executable_short[0] = False
        executable_long[-1] = False
        executable_short[-1] = False

    return executable_long, executable_short


def estimate_trade_count(executable_long: np.ndarray, executable_short: np.ndarray, cooldown_bars: int) -> int:
    n = len(executable_long)
    trade_count = 0
    i = 1
    cooldown_until = -1

    while i < n - 1:
        if i <= cooldown_until:
            i += 1
            continue

        if executable_long[i] or executable_short[i]:
            trade_count += 1
            cooldown_until = i + max(0, int(cooldown_bars))
        i += 1

    return trade_count


def determine_kill_stage(
    base_total_count: int,
    entry_logic_total_count: int,
    common_mask_pass_count: int,
    pre_side_total_count: int,
    final_total_count: int,
    executable_total_count: int,
    estimated_trade_count: int,
) -> Tuple[str, str]:
    if base_total_count == 0:
        return "BASE_STRATEGY", "Base strategy produced zero raw signals"
    if entry_logic_total_count == 0:
        return "ENTRY_LOGIC", "Entry logic produced zero bars"
    if common_mask_pass_count == 0:
        return "COMMON_MASK", "Common masks removed all bars"
    if pre_side_total_count == 0:
        return "FILTER_INTERSECTION", "Base + entry + common masks had no overlap"
    if final_total_count == 0:
        return "SIDE_POLICY", "Side policy removed remaining signals"
    if executable_total_count == 0:
        return "EXECUTABLE_NEXT_BAR", "Signals existed but none were executable next bar"
    if estimated_trade_count == 0:
        return "COOLDOWN_OR_COLLISION", "Signals existed but collapsed after cooldown estimate"
    return "SURVIVED", "Signal path produced executable entries"


def relaxed_exec_count(
    base_long: np.ndarray,
    base_short: np.ndarray,
    logic_long: np.ndarray,
    logic_short: np.ndarray,
    regime_mask: np.ndarray,
    volatility_mask: np.ndarray,
    trend_strength_mask: np.ndarray,
    side_policy: str,
    cooldown_bars: int,
    drop_regime: bool = False,
    drop_vol: bool = False,
    drop_trend: bool = False,
    entry_only: bool = False,
) -> int:
    if entry_only:
        pre_side_long = base_long & logic_long
        pre_side_short = base_short & logic_short
    else:
        mask = np.ones_like(regime_mask, dtype=np.bool_)
        if not drop_regime:
            mask &= regime_mask
        if not drop_vol:
            mask &= volatility_mask
        if not drop_trend:
            mask &= trend_strength_mask
        pre_side_long = base_long & logic_long & mask
        pre_side_short = base_short & logic_short & mask

    job_stub = {"side_policy": side_policy}
    final_long, final_short = apply_side_policy(job_stub, pre_side_long, pre_side_short)
    exec_long, exec_short = build_executable_signals(final_long, final_short)
    return estimate_trade_count(exec_long, exec_short, cooldown_bars)


def debug_one_job(job: Dict[str, Any], a: Dict[str, np.ndarray]) -> JobDebugRow:
    base_long, base_short = build_strategy_base_signals(job, a)
    logic_long, logic_short = build_entry_logic_filter(job, a)
    regime_mask = build_regime_mask(job, a)
    volatility_mask = build_volatility_mask(job, a)
    trend_strength_mask = build_trend_strength_mask(job, a)

    common_mask = regime_mask & volatility_mask & trend_strength_mask

    pre_side_long = base_long & logic_long & common_mask
    pre_side_short = base_short & logic_short & common_mask

    final_long, final_short = apply_side_policy(job, pre_side_long, pre_side_short)
    executable_long, executable_short = build_executable_signals(final_long, final_short)

    estimated_trade_count = estimate_trade_count(
        executable_long=executable_long,
        executable_short=executable_short,
        cooldown_bars=int(job["cooldown_bars"]),
    )

    relaxed_no_trend_exec_count = relaxed_exec_count(
        base_long, base_short, logic_long, logic_short,
        regime_mask, volatility_mask, trend_strength_mask,
        str(job["side_policy"]), int(job["cooldown_bars"]),
        drop_trend=True
    )
    relaxed_no_vol_exec_count = relaxed_exec_count(
        base_long, base_short, logic_long, logic_short,
        regime_mask, volatility_mask, trend_strength_mask,
        str(job["side_policy"]), int(job["cooldown_bars"]),
        drop_vol=True
    )
    relaxed_no_regime_exec_count = relaxed_exec_count(
        base_long, base_short, logic_long, logic_short,
        regime_mask, volatility_mask, trend_strength_mask,
        str(job["side_policy"]), int(job["cooldown_bars"]),
        drop_regime=True
    )
    relaxed_entry_only_exec_count = relaxed_exec_count(
        base_long, base_short, logic_long, logic_short,
        regime_mask, volatility_mask, trend_strength_mask,
        str(job["side_policy"]), int(job["cooldown_bars"]),
        entry_only=True
    )

    base_long_count = count_true(base_long)
    base_short_count = count_true(base_short)
    base_total_count = base_long_count + base_short_count

    entry_logic_long_count = count_true(logic_long)
    entry_logic_short_count = count_true(logic_short)
    entry_logic_total_count = entry_logic_long_count + entry_logic_short_count

    regime_pass_count = count_true(regime_mask)
    volatility_pass_count = count_true(volatility_mask)
    trend_strength_pass_count = count_true(trend_strength_mask)
    common_mask_pass_count = count_true(common_mask)

    pre_side_long_count = count_true(pre_side_long)
    pre_side_short_count = count_true(pre_side_short)
    pre_side_total_count = pre_side_long_count + pre_side_short_count

    final_long_count = count_true(final_long)
    final_short_count = count_true(final_short)
    final_total_count = final_long_count + final_short_count

    executable_long_count = count_true(executable_long)
    executable_short_count = count_true(executable_short)
    executable_total_count = executable_long_count + executable_short_count

    kill_stage, kill_reason = determine_kill_stage(
        base_total_count=base_total_count,
        entry_logic_total_count=entry_logic_total_count,
        common_mask_pass_count=common_mask_pass_count,
        pre_side_total_count=pre_side_total_count,
        final_total_count=final_total_count,
        executable_total_count=executable_total_count,
        estimated_trade_count=estimated_trade_count,
    )

    seed_candidate_flag = bool(
        estimated_trade_count > 0
        or relaxed_no_trend_exec_count > 0
        or relaxed_no_vol_exec_count > 0
        or relaxed_no_regime_exec_count > 0
        or relaxed_entry_only_exec_count > 0
    )

    return JobDebugRow(
        version=VERSION,
        job_id=job["job_id"],
        timeframe=job["timeframe"],
        feature_cache_path=job["feature_cache_path"],
        strategy_family=str(job["strategy_family"]),
        entry_logic=str(job["entry_logic"]),
        micro_exit=str(job["micro_exit"]),
        regime_filter=str(job["regime_filter"]),
        cooldown_bars=int(job["cooldown_bars"]),
        side_policy=str(job["side_policy"]),
        volatility_filter=str(job["volatility_filter"]),
        trend_strength_filter=str(job["trend_strength_filter"]),
        bars_total=len(a["close"]),
        base_long_count=base_long_count,
        base_short_count=base_short_count,
        base_total_count=base_total_count,
        entry_logic_long_count=entry_logic_long_count,
        entry_logic_short_count=entry_logic_short_count,
        entry_logic_total_count=entry_logic_total_count,
        regime_pass_count=regime_pass_count,
        volatility_pass_count=volatility_pass_count,
        trend_strength_pass_count=trend_strength_pass_count,
        common_mask_pass_count=common_mask_pass_count,
        pre_side_long_count=pre_side_long_count,
        pre_side_short_count=pre_side_short_count,
        pre_side_total_count=pre_side_total_count,
        final_long_count=final_long_count,
        final_short_count=final_short_count,
        final_total_count=final_total_count,
        executable_long_count=executable_long_count,
        executable_short_count=executable_short_count,
        executable_total_count=executable_total_count,
        estimated_trade_count=estimated_trade_count,
        relaxed_no_trend_exec_count=relaxed_no_trend_exec_count,
        relaxed_no_vol_exec_count=relaxed_no_vol_exec_count,
        relaxed_no_regime_exec_count=relaxed_no_regime_exec_count,
        relaxed_entry_only_exec_count=relaxed_entry_only_exec_count,
        seed_candidate_flag=seed_candidate_flag,
        survival_base_to_logic_pct=safe_pct(pre_side_total_count, max(base_total_count, 1)),
        survival_logic_to_common_pct=safe_pct(pre_side_total_count, max(entry_logic_total_count, 1)),
        survival_common_to_final_pct=safe_pct(final_total_count, max(pre_side_total_count, 1)),
        survival_final_to_executable_pct=safe_pct(executable_total_count, max(final_total_count, 1)),
        kill_stage=kill_stage,
        kill_reason=kill_reason,
        generated_at_utc=utc_now_iso(),
    )


def build_stage_summary(df: pl.DataFrame) -> Dict[str, Any]:
    kill_stage_counts = (
        df.group_by("kill_stage")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .to_dicts()
    )

    seed_candidates = int(df.filter(pl.col("seed_candidate_flag") == True).height)

    return {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "total_jobs": int(df.height),
        "strict_survived_jobs": int(df.filter(pl.col("estimated_trade_count") > 0).height),
        "seed_candidate_jobs": seed_candidates,
        "relaxed_no_trend_positive": int(df.filter(pl.col("relaxed_no_trend_exec_count") > 0).height),
        "relaxed_no_vol_positive": int(df.filter(pl.col("relaxed_no_vol_exec_count") > 0).height),
        "relaxed_no_regime_positive": int(df.filter(pl.col("relaxed_no_regime_exec_count") > 0).height),
        "relaxed_entry_only_positive": int(df.filter(pl.col("relaxed_entry_only_exec_count") > 0).height),
        "kill_stage_counts": kill_stage_counts,
    }


def build_layer_failure_counts(df: pl.DataFrame) -> Dict[str, Any]:
    return {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "base_strategy_zero": int(df.filter(pl.col("base_total_count") == 0).height),
        "entry_logic_zero": int(df.filter(pl.col("entry_logic_total_count") == 0).height),
        "filter_intersection_zero": int(df.filter(pl.col("kill_stage") == "FILTER_INTERSECTION").height),
        "strict_survived": int(df.filter(pl.col("estimated_trade_count") > 0).height),
        "seed_candidate_jobs": int(df.filter(pl.col("seed_candidate_flag") == True).height),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug signal pipeline for discovery manifest jobs.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    outdir = Path(args.outdir)
    job_debug_path = outdir / JOB_DEBUG_FILENAME
    leaderboard_path = outdir / LEADERBOARD_FILENAME
    stage_summary_path = outdir / STAGE_SUMMARY_FILENAME
    layer_failure_path = outdir / LAYER_FAILURE_FILENAME
    run_summary_path = outdir / RUN_SUMMARY_FILENAME

    if not manifest_path.exists():
        raise RuntimeError(f"Manifest not found: {manifest_path}")

    ensure_dir(outdir)
    started = time.perf_counter()

    jobs = read_manifest_slice(manifest_path=manifest_path, start=args.start, end=args.end)
    grouped = group_jobs_by_feature_path(jobs)

    print("=" * 120)
    print(f"[START] version={VERSION}")
    print(f"[START] manifest={manifest_path}")
    print(f"[START] outdir={outdir}")
    print(f"[START] jobs={len(jobs)}")
    print(f"[START] feature_cache_groups={len(grouped)}")
    print("=" * 120)

    processed = 0
    all_rows: List[Dict[str, Any]] = []

    for group_idx, (feature_cache_path, group_jobs) in enumerate(grouped.items(), start=1):
        print(f"[LOAD] {group_idx}/{len(grouped)} feature_cache={feature_cache_path} jobs={len(group_jobs)}")
        df = load_feature_frame(feature_cache_path)
        arrays = build_arrays(df)
        print(f"[LOAD-DONE] rows={df.height} columns={len(df.columns)}")

        for job in group_jobs:
            row = debug_one_job(job=job, a=arrays)
            row_dict = asdict(row)
            append_jsonl(job_debug_path, row_dict)
            all_rows.append(row_dict)

            processed += 1
            if processed % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                rate = processed / elapsed if elapsed > 0 else 0.0
                print(f"[PROGRESS] processed={processed}/{len(jobs)} elapsed_sec={elapsed:.2f} jobs_per_sec={rate:.2f}")

    result_df = pl.DataFrame(all_rows)

    leaderboard_df = result_df.sort(
        by=[
            "seed_candidate_flag",
            "estimated_trade_count",
            "relaxed_entry_only_exec_count",
            "relaxed_no_regime_exec_count",
            "relaxed_no_vol_exec_count",
            "relaxed_no_trend_exec_count",
            "executable_total_count",
            "final_total_count",
            "base_total_count",
        ],
        descending=[True, True, True, True, True, True, True, True, True],
    )
    leaderboard_df.write_csv(leaderboard_path)

    stage_summary = build_stage_summary(result_df)
    layer_failure_counts = build_layer_failure_counts(result_df)

    write_json(stage_summary_path, stage_summary)
    write_json(layer_failure_path, layer_failure_counts)

    total_elapsed_sec = round(time.perf_counter() - started, 4)
    run_summary = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "manifest": str(manifest_path),
        "outdir": str(outdir),
        "start": args.start,
        "end": args.end,
        "jobs": len(jobs),
        "feature_cache_groups": len(grouped),
        "job_debug_path": str(job_debug_path),
        "leaderboard_path": str(leaderboard_path),
        "stage_summary_path": str(stage_summary_path),
        "layer_failure_path": str(layer_failure_path),
        "total_elapsed_sec": total_elapsed_sec,
    }
    write_json(run_summary_path, run_summary)

    print("=" * 120)
    print(f"[DONE] job_debug_rows={job_debug_path}")
    print(f"[DONE] leaderboard={leaderboard_path}")
    print(f"[DONE] stage_summary={stage_summary_path}")
    print(f"[DONE] layer_failure_counts={layer_failure_path}")
    print(f"[DONE] run_summary={run_summary_path}")
    print(f"[DONE] processed_jobs={processed}")
    print(f"[DONE] total_elapsed_sec={total_elapsed_sec:.2f}")
    print("=" * 120)


if __name__ == "__main__":
    main()
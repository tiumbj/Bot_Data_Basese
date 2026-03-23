# ==================================================================================================
# FILE: run_vectorbt_seed_validation_pullback_trend_continuation_family_v1_0_0.py
# PATH: C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_seed_validation_pullback_trend_continuation_family_v1_0_0.py
# VERSION: v1.0.0
#
# CHANGELOG:
# - v1.0.0
#   1) New production-grade research runner for pullback trend continuation family on M30
#   2) Keep stable argparse contract using args.feature_cache
#   3) Replace failed breakout family with pullback-to-EMA20 continuation entry structure
#   4) Long setup:
#      - bull_stack = True
#      - close > EMA20
#      - pullback low touches EMA20 zone
#      - close reclaims above EMA20
#      - ADX >= 23
#      - MID_VOL or HIGH_VOL
#      - RSI in 48-64
#      - distance from EMA20 <= 0.80 ATR
#   5) Short setup:
#      - bear_stack = True
#      - close < EMA20
#      - pullback high touches EMA20 zone
#      - close reclaims below EMA20
#      - ADX >= 23
#      - MID_VOL or HIGH_VOL
#      - RSI in 36-52
#      - distance from EMA20 <= 0.80 ATR
#   6) Keep exit architecture:
#      - ATR trailing stop
#      - trend failure exit
#      - structure failure exit
#   7) Keep output contract:
#      - vectorbt_pullback_trend_continuation_family_v1_0_0_results.jsonl
#      - vectorbt_pullback_trend_continuation_family_v1_0_0_results.csv
#      - vectorbt_pullback_trend_continuation_family_v1_0_0_top20.csv
#      - vectorbt_pullback_trend_continuation_family_v1_0_0_summary.json
#      - vectorbt_pullback_trend_continuation_family_v1_0_0_recommendation.txt
# ==================================================================================================

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
import vectorbt as vbt

VERSION = "v1.0.0"
DEFAULT_PROGRESS_EVERY = 25

OUT_RESULTS_JSONL = "vectorbt_pullback_trend_continuation_family_v1_0_0_results.jsonl"
OUT_RESULTS_CSV = "vectorbt_pullback_trend_continuation_family_v1_0_0_results.csv"
OUT_TOP20_CSV = "vectorbt_pullback_trend_continuation_family_v1_0_0_top20.csv"
OUT_SUMMARY_JSON = "vectorbt_pullback_trend_continuation_family_v1_0_0_summary.json"
OUT_RECOMMENDATION_TXT = "vectorbt_pullback_trend_continuation_family_v1_0_0_recommendation.txt"

PROMOTION_MIN_TRADES = 5
PROMOTION_MIN_PF = 1.05
PROMOTION_MIN_RETURN_PCT = 0.0
PROMOTION_MAX_DD_PCT = 8.0
PROMOTION_MIN_EXPECTANCY = 0.0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def normalize_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def normalize_key(value: Any, default: str = "") -> str:
    text = normalize_text(value, default=default)
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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
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
        self.time = pd.to_datetime(df["time"].to_list())
        self.open = df["open"].to_numpy().astype(np.float64, copy=False)
        self.high = df["high"].to_numpy().astype(np.float64, copy=False)
        self.low = df["low"].to_numpy().astype(np.float64, copy=False)
        self.close = df["close"].to_numpy().astype(np.float64, copy=False)

        self.ema_20 = df["ema_20"].to_numpy().astype(np.float64, copy=False)
        self.ema_50 = df["ema_50"].to_numpy().astype(np.float64, copy=False)
        self.atr_14 = df["atr_14"].to_numpy().astype(np.float64, copy=False)
        self.adx_14 = df["adx_14"].to_numpy().astype(np.float64, copy=False)
        self.rsi_14 = df["rsi_14"].to_numpy().astype(np.float64, copy=False)
        self.return_1 = df["return_1"].to_numpy().astype(np.float64, copy=False)

        self.bull_stack = df["bull_stack"].to_numpy().astype(bool, copy=False)
        self.bear_stack = df["bear_stack"].to_numpy().astype(bool, copy=False)

        self.swing_high_10 = df["swing_high_10"].to_numpy().astype(np.float64, copy=False)
        self.swing_low_10 = df["swing_low_10"].to_numpy().astype(np.float64, copy=False)

        self.vol_bucket = np.array(df["vol_bucket"].to_list(), dtype=object)
        self.trend_bucket = np.array(df["trend_bucket"].to_list(), dtype=object)
        self.price_location_bucket = np.array(df["price_location_bucket"].to_list(), dtype=object)

        self.base_true = np.ones(self.row_count, dtype=bool)

        self.range_abs = np.abs(self.high - self.low)
        self.body_abs = np.abs(self.close - self.open)
        self.upper_wick = self.high - np.maximum(self.open, self.close)
        self.lower_wick = np.minimum(self.open, self.close) - self.low

        self.body_ratio = np.zeros(self.row_count, dtype=np.float64)
        np.divide(
            self.body_abs,
            self.range_abs,
            out=self.body_ratio,
            where=self.range_abs > 1e-12,
        )

        self.distance_from_ema20_abs = np.abs(self.close - self.ema_20)

        self.mask_low_vol = self.vol_bucket == "LOW_VOL"
        self.mask_mid_vol = self.vol_bucket == "MID_VOL"
        self.mask_high_vol = self.vol_bucket == "HIGH_VOL"

        self.mask_weak_trend = self.trend_bucket == "WEAK_TREND"
        self.mask_mid_trend = self.trend_bucket == "MID_TREND"
        self.mask_strong_trend = self.trend_bucket == "STRONG_TREND"

        self.mask_above_ema = self.price_location_bucket == "ABOVE_EMA_STACK"
        self.mask_below_ema = self.price_location_bucket == "BELOW_EMA_STACK"
        self.mask_near_ema = self.price_location_bucket == "NEAR_EMA_STACK"

        self.pullback_band = self._pullback_band()
        self.long_pullback_base = self._build_long_pullback_base()
        self.short_pullback_base = self._build_short_pullback_base()

    def _pullback_band(self) -> np.ndarray:
        return np.maximum(self.atr_14 * 0.30, self.close * 0.0005)

    def _build_long_pullback_base(self) -> np.ndarray:
        valid = self.atr_14 > 1e-12
        trend_ok = self.bull_stack & (self.close > self.ema_20)
        pullback_touch = self.low <= (self.ema_20 + self.pullback_band)
        reclaim_ok = self.close >= self.ema_20
        adx_ok = self.adx_14 >= 23.0
        vol_ok = self.mask_mid_vol | self.mask_high_vol
        rsi_ok = (self.rsi_14 >= 48.0) & (self.rsi_14 <= 64.0)
        not_extended = self.distance_from_ema20_abs <= (0.80 * self.atr_14)
        bullish_reaction = (self.close > self.open) & (self.body_ratio >= 0.30)
        lower_wick_ok = self.lower_wick >= 0.10 * self.range_abs
        momentum_ok = self.return_1 > -0.0015

        return (
            valid
            & trend_ok
            & pullback_touch
            & reclaim_ok
            & adx_ok
            & vol_ok
            & rsi_ok
            & not_extended
            & bullish_reaction
            & lower_wick_ok
            & momentum_ok
        )

    def _build_short_pullback_base(self) -> np.ndarray:
        valid = self.atr_14 > 1e-12
        trend_ok = self.bear_stack & (self.close < self.ema_20)
        pullback_touch = self.high >= (self.ema_20 - self.pullback_band)
        reclaim_ok = self.close <= self.ema_20
        adx_ok = self.adx_14 >= 23.0
        vol_ok = self.mask_mid_vol | self.mask_high_vol
        rsi_ok = (self.rsi_14 >= 36.0) & (self.rsi_14 <= 52.0)
        not_extended = self.distance_from_ema20_abs <= (0.80 * self.atr_14)
        bearish_reaction = (self.close < self.open) & (self.body_ratio >= 0.30)
        upper_wick_ok = self.upper_wick >= 0.10 * self.range_abs
        momentum_ok = self.return_1 < 0.0015

        return (
            valid
            & trend_ok
            & pullback_touch
            & reclaim_ok
            & adx_ok
            & vol_ok
            & rsi_ok
            & not_extended
            & bearish_reaction
            & upper_wick_ok
            & momentum_ok
        )


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
            jobs.append(
                JobSpec(
                    job_id=normalize_text(first_present(raw, ["job_id", "id", "name", "strategy_id"], f"job_{idx:06d}")),
                    timeframe=normalize_key(first_present(raw, ["timeframe", "tf"], "M30"), "M30"),
                    strategy_family=normalize_key(
                        first_present(raw, ["strategy_family", "strategy", "strategy_id"], "PULLBACK_TREND_CONTINUATION_FAMILY_V1_0_0"),
                        "PULLBACK_TREND_CONTINUATION_FAMILY_V1_0_0",
                    ),
                    entry_logic=normalize_key(
                        first_present(raw, ["entry_logic", "entry", "entry_type"], "PULLBACK_TREND_CONTINUATION"),
                        "PULLBACK_TREND_CONTINUATION",
                    ),
                    micro_exit=normalize_key(first_present(raw, ["micro_exit", "exit_logic", "exit"], "LET_WINNERS_RUN"), "LET_WINNERS_RUN"),
                    regime_filter=normalize_key(first_present(raw, ["regime_filter", "regime", "market_regime"], "ALL"), "ALL"),
                    cooldown_bars=safe_int(first_present(raw, ["cooldown_bars", "cooldown", "entry_cooldown_bars"], 0), 0),
                    side_policy=normalize_key(first_present(raw, ["side_policy", "side", "trade_side"], "BOTH"), "BOTH"),
                    volatility_filter=normalize_key(first_present(raw, ["volatility_filter", "vol_filter", "vol_bucket"], "ALL"), "ALL"),
                    trend_strength_filter=normalize_key(first_present(raw, ["trend_strength_filter", "trend_filter", "trend_bucket"], "ALL"), "ALL"),
                )
            )

    if not jobs:
        raise RuntimeError(f"No jobs loaded from manifest: {manifest_path}")
    return jobs


def load_feature_cache(feature_cache_path: Path) -> FeatureArrays:
    if not feature_cache_path.exists():
        raise RuntimeError(f"Feature cache not found: {feature_cache_path}")

    df = pl.read_parquet(feature_cache_path)

    required_cols = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "ema_20",
        "ema_50",
        "atr_14",
        "adx_14",
        "rsi_14",
        "return_1",
        "bull_stack",
        "bear_stack",
        "vol_bucket",
        "trend_bucket",
        "price_location_bucket",
        "swing_high_10",
        "swing_low_10",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Feature cache missing columns: {missing}")

    return FeatureArrays(df)


def apply_cooldown(mask: np.ndarray, cooldown_bars: int) -> np.ndarray:
    if cooldown_bars <= 0:
        return mask.copy()

    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.zeros(mask.shape[0], dtype=bool)

    keep_idx: List[int] = []
    last_kept = -10**18
    for i in idx:
        if i - last_kept > cooldown_bars:
            keep_idx.append(int(i))
            last_kept = int(i)

    out = np.zeros(mask.shape[0], dtype=bool)
    out[np.array(keep_idx, dtype=np.int64)] = True
    return out


def side_list(side_policy: str) -> List[str]:
    key = normalize_key(side_policy, "BOTH")
    if key in ("LONG_ONLY", "LONG", "BUY_ONLY", "BUY"):
        return ["LONG"]
    if key in ("SHORT_ONLY", "SHORT", "SELL_ONLY", "SELL"):
        return ["SHORT"]
    return ["LONG", "SHORT"]


def regime_mask(arr: FeatureArrays, regime_filter: str) -> np.ndarray:
    key = normalize_key(regime_filter, "ALL")
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
    return arr.base_true


def volatility_mask(arr: FeatureArrays, volatility_filter: str) -> np.ndarray:
    key = normalize_key(volatility_filter, "ALL")
    if key in ("ALL", "", "NONE"):
        return arr.base_true
    if key == "LOW_VOL":
        return arr.mask_low_vol
    if key == "MID_VOL":
        return arr.mask_mid_vol
    if key == "HIGH_VOL":
        return arr.mask_high_vol
    if key == "NOT_LOW_VOL":
        return arr.mask_mid_vol | arr.mask_high_vol
    return arr.base_true


def trend_strength_mask(arr: FeatureArrays, trend_strength_filter: str) -> np.ndarray:
    key = normalize_key(trend_strength_filter, "ALL")
    if key in ("ALL", "", "NONE"):
        return arr.base_true
    if key == "WEAK_TREND":
        return arr.mask_weak_trend
    if key == "MID_TREND":
        return arr.mask_mid_trend
    if key == "STRONG_TREND":
        return arr.mask_strong_trend
    if key == "MID_OR_STRONG":
        return arr.mask_mid_trend | arr.mask_strong_trend
    if key == "TREND_23":
        return arr.adx_14 >= 23.0
    if key == "TREND_25":
        return arr.adx_14 >= 25.0
    return arr.base_true


def entry_mask(arr: FeatureArrays, side: str) -> np.ndarray:
    if side == "LONG":
        return arr.long_pullback_base
    if side == "SHORT":
        return arr.short_pullback_base
    raise RuntimeError(f"Unsupported side: {side}")


def build_side_entries(job: JobSpec, arr: FeatureArrays, side: str) -> np.ndarray:
    mask = (
        regime_mask(arr, job.regime_filter)
        & volatility_mask(arr, job.volatility_filter)
        & trend_strength_mask(arr, job.trend_strength_filter)
        & entry_mask(arr, side)
    )
    return apply_cooldown(mask, job.cooldown_bars)


def micro_exit_profile(micro_exit: str) -> Tuple[float, float]:
    key = normalize_key(micro_exit, "LET_WINNERS_RUN")
    if "FAST_INVALIDATION" in key:
        return 0.90, 0.20
    if "MOMENTUM_FADE" in key:
        return 1.00, 0.25
    if "LET_WINNERS_RUN" in key:
        return 1.20, 0.35
    if "MICRO_EXIT" in key:
        return 1.00, 0.25
    return 1.20, 0.35


def build_exit_mask(job: JobSpec, arr: FeatureArrays, side: str) -> np.ndarray:
    _, failure_mult = micro_exit_profile(job.micro_exit)

    if side == "LONG":
        structure_fail = arr.close < (arr.ema_20 - failure_mult * arr.atr_14)
        trend_fail = (arr.close < arr.ema_20) & (arr.rsi_14 < 47.0)
        hard_fail = arr.close < (arr.swing_low_10 - 0.20 * arr.atr_14)
        reversal = arr.bear_stack
        return structure_fail | trend_fail | hard_fail | reversal

    if side == "SHORT":
        structure_fail = arr.close > (arr.ema_20 + failure_mult * arr.atr_14)
        trend_fail = (arr.close > arr.ema_20) & (arr.rsi_14 > 53.0)
        hard_fail = arr.close > (arr.swing_high_10 + 0.20 * arr.atr_14)
        reversal = arr.bull_stack
        return structure_fail | trend_fail | hard_fail | reversal

    raise RuntimeError(f"Unsupported side: {side}")


def build_sl_stop(job: JobSpec, arr: FeatureArrays) -> pd.Series:
    trail_mult, _ = micro_exit_profile(job.micro_exit)
    stop_pct = np.zeros(arr.row_count, dtype=np.float64)
    np.divide(
        trail_mult * arr.atr_14,
        arr.close,
        out=stop_pct,
        where=np.abs(arr.close) > 1e-12,
    )
    stop_pct = np.clip(stop_pct, 0.0010, 0.20)
    return pd.Series(stop_pct, index=arr.time, dtype=np.float64)


def portfolio_metrics(
    price: pd.Series,
    entries: np.ndarray,
    exits: np.ndarray,
    sl_stop: pd.Series,
    direction: str,
) -> Dict[str, Any]:
    entry_series = pd.Series(entries, index=price.index, dtype=bool)
    exit_series = pd.Series(exits, index=price.index, dtype=bool)

    if direction == "LONG":
        pf = vbt.Portfolio.from_signals(
            close=price,
            entries=entry_series,
            exits=exit_series,
            sl_stop=sl_stop,
            sl_trail=True,
            init_cash=100000.0,
            fees=0.0005,
            slippage=0.0002,
            freq=None,
        )
    elif direction == "SHORT":
        pf = vbt.Portfolio.from_signals(
            close=price,
            short_entries=entry_series,
            short_exits=exit_series,
            sl_stop=sl_stop,
            sl_trail=True,
            init_cash=100000.0,
            fees=0.0005,
            slippage=0.0002,
            freq=None,
        )
    else:
        raise RuntimeError(f"Unsupported direction: {direction}")

    closed_trades = pf.trades.closed
    trades = int(safe_float(closed_trades.count()))
    total_return_pct = safe_float(pf.total_return()) * 100.0
    max_drawdown_pct = abs(safe_float(pf.max_drawdown()) * 100.0)
    win_rate_pct = safe_float(closed_trades.win_rate()) * 100.0 if trades > 0 else 0.0
    profit_factor = safe_float(closed_trades.profit_factor()) if trades > 0 else 0.0
    expectancy = safe_float(closed_trades.expectancy()) if trades > 0 else 0.0
    avg_trade_return_pct = safe_float(closed_trades.returns.mean()) * 100.0 if trades > 0 else 0.0
    total_profit = safe_float(pf.total_profit())

    return {
        "trade_count": trades,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_trade_return_pct": avg_trade_return_pct,
        "total_profit": total_profit,
    }


def is_promoted(row: Dict[str, Any]) -> bool:
    return (
        int(row["trade_count"]) >= PROMOTION_MIN_TRADES
        and float(row["profit_factor"]) > PROMOTION_MIN_PF
        and float(row["total_return_pct"]) > PROMOTION_MIN_RETURN_PCT
        and float(row["max_drawdown_pct"]) <= PROMOTION_MAX_DD_PCT
        and float(row["expectancy"]) > PROMOTION_MIN_EXPECTANCY
    )


def evaluate_job(job: JobSpec, arr: FeatureArrays, price: pd.Series, sl_stop: pd.Series) -> Dict[str, Any]:
    side_rows: List[Dict[str, Any]] = []

    for side in side_list(job.side_policy):
        entries = build_side_entries(job, arr, side)
        exits = build_exit_mask(job, arr, side)

        entry_count = int(np.count_nonzero(entries))
        if entry_count == 0:
            side_rows.append(
                {
                    "side": side,
                    "entry_count": 0,
                    "trade_count": 0,
                    "total_return_pct": 0.0,
                    "max_drawdown_pct": 0.0,
                    "win_rate_pct": 0.0,
                    "profit_factor": 0.0,
                    "expectancy": 0.0,
                    "avg_trade_return_pct": 0.0,
                    "total_profit": 0.0,
                }
            )
            continue

        metrics = portfolio_metrics(price, entries, exits, sl_stop, side)
        metrics["side"] = side
        metrics["entry_count"] = entry_count
        side_rows.append(metrics)

    best = sorted(
        side_rows,
        key=lambda x: (
            safe_float(x["profit_factor"]),
            safe_float(x["total_return_pct"]),
            -safe_float(x["max_drawdown_pct"]),
            safe_float(x["trade_count"]),
            safe_float(x["total_profit"]),
        ),
        reverse=True,
    )[0]

    score = (
        safe_float(best["total_return_pct"]) * 1.00
        + safe_float(best["profit_factor"]) * 25.00
        + safe_float(best["win_rate_pct"]) * 0.10
        - safe_float(best["max_drawdown_pct"]) * 3.00
        + min(20.0, safe_float(best["trade_count"])) * 0.50
    )

    row = {
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
        "selected_side": best["side"],
        "entry_count": int(best["entry_count"]),
        "trade_count": int(best["trade_count"]),
        "total_return_pct": round(safe_float(best["total_return_pct"]), 6),
        "max_drawdown_pct": round(safe_float(best["max_drawdown_pct"]), 6),
        "win_rate_pct": round(safe_float(best["win_rate_pct"]), 6),
        "profit_factor": round(safe_float(best["profit_factor"]), 6),
        "expectancy": round(safe_float(best["expectancy"]), 6),
        "avg_trade_return_pct": round(safe_float(best["avg_trade_return_pct"]), 6),
        "total_profit": round(safe_float(best["total_profit"]), 6),
        "score": round(score, 6),
        "status": "PASS" if int(best["trade_count"]) > 0 else "NO_TRADES",
    }
    row["promoted"] = is_promoted(row)
    return row


def build_recommendation(df: pl.DataFrame, summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"VERSION: {VERSION}")
    lines.append("PURPOSE: Pullback trend continuation family validation recommendation")
    lines.append("")
    lines.append(f"TOTAL_JOBS: {summary['total_jobs']}")
    lines.append(f"PROMOTED_JOBS: {summary['promoted_jobs']}")
    lines.append(f"PASS_JOBS: {summary['pass_jobs']}")
    lines.append(f"NO_TRADE_JOBS: {summary['no_trade_jobs']}")
    lines.append(f"BEST_JOB_ID: {summary['best_job_id']}")
    lines.append("")
    lines.append("TOP 10 JOBS:")

    top_rows = df.sort("score", descending=True).head(10).to_dicts()
    for idx, row in enumerate(top_rows, start=1):
        lines.append(
            f"- #{idx} job_id={row['job_id']} tf={row['timeframe']} "
            f"side={row['selected_side']} trades={row['trade_count']} "
            f"ret%={row['total_return_pct']} pf={row['profit_factor']} "
            f"dd%={row['max_drawdown_pct']} promoted={row['promoted']}"
        )

    lines.append("")
    lines.append("DECISION:")
    if int(summary["promoted_jobs"]) > 0:
        lines.append("- Pullback trend continuation family has promotable jobs on M30.")
        lines.append("- Promote only rows that pass promotion gate.")
    else:
        lines.append("- Pullback trend continuation family still has no promotable jobs on M30.")
        lines.append("- If pullback continuation also fails, next layer must add regime/session intelligence.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VectorBT validation for pullback trend continuation family v1.0.0")
    parser.add_argument("--manifest", required=True, help="Input manifest JSONL")
    parser.add_argument("--feature-cache", required=True, dest="feature_cache", help="Feature cache parquet")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY, help="Progress interval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    feature_cache_path = Path(args.feature_cache)
    outdir = Path(args.outdir)
    progress_every = max(1, int(args.progress_every))

    ensure_dir(outdir)

    out_jsonl = outdir / OUT_RESULTS_JSONL
    out_csv = outdir / OUT_RESULTS_CSV
    out_top20 = outdir / OUT_TOP20_CSV
    out_summary = outdir / OUT_SUMMARY_JSON
    out_reco = outdir / OUT_RECOMMENDATION_TXT

    reset_file(out_jsonl)

    jobs = load_jobs(manifest_path)
    arr = load_feature_cache(feature_cache_path)
    price = pd.Series(arr.close, index=arr.time, name="close")
    sl_stop = build_sl_stop(JobSpec("", "", "", "", "LET_WINNERS_RUN", "", 0, "", "", ""), arr)

    print(f"[LOAD] manifest={manifest_path}")
    print(f"[LOAD] feature_cache={feature_cache_path}")
    print(f"[LOAD-DONE] jobs={len(jobs)} rows={arr.row_count}")

    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []

    for idx, job in enumerate(jobs, start=1):
        row = evaluate_job(job, arr, price, sl_stop)
        rows.append(row)
        append_jsonl(out_jsonl, row)

        if idx % progress_every == 0 or idx == len(jobs):
            elapsed = time.perf_counter() - started
            jobs_per_sec = idx / elapsed if elapsed > 0 else 0.0
            print(
                f"[PROGRESS] processed={idx}/{len(jobs)} "
                f"elapsed_sec={elapsed:.2f} jobs_per_sec={jobs_per_sec:.2f}"
            )

    df = pl.DataFrame(rows)
    df.write_csv(out_csv)
    df.sort("score", descending=True).head(20).write_csv(out_top20)

    total_jobs = int(df.height)
    promoted_jobs = int(df.filter(pl.col("promoted") == True).height)
    pass_jobs = int(df.filter(pl.col("trade_count") > 0).height)
    no_trade_jobs = int(df.filter(pl.col("trade_count") <= 0).height)
    best_job_id = str(df.sort("score", descending=True).select("job_id").item(0, 0)) if total_jobs > 0 else ""

    summary = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "manifest_path": str(manifest_path),
        "feature_cache_path": str(feature_cache_path),
        "outdir": str(outdir),
        "total_jobs": total_jobs,
        "promoted_jobs": promoted_jobs,
        "pass_jobs": pass_jobs,
        "no_trade_jobs": no_trade_jobs,
        "top_score": round(safe_float(df.select(pl.col("score").max()).item()), 6) if total_jobs > 0 else 0.0,
        "best_job_id": best_job_id,
        "mean_trade_count": round(safe_float(df.select(pl.col("trade_count").mean()).item()), 6) if total_jobs > 0 else 0.0,
        "mean_profit_factor": round(safe_float(df.select(pl.col("profit_factor").mean()).item()), 6) if total_jobs > 0 else 0.0,
        "mean_total_return_pct": round(safe_float(df.select(pl.col("total_return_pct").mean()).item()), 6) if total_jobs > 0 else 0.0,
        "mean_max_drawdown_pct": round(safe_float(df.select(pl.col("max_drawdown_pct").mean()).item()), 6) if total_jobs > 0 else 0.0,
        "elapsed_sec": round(time.perf_counter() - started, 4),
    }

    write_json(out_summary, summary)
    write_text(out_reco, build_recommendation(df, summary))

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] results_jsonl={out_jsonl}")
    print(f"[DONE] results_csv={out_csv}")
    print(f"[DONE] top20_csv={out_top20}")
    print(f"[DONE] summary_json={out_summary}")
    print(f"[DONE] recommendation_txt={out_reco}")
    print(f"[DONE] total_jobs={summary['total_jobs']}")
    print(f"[DONE] promoted_jobs={summary['promoted_jobs']}")
    print(f"[DONE] pass_jobs={summary['pass_jobs']}")
    print(f"[DONE] no_trade_jobs={summary['no_trade_jobs']}")
    print(f"[DONE] best_job_id={summary['best_job_id']}")
    print("=" * 120)


if __name__ == "__main__":
    main()
# ==================================================================================================
# FILE: run_vectorbt_seed_validation.py
# PATH: C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_seed_validation.py
# VERSION: v1.0.1
#
# CHANGELOG:
# - v1.0.1
#   1) Fix VectorBT API usage: trades.returns is a property, not a callable
#   2) Use pf.trades.closed for trade metrics to avoid mixing open + closed trades
#   3) Keep output contract and CLI identical to v1.0.0
#
# PURPOSE:
# - Take a debug manifest shard and run true VectorBT validation on each job
# - Rank jobs by pnl/profit_factor/drawdown/trade_count
# - Produce seed shortlist for the next full validation phase
# ==================================================================================================

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import polars as pl
import vectorbt as vbt

VERSION = "v1.0.1"
DEFAULT_PROGRESS_EVERY = 25

OUT_RESULTS_JSONL = "vectorbt_seed_results.jsonl"
OUT_RESULTS_CSV = "vectorbt_seed_results.csv"
OUT_SUMMARY_JSON = "vectorbt_seed_summary.json"
OUT_TOP20_CSV = "vectorbt_seed_top20.csv"
OUT_RECOMMENDATION_TXT = "vectorbt_seed_recommendation.txt"


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
        near = self.mask_near_ema | (self.close <= (self.ema_20 + self.atr_14 * 0.20))
        momentum_ok = self.return_1 > -0.01
        return self.bull_stack & near & momentum_ok

    def _build_short_pullback_deep(self) -> np.ndarray:
        near = self.mask_near_ema | (self.close >= (self.ema_20 - self.atr_14 * 0.20))
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

            jobs.append(
                JobSpec(
                    job_id=normalize_text(first_present(raw, ["job_id", "id", "name", "strategy_id"], f"job_{idx:06d}")),
                    timeframe=normalize_key(first_present(raw, ["timeframe", "tf"], "M1"), "M1"),
                    strategy_family=normalize_key(first_present(raw, ["strategy_family", "strategy", "strategy_id"], "UNKNOWN"), "UNKNOWN"),
                    entry_logic=normalize_key(first_present(raw, ["entry_logic", "entry", "entry_type"], "DEFAULT"), "DEFAULT"),
                    micro_exit=normalize_key(first_present(raw, ["micro_exit", "exit_logic", "exit"], "NONE"), "NONE"),
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

    required_cols = [
        "time", "close", "ema_9", "ema_20", "ema_50", "ema_200",
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
    if key == "TREND_20":
        return arr.adx_14 >= 20.0
    if key == "TREND_25":
        return arr.adx_14 >= 25.0
    return arr.base_true


def entry_mask(arr: FeatureArrays, entry_logic: str, side: str) -> np.ndarray:
    key = normalize_key(entry_logic, "DEFAULT")

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

    raise RuntimeError(f"Unsupported side: {side}")


def build_side_entries(job: JobSpec, arr: FeatureArrays, side: str) -> np.ndarray:
    mask = (
        regime_mask(arr, job.regime_filter)
        & volatility_mask(arr, job.volatility_filter)
        & trend_strength_mask(arr, job.trend_strength_filter)
        & entry_mask(arr, job.entry_logic, side)
    )
    return apply_cooldown(mask, job.cooldown_bars)


def build_exit_mask(arr: FeatureArrays, side: str, micro_exit: str) -> np.ndarray:
    key = normalize_key(micro_exit, "NONE")

    if side == "LONG":
        base_exit = arr.bear_stack | (arr.close < arr.ema_20)
        if "FAST_INVALIDATION" in key:
            return base_exit | (arr.close < (arr.ema_20 - arr.atr_14 * 0.30)) | (arr.rsi_14 < 45.0)
        if "MOMENTUM_FADE" in key:
            return base_exit | (arr.rsi_14 < 50.0) | (arr.return_1 < -0.0025)
        if "MICRO_EXIT" in key:
            return base_exit | (arr.close < arr.ema_9)
        return base_exit

    if side == "SHORT":
        base_exit = arr.bull_stack | (arr.close > arr.ema_20)
        if "FAST_INVALIDATION" in key:
            return base_exit | (arr.close > (arr.ema_20 + arr.atr_14 * 0.30)) | (arr.rsi_14 > 55.0)
        if "MOMENTUM_FADE" in key:
            return base_exit | (arr.rsi_14 > 50.0) | (arr.return_1 > 0.0025)
        if "MICRO_EXIT" in key:
            return base_exit | (arr.close > arr.ema_9)
        return base_exit

    raise RuntimeError(f"Unsupported side: {side}")


def portfolio_metrics(
    price: pd.Series,
    entries: np.ndarray,
    exits: np.ndarray,
    direction: str,
) -> Dict[str, Any]:
    entry_series = pd.Series(entries, index=price.index, dtype=bool)
    exit_series = pd.Series(exits, index=price.index, dtype=bool)

    if direction == "LONG":
        pf = vbt.Portfolio.from_signals(
            close=price,
            entries=entry_series,
            exits=exit_series,
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
    max_dd_pct = safe_float(pf.max_drawdown()) * 100.0
    win_rate_pct = safe_float(closed_trades.win_rate()) * 100.0 if trades > 0 else 0.0
    profit_factor = safe_float(closed_trades.profit_factor()) if trades > 0 else 0.0
    expectancy = safe_float(closed_trades.expectancy()) if trades > 0 else 0.0
    avg_trade_return_pct = safe_float(closed_trades.returns.mean()) * 100.0 if trades > 0 else 0.0
    total_profit = safe_float(pf.total_profit())

    return {
        "trade_count": trades,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd_pct,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_trade_return_pct": avg_trade_return_pct,
        "total_profit": total_profit,
    }


def evaluate_job(job: JobSpec, arr: FeatureArrays, price: pd.Series) -> Dict[str, Any]:
    result_rows: List[Dict[str, Any]] = []

    for side in side_list(job.side_policy):
        entries = build_side_entries(job, arr, side)
        exits = build_exit_mask(arr, side, job.micro_exit)

        entry_count = int(np.count_nonzero(entries))
        if entry_count == 0:
            result_rows.append(
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

        metrics = portfolio_metrics(price, entries, exits, side)
        metrics["side"] = side
        metrics["entry_count"] = entry_count
        result_rows.append(metrics)

    best = sorted(
        result_rows,
        key=lambda x: (
            safe_float(x["total_profit"]),
            safe_float(x["profit_factor"]),
            -abs(safe_float(x["max_drawdown_pct"])),
            safe_float(x["trade_count"]),
        ),
        reverse=True,
    )[0]

    score = (
        safe_float(best["total_return_pct"]) * 1.0
        + safe_float(best["profit_factor"]) * 15.0
        + safe_float(best["win_rate_pct"]) * 0.2
        - max(0.0, safe_float(best["max_drawdown_pct"])) * 1.5
        + min(200.0, safe_float(best["trade_count"])) * 0.05
    )

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
        "status": "PASS" if best["trade_count"] > 0 else "NO_TRADES",
    }


def build_recommendation(df: pl.DataFrame, summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"VERSION: {VERSION}")
    lines.append("PURPOSE: VectorBT seed validation recommendation")
    lines.append("")
    lines.append(f"TOTAL_JOBS: {summary['total_jobs']}")
    lines.append(f"PASS_JOBS: {summary['pass_jobs']}")
    lines.append(f"NO_TRADES: {summary['no_trade_jobs']}")
    lines.append(f"TOP_SCORE: {summary['top_score']}")
    lines.append("")

    top_rows = df.sort("score", descending=True).head(10).to_dicts()
    lines.append("TOP 10 JOBS:")
    for idx, row in enumerate(top_rows, start=1):
        lines.append(
            f"- #{idx} job_id={row['job_id']} tf={row['timeframe']} strategy={row['strategy_family']} "
            f"entry={row['entry_logic']} side={row['selected_side']} trades={row['trade_count']} "
            f"ret%={row['total_return_pct']} pf={row['profit_factor']} dd%={row['max_drawdown_pct']} score={row['score']}"
        )

    lines.append("")
    lines.append("DECISION:")
    if summary["pass_jobs"] == 0:
        lines.append("- No promotable jobs. Tighten entry realism or exit logic before next batch.")
    else:
        lines.append("- Promote top-ranked jobs to the next validation phase.")
        lines.append("- Use profit factor, drawdown, and trade count together. Do not rank by return only.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VectorBT seed validation on a manifest shard.")
    parser.add_argument("--manifest", required=True, help="Input manifest JSONL")
    parser.add_argument("--feature-cache", required=True, help="Feature cache parquet")
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
    out_summary = outdir / OUT_SUMMARY_JSON
    out_top20 = outdir / OUT_TOP20_CSV
    out_reco = outdir / OUT_RECOMMENDATION_TXT

    reset_file(out_jsonl)

    jobs = load_jobs(manifest_path)
    arr = load_feature_cache(feature_cache_path)

    price = pd.Series(arr.close, index=arr.time, name="close")

    print(f"[LOAD] manifest={manifest_path}")
    print(f"[LOAD] feature_cache={feature_cache_path}")
    print(f"[LOAD-DONE] jobs={len(jobs)} rows={arr.row_count}")

    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []

    for idx, job in enumerate(jobs, start=1):
        row = evaluate_job(job, arr, price)
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

    pass_jobs = int(df.filter(pl.col("trade_count") > 0).height)
    no_trade_jobs = int(df.filter(pl.col("trade_count") <= 0).height)

    summary = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "manifest_path": str(manifest_path),
        "feature_cache_path": str(feature_cache_path),
        "outdir": str(outdir),
        "total_jobs": int(df.height),
        "pass_jobs": pass_jobs,
        "no_trade_jobs": no_trade_jobs,
        "top_score": round(safe_float(df.select(pl.col("score").max()).item()), 6),
        "best_job_id": str(df.sort("score", descending=True).select("job_id").item(0, 0)) if df.height > 0 else "",
        "mean_trade_count": round(safe_float(df.select(pl.col("trade_count").mean()).item()), 6),
        "mean_profit_factor": round(safe_float(df.select(pl.col("profit_factor").mean()).item()), 6),
        "mean_total_return_pct": round(safe_float(df.select(pl.col("total_return_pct").mean()).item()), 6),
        "mean_max_drawdown_pct": round(safe_float(df.select(pl.col("max_drawdown_pct").mean()).item()), 6),
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
    print(f"[DONE] pass_jobs={summary['pass_jobs']}")
    print(f"[DONE] no_trade_jobs={summary['no_trade_jobs']}")
    print(f"[DONE] best_job_id={summary['best_job_id']}")
    print("=" * 120)


if __name__ == "__main__":
    main()
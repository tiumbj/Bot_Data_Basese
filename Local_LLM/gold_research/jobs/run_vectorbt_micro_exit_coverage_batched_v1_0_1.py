#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Code Name : run_vectorbt_micro_exit_coverage_batched_v1_0_1.py
Version   : v1.0.1
Path      : C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_micro_exit_coverage_batched_v1_0_1.py
Run       : python C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_micro_exit_coverage_batched_v1_0_1.py --manifest C:\Data\Bot\central_backtest_results\research_coverage_master_v1_0_0\research_coverage_master_manifest.csv --data-root C:\Data\Bot\central_market_data\parquet --feature-root C:\Data\Bot\central_feature_cache --outdir C:\Data\Bot\central_backtest_results\coverage_master_runs_v1_0_0\micro_exit_expansion_batched_v1_0_1\run_0 --phase micro_exit_expansion --portfolio-chunk-size 64 --progress-every-groups 1 --continue-on-error

Purpose
-------
Compatibility-safe batched coverage runner for the componentized micro-exit manifest.
This version does not require legacy `logic_variant` in the manifest. Instead it reads
component fields directly, builds a canonical logic key internally, groups jobs by
(timeframe, strategy_family, logic components, EMA pair), then evaluates each job batch
against shared entry signals and per-job exit rules.

Scope
-----
1) Fix schema mismatch between coverage manifest and legacy runner.
2) Preserve one-process batched execution style.
3) Keep resume/log/progress structure simple and production-usable.
4) Support the currently observed manifest family:
   - strategy_family = adx_ema_trend_continuation
   - entry_variant   = confirm_entry
   - swing_variant   = long / short
   - pullback_zone_variant = narrow / medium / wide
   - logic_strictness = loose / medium / strict (or synonyms)
   - micro_exit_variant = adx_fade_exit / fast_invalidation / momentum_fade / no_micro_exit

Notes
-----
- This file is intentionally self-contained.
- It prefers compatibility and clean failure reporting over hidden assumptions.
- It will log unsupported component combinations explicitly to bootstrap.log.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

VERSION = "v1.0.1"
FINAL_DONE = "DONE"
FINAL_DONE_WITH_ERRORS = "DONE_WITH_ERRORS"
FINAL_FAILED = "FAILED"

TIMEFRAME_ORDER = {
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

STRICTNESS_THRESHOLD = {
    "loose": 18.0,
    "light": 18.0,
    "medium": 25.0,
    "mid": 25.0,
    "strict": 30.0,
    "hard": 30.0,
}

PULLBACK_RATIO_BANDS = {
    "narrow": (0.00, 0.35),
    "medium": (0.15, 0.70),
    "wide": (0.35, 1.25),
}

DEFAULT_RESULT_COLUMNS = [
    "job_id",
    "manifest_id",
    "manifest_rank",
    "row_index",
    "symbol",
    "timeframe",
    "strategy_family",
    "logic_variant",
    "logic_strictness",
    "swing_variant",
    "pullback_zone_variant",
    "entry_variant",
    "micro_exit_variant",
    "management_variant",
    "regime_variant",
    "robustness_variant",
    "side",
    "ema_fast",
    "ema_slow",
    "entry_count",
    "trade_count",
    "wins",
    "losses",
    "win_rate_pct",
    "pnl_sum",
    "avg_pnl",
    "profit_factor",
    "payoff_ratio",
    "max_consecutive_losses",
    "avg_bars_held",
    "median_bars_held",
    "first_trade_time",
    "last_trade_time",
    "status",
    "status_reason",
]


@dataclass(frozen=True)
class Job:
    row_index: int
    manifest_rank: int
    manifest_id: str
    symbol: str
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
    phase: str
    logic_variant: str
    source_row: Dict[str, str]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_text(path: Path, line: str) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def write_bootstrap_log(path: Path, message: str) -> None:
    append_text(path, f"{now_utc_iso()} {message}")


def chunked(seq: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def hash_job_id(manifest_id: str, micro_exit_variant: str) -> str:
    raw = f"{manifest_id}__{micro_exit_variant}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def sanitize_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def to_int(value: object, field_name: str) -> int:
    text = sanitize_text(value)
    if not text:
        raise ValueError(f"missing {field_name}")
    try:
        return int(float(text))
    except Exception as exc:
        raise ValueError(f"invalid {field_name}={text}") from exc


def canonical_logic_variant(strategy_family: str, logic_strictness: str, swing_variant: str, pullback_zone_variant: str, entry_variant: str) -> str:
    return "__".join(
        [
            sanitize_text(strategy_family, "unknown_family"),
            sanitize_text(logic_strictness, "unknown_strictness"),
            sanitize_text(swing_variant, "unknown_swing"),
            sanitize_text(pullback_zone_variant, "unknown_pullback"),
            sanitize_text(entry_variant, "unknown_entry"),
        ]
    )


def normalize_job_row(row_index: int, row: Dict[str, str], default_symbol: str, default_phase: str) -> Job:
    timeframe = sanitize_text(row.get("timeframe"))
    if not timeframe:
        raise ValueError("missing timeframe")

    strategy_family = sanitize_text(row.get("strategy_family"), "micro_exit_expansion")
    logic_strictness = sanitize_text(row.get("logic_strictness")).lower()
    swing_variant = sanitize_text(row.get("swing_variant")).lower()
    pullback_zone_variant = sanitize_text(row.get("pullback_zone_variant")).lower()
    entry_variant = sanitize_text(row.get("entry_variant")).lower()
    micro_exit_variant = sanitize_text(row.get("micro_exit_variant"), "no_micro_exit").lower()
    management_variant = sanitize_text(row.get("management_variant"), "no_extra_management").lower()
    regime_variant = sanitize_text(row.get("regime_variant"), "")
    robustness_variant = sanitize_text(row.get("robustness_variant"), "")
    symbol = sanitize_text(row.get("symbol"), default_symbol)
    phase = sanitize_text(row.get("phase"), default_phase)
    manifest_id = sanitize_text(row.get("manifest_id"), f"row_{row_index}")
    manifest_rank = to_int(row.get("manifest_rank", row_index + 1), "manifest_rank")
    ema_fast = to_int(row.get("ema_fast"), "ema_fast")
    ema_slow = to_int(row.get("ema_slow"), "ema_slow")

    if not logic_strictness:
        raise ValueError("missing logic_strictness")
    if not swing_variant:
        raise ValueError("missing swing_variant")
    if not pullback_zone_variant:
        raise ValueError("missing pullback_zone_variant")
    if not entry_variant:
        raise ValueError("missing entry_variant")
    if ema_fast <= 0 or ema_slow <= 0:
        raise ValueError("ema periods must be positive")
    if ema_fast >= ema_slow:
        raise ValueError(f"invalid ema pair fast={ema_fast} slow={ema_slow}")

    logic_variant = canonical_logic_variant(strategy_family, logic_strictness, swing_variant, pullback_zone_variant, entry_variant)

    return Job(
        row_index=row_index,
        manifest_rank=manifest_rank,
        manifest_id=manifest_id,
        symbol=symbol,
        timeframe=timeframe,
        strategy_family=strategy_family,
        logic_strictness=logic_strictness,
        swing_variant=swing_variant,
        pullback_zone_variant=pullback_zone_variant,
        entry_variant=entry_variant,
        micro_exit_variant=micro_exit_variant,
        management_variant=management_variant,
        regime_variant=regime_variant,
        robustness_variant=robustness_variant,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        phase=phase,
        logic_variant=logic_variant,
        source_row=dict(row),
    )


def read_manifest_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_price_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    needed = ["time", "open", "high", "low", "close"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"price parquet missing columns={missing} path={path}")
    out = df[needed].copy()
    out["time"] = pd.to_datetime(out["time"], utc=False, errors="coerce")
    out = out.dropna(subset=["time"]).reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return out


def read_feature_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=False, errors="coerce")
    return df.reset_index(drop=True)


def align_feature_series(price_df: pd.DataFrame, feat_df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in feat_df.columns:
        s = pd.to_numeric(feat_df[column], errors="coerce")
        s = s.reindex(range(len(price_df)))
        return s.fillna(default).astype(float)
    return pd.Series(np.full(len(price_df), default, dtype=float), index=price_df.index)


def make_ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean().astype(float)


def bars_since(condition: pd.Series) -> pd.Series:
    values = condition.fillna(False).to_numpy(dtype=bool)
    out = np.full(len(values), np.nan, dtype=float)
    last_true = -1
    for i, flag in enumerate(values):
        if flag:
            last_true = i
            out[i] = 0.0
        elif last_true >= 0:
            out[i] = float(i - last_true)
    return pd.Series(out, index=condition.index)


def build_component_entries(
    feat_df: pd.DataFrame,
    close: pd.Series,
    ema_fast: pd.Series,
    ema_slow: pd.Series,
    strategy_family: str,
    logic_strictness: str,
    swing_variant: str,
    pullback_zone_variant: str,
    entry_variant: str,
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    if strategy_family != "adx_ema_trend_continuation":
        raise ValueError(f"unsupported strategy_family={strategy_family}")
    if entry_variant != "confirm_entry":
        raise ValueError(f"unsupported entry_variant={entry_variant}")
    if swing_variant not in {"long", "short"}:
        raise ValueError(f"unsupported swing_variant={swing_variant}")
    if pullback_zone_variant not in PULLBACK_RATIO_BANDS:
        raise ValueError(f"unsupported pullback_zone_variant={pullback_zone_variant}")

    adx_threshold = STRICTNESS_THRESHOLD.get(logic_strictness)
    if adx_threshold is None:
        raise ValueError(f"unsupported logic_strictness={logic_strictness}")

    adx = align_feature_series(price_df=pd.DataFrame(index=close.index), feat_df=feat_df, column="adx_14", default=0.0)
    ema_spread = (ema_slow - ema_fast).abs().replace(0.0, np.nan)
    ratio_lo, ratio_hi = PULLBACK_RATIO_BANDS[pullback_zone_variant]
    bars_above_fast = bars_since(close > ema_fast)
    bars_below_fast = bars_since(close < ema_fast)

    if swing_variant == "long":
        trend_ok = (ema_fast > ema_slow) & (adx >= adx_threshold)
        pullback_depth = (ema_fast - close) / ema_spread
        pullback_ok = (
            close <= ema_fast
        ) & (
            close >= ema_slow
        ) & (
            pullback_depth >= ratio_lo
        ) & (
            pullback_depth <= ratio_hi
        )
        reclaim = (close > ema_fast) & (close.shift(1) <= ema_fast.shift(1))
        recent_pullback = (bars_below_fast <= 6).fillna(False)
        signal = (trend_ok & reclaim & recent_pullback & pullback_ok.shift(1).fillna(False)).fillna(False)
        prev = signal.shift(1, fill_value=False).astype(bool)
        return (signal & (~prev)).astype(bool), None

    trend_ok = (ema_fast < ema_slow) & (adx >= adx_threshold)
    pullback_depth = (close - ema_fast) / ema_spread
    pullback_ok = (
        close >= ema_fast
    ) & (
        close <= ema_slow
    ) & (
        pullback_depth >= ratio_lo
    ) & (
        pullback_depth <= ratio_hi
    )
    reclaim = (close < ema_fast) & (close.shift(1) >= ema_fast.shift(1))
    recent_pullback = (bars_above_fast <= 6).fillna(False)
    signal = (trend_ok & reclaim & recent_pullback & pullback_ok.shift(1).fillna(False)).fillna(False)
    prev = signal.shift(1, fill_value=False).astype(bool)
    return None, (signal & (~prev)).astype(bool)


def build_exit_series(
    micro_exit_variant: str,
    side: str,
    close: pd.Series,
    ema_fast: pd.Series,
    ema_slow: pd.Series,
    adx: pd.Series,
) -> pd.Series:
    side = side.upper()
    variant = micro_exit_variant.lower()

    if variant in {"", "none", "no_micro_exit", "hold_until_end"}:
        return pd.Series(False, index=close.index, dtype=bool)

    if variant == "adx_fade_exit":
        if side == "LONG":
            return ((adx < 20.0) | (close < ema_fast)).fillna(False).astype(bool)
        return ((adx < 20.0) | (close > ema_fast)).fillna(False).astype(bool)

    if variant == "fast_invalidation":
        if side == "LONG":
            return (close < ema_fast).fillna(False).astype(bool)
        return (close > ema_fast).fillna(False).astype(bool)

    if variant == "momentum_fade":
        fast_slope = ema_fast.diff()
        slow_slope = ema_slow.diff()
        if side == "LONG":
            return ((fast_slope <= 0.0) | (close < ema_fast) | (slow_slope <= 0.0)).fillna(False).astype(bool)
        return ((fast_slope >= 0.0) | (close > ema_fast) | (slow_slope >= 0.0)).fillna(False).astype(bool)

    raise ValueError(f"unsupported micro_exit_variant={micro_exit_variant}")


def simulate_trades(
    time_index: pd.Series,
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    side: str,
) -> Dict[str, object]:
    entry_idx = np.flatnonzero(entries.fillna(False).to_numpy(dtype=bool))
    exit_flags = exits.fillna(False).to_numpy(dtype=bool)
    close_np = close.to_numpy(dtype=float)
    time_np = pd.to_datetime(time_index).to_numpy()

    trade_pnls: List[float] = []
    bars_held: List[int] = []
    wins = 0
    losses = 0
    max_consecutive_losses = 0
    current_loss_streak = 0
    first_trade_time: Optional[str] = None
    last_trade_time: Optional[str] = None

    for i in entry_idx:
        if i >= len(close_np) - 1:
            continue
        start = i + 1
        j = start
        while j < len(close_np) - 1 and not exit_flags[j]:
            j += 1
        exit_i = j if j < len(close_np) else len(close_np) - 1
        if exit_i <= i:
            continue

        entry_price = close_np[start]
        exit_price = close_np[exit_i]
        pnl = (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)
        trade_pnls.append(float(pnl))
        bars_held.append(int(max(exit_i - start, 0)))

        entry_time = pd.Timestamp(time_np[start]).isoformat() if start < len(time_np) else None
        exit_time = pd.Timestamp(time_np[exit_i]).isoformat() if exit_i < len(time_np) else None
        if first_trade_time is None:
            first_trade_time = entry_time
        last_trade_time = exit_time

        if pnl > 0:
            wins += 1
            current_loss_streak = 0
        elif pnl < 0:
            losses += 1
            current_loss_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_loss_streak)

    trade_count = len(trade_pnls)
    pnl_sum = float(np.sum(trade_pnls)) if trade_pnls else 0.0
    avg_pnl = float(np.mean(trade_pnls)) if trade_pnls else 0.0
    gross_profit = float(np.sum([x for x in trade_pnls if x > 0]))
    gross_loss_abs = abs(float(np.sum([x for x in trade_pnls if x < 0])))
    profit_factor = gross_profit / gross_loss_abs if gross_loss_abs > 0 else (999.0 if gross_profit > 0 else 0.0)
    avg_win = float(np.mean([x for x in trade_pnls if x > 0])) if wins > 0 else 0.0
    avg_loss_abs = abs(float(np.mean([x for x in trade_pnls if x < 0]))) if losses > 0 else 0.0
    payoff_ratio = avg_win / avg_loss_abs if avg_loss_abs > 0 else (999.0 if avg_win > 0 else 0.0)
    win_rate_pct = (wins / trade_count * 100.0) if trade_count else 0.0
    avg_bars_held = float(np.mean(bars_held)) if bars_held else 0.0
    median_bars_held = float(np.median(bars_held)) if bars_held else 0.0

    return {
        "entry_count": int(entries.fillna(False).sum()),
        "trade_count": trade_count,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate_pct, 4),
        "pnl_sum": round(pnl_sum, 8),
        "avg_pnl": round(avg_pnl, 8),
        "profit_factor": round(profit_factor, 8),
        "payoff_ratio": round(payoff_ratio, 8),
        "max_consecutive_losses": max_consecutive_losses,
        "avg_bars_held": round(avg_bars_held, 4),
        "median_bars_held": round(median_bars_held, 4),
        "first_trade_time": first_trade_time or "",
        "last_trade_time": last_trade_time or "",
    }


def build_live_progress(
    version: str,
    outdir: Path,
    phase: str,
    current_timeframe: str,
    overall_total: int,
    overall_completed: int,
    overall_failed: int,
    overall_skipped: int,
    start_ts: float,
) -> Dict[str, object]:
    elapsed_min = (time.perf_counter() - start_ts) / 60.0
    progress_pct = (overall_completed / overall_total * 100.0) if overall_total else 100.0
    rate_per_min = (overall_completed / elapsed_min) if elapsed_min > 0 else 0.0
    remaining = max(overall_total - overall_completed, 0)
    eta_min = (remaining / rate_per_min) if rate_per_min > 0 else None
    payload = {
        "version": version,
        "phase": phase,
        "current_timeframe": current_timeframe,
        "overall_total": overall_total,
        "overall_completed": overall_completed,
        "overall_failed": overall_failed,
        "overall_skipped": overall_skipped,
        "overall_remaining": remaining,
        "overall_progress_pct": round(progress_pct, 4),
        "observed_elapsed_min": round(elapsed_min, 4),
        "observed_rate_jobs_per_min": round(rate_per_min, 4),
        "overall_eta_remaining_min": round(eta_min, 4) if eta_min is not None else None,
        "updated_at_utc": now_utc_iso(),
        "outdir": str(outdir),
    }
    save_json(outdir / "live_progress.json", payload)
    return payload


def append_frame_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    write_header = not path.exists()
    frame.to_csv(path, mode="a", index=False, header=write_header, encoding="utf-8")


def load_completed_ids(result_csv: Path) -> set[str]:
    if not result_csv.exists():
        return set()
    try:
        df = pd.read_csv(result_csv, usecols=["job_id"])
        return set(df["job_id"].dropna().astype(str).tolist())
    except Exception:
        return set()


def evaluate_group_batch(
    time_index: pd.Series,
    close: pd.Series,
    adx: pd.Series,
    ema_fast: pd.Series,
    ema_slow: pd.Series,
    jobs: Sequence[Job],
    side: str,
    entries: pd.Series,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for job in jobs:
        job_id = hash_job_id(job.manifest_id, job.micro_exit_variant)
        try:
            exits = build_exit_series(job.micro_exit_variant, side, close, ema_fast, ema_slow, adx)
            stats = simulate_trades(time_index, close, entries, exits, side)
            rows.append(
                {
                    "job_id": job_id,
                    "manifest_id": job.manifest_id,
                    "manifest_rank": job.manifest_rank,
                    "row_index": job.row_index,
                    "symbol": job.symbol,
                    "timeframe": job.timeframe,
                    "strategy_family": job.strategy_family,
                    "logic_variant": job.logic_variant,
                    "logic_strictness": job.logic_strictness,
                    "swing_variant": job.swing_variant,
                    "pullback_zone_variant": job.pullback_zone_variant,
                    "entry_variant": job.entry_variant,
                    "micro_exit_variant": job.micro_exit_variant,
                    "management_variant": job.management_variant,
                    "regime_variant": job.regime_variant,
                    "robustness_variant": job.robustness_variant,
                    "side": side,
                    "ema_fast": job.ema_fast,
                    "ema_slow": job.ema_slow,
                    **stats,
                    "status": "DONE",
                    "status_reason": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "job_id": job_id,
                    "manifest_id": job.manifest_id,
                    "manifest_rank": job.manifest_rank,
                    "row_index": job.row_index,
                    "symbol": job.symbol,
                    "timeframe": job.timeframe,
                    "strategy_family": job.strategy_family,
                    "logic_variant": job.logic_variant,
                    "logic_strictness": job.logic_strictness,
                    "swing_variant": job.swing_variant,
                    "pullback_zone_variant": job.pullback_zone_variant,
                    "entry_variant": job.entry_variant,
                    "micro_exit_variant": job.micro_exit_variant,
                    "management_variant": job.management_variant,
                    "regime_variant": job.regime_variant,
                    "robustness_variant": job.robustness_variant,
                    "side": side,
                    "ema_fast": job.ema_fast,
                    "ema_slow": job.ema_slow,
                    "entry_count": 0,
                    "trade_count": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate_pct": 0.0,
                    "pnl_sum": 0.0,
                    "avg_pnl": 0.0,
                    "profit_factor": 0.0,
                    "payoff_ratio": 0.0,
                    "max_consecutive_losses": 0,
                    "avg_bars_held": 0.0,
                    "median_bars_held": 0.0,
                    "first_trade_time": "",
                    "last_trade_time": "",
                    "status": "ERROR",
                    "status_reason": f"{type(exc).__name__}:{exc}",
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Component-schema batched micro-exit coverage runner")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--feature-root", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--phase", default="micro_exit_expansion")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--portfolio-chunk-size", type=int, default=64)
    parser.add_argument("--progress-every-groups", type=int, default=1)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    ensure_dir(outdir)
    ensure_dir(outdir / "per_timeframe")
    bootstrap_log_path = outdir / "bootstrap.log"
    stdout_result_csv = outdir / "coverage_results_all.csv"
    summary_json = outdir / "summary.json"
    state_json = outdir / "run_state.json"
    summary_csv = outdir / "summary_by_timeframe.csv"

    start_ts = time.perf_counter()
    started_at_utc = now_utc_iso()
    write_bootstrap_log(bootstrap_log_path, f"START version={VERSION} manifest={args.manifest} phase={args.phase}")

    rows = read_manifest_csv(args.manifest)
    jobs: List[Job] = []
    bad_rows = 0
    for row_index, row in enumerate(rows):
        try:
            job = normalize_job_row(row_index=row_index, row=row, default_symbol=args.symbol, default_phase=args.phase)
            if sanitize_text(job.phase).lower() != sanitize_text(args.phase).lower():
                continue
            jobs.append(job)
        except Exception as exc:
            bad_rows += 1
            write_bootstrap_log(bootstrap_log_path, f"ROW_SKIP row_index={row_index} reason={type(exc).__name__}:{exc}")

    if not jobs:
        payload = {
            "version": VERSION,
            "manifest": str(args.manifest),
            "phase": args.phase,
            "status": FINAL_DONE,
            "started_at_utc": started_at_utc,
            "finished_at_utc": now_utc_iso(),
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "skipped_bad_rows": bad_rows,
        }
        save_json(summary_json, payload)
        save_json(state_json, payload)
        write_bootstrap_log(bootstrap_log_path, "FINISH final_status=DONE total_jobs=0")
        return

    jobs = sorted(jobs, key=lambda j: (TIMEFRAME_ORDER.get(j.timeframe, 99999), j.strategy_family, j.logic_variant, j.ema_fast, j.ema_slow, j.micro_exit_variant, j.manifest_rank))

    grouped: Dict[Tuple[str, str, str, str, str, str, int, int], List[Job]] = {}
    for job in jobs:
        key = (
            job.timeframe,
            job.strategy_family,
            job.logic_strictness,
            job.swing_variant,
            job.pullback_zone_variant,
            job.entry_variant,
            job.ema_fast,
            job.ema_slow,
        )
        grouped.setdefault(key, []).append(job)

    write_bootstrap_log(bootstrap_log_path, f"LOAD_DONE total_jobs={len(jobs)} groups={len(grouped)} skipped_bad_rows={bad_rows}")

    completed_ids = load_completed_ids(stdout_result_csv)
    timeframe_cache: Dict[str, Dict[str, object]] = {}
    timeframe_summary: List[Dict[str, object]] = []
    overall_total = len(jobs)
    overall_completed = 0
    overall_failed = 0
    overall_skipped = bad_rows
    processed_groups = 0
    encountered_error = False

    for group_key, group_jobs in grouped.items():
        processed_groups += 1
        timeframe, strategy_family, logic_strictness, swing_variant, pullback_zone_variant, entry_variant, ema_fast_n, ema_slow_n = group_key
        current_timeframe = timeframe
        try:
            symbol = group_jobs[0].symbol
            if timeframe not in timeframe_cache:
                price_path = args.data_root / f"{symbol}_{timeframe}.parquet"
                feature_path = args.feature_root / f"{symbol}_{timeframe}_base_features.parquet"
                if not price_path.exists():
                    raise FileNotFoundError(f"missing price file: {price_path}")
                if not feature_path.exists():
                    raise FileNotFoundError(f"missing feature file: {feature_path}")
                price_df = read_price_parquet(price_path)
                feat_df = read_feature_parquet(feature_path)
                close = price_df["close"].astype(float)
                adx = align_feature_series(price_df, feat_df, "adx_14", default=0.0)
                timeframe_cache[timeframe] = {
                    "symbol": symbol,
                    "price_df": price_df,
                    "feat_df": feat_df,
                    "close": close,
                    "adx": adx,
                    "ema_cache": {},
                }
                write_bootstrap_log(bootstrap_log_path, f"TIMEFRAME_LOAD timeframe={timeframe} price_path={price_path} feature_path={feature_path} rows={len(price_df)}")

            tfc = timeframe_cache[timeframe]
            ema_cache = tfc["ema_cache"]
            if ema_fast_n not in ema_cache:
                ema_cache[ema_fast_n] = make_ema(tfc["close"], ema_fast_n)
            if ema_slow_n not in ema_cache:
                ema_cache[ema_slow_n] = make_ema(tfc["close"], ema_slow_n)
            ema_fast = ema_cache[ema_fast_n]
            ema_slow = ema_cache[ema_slow_n]

            long_entries, short_entries = build_component_entries(
                feat_df=tfc["feat_df"],
                close=tfc["close"],
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                strategy_family=strategy_family,
                logic_strictness=logic_strictness,
                swing_variant=swing_variant,
                pullback_zone_variant=pullback_zone_variant,
                entry_variant=entry_variant,
            )
            if long_entries is not None:
                side = "LONG"
                entry_series = long_entries
            elif short_entries is not None:
                side = "SHORT"
                entry_series = short_entries
            else:
                raise RuntimeError("unable to determine side from component entries")

            pending_jobs = [j for j in group_jobs if hash_job_id(j.manifest_id, j.micro_exit_variant) not in completed_ids]
            resumed_jobs = len(group_jobs) - len(pending_jobs)
            overall_completed += resumed_jobs

            if not pending_jobs:
                write_bootstrap_log(bootstrap_log_path, f"GROUP_RESUME_SKIP timeframe={timeframe} strategy_family={strategy_family} jobs={len(group_jobs)}")
            else:
                for job_chunk in chunked(pending_jobs, max(1, int(args.portfolio_chunk_size))):
                    result_rows = evaluate_group_batch(
                        time_index=tfc["price_df"]["time"],
                        close=tfc["close"],
                        adx=tfc["adx"],
                        ema_fast=ema_fast,
                        ema_slow=ema_slow,
                        jobs=job_chunk,
                        side=side,
                        entries=entry_series,
                    )
                    frame = pd.DataFrame(result_rows)
                    frame = frame.reindex(columns=DEFAULT_RESULT_COLUMNS)
                    append_frame_csv(stdout_result_csv, frame)
                    tf_result_csv = outdir / "per_timeframe" / f"{symbol}_{timeframe}_micro_exit_coverage_results.csv"
                    append_frame_csv(tf_result_csv, frame)
                    overall_completed += int((frame["status"] == "DONE").sum())
                    overall_failed += int((frame["status"] == "ERROR").sum())
                    for jid in frame["job_id"].astype(str).tolist():
                        completed_ids.add(jid)

            timeframe_summary.append(
                {
                    "timeframe": timeframe,
                    "symbol": symbol,
                    "group_key": "|".join(map(str, group_key)),
                    "group_jobs": len(group_jobs),
                    "pending_jobs": len(pending_jobs),
                    "resumed_jobs": resumed_jobs,
                    "side": side,
                    "entry_count": int(entry_series.sum()),
                    "groups_completed": processed_groups,
                    "groups_total": len(grouped),
                    "status": "DONE",
                }
            )
        except Exception as exc:
            encountered_error = True
            overall_failed += len(group_jobs)
            timeframe_summary.append(
                {
                    "timeframe": timeframe,
                    "symbol": group_jobs[0].symbol if group_jobs else "",
                    "group_key": "|".join(map(str, group_key)),
                    "group_jobs": len(group_jobs),
                    "pending_jobs": len(group_jobs),
                    "resumed_jobs": 0,
                    "side": "",
                    "entry_count": 0,
                    "groups_completed": processed_groups,
                    "groups_total": len(grouped),
                    "status": f"ERROR:{type(exc).__name__}:{exc}",
                }
            )
            write_bootstrap_log(bootstrap_log_path, f"GROUP_ERROR group={group_key} reason={type(exc).__name__}:{exc}")
            if not args.continue_on_error:
                raise

        live = build_live_progress(
            version=VERSION,
            outdir=outdir,
            phase=args.phase,
            current_timeframe=current_timeframe,
            overall_total=overall_total,
            overall_completed=overall_completed,
            overall_failed=overall_failed,
            overall_skipped=overall_skipped,
            start_ts=start_ts,
        )
        if processed_groups % max(1, int(args.progress_every_groups)) == 0 or processed_groups == len(grouped):
            print(
                f"[PROGRESS] groups={processed_groups}/{len(grouped)} "
                f"overall_completed={overall_completed}/{overall_total} "
                f"overall_failed={overall_failed} overall_skipped={overall_skipped} "
                f"overall_progress_pct={live['overall_progress_pct']:.2f} "
                f"overall_eta_remaining_min={live['overall_eta_remaining_min']}"
            )

    summary_frame = pd.DataFrame(timeframe_summary)
    if not summary_frame.empty:
        summary_frame.to_csv(summary_csv, index=False, encoding="utf-8")

    final_status = FINAL_DONE_WITH_ERRORS if encountered_error or overall_failed > 0 else FINAL_DONE
    summary_payload = {
        "version": VERSION,
        "manifest": str(args.manifest),
        "phase": args.phase,
        "started_at_utc": started_at_utc,
        "finished_at_utc": now_utc_iso(),
        "status": final_status,
        "total_jobs": overall_total,
        "completed_jobs": overall_completed,
        "failed_jobs": overall_failed,
        "skipped_bad_rows": overall_skipped,
        "groups_total": len(grouped),
        "groups_completed": processed_groups,
        "coverage_results_all_csv": str(stdout_result_csv),
        "summary_by_timeframe_csv": str(summary_csv),
    }
    save_json(summary_json, summary_payload)
    save_json(state_json, summary_payload)
    write_bootstrap_log(bootstrap_log_path, f"FINISH final_status={final_status} total_jobs={overall_total} completed={overall_completed} failed={overall_failed} skipped_bad_rows={overall_skipped}")


if __name__ == "__main__":
    main()

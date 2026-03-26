#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code Name: run_vectorbt_pending_master_coverage_v1_0_0.py
File Path: C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_pending_master_coverage_v1_0_0.py
Run Command:
    python C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_pending_master_coverage_v1_0_0.py
Version: v1.0.0

Strategy Header
- Purpose:
  Run the next locked phase after EMA baseline:
  1) load top EMA pairs from prior uncovered backtest evidence,
  2) cover pending strategy logic families,
  3) cover micro-exit family variants,
  4) save database-ready outputs,
  5) auto-resume after stop/restart,
  6) keep live progress % and ETA.
- Locked Direction:
  * Use VectorBT as execution/backtest engine.
  * Run automatically across pending/available research timeframes.
  * Record results automatically.
  * Resume automatically from saved state/results.
  * Publish progress and ETA as machine-readable JSON.
- Changelog:
  * v1.0.0
    1) First production-style pending coverage master runner.
    2) Uses prior EMA evidence as shortlist input to avoid exploding job count.
    3) Covers strategy logic families + micro-exit family in one runner.
    4) Saves per-timeframe results, per-timeframe state, and overall live progress.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
except Exception as exc:
    raise SystemExit(
        "VectorBT import failed. Please install vectorbt first. "
        f"Original error: {exc}"
    ) from exc


VERSION = "v1.0.0"
SYMBOL = "XAUUSD"
RESEARCH_TIMEFRAMES = ["M1", "M2", "M3", "M4", "M5", "M6", "M10", "M15", "M30", "H1", "H4", "D1"]

DEFAULT_DATA_ROOT = Path(r"C:\Data\Bot\central_market_data\parquet")
DEFAULT_EMA_SOURCE_DIR = Path(r"C:\Data\Bot\central_backtest_results\vectorbt_uncovered_parallel_v1_1_2")
DEFAULT_OUTDIR = Path(r"C:\Data\Bot\central_backtest_results\vectorbt_pending_master_coverage_v1_0_0")

DEFAULT_TOP_N_EMA = 20
DEFAULT_BATCH_SIZE = 24
INITIAL_CASH = 100_000.0
FEES = 0.0
SLIPPAGE = 0.0

DEFAULT_ADX_WINDOW = 14
DEFAULT_ATR_WINDOW = 14
DEFAULT_ADX_THRESHOLD = 20.0

STRATEGY_FAMILIES = [
    "ema_cross_long_only",
    "ema_cross_short_only",
    "ema_pullback_long_only",
    "ema_pullback_short_only",
    "adx_ema_cross_long_only",
    "adx_ema_cross_short_only",
]

MICRO_EXIT_VARIANTS = [
    "reverse_signal_exit",
    "price_cross_fast_exit",
    "price_cross_slow_exit",
    "adx_fade_exit",
]


@dataclass(frozen=True)
class EmaPair:
    ema_fast: int
    ema_slow: int


@dataclass(frozen=True)
class Job:
    timeframe: str
    ema_fast: int
    ema_slow: int
    strategy_family: str
    micro_exit_variant: str

    @property
    def job_id(self) -> str:
        return (
            f"{SYMBOL}_{self.timeframe}"
            f"_ema_fast_{self.ema_fast:03d}"
            f"_ema_slow_{self.ema_slow:03d}"
            f"_{self.strategy_family}"
            f"_{self.micro_exit_variant}"
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pending strategy/micro-exit coverage with VectorBT, auto-resume, and live progress."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ema-source-dir", type=Path, default=DEFAULT_EMA_SOURCE_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--timeframes", nargs="*", default=RESEARCH_TIMEFRAMES)
    parser.add_argument("--top-n-ema", type=int, default=DEFAULT_TOP_N_EMA)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--initial-cash", type=float, default=INITIAL_CASH)
    parser.add_argument("--fees", type=float, default=FEES)
    parser.add_argument("--slippage", type=float, default=SLIPPAGE)
    parser.add_argument("--adx-window", type=int, default=DEFAULT_ADX_WINDOW)
    parser.add_argument("--atr-window", type=int, default=DEFAULT_ATR_WINDOW)
    parser.add_argument("--adx-threshold", type=float, default=DEFAULT_ADX_THRESHOLD)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def safe_metric(value: object) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def find_price_file(data_root: Path, timeframe: str) -> Path:
    candidates = [
        data_root / f"{SYMBOL}_{timeframe}.parquet",
        data_root / f"{SYMBOL}_{timeframe}.pq",
        data_root / f"{SYMBOL}_{timeframe}.csv",
        data_root / timeframe / f"{SYMBOL}_{timeframe}.parquet",
        data_root / timeframe / f"{SYMBOL}_{timeframe}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Price file not found for timeframe={timeframe}. Checked: " + ", ".join(str(x) for x in candidates)
    )


def load_ohlc_dataframe(price_path: Path) -> pd.DataFrame:
    if price_path.suffix.lower() == ".csv":
        df = pd.read_csv(price_path)
    else:
        df = pd.read_parquet(price_path)

    lower_map = {str(col).lower(): col for col in df.columns}

    def find_col(names: Sequence[str]) -> Optional[str]:
        for name in names:
            if name.lower() in lower_map:
                return lower_map[name.lower()]
        return None

    time_col = find_col(["time", "datetime", "timestamp", "date"])
    open_col = find_col(["open"])
    high_col = find_col(["high"])
    low_col = find_col(["low"])
    close_col = find_col(["close"])

    if close_col is None or high_col is None or low_col is None:
        raise ValueError(f"Missing OHLC columns in {price_path}")

    out = pd.DataFrame({
        "open": pd.to_numeric(df[open_col], errors="coerce") if open_col else pd.to_numeric(df[close_col], errors="coerce"),
        "high": pd.to_numeric(df[high_col], errors="coerce"),
        "low": pd.to_numeric(df[low_col], errors="coerce"),
        "close": pd.to_numeric(df[close_col], errors="coerce"),
    })
    if time_col is not None:
        idx = pd.to_datetime(df[time_col], errors="coerce")
        out.index = idx
        out = out[~out.index.isna()]
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    if out.empty:
        raise ValueError(f"OHLC dataframe empty after cleaning: {price_path}")
    return out.astype(float)


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, window: int) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def adx(df: pd.DataFrame, window: int) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr = true_range(df)
    atr_s = tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    plus_di = 100.0 * (plus_dm.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean() / atr_s.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean() / atr_s.replace(0, np.nan))

    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return dx.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def rank_score(df: pd.DataFrame) -> pd.Series:
    return (
        df["profit_factor"].fillna(0.0) * 1000.0
        + df["expectancy"].fillna(0.0) * 100.0
        + df["pnl_sum"].fillna(0.0)
        - df["max_drawdown"].fillna(0.0) * 100.0
    )


def load_top_ema_pairs_for_timeframe(ema_source_dir: Path, timeframe: str, top_n: int) -> List[EmaPair]:
    candidates = [
        ema_source_dir / "per_timeframe" / f"{SYMBOL}_{timeframe}_vectorbt_uncovered_results.csv",
        ema_source_dir / f"{SYMBOL}_{timeframe}_vectorbt_uncovered_results.csv",
    ]
    source_path = None
    for candidate in candidates:
        if candidate.exists():
            source_path = candidate
            break

    if source_path is None:
        return []

    df = pd.read_csv(source_path)
    if df.empty:
        return []

    required_cols = {"ema_fast", "ema_slow", "profit_factor", "expectancy", "pnl_sum", "max_drawdown"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"EMA source missing columns {missing} in {source_path}")

    df = df.copy()
    df["rank_score"] = rank_score(df)
    df = (
        df.sort_values(
            by=["rank_score", "profit_factor", "expectancy", "pnl_sum"],
            ascending=[False, False, False, False],
        )
        .drop_duplicates(subset=["ema_fast", "ema_slow"])
        .head(top_n)
    )
    pairs = [EmaPair(ema_fast=int(r.ema_fast), ema_slow=int(r.ema_slow)) for r in df.itertuples(index=False)]
    return pairs


def build_jobs(timeframes: Sequence[str], ema_source_dir: Path, top_n_ema: int) -> List[Job]:
    jobs: List[Job] = []
    for timeframe in timeframes:
        pairs = load_top_ema_pairs_for_timeframe(ema_source_dir, timeframe, top_n_ema)
        for pair in pairs:
            for strategy_family in STRATEGY_FAMILIES:
                for micro_exit_variant in MICRO_EXIT_VARIANTS:
                    jobs.append(
                        Job(
                            timeframe=timeframe,
                            ema_fast=pair.ema_fast,
                            ema_slow=pair.ema_slow,
                            strategy_family=strategy_family,
                            micro_exit_variant=micro_exit_variant,
                        )
                    )
    return jobs


def read_completed_job_ids(result_csv: Path) -> Set[str]:
    if not result_csv.exists():
        return set()
    try:
        df = pd.read_csv(result_csv, usecols=["job_id"])
        return set(df["job_id"].astype(str).tolist())
    except Exception:
        return set()


def append_rows_csv(result_csv: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not result_csv.exists()
    df.to_csv(result_csv, mode="a", index=False, header=header)


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def chunked(seq: Sequence[Job], size: int) -> Iterable[List[Job]]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def estimate_eta(elapsed_sec: float, completed_jobs: int, remaining_jobs: int) -> Optional[float]:
    if completed_jobs <= 0 or elapsed_sec <= 0:
        return None
    sec_per_job = elapsed_sec / completed_jobs
    return sec_per_job * remaining_jobs


def format_minutes(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value / 60.0, 2)


def build_manifest(args: argparse.Namespace, timeframes: Sequence[str], outdir: Path, total_jobs: int) -> Dict[str, object]:
    return {
        "version": VERSION,
        "symbol": SYMBOL,
        "timeframes": list(timeframes),
        "ema_source_dir": str(args.ema_source_dir),
        "data_root": str(args.data_root),
        "outdir": str(outdir),
        "top_n_ema": args.top_n_ema,
        "batch_size": args.batch_size,
        "initial_cash": args.initial_cash,
        "fees": args.fees,
        "slippage": args.slippage,
        "adx_window": args.adx_window,
        "atr_window": args.atr_window,
        "adx_threshold": args.adx_threshold,
        "strategy_families": STRATEGY_FAMILIES,
        "micro_exit_variants": MICRO_EXIT_VARIANTS,
        "total_jobs_planned": total_jobs,
        "created_at_utc": utc_now_iso(),
    }


def precompute_context(
    df: pd.DataFrame,
    pair_set: Sequence[EmaPair],
    adx_window: int,
    atr_window: int,
) -> Dict[str, object]:
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema_cache: Dict[int, pd.Series] = {}
    needed_windows = sorted(set([p.ema_fast for p in pair_set] + [p.ema_slow for p in pair_set]))
    for window in needed_windows:
        ema_cache[window] = ema(close, window)

    adx_series = adx(df, adx_window)
    atr_series = atr(df, atr_window)

    return {
        "close": close,
        "high": high,
        "low": low,
        "ema_cache": ema_cache,
        "adx": adx_series,
        "atr": atr_series,
    }


def build_strategy_signals(
    strategy_family: str,
    close: pd.Series,
    ema_fast: pd.Series,
    ema_slow: pd.Series,
    adx_series: pd.Series,
    adx_threshold: float,
) -> Tuple[pd.Series, pd.Series, str]:
    fast_gt_slow = (ema_fast > ema_slow).fillna(False)
    fast_lt_slow = (ema_fast < ema_slow).fillna(False)

    fast_gt_prev = fast_gt_slow.shift(1).fillna(False).astype(bool)
    fast_lt_prev = fast_lt_slow.shift(1).fillna(False).astype(bool)

    close_gt_fast = (close > ema_fast).fillna(False)
    close_lt_fast = (close < ema_fast).fillna(False)
    close_gt_fast_prev = close_gt_fast.shift(1).fillna(False).astype(bool)
    close_lt_fast_prev = close_lt_fast.shift(1).fillna(False).astype(bool)

    adx_ok = (adx_series >= adx_threshold).fillna(False)

    if strategy_family == "ema_cross_long_only":
        entries = fast_gt_slow & (~fast_gt_prev)
        reverse_signal = fast_lt_slow & (~fast_lt_prev)
        side = "longonly"
    elif strategy_family == "ema_cross_short_only":
        entries = fast_lt_slow & (~fast_lt_prev)
        reverse_signal = fast_gt_slow & (~fast_gt_prev)
        side = "shortonly"
    elif strategy_family == "ema_pullback_long_only":
        entries = fast_gt_slow & close_gt_fast & close_lt_fast_prev
        reverse_signal = fast_lt_slow & (~fast_lt_prev)
        side = "longonly"
    elif strategy_family == "ema_pullback_short_only":
        entries = fast_lt_slow & close_lt_fast & close_gt_fast_prev
        reverse_signal = fast_gt_slow & (~fast_gt_prev)
        side = "shortonly"
    elif strategy_family == "adx_ema_cross_long_only":
        entries = fast_gt_slow & (~fast_gt_prev) & adx_ok
        reverse_signal = fast_lt_slow & (~fast_lt_prev)
        side = "longonly"
    elif strategy_family == "adx_ema_cross_short_only":
        entries = fast_lt_slow & (~fast_lt_prev) & adx_ok
        reverse_signal = fast_gt_slow & (~fast_gt_prev)
        side = "shortonly"
    else:
        raise ValueError(f"Unsupported strategy_family={strategy_family}")

    return entries.fillna(False).astype(bool), reverse_signal.fillna(False).astype(bool), side


def build_exit_signals(
    micro_exit_variant: str,
    side: str,
    close: pd.Series,
    ema_fast: pd.Series,
    ema_slow: pd.Series,
    reverse_signal: pd.Series,
    adx_series: pd.Series,
    adx_threshold: float,
) -> pd.Series:
    if micro_exit_variant == "reverse_signal_exit":
        exits = reverse_signal
    elif micro_exit_variant == "price_cross_fast_exit":
        exits = (close < ema_fast) if side == "longonly" else (close > ema_fast)
    elif micro_exit_variant == "price_cross_slow_exit":
        exits = (close < ema_slow) if side == "longonly" else (close > ema_slow)
    elif micro_exit_variant == "adx_fade_exit":
        exits = (adx_series < adx_threshold).fillna(False)
    else:
        raise ValueError(f"Unsupported micro_exit_variant={micro_exit_variant}")

    return exits.fillna(False).astype(bool)


def evaluate_job(
    job: Job,
    context: Dict[str, object],
    initial_cash: float,
    fees: float,
    slippage: float,
    adx_threshold: float,
) -> Dict[str, object]:
    close = context["close"]
    ema_fast = context["ema_cache"][job.ema_fast]
    ema_slow = context["ema_cache"][job.ema_slow]
    adx_series = context["adx"]

    entries, reverse_signal, side = build_strategy_signals(
        strategy_family=job.strategy_family,
        close=close,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        adx_series=adx_series,
        adx_threshold=adx_threshold,
    )
    exits = build_exit_signals(
        micro_exit_variant=job.micro_exit_variant,
        side=side,
        close=close,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        reverse_signal=reverse_signal,
        adx_series=adx_series,
        adx_threshold=adx_threshold,
    )

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=initial_cash,
        fees=fees,
        slippage=slippage,
        direction=side,
        freq=None,
    )

    trade_count = int(safe_metric(pf.trades.count()))
    win_rate_raw = safe_metric(pf.trades.win_rate())

    row = {
        "job_id": job.job_id,
        "symbol": SYMBOL,
        "timeframe": job.timeframe,
        "strategy_family": job.strategy_family,
        "logic_variant": job.strategy_family,
        "ema_fast": job.ema_fast,
        "ema_slow": job.ema_slow,
        "micro_exit_variant": job.micro_exit_variant,
        "trade_count": trade_count,
        "win_rate_pct": win_rate_raw * 100.0 if pd.notna(win_rate_raw) else float("nan"),
        "profit_factor": safe_metric(pf.trades.profit_factor()),
        "expectancy": safe_metric(pf.trades.expectancy()),
        "pnl_sum": safe_metric(pf.total_profit()),
        "max_drawdown": safe_metric(pf.max_drawdown()),
        "avg_win": safe_metric(pf.trades.winning.pnl.mean()),
        "avg_loss": safe_metric(pf.trades.losing.pnl.mean()),
        "side": side,
        "status": "DONE",
        "created_at_utc": utc_now_iso(),
        "regime_summary": "",
    }
    return row


def update_overall_progress(
    live_progress_json: Path,
    total_jobs: int,
    completed_jobs: int,
    current_timeframe: str,
    started_at: float,
    timeframe_status: Dict[str, Dict[str, object]],
    outdir: Path,
) -> None:
    remaining_jobs = max(total_jobs - completed_jobs, 0)
    elapsed_sec = time.time() - started_at
    overall_progress_pct = (completed_jobs / total_jobs * 100.0) if total_jobs > 0 else 0.0
    eta_sec = estimate_eta(elapsed_sec=elapsed_sec, completed_jobs=completed_jobs, remaining_jobs=remaining_jobs)

    progress_obj = {
        "version": VERSION,
        "checked_at_utc": utc_now_iso(),
        "outdir": str(outdir),
        "current_timeframe": current_timeframe,
        "overall_jobs_total": int(total_jobs),
        "overall_jobs_completed": int(completed_jobs),
        "overall_jobs_remaining": int(remaining_jobs),
        "overall_progress_pct": round(overall_progress_pct, 6),
        "observed_elapsed_hours": round(elapsed_sec / 3600.0, 2),
        "rough_eta_remaining_minutes": format_minutes(eta_sec),
        "timeframe_status": timeframe_status,
    }
    write_json(live_progress_json, progress_obj)


def process_timeframe(
    timeframe: str,
    tf_jobs: Sequence[Job],
    args: argparse.Namespace,
    per_tf_dir: Path,
    state_dir: Path,
    timeframe_status: Dict[str, Dict[str, object]],
    live_progress_json: Path,
    total_jobs: int,
    global_completed_before_start: int,
    started_at: float,
    outdir: Path,
) -> Tuple[Dict[str, object], int]:
    result_csv = per_tf_dir / f"{SYMBOL}_{timeframe}_pending_master_coverage_results.csv"
    state_json = state_dir / f"{SYMBOL}_{timeframe}_pending_master_coverage_state.json"

    completed_ids = set()
    if not args.no_resume:
        completed_ids = read_completed_job_ids(result_csv)

    remaining_jobs = [job for job in tf_jobs if job.job_id not in completed_ids]
    price_path = find_price_file(args.data_root, timeframe)
    df = load_ohlc_dataframe(price_path)

    pair_set = sorted({EmaPair(job.ema_fast, job.ema_slow) for job in tf_jobs}, key=lambda x: (x.ema_fast, x.ema_slow))
    context = precompute_context(
        df=df,
        pair_set=pair_set,
        adx_window=args.adx_window,
        atr_window=args.atr_window,
    )

    tf_total = len(tf_jobs)
    processed_now = 0
    rows_written = 0
    tf_start = time.time()

    log(f"[START] timeframe={timeframe} bars={len(df)} jobs_total={tf_total} price_path={price_path}")
    if completed_ids:
        log(f"[RESUME] timeframe={timeframe} already_completed={len(completed_ids)}/{tf_total}")

    if not remaining_jobs:
        tf_summary = {
            "timeframe": timeframe,
            "status": "ALREADY_DONE",
            "jobs_total": tf_total,
            "jobs_completed": tf_total,
            "jobs_remaining": 0,
            "bars": int(len(df)),
            "price_path": str(price_path),
            "elapsed_sec": round(time.time() - tf_start, 3),
            "updated_at_utc": utc_now_iso(),
        }
        write_json(state_json, tf_summary)
        timeframe_status[timeframe] = tf_summary
        update_overall_progress(
            live_progress_json=live_progress_json,
            total_jobs=total_jobs,
            completed_jobs=global_completed_before_start + tf_total,
            current_timeframe=timeframe,
            started_at=started_at,
            timeframe_status=timeframe_status,
            outdir=outdir,
        )
        log(f"[DONE] timeframe={timeframe} status=ALREADY_DONE jobs_total={tf_total}")
        return tf_summary, tf_total

    for batch_index, batch_jobs in enumerate(chunked(remaining_jobs, args.batch_size), start=1):
        batch_rows: List[Dict[str, object]] = []
        for idx, job in enumerate(batch_jobs, start=1):
            batch_rows.append(
                evaluate_job(
                    job=job,
                    context=context,
                    initial_cash=args.initial_cash,
                    fees=args.fees,
                    slippage=args.slippage,
                    adx_threshold=args.adx_threshold,
                )
            )
            if idx == 1 or idx == len(batch_jobs) or idx % 10 == 0:
                log(
                    f"[JOB] timeframe={timeframe} batch={batch_index} processed_in_batch={idx}/{len(batch_jobs)} "
                    f"job_id={job.job_id}"
                )

        append_rows_csv(result_csv, batch_rows)
        rows_written += len(batch_rows)
        processed_now += len(batch_rows)

        tf_completed = len(completed_ids) + processed_now
        tf_remaining = tf_total - tf_completed
        tf_progress_pct = (tf_completed / tf_total * 100.0) if tf_total > 0 else 0.0

        global_completed = global_completed_before_start + processed_now
        update_overall_progress(
            live_progress_json=live_progress_json,
            total_jobs=total_jobs,
            completed_jobs=global_completed,
            current_timeframe=timeframe,
            started_at=started_at,
            timeframe_status=timeframe_status,
            outdir=outdir,
        )

        elapsed_tf = time.time() - tf_start
        tf_eta_sec = estimate_eta(elapsed_sec=elapsed_tf, completed_jobs=max(processed_now, 1), remaining_jobs=tf_remaining)
        tf_state = {
            "version": VERSION,
            "symbol": SYMBOL,
            "timeframe": timeframe,
            "jobs_total": tf_total,
            "jobs_completed": tf_completed,
            "jobs_remaining": tf_remaining,
            "progress_pct": round(tf_progress_pct, 6),
            "rows_written_this_run": rows_written,
            "last_completed_job_id": batch_rows[-1]["job_id"] if batch_rows else "",
            "price_path": str(price_path),
            "bars": int(len(df)),
            "elapsed_hours": round(elapsed_tf / 3600.0, 2),
            "rough_eta_remaining_minutes": format_minutes(tf_eta_sec),
            "updated_at_utc": utc_now_iso(),
            "result_csv": str(result_csv),
        }
        write_json(state_json, tf_state)
        timeframe_status[timeframe] = tf_state

        overall_progress_pct = (global_completed / total_jobs * 100.0) if total_jobs > 0 else 0.0
        overall_eta_sec = estimate_eta(
            elapsed_sec=time.time() - started_at,
            completed_jobs=max(global_completed, 1),
            remaining_jobs=max(total_jobs - global_completed, 0),
        )
        log(
            f"[PROGRESS] timeframe={timeframe} batch={batch_index} "
            f"tf_completed={tf_completed}/{tf_total} tf_progress_pct={tf_progress_pct:.2f} "
            f"overall_completed={global_completed}/{total_jobs} overall_progress_pct={overall_progress_pct:.2f} "
            f"overall_eta_remaining_min={format_minutes(overall_eta_sec)} "
            f"last_job_id={tf_state['last_completed_job_id']}"
        )

    tf_elapsed = time.time() - tf_start
    final_summary = {
        "timeframe": timeframe,
        "status": "DONE",
        "jobs_total": tf_total,
        "jobs_completed": tf_total,
        "jobs_remaining": 0,
        "rows_written_this_run": rows_written,
        "bars": int(len(df)),
        "price_path": str(price_path),
        "elapsed_sec": round(tf_elapsed, 3),
        "updated_at_utc": utc_now_iso(),
        "result_csv": str(result_csv),
    }
    write_json(state_json, final_summary)
    timeframe_status[timeframe] = final_summary
    log(f"[DONE] timeframe={timeframe} jobs_total={tf_total} rows_written_this_run={rows_written} elapsed_sec={tf_elapsed:.2f}")
    return final_summary, processed_now


def build_leaderboards(per_tf_dir: Path, outdir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(per_tf_dir.glob(f"{SYMBOL}_*_pending_master_coverage_results.csv")):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return None, None

    full_df = pd.concat(frames, ignore_index=True)
    full_df["rank_score"] = rank_score(full_df)

    full_path = outdir / "leaderboard_all_results.csv"
    full_df.sort_values(
        by=["rank_score", "profit_factor", "expectancy", "pnl_sum"],
        ascending=[False, False, False, False],
    ).to_csv(full_path, index=False)

    top_frames: List[pd.DataFrame] = []
    for timeframe, tf_df in full_df.groupby("timeframe", sort=True):
        tf_top = tf_df.sort_values(
            by=["rank_score", "profit_factor", "expectancy", "pnl_sum"],
            ascending=[False, False, False, False],
        ).head(50)
        top_frames.append(tf_top)

    top_path = outdir / "leaderboard_top50_per_tf.csv"
    pd.concat(top_frames, ignore_index=True).to_csv(top_path, index=False)
    return full_path, top_path


def main() -> None:
    args = parse_args()
    args.timeframes = [tf.strip() for tf in args.timeframes]

    outdir = args.outdir
    per_tf_dir = outdir / "per_timeframe"
    state_dir = outdir / "state"
    ensure_dir(outdir)
    ensure_dir(per_tf_dir)
    ensure_dir(state_dir)

    all_jobs = build_jobs(
        timeframes=args.timeframes,
        ema_source_dir=args.ema_source_dir,
        top_n_ema=args.top_n_ema,
    )
    if not all_jobs:
        raise SystemExit(
            "No jobs were built. Check EMA source results directory and timeframe list."
        )

    jobs_by_tf: Dict[str, List[Job]] = {}
    for job in all_jobs:
        jobs_by_tf.setdefault(job.timeframe, []).append(job)

    total_jobs = len(all_jobs)
    manifest = build_manifest(args=args, timeframes=list(jobs_by_tf.keys()), outdir=outdir, total_jobs=total_jobs)
    write_json(outdir / "run_manifest.json", manifest)

    live_progress_json = outdir / "live_progress.json"
    timeframe_status: Dict[str, Dict[str, object]] = {}
    total_start = time.time()

    global_completed = 0
    if not args.no_resume:
        for timeframe, tf_jobs in jobs_by_tf.items():
            result_csv = per_tf_dir / f"{SYMBOL}_{timeframe}_pending_master_coverage_results.csv"
            completed_ids = read_completed_job_ids(result_csv)
            already_done = len([job for job in tf_jobs if job.job_id in completed_ids])
            global_completed += already_done
            timeframe_status[timeframe] = {
                "timeframe": timeframe,
                "jobs_total": len(tf_jobs),
                "jobs_completed": already_done,
                "jobs_remaining": len(tf_jobs) - already_done,
                "progress_pct": round((already_done / len(tf_jobs) * 100.0), 6) if tf_jobs else 0.0,
                "status": "PENDING" if already_done < len(tf_jobs) else "DONE",
            }

    update_overall_progress(
        live_progress_json=live_progress_json,
        total_jobs=total_jobs,
        completed_jobs=global_completed,
        current_timeframe="INIT",
        started_at=total_start,
        timeframe_status=timeframe_status,
        outdir=outdir,
    )

    tf_summaries: List[Dict[str, object]] = []
    completed_running = global_completed

    for timeframe in sorted(jobs_by_tf.keys(), key=lambda x: RESEARCH_TIMEFRAMES.index(x) if x in RESEARCH_TIMEFRAMES else 999):
        tf_summary, newly_processed = process_timeframe(
            timeframe=timeframe,
            tf_jobs=jobs_by_tf[timeframe],
            args=args,
            per_tf_dir=per_tf_dir,
            state_dir=state_dir,
            timeframe_status=timeframe_status,
            live_progress_json=live_progress_json,
            total_jobs=total_jobs,
            global_completed_before_start=completed_running,
            started_at=total_start,
            outdir=outdir,
        )
        tf_summaries.append(tf_summary)
        if tf_summary.get("status") == "ALREADY_DONE":
            completed_running += len(jobs_by_tf[timeframe])
        else:
            completed_running += newly_processed

    all_path, top_path = build_leaderboards(per_tf_dir=per_tf_dir, outdir=outdir)
    total_elapsed = time.time() - total_start

    final_summary = {
        "version": VERSION,
        "symbol": SYMBOL,
        "outdir": str(outdir),
        "ema_source_dir": str(args.ema_source_dir),
        "timeframes_processed": list(jobs_by_tf.keys()),
        "jobs_total": total_jobs,
        "jobs_completed": completed_running,
        "overall_progress_pct": round((completed_running / total_jobs * 100.0), 6) if total_jobs > 0 else 0.0,
        "elapsed_sec_total": round(total_elapsed, 3),
        "leaderboard_all_results_csv": str(all_path) if all_path else "",
        "leaderboard_top50_per_tf_csv": str(top_path) if top_path else "",
        "live_progress_json": str(live_progress_json),
        "timeframe_summaries": tf_summaries,
        "finished_at_utc": utc_now_iso(),
    }
    write_json(outdir / "summary.json", final_summary)
    update_overall_progress(
        live_progress_json=live_progress_json,
        total_jobs=total_jobs,
        completed_jobs=completed_running,
        current_timeframe="DONE",
        started_at=total_start,
        timeframe_status=timeframe_status,
        outdir=outdir,
    )

    log("=" * 120)
    log(f"[DONE] version={VERSION}")
    log(f"[DONE] outdir={outdir}")
    log(f"[DONE] total_jobs={total_jobs}")
    log(f"[DONE] jobs_completed={completed_running}")
    if all_path:
        log(f"[DONE] leaderboard_all_results_csv={all_path}")
    if top_path:
        log(f"[DONE] leaderboard_top50_per_tf_csv={top_path}")
    log(f"[DONE] live_progress_json={live_progress_json}")
    log(f"[DONE] summary_json={outdir / 'summary.json'}")
    log("=" * 120)


if __name__ == "__main__":
    main()

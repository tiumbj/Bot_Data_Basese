
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code Name: run_vectorbt_uncovered_parallel_backtests_v1_1_1.py
File Path: C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_uncovered_parallel_backtests_v1_1_1.py
Run Command:
    python C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_uncovered_parallel_backtests_v1_1_1.py
Version: v1.1.1

Strategy Header
- Purpose:
  Run VectorBT uncovered parallel backtests for the still-uncovered timeframes.
- Locked Scope:
  * Timeframes: M2, M3, M4, M6, M30, D1
  * EMA sweep: fast 1-50, slow 20-100, fast < slow
  * Save results in database-ready CSV/JSON outputs
  * Auto resume if stopped and re-run
- Changelog:
  * v1.1.0 -> v1.1.1
    1) Fix tuple/MultiIndex column parsing from vectorbt MA output.
    2) Keep batch + resume architecture.
    3) Improve progress logging and state writing.
    4) Add safer column normalization and robust CSV appends.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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


VERSION = "v1.1.1"
SYMBOL = "XAUUSD"
UNCOVERED_TIMEFRAMES = ["M2", "M3", "M4", "M6", "M30", "D1"]
UNCOVERED_TIMEFRAMES = [x.strip() for x in UNCOVERED_TIMEFRAMES]  # keep formatter happy

DEFAULT_DATA_ROOT = Path(r"C:\Data\Bot\central_market_data\parquet")
DEFAULT_OUTDIR = Path(r"C:\Data\Bot\central_backtest_results\vectorbt_uncovered_parallel_v1_1_1")

FAST_VALUES = list(range(1, 51))
SLOW_VALUES = list(range(20, 101))
INITIAL_CASH = 100_000.0
FEES = 0.0
SLIPPAGE = 0.0

# Windows batch controls
DEFAULT_BATCH_SIZE = 32
DEFAULT_MA_CHUNK_SIZE = 8


@dataclass(frozen=True)
class Job:
    timeframe: str
    ema_fast: int
    ema_slow: int

    @property
    def job_id(self) -> str:
        return f"{SYMBOL}_{self.timeframe}_ema_fast_{self.ema_fast:03d}_ema_slow_{self.ema_slow:03d}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run uncovered VectorBT EMA backtests with auto-resume.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--timeframes", nargs="*", default=UNCOVERED_TIMEFRAMES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--ma-chunk-size", type=int, default=DEFAULT_MA_CHUNK_SIZE)
    parser.add_argument("--initial-cash", type=float, default=INITIAL_CASH)
    parser.add_argument("--fees", type=float, default=FEES)
    parser.add_argument("--slippage", type=float, default=SLIPPAGE)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_jobs_for_timeframe(timeframe: str) -> List[Job]:
    jobs: List[Job] = []
    for fast in FAST_VALUES:
        for slow in SLOW_VALUES:
            if fast < slow:
                jobs.append(Job(timeframe=timeframe, ema_fast=fast, ema_slow=slow))
    return jobs


def find_price_file(data_root: Path, timeframe: str) -> Path:
    candidates = [
        data_root / f"{SYMBOL}_{timeframe}.parquet",
        data_root / f"{SYMBOL}_{timeframe}.pq",
        data_root / f"{SYMBOL}_{timeframe}.csv",
        data_root / timeframe / f"{SYMBOL}_{timeframe}.parquet",
        data_root / timeframe / f"{SYMBOL}_{timeframe}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Price file not found for timeframe={timeframe}. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def load_price_series(price_path: Path) -> pd.Series:
    if price_path.suffix.lower() == ".csv":
        df = pd.read_csv(price_path)
    else:
        df = pd.read_parquet(price_path)

    # normalize columns
    lower_map = {c.lower(): c for c in df.columns}
    time_col = None
    close_col = None
    for candidate in ["time", "datetime", "timestamp", "date"]:
        if candidate in lower_map:
            time_col = lower_map[candidate]
            break
    for candidate in ["close", "Close", "CLOSE"]:
        if candidate.lower() in lower_map:
            close_col = lower_map[candidate.lower()]
            break

    if close_col is None:
        raise ValueError(f"Could not find close column in {price_path}")

    if time_col is not None:
        dt_index = pd.to_datetime(df[time_col], errors="coerce")
        price = pd.Series(df[close_col].astype(float).to_numpy(), index=dt_index, name="close")
        price = price[~price.index.isna()]
    else:
        price = pd.Series(df[close_col].astype(float).to_numpy(), name="close")

    price = price.replace([np.inf, -np.inf], np.nan).dropna()
    if price.empty:
        raise ValueError(f"Price series is empty after cleaning: {price_path}")
    return price.astype(float)


def extract_window_from_column(col: object) -> int:
    """Handle vectorbt MA columns that may be scalar, tuple, or MultiIndex-like."""
    if isinstance(col, (int, np.integer)):
        return int(col)

    if isinstance(col, float):
        return int(col)

    if isinstance(col, str):
        # common cases: "10", "ma_10", "(10,)"
        digits = "".join(ch for ch in col if ch.isdigit())
        if digits:
            return int(digits)
        raise ValueError(f"Cannot parse MA window from string column: {col!r}")

    if isinstance(col, tuple):
        # common vectorbt MultiIndex column tuple forms
        for item in reversed(col):
            try:
                return extract_window_from_column(item)
            except Exception:
                continue
        raise ValueError(f"Cannot parse MA window from tuple column: {col!r}")

    # pandas may pass namedtuple-like or numpy scalar objects
    try:
        return int(col)  # fallback
    except Exception as exc:
        raise ValueError(f"Cannot parse MA window from column: {col!r}") from exc


def normalize_ma_columns(ma_df: pd.DataFrame) -> pd.DataFrame:
    normalized = ma_df.copy()
    normalized.columns = [extract_window_from_column(c) for c in normalized.columns]
    return normalized


def compute_ema_map_vectorbt(
    price: pd.Series,
    windows: Sequence[int],
    ma_chunk_size: int,
) -> Dict[int, pd.Series]:
    unique_windows = sorted(set(int(w) for w in windows))
    ema_map: Dict[int, pd.Series] = {}

    for start in range(0, len(unique_windows), ma_chunk_size):
        chunk = unique_windows[start : start + ma_chunk_size]
        ma_obj = vbt.MA.run(price, window=chunk, ewm=True, short_name="ema")
        ma_df = ma_obj.ma
        if isinstance(ma_df, pd.Series):
            ma_df = ma_df.to_frame()
        ma_df = normalize_ma_columns(ma_df)

        for col in ma_df.columns:
            ema_map[int(col)] = ma_df[col].copy()

    missing = [w for w in unique_windows if w not in ema_map]
    if missing:
        raise RuntimeError(f"EMA windows missing after compute: {missing[:20]}")
    return ema_map


def build_entries_exits(
    price: pd.Series,
    fast_series: pd.Series,
    slow_series: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    long_signal = fast_series > slow_series
    prev_long_signal = long_signal.shift(1).fillna(False)
    entries = long_signal & (~prev_long_signal)
    exits = (~long_signal) & prev_long_signal
    return entries.astype(bool), exits.astype(bool)


def safe_metric(value: object) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def evaluate_job(
    job: Job,
    price: pd.Series,
    ema_map: Dict[int, pd.Series],
    initial_cash: float,
    fees: float,
    slippage: float,
) -> Dict[str, object]:
    fast_series = ema_map[job.ema_fast]
    slow_series = ema_map[job.ema_slow]
    entries, exits = build_entries_exits(price=price, fast_series=fast_series, slow_series=slow_series)

    pf = vbt.Portfolio.from_signals(
        close=price,
        entries=entries,
        exits=exits,
        init_cash=initial_cash,
        fees=fees,
        slippage=slippage,
        direction="longonly",
        freq=None,
    )

    stats = {
        "job_id": job.job_id,
        "symbol": SYMBOL,
        "timeframe": job.timeframe,
        "strategy_family": "ema_filter_uncovered_vbt",
        "logic_variant": "ema_fast_slow_cross_long_only",
        "ema_fast": job.ema_fast,
        "ema_slow": job.ema_slow,
        "micro_exit_variant": "reverse_signal_exit",
        "trade_count": int(safe_metric(pf.trades.count())),
        "win_rate_pct": safe_metric(pf.trades.win_rate()) * 100.0,
        "profit_factor": safe_metric(pf.trades.profit_factor()),
        "expectancy": safe_metric(pf.trades.expectancy()),
        "pnl_sum": safe_metric(pf.total_profit()),
        "max_drawdown": safe_metric(pf.max_drawdown()),
        "avg_win": safe_metric(pf.trades.winning.pnl.mean()),
        "avg_loss": safe_metric(pf.trades.losing.pnl.mean()),
        "status": "DONE",
        "created_at_utc": utc_now_iso(),
        "regime_summary": "",
    }
    return stats


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


def build_manifest(args: argparse.Namespace, outdir: Path) -> Dict[str, object]:
    return {
        "version": VERSION,
        "symbol": SYMBOL,
        "timeframes": list(args.timeframes),
        "data_root": str(args.data_root),
        "outdir": str(outdir),
        "batch_size": args.batch_size,
        "ma_chunk_size": args.ma_chunk_size,
        "initial_cash": args.initial_cash,
        "fees": args.fees,
        "slippage": args.slippage,
        "created_at_utc": utc_now_iso(),
        "locked_scope": {
            "ema_fast_range": [1, 50],
            "ema_slow_range": [20, 100],
            "rule": "fast < slow",
            "micro_exit_variant": "reverse_signal_exit",
        },
    }


def process_timeframe(
    timeframe: str,
    args: argparse.Namespace,
    per_tf_dir: Path,
    state_dir: Path,
) -> Dict[str, object]:
    price_path = find_price_file(args.data_root, timeframe)
    price = load_price_series(price_path)
    jobs = build_jobs_for_timeframe(timeframe)

    result_csv = per_tf_dir / f"{SYMBOL}_{timeframe}_vectorbt_uncovered_results.csv"
    state_json = state_dir / f"{SYMBOL}_{timeframe}_state.json"

    completed_ids: Set[str] = set()
    if not args.no_resume:
        completed_ids = read_completed_job_ids(result_csv)

    total_jobs = len(jobs)
    rows_written = 0
    t0 = time.time()

    log(f"[START] timeframe={timeframe} bars={len(price)} price_path={price_path}")
    if completed_ids:
        log(f"[RESUME] timeframe={timeframe} already_completed={len(completed_ids)}/{total_jobs}")

    processed_now = 0
    remaining_jobs = [job for job in jobs if job.job_id not in completed_ids]

    if not remaining_jobs:
        summary = {
            "timeframe": timeframe,
            "status": "ALREADY_DONE",
            "bars": int(len(price)),
            "jobs_total": total_jobs,
            "jobs_completed": total_jobs,
            "jobs_remaining": 0,
            "elapsed_sec": round(time.time() - t0, 3),
            "result_csv": str(result_csv),
            "price_path": str(price_path),
            "updated_at_utc": utc_now_iso(),
        }
        write_json(state_json, summary)
        log(f"[DONE] timeframe={timeframe} jobs_total={total_jobs} jobs_completed={total_jobs} status=ALREADY_DONE")
        return summary

    batch_index = 0
    for batch_jobs in chunked(remaining_jobs, args.batch_size):
        batch_index += 1
        batch_windows = sorted(set([job.ema_fast for job in batch_jobs] + [job.ema_slow for job in batch_jobs]))
        ema_map = compute_ema_map_vectorbt(price=price, windows=batch_windows, ma_chunk_size=args.ma_chunk_size)

        batch_rows: List[Dict[str, object]] = []
        for job in batch_jobs:
            batch_rows.append(
                evaluate_job(
                    job=job,
                    price=price,
                    ema_map=ema_map,
                    initial_cash=args.initial_cash,
                    fees=args.fees,
                    slippage=args.slippage,
                )
            )

        append_rows_csv(result_csv, batch_rows)
        rows_written += len(batch_rows)
        processed_now += len(batch_rows)

        completed_total = len(completed_ids) + processed_now
        progress_pct = (completed_total / total_jobs) * 100.0
        last_job_id = batch_rows[-1]["job_id"] if batch_rows else ""

        state = {
            "version": VERSION,
            "symbol": SYMBOL,
            "timeframe": timeframe,
            "bars": int(len(price)),
            "price_path": str(price_path),
            "jobs_total": total_jobs,
            "jobs_completed": completed_total,
            "jobs_remaining": total_jobs - completed_total,
            "progress_pct": round(progress_pct, 4),
            "last_completed_job_id": last_job_id,
            "last_batch_index": batch_index,
            "rows_written_this_run": rows_written,
            "result_csv": str(result_csv),
            "updated_at_utc": utc_now_iso(),
        }
        write_json(state_json, state)
        log(
            f"[PROGRESS] timeframe={timeframe} batch={batch_index} batch_jobs={len(batch_rows)} "
            f"completed={completed_total}/{total_jobs} progress_pct={progress_pct:.2f} "
            f"last_job_id={last_job_id}"
        )

    elapsed = time.time() - t0
    final_summary = {
        "timeframe": timeframe,
        "status": "DONE",
        "bars": int(len(price)),
        "jobs_total": total_jobs,
        "jobs_completed": total_jobs,
        "jobs_remaining": 0,
        "rows_written_this_run": rows_written,
        "elapsed_sec": round(elapsed, 3),
        "result_csv": str(result_csv),
        "price_path": str(price_path),
        "updated_at_utc": utc_now_iso(),
    }
    write_json(state_json, final_summary)
    log(
        f"[DONE] timeframe={timeframe} jobs_total={total_jobs} jobs_completed={total_jobs} "
        f"rows_written_this_run={rows_written} elapsed_sec={elapsed:.2f}"
    )
    return final_summary


def build_leaderboard(per_tf_dir: Path, outdir: Path) -> Optional[Path]:
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(per_tf_dir.glob(f"{SYMBOL}_*_vectorbt_uncovered_results.csv")):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return None

    full_df = pd.concat(frames, ignore_index=True)
    # database-ready ranking
    full_df["rank_score"] = (
        full_df["profit_factor"].fillna(0.0) * 1000.0
        + full_df["expectancy"].fillna(0.0) * 100.0
        + full_df["pnl_sum"].fillna(0.0) * 1.0
        - full_df["max_drawdown"].fillna(0.0) * 100.0
    )

    top_frames = []
    for timeframe, tf_df in full_df.groupby("timeframe", sort=True):
        tf_top = tf_df.sort_values(
            by=["rank_score", "profit_factor", "expectancy", "pnl_sum"],
            ascending=[False, False, False, False],
        ).head(50)
        top_frames.append(tf_top)

    leaderboard = pd.concat(top_frames, ignore_index=True)
    leaderboard_path = outdir / "leaderboard_top50_per_tf.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    return leaderboard_path


def main() -> None:
    args = parse_args()

    # fix accidental whitespace safety
    args.timeframes = [tf.strip() for tf in args.timeframes]

    outdir: Path = args.outdir
    per_tf_dir = outdir / "per_timeframe"
    state_dir = outdir / "state"
    ensure_dir(outdir)
    ensure_dir(per_tf_dir)
    ensure_dir(state_dir)

    manifest = build_manifest(args=args, outdir=outdir)
    write_json(outdir / "run_manifest.json", manifest)

    tf_summaries: List[Dict[str, object]] = []
    total_start = time.time()

    for timeframe in args.timeframes:
        tf_summary = process_timeframe(
            timeframe=timeframe,
            args=args,
            per_tf_dir=per_tf_dir,
            state_dir=state_dir,
        )
        tf_summaries.append(tf_summary)

    leaderboard_path = build_leaderboard(per_tf_dir=per_tf_dir, outdir=outdir)

    summary = {
        "version": VERSION,
        "symbol": SYMBOL,
        "timeframes": args.timeframes,
        "timeframes_processed": len(tf_summaries),
        "leaderboard_path": str(leaderboard_path) if leaderboard_path else "",
        "outdir": str(outdir),
        "elapsed_sec_total": round(time.time() - total_start, 3),
        "timeframe_summaries": tf_summaries,
        "finished_at_utc": utc_now_iso(),
    }
    write_json(outdir / "summary.json", summary)

    log("========================================================================================================================")
    log(f"[DONE] version={VERSION}")
    log(f"[DONE] outdir={outdir}")
    if leaderboard_path:
        log(f"[DONE] leaderboard_top50_per_tf={leaderboard_path}")
    log(f"[DONE] summary={outdir / 'summary.json'}")
    log("========================================================================================================================")


if __name__ == "__main__":
    main()

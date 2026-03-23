#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: run_vectorbt_uncovered_parallel_backtests_v1_1_0.py
Version: v1.1.0
Path: C:/Data/Bot/Local_LLM/gold_research/jobs/run_vectorbt_uncovered_parallel_backtests_v1_1_0.py

Purpose:
- Run uncovered research backtests using VectorBT with memory-safe batching.
- Auto-resume if the process stops and is started again.
- Show visible progress while running.
- Write database-ready results incrementally so work is never lost.

Locked coverage in this file:
- Timeframes: M2, M3, M4, M6, M30, D1
- EMA rule: fast 1-50, slow 20-100, fast < slow
- Micro-exit placeholder baseline: reverse_signal_exit

Important design note:
- This file uses VectorBT as the speed layer for uncovered parallel backtests.
- It avoids creating one huge EMA matrix in memory.
- Instead it processes parameter pairs in small batches and writes progress after every batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "[ERROR] vectorbt is not installed. Install it first with: pip install vectorbt"
    ) from exc

VERSION = "v1.1.0"
DEFAULT_TIMEFRAMES = ["M2", "M3", "M4", "M6", "M30", "D1"]
DEFAULT_DATA_ROOT = Path(r"C:\Data\Bot\central_market_data\parquet")
DEFAULT_OUTDIR = Path(r"C:\Data\Bot\central_backtest_results\vectorbt_uncovered_parallel_v1_1_0")
DEFAULT_SYMBOL = "XAUUSD"
DEFAULT_MICRO_EXIT_VARIANT = "reverse_signal_exit"
ALLOWED_EXTENSIONS = (".parquet", ".csv")
RESULT_COLUMNS = [
    "job_id",
    "symbol",
    "timeframe",
    "strategy_family",
    "logic_variant",
    "ema_fast",
    "ema_slow",
    "micro_exit_variant",
    "status",
    "created_at_utc",
    "data_path",
    "total_bars",
    "trade_count",
    "win_rate_pct",
    "profit_factor",
    "expectancy",
    "pnl_sum",
    "max_drawdown",
    "avg_win",
    "avg_loss",
    "sharpe_ratio",
    "score",
    "result_note",
]


@dataclass(frozen=True)
class TimeframeState:
    timeframe: str
    symbol: str
    version: str
    created_at_utc: str
    updated_at_utc: str
    total_jobs: int
    completed_jobs: int
    remaining_jobs: int
    resume_enabled: bool
    result_csv: str
    progress_pct: float
    last_completed_job_id: str
    elapsed_sec: float
    status: str


@dataclass(frozen=True)
class FinalSummary:
    timeframe: str
    jobs_total: int
    jobs_completed: int
    jobs_with_trades: int
    best_job_id: str
    best_score: float
    best_pnl_sum: float
    best_profit_factor: float
    best_expectancy: float
    best_trade_count: int
    result_csv: str
    state_json: str


# -----------------------------
# Utility helpers
# -----------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")



def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")



def safe_float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        value = float(value)
    except Exception:
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value



def safe_int(value: object) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return 0



def compute_score(
    trade_count: int,
    profit_factor: float,
    expectancy: float,
    pnl_sum: float,
    max_drawdown: float,
    sharpe_ratio: float,
) -> float:
    pf_capped = min(safe_float(profit_factor), 10.0)
    exp_value = safe_float(expectancy)
    pnl_value = safe_float(pnl_sum)
    dd_penalty = abs(safe_float(max_drawdown))
    sharpe_value = safe_float(sharpe_ratio)
    trade_bonus = min(max(trade_count, 0) / 100.0, 2.0)
    return float((exp_value * pf_capped) + (pnl_value * 0.20) + (sharpe_value * 0.25) - (dd_penalty * 0.10) + trade_bonus)



def print_line(message: str) -> None:
    print(message, flush=True)


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run memory-safe, resumable VectorBT uncovered parallel backtests."
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--fast-min", type=int, default=1)
    parser.add_argument("--fast-max", type=int, default=50)
    parser.add_argument("--slow-min", type=int, default=20)
    parser.add_argument("--slow-max", type=int, default=100)
    parser.add_argument("--min-bars", type=int, default=500)
    parser.add_argument("--price-col", default="close")
    parser.add_argument("--fee-pct", type=float, default=0.0)
    parser.add_argument("--slippage-pct", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64, help="EMA pair batch size for memory-safe processing.")
    parser.add_argument(
        "--ma-chunk-size",
        type=int,
        default=16,
        help="How many unique EMA windows VectorBT computes in one MA batch.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from previous partial results if files already exist.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore previous progress and start a fresh run into the same folder.",
    )
    parser.add_argument(
        "--freq-map-json",
        default="",
        help='Optional JSON string mapping timeframe -> pandas frequency alias, e.g. {"M30":"30T","D1":"1D"}',
    )
    return parser.parse_args()



def load_freq_map(raw: str) -> Dict[str, str]:
    if not raw.strip():
        return {
            "M1": "1T",
            "M2": "2T",
            "M3": "3T",
            "M4": "4T",
            "M5": "5T",
            "M6": "6T",
            "M10": "10T",
            "M15": "15T",
            "M30": "30T",
            "H1": "1H",
            "H4": "4H",
            "D1": "1D",
        }
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--freq-map-json must decode to a JSON object")
    return {str(k).upper().strip(): str(v) for k, v in parsed.items()}


# -----------------------------
# Data loading
# -----------------------------


def resolve_data_path(data_root: Path, symbol: str, timeframe: str) -> Path:
    candidates = [
        data_root / f"{symbol}_{timeframe}.parquet",
        data_root / f"{symbol}_{timeframe}.csv",
        data_root / timeframe / f"{symbol}_{timeframe}.parquet",
        data_root / timeframe / f"{symbol}_{timeframe}.csv",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.suffix.lower() in ALLOWED_EXTENSIONS:
            return candidate
    raise FileNotFoundError(
        f"No canonical dataset found for timeframe={timeframe}. Checked: {', '.join(str(p) for p in candidates)}"
    )



def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=renamed)

    time_candidates = ["datetime", "time", "timestamp", "date"]
    time_col = next((c for c in time_candidates if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"No datetime column found. Available columns: {list(df.columns)}")

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).copy()
    df = df.drop_duplicates(subset=[time_col], keep="first")
    df = df.sort_values(time_col).reset_index(drop=True)
    df = df.rename(columns={time_col: "datetime"})
    return df



def load_market_data(path: Path, price_col: str, min_bars: int) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    df = normalize_columns(df)
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in {path}. Available columns: {list(df.columns)}")

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col]).copy()

    if len(df) < min_bars:
        raise ValueError(f"Timeframe file has only {len(df)} bars which is less than --min-bars={min_bars}")
    return df


# -----------------------------
# Resume / state helpers
# -----------------------------


def build_param_pairs(fast_min: int, fast_max: int, slow_min: int, slow_max: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for fast in range(fast_min, fast_max + 1):
        for slow in range(slow_min, slow_max + 1):
            if fast < slow:
                pairs.append((fast, slow))
    if not pairs:
        raise ValueError("No valid EMA pairs generated. Check fast/slow ranges.")
    return pairs



def make_job_id(timeframe: str, fast: int, slow: int) -> str:
    return f"vectorbt_uncovered_{timeframe}_{fast:02d}_{slow:03d}"



def load_completed_job_ids(result_csv: Path) -> Set[str]:
    if not result_csv.exists():
        return set()
    try:
        df = pd.read_csv(result_csv, usecols=["job_id"])
    except Exception:
        return set()
    if "job_id" not in df.columns:
        return set()
    return set(df["job_id"].astype(str).tolist())



def append_result_rows(result_csv: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    file_exists = result_csv.exists()
    with result_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)



def write_timeframe_state(
    state_json: Path,
    timeframe: str,
    symbol: str,
    result_csv: Path,
    total_jobs: int,
    completed_jobs: int,
    created_at: str,
    started_monotonic: float,
    last_completed_job_id: str,
    status: str,
    resume_enabled: bool,
) -> None:
    elapsed_sec = max(time.monotonic() - started_monotonic, 0.0)
    remaining_jobs = max(total_jobs - completed_jobs, 0)
    progress_pct = round((completed_jobs / total_jobs) * 100.0, 6) if total_jobs > 0 else 0.0
    payload = asdict(
        TimeframeState(
            timeframe=timeframe,
            symbol=symbol,
            version=VERSION,
            created_at_utc=created_at,
            updated_at_utc=utc_now_iso(),
            total_jobs=total_jobs,
            completed_jobs=completed_jobs,
            remaining_jobs=remaining_jobs,
            resume_enabled=resume_enabled,
            result_csv=str(result_csv),
            progress_pct=progress_pct,
            last_completed_job_id=last_completed_job_id,
            elapsed_sec=round(elapsed_sec, 3),
            status=status,
        )
    )
    write_json(state_json, payload)


# -----------------------------
# VectorBT execution
# -----------------------------


def chunked(items: Sequence[Tuple[int, int]], batch_size: int) -> Iterable[List[Tuple[int, int]]]:
    for i in range(0, len(items), batch_size):
        yield list(items[i : i + batch_size])



def compute_ema_map_vectorbt(price: pd.Series, windows: Sequence[int], ma_chunk_size: int) -> Dict[int, pd.Series]:
    unique_windows = sorted(set(int(w) for w in windows))
    ema_map: Dict[int, pd.Series] = {}
    for i in range(0, len(unique_windows), ma_chunk_size):
        sub_windows = unique_windows[i : i + ma_chunk_size]
        ma = vbt.MA.run(price, window=sub_windows, ewm=True, short_name="ema")
        ma_df = ma.ma
        if isinstance(ma_df, pd.Series):
            ma_df = ma_df.to_frame(name=sub_windows[0])
        ma_df = ma_df.copy()
        ma_df.columns = [int(c) for c in ma_df.columns]
        for window in sub_windows:
            ema_map[int(window)] = ma_df[int(window)]
    return ema_map



def run_pair_backtest(
    price: pd.Series,
    freq_alias: str,
    fast_ema: pd.Series,
    slow_ema: pd.Series,
    fee_pct: float,
    slippage_pct: float,
) -> Dict[str, float]:
    entries = fast_ema > slow_ema
    exits = fast_ema < slow_ema

    # crossover-style signal edges
    long_entries = entries & (~entries.shift(1, fill_value=False))
    long_exits = exits & (~exits.shift(1, fill_value=False))
    short_entries = long_exits
    short_exits = long_entries

    pf = vbt.Portfolio.from_signals(
        close=price,
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        fees=fee_pct,
        slippage=slippage_pct,
        freq=freq_alias,
        init_cash=1.0,
        size=np.inf,
        upon_opposite_entry="close",
        direction="both",
    )

    trades = pf.trades
    trade_count = safe_int(trades.count())
    win_rate_pct = safe_float(trades.win_rate()) * 100.0
    profit_factor = safe_float(trades.profit_factor())
    expectancy = safe_float(trades.expectancy())
    pnl_sum = safe_float(trades.pnl.sum())
    max_drawdown = safe_float(pf.max_drawdown())
    sharpe_ratio = safe_float(pf.sharpe_ratio())
    avg_win = safe_float(trades.winning.pnl.mean())
    avg_loss = safe_float(trades.losing.pnl.mean())
    score = compute_score(
        trade_count=trade_count,
        profit_factor=profit_factor,
        expectancy=expectancy,
        pnl_sum=pnl_sum,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
    )

    return {
        "trade_count": trade_count,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "pnl_sum": pnl_sum,
        "max_drawdown": max_drawdown,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "sharpe_ratio": sharpe_ratio,
        "score": score,
    }


# -----------------------------
# Per-timeframe execution
# -----------------------------


def process_timeframe(
    symbol: str,
    timeframe: str,
    data_root: Path,
    outdir: Path,
    price_col: str,
    min_bars: int,
    freq_alias: str,
    pairs: List[Tuple[int, int]],
    fee_pct: float,
    slippage_pct: float,
    batch_size: int,
    ma_chunk_size: int,
    resume_enabled: bool,
) -> FinalSummary:
    timeframe = timeframe.upper().strip()
    ensure_dir(outdir)
    per_tf_dir = outdir / "per_timeframe"
    state_dir = outdir / "state"
    ensure_dir(per_tf_dir)
    ensure_dir(state_dir)

    result_csv = per_tf_dir / f"{symbol}_{timeframe}_vectorbt_uncovered_results.csv"
    state_json = state_dir / f"{symbol}_{timeframe}_state.json"

    data_path = resolve_data_path(data_root=data_root, symbol=symbol, timeframe=timeframe)
    df = load_market_data(path=data_path, price_col=price_col, min_bars=min_bars)
    price = df.set_index("datetime")[price_col].astype(float)
    price = price[~price.index.duplicated(keep="first")]

    all_job_ids = {make_job_id(timeframe, fast, slow) for fast, slow in pairs}
    completed_job_ids = load_completed_job_ids(result_csv) if resume_enabled else set()
    unknown_done = completed_job_ids - all_job_ids
    if unknown_done:
        print_line(f"[WARN] timeframe={timeframe} found {len(unknown_done)} completed jobs not in current parameter range. They will be ignored.")
        completed_job_ids = completed_job_ids & all_job_ids

    remaining_pairs = [(fast, slow) for fast, slow in pairs if make_job_id(timeframe, fast, slow) not in completed_job_ids]
    total_jobs = len(pairs)
    started_monotonic = time.monotonic()
    created_at = utc_now_iso()

    print_line("=" * 120)
    print_line(f"[START] timeframe={timeframe} bars={len(price)} total_jobs={total_jobs} completed_before_resume={len(completed_job_ids)} remaining={len(remaining_pairs)}")
    print_line(f"[START] data_path={data_path}")
    print_line(f"[START] result_csv={result_csv}")
    print_line(f"[START] state_json={state_json}")

    if len(remaining_pairs) == 0:
        write_timeframe_state(
            state_json=state_json,
            timeframe=timeframe,
            symbol=symbol,
            result_csv=result_csv,
            total_jobs=total_jobs,
            completed_jobs=total_jobs,
            created_at=created_at,
            started_monotonic=started_monotonic,
            last_completed_job_id="RESUME_ALREADY_COMPLETE",
            status="DONE",
            resume_enabled=resume_enabled,
        )
        result_df = pd.read_csv(result_csv)
        result_df = result_df.sort_values(["score", "pnl_sum", "profit_factor", "expectancy"], ascending=[False, False, False, False])
        top_row = result_df.iloc[0]
        return FinalSummary(
            timeframe=timeframe,
            jobs_total=total_jobs,
            jobs_completed=total_jobs,
            jobs_with_trades=safe_int((result_df["trade_count"] > 0).sum()),
            best_job_id=str(top_row["job_id"]),
            best_score=safe_float(top_row["score"]),
            best_pnl_sum=safe_float(top_row["pnl_sum"]),
            best_profit_factor=safe_float(top_row["profit_factor"]),
            best_expectancy=safe_float(top_row["expectancy"]),
            best_trade_count=safe_int(top_row["trade_count"]),
            result_csv=str(result_csv),
            state_json=str(state_json),
        )

    completed_count = len(completed_job_ids)
    last_completed_job_id = ""

    for batch_index, pair_batch in enumerate(chunked(remaining_pairs, batch_size), start=1):
        batch_started = time.monotonic()
        batch_windows: List[int] = []
        for fast, slow in pair_batch:
            batch_windows.append(fast)
            batch_windows.append(slow)
        ema_map = compute_ema_map_vectorbt(price=price, windows=batch_windows, ma_chunk_size=ma_chunk_size)

        rows_to_append: List[Dict[str, object]] = []
        for fast, slow in pair_batch:
            job_id = make_job_id(timeframe, fast, slow)
            metrics = run_pair_backtest(
                price=price,
                freq_alias=freq_alias,
                fast_ema=ema_map[int(fast)],
                slow_ema=ema_map[int(slow)],
                fee_pct=fee_pct,
                slippage_pct=slippage_pct,
            )
            row: Dict[str, object] = {
                "job_id": job_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy_family": "ema_filter_research",
                "logic_variant": "ema_fast_slow_filter_only",
                "ema_fast": int(fast),
                "ema_slow": int(slow),
                "micro_exit_variant": DEFAULT_MICRO_EXIT_VARIANT,
                "status": "DONE",
                "created_at_utc": utc_now_iso(),
                "data_path": str(data_path),
                "total_bars": int(len(price)),
                "trade_count": int(metrics["trade_count"]),
                "win_rate_pct": float(metrics["win_rate_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "expectancy": float(metrics["expectancy"]),
                "pnl_sum": float(metrics["pnl_sum"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "avg_win": float(metrics["avg_win"]),
                "avg_loss": float(metrics["avg_loss"]),
                "sharpe_ratio": float(metrics["sharpe_ratio"]),
                "score": float(metrics["score"]),
                "result_note": "incremental_append_resume_safe",
            }
            rows_to_append.append(row)
            completed_count += 1
            last_completed_job_id = job_id

        append_result_rows(result_csv=result_csv, rows=rows_to_append)
        write_timeframe_state(
            state_json=state_json,
            timeframe=timeframe,
            symbol=symbol,
            result_csv=result_csv,
            total_jobs=total_jobs,
            completed_jobs=completed_count,
            created_at=created_at,
            started_monotonic=started_monotonic,
            last_completed_job_id=last_completed_job_id,
            status="RUNNING",
            resume_enabled=resume_enabled,
        )

        batch_elapsed = max(time.monotonic() - batch_started, 0.0)
        progress_pct = (completed_count / total_jobs) * 100.0 if total_jobs > 0 else 0.0
        print_line(
            f"[PROGRESS] timeframe={timeframe} batch={batch_index} batch_jobs={len(pair_batch)} completed={completed_count}/{total_jobs} progress_pct={progress_pct:.2f} last_job_id={last_completed_job_id} batch_sec={batch_elapsed:.2f}"
        )

    result_df = pd.read_csv(result_csv)
    result_df = result_df.sort_values(["score", "pnl_sum", "profit_factor", "expectancy"], ascending=[False, False, False, False])
    result_df.to_csv(result_csv, index=False, encoding="utf-8")

    top_row = result_df.iloc[0]
    jobs_with_trades = safe_int((result_df["trade_count"] > 0).sum())

    write_timeframe_state(
        state_json=state_json,
        timeframe=timeframe,
        symbol=symbol,
        result_csv=result_csv,
        total_jobs=total_jobs,
        completed_jobs=completed_count,
        created_at=created_at,
        started_monotonic=started_monotonic,
        last_completed_job_id=str(top_row["job_id"]),
        status="DONE",
        resume_enabled=resume_enabled,
    )

    print_line(
        f"[DONE] timeframe={timeframe} jobs_total={total_jobs} jobs_completed={completed_count} jobs_with_trades={jobs_with_trades} best_job_id={top_row['job_id']}"
    )

    return FinalSummary(
        timeframe=timeframe,
        jobs_total=total_jobs,
        jobs_completed=completed_count,
        jobs_with_trades=jobs_with_trades,
        best_job_id=str(top_row["job_id"]),
        best_score=safe_float(top_row["score"]),
        best_pnl_sum=safe_float(top_row["pnl_sum"]),
        best_profit_factor=safe_float(top_row["profit_factor"]),
        best_expectancy=safe_float(top_row["expectancy"]),
        best_trade_count=safe_int(top_row["trade_count"]),
        result_csv=str(result_csv),
        state_json=str(state_json),
    )


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    args = parse_args()
    symbol = str(args.symbol).upper().strip()
    timeframes = [str(tf).upper().strip() for tf in args.timeframes]
    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    ensure_dir(outdir / "per_timeframe")
    ensure_dir(outdir / "state")

    freq_map = load_freq_map(args.freq_map_json)
    pairs = build_param_pairs(args.fast_min, args.fast_max, args.slow_min, args.slow_max)
    resume_enabled = bool(args.resume) and not bool(args.no_resume)
    started_at = utc_now_iso()
    started_monotonic = time.monotonic()

    print_line("=" * 120)
    print_line(f"[START] version={VERSION}")
    print_line(f"[START] symbol={symbol}")
    print_line(f"[START] outdir={outdir}")
    print_line(f"[START] data_root={data_root}")
    print_line(f"[START] timeframes={','.join(timeframes)}")
    print_line(f"[START] ema_pairs_total={len(pairs)}")
    print_line(f"[START] batch_size={args.batch_size}")
    print_line(f"[START] ma_chunk_size={args.ma_chunk_size}")
    print_line(f"[START] resume_enabled={resume_enabled}")
    print_line("=" * 120)

    summary_rows: List[FinalSummary] = []
    manifest_rows: List[Dict[str, object]] = []

    for timeframe in timeframes:
        if timeframe not in freq_map:
            raise KeyError(f"No frequency mapping found for timeframe={timeframe}")

        tf_summary = process_timeframe(
            symbol=symbol,
            timeframe=timeframe,
            data_root=data_root,
            outdir=outdir,
            price_col=args.price_col,
            min_bars=args.min_bars,
            freq_alias=freq_map[timeframe],
            pairs=pairs,
            fee_pct=args.fee_pct,
            slippage_pct=args.slippage_pct,
            batch_size=max(1, int(args.batch_size)),
            ma_chunk_size=max(1, int(args.ma_chunk_size)),
            resume_enabled=resume_enabled,
        )
        summary_rows.append(tf_summary)
        manifest_rows.append(asdict(tf_summary))

    summary_payload = {
        "version": VERSION,
        "symbol": symbol,
        "timeframes": timeframes,
        "timeframes_total": len(summary_rows),
        "jobs_total": int(sum(row.jobs_total for row in summary_rows)),
        "jobs_completed": int(sum(row.jobs_completed for row in summary_rows)),
        "jobs_with_trades_total": int(sum(row.jobs_with_trades for row in summary_rows)),
        "created_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "elapsed_sec": round(max(time.monotonic() - started_monotonic, 0.0), 3),
        "timeframe_summaries": [asdict(row) for row in summary_rows],
    }
    manifest_payload = {
        "version": VERSION,
        "symbol": symbol,
        "data_root": str(data_root),
        "outdir": str(outdir),
        "timeframes": timeframes,
        "fast_min": args.fast_min,
        "fast_max": args.fast_max,
        "slow_min": args.slow_min,
        "slow_max": args.slow_max,
        "ema_pairs_total": len(pairs),
        "micro_exit_variant": DEFAULT_MICRO_EXIT_VARIANT,
        "resume_enabled": resume_enabled,
        "batch_size": int(args.batch_size),
        "ma_chunk_size": int(args.ma_chunk_size),
        "created_at_utc": started_at,
        "timeframe_runs": manifest_rows,
    }

    summary_json = outdir / "summary.json"
    manifest_json = outdir / "run_manifest.json"
    write_json(summary_json, summary_payload)
    write_json(manifest_json, manifest_payload)

    leaderboard_rows: List[pd.DataFrame] = []
    for row in summary_rows:
        df = pd.read_csv(row.result_csv)
        df = df.sort_values(["score", "pnl_sum", "profit_factor", "expectancy"], ascending=[False, False, False, False]).head(50)
        leaderboard_rows.append(df)
    leaderboard_df = pd.concat(leaderboard_rows, ignore_index=True) if leaderboard_rows else pd.DataFrame(columns=RESULT_COLUMNS)
    leaderboard_csv = outdir / "leaderboard_top50_per_tf.csv"
    leaderboard_df.to_csv(leaderboard_csv, index=False, encoding="utf-8")

    print_line("=" * 120)
    print_line(f"[DONE] version={VERSION}")
    print_line(f"[DONE] outdir={outdir}")
    print_line(f"[DONE] manifest_json={manifest_json}")
    print_line(f"[DONE] summary_json={summary_json}")
    print_line(f"[DONE] leaderboard_csv={leaderboard_csv}")
    print_line(f"[DONE] total_jobs={summary_payload['jobs_total']}")
    print_line(f"[DONE] total_jobs_completed={summary_payload['jobs_completed']}")
    print_line(f"[DONE] total_jobs_with_trades={summary_payload['jobs_with_trades_total']}")
    print_line("=" * 120)


if __name__ == "__main__":
    main()

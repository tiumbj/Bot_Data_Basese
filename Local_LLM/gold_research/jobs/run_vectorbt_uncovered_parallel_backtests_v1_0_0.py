#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: run_vectorbt_uncovered_parallel_backtests_v1_0_0.py
Version: v1.0.0
Path: C:/Data/Bot/Local_LLM/gold_research/jobs/run_vectorbt_uncovered_parallel_backtests_v1_0_0.py

Purpose:
- Run uncovered research backtests in parallel using VectorBT for speed.
- Cover the locked uncovered timeframes first: M2, M3, M4, M6, M30, D1.
- Sweep EMA fast/slow combinations under the locked rule: fast 1-50, slow 20-100, fast < slow.
- Produce database-ready outputs immediately: manifest, per-timeframe result files, summary JSON, leaderboard CSV.

Notes:
- This file is the first production-oriented VBT speed layer for uncovered backtests.
- It does not replace the locked custom research core. It accelerates uncovered evidence collection.
- Micro-exit is kept as an explicit output field. v1.0.0 starts with the baseline "reverse_signal_exit"
  so results can be ingested now. Additional micro-exit families can be layered on top in later files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
except ImportError as exc:  # pragma: no cover - runtime environment dependent
    raise SystemExit(
        "[ERROR] vectorbt is not installed. Install it first with: pip install vectorbt"
    ) from exc


VERSION = "v1.0.0"
DEFAULT_TIMEFRAMES = ["M2", "M3", "M4", "M6", "M30", "D1"]
DEFAULT_DATA_ROOT = Path(r"C:\Data\Bot\central_market_data\parquet")
DEFAULT_OUTDIR = Path(r"C:\Data\Bot\central_backtest_results\vectorbt_uncovered_parallel_v1_0_0")
DEFAULT_SYMBOL = "XAUUSD"
DEFAULT_MICRO_EXIT_VARIANT = "reverse_signal_exit"
ALLOWED_EXTENSIONS = (".parquet", ".csv")


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    symbol: str
    timeframe: str
    strategy_family: str
    logic_variant: str
    ema_fast: int
    ema_slow: int
    micro_exit_variant: str
    data_path: str
    result_path: str
    status: str
    total_bars: int
    trade_count: int
    win_rate_pct: float
    profit_factor: float
    expectancy: float
    pnl_sum: float
    max_drawdown: float
    avg_win: float
    avg_loss: float
    sharpe_ratio: float
    created_at_utc: str


@dataclass(frozen=True)
class TimeframeSummary:
    timeframe: str
    jobs_total: int
    jobs_with_trades: int
    best_job_id: str
    best_score: float
    best_pnl_sum: float
    best_profit_factor: float
    best_expectancy: float
    best_trade_count: int
    result_csv: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VectorBT uncovered parallel backtests and write database-ready outputs."
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Trading symbol prefix in file names.")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=DEFAULT_TIMEFRAMES,
        help="Timeframes to run. Default covers the currently uncovered matrix.",
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Directory containing canonical TF parquet/csv files.",
    )
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help="Output directory for manifests, summaries, and result tables.",
    )
    parser.add_argument("--fast-min", type=int, default=1, help="Minimum fast EMA period.")
    parser.add_argument("--fast-max", type=int, default=50, help="Maximum fast EMA period.")
    parser.add_argument("--slow-min", type=int, default=20, help="Minimum slow EMA period.")
    parser.add_argument("--slow-max", type=int, default=100, help="Maximum slow EMA period.")
    parser.add_argument(
        "--min-bars",
        type=int,
        default=500,
        help="Minimum required bars for a timeframe file to be accepted.",
    )
    parser.add_argument(
        "--price-col",
        default="close",
        help="Price column used for VectorBT signal backtests.",
    )
    parser.add_argument(
        "--fee-pct",
        type=float,
        default=0.0,
        help="Per-side fee fraction passed to VectorBT (example 0.0002 = 2 bps).",
    )
    parser.add_argument(
        "--slippage-pct",
        type=float,
        default=0.0,
        help="Per-side slippage fraction passed to VectorBT.",
    )
    parser.add_argument(
        "--freq-map-json",
        default="",
        help="Optional JSON string mapping timeframe -> pandas frequency alias, e.g. '{\"M30\":\"30T\",\"D1\":\"1D\"}'.",
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
    return {str(k): str(v) for k, v in parsed.items()}


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
    renamed = {c: c.strip().lower() for c in df.columns}
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


def build_param_grid(fast_min: int, fast_max: int, slow_min: int, slow_max: int) -> Tuple[np.ndarray, np.ndarray]:
    fast_values: List[int] = []
    slow_values: List[int] = []
    for fast in range(fast_min, fast_max + 1):
        for slow in range(slow_min, slow_max + 1):
            if fast < slow:
                fast_values.append(fast)
                slow_values.append(slow)
    if not fast_values:
        raise ValueError("No valid EMA combinations generated. Check fast/slow ranges.")
    return np.asarray(fast_values, dtype=int), np.asarray(slow_values, dtype=int)


def nan_to_zero(value: float) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (float, np.floating)) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    return float(value)


def safe_ratio(num: float, denom: float) -> float:
    if denom == 0:
        return 0.0
    return float(num / denom)


def compute_score(row: pd.Series) -> float:
    pf_capped = min(float(row["profit_factor"]), 10.0)
    expectancy = float(row["expectancy"])
    max_dd = float(row["max_drawdown"])
    trades = float(row["trade_count"])
    trade_bonus = min(trades / 100.0, 2.0)
    return float((expectancy * pf_capped) - (max_dd * 0.10) + trade_bonus)


def run_vectorbt_grid(
    df: pd.DataFrame,
    price_col: str,
    freq_alias: str,
    fast_values: np.ndarray,
    slow_values: np.ndarray,
    fee_pct: float,
    slippage_pct: float,
) -> pd.DataFrame:
    price = df.set_index("datetime")[price_col].astype(float)
    price = price[~price.index.duplicated(keep="first")]

    fast_ma = vbt.MA.run(price, window=fast_values, short_name="ema_fast", ewm=True)
    slow_ma = vbt.MA.run(price, window=slow_values, short_name="ema_slow", ewm=True)

    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    short_entries = exits
    short_exits = entries

    portfolio = vbt.Portfolio.from_signals(
        close=price,
        entries=entries,
        exits=exits,
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

    trades = portfolio.trades
    trade_count = trades.count()
    win_rate = trades.win_rate() * 100.0
    profit_factor = trades.profit_factor()
    expectancy = trades.expectancy()
    pnl_sum = trades.pnl.sum()
    max_drawdown = portfolio.max_drawdown()
    sharpe_ratio = portfolio.sharpe_ratio()
    avg_win = trades.winning.pnl.mean()
    avg_loss = trades.losing.pnl.mean()

    result = pd.DataFrame(
        {
            "ema_fast": fast_values,
            "ema_slow": slow_values,
            "trade_count": np.asarray(trade_count, dtype=float),
            "win_rate_pct": np.asarray(win_rate, dtype=float),
            "profit_factor": np.asarray(profit_factor, dtype=float),
            "expectancy": np.asarray(expectancy, dtype=float),
            "pnl_sum": np.asarray(pnl_sum, dtype=float),
            "max_drawdown": np.asarray(max_drawdown, dtype=float),
            "sharpe_ratio": np.asarray(sharpe_ratio, dtype=float),
            "avg_win": np.asarray(avg_win, dtype=float),
            "avg_loss": np.asarray(avg_loss, dtype=float),
        }
    )

    for col in [
        "trade_count",
        "win_rate_pct",
        "profit_factor",
        "expectancy",
        "pnl_sum",
        "max_drawdown",
        "sharpe_ratio",
        "avg_win",
        "avg_loss",
    ]:
        result[col] = result[col].map(nan_to_zero)

    result["trade_count"] = result["trade_count"].round().astype(int)
    result["score"] = result.apply(compute_score, axis=1)
    return result.sort_values(["score", "pnl_sum", "profit_factor"], ascending=[False, False, False]).reset_index(drop=True)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    per_tf_dir = outdir / "per_timeframe"
    ensure_dir(per_tf_dir)

    freq_map = load_freq_map(args.freq_map_json)
    fast_values, slow_values = build_param_grid(args.fast_min, args.fast_max, args.slow_min, args.slow_max)

    manifest_rows: List[Dict[str, object]] = []
    leaderboard_rows: List[JobRecord] = []
    summary_rows: List[TimeframeSummary] = []
    run_created_at = utc_now_iso()

    for timeframe in args.timeframes:
        timeframe = str(timeframe).upper().strip()
        freq_alias = freq_map.get(timeframe)
        if not freq_alias:
            raise KeyError(f"No frequency mapping found for timeframe={timeframe}")

        data_path = resolve_data_path(data_root=data_root, symbol=args.symbol, timeframe=timeframe)
        df = load_market_data(path=data_path, price_col=args.price_col, min_bars=args.min_bars)

        result_df = run_vectorbt_grid(
            df=df,
            price_col=args.price_col,
            freq_alias=freq_alias,
            fast_values=fast_values,
            slow_values=slow_values,
            fee_pct=args.fee_pct,
            slippage_pct=args.slippage_pct,
        )

        result_df.insert(0, "symbol", args.symbol)
        result_df.insert(1, "timeframe", timeframe)
        result_df.insert(2, "strategy_family", "ema_filter_research")
        result_df.insert(3, "logic_variant", "ema_fast_slow_filter_only")
        result_df.insert(6, "micro_exit_variant", DEFAULT_MICRO_EXIT_VARIANT)
        result_df.insert(7, "status", "DONE")
        result_df.insert(8, "created_at_utc", run_created_at)
        result_df.insert(9, "data_path", str(data_path))

        tf_result_path = per_tf_dir / f"{args.symbol}_{timeframe}_vectorbt_uncovered_results.csv"
        result_df.to_csv(tf_result_path, index=False, encoding="utf-8")

        top_row = result_df.iloc[0]
        tf_job_id = f"vectorbt_uncovered_{timeframe}_{int(top_row['ema_fast']):02d}_{int(top_row['ema_slow']):03d}"
        summary_rows.append(
            TimeframeSummary(
                timeframe=timeframe,
                jobs_total=int(len(result_df)),
                jobs_with_trades=int((result_df["trade_count"] > 0).sum()),
                best_job_id=tf_job_id,
                best_score=float(top_row["score"]),
                best_pnl_sum=float(top_row["pnl_sum"]),
                best_profit_factor=float(top_row["profit_factor"]),
                best_expectancy=float(top_row["expectancy"]),
                best_trade_count=int(top_row["trade_count"]),
                result_csv=str(tf_result_path),
            )
        )

        top_n = min(50, len(result_df))
        for idx, row in result_df.head(top_n).iterrows():
            job_id = f"vectorbt_uncovered_{timeframe}_{int(row['ema_fast']):02d}_{int(row['ema_slow']):03d}"
            record = JobRecord(
                job_id=job_id,
                symbol=args.symbol,
                timeframe=timeframe,
                strategy_family="ema_filter_research",
                logic_variant="ema_fast_slow_filter_only",
                ema_fast=int(row["ema_fast"]),
                ema_slow=int(row["ema_slow"]),
                micro_exit_variant=DEFAULT_MICRO_EXIT_VARIANT,
                data_path=str(data_path),
                result_path=str(tf_result_path),
                status="DONE",
                total_bars=int(len(df)),
                trade_count=int(row["trade_count"]),
                win_rate_pct=float(row["win_rate_pct"]),
                profit_factor=float(row["profit_factor"]),
                expectancy=float(row["expectancy"]),
                pnl_sum=float(row["pnl_sum"]),
                max_drawdown=float(row["max_drawdown"]),
                avg_win=float(row["avg_win"]),
                avg_loss=float(row["avg_loss"]),
                sharpe_ratio=float(row["sharpe_ratio"]),
                created_at_utc=run_created_at,
            )
            leaderboard_rows.append(record)

        manifest_rows.append(
            {
                "timeframe": timeframe,
                "symbol": args.symbol,
                "data_path": str(data_path),
                "bars": int(len(df)),
                "freq_alias": freq_alias,
                "jobs_generated": int(len(result_df)),
                "result_csv": str(tf_result_path),
                "created_at_utc": run_created_at,
                "version": VERSION,
            }
        )

    leaderboard_df = pd.DataFrame([asdict(r) for r in leaderboard_rows]).sort_values(
        ["pnl_sum", "profit_factor", "expectancy", "trade_count"],
        ascending=[False, False, False, False],
    )
    leaderboard_path = outdir / "leaderboard_top50_per_tf.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False, encoding="utf-8")

    manifest_path = outdir / "run_manifest.json"
    write_json(
        manifest_path,
        {
            "version": VERSION,
            "symbol": args.symbol,
            "timeframes": [str(tf).upper().strip() for tf in args.timeframes],
            "data_root": str(data_root),
            "outdir": str(outdir),
            "fast_min": args.fast_min,
            "fast_max": args.fast_max,
            "slow_min": args.slow_min,
            "slow_max": args.slow_max,
            "micro_exit_variant": DEFAULT_MICRO_EXIT_VARIANT,
            "fee_pct": args.fee_pct,
            "slippage_pct": args.slippage_pct,
            "created_at_utc": run_created_at,
            "timeframe_runs": manifest_rows,
        },
    )

    summary_path = outdir / "summary.json"
    write_json(
        summary_path,
        {
            "version": VERSION,
            "symbol": args.symbol,
            "timeframes_total": len(summary_rows),
            "jobs_total": int(sum(row.jobs_total for row in summary_rows)),
            "jobs_with_trades_total": int(sum(row.jobs_with_trades for row in summary_rows)),
            "created_at_utc": run_created_at,
            "leaderboard_csv": str(leaderboard_path),
            "timeframe_summaries": [asdict(row) for row in summary_rows],
        },
    )

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] outdir={outdir}")
    print(f"[DONE] manifest_json={manifest_path}")
    print(f"[DONE] summary_json={summary_path}")
    print(f"[DONE] leaderboard_csv={leaderboard_path}")
    print(f"[DONE] timeframes={','.join([str(tf).upper().strip() for tf in args.timeframes])}")
    print(f"[DONE] total_jobs={sum(row.jobs_total for row in summary_rows)}")
    print(f"[DONE] total_jobs_with_trades={sum(row.jobs_with_trades for row in summary_rows)}")
    print("=" * 120)


if __name__ == "__main__":
    main()

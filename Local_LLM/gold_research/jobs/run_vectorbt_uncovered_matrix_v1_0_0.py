
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_vectorbt_uncovered_matrix_v1_0_0.py
Version: v1.0.0

Uncovered baseline sweep:
- TF: M2,M3,M4,M6,M30,D1 by default
- EMA fast 1..50
- EMA slow 20..100
- fast < slow
- auto save
- auto resume
- live progress + ETA
"""


from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def append_frame_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def load_completed_job_ids(result_csv: Path) -> set:
    if not result_csv.exists():
        return set()
    try:
        df = pd.read_csv(result_csv, usecols=["job_id"])
        return set(df["job_id"].astype(str).tolist())
    except Exception:
        return set()


def read_price_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    cols = {c.lower(): c for c in df.columns}
    rename_map = {}
    for target in ["time", "open", "high", "low", "close"]:
        if target in cols:
            rename_map[cols[target]] = target
    df = df.rename(columns=rename_map)
    required = ["time", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns in {path}: {missing}")
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def read_feature_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def make_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def run_pf_from_signals(close: pd.Series, entries: pd.Series, exits: pd.Series, short_entries=None, short_exits=None) -> vbt.Portfolio:
    entries = entries.fillna(False).astype(bool)
    exits = exits.fillna(False).astype(bool)
    if short_entries is None:
        pf = vbt.Portfolio.from_signals(close=close, entries=entries, exits=exits, init_cash=100_000.0, fees=0.0, slippage=0.0)
    else:
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            short_entries=short_entries.fillna(False).astype(bool),
            short_exits=short_exits.fillna(False).astype(bool),
            init_cash=100_000.0,
            fees=0.0,
            slippage=0.0,
        )
    return pf


def safe_stat(callable_obj, default=np.nan):
    try:
        return callable_obj()
    except Exception:
        return default


def portfolio_metrics(pf: vbt.Portfolio) -> Dict[str, float]:
    total_return = safe_stat(lambda: float(pf.total_return()))
    trade_count = safe_stat(lambda: int(pf.trades.count()))
    win_rate = safe_stat(lambda: float(pf.trades.win_rate()))
    profit_factor = safe_stat(lambda: float(pf.trades.profit_factor()))
    expectancy = safe_stat(lambda: float(pf.trades.expectancy()))
    max_dd = safe_stat(lambda: float(pf.max_drawdown()))
    avg_win = safe_stat(lambda: float(pf.trades.winning.pnl.mean()))
    avg_loss = safe_stat(lambda: float(pf.trades.losing.pnl.mean()))
    return {
        "trade_count": trade_count,
        "win_rate_pct": win_rate * 100.0 if win_rate == win_rate else np.nan,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "pnl_sum": total_return,
        "max_drawdown": max_dd,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def build_live_progress(
    version: str,
    outdir: Path,
    current_timeframe: str,
    current_phase: str,
    overall_total: int,
    overall_completed: int,
    start_ts: float,
) -> Dict:
    elapsed_sec = max(time.time() - start_ts, 1e-9)
    rate = overall_completed / elapsed_sec if overall_completed > 0 else 0.0
    remaining = max(overall_total - overall_completed, 0)
    eta_min = (remaining / rate / 60.0) if rate > 0 else None
    payload = {
        "version": version,
        "updated_at_utc": now_utc_iso(),
        "current_phase": current_phase,
        "current_timeframe": current_timeframe,
        "overall_total": overall_total,
        "overall_completed": overall_completed,
        "overall_remaining": remaining,
        "overall_progress_pct": round((overall_completed / overall_total) * 100.0, 4) if overall_total else 100.0,
        "observed_elapsed_min": round(elapsed_sec / 60.0, 4),
        "overall_eta_remaining_min": round(eta_min, 4) if eta_min is not None else None,
        "outdir": str(outdir),
    }
    save_json(outdir / "live_progress.json", payload)
    return payload


import argparse
from pathlib import Path


VERSION = "v1.0.0"


def generate_jobs() -> List[Dict]:
    jobs = []
    for fast in range(1, 51):
        for slow in range(20, 101):
            if fast < slow:
                jobs.append({
                    "ema_fast": fast,
                    "ema_slow": slow,
                    "strategy_family": "ema_cross_long_only",
                    "logic_variant": "ema_fast_slow_cross_long_only",
                    "micro_exit_variant": "reverse_signal_exit",
                })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run uncovered baseline matrix with VectorBT")
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\Data\Bot\central_market_data\parquet"))
    parser.add_argument("--feature-root", type=Path, default=Path(r"C:\Data\Bot\central_feature_cache\base_features_v1_0_0"))
    parser.add_argument("--outdir", type=Path, default=Path(r"C:\Data\Bot\central_backtest_results\vectorbt_uncovered_matrix_v1_0_0"))
    parser.add_argument("--timeframes", default="M2,M3,M4,M6,M30,D1")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    outdir = args.outdir
    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]
    jobs = generate_jobs()
    overall_total = len(jobs) * len(timeframes)
    overall_completed = 0
    start_ts = time.time()

    save_json(outdir / "run_manifest.json", {
        "version": VERSION,
        "generated_at_utc": now_utc_iso(),
        "symbol": args.symbol,
        "timeframes": timeframes,
        "job_count_per_tf": len(jobs),
        "overall_total_jobs": overall_total,
    })

    summary_rows = []

    for tf in timeframes:
        price_path = args.data_root / f"{args.symbol}_{tf}.parquet"
        feature_path = args.feature_root / f"{args.symbol}_{tf}_base_features.parquet"
        result_csv = outdir / "per_timeframe" / f"{args.symbol}_{tf}_uncovered_results.csv"
        state_path = outdir / "state" / f"{args.symbol}_{tf}_state.json"
        if not price_path.exists():
            save_json(state_path, {"version": VERSION, "status": "MISSING_DATA", "timeframe": tf, "price_path": str(price_path)})
            summary_rows.append({"timeframe": tf, "status": "MISSING_DATA"})
            continue
        price_df = read_price_parquet(price_path)
        close = price_df["close"].astype(float)
        completed_ids = set() if args.no_resume else load_completed_job_ids(result_csv)

        print(f"[START] timeframe={tf} bars={len(price_df)} price_path={price_path}")
        ema_cache: Dict[int, pd.Series] = {}
        tf_done = 0

        for batch_start in range(0, len(jobs), args.batch_size):
            batch = jobs[batch_start:batch_start + args.batch_size]
            batch_rows = []
            for job in batch:
                job_id = f"{args.symbol}_{tf}_ema_fast_{job['ema_fast']:03d}_ema_slow_{job['ema_slow']:03d}"
                if job_id in completed_ids:
                    tf_done += 1
                    overall_completed += 1
                    continue

                if job["ema_fast"] not in ema_cache:
                    ema_cache[job["ema_fast"]] = make_ema(close, job["ema_fast"])
                if job["ema_slow"] not in ema_cache:
                    ema_cache[job["ema_slow"]] = make_ema(close, job["ema_slow"])

                ema_fast = ema_cache[job["ema_fast"]]
                ema_slow = ema_cache[job["ema_slow"]]
                long_signal = (ema_fast > ema_slow).astype(bool)
                prev_long = long_signal.shift(1).fillna(False).astype(bool)
                entries = long_signal & (~prev_long)
                exits = (~long_signal) & prev_long

                pf = run_pf_from_signals(close=close, entries=entries, exits=exits)
                metrics = portfolio_metrics(pf)
                batch_rows.append({
                    "job_id": job_id,
                    "timeframe": tf,
                    "strategy_family": job["strategy_family"],
                    "logic_variant": job["logic_variant"],
                    "ema_fast": job["ema_fast"],
                    "ema_slow": job["ema_slow"],
                    "micro_exit_variant": job["micro_exit_variant"],
                    **metrics,
                    "status": "DONE",
                    "created_at_utc": now_utc_iso(),
                })
                tf_done += 1
                overall_completed += 1

            if batch_rows:
                append_frame_csv(result_csv, pd.DataFrame(batch_rows))

            live = build_live_progress(VERSION, outdir, tf, "run_vectorbt_uncovered_matrix", overall_total, overall_completed, start_ts)
            save_json(state_path, {
                "version": VERSION,
                "updated_at_utc": now_utc_iso(),
                "timeframe": tf,
                "completed_jobs": tf_done,
                "total_jobs": len(jobs),
                "remaining_jobs": len(jobs) - tf_done,
                "progress_pct": round((tf_done / len(jobs)) * 100.0, 4),
                "last_batch_end_index": batch_start + len(batch),
                "result_csv": str(result_csv),
            })
            print(f"[PROGRESS] timeframe={tf} batch={(batch_start // args.batch_size) + 1} tf_completed={tf_done}/{len(jobs)} tf_progress_pct={(tf_done / len(jobs)) * 100.0:.2f} overall_completed={overall_completed}/{overall_total} overall_progress_pct={live['overall_progress_pct']:.2f} overall_eta_remaining_min={live['overall_eta_remaining_min']}")

        summary_rows.append({"timeframe": tf, "status": "DONE", "completed_jobs": tf_done, "total_jobs": len(jobs)})
        print(f"[DONE] timeframe={tf} jobs_total={len(jobs)} jobs_completed={tf_done} result_csv={result_csv}")

    leaderboard_rows = []
    for tf in timeframes:
        result_csv = outdir / "per_timeframe" / f"{args.symbol}_{tf}_uncovered_results.csv"
        if result_csv.exists():
            df = pd.read_csv(result_csv)
            if not df.empty:
                rank_df = df.sort_values(["profit_factor", "expectancy", "pnl_sum"], ascending=[False, False, False]).head(50)
                rank_df.insert(0, "leaderboard_rank", range(1, len(rank_df) + 1))
                leaderboard_rows.append(rank_df)
    if leaderboard_rows:
        pd.concat(leaderboard_rows, ignore_index=True).to_csv(outdir / "leaderboard_top50_per_tf.csv", index=False)

    summary = {
        "version": VERSION,
        "generated_at_utc": now_utc_iso(),
        "timeframes": timeframes,
        "overall_total_jobs": overall_total,
        "overall_completed_jobs": overall_completed,
        "overall_progress_pct": round((overall_completed / overall_total) * 100.0, 4) if overall_total else 100.0,
        "rows": summary_rows,
    }
    save_json(outdir / "summary.json", summary)
    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] outdir={outdir}")
    print(f"[DONE] overall_total_jobs={overall_total}")
    print(f"[DONE] overall_completed_jobs={overall_completed}")
    print(f"[DONE] summary_json={outdir / 'summary.json'}")
    print("=" * 120)


if __name__ == "__main__":
    main()

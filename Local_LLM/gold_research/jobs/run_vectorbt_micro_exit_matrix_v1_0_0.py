
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_vectorbt_micro_exit_matrix_v1_0_0.py
Version: v1.0.0

Test micro-exit family across all research TF using top pending candidates.
Auto save / auto resume / progress / ETA included.
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

MICRO_EXITS = [
    "reverse_signal_exit",
    "price_cross_fast_exit",
    "price_cross_slow_exit",
    "adx_fade_exit",
    "atr_guard_exit",
]


def load_top_pending_candidates(candidate_source_dir: Path, symbol: str, timeframe: str, top_n: int) -> List[Dict]:
    tf_csv = candidate_source_dir / "per_timeframe" / f"{symbol}_{timeframe}_pending_logic_results.csv"
    if not tf_csv.exists():
        return []
    df = pd.read_csv(tf_csv)
    if df.empty:
        return []
    df = df.sort_values(["profit_factor", "expectancy", "pnl_sum"], ascending=[False, False, False]).head(top_n)
    return df.to_dict("records")


def rebuild_entries(price_df: pd.DataFrame, feat_df: pd.DataFrame, row: Dict, ema_fast: pd.Series, ema_slow: pd.Series):
    logic = row["logic_variant"]
    close = price_df["close"]
    adx = feat_df["adx_14"].fillna(0.0)
    pb_long = feat_df["pullback_to_ema20_long"].fillna(0).astype(bool)
    pb_short = feat_df["pullback_to_ema20_short"].fillna(0).astype(bool)

    if logic == "ema_cross_long_only":
        signal = (ema_fast > ema_slow).astype(bool)
        prev = signal.shift(1).fillna(False).astype(bool)
        return signal & (~prev), None

    if logic == "ema_cross_short_only":
        signal = (ema_fast < ema_slow).astype(bool)
        prev = signal.shift(1).fillna(False).astype(bool)
        return None, signal & (~prev)

    if logic == "ema_pullback_long_only":
        signal = ((ema_fast > ema_slow) & pb_long).astype(bool)
        prev = signal.shift(1).fillna(False).astype(bool)
        return signal & (~prev), None

    if logic == "ema_pullback_short_only":
        signal = ((ema_fast < ema_slow) & pb_short).astype(bool)
        prev = signal.shift(1).fillna(False).astype(bool)
        return None, signal & (~prev)

    if logic == "adx_ema_cross_long_only":
        signal = ((ema_fast > ema_slow) & (adx >= 25.0)).astype(bool)
        prev = signal.shift(1).fillna(False).astype(bool)
        return signal & (~prev), None

    if logic == "adx_ema_cross_short_only":
        signal = ((ema_fast < ema_slow) & (adx >= 25.0)).astype(bool)
        prev = signal.shift(1).fillna(False).astype(bool)
        return None, signal & (~prev)

    raise ValueError(f"unsupported logic={logic}")


def build_exits(micro_exit_variant: str, price_df: pd.DataFrame, feat_df: pd.DataFrame, ema_fast: pd.Series, ema_slow: pd.Series, is_long: bool):
    close = price_df["close"]
    adx = feat_df["adx_14"].fillna(0.0)
    atr_pct = feat_df["atr_pct_14"].fillna(0.0)

    if micro_exit_variant == "reverse_signal_exit":
        return (ema_fast <= ema_slow) if is_long else (ema_fast >= ema_slow)

    if micro_exit_variant == "price_cross_fast_exit":
        return (close < ema_fast) if is_long else (close > ema_fast)

    if micro_exit_variant == "price_cross_slow_exit":
        return (close < ema_slow) if is_long else (close > ema_slow)

    if micro_exit_variant == "adx_fade_exit":
        return (adx < 20.0)

    if micro_exit_variant == "atr_guard_exit":
        return (atr_pct > atr_pct.rolling(20).mean().fillna(method="bfill") * 1.25)

    raise ValueError(f"unsupported micro_exit_variant={micro_exit_variant}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run micro exit family matrix with VectorBT")
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\Data\Bot\central_market_data\parquet"))
    parser.add_argument("--feature-root", type=Path, default=Path(r"C:\Data\Bot\central_feature_cache\base_features_v1_0_0"))
    parser.add_argument("--candidate-source-dir", type=Path, default=Path(r"C:\Data\Bot\central_backtest_results\vectorbt_pending_logic_matrix_v1_0_0"))
    parser.add_argument("--outdir", type=Path, default=Path(r"C:\Data\Bot\central_backtest_results\vectorbt_micro_exit_matrix_v1_0_0"))
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframes", default="M1,M2,M3,M4,M5,M6,M10,M15,M30,H1,H4,D1")
    parser.add_argument("--top-n-candidates", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    outdir = args.outdir
    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]
    candidate_jobs = []
    for tf in timeframes:
        for row in load_top_pending_candidates(args.candidate_source_dir, args.symbol, tf, args.top_n_candidates):
            for micro_exit in MICRO_EXITS:
                candidate_jobs.append((tf, row, micro_exit))

    overall_total = len(candidate_jobs)
    overall_completed = 0
    start_ts = time.time()
    save_json(outdir / "run_manifest.json", {
        "version": VERSION,
        "generated_at_utc": now_utc_iso(),
        "overall_total_jobs": overall_total,
        "micro_exits": MICRO_EXITS,
        "timeframes": timeframes,
    })

    grouped = {}
    for tf, row, micro_exit in candidate_jobs:
        grouped.setdefault(tf, []).append((row, micro_exit))

    price_cache = {}
    feat_cache = {}
    ema_cache = {}
    summary_rows = []

    for tf in timeframes:
        tf_jobs = grouped.get(tf, [])
        result_csv = outdir / "per_timeframe" / f"{args.symbol}_{tf}_micro_exit_results.csv"
        state_path = outdir / "state" / f"{args.symbol}_{tf}_micro_exit_state.json"
        completed_ids = set() if args.no_resume else load_completed_job_ids(result_csv)

        price_path = args.data_root / f"{args.symbol}_{tf}.parquet"
        feature_path = args.feature_root / f"{args.symbol}_{tf}_base_features.parquet"
        if not price_path.exists() or not feature_path.exists():
            save_json(state_path, {"version": VERSION, "timeframe": tf, "status": "MISSING_INPUT"})
            summary_rows.append({"timeframe": tf, "status": "MISSING_INPUT"})
            continue

        price_cache.setdefault(tf, read_price_parquet(price_path))
        feat_cache.setdefault(tf, read_feature_parquet(feature_path))
        price_df = price_cache[tf]
        feat_df = feat_cache[tf]
        close = price_df["close"].astype(float)
        tf_done = 0

        print(f"[START] timeframe={tf} tf_jobs={len(tf_jobs)}")

        for batch_start in range(0, len(tf_jobs), args.batch_size):
            batch = tf_jobs[batch_start:batch_start + args.batch_size]
            batch_rows = []

            for row, micro_exit in batch:
                fast = int(row["ema_fast"])
                slow = int(row["ema_slow"])
                logic_variant = row["logic_variant"]
                strategy_family = row["strategy_family"]
                job_id = f"{args.symbol}_{tf}_{logic_variant}_ema_fast_{fast:03d}_ema_slow_{slow:03d}_{micro_exit}"
                if job_id in completed_ids:
                    tf_done += 1
                    overall_completed += 1
                    continue

                ema_cache.setdefault((tf, fast), make_ema(close, fast))
                ema_cache.setdefault((tf, slow), make_ema(close, slow))
                ema_fast = ema_cache[(tf, fast)]
                ema_slow = ema_cache[(tf, slow)]

                long_entries, short_entries = rebuild_entries(price_df, feat_df, row, ema_fast, ema_slow)
                if long_entries is not None:
                    exits = build_exits(micro_exit, price_df, feat_df, ema_fast, ema_slow, is_long=True)
                    pf = run_pf_from_signals(close=close, entries=long_entries, exits=exits)
                    side = "LONG"
                else:
                    short_exits = build_exits(micro_exit, price_df, feat_df, ema_fast, ema_slow, is_long=False)
                    pf = run_pf_from_signals(close=close, entries=pd.Series(False, index=close.index), exits=pd.Series(False, index=close.index), short_entries=short_entries, short_exits=short_exits)
                    side = "SHORT"

                metrics = portfolio_metrics(pf)
                batch_rows.append({
                    "job_id": job_id,
                    "timeframe": tf,
                    "strategy_family": strategy_family,
                    "logic_variant": logic_variant,
                    "side": side,
                    "ema_fast": fast,
                    "ema_slow": slow,
                    "micro_exit_variant": micro_exit,
                    "regime_summary": row.get("regime_summary"),
                    **metrics,
                    "status": "DONE",
                    "created_at_utc": now_utc_iso(),
                })
                tf_done += 1
                overall_completed += 1

            if batch_rows:
                append_frame_csv(result_csv, pd.DataFrame(batch_rows))

            live = build_live_progress(VERSION, outdir, tf, "run_vectorbt_micro_exit_matrix", overall_total, overall_completed, start_ts)
            save_json(state_path, {
                "version": VERSION,
                "updated_at_utc": now_utc_iso(),
                "timeframe": tf,
                "completed_jobs": tf_done,
                "total_jobs": len(tf_jobs),
                "remaining_jobs": len(tf_jobs) - tf_done,
                "progress_pct": round((tf_done / len(tf_jobs)) * 100.0, 4) if tf_jobs else 100.0,
                "result_csv": str(result_csv),
            })
            print(f"[PROGRESS] timeframe={tf} batch={(batch_start // args.batch_size)+1} tf_completed={tf_done}/{len(tf_jobs)} tf_progress_pct={(tf_done / len(tf_jobs) * 100.0) if tf_jobs else 100.0:.2f} overall_completed={overall_completed}/{overall_total} overall_progress_pct={live['overall_progress_pct']:.2f} overall_eta_remaining_min={live['overall_eta_remaining_min']}")

        summary_rows.append({"timeframe": tf, "status": "DONE", "completed_jobs": tf_done, "total_jobs": len(tf_jobs)})
        print(f"[DONE] timeframe={tf} jobs_total={len(tf_jobs)} jobs_completed={tf_done} result_csv={result_csv}")

    leaderboard_rows = []
    for tf in timeframes:
        result_csv = outdir / "per_timeframe" / f"{args.symbol}_{tf}_micro_exit_results.csv"
        if result_csv.exists():
            df = pd.read_csv(result_csv)
            if not df.empty:
                top_df = df.sort_values(["profit_factor", "expectancy", "pnl_sum"], ascending=[False, False, False]).head(50)
                top_df.insert(0, "leaderboard_rank", range(1, len(top_df) + 1))
                leaderboard_rows.append(top_df)
    if leaderboard_rows:
        pd.concat(leaderboard_rows, ignore_index=True).to_csv(outdir / "leaderboard_top50_per_tf.csv", index=False)

    summary = {
        "version": VERSION,
        "generated_at_utc": now_utc_iso(),
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

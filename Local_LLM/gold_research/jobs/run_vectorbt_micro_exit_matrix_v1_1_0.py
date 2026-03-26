#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code Name : run_vectorbt_micro_exit_matrix_v1_1_0.py
Version   : v1.1.0
Path      : C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_micro_exit_matrix_v1_1_0.py
Run       : python C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_micro_exit_matrix_v1_1_0.py --data-root C:\Data\Bot\central_market_data\parquet --feature-root C:\Data\Bot\central_feature_cache\base_features_v1_0_0 --candidate-source-dir C:\Data\Bot\central_backtest_results\vectorbt_pending_logic_matrix_v1_0_0 --outdir C:\Data\Bot\central_backtest_results\vectorbt_micro_exit_matrix_v1_1_0

Production-safe speed upgrade for micro-exit research without changing the external folder/result structure.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

VERSION = "v1.1.0"

MICRO_EXITS: List[str] = [
    "reverse_signal_exit",
    "price_cross_fast_exit",
    "price_cross_slow_exit",
    "adx_fade_exit",
    "atr_guard_exit",
]


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


def load_top_pending_candidates(candidate_source_dir: Path, symbol: str, timeframe: str, top_n: int) -> List[Dict]:
    tf_csv = candidate_source_dir / "per_timeframe" / f"{symbol}_{timeframe}_pending_logic_results.csv"
    if not tf_csv.exists():
        return []
    df = pd.read_csv(tf_csv)
    if df.empty:
        return []
    df = df.sort_values(["profit_factor", "expectancy", "pnl_sum"], ascending=[False, False, False]).head(top_n)
    return df.to_dict("records")


def rebuild_entries(
    feat_df: pd.DataFrame,
    row: Dict,
    ema_fast: pd.Series,
    ema_slow: pd.Series,
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    logic = str(row["logic_variant"]).strip()
    adx = feat_df["adx_14"].fillna(0.0)
    pb_long = feat_df["pullback_to_ema20_long"].fillna(0).astype(bool)
    pb_short = feat_df["pullback_to_ema20_short"].fillna(0).astype(bool)

    if logic == "ema_cross_long_only":
        signal = (ema_fast > ema_slow).astype(bool)
        prev = signal.shift(1, fill_value=False).astype(bool)
        return (signal & (~prev)).astype(bool), None

    if logic == "ema_cross_short_only":
        signal = (ema_fast < ema_slow).astype(bool)
        prev = signal.shift(1, fill_value=False).astype(bool)
        return None, (signal & (~prev)).astype(bool)

    if logic == "ema_pullback_long_only":
        signal = ((ema_fast > ema_slow) & pb_long).astype(bool)
        prev = signal.shift(1, fill_value=False).astype(bool)
        return (signal & (~prev)).astype(bool), None

    if logic == "ema_pullback_short_only":
        signal = ((ema_fast < ema_slow) & pb_short).astype(bool)
        prev = signal.shift(1, fill_value=False).astype(bool)
        return None, (signal & (~prev)).astype(bool)

    if logic == "adx_ema_cross_long_only":
        signal = ((ema_fast > ema_slow) & (adx >= 25.0)).astype(bool)
        prev = signal.shift(1, fill_value=False).astype(bool)
        return (signal & (~prev)).astype(bool), None

    if logic == "adx_ema_cross_short_only":
        signal = ((ema_fast < ema_slow) & (adx >= 25.0)).astype(bool)
        prev = signal.shift(1, fill_value=False).astype(bool)
        return None, (signal & (~prev)).astype(bool)

    raise ValueError(f"unsupported logic={logic}")


def build_exit_series(
    micro_exit_variant: str,
    close: pd.Series,
    adx: pd.Series,
    atr_pct: pd.Series,
    atr_pct_roll20_mean: pd.Series,
    ema_fast: pd.Series,
    ema_slow: pd.Series,
    is_long: bool,
) -> pd.Series:
    if micro_exit_variant == "reverse_signal_exit":
        return ((ema_fast <= ema_slow) if is_long else (ema_fast >= ema_slow)).astype(bool)

    if micro_exit_variant == "price_cross_fast_exit":
        return ((close < ema_fast) if is_long else (close > ema_fast)).astype(bool)

    if micro_exit_variant == "price_cross_slow_exit":
        return ((close < ema_slow) if is_long else (close > ema_slow)).astype(bool)

    if micro_exit_variant == "adx_fade_exit":
        return (adx < 20.0).astype(bool)

    if micro_exit_variant == "atr_guard_exit":
        return (atr_pct > (atr_pct_roll20_mean * 1.25)).astype(bool)

    raise ValueError(f"unsupported micro_exit_variant={micro_exit_variant}")


def chunked(items: Sequence[Dict], size: int) -> Iterable[Sequence[Dict]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def to_stat_map(obj, columns: Sequence[str], default=np.nan) -> Dict[str, float]:
    if isinstance(obj, pd.Series):
        return {str(col): float(obj.get(col, default)) for col in columns}
    try:
        scalar = float(obj)
        return {str(col): scalar for col in columns}
    except Exception:
        return {str(col): default for col in columns}


def extract_portfolio_metrics_batch(pf: vbt.Portfolio, columns: Sequence[str]) -> Dict[str, Dict[str, float]]:
    total_return_map = to_stat_map(pf.total_return(), columns)
    max_dd_map = to_stat_map(pf.max_drawdown(), columns)
    trade_count_map = to_stat_map(pf.trades.count(), columns, default=0.0)
    win_rate_map = to_stat_map(pf.trades.win_rate(), columns)
    profit_factor_map = to_stat_map(pf.trades.profit_factor(), columns)
    expectancy_map = to_stat_map(pf.trades.expectancy(), columns)
    avg_win_map = to_stat_map(pf.trades.winning.pnl.mean(), columns)
    avg_loss_map = to_stat_map(pf.trades.losing.pnl.mean(), columns)

    out: Dict[str, Dict[str, float]] = {}
    for col in columns:
        win_rate_value = float(win_rate_map.get(col, np.nan))
        out[str(col)] = {
            "trade_count": int(round(float(trade_count_map.get(col, 0.0)))),
            "win_rate_pct": win_rate_value * 100.0 if win_rate_value == win_rate_value else np.nan,
            "profit_factor": float(profit_factor_map.get(col, np.nan)),
            "expectancy": float(expectancy_map.get(col, np.nan)),
            "pnl_sum": float(total_return_map.get(col, np.nan)),
            "max_drawdown": float(max_dd_map.get(col, np.nan)),
            "avg_win": float(avg_win_map.get(col, np.nan)),
            "avg_loss": float(avg_loss_map.get(col, np.nan)),
        }
    return out


def evaluate_group_batch(
    close: pd.Series,
    adx: pd.Series,
    atr_pct: pd.Series,
    atr_pct_roll20_mean: pd.Series,
    ema_fast: pd.Series,
    ema_slow: pd.Series,
    entries: pd.Series,
    jobs: Sequence[Dict],
    side: str,
) -> List[Dict]:
    if not jobs:
        return []

    columns = [str(job["job_id"]) for job in jobs]
    entries_bool = entries.fillna(False).astype(bool)
    entries_df = pd.concat([entries_bool.rename(col) for col in columns], axis=1)
    exits_df = pd.concat(
        [
            build_exit_series(
                micro_exit_variant=str(job["micro_exit_variant"]),
                close=close,
                adx=adx,
                atr_pct=atr_pct,
                atr_pct_roll20_mean=atr_pct_roll20_mean,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                is_long=(side == "LONG"),
            ).fillna(False).astype(bool).rename(str(job["job_id"]))
            for job in jobs
        ],
        axis=1,
    )

    if side == "LONG":
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries_df,
            exits=exits_df,
            init_cash=100_000.0,
            fees=0.0,
            slippage=0.0,
        )
    else:
        false_df = pd.DataFrame(False, index=close.index, columns=columns)
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=false_df,
            exits=false_df,
            short_entries=entries_df,
            short_exits=exits_df,
            init_cash=100_000.0,
            fees=0.0,
            slippage=0.0,
        )

    metrics_map = extract_portfolio_metrics_batch(pf, columns)
    created_at = now_utc_iso()
    rows: List[Dict] = []
    for job in jobs:
        job_id = str(job["job_id"])
        rows.append(
            {
                "job_id": job_id,
                "timeframe": str(job["timeframe"]),
                "strategy_family": str(job["strategy_family"]),
                "logic_variant": str(job["logic_variant"]),
                "side": side,
                "ema_fast": int(job["ema_fast"]),
                "ema_slow": int(job["ema_slow"]),
                "micro_exit_variant": str(job["micro_exit_variant"]),
                "regime_summary": job.get("regime_summary"),
                **metrics_map[job_id],
                "status": "DONE",
                "created_at_utc": created_at,
            }
        )
    return rows


def build_tf_candidate_jobs(
    symbol: str,
    timeframe: str,
    candidate_rows: Sequence[Dict],
    completed_ids: set,
) -> Tuple[List[Dict], int]:
    jobs: List[Dict] = []
    resume_hits = 0
    for row in candidate_rows:
        fast = int(row["ema_fast"])
        slow = int(row["ema_slow"])
        logic_variant = str(row["logic_variant"])
        strategy_family = str(row.get("strategy_family", "pending_logic"))
        for micro_exit in MICRO_EXITS:
            job_id = f"{symbol}_{timeframe}_{logic_variant}_ema_fast_{fast:03d}_ema_slow_{slow:03d}_{micro_exit}"
            payload = {
                "job_id": job_id,
                "timeframe": timeframe,
                "strategy_family": strategy_family,
                "logic_variant": logic_variant,
                "ema_fast": fast,
                "ema_slow": slow,
                "micro_exit_variant": micro_exit,
                "regime_summary": row.get("regime_summary"),
                "_source_row": row,
            }
            if job_id in completed_ids:
                resume_hits += 1
            else:
                jobs.append(payload)
    return jobs, resume_hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Run micro exit family matrix with VectorBT (batched v1.1.0)")
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\Data\Bot\central_market_data\parquet"))
    parser.add_argument("--feature-root", type=Path, default=Path(r"C:\Data\Bot\central_feature_cache\base_features_v1_0_0"))
    parser.add_argument("--candidate-source-dir", type=Path, default=Path(r"C:\Data\Bot\central_backtest_results\vectorbt_pending_logic_matrix_v1_0_0"))
    parser.add_argument("--outdir", type=Path, default=Path(r"C:\Data\Bot\central_backtest_results\vectorbt_micro_exit_matrix_v1_1_0"))
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframes", default="M1,M2,M3,M4,M5,M6,M10,M15,M30,H1,H4,D1")
    parser.add_argument("--top-n-candidates", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--portfolio-chunk-size", type=int, default=64)
    parser.add_argument("--progress-every-batches", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    outdir = args.outdir
    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]
    outdir.mkdir(parents=True, exist_ok=True)

    overall_total = 0
    for tf in timeframes:
        overall_total += len(load_top_pending_candidates(args.candidate_source_dir, args.symbol, tf, args.top_n_candidates)) * len(MICRO_EXITS)

    overall_completed = 0
    start_ts = time.time()
    save_json(
        outdir / "run_manifest.json",
        {
            "version": VERSION,
            "generated_at_utc": now_utc_iso(),
            "overall_total_jobs": overall_total,
            "micro_exits": MICRO_EXITS,
            "timeframes": timeframes,
            "batch_size": int(args.batch_size),
            "portfolio_chunk_size": int(args.portfolio_chunk_size),
        },
    )

    summary_rows: List[Dict] = []

    for tf in timeframes:
        result_csv = outdir / "per_timeframe" / f"{args.symbol}_{tf}_micro_exit_results.csv"
        state_path = outdir / "state" / f"{args.symbol}_{tf}_micro_exit_state.json"
        completed_ids = set() if args.no_resume else load_completed_job_ids(result_csv)

        candidate_rows = load_top_pending_candidates(args.candidate_source_dir, args.symbol, tf, args.top_n_candidates)
        tf_total = len(candidate_rows) * len(MICRO_EXITS)

        if tf_total == 0:
            save_json(
                state_path,
                {
                    "version": VERSION,
                    "updated_at_utc": now_utc_iso(),
                    "timeframe": tf,
                    "completed_jobs": 0,
                    "total_jobs": 0,
                    "remaining_jobs": 0,
                    "progress_pct": 100.0,
                    "result_csv": str(result_csv),
                    "status": "EMPTY",
                },
            )
            summary_rows.append({"timeframe": tf, "status": "EMPTY", "completed_jobs": 0, "total_jobs": 0})
            print(f"[SKIP] timeframe={tf} no candidate jobs")
            continue

        price_path = args.data_root / f"{args.symbol}_{tf}.parquet"
        feature_path = args.feature_root / f"{args.symbol}_{tf}_base_features.parquet"
        if not price_path.exists() or not feature_path.exists():
            save_json(
                state_path,
                {
                    "version": VERSION,
                    "timeframe": tf,
                    "status": "MISSING_INPUT",
                    "updated_at_utc": now_utc_iso(),
                    "price_path": str(price_path),
                    "feature_path": str(feature_path),
                },
            )
            summary_rows.append({"timeframe": tf, "status": "MISSING_INPUT", "completed_jobs": 0, "total_jobs": tf_total})
            print(f"[MISSING_INPUT] timeframe={tf} price_exists={price_path.exists()} feature_exists={feature_path.exists()}")
            continue

        price_df = read_price_parquet(price_path)
        feat_df = read_feature_parquet(feature_path)
        close = price_df["close"].astype(float)
        adx = feat_df["adx_14"].fillna(0.0).astype(float)
        atr_pct = feat_df["atr_pct_14"].fillna(0.0).astype(float)
        atr_pct_roll20_mean = atr_pct.rolling(20).mean().bfill().astype(float)

        tf_jobs, resume_hits = build_tf_candidate_jobs(args.symbol, tf, candidate_rows, completed_ids)
        tf_done = resume_hits
        overall_completed += resume_hits

        print(f"[START] timeframe={tf} total_jobs={tf_total} pending_jobs={len(tf_jobs)} resume_hits={resume_hits}")

        ema_cache: Dict[int, pd.Series] = {}
        grouped_jobs: Dict[Tuple[str, int, int], List[Dict]] = {}
        for job in tf_jobs:
            key = (str(job["logic_variant"]), int(job["ema_fast"]), int(job["ema_slow"]))
            grouped_jobs.setdefault(key, []).append(job)

        processed_groups = 0
        total_groups = len(grouped_jobs)

        for (logic_variant, fast, slow), group_jobs in grouped_jobs.items():
            if fast not in ema_cache:
                ema_cache[fast] = make_ema(close, fast)
            if slow not in ema_cache:
                ema_cache[slow] = make_ema(close, slow)
            ema_fast = ema_cache[fast]
            ema_slow = ema_cache[slow]

            source_row = dict(group_jobs[0]["_source_row"])
            source_row["logic_variant"] = logic_variant
            long_entries, short_entries = rebuild_entries(feat_df, source_row, ema_fast, ema_slow)

            if long_entries is not None:
                side = "LONG"
                entry_series = long_entries
            elif short_entries is not None:
                side = "SHORT"
                entry_series = short_entries
            else:
                raise RuntimeError(f"Unable to determine side for logic_variant={logic_variant}")

            jobs_sorted = sorted(group_jobs, key=lambda x: str(x["micro_exit_variant"]))

            for job_chunk in chunked(jobs_sorted, max(1, int(args.portfolio_chunk_size))):
                rows = evaluate_group_batch(
                    close=close,
                    adx=adx,
                    atr_pct=atr_pct,
                    atr_pct_roll20_mean=atr_pct_roll20_mean,
                    ema_fast=ema_fast,
                    ema_slow=ema_slow,
                    entries=entry_series,
                    jobs=job_chunk,
                    side=side,
                )
                if rows:
                    append_frame_csv(result_csv, pd.DataFrame(rows))
                    tf_done += len(rows)
                    overall_completed += len(rows)

            processed_groups += 1
            if processed_groups % max(1, int(args.progress_every_batches)) == 0 or processed_groups == total_groups:
                live = build_live_progress(VERSION, outdir, tf, "run_vectorbt_micro_exit_matrix", overall_total, overall_completed, start_ts)
                save_json(
                    state_path,
                    {
                        "version": VERSION,
                        "updated_at_utc": now_utc_iso(),
                        "timeframe": tf,
                        "completed_jobs": tf_done,
                        "total_jobs": tf_total,
                        "remaining_jobs": max(tf_total - tf_done, 0),
                        "progress_pct": round((tf_done / tf_total) * 100.0, 4) if tf_total else 100.0,
                        "result_csv": str(result_csv),
                        "groups_completed": processed_groups,
                        "groups_total": total_groups,
                    },
                )
                print(
                    f"[PROGRESS] timeframe={tf} groups={processed_groups}/{total_groups} "
                    f"tf_completed={tf_done}/{tf_total} tf_progress_pct={(tf_done / tf_total * 100.0) if tf_total else 100.0:.2f} "
                    f"overall_completed={overall_completed}/{overall_total} overall_progress_pct={live['overall_progress_pct']:.2f} "
                    f"overall_eta_remaining_min={live['overall_eta_remaining_min']}"
                )

        summary_rows.append({"timeframe": tf, "status": "DONE", "completed_jobs": tf_done, "total_jobs": tf_total})
        print(f"[DONE] timeframe={tf} jobs_total={tf_total} jobs_completed={tf_done} result_csv={result_csv}")

    leaderboard_rows: List[pd.DataFrame] = []
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

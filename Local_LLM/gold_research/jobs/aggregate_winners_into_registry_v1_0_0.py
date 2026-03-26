
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_winners_into_registry_v1_0_0.py
Version: v1.0.0

Aggregate backtest outputs into registry/database-ready tables.
Inputs:
- uncovered results
- pending logic results
- micro exit results
Outputs:
- registry_candidates.csv
- active_winner_registry.csv
- active_winner_registry.json
- timeframe_leaderboard.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


VERSION = "v1.0.0"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def load_result_group(root: Path, suffix: str) -> pd.DataFrame:
    per_tf = root / "per_timeframe"
    if not per_tf.exists():
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for file in sorted(per_tf.glob(f"*{suffix}.csv")):
        try:
            df = pd.read_csv(file)
            if not df.empty:
                df["source_file"] = str(file)
                frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_score(df: pd.DataFrame) -> pd.Series:
    pf = df["profit_factor"].fillna(0.0).clip(lower=0.0, upper=10.0)
    exp = df["expectancy"].fillna(0.0).clip(lower=-10.0, upper=10.0)
    pnl = df["pnl_sum"].fillna(0.0).clip(lower=-10.0, upper=10.0)
    dd = df["max_drawdown"].fillna(0.0).abs().clip(lower=0.0, upper=1.0)
    trades = df["trade_count"].fillna(0.0).clip(lower=0.0, upper=5000.0)
    trade_bonus = np.log1p(trades) / np.log(5001.0)
    score = (pf * 0.40) + (exp * 0.25) + (pnl * 0.20) + (trade_bonus * 0.15) - (dd * 0.50)
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate winners into registry")
    parser.add_argument("--feature-root", type=Path, default=Path(r"C:\Data\Bot\central_feature_cache\base_features_v1_0_0"))
    parser.add_argument("--uncovered-dir", type=Path, default=Path(r"C:\Data\Bot\central_backtest_results\vectorbt_uncovered_matrix_v1_0_0"))
    parser.add_argument("--pending-dir", type=Path, default=Path(r"C:\Data\Bot\central_backtest_results\vectorbt_pending_logic_matrix_v1_0_0"))
    parser.add_argument("--micro-exit-dir", type=Path, default=Path(r"C:\Data\Bot\central_backtest_results\vectorbt_micro_exit_matrix_v1_0_0"))
    parser.add_argument("--registry-root", type=Path, default=Path(r"C:\Data\Bot\central_strategy_registry\winner_registry_v1_0_0"))
    parser.add_argument("--top-per-timeframe", type=int, default=5)
    args = parser.parse_args()

    reg = args.registry_root
    reg.mkdir(parents=True, exist_ok=True)

    uncovered = load_result_group(args.uncovered_dir, "_uncovered_results")
    pending = load_result_group(args.pending_dir, "_pending_logic_results")
    micro = load_result_group(args.micro_exit_dir, "_micro_exit_results")

    frames = []
    if not uncovered.empty:
        uncovered["result_stage"] = "uncovered"
        frames.append(uncovered)
    if not pending.empty:
        pending["result_stage"] = "pending_logic"
        frames.append(pending)
    if not micro.empty:
        micro["result_stage"] = "micro_exit"
        frames.append(micro)

    if not frames:
        summary = {
            "version": VERSION,
            "generated_at_utc": now_utc_iso(),
            "status": "NO_RESULTS",
            "registry_root": str(reg),
        }
        save_json(reg / "summary.json", summary)
        print(f"[DONE] version={VERSION}")
        print(f"[DONE] status=NO_RESULTS")
        print(f"[DONE] summary_json={reg / 'summary.json'}")
        return

    all_results = pd.concat(frames, ignore_index=True)
    for col in ["profit_factor", "expectancy", "pnl_sum", "max_drawdown", "trade_count"]:
        if col not in all_results.columns:
            all_results[col] = np.nan
    all_results["score"] = compute_score(all_results)
    all_results["registry_version"] = VERSION

    all_results.to_csv(reg / "registry_candidates.csv", index=False)

    leaderboards = []
    winners = []
    for timeframe, tf_df in all_results.groupby("timeframe"):
        tf_df = tf_df.sort_values(["score", "profit_factor", "expectancy", "pnl_sum"], ascending=[False, False, False, False]).reset_index(drop=True)
        tf_df["timeframe_rank"] = range(1, len(tf_df) + 1)
        leaderboards.append(tf_df)
        winners.append(tf_df.head(args.top_per_timeframe))

    timeframe_leaderboard = pd.concat(leaderboards, ignore_index=True)
    timeframe_leaderboard.to_csv(reg / "timeframe_leaderboard.csv", index=False)

    active_winners = pd.concat(winners, ignore_index=True)
    active_winners["winner_status"] = "ACTIVE"
    active_winners["approved_at_utc"] = now_utc_iso()
    active_winners.to_csv(reg / "active_winner_registry.csv", index=False)

    active_json = active_winners.to_dict("records")
    save_json(reg / "active_winner_registry.json", active_json)

    package_rows = []
    for _, row in active_winners.iterrows():
        package_rows.append({
            "strategy_id": row.get("job_id"),
            "timeframe": row.get("timeframe"),
            "strategy_family": row.get("strategy_family"),
            "logic_variant": row.get("logic_variant"),
            "ema_fast": row.get("ema_fast"),
            "ema_slow": row.get("ema_slow"),
            "micro_exit_variant": row.get("micro_exit_variant"),
            "score": row.get("score"),
            "profit_factor": row.get("profit_factor"),
            "expectancy": row.get("expectancy"),
            "trade_count": row.get("trade_count"),
            "registry_version": VERSION,
            "is_active": True,
        })
    pd.DataFrame(package_rows).to_csv(reg / "database_ready_strategy_packages.csv", index=False)

    summary = {
        "version": VERSION,
        "generated_at_utc": now_utc_iso(),
        "registry_root": str(reg),
        "candidate_rows": int(len(all_results)),
        "active_winners": int(len(active_winners)),
        "top_per_timeframe": int(args.top_per_timeframe),
        "files": {
            "registry_candidates_csv": str(reg / "registry_candidates.csv"),
            "timeframe_leaderboard_csv": str(reg / "timeframe_leaderboard.csv"),
            "active_winner_registry_csv": str(reg / "active_winner_registry.csv"),
            "active_winner_registry_json": str(reg / "active_winner_registry.json"),
            "database_ready_strategy_packages_csv": str(reg / "database_ready_strategy_packages.csv"),
        }
    }
    save_json(reg / "summary.json", summary)
    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] registry_root={reg}")
    print(f"[DONE] candidate_rows={len(all_results)}")
    print(f"[DONE] active_winners={len(active_winners)}")
    print(f"[DONE] summary_json={reg / 'summary.json'}")
    print("=" * 120)


if __name__ == "__main__":
    main()

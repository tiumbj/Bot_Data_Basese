# ==================================================================================================
# FILE: summarize_signal_debug.py
# PATH: C:\Data\Bot\Local_LLM\gold_research\jobs\summarize_signal_debug.py
# VERSION: v1.0.2
#
# CHANGELOG:
# - v1.0.2
#   1) Rewrite whole file to match run_signal_debug_shard_fast.py v2.0.0 output schema
#   2) Remove dependency on deprecated/old column names such as base_total_count
#   3) Keep summary outputs decision-ready for next-stage VectorBT promotion
#   4) Make sorting and grouping robust against minor schema drift
#
# PURPOSE:
# - Summarize signal debug shard outputs
# - Identify survivors and dominant kill stages
# - Produce promotion-ready CSV/JSON/TXT artifacts
# ==================================================================================================

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import polars as pl

VERSION = "v1.0.2"

INPUT_JOB_DEBUG_ROWS = "job_debug_rows.jsonl"
INPUT_LAYER_FAILURE_COUNTS = "layer_failure_counts.json"
INPUT_STAGE_SUMMARY = "stage_summary.json"
INPUT_RUN_SUMMARY = "run_summary.json"

OUT_DECISION_SUMMARY_JSON = "signal_debug_decision_summary.json"
OUT_TOP_SURVIVORS_CSV = "top_surviving_jobs.csv"
OUT_TOP_GROUPS_CSV = "top_surviving_groups.csv"
OUT_KILL_BY_TF_CSV = "kill_stage_by_timeframe.csv"
OUT_RECOMMENDATION_TXT = "signal_debug_recommendation.txt"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_file(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"Required file not found: {path}")


def safe_float(value: Any, digits: int = 6) -> float:
    try:
        return round(float(value), digits)
    except Exception:
        return 0.0


def existing_columns(df: pl.DataFrame, wanted: List[str]) -> List[str]:
    return [c for c in wanted if c in df.columns]


def build_recommendation_text(
    total_jobs: int,
    survived_jobs: int,
    top_kill_stage: str,
    top_survivor_rows: List[Dict[str, Any]],
    top_group_rows: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append(f"VERSION: {VERSION}")
    lines.append("PURPOSE: Decision-ready summary from signal debugger")
    lines.append("")

    lines.append("CORE VERDICT:")
    if survived_jobs == 0:
        lines.append("- No jobs survived to executable trades in this shard.")
        lines.append(f"- Main blocker stage: {top_kill_stage}")
        lines.append("- Do NOT promote this shard to VectorBT yet.")
        lines.append("- Relax the dominant killer layer first.")
    elif survived_jobs < total_jobs:
        lines.append(f"- Survived jobs in shard: {survived_jobs}/{total_jobs}")
        lines.append(f"- Main blocker stage among failed jobs: {top_kill_stage}")
        lines.append("- Promote only survivor groups to next-stage VectorBT.")
    else:
        lines.append(f"- All jobs survived in shard: {survived_jobs}/{total_jobs}")
        lines.append("- This shard is too permissive for kill-stage diagnosis.")
        lines.append("- Next step: move to true VectorBT validation or tighten screening rules.")

    lines.append("")
    lines.append("TOP SURVIVING JOBS:")
    if not top_survivor_rows:
        lines.append("- None")
    else:
        for idx, row in enumerate(top_survivor_rows[:10], start=1):
            lines.append(
                f"- #{idx} tf={row.get('timeframe', '')} "
                f"strategy={row.get('strategy_family', '')} "
                f"entry={row.get('entry_logic', '')} "
                f"side={row.get('side_policy', '')} "
                f"pre_side={row.get('pre_side_total_count', 0)} "
                f"exec={row.get('executable_total_count', 0)} "
                f"est_trades={row.get('estimated_trade_count', 0)} "
                f"kill_stage={row.get('kill_stage', '')}"
            )

    lines.append("")
    lines.append("TOP SURVIVING GROUPS:")
    if not top_group_rows:
        lines.append("- None")
    else:
        for idx, row in enumerate(top_group_rows[:10], start=1):
            lines.append(
                f"- #{idx} tf={row.get('timeframe', '')} "
                f"strategy={row.get('strategy_family', '')} "
                f"entry={row.get('entry_logic', '')} "
                f"side={row.get('side_policy', '')} "
                f"survived_jobs={row.get('survived_jobs', 0)} "
                f"survival_pct={safe_float(row.get('survival_pct', 0.0))} "
                f"mean_exec={safe_float(row.get('mean_executable_total_count', 0.0))} "
                f"mean_est_trades={safe_float(row.get('mean_estimated_trade_count', 0.0))}"
            )

    lines.append("")
    lines.append("NEXT DECISION:")
    if survived_jobs == 0:
        lines.append("- Build relaxed signal logic variant first.")
        lines.append("- Do not start large-scale VectorBT yet.")
    elif survived_jobs < total_jobs:
        lines.append("- Promote surviving groups into the initial VectorBT seed set.")
        lines.append("- Keep micro exit as the base layer.")
    else:
        lines.append("- Promote this shard to VectorBT directly.")
        lines.append("- Use real PnL, PF, DD, and trade distribution to separate winners from false positives.")

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize signal-debug outputs into decision-ready artifacts.")
    parser.add_argument(
        "--debug-dir",
        required=True,
        help="Directory that contains job_debug_rows.jsonl and other debug outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug_dir = Path(args.debug_dir)

    job_debug_path = debug_dir / INPUT_JOB_DEBUG_ROWS
    layer_failure_path = debug_dir / INPUT_LAYER_FAILURE_COUNTS
    stage_summary_path = debug_dir / INPUT_STAGE_SUMMARY
    run_summary_path = debug_dir / INPUT_RUN_SUMMARY

    out_summary_json = debug_dir / OUT_DECISION_SUMMARY_JSON
    out_top_survivors_csv = debug_dir / OUT_TOP_SURVIVORS_CSV
    out_top_groups_csv = debug_dir / OUT_TOP_GROUPS_CSV
    out_kill_by_tf_csv = debug_dir / OUT_KILL_BY_TF_CSV
    out_recommendation_txt = debug_dir / OUT_RECOMMENDATION_TXT

    require_file(job_debug_path)
    require_file(layer_failure_path)
    require_file(stage_summary_path)
    require_file(run_summary_path)

    df = pl.read_ndjson(job_debug_path)
    if df.height == 0:
        raise RuntimeError(f"Empty debugger output: {job_debug_path}")

    required_cols = [
        "job_id",
        "timeframe",
        "strategy_family",
        "entry_logic",
        "micro_exit",
        "regime_filter",
        "cooldown_bars",
        "side_policy",
        "volatility_filter",
        "trend_strength_filter",
        "pre_side_total_count",
        "final_total_count",
        "executable_total_count",
        "estimated_trade_count",
        "kill_stage",
        "kill_reason",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in job_debug_rows.jsonl: {missing}")

    layer_failure = read_json(layer_failure_path)
    stage_summary = read_json(stage_summary_path)
    run_summary = read_json(run_summary_path)

    total_jobs = int(df.height)
    survived_jobs = int(
        df.select(
            (pl.col("estimated_trade_count") > 0)
            .cast(pl.Int64)
            .sum()
            .alias("survived_jobs")
        ).item()
    )

    survived_df = (
        df.filter(pl.col("estimated_trade_count") > 0)
        .sort(
            by=["estimated_trade_count", "executable_total_count", "pre_side_total_count"],
            descending=[True, True, True],
        )
    )

    failed_df = df.filter(pl.col("estimated_trade_count") <= 0)

    if failed_df.height > 0:
        top_kill_stage = (
            failed_df.group_by("kill_stage")
            .agg(pl.len().alias("count"))
            .sort(by="count", descending=True)
            .select("kill_stage")
            .item()
        )
    else:
        top_kill_stage = "SURVIVED"

    top_survivor_cols = existing_columns(
        survived_df,
        [
            "job_id",
            "timeframe",
            "strategy_family",
            "entry_logic",
            "micro_exit",
            "regime_filter",
            "cooldown_bars",
            "side_policy",
            "volatility_filter",
            "trend_strength_filter",
            "pre_side_total_count",
            "final_total_count",
            "executable_total_count",
            "estimated_trade_count",
            "kill_stage",
            "kill_reason",
        ],
    )
    top_survivors_df = survived_df.select(top_survivor_cols)
    top_survivors_df.write_csv(out_top_survivors_csv)

    top_groups_df = (
        df.with_columns(
            (pl.col("estimated_trade_count") > 0).cast(pl.Int64).alias("survived_flag")
        )
        .group_by(
            [
                "timeframe",
                "strategy_family",
                "entry_logic",
                "micro_exit",
                "side_policy",
                "regime_filter",
                "volatility_filter",
                "trend_strength_filter",
            ]
        )
        .agg(
            [
                pl.len().alias("jobs"),
                pl.col("survived_flag").sum().alias("survived_jobs"),
                pl.col("pre_side_total_count").mean().alias("mean_pre_side_total_count"),
                pl.col("final_total_count").mean().alias("mean_final_total_count"),
                pl.col("executable_total_count").mean().alias("mean_executable_total_count"),
                pl.col("estimated_trade_count").mean().alias("mean_estimated_trade_count"),
            ]
        )
        .with_columns(
            pl.when(pl.col("jobs") > 0)
            .then((pl.col("survived_jobs") / pl.col("jobs")) * 100.0)
            .otherwise(0.0)
            .alias("survival_pct")
        )
        .sort(
            by=["survival_pct", "mean_estimated_trade_count", "mean_executable_total_count"],
            descending=[True, True, True],
        )
    )
    top_groups_df.write_csv(out_top_groups_csv)

    kill_by_tf_df = (
        df.group_by(["timeframe", "kill_stage"])
        .agg(pl.len().alias("jobs"))
        .sort(by=["timeframe", "jobs"], descending=[False, True])
    )
    kill_by_tf_df.write_csv(out_kill_by_tf_csv)

    best_timeframes_df = (
        df.with_columns(
            (pl.col("estimated_trade_count") > 0).cast(pl.Int64).alias("survived_flag")
        )
        .group_by("timeframe")
        .agg(
            [
                pl.len().alias("jobs"),
                pl.col("survived_flag").sum().alias("survived_jobs"),
                pl.col("pre_side_total_count").mean().alias("mean_pre_side_total_count"),
                pl.col("executable_total_count").mean().alias("mean_executable_total_count"),
                pl.col("estimated_trade_count").mean().alias("mean_estimated_trade_count"),
            ]
        )
        .with_columns(
            pl.when(pl.col("jobs") > 0)
            .then((pl.col("survived_jobs") / pl.col("jobs")) * 100.0)
            .otherwise(0.0)
            .alias("survival_pct")
        )
        .sort(by=["survival_pct", "mean_estimated_trade_count"], descending=[True, True])
    )

    best_strategies_df = (
        df.with_columns(
            (pl.col("estimated_trade_count") > 0).cast(pl.Int64).alias("survived_flag")
        )
        .group_by("strategy_family")
        .agg(
            [
                pl.len().alias("jobs"),
                pl.col("survived_flag").sum().alias("survived_jobs"),
                pl.col("pre_side_total_count").mean().alias("mean_pre_side_total_count"),
                pl.col("executable_total_count").mean().alias("mean_executable_total_count"),
                pl.col("estimated_trade_count").mean().alias("mean_estimated_trade_count"),
            ]
        )
        .with_columns(
            pl.when(pl.col("jobs") > 0)
            .then((pl.col("survived_jobs") / pl.col("jobs")) * 100.0)
            .otherwise(0.0)
            .alias("survival_pct")
        )
        .sort(by=["survival_pct", "mean_estimated_trade_count"], descending=[True, True])
    )

    best_entries_df = (
        df.with_columns(
            (pl.col("estimated_trade_count") > 0).cast(pl.Int64).alias("survived_flag")
        )
        .group_by("entry_logic")
        .agg(
            [
                pl.len().alias("jobs"),
                pl.col("survived_flag").sum().alias("survived_jobs"),
                pl.col("pre_side_total_count").mean().alias("mean_pre_side_total_count"),
                pl.col("executable_total_count").mean().alias("mean_executable_total_count"),
                pl.col("estimated_trade_count").mean().alias("mean_estimated_trade_count"),
            ]
        )
        .with_columns(
            pl.when(pl.col("jobs") > 0)
            .then((pl.col("survived_jobs") / pl.col("jobs")) * 100.0)
            .otherwise(0.0)
            .alias("survival_pct")
        )
        .sort(by=["survival_pct", "mean_estimated_trade_count"], descending=[True, True])
    )

    decision_summary = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "debug_dir": str(debug_dir),
        "total_jobs": total_jobs,
        "survived_jobs": survived_jobs,
        "survival_pct": safe_float((survived_jobs / total_jobs) * 100.0 if total_jobs > 0 else 0.0),
        "top_kill_stage": top_kill_stage,
        "layer_failure_counts": layer_failure,
        "stage_summary": stage_summary,
        "run_summary": run_summary,
        "best_timeframes": best_timeframes_df.head(20).to_dicts(),
        "best_strategies": best_strategies_df.head(20).to_dicts(),
        "best_entry_logics": best_entries_df.head(20).to_dicts(),
        "top_surviving_jobs_preview": top_survivors_df.head(20).to_dicts(),
        "top_surviving_groups_preview": top_groups_df.head(20).to_dicts(),
    }
    write_json(out_summary_json, decision_summary)

    recommendation_text = build_recommendation_text(
        total_jobs=total_jobs,
        survived_jobs=survived_jobs,
        top_kill_stage=top_kill_stage,
        top_survivor_rows=top_survivors_df.head(20).to_dicts(),
        top_group_rows=top_groups_df.head(20).to_dicts(),
    )
    write_text(out_recommendation_txt, recommendation_text)

    print("=" * 120)
    print(f"[DONE] decision_summary={out_summary_json}")
    print(f"[DONE] top_survivors={out_top_survivors_csv}")
    print(f"[DONE] top_groups={out_top_groups_csv}")
    print(f"[DONE] kill_by_timeframe={out_kill_by_tf_csv}")
    print(f"[DONE] recommendation={out_recommendation_txt}")
    print(f"[DONE] total_jobs={total_jobs}")
    print(f"[DONE] survived_jobs={survived_jobs}")
    print(f"[DONE] top_kill_stage={top_kill_stage}")
    print("=" * 120)


if __name__ == "__main__":
    main()
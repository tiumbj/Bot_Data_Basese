# version: v1.0.0
# file: C:\Data\Bot\Local_LLM\gold_research\jobs\execute_winner_benchmark_job.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VERSION = "v1.0.0"
WINNER_STRATEGY_ID = (
    "locked_ms_bos_choch_pullback_atr_adx_ema_entry_v2_pullback_deep_"
    "regime_block_conditional_max_hold_4h_cooldown_2loss_skip1"
)
WINNER_FREEZE_SUMMARY = Path(
    r"C:\Data\Bot\central_backtest_results\winner_variant_freeze\winner_variant_summary.json"
)


def load_winner_summary() -> dict[str, Any]:
    if not WINNER_FREEZE_SUMMARY.exists():
        raise FileNotFoundError(f"Winner freeze summary not found: {WINNER_FREEZE_SUMMARY}")
    return json.loads(WINNER_FREEZE_SUMMARY.read_text(encoding="utf-8"))


def write_summary_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_real_benchmark_summary(job: dict[str, Any]) -> dict[str, Any]:
    winner_payload = load_winner_summary()
    winner_summary = winner_payload["summary"]

    return {
        "runner_version": VERSION,
        "job_id": job["job_id"],
        "status": "done",
        "mode": "real_benchmark_reference",
        "strategy_id": WINNER_STRATEGY_ID,
        "stage": job["stage"],
        "metrics": {
            "pnl_sum": winner_summary["pnl_sum"],
            "payoff_ratio": winner_summary["payoff_ratio"],
            "max_consecutive_losses": winner_summary["max_consecutive_losses"],
            "trade_count": winner_summary["trades"],
            "win_rate_pct": winner_summary["win_rate_pct"],
            "wins": winner_summary["wins"],
            "losses": winner_summary["losses"],
        },
        "artifact_links": {
            "winner_freeze_summary": str(WINNER_FREEZE_SUMMARY),
            "winner_active_rows": r"C:\Data\Bot\central_backtest_results\winner_variant_freeze\winner_variant_active_rows.jsonl",
            "winner_audit_rows": r"C:\Data\Bot\central_backtest_results\winner_variant_freeze\winner_variant_audit_all_rows.jsonl",
        },
        "notes": [
            "Real benchmark summary loaded from winner_variant_freeze",
            "This is the locked winner reference for research tournament v1",
        ],
    }


def is_winner_mapped_deep_pullback_job(job: dict[str, Any]) -> bool:
    if job.get("family_id") != "deep_pullback_continuation":
        return False

    entry_style = job.get("entry_style")
    micro_exit = job.get("micro_exit", {})
    cooldown = job.get("cooldown", {})
    regime_filter = job.get("regime_filter", {})
    timeframe = job.get("timeframe")

    return (
        timeframe == "M30"
        and entry_style == "deep"
        and micro_exit.get("exit_id") == "conditional_max_hold_4h"
        and cooldown.get("cooldown_id") == "cooldown_2L_skip1"
        and regime_filter.get("regime_filter_id") == "strict"
    )


def build_real_mapped_family_summary(job: dict[str, Any]) -> dict[str, Any]:
    winner_payload = load_winner_summary()
    winner_summary = winner_payload["summary"]

    return {
        "runner_version": VERSION,
        "job_id": job["job_id"],
        "status": "done",
        "mode": "real_family_reference_from_winner_mapping",
        "family_id": job["family_id"],
        "stage": job["stage"],
        "symbol": job["symbol"],
        "timeframe": job["timeframe"],
        "entry_style": job["entry_style"],
        "micro_exit": job["micro_exit"],
        "cooldown": job["cooldown"],
        "regime_filter": job["regime_filter"],
        "metrics": {
            "pnl_sum": winner_summary["pnl_sum"],
            "payoff_ratio": winner_summary["payoff_ratio"],
            "max_consecutive_losses": winner_summary["max_consecutive_losses"],
            "trade_count": winner_summary["trades"],
            "win_rate_pct": winner_summary["win_rate_pct"],
            "wins": winner_summary["wins"],
            "losses": winner_summary["losses"],
        },
        "artifact_links": {
            "winner_freeze_summary": str(WINNER_FREEZE_SUMMARY),
        },
        "notes": [
            "This deep_pullback_continuation job is mapped to the locked winner implementation",
            "Only strict M30 + deep + conditional_max_hold_4h + cooldown_2L_skip1 is mapped in v1",
        ],
    }


def save_job_summary(job: dict[str, Any], summary: dict[str, Any]) -> None:
    write_summary_json(Path(job["artifact_paths"]["summary_json"]), summary)
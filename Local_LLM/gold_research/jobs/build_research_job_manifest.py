# version: v1.0.0
# file: C:\Data\Bot\Local_LLM\gold_research\jobs\build_research_job_manifest.py
# run:
# python C:\Data\Bot\Local_LLM\gold_research\jobs\build_research_job_manifest.py --config C:\Data\Bot\Local_LLM\gold_research\configs\research_requirements.yaml

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import yaml


VERSION = "v1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build research job manifest from YAML requirements")
    parser.add_argument("--config", required=True, help="Path to research_requirements.yaml")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def enabled_list(items: list[dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        if bool(item.get("enabled", True)):
            if key_name not in item:
                raise ValueError(f"Missing key '{key_name}' in item: {item}")
            out.append(item)
    return out


def build_stage_jobs(config: dict[str, Any], stage_name: str, timeframes: list[str]) -> list[dict[str, Any]]:
    families = enabled_list(config["families"], "family_id")
    micro_exits = enabled_list(config["micro_exits"], "exit_id")
    cooldowns = enabled_list(config["cooldowns"], "cooldown_id")
    regime_filters = enabled_list(config["regime_filters"], "regime_filter_id")
    entry_styles = list(config["entry_styles"])

    jobs: list[dict[str, Any]] = []

    for tf, family, entry_style, micro_exit, cooldown, regime_filter in product(
        timeframes,
        families,
        entry_styles,
        micro_exits,
        cooldowns,
        regime_filters,
    ):
        payload = {
            "project_id": config["project_id"],
            "stage": stage_name,
            "symbol": config["data"]["symbol"],
            "timeframe": tf,
            "family_id": family["family_id"],
            "family_category": family["category"],
            "entry_style": entry_style,
            "micro_exit": micro_exit,
            "cooldown": cooldown,
            "regime_filter": regime_filter,
            "cost_model": config["cost_model"],
            "walkforward": config["walkforward"],
            "ranking_hard_gates": config["ranking"]["hard_gates"],
        }

        job_id = f"{stage_name}_{tf}_{family['family_id']}_{entry_style}_{micro_exit['exit_id']}_{cooldown['cooldown_id']}_{regime_filter['regime_filter_id']}_{stable_hash(payload)}"

        job = {
            "job_id": job_id,
            "status": "pending",
            "created_at_utc": utc_now_iso(),
            "project_id": config["project_id"],
            "project_name": config["project_name"],
            "stage": stage_name,
            "symbol": config["data"]["symbol"],
            "timeframe": tf,
            "family_id": family["family_id"],
            "family_category": family["category"],
            "family_description": family["description"],
            "entry_style": entry_style,
            "micro_exit": micro_exit,
            "cooldown": cooldown,
            "regime_filter": regime_filter,
            "cost_model": config["cost_model"],
            "walkforward": config["walkforward"],
            "ranking_hard_gates": config["ranking"]["hard_gates"],
            "artifact_paths": {
                "job_result_dir": str(
                    Path(config["output"]["manifest_dir"]) / "job_results" / job_id
                ),
                "summary_json": str(
                    Path(config["output"]["manifest_dir"]) / "job_results" / job_id / "summary.json"
                ),
                "trade_log_parquet": str(
                    Path(config["output"]["manifest_dir"]) / "job_results" / job_id / "trade_log.parquet"
                ),
                "regime_summary_csv": str(
                    Path(config["output"]["manifest_dir"]) / "job_results" / job_id / "regime_summary.csv"
                ),
                "walkforward_summary_json": str(
                    Path(config["output"]["manifest_dir"]) / "job_results" / job_id / "walkforward_summary.json"
                ),
            },
        }
        jobs.append(job)

    return jobs


def build_benchmark_job(config: dict[str, Any]) -> dict[str, Any]:
    benchmark = config["benchmark"]
    payload = {
        "project_id": config["project_id"],
        "stage": "benchmark",
        "strategy_id": benchmark["strategy_id"],
        "family": benchmark["family"],
        "locked_winner_logic": benchmark["locked_winner_logic"],
    }
    job_id = f"benchmark_{benchmark['strategy_id']}_{stable_hash(payload)}"

    return {
        "job_id": job_id,
        "status": "pending",
        "created_at_utc": utc_now_iso(),
        "project_id": config["project_id"],
        "project_name": config["project_name"],
        "stage": "benchmark",
        "strategy_id": benchmark["strategy_id"],
        "family": benchmark["family"],
        "locked_winner_logic": benchmark["locked_winner_logic"],
        "artifact_paths": {
            "job_result_dir": str(
                Path(config["output"]["manifest_dir"]) / "job_results" / job_id
            ),
            "summary_json": str(
                Path(config["output"]["manifest_dir"]) / "job_results" / job_id / "summary.json"
            ),
        },
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml(config_path)

    manifest_dir = Path(config["output"]["manifest_dir"])
    ensure_dir(manifest_dir)
    ensure_dir(manifest_dir / "job_results")

    jobs: list[dict[str, Any]] = []

    if bool(config["benchmark"]["enabled"]):
        jobs.append(build_benchmark_job(config))

    if bool(config["research"]["stage_1_scan"]["enabled"]):
        jobs.extend(
            build_stage_jobs(
                config=config,
                stage_name="stage_1_scan",
                timeframes=list(config["research"]["stage_1_scan"]["timeframes"]),
            )
        )

    if bool(config["research"]["stage_2_deepen"]["enabled"]):
        jobs.extend(
            build_stage_jobs(
                config=config,
                stage_name="stage_2_deepen",
                timeframes=list(config["research"]["stage_2_deepen"]["timeframes"]),
            )
        )

    manifest_jsonl = manifest_dir / "research_job_manifest.jsonl"
    manifest_summary = manifest_dir / "research_job_manifest_summary.json"

    with manifest_jsonl.open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")

    stage_counts: dict[str, int] = {}
    tf_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}

    for job in jobs:
        stage = job["stage"]
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

        if "timeframe" in job:
            tf = job["timeframe"]
            tf_counts[tf] = tf_counts.get(tf, 0) + 1

        family = job.get("family_id", job.get("family", "benchmark"))
        family_counts[family] = family_counts.get(family, 0) + 1

    summary = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "config_path": str(config_path),
        "project_id": config["project_id"],
        "project_name": config["project_name"],
        "total_jobs": len(jobs),
        "stage_counts": stage_counts,
        "timeframe_counts": tf_counts,
        "family_counts": family_counts,
        "manifest_jsonl": str(manifest_jsonl),
    }

    manifest_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] config={config_path}")
    print(f"[DONE] total_jobs={len(jobs)}")
    print(f"[DONE] manifest={manifest_jsonl}")
    print(f"[DONE] summary={manifest_summary}")
    print("=" * 120)


if __name__ == "__main__":
    main()
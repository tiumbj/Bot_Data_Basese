# version: v1.0.4
# file: C:\Data\Bot\Local_LLM\gold_research\jobs\run_research_jobs.py
# run:
# python C:\Data\Bot\Local_LLM\gold_research\jobs\run_research_jobs.py --manifest C:\Data\Bot\central_backtest_results\research_jobs\research_job_manifest_multitf_winner.jsonl --state C:\Data\Bot\central_backtest_results\research_jobs\job_state_multitf_winner.jsonl --max-jobs 3

from __future__ import annotations

import argparse
import copy
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from execute_winner_benchmark_job import (
    VERSION as EXECUTOR_VERSION,
    build_real_benchmark_summary,
    save_job_summary,
)
from execute_deep_pullback_m30_family import (
    VERSION as DEEP_PULLBACK_EXECUTOR_VERSION,
    execute_deep_pullback_m30_family_job,
)


VERSION = "v1.0.4"
SUPPORTED_TIMEFRAMES = {"M1", "M2", "M3", "M4", "M5", "M6", "M10", "M15", "M30", "H1", "H4", "D1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run research jobs with checkpoint/resume")
    parser.add_argument("--manifest", required=True, help="Path to research_job_manifest.jsonl")
    parser.add_argument("--state", required=True, help="Path to job_state.jsonl")
    parser.add_argument("--max-jobs", type=int, default=0, help="Limit number of pending jobs to run in this session")
    parser.add_argument(
        "--allow-stub-families",
        action="store_true",
        help="Allow placeholder execution for families not implemented yet",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Manifest is empty or missing: {path}")
    return rows


def load_latest_state(state_path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(state_path)
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest[row["job_id"]] = row
    return latest


def get_pending_jobs(
    manifest_rows: list[dict[str, Any]],
    latest_state: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for job in manifest_rows:
        state = latest_state.get(job["job_id"])
        if state is None:
            pending.append(job)
            continue
        if state.get("status") not in {"done", "skipped"}:
            pending.append(job)
    return pending


def write_summary_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_variant_dict(job: dict[str, Any]) -> dict[str, Any]:
    variant = job.get("variant")
    if isinstance(variant, dict):
        return variant
    return {}


def extract_variant_fields(job: dict[str, Any]) -> dict[str, Any]:
    variant = get_variant_dict(job)

    micro_exit_id = variant.get("micro_exit_id")
    cooldown_bars = variant.get("cooldown_bars")
    regime_filter_id = variant.get("regime_filter_id")

    if micro_exit_id is None:
        legacy_micro_exit = job.get("micro_exit", {})
        if isinstance(legacy_micro_exit, dict):
            micro_exit_id = legacy_micro_exit.get("exit_id") or legacy_micro_exit.get("micro_exit_id")

    if cooldown_bars is None:
        legacy_cooldown = job.get("cooldown", {})
        if isinstance(legacy_cooldown, dict):
            legacy_cooldown_id = legacy_cooldown.get("cooldown_id")
            legacy_map = {
                "none": 0,
                "cooldown_2L_skip1": 2,
                "cooldown_3L_skip1": 3,
            }
            cooldown_bars = legacy_map.get(legacy_cooldown_id)

    if regime_filter_id is None:
        legacy_regime = job.get("regime_filter", {})
        if isinstance(legacy_regime, dict):
            regime_filter_id = legacy_regime.get("regime_filter_id")

    return {
        "micro_exit_id": micro_exit_id,
        "cooldown_bars": cooldown_bars,
        "regime_filter_id": regime_filter_id,
    }


def build_legacy_micro_exit(micro_exit_id: str | None) -> dict[str, Any]:
    return {
        "exit_id": micro_exit_id or "",
        "micro_exit_id": micro_exit_id or "",
    }


def build_legacy_cooldown(cooldown_bars: Any) -> dict[str, Any]:
    cooldown_value = ""
    if isinstance(cooldown_bars, int):
        cooldown_value = f"cooldown_{cooldown_bars}"
    elif cooldown_bars is not None:
        cooldown_value = str(cooldown_bars)

    return {
        "cooldown_bars": cooldown_bars,
        "cooldown_id": cooldown_value,
    }


def build_legacy_regime_filter(regime_filter_id: str | None) -> dict[str, Any]:
    return {
        "regime_filter_id": regime_filter_id or "",
    }


def normalize_job_for_executor(job: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(job)
    extracted = extract_variant_fields(normalized)

    normalized["micro_exit"] = build_legacy_micro_exit(extracted["micro_exit_id"])
    normalized["cooldown"] = build_legacy_cooldown(extracted["cooldown_bars"])
    normalized["regime_filter"] = build_legacy_regime_filter(extracted["regime_filter_id"])

    if "stage" not in normalized and "stage_name" in normalized:
        normalized["stage"] = normalized["stage_name"]

    return normalized


def run_stub_job(job: dict[str, Any], skip_reason: str) -> dict[str, Any]:
    normalized_job = normalize_job_for_executor(job)

    result_dir = Path(normalized_job["artifact_paths"]["job_result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "runner_version": VERSION,
        "benchmark_executor_version": EXECUTOR_VERSION,
        "deep_pullback_executor_version": DEEP_PULLBACK_EXECUTOR_VERSION,
        "job_id": normalized_job["job_id"],
        "status": "skipped",
        "skip_reason": skip_reason,
        "stage": normalized_job.get("stage", ""),
        "symbol": normalized_job.get("symbol", ""),
        "timeframe": normalized_job.get("timeframe", ""),
        "family_id": normalized_job.get("family_id", ""),
        "entry_style": normalized_job.get("entry_style", ""),
        "variant": normalized_job.get("variant", {}),
        "micro_exit": normalized_job.get("micro_exit", {}),
        "cooldown": normalized_job.get("cooldown", {}),
        "regime_filter": normalized_job.get("regime_filter", {}),
        "dataset": normalized_job.get("dataset", {}),
    }
    write_summary_json(Path(normalized_job["artifact_paths"]["summary_json"]), summary)
    return summary


def deep_pullback_job_is_really_supported(job: dict[str, Any]) -> bool:
    if job.get("family_id") != "deep_pullback_continuation":
        return False
    if job.get("entry_style") != "deep":
        return False
    if str(job.get("timeframe", "")).upper() not in SUPPORTED_TIMEFRAMES:
        return False

    dataset = job.get("dataset", {})
    if not isinstance(dataset, dict):
        return False
    ohlc_csv = dataset.get("ohlc_csv")
    if not ohlc_csv:
        return False

    extracted = extract_variant_fields(job)
    micro_exit_id = extracted["micro_exit_id"]
    cooldown_bars = extracted["cooldown_bars"]
    regime_filter_id = extracted["regime_filter_id"]

    supported_micro_exits = {
        "micro_exit_v2_fast_invalidation",
        "micro_exit_v2_momentum_fade",
        "micro_exit_v2_structure_trail",
    }
    supported_cooldowns = {0, 3, 6}
    supported_regimes = {
        "regime_filter_off",
        "regime_filter_trend_only",
        "regime_filter_trend_or_neutral",
    }

    return (
        micro_exit_id in supported_micro_exits
        and cooldown_bars in supported_cooldowns
        and regime_filter_id in supported_regimes
    )


def execute_job(job: dict[str, Any], allow_stub_families: bool) -> dict[str, Any]:
    normalized_job = normalize_job_for_executor(job)

    if normalized_job["stage"] == "benchmark":
        summary = build_real_benchmark_summary(normalized_job)
        save_job_summary(normalized_job, summary)
        return summary

    family_id = normalized_job.get("family_id", "")

    if family_id == "deep_pullback_continuation":
        if deep_pullback_job_is_really_supported(normalized_job):
            return execute_deep_pullback_m30_family_job(normalized_job)

        if allow_stub_families:
            return run_stub_job(normalized_job, "deep_pullback_family_variant_not_supported_by_real_executor")

        raise NotImplementedError("deep_pullback_continuation variant not supported by real executor")

    if allow_stub_families:
        return run_stub_job(normalized_job, "family_not_implemented_in_runner_v1_0_4")

    raise NotImplementedError(f"Family '{family_id}' is not implemented yet.")


def build_state_row(
    job: dict[str, Any],
    status: str,
    message: str,
    summary_path: str | None = None,
) -> dict[str, Any]:
    return {
        "ts_utc": utc_now_iso(),
        "runner_version": VERSION,
        "job_id": job["job_id"],
        "status": status,
        "stage": job.get("stage", ""),
        "family_id": job.get("family_id", ""),
        "timeframe": job.get("timeframe", ""),
        "message": message,
        "summary_json": summary_path or "",
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    state_path = Path(args.state)

    manifest_rows = load_manifest(manifest_path)
    latest_state = load_latest_state(state_path)
    pending_jobs = get_pending_jobs(manifest_rows, latest_state)

    total_pending = len(pending_jobs)
    if args.max_jobs > 0:
        pending_jobs = pending_jobs[: args.max_jobs]

    print("=" * 120)
    print(f"[INFO] runner_version={VERSION}")
    print(f"[INFO] benchmark_executor_version={EXECUTOR_VERSION}")
    print(f"[INFO] deep_pullback_executor_version={DEEP_PULLBACK_EXECUTOR_VERSION}")
    print(f"[INFO] manifest={manifest_path}")
    print(f"[INFO] state={state_path}")
    print(f"[INFO] total_jobs_in_manifest={len(manifest_rows)}")
    print(f"[INFO] total_pending_before_limit={total_pending}")
    print(f"[INFO] jobs_to_run_now={len(pending_jobs)}")
    print("=" * 120)

    done_count = 0
    skipped_count = 0
    failed_count = 0

    for idx, job in enumerate(pending_jobs, start=1):
        print(f"[RUN] {idx}/{len(pending_jobs)} job_id={job['job_id']}")
        append_jsonl(
            state_path,
            build_state_row(job=job, status="running", message="job_started"),
        )

        try:
            summary = execute_job(job, allow_stub_families=args.allow_stub_families)
            summary_json = job["artifact_paths"]["summary_json"]
            final_status = summary.get("status", "done")

            if final_status == "skipped":
                skipped_count += 1
            else:
                done_count += 1

            append_jsonl(
                state_path,
                build_state_row(
                    job=job,
                    status=final_status,
                    message=summary.get("skip_reason", "job_finished"),
                    summary_path=summary_json,
                ),
            )

        except Exception as exc:
            failed_count += 1
            error_dir = Path(job["artifact_paths"]["job_result_dir"])
            error_dir.mkdir(parents=True, exist_ok=True)

            error_payload = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            write_summary_json(error_dir / "error.json", error_payload)

            append_jsonl(
                state_path,
                build_state_row(
                    job=job,
                    status="failed",
                    message=f"{type(exc).__name__}: {exc}",
                    summary_path=str(error_dir / "error.json"),
                ),
            )

    print("=" * 120)
    print(f"[DONE] runner_version={VERSION}")
    print(f"[DONE] processed_now={len(pending_jobs)}")
    print(f"[DONE] done_count={done_count}")
    print(f"[DONE] skipped_count={skipped_count}")
    print(f"[DONE] failed_count={failed_count}")
    print("=" * 120)


if __name__ == "__main__":
    main()
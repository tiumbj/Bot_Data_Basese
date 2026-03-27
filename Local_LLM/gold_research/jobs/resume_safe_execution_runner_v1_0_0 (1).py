"""
resume_safe_execution_runner_v1_0_0.py
Version: v1.0.0
Purpose:
    Gate 4 promoted resume-safe execution runner.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

from execution_state_store_v1_0_0 import ExecutionStateStore, RunIdentity, file_sha256
from standardized_result_pack_v1_0_0 import (
    StandardizedResultPackWriter,
    build_standardized_result,
    deterministic_result_key,
)


RUNNER_VERSION = "v1.0.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_executor(executor_ref: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    if ":" not in executor_ref:
        raise ValueError("executor_ref must use 'module:function' format")
    module_name, func_name = executor_ref.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    if not callable(func):
        raise TypeError(f"Executor {executor_ref!r} is not callable")
    return func


def load_manifest(manifest_path: str | Path) -> List[Dict[str, Any]]:
    manifest_path = Path(manifest_path)
    suffix = manifest_path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError("manifest must be .jsonl or .csv")


def shard_jobs(rows: List[Dict[str, Any]], shard_index: int, shard_count: int) -> List[Dict[str, Any]]:
    if shard_count <= 0:
        raise ValueError("shard_count must be > 0")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index out of range")
    return [row for i, row in enumerate(rows) if i % shard_count == shard_index]


def ensure_required_job_fields(job: Dict[str, Any]) -> Dict[str, Any]:
    job = dict(job)
    if not job.get("job_id"):
        job["job_id"] = deterministic_result_key(job)
    if not job.get("group_id"):
        job["group_id"] = f"{job.get('timeframe','')}|{job.get('strategy','')}|{job.get('entry','')}|{job.get('micro_exit','')}"
    return job


def safe_execute(executor: Callable[[Dict[str, Any]], Dict[str, Any]], job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = executor(job)
        if not isinstance(result, dict):
            raise TypeError("executor result must be dict")
        return result
    except Exception as exc:
        return {
            "status": "failed",
            "metrics": {},
            "reject_reason": "",
            "error_reason": f"{type(exc).__name__}: {exc}",
            "missing_feature_reason": "",
        }


def run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    outdir = Path(args.outdir).resolve()
    executor_ref = args.executor
    executor = load_executor(executor_ref)

    manifest_rows = [ensure_required_job_fields(r) for r in load_manifest(manifest_path)]
    shard_rows = shard_jobs(manifest_rows, args.shard_index, args.shard_count)

    state_store = ExecutionStateStore(outdir)
    result_writer = StandardizedResultPackWriter(outdir)
    result_writer.ensure_layout()

    run_identity = RunIdentity(
        runner_version=RUNNER_VERSION,
        manifest_path=str(manifest_path),
        manifest_hash=file_sha256(manifest_path),
        outdir=str(outdir),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        executor_ref=executor_ref,
    )
    state = state_store.initialize_or_resume(run_identity, total_jobs=len(shard_rows))
    run_id = state["run_meta"]["run_id"]

    completed_job_ids = state_store.load_completed_job_ids()
    completed_group_ids = state_store.load_completed_group_ids()
    written_result_keys = state_store.load_written_result_keys()

    skipped_completed_jobs = 0
    skipped_completed_groups = 0
    executed_in_this_session = 0

    for job in shard_rows:
        job_id = str(job["job_id"])
        group_id = str(job["group_id"])

        if job_id in completed_job_ids:
            skipped_completed_jobs += 1
            continue

        if args.skip_completed_groups and group_id in completed_group_ids:
            skipped_completed_groups += 1
            completed_job_ids.add(job_id)
            state_store.append_completed_job_id(job_id)
            state = state_store.update_progress_counters(
                state=state,
                status="success",
                completed_job_ids_count=len(completed_job_ids),
                completed_group_ids_count=len(completed_group_ids),
                written_result_keys_count=len(written_result_keys),
                skipped_completed_jobs=skipped_completed_jobs,
                skipped_completed_groups=skipped_completed_groups,
            )
            state_store.save_state(state)
            continue

        started_at = utc_now_iso()
        start_ts = time.perf_counter()
        executor_result = safe_execute(executor, job)
        duration_sec = time.perf_counter() - start_ts
        finished_at = utc_now_iso()

        row = build_standardized_result(
            run_id=run_id,
            job_payload=job,
            executor_result=executor_result,
            recovery_status=state["recovery"]["recovery_status"],
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            duration_sec=duration_sec,
        )

        if row["result_key"] not in written_result_keys:
            result_writer.append_result(row)
            written_result_keys.add(row["result_key"])
            state_store.save_written_result_keys(written_result_keys)

        completed_job_ids.add(job_id)
        state_store.append_completed_job_id(job_id)

        if args.mark_group_complete_on_job_finish:
            completed_group_ids.add(group_id)
            state_store.append_completed_group_id(group_id)

        state = state_store.update_progress_counters(
            state=state,
            status=row["status"],
            completed_job_ids_count=len(completed_job_ids),
            completed_group_ids_count=len(completed_group_ids),
            written_result_keys_count=len(written_result_keys),
            skipped_completed_jobs=skipped_completed_jobs,
            skipped_completed_groups=skipped_completed_groups,
        )
        state_store.save_state(state)
        result_writer.write_summaries(state, result_writer.read_all_results())

        executed_in_this_session += 1
        if args.stop_after_n is not None and executed_in_this_session >= args.stop_after_n:
            print(f"[INTENTIONAL_STOP] stop_after_n={args.stop_after_n}")
            return 0

    state["recovery"]["skipped_completed_jobs"] = skipped_completed_jobs
    state["recovery"]["skipped_completed_groups"] = skipped_completed_groups
    state_store.save_state(state)
    result_writer.write_summaries(state, result_writer.read_all_results())

    print("=" * 100)
    print(f"[DONE] version={RUNNER_VERSION}")
    print(f"[DONE] outdir={outdir}")
    print(f"[DONE] manifest={manifest_path}")
    print(f"[DONE] total_jobs={len(shard_rows)}")
    print(f"[DONE] completed_jobs={state['progress']['completed_jobs']}")
    print(f"[DONE] success_jobs={state['progress']['success_jobs']}")
    print(f"[DONE] failed_jobs={state['progress']['failed_jobs']}")
    print(f"[DONE] rejected_jobs={state['progress']['rejected_jobs']}")
    print(f"[DONE] missing_feature_jobs={state['progress']['missing_feature_jobs']}")
    print(f"[DONE] resumed_jobs={state['recovery']['resumed_jobs']}")
    print(f"[DONE] skipped_completed_jobs={state['recovery']['skipped_completed_jobs']}")
    print("=" * 100)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate 4 promoted resume-safe execution runner")
    parser.add_argument("--manifest", required=True, help="Path to seed manifest .jsonl or .csv")
    parser.add_argument("--outdir", required=True, help="Same outdir must be reused for restart")
    parser.add_argument("--executor", required=True, help="Executor callable in module:function format")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index")
    parser.add_argument("--shard-count", type=int, default=1, help="Shard count")
    parser.add_argument("--skip-completed-groups", action="store_true", help="Skip jobs when their group_id is already complete")
    parser.add_argument("--mark-group-complete-on-job-finish", action="store_true", help="Mark group_id complete on job finish")
    parser.add_argument("--stop-after-n", type=int, default=None, help="Intentional partial stop for acceptance testing")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

"""
ชื่อโค้ด: run_intelligent_backtest_batch.py
เวอร์ชัน: v1.0.1
เป้าหมาย: Production-grade batch runner ที่รองรับ manifest schema เก่าและใหม่แบบ backward-compatible
changelog:
- v1.0.1
  1) รองรับ schema ใหม่ที่ใช้ feature_cache_path / htf_feature_cache_path / parameter_fingerprint / one_axis_group_key
  2) ยกเลิกการบังคับ ohlc_csv และ result_key แบบเดิม
  3) เพิ่ม normalize_manifest_row(row) พร้อม validation และ auto result_key injection
  4) คง contract CLI และ worker invocation เดิมทั้งหมด
  5) ปรับ load_manifest ให้ parse+normalize ทีละบรรทัดพร้อม error แสดงเลขแถวชัดเจน

ที่อยู่ไฟล์:
C:\Data\Bot\Local_LLM\gold_research\jobs\run_intelligent_backtest_batch.py

คำสั่งรัน:
python C:\Data\Bot\Local_LLM\gold_research\jobs\run_intelligent_backtest_batch.py --manifest C:\Data\Bot\central_backtest_results\research_jobs_full_discovery\research_job_manifest_full_discovery.jsonl --worker-script C:\Data\Bot\Local_LLM\gold_research\jobs\run_single_research_job.py --outdir C:\Data\Bot\central_backtest_results\intelligent_backtest_batch_run_01 --max-jobs 10 --timeout-sec 1800 --sleep-sec 1
"""

import argparse
import json
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

VERSION = "v1.0.1"

REQUIRED_BASE_KEYS = {
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
}

OPTIONAL_FORWARD_KEYS = {
    "feature_cache_path",
    "htf_context_timeframe",
    "htf_feature_cache_path",
    "parameter_fingerprint",
    "one_axis_group_key",
    "rank_priority",
    "status",
}

STRING_NORMALIZE_KEYS = REQUIRED_BASE_KEYS | OPTIONAL_FORWARD_KEYS | {
    "result_key",
    "manifest_id",
    "job_name",
    "phase",
    "symbol",
    "logic_variant",
    "pullback_zone_variant",
    "management_variant",
    "regime_variant",
    "robustness_variant",
    "ohlc_csv",
}


@dataclass
class JobRunResult:
    job_id: str
    result_key: str
    status: str
    returncode: int
    duration_sec: float
    stdout_log_path: str
    stderr_log_path: str
    result_dir: str
    error_message: str = ""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path} line {line_no}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                raise RuntimeError(f"Invalid JSONL object at {path} line {line_no}: expected object")
    return rows


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_manifest_row(row: dict) -> dict:
    if not isinstance(row, dict):
        raise ValueError("row is not an object")

    normalized: Dict[str, Any] = dict(row)
    for key in STRING_NORMALIZE_KEYS:
        if key in normalized:
            normalized[key] = clean_str(normalized.get(key))

    missing_required = [key for key in REQUIRED_BASE_KEYS if clean_str(normalized.get(key)) == ""]
    if missing_required:
        raise ValueError(f"missing required keys: {sorted(missing_required)}")

    has_feature_cache = clean_str(normalized.get("feature_cache_path")) != ""
    has_ohlc_csv = clean_str(normalized.get("ohlc_csv")) != ""
    if not (has_feature_cache or has_ohlc_csv):
        raise ValueError("missing data reference: require one of ['feature_cache_path', 'ohlc_csv']")

    derived_result_key = (
        clean_str(normalized.get("result_key"))
        or clean_str(normalized.get("parameter_fingerprint"))
        or clean_str(normalized.get("job_id"))
    )
    if derived_result_key == "":
        raise ValueError("unable to derive result_key")
    normalized["result_key"] = derived_result_key
    normalized["job_id"] = clean_str(normalized.get("job_id"))
    return normalized


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Manifest not found: {path}")

    rows: List[Dict[str, Any]] = []
    seen_job_ids = set()
    seen_result_keys = set()

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Manifest row #{line_no} invalid JSON: {exc}") from exc

            try:
                normalized = normalize_manifest_row(parsed)
            except Exception as exc:
                raise RuntimeError(f"Manifest row #{line_no} schema error: {exc}") from exc

            job_id = normalized["job_id"]
            result_key = normalized["result_key"]

            if job_id in seen_job_ids:
                raise RuntimeError(f"Manifest row #{line_no} duplicate job_id: {job_id}")
            if result_key in seen_result_keys:
                raise RuntimeError(f"Manifest row #{line_no} duplicate result_key: {result_key}")

            seen_job_ids.add(job_id)
            seen_result_keys.add(result_key)
            rows.append(normalized)

    if not rows:
        raise RuntimeError(f"Manifest is empty or missing: {path}")
    return rows


def build_latest_status_map(state_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for row in state_rows:
        job_id = clean_str(row.get("job_id"))
        if job_id:
            latest[job_id] = row
    return latest


def choose_jobs(
    manifest_rows: List[Dict[str, Any]],
    latest_state_map: Dict[str, Dict[str, Any]],
    retry_failed: bool,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for row in manifest_rows:
        job_id = row["job_id"]
        last = latest_state_map.get(job_id)
        if last is None:
            selected.append(row)
            continue
        status = clean_str(last.get("status")).upper()
        if status == "SUCCESS":
            continue
        if status == "FAILED":
            if retry_failed:
                selected.append(row)
            continue
        if status in {"RUNNING", "PENDING"}:
            selected.append(row)
            continue
        selected.append(row)
    return selected


def safe_filename(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def build_worker_command(
    python_exe: str,
    worker_script: Path,
    job_json_path: Path,
    result_root: Path,
) -> List[str]:
    return [
        python_exe,
        str(worker_script),
        "--job",
        str(job_json_path),
        "--result-root",
        str(result_root),
    ]


def write_job_spec(job_specs_dir: Path, job: Dict[str, Any]) -> Path:
    path = job_specs_dir / f"{safe_filename(job['job_id'])}.json"
    write_json(path, job)
    return path


def summarize_latest_statuses(latest_state_map: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for row in latest_state_map.values():
        counter[clean_str(row.get("status", "UNKNOWN")).upper()] += 1
    return dict(sorted(counter.items(), key=lambda kv: kv[0]))


def write_progress_summary(
    path: Path,
    *,
    manifest_total: int,
    latest_state_map: Dict[str, Dict[str, Any]],
    run_started_at_utc: str,
    current_job_index: int,
    total_selected_jobs: int,
) -> None:
    status_counts = summarize_latest_statuses(latest_state_map)
    success_count = status_counts.get("SUCCESS", 0)
    failed_count = status_counts.get("FAILED", 0)
    running_count = status_counts.get("RUNNING", 0)
    pending_like = manifest_total - success_count - failed_count - running_count

    write_json(
        path,
        {
            "version": VERSION,
            "generated_at_utc": utc_now_iso(),
            "run_started_at_utc": run_started_at_utc,
            "manifest_total_jobs": manifest_total,
            "status_counts": status_counts,
            "derived": {
                "success_count": success_count,
                "failed_count": failed_count,
                "running_count": running_count,
                "pending_like_count": pending_like,
            },
            "current_batch_progress": {
                "current_job_index_1_based": current_job_index,
                "selected_jobs_in_this_run": total_selected_jobs,
            },
        },
    )


def run_one_job(
    *,
    python_exe: str,
    worker_script: Path,
    job: Dict[str, Any],
    job_json_path: Path,
    result_root: Path,
    logs_dir: Path,
    timeout_sec: Optional[int],
) -> JobRunResult:
    job_id = clean_str(job["job_id"])
    result_key = clean_str(job["result_key"])
    result_dir = result_root / safe_filename(result_key)
    ensure_dir(result_dir)

    stdout_log_path = logs_dir / f"{safe_filename(job_id)}__stdout.log"
    stderr_log_path = logs_dir / f"{safe_filename(job_id)}__stderr.log"

    cmd = build_worker_command(
        python_exe=python_exe,
        worker_script=worker_script,
        job_json_path=job_json_path,
        result_root=result_root,
    )

    started = time.perf_counter()
    error_message = ""
    with stdout_log_path.open("w", encoding="utf-8") as stdout_f, stderr_log_path.open("w", encoding="utf-8") as stderr_f:
        try:
            completed = subprocess.run(
                cmd,
                stdout=stdout_f,
                stderr=stderr_f,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
            error_message = f"Worker timed out after {timeout_sec} seconds"
        except Exception as exc:
            returncode = 1
            error_message = f"Worker execution error: {exc}"

    duration_sec = time.perf_counter() - started
    status = "SUCCESS" if returncode == 0 else "FAILED"
    return JobRunResult(
        job_id=job_id,
        result_key=result_key,
        status=status,
        returncode=returncode,
        duration_sec=duration_sec,
        stdout_log_path=str(stdout_log_path),
        stderr_log_path=str(stderr_log_path),
        result_dir=str(result_dir),
        error_message=error_message,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run long-running intelligent backtest jobs from a JSONL manifest.")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSONL")
    parser.add_argument("--worker-script", required=True, help="Path to worker script")
    parser.add_argument("--outdir", required=True, help="Output directory for run state/results/logs")
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used to launch worker script (default: current interpreter)",
    )
    parser.add_argument("--max-jobs", type=int, default=0, help="Maximum number of jobs to run in this invocation (0 = no limit)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle pending jobs before execution")
    parser.add_argument("--seed", type=int, default=20260321, help="Shuffle seed when --shuffle is enabled")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately on first job failure")
    parser.add_argument("--retry-failed", action="store_true", help="Include FAILED jobs again when selecting pending jobs")
    parser.add_argument("--timeout-sec", type=int, default=0, help="Per-job worker timeout in seconds (0 = no timeout)")
    parser.add_argument("--sleep-sec", type=float, default=0.0, help="Sleep between jobs to reduce system pressure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    worker_script = Path(args.worker_script)
    outdir = Path(args.outdir)
    python_exe = str(args.python_exe)
    max_jobs = int(args.max_jobs)
    timeout_sec = int(args.timeout_sec) if int(args.timeout_sec) > 0 else None

    ensure_dir(outdir)
    job_specs_dir = outdir / "job_specs"
    logs_dir = outdir / "logs"
    results_dir = outdir / "results"
    reports_dir = outdir / "reports"
    ensure_dir(job_specs_dir)
    ensure_dir(logs_dir)
    ensure_dir(results_dir)
    ensure_dir(reports_dir)

    state_path = outdir / "job_state.jsonl"
    progress_summary_path = reports_dir / "progress_summary.json"
    launch_metadata_path = reports_dir / "launch_metadata.json"

    if not worker_script.exists():
        raise RuntimeError(f"Worker script not found: {worker_script}")

    manifest_rows = load_manifest(manifest_path)
    existing_state_rows = read_jsonl(state_path)
    latest_state_map = build_latest_status_map(existing_state_rows)

    selected_jobs = choose_jobs(
        manifest_rows=manifest_rows,
        latest_state_map=latest_state_map,
        retry_failed=bool(args.retry_failed),
    )
    if args.shuffle:
        random.Random(args.seed).shuffle(selected_jobs)
    if max_jobs > 0:
        selected_jobs = selected_jobs[:max_jobs]

    run_started_at_utc = utc_now_iso()
    write_json(
        launch_metadata_path,
        {
            "version": VERSION,
            "run_started_at_utc": run_started_at_utc,
            "manifest": str(manifest_path),
            "worker_script": str(worker_script),
            "outdir": str(outdir),
            "python_exe": python_exe,
            "max_jobs": max_jobs,
            "shuffle": bool(args.shuffle),
            "seed": int(args.seed),
            "fail_fast": bool(args.fail_fast),
            "retry_failed": bool(args.retry_failed),
            "timeout_sec": timeout_sec,
            "sleep_sec": float(args.sleep_sec),
            "manifest_total_jobs": len(manifest_rows),
            "selected_jobs_in_this_run": len(selected_jobs),
            "existing_state_rows": len(existing_state_rows),
            "existing_status_counts": summarize_latest_statuses(latest_state_map),
        },
    )

    print("=" * 120)
    print(f"[START] version={VERSION}")
    print(f"[START] manifest={manifest_path}")
    print(f"[START] worker_script={worker_script}")
    print(f"[START] outdir={outdir}")
    print(f"[START] manifest_total_jobs={len(manifest_rows)}")
    print(f"[START] selected_jobs_in_this_run={len(selected_jobs)}")
    print(f"[START] existing_status_counts={summarize_latest_statuses(latest_state_map)}")
    print("=" * 120)

    if not selected_jobs:
        write_progress_summary(
            progress_summary_path,
            manifest_total=len(manifest_rows),
            latest_state_map=latest_state_map,
            run_started_at_utc=run_started_at_utc,
            current_job_index=0,
            total_selected_jobs=0,
        )
        print("[DONE] No pending jobs to run.")
        return

    run_success = 0
    run_failed = 0
    for idx, job in enumerate(selected_jobs, start=1):
        job_id = clean_str(job["job_id"])
        result_key = clean_str(job["result_key"])
        job_json_path = write_job_spec(job_specs_dir, job)

        append_jsonl(
            state_path,
            {
                "ts_utc": utc_now_iso(),
                "version": VERSION,
                "event": "job_started",
                "status": "RUNNING",
                "job_id": job_id,
                "result_key": result_key,
                "job_json_path": str(job_json_path),
                "worker_script": str(worker_script),
                "result_root": str(results_dir),
                "queue_index_1_based": idx,
                "queue_total": len(selected_jobs),
            },
        )

        print(
            f"[RUN] {idx}/{len(selected_jobs)} "
            f"job_id={job_id} "
            f"timeframe={clean_str(job.get('timeframe'))} "
            f"strategy={clean_str(job.get('strategy_family'))} "
            f"entry={clean_str(job.get('entry_logic'))} "
            f"micro_exit={clean_str(job.get('micro_exit'))}"
        )

        result = run_one_job(
            python_exe=python_exe,
            worker_script=worker_script,
            job=job,
            job_json_path=job_json_path,
            result_root=results_dir,
            logs_dir=logs_dir,
            timeout_sec=timeout_sec,
        )

        append_jsonl(
            state_path,
            {
                "ts_utc": utc_now_iso(),
                "version": VERSION,
                "event": "job_finished",
                "status": result.status,
                "job_id": result.job_id,
                "result_key": result.result_key,
                "returncode": result.returncode,
                "duration_sec": round(result.duration_sec, 6),
                "stdout_log_path": result.stdout_log_path,
                "stderr_log_path": result.stderr_log_path,
                "result_dir": result.result_dir,
                "error_message": result.error_message,
                "queue_index_1_based": idx,
                "queue_total": len(selected_jobs),
            },
        )

        latest_state_map[result.job_id] = {
            "status": result.status,
            "returncode": result.returncode,
            "duration_sec": round(result.duration_sec, 6),
            "result_dir": result.result_dir,
            "stdout_log_path": result.stdout_log_path,
            "stderr_log_path": result.stderr_log_path,
            "error_message": result.error_message,
        }

        write_progress_summary(
            progress_summary_path,
            manifest_total=len(manifest_rows),
            latest_state_map=latest_state_map,
            run_started_at_utc=run_started_at_utc,
            current_job_index=idx,
            total_selected_jobs=len(selected_jobs),
        )

        if result.status == "SUCCESS":
            run_success += 1
            print(
                f"[OK] {idx}/{len(selected_jobs)} "
                f"job_id={result.job_id} "
                f"duration_sec={result.duration_sec:.2f} "
                f"result_dir={result.result_dir}"
            )
        else:
            run_failed += 1
            print(
                f"[FAIL] {idx}/{len(selected_jobs)} "
                f"job_id={result.job_id} "
                f"returncode={result.returncode} "
                f"duration_sec={result.duration_sec:.2f} "
                f"stderr={result.stderr_log_path}"
            )
            if result.error_message:
                print(f"[FAIL-REASON] {result.error_message}")
            if args.fail_fast:
                print("[STOP] fail-fast enabled, stopping on first failure.")
                break

        if args.sleep_sec > 0:
            time.sleep(float(args.sleep_sec))

    latest_state_rows = read_jsonl(state_path)
    latest_state_map = build_latest_status_map(latest_state_rows)
    final_status_counts = summarize_latest_statuses(latest_state_map)

    write_progress_summary(
        progress_summary_path,
        manifest_total=len(manifest_rows),
        latest_state_map=latest_state_map,
        run_started_at_utc=run_started_at_utc,
        current_job_index=min(len(selected_jobs), run_success + run_failed),
        total_selected_jobs=len(selected_jobs),
    )

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] run_started_at_utc={run_started_at_utc}")
    print(f"[DONE] manifest_total_jobs={len(manifest_rows)}")
    print(f"[DONE] selected_jobs_in_this_run={len(selected_jobs)}")
    print(f"[DONE] run_success={run_success}")
    print(f"[DONE] run_failed={run_failed}")
    print(f"[DONE] state_path={state_path}")
    print(f"[DONE] progress_summary={progress_summary_path}")
    print(f"[DONE] final_status_counts={final_status_counts}")
    print("=" * 120)


if __name__ == "__main__":
    main()

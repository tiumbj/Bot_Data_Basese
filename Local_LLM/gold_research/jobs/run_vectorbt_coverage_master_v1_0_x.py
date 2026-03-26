# -*- coding: utf-8 -*-
"""
run_vectorbt_coverage_master_v1_0_0.py
Version: v1.0.1

Production-first coverage master runner for Local_LLM / gold_research.

Purpose
-------
1) Read coverage manifest with robust encoding fallback.
2) Enforce full-history / no-sampling policy at orchestration layer.
3) Run manifest-defined jobs with shard support.
4) Persist live progress, ETA, job state, logs, and auto-resume.
5) Re-run only missing shards without touching currently running shards.

Design Notes
------------
- This runner is an orchestrator. It does NOT itself perform VectorBT calculations.
- "GPU usage" can only be surfaced/detected here. Real GPU acceleration requires
  the downstream worker/backtest code to support CuPy/RAPIDS/Numba-CUDA.
- To keep this runner robust against manifest schema drift, it supports:
    A) exact shell command via `command`
    B) python script path via `script_path` / `python_file`
       plus optional args_json
    C) parameter columns converted to CLI flags automatically
- The runner explicitly blocks sampling-like manifest directives.

Locked Behavior
---------------
- Full-history only
- No sampling
- Resume-safe
- Progress/ETA
- Shard-aware
- Source-of-truth output under outroot/shards/shard_{i}
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import threading
import time
import traceback
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd
except Exception as exc:
    print(f"[FATAL] pandas import failed: {exc}", file=sys.stderr)
    raise


VERSION = "v1.0.1"
DEFAULT_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
DEFAULT_POLL_SECONDS = 5.0
DEFAULT_MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)

STATE_DONE = "DONE"
STATE_FAILED = "FAILED"
STATE_RUNNING = "RUNNING"
STATE_PENDING = "PENDING"
STATE_SKIPPED = "SKIPPED"

LIVE_PROGRESS_BASENAME = "live_progress.json"
STATE_JSONL_BASENAME = "job_state.jsonl"
SUMMARY_JSON_BASENAME = "summary.json"
CHECKLIST_TXT_BASENAME = "progression_checklist.txt"
RUNNER_LOG_BASENAME = "runner.log"


# ---------------------------------------------------------------------
# Warning suppression
# ---------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message=r".*direction has no effect if short_entries and short_exits are set.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*direction has no effect.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*Downcasting object dtype arrays on \.fillna.*",
)


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------
@dataclass
class JobRecord:
    job_id: str
    row_index: int
    phase: str
    timeframe: str
    strategy_family: str
    command: str
    raw: Dict[str, Any]


@dataclass
class JobResult:
    job_id: str
    status: str
    returncode: int
    start_ts_utc: str
    end_ts_utc: str
    duration_sec: float
    stdout_path: str
    stderr_path: str
    command: str
    error: str = ""


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def stable_shard(job_id: str, num_shards: int) -> int:
    digest = hashlib.md5(job_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_shards


def detect_gpu() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "gpu_mode": "cpu_only",
        "cupy_available": False,
        "cupy_device_count": 0,
        "cupy_error": "",
    }
    try:
        import cupy  # type: ignore

        count = int(cupy.cuda.runtime.getDeviceCount())
        info["gpu_mode"] = "cupy_detected" if count > 0 else "cpu_only"
        info["cupy_available"] = True
        info["cupy_device_count"] = count
        return info
    except Exception as exc:
        info["cupy_error"] = str(exc)
        return info


def sanitize_filename(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("._")
    if not text:
        text = "job"
    return text[:max_len]


def shell_join(parts: List[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return " ".join(shlex.quote(x) for x in parts)


def parse_args_json(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if not text:
        return {}
    return json.loads(text)


def normalize_bool_like(value: Any) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


# ---------------------------------------------------------------------
# Manifest loading and validation
# ---------------------------------------------------------------------
def load_manifest_csv(path: Path, encodings: List[str]) -> Tuple[pd.DataFrame, str]:
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df, enc
        except UnicodeDecodeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
            # parse errors should still stop fallback loop only after trying all encodings
    raise RuntimeError(
        f"Unable to read manifest: {path} | tried encodings={encodings} | last_error={last_error}"
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def block_sampling_columns(df: pd.DataFrame) -> None:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    bad_hits: List[str] = []

    direct_cols = [
        "sample",
        "sample_pct",
        "sample_ratio",
        "sample_rows",
        "sample_size",
        "sampling",
        "sampling_pct",
        "sampling_ratio",
        "random_sample",
        "use_sample",
        "max_bars",
        "bar_limit",
        "row_limit",
        "nrows",
    ]
    for c in direct_cols:
        if c in lowered:
            col = lowered[c]
            series = df[col]
            for value in series.tolist():
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    continue
                if isinstance(value, (int, float)):
                    if float(value) not in (0.0, 1.0):
                        bad_hits.append(f"{col}={value}")
                        break
                else:
                    text = str(value).strip().lower()
                    if text in {"true", "yes", "y"}:
                        bad_hits.append(f"{col}={value}")
                        break
                    if text not in {"", "0", "0.0", "false", "no", "n", "none", "all", "full", "full_history"}:
                        if c not in {"max_bars", "bar_limit", "row_limit", "nrows"}:
                            bad_hits.append(f"{col}={value}")
                            break
                        if text not in {"0", "none", ""}:
                            bad_hits.append(f"{col}={value}")
                            break

    text_scan_cols = [c for c in df.columns if df[c].dtype == object]
    risky_tokens = [
        "sample=",
        "sampling=",
        "--sample",
        "--sample-pct",
        "--sample-ratio",
        "--sample-rows",
        "--max-bars",
        "--bar-limit",
        "--row-limit",
        "--nrows",
    ]
    for col in text_scan_cols:
        values = df[col].fillna("").astype(str).tolist()
        for value in values:
            value_lower = value.lower()
            if any(tok in value_lower for tok in risky_tokens):
                bad_hits.append(f"{col} contains sampling directive")
                break

    if bad_hits:
        raise ValueError(
            "Manifest violates full-history / no-sampling policy. Hits: " + " | ".join(sorted(set(bad_hits)))
        )


def build_job_id(row: Dict[str, Any], row_index: int) -> str:
    preferred_keys = [
        "job_id",
        "job_key",
        "id",
        "task_id",
    ]
    for key in preferred_keys:
        raw = row.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()

    basis = {
        "phase": row.get("phase", ""),
        "timeframe": row.get("timeframe", ""),
        "strategy_family": row.get("strategy_family", ""),
        "command": row.get("command", ""),
        "script_path": row.get("script_path", row.get("python_file", "")),
        "args_json": row.get("args_json", ""),
        "row_index": row_index,
    }
    digest = hashlib.md5(json.dumps(basis, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    return f"job_{row_index:06d}_{digest}"


def build_command_from_row(row: Dict[str, Any]) -> str:
    if row.get("command") is not None and str(row.get("command")).strip():
        return str(row["command"]).strip()

    script_path = row.get("script_path", row.get("python_file", row.get("script", "")))
    if script_path is None or not str(script_path).strip():
        raise ValueError("Manifest row has neither `command` nor `script_path`/`python_file`.")

    script_path = str(script_path).strip()
    python_exe = str(row.get("python_exe", sys.executable)).strip() or sys.executable
    args_json = parse_args_json(row.get("args_json", ""))

    reserved = {
        "command",
        "script_path",
        "python_file",
        "script",
        "python_exe",
        "job_id",
        "job_key",
        "id",
        "task_id",
        "phase",
        "timeframe",
        "strategy_family",
        "args_json",
    }

    cli_parts: List[str] = [python_exe, script_path]

    merged_args: Dict[str, Any] = {}
    for k, v in row.items():
        if k in reserved:
            continue
        merged_args[k] = v
    merged_args.update(args_json)

    for key, value in merged_args.items():
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue

        flag = "--" + str(key).replace("_", "-")
        bool_like = normalize_bool_like(value)
        if bool_like is True:
            cli_parts.append(flag)
            continue
        if bool_like is False:
            continue

        if isinstance(value, (list, tuple)):
            for item in value:
                cli_parts.extend([flag, str(item)])
            continue

        cli_parts.extend([flag, str(value)])

    return shell_join(cli_parts)


def manifest_to_jobs(df: pd.DataFrame) -> List[JobRecord]:
    jobs: List[JobRecord] = []
    for idx, row_series in df.iterrows():
        row = {str(k): (None if pd.isna(v) else v) for k, v in row_series.to_dict().items()}
        job_id = build_job_id(row, idx)
        phase = str(row.get("phase", "default")).strip() or "default"
        timeframe = str(row.get("timeframe", "")).strip()
        strategy_family = str(row.get("strategy_family", "")).strip()
        command = build_command_from_row(row)
        jobs.append(
            JobRecord(
                job_id=job_id,
                row_index=idx,
                phase=phase,
                timeframe=timeframe,
                strategy_family=strategy_family,
                command=command,
                raw=row,
            )
        )
    return jobs


def select_jobs(
    jobs: List[JobRecord],
    phase: Optional[str],
    shard_index: int,
    num_shards: int,
) -> List[JobRecord]:
    selected: List[JobRecord] = []
    for job in jobs:
        if phase and str(job.phase).strip() != str(phase).strip():
            continue
        if stable_shard(job.job_id, num_shards) != shard_index:
            continue
        selected.append(job)
    return selected


# ---------------------------------------------------------------------
# State handling
# ---------------------------------------------------------------------
def load_existing_state(state_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    state: Dict[str, Dict[str, Any]] = {}
    if not state_jsonl.exists():
        return state
    with state_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                job_id = str(row.get("job_id", "")).strip()
                if job_id:
                    state[job_id] = row
            except Exception:
                continue
    return state


def make_progress_payload(
    *,
    version: str,
    manifest_path: Path,
    manifest_encoding: str,
    outroot: Path,
    shard_dir: Path,
    phase: Optional[str],
    shard_index: int,
    num_shards: int,
    workers: int,
    total_jobs: int,
    done_jobs: int,
    failed_jobs: int,
    running_jobs: int,
    skipped_jobs: int,
    pending_jobs: int,
    elapsed_sec: float,
    avg_done_duration_sec: float,
    gpu_info: Dict[str, Any],
) -> Dict[str, Any]:
    throughput = (done_jobs + failed_jobs + skipped_jobs) / elapsed_sec if elapsed_sec > 0 else 0.0
    remaining = max(0, total_jobs - done_jobs - failed_jobs - skipped_jobs)
    eta_sec = 0.0
    if throughput > 0:
        eta_sec = remaining / throughput
    elif avg_done_duration_sec > 0 and workers > 0:
        eta_sec = (remaining / workers) * avg_done_duration_sec

    pct = (100.0 * (done_jobs + failed_jobs + skipped_jobs) / total_jobs) if total_jobs > 0 else 100.0

    return {
        "version": version,
        "generated_at_utc": utc_now_iso(),
        "manifest_path": str(manifest_path),
        "manifest_encoding": manifest_encoding,
        "outroot": str(outroot),
        "shard_dir": str(shard_dir),
        "phase": phase,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "workers": workers,
        "total_jobs": total_jobs,
        "done_jobs": done_jobs,
        "failed_jobs": failed_jobs,
        "running_jobs": running_jobs,
        "skipped_jobs": skipped_jobs,
        "pending_jobs": pending_jobs,
        "progress_pct": round(pct, 4),
        "elapsed_sec": round(elapsed_sec, 4),
        "avg_done_duration_sec": round(avg_done_duration_sec, 4),
        "eta_sec": round(eta_sec, 4),
        "eta_hms": str(dt.timedelta(seconds=int(eta_sec))) if eta_sec > 0 else "0:00:00",
        "throughput_jobs_per_sec": round(throughput, 6),
        "gpu_info": gpu_info,
        "policy": {
            "full_history_only": True,
            "sampling_allowed": False,
            "resume_enabled": True,
        },
    }


# ---------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------
def write_runner_log(log_path: Path, message: str) -> None:
    ts = utc_now_iso()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{ts} {message}\n")


def run_one_job(
    job: JobRecord,
    logs_dir: Path,
    extra_env: Dict[str, str],
    runner_log_path: Path,
) -> JobResult:
    start_perf = time.perf_counter()
    start_ts = utc_now_iso()

    job_slug = sanitize_filename(job.job_id)
    stdout_path = logs_dir / f"{job_slug}.stdout.log"
    stderr_path = logs_dir / f"{job_slug}.stderr.log"

    env = os.environ.copy()
    env.update(extra_env)

    write_runner_log(runner_log_path, f"[RUNNING] job_id={job.job_id} command={job.command}")

    with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("w", encoding="utf-8") as err_f:
        try:
            completed = subprocess.run(
                job.command,
                shell=True,
                stdout=out_f,
                stderr=err_f,
                env=env,
                check=False,
            )
            returncode = int(completed.returncode)
            status = STATE_DONE if returncode == 0 else STATE_FAILED
            error = ""
        except Exception as exc:
            returncode = -1
            status = STATE_FAILED
            error = f"{type(exc).__name__}: {exc}"
            err_f.write(error + "\n")
            err_f.write(traceback.format_exc())

    end_ts = utc_now_iso()
    duration = max(0.0, time.perf_counter() - start_perf)

    result = JobResult(
        job_id=job.job_id,
        status=status,
        returncode=returncode,
        start_ts_utc=start_ts,
        end_ts_utc=end_ts,
        duration_sec=duration,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        command=job.command,
        error=error,
    )

    write_runner_log(
        runner_log_path,
        f"[{status}] job_id={job.job_id} returncode={returncode} duration_sec={duration:.4f}",
    )
    return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VectorBT coverage master runner with robust manifest loading.")
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV.")
    parser.add_argument("--outroot", required=True, help="Root output directory.")
    parser.add_argument("--phase", default=None, help="Phase filter, e.g. micro_exit_expansion")
    parser.add_argument("--num-shards", type=int, default=8, help="Total shard count.")
    parser.add_argument("--shard-index", type=int, required=True, help="Target shard index.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent subprocess workers for this shard.")
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS, help="Progress flush interval.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep running after failed jobs.")
    parser.add_argument("--rerun-failed", action="store_true", help="Re-run jobs previously marked FAILED.")
    parser.add_argument("--force-rerun-done", action="store_true", help="Re-run jobs previously marked DONE.")
    parser.add_argument("--encoding-order", default=",".join(DEFAULT_ENCODINGS), help="Manifest encoding fallback order.")
    parser.add_argument("--gpu-mode", default="auto", choices=["auto", "off"], help="Detect GPU availability only.")
    parser.add_argument("--print-manifest-head", type=int, default=3, help="Print N manifest rows for inspection.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    manifest_path = Path(args.manifest)
    outroot = ensure_dir(Path(args.outroot))
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be > 0")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")

    shard_dir = ensure_dir(outroot / "shards" / f"shard_{args.shard_index}")
    logs_dir = ensure_dir(shard_dir / "logs")
    state_jsonl = shard_dir / STATE_JSONL_BASENAME
    live_progress_path = shard_dir / LIVE_PROGRESS_BASENAME
    summary_json_path = shard_dir / SUMMARY_JSON_BASENAME
    checklist_txt_path = shard_dir / CHECKLIST_TXT_BASENAME
    runner_log_path = shard_dir / RUNNER_LOG_BASENAME

    encodings = [x.strip() for x in str(args.encoding_order).split(",") if x.strip()]
    start_perf = time.perf_counter()

    write_runner_log(runner_log_path, f"[START] version={VERSION}")
    write_runner_log(runner_log_path, f"[START] manifest={manifest_path}")
    write_runner_log(runner_log_path, f"[START] outroot={outroot}")
    write_runner_log(runner_log_path, f"[START] phase={args.phase}")
    write_runner_log(runner_log_path, f"[START] shard_index={args.shard_index}/{args.num_shards}")
    write_runner_log(runner_log_path, f"[START] workers={args.workers}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df, used_encoding = load_manifest_csv(manifest_path, encodings)
    df = normalize_columns(df)
    block_sampling_columns(df)

    jobs_all = manifest_to_jobs(df)
    jobs_selected = select_jobs(
        jobs=jobs_all,
        phase=args.phase,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )

    existing_state = load_existing_state(state_jsonl)
    pending_jobs: List[JobRecord] = []
    skipped_done = 0
    skipped_failed = 0

    for job in jobs_selected:
        prev = existing_state.get(job.job_id)
        if not prev:
            pending_jobs.append(job)
            continue

        prev_status = str(prev.get("status", "")).strip().upper()
        if prev_status == STATE_DONE and not args.force_rerun_done:
            skipped_done += 1
            continue
        if prev_status == STATE_FAILED and not args.rerun_failed:
            skipped_failed += 1
            continue
        pending_jobs.append(job)

    checklist_lines = [
        f"version={VERSION}",
        f"generated_at_utc={utc_now_iso()}",
        "coverage_policy_checklist:",
        "- full_history_only = PASS",
        "- no_sampling = PASS",
        "- manifest_loaded = PASS",
        f"- manifest_encoding = {used_encoding}",
        f"- phase_filter = {args.phase}",
        f"- shard_filter = shard_{args.shard_index}_of_{args.num_shards}",
        "- resume_enabled = PASS",
        "- progress_eta_enabled = PASS",
    ]
    checklist_txt_path.write_text("\n".join(checklist_lines) + "\n", encoding="utf-8")

    gpu_info = detect_gpu() if args.gpu_mode == "auto" else {
        "gpu_mode": "off",
        "cupy_available": False,
        "cupy_device_count": 0,
        "cupy_error": "",
    }

    extra_env = {
        "LOCAL_LLM_COVERAGE_VERSION": VERSION,
        "LOCAL_LLM_COVERAGE_PHASE": str(args.phase or ""),
        "LOCAL_LLM_COVERAGE_SHARD_INDEX": str(args.shard_index),
        "LOCAL_LLM_COVERAGE_NUM_SHARDS": str(args.num_shards),
        "LOCAL_LLM_COVERAGE_FULL_HISTORY_ONLY": "1",
        "LOCAL_LLM_COVERAGE_NO_SAMPLING": "1",
        "LOCAL_LLM_COVERAGE_GPU_MODE": str(gpu_info.get("gpu_mode", "cpu_only")),
    }

    total_jobs = len(jobs_selected)
    done_jobs = sum(1 for x in existing_state.values() if x.get("status") == STATE_DONE and x.get("job_id") in {j.job_id for j in jobs_selected})
    failed_jobs = 0
    skipped_jobs = skipped_done + skipped_failed
    running_jobs = 0
    duration_samples: List[float] = []

    if args.print_manifest_head > 0:
        head_rows = min(args.print_manifest_head, len(jobs_selected))
        print("=" * 120)
        print(f"[INFO] version={VERSION}")
        print(f"[INFO] manifest_path={manifest_path}")
        print(f"[INFO] manifest_encoding={used_encoding}")
        print(f"[INFO] total_manifest_rows={len(df)}")
        print(f"[INFO] selected_rows={len(jobs_selected)}")
        print(f"[INFO] pending_rows={len(pending_jobs)}")
        print(f"[INFO] skipped_done={skipped_done}")
        print(f"[INFO] skipped_failed={skipped_failed}")
        print(f"[INFO] shard={args.shard_index}/{args.num_shards}")
        print(f"[INFO] gpu_info={json.dumps(gpu_info, ensure_ascii=False)}")
        print("-" * 120)
        for job in jobs_selected[:head_rows]:
            print(
                f"[JOB] row_index={job.row_index} job_id={job.job_id} phase={job.phase} "
                f"timeframe={job.timeframe} strategy_family={job.strategy_family}"
            )
            print(f"[JOB] command={job.command}")
        print("=" * 120)

    stop_event = threading.Event()
    lock = threading.Lock()

    def flush_progress() -> None:
        elapsed = max(0.0, time.perf_counter() - start_perf)
        avg_done = (sum(duration_samples) / len(duration_samples)) if duration_samples else 0.0
        remaining_pending = max(0, total_jobs - done_jobs - failed_jobs - skipped_jobs - running_jobs)
        payload = make_progress_payload(
            version=VERSION,
            manifest_path=manifest_path,
            manifest_encoding=used_encoding,
            outroot=outroot,
            shard_dir=shard_dir,
            phase=args.phase,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            workers=args.workers,
            total_jobs=total_jobs,
            done_jobs=done_jobs,
            failed_jobs=failed_jobs,
            running_jobs=running_jobs,
            skipped_jobs=skipped_jobs,
            pending_jobs=remaining_pending,
            elapsed_sec=elapsed,
            avg_done_duration_sec=avg_done,
            gpu_info=gpu_info,
        )
        json_dump(live_progress_path, payload)
        json_dump(summary_json_path, payload)

    def progress_loop() -> None:
        while not stop_event.is_set():
            with lock:
                flush_progress()
            stop_event.wait(args.poll_seconds)
        with lock:
            flush_progress()

    progress_thread = threading.Thread(target=progress_loop, daemon=True)
    progress_thread.start()

    try:
        if not pending_jobs:
            with lock:
                flush_progress()
            print("=" * 120)
            print(f"[DONE] version={VERSION}")
            print(f"[DONE] manifest={manifest_path}")
            print(f"[DONE] manifest_encoding={used_encoding}")
            print(f"[DONE] outroot={outroot}")
            print(f"[DONE] phase={args.phase}")
            print(f"[DONE] shard_dir={shard_dir}")
            print(f"[DONE] total_jobs={total_jobs}")
            print(f"[DONE] pending_jobs=0")
            print(f"[DONE] skipped_done={skipped_done}")
            print(f"[DONE] skipped_failed={skipped_failed}")
            print(f"[DONE] live_progress={live_progress_path}")
            print("=" * 120)
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map: Dict[concurrent.futures.Future[JobResult], JobRecord] = {}
            queue_iter = iter(pending_jobs)

            def submit_next() -> bool:
                nonlocal running_jobs
                try:
                    job = next(queue_iter)
                except StopIteration:
                    return False

                payload = {
                    "job_id": job.job_id,
                    "status": STATE_RUNNING,
                    "row_index": job.row_index,
                    "phase": job.phase,
                    "timeframe": job.timeframe,
                    "strategy_family": job.strategy_family,
                    "command": job.command,
                    "started_at_utc": utc_now_iso(),
                }
                append_jsonl(state_jsonl, payload)

                future = executor.submit(
                    run_one_job,
                    job=job,
                    logs_dir=logs_dir,
                    extra_env=extra_env,
                    runner_log_path=runner_log_path,
                )
                future_map[future] = job
                running_jobs += 1
                return True

            for _ in range(min(args.workers, len(pending_jobs))):
                submit_next()

            while future_map:
                done_set, _ = concurrent.futures.wait(
                    future_map.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for future in done_set:
                    job = future_map.pop(future)
                    running_jobs -= 1

                    try:
                        result = future.result()
                    except Exception as exc:
                        result = JobResult(
                            job_id=job.job_id,
                            status=STATE_FAILED,
                            returncode=-1,
                            start_ts_utc=utc_now_iso(),
                            end_ts_utc=utc_now_iso(),
                            duration_sec=0.0,
                            stdout_path="",
                            stderr_path="",
                            command=job.command,
                            error=f"{type(exc).__name__}: {exc}",
                        )

                    record = {
                        "job_id": result.job_id,
                        "status": result.status,
                        "returncode": result.returncode,
                        "row_index": job.row_index,
                        "phase": job.phase,
                        "timeframe": job.timeframe,
                        "strategy_family": job.strategy_family,
                        "start_ts_utc": result.start_ts_utc,
                        "end_ts_utc": result.end_ts_utc,
                        "duration_sec": round(result.duration_sec, 6),
                        "stdout_path": result.stdout_path,
                        "stderr_path": result.stderr_path,
                        "command": result.command,
                        "error": result.error,
                    }
                    append_jsonl(state_jsonl, record)

                    if result.status == STATE_DONE:
                        done_jobs += 1
                        duration_samples.append(result.duration_sec)
                    elif result.status == STATE_FAILED:
                        failed_jobs += 1
                        if not args.continue_on_error:
                            with lock:
                                flush_progress()
                            raise RuntimeError(f"Job failed and continue-on-error is disabled: {job.job_id}")

                    submit_next()

                with lock:
                    flush_progress()

    finally:
        stop_event.set()
        progress_thread.join(timeout=max(1.0, args.poll_seconds + 1.0))
        with contextlib.suppress(Exception):
            with lock:
                flush_progress()

    elapsed = max(0.0, time.perf_counter() - start_perf)
    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] manifest={manifest_path}")
    print(f"[DONE] manifest_encoding={used_encoding}")
    print(f"[DONE] outroot={outroot}")
    print(f"[DONE] phase={args.phase}")
    print(f"[DONE] shard_dir={shard_dir}")
    print(f"[DONE] total_jobs={total_jobs}")
    print(f"[DONE] done_jobs={done_jobs}")
    print(f"[DONE] failed_jobs={failed_jobs}")
    print(f"[DONE] skipped_jobs={skipped_jobs}")
    print(f"[DONE] elapsed_sec={elapsed:.4f}")
    print(f"[DONE] state_jsonl={state_jsonl}")
    print(f"[DONE] live_progress={live_progress_path}")
    print(f"[DONE] summary_json={summary_json_path}")
    print(f"[DONE] checklist_txt={checklist_txt_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()
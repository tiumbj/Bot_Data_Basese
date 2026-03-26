#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Code Name : run_vectorbt_coverage_master_v1_0_9.py
Version   : v1.0.9
Path      : C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_coverage_master_v1_0_0.py
Run       : python C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_coverage_master_v1_0_0.py --manifest C:\Data\Bot\central_backtest_results\research_coverage_master_v1_0_0\research_coverage_master_manifest.csv --data-root C:\Data\Bot\central_market_data\parquet --feature-root C:\Data\Bot\central_feature_cache --outdir C:\Data\Bot\central_backtest_results\coverage_master_runs_v1_0_0\micro_exit_expansion_single\run_0 --phase micro_exit_expansion --single-shard-only --progress-every 25 --continue-on-error --gpu-mode auto --worker-script-override C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_micro_exit_matrix_v1_0_0.py

Production notes
----------------
1) Stable single-process runner focused on resumability and durability.
2) Supports both explicit command manifests and schema-only manifests.
3) Fixes Windows child execution by storing command as args list and running subprocess with shell=False.
4) Worker override now wins over manifest script_path unless --prefer-manifest-script is explicitly enabled.
5) Phase-aware and worker-aware argument allowlist reduces immediate argparse/path failures in child workers.

Changelog v1.0.9
----------------
- Replaced string command execution with argv list execution on Windows and Linux.
- Added worker override precedence over manifest script fields by default.
- Added manifest-script opt-in flag: --prefer-manifest-script.
- Added safe phase-aware CLI builder with allowlist / denylist control.
- Added command preview files with argv payload for direct debugging.
- Added per-job command json snapshot beside stdout/stderr logs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

VERSION = "v1.0.9"
DEFAULT_ENCODINGS: Tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
STATE_RUNNING = "RUNNING"
STATE_DONE = "DONE"
STATE_FAILED = "FAILED"
FINAL_DONE = "DONE"
FINAL_DONE_WITH_ERRORS = "DONE_WITH_ERRORS"
FINAL_FAILED = "FAILED"

DEFAULT_PHASE_TO_WORKER = {
    "micro_exit_expansion": "run_vectorbt_micro_exit_matrix_v1_0_0.py",
    "micro_exit": "run_vectorbt_micro_exit_matrix_v1_0_0.py",
    "pending_logic": "run_vectorbt_pending_logic_matrix.py",
    "pending": "run_vectorbt_pending_logic_matrix.py",
    "uncovered": "run_vectorbt_uncovered_matrix.py",
    "uncovered_baseline": "run_vectorbt_uncovered_matrix.py",
    "coverage": "run_vectorbt_uncovered_matrix.py",
}

DIRECT_COMMAND_KEYS = [
    "command",
    "cmd",
    "run_command",
    "python_command",
    "job_command",
]
SCRIPT_KEYS = [
    "script_path",
    "script",
    "python_file",
    "python_path",
    "worker_script",
    "entry_script",
]
PYTHON_KEYS = ["python_exe", "python", "python_path", "py_exe"]
PHASE_KEYS = ["phase", "research_phase", "family", "category"]
JOB_ID_KEYS = ["job_id", "strategy_id", "id", "run_id", "name", "strategy_name"]
SYMBOL_KEYS = ["symbol", "ticker", "asset"]
TIMEFRAME_KEYS = ["timeframe", "tf", "bar_tf"]
OUTDIR_KEYS = ["outdir", "output_dir", "result_dir", "job_outdir"]

COMMON_SKIP_KEYS_FOR_ARGS = set(
    DIRECT_COMMAND_KEYS
    + SCRIPT_KEYS
    + PYTHON_KEYS
    + PHASE_KEYS
    + JOB_ID_KEYS
    + [
        "status",
        "state",
        "notes",
        "error",
        "command_preview",
        "selected",
        "rows_seen",
        "rows_selected",
    ]
)

BOOL_TRUE = {"true", "yes", "y", "on"}
BOOL_FALSE = {"false", "no", "n", "off"}
NUMERIC_BOOL_TRUE = {"1"}
NUMERIC_BOOL_FALSE = {"0"}

SAFE_COMMON_ARG_ALLOWLIST = {
    "symbol",
    "timeframe",
    "tf",
    "bar_tf",
    "start",
    "end",
    "date_from",
    "date_to",
    "market",
    "side",
    "direction",
    "family",
    "category",
    "regime",
    "session",
    "entry_name",
    "entry_family",
    "exit_name",
    "exit_family",
    "strategy",
    "strategy_name",
    "strategy_id",
    "template",
    "variant",
    "window_name",
    "validation_mode",
    "lookback",
    "atr_period",
    "adx_period",
    "ema_fast",
    "ema_slow",
    "ema_period",
    "bb_period",
    "bb_std",
    "risk_pct",
    "sl_atr",
    "tp_atr",
    "rr",
    "hold_bars",
    "cooldown_bars",
    "min_adx",
    "max_spread",
    "min_vol",
    "max_vol",
    "pending_logic",
    "micro_exit_name",
    "micro_exit_family",
    "micro_exit_mode",
    "micro_exit_profile",
    "micro_exit_version",
}
SAFE_COMMON_EXACT_DENYLIST = {
    "manifest",
    "manifest_path",
    "data_root",
    "feature_root",
    "phase",
    "outdir",
    "output_dir",
    "result_dir",
    "job_outdir",
    "stdout",
    "stderr",
    "bootstrap_log",
}
MICRO_EXIT_LEGACY_ALLOWLIST = {
    "symbol",
    "timeframe",
    "tf",
    "bar_tf",
    "micro_exit_name",
    "micro_exit_family",
    "micro_exit_mode",
    "micro_exit_profile",
    "micro_exit_version",
    "session",
    "regime",
    "side",
    "direction",
    "entry_name",
    "entry_family",
    "variant",
    "template",
    "strategy",
    "strategy_name",
    "strategy_id",
    "window_name",
    "validation_mode",
}
PENDING_ALLOWLIST = SAFE_COMMON_ARG_ALLOWLIST | {
    "signal_name",
    "signal_family",
    "pending_entry",
    "pending_offset",
    "pending_window",
}
UNCOVERED_ALLOWLIST = SAFE_COMMON_ARG_ALLOWLIST | {
    "baseline_name",
    "baseline_family",
    "entry_logic",
    "exit_logic",
}
MICRO_EXIT_FORCE_IGNORED_PREFIXES = ("path", "dir", "file", "manifest", "command", "script", "python")
MAX_SAMPLE_VALUE_CHARS = 300


@dataclass(frozen=True)
class Job:
    row_index: int
    job_id: str
    phase: str
    command_args: List[str]
    command_text: str
    script_path: str
    symbol: str
    timeframe: str
    outdir_hint: str
    raw_row: Dict[str, str]


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
    command_json_path: str
    command_text: str
    error: str
    row_index: int
    phase: str
    symbol: str
    timeframe: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: Dict) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def append_line(path: Path, line: str) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line.rstrip("\n") + "\n")


def write_bootstrap_log(path: Path, message: str) -> None:
    append_line(path, f"{utc_now_iso()} {message}")


def sanitize_filename(value: str) -> str:
    safe = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    joined = "".join(safe).strip("._")
    return joined or "job"


def stable_job_id(row_index: int, command_text: str, row: Dict[str, str]) -> str:
    for key in JOB_ID_KEYS:
        value = (row.get(key) or "").strip()
        if value:
            return sanitize_filename(value)
    payload = f"{row_index}|{command_text}|{json.dumps(row, sort_keys=True, ensure_ascii=False)}"
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:20]


def detect_csv_delimiter(header_line: str) -> str:
    counts = {
        ",": header_line.count(","),
        ";": header_line.count(";"),
        "\t": header_line.count("\t"),
        "|": header_line.count("|"),
    }
    return max(counts, key=counts.get)


def sniff_encoding_and_header(manifest_path: Path, encodings: Sequence[str]) -> Tuple[str, str, List[str]]:
    errors: List[str] = []
    for enc in encodings:
        try:
            with manifest_path.open("r", encoding=enc, newline="") as fh:
                header_line = fh.readline()
                if not header_line:
                    raise ValueError("Manifest is empty")
                delimiter = detect_csv_delimiter(header_line)
                reader = csv.reader([header_line], delimiter=delimiter)
                headers = next(reader)
                if len(headers) < 2:
                    raise ValueError(f"Header parse failed for encoding={enc}")
                return enc, delimiter, headers
        except Exception as exc:  # noqa: BLE001
            errors.append(f"encoding={enc} error={type(exc).__name__}: {exc}")
    raise RuntimeError("Unable to read manifest header. " + " | ".join(errors))


def iter_manifest_rows(manifest_path: Path, encoding: str, delimiter: str) -> Iterator[Tuple[int, Dict[str, str]]]:
    with manifest_path.open("r", encoding=encoding, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        for row_index, row in enumerate(reader):
            normalized = {str(k).strip(): ("" if v is None else str(v).strip()) for k, v in row.items()}
            yield row_index, normalized


def pick_first_nonempty(row: Dict[str, str], keys: Sequence[str]) -> str:
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def choose_phase(row: Dict[str, str], explicit_phase: Optional[str]) -> str:
    row_phase = pick_first_nonempty(row, PHASE_KEYS)
    if explicit_phase:
        return explicit_phase
    return row_phase or "UNKNOWN"


def matches_phase(row: Dict[str, str], explicit_phase: Optional[str]) -> bool:
    if not explicit_phase:
        return True
    explicit = explicit_phase.strip().lower()
    for key in PHASE_KEYS:
        value = (row.get(key) or "").strip().lower()
        if value == explicit:
            return True
    return False


def boolish(value: str) -> Optional[bool]:
    val = (value or "").strip().lower()
    if not val:
        return None
    if val in BOOL_TRUE or val in NUMERIC_BOOL_TRUE:
        return True
    if val in BOOL_FALSE or val in NUMERIC_BOOL_FALSE:
        return False
    return None


def normalize_flag_name(key: str) -> str:
    key = key.strip().replace(" ", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return "--" + key.replace("_", "-")


def should_skip_common_arg(key: str, value: str) -> bool:
    if key in COMMON_SKIP_KEYS_FOR_ARGS:
        return True
    if key in SAFE_COMMON_EXACT_DENYLIST:
        return True
    if value is None:
        return True
    text = str(value).strip()
    if text == "":
        return True
    return False


def should_allow_key_for_worker(key: str, worker_name: str, phase: str, value: str) -> bool:
    key_norm = key.strip().lower()
    if should_skip_common_arg(key_norm, value):
        return False

    if worker_name == "run_vectorbt_micro_exit_matrix_v1_0_0.py":
        if key_norm.startswith(MICRO_EXIT_FORCE_IGNORED_PREFIXES):
            return False
        return key_norm in MICRO_EXIT_LEGACY_ALLOWLIST

    phase_norm = (phase or "").strip().lower()
    if phase_norm in {"micro_exit_expansion", "micro_exit"}:
        if key_norm.startswith(MICRO_EXIT_FORCE_IGNORED_PREFIXES):
            return False
        return key_norm in MICRO_EXIT_LEGACY_ALLOWLIST

    if phase_norm in {"pending_logic", "pending"}:
        return key_norm in PENDING_ALLOWLIST

    if phase_norm in {"uncovered", "uncovered_baseline", "coverage"}:
        return key_norm in UNCOVERED_ALLOWLIST

    return key_norm in SAFE_COMMON_ARG_ALLOWLIST


def row_to_cli_parts(row: Dict[str, str], worker_name: str, phase: str) -> List[str]:
    parts: List[str] = []
    for raw_key, raw_value in row.items():
        key = raw_key.strip().lower()
        value = str(raw_value).strip()
        if not should_allow_key_for_worker(key, worker_name, phase, value):
            continue

        flag = normalize_flag_name(key)
        b = boolish(value)
        if b is True:
            parts.append(flag)
            continue
        if b is False:
            continue

        if len(value) > MAX_SAMPLE_VALUE_CHARS:
            value = value[:MAX_SAMPLE_VALUE_CHARS]
        parts.extend([flag, value])
    return parts


def choose_worker_path(args: argparse.Namespace, row: Dict[str, str], phase: str) -> str:
    if args.worker_script_override:
        return str(Path(args.worker_script_override).resolve())

    if args.prefer_manifest_script:
        explicit = pick_first_nonempty(row, SCRIPT_KEYS)
        if explicit:
            return explicit

    key = (phase or args.phase or "").strip().lower()
    worker_name = DEFAULT_PHASE_TO_WORKER.get(key, "")
    if worker_name:
        base_dir = Path(__file__).resolve().parent
        return str((base_dir / worker_name).resolve())

    explicit = pick_first_nonempty(row, SCRIPT_KEYS)
    if explicit:
        return explicit

    raise ValueError(
        f"Manifest row has no usable worker for phase={phase!r}. "
        f"Use --worker-script-override or enable --prefer-manifest-script."
    )


def build_direct_command_args(command_text: str) -> List[str]:
    if not command_text.strip():
        raise ValueError("Empty direct command")
    if os.name == "nt":
        import shlex
        return shlex.split(command_text, posix=False)
    import shlex
    return shlex.split(command_text)


def add_arg_if_missing(args_list: List[str], flag: str, value: Optional[str] = None) -> None:
    if flag in args_list:
        return
    args_list.append(flag)
    if value is not None:
        args_list.append(value)


def build_micro_exit_legacy_args(
    row_index: int,
    row: Dict[str, str],
    worker: str,
    args: argparse.Namespace,
    root_outdir: Path,
) -> List[str]:
    py_exe = pick_first_nonempty(row, PYTHON_KEYS) or sys.executable
    argv: List[str] = [py_exe, worker]
    symbol = pick_first_nonempty(row, SYMBOL_KEYS)
    timeframe = pick_first_nonempty(row, TIMEFRAME_KEYS)

    if symbol:
        argv.extend(["--symbol", symbol])
    if timeframe:
        argv.extend(["--timeframes", timeframe])

    passthrough = row_to_cli_parts(row=row, worker_name=Path(worker).name.lower(), phase=args.phase or "")
    argv.extend(passthrough)

    job_id_seed = pick_first_nonempty(row, JOB_ID_KEYS) or f"row_{row_index:08d}"
    job_slug = sanitize_filename(job_id_seed)
    job_outdir = pick_first_nonempty(row, OUTDIR_KEYS)
    if not job_outdir:
        job_outdir = str(root_outdir / "jobs" / job_slug)

    add_arg_if_missing(argv, "--outdir", job_outdir)
    return argv


def build_general_worker_args(
    row_index: int,
    row: Dict[str, str],
    worker: str,
    phase: str,
    args: argparse.Namespace,
    root_outdir: Path,
) -> List[str]:
    py_exe = pick_first_nonempty(row, PYTHON_KEYS) or sys.executable
    argv: List[str] = [py_exe, worker]
    worker_name = Path(worker).name.lower()
    argv.extend(row_to_cli_parts(row=row, worker_name=worker_name, phase=phase))

    if args.phase:
        add_arg_if_missing(argv, "--phase", args.phase)
    if args.data_root:
        add_arg_if_missing(argv, "--data-root", str(args.data_root))
    if args.feature_root:
        add_arg_if_missing(argv, "--feature-root", str(args.feature_root))

    job_id_seed = pick_first_nonempty(row, JOB_ID_KEYS) or f"row_{row_index:08d}"
    job_slug = sanitize_filename(job_id_seed)
    job_outdir = pick_first_nonempty(row, OUTDIR_KEYS)
    if not job_outdir:
        job_outdir = str(root_outdir / "jobs" / job_slug)
    add_arg_if_missing(argv, "--outdir", job_outdir)

    if args.child_continue_on_error:
        add_arg_if_missing(argv, "--continue-on-error")
    if args.child_gpu_mode:
        add_arg_if_missing(argv, "--gpu-mode", args.child_gpu_mode)
    if args.child_progress_every:
        add_arg_if_missing(argv, "--progress-every", str(args.child_progress_every))
    add_arg_if_missing(argv, "--manifest-row-index", str(row_index))
    add_arg_if_missing(argv, "--runner-version", VERSION)
    return argv


def argv_to_text(argv: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(x) for x in argv])


def build_fallback_command(
    row_index: int,
    row: Dict[str, str],
    phase: str,
    args: argparse.Namespace,
    root_outdir: Path,
) -> Tuple[List[str], str]:
    worker = choose_worker_path(args=args, row=row, phase=phase)
    worker_name = Path(worker).name.lower()

    if worker_name == "run_vectorbt_micro_exit_matrix_v1_0_0.py":
        argv = build_micro_exit_legacy_args(row_index=row_index, row=row, worker=worker, args=args, root_outdir=root_outdir)
    else:
        argv = build_general_worker_args(row_index=row_index, row=row, worker=worker, phase=phase, args=args, root_outdir=root_outdir)

    if len(argv) < 2:
        raise ValueError("Command builder produced empty argv")
    return argv, worker


def extract_command(
    row_index: int,
    row: Dict[str, str],
    phase: str,
    args: argparse.Namespace,
    root_outdir: Path,
) -> Tuple[List[str], str]:
    if not args.worker_script_override:
        direct = pick_first_nonempty(row, DIRECT_COMMAND_KEYS)
        if direct:
            return build_direct_command_args(direct), pick_first_nonempty(row, SCRIPT_KEYS)

    argv, worker = build_fallback_command(
        row_index=row_index,
        row=row,
        phase=phase,
        args=args,
        root_outdir=root_outdir,
    )
    return argv, worker


def build_job(
    row_index: int,
    row: Dict[str, str],
    explicit_phase: Optional[str],
    args: argparse.Namespace,
    root_outdir: Path,
) -> Job:
    phase = choose_phase(row, explicit_phase)
    command_args, script_path = extract_command(
        row_index=row_index,
        row=row,
        phase=phase,
        args=args,
        root_outdir=root_outdir,
    )
    command_text = argv_to_text(command_args)
    job_id = stable_job_id(row_index=row_index, command_text=command_text, row=row)
    symbol = pick_first_nonempty(row, SYMBOL_KEYS)
    timeframe = pick_first_nonempty(row, TIMEFRAME_KEYS)
    outdir_hint = pick_first_nonempty(row, OUTDIR_KEYS)
    return Job(
        row_index=row_index,
        job_id=job_id,
        phase=phase,
        command_args=command_args,
        command_text=command_text,
        script_path=script_path,
        symbol=symbol,
        timeframe=timeframe,
        outdir_hint=outdir_hint,
        raw_row=row,
    )


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_last_states(path: Path) -> Dict[str, str]:
    states: Dict[str, str] = {}
    if not path.exists():
        return states
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                job_id = str(payload.get("job_id", "")).strip()
                status = str(payload.get("status", "")).strip()
                if job_id and status:
                    states[job_id] = status
            except Exception:
                continue
    return states


def save_state_record(path: Path, payload: Dict) -> None:
    append_line(path, json.dumps(payload, ensure_ascii=False))


def probe_gpu(mode: str, bootstrap_log_path: Path) -> Dict[str, str]:
    info: Dict[str, str] = {
        "requested_mode": mode,
        "active_mode": "off",
        "cupy_available": "0",
        "device_count": "0",
        "device_name": "",
        "cuda_runtime_version": "",
        "error": "",
    }
    if mode == "off":
        write_bootstrap_log(bootstrap_log_path, "GPU_PROBE mode=off")
        return info
    try:
        import cupy as cp  # type: ignore

        info["cupy_available"] = "1"
        device_count = int(cp.cuda.runtime.getDeviceCount())
        info["device_count"] = str(device_count)
        if device_count <= 0:
            info["error"] = "No CUDA device found"
            write_bootstrap_log(bootstrap_log_path, "GPU_PROBE no_device_found")
            return info
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props.get("name", b"")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")
        info["device_name"] = str(name)
        runtime_ver = cp.cuda.runtime.runtimeGetVersion()
        info["cuda_runtime_version"] = str(runtime_ver)
        arr = cp.arange(1024, dtype=cp.float32)
        _ = float(arr.sum().get())
        info["active_mode"] = "cupy"
        write_bootstrap_log(
            bootstrap_log_path,
            f"GPU_PROBE active_mode=cupy device_count={device_count} device_name={info['device_name']} runtime={runtime_ver}",
        )
        return info
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
        write_bootstrap_log(bootstrap_log_path, f"GPU_PROBE_FAILED error={info['error']}")
        return info


def build_child_env(args: argparse.Namespace, gpu_info: Dict[str, str]) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["LOCAL_LLM_COVERAGE_RUNNER_VERSION"] = VERSION
    env["LOCAL_LLM_COVERAGE_PHASE"] = args.phase or ""
    env["LOCAL_LLM_GPU_MODE"] = gpu_info.get("active_mode", "off")
    env["LOCAL_LLM_CUPY_AVAILABLE"] = gpu_info.get("cupy_available", "0")
    env["LOCAL_LLM_CUDA_DEVICE_COUNT"] = gpu_info.get("device_count", "0")
    if gpu_info.get("device_name"):
        env["LOCAL_LLM_CUDA_DEVICE_NAME"] = gpu_info["device_name"]
    if args.data_root:
        env["LOCAL_LLM_DATA_ROOT"] = str(args.data_root)
    if args.feature_root:
        env["LOCAL_LLM_FEATURE_ROOT"] = str(args.feature_root)
    if args.cuda_path:
        env["CUDA_PATH"] = str(args.cuda_path)
        env["CUDA_HOME"] = str(args.cuda_path)
    return env


def flush_progress(
    path: Path,
    *,
    version: str,
    status: str,
    manifest_path: Path,
    outdir: Path,
    phase: Optional[str],
    total_jobs: int,
    done_jobs: int,
    failed_jobs: int,
    skipped_jobs: int,
    current_job_id: str,
    current_row_index: int,
    gpu_info: Dict[str, str],
    started_at_utc: str,
) -> None:
    progress_pct = 0.0 if total_jobs <= 0 else round((done_jobs / total_jobs) * 100.0, 4)
    payload = {
        "version": version,
        "status": status,
        "manifest_path": str(manifest_path),
        "outdir": str(outdir),
        "phase": phase,
        "single_shard_only": True,
        "total_jobs": total_jobs,
        "done_jobs": done_jobs,
        "failed_jobs": failed_jobs,
        "skipped_jobs": skipped_jobs,
        "remaining_jobs": max(total_jobs - done_jobs - skipped_jobs, 0),
        "progress_pct": progress_pct,
        "current_job_id": current_job_id,
        "current_row_index": current_row_index,
        "gpu": gpu_info,
        "started_at_utc": started_at_utc,
        "updated_at_utc": utc_now_iso(),
    }
    atomic_write_json(path, payload)


def flush_summary(
    path: Path,
    *,
    version: str,
    final_status: str,
    manifest_path: Path,
    outdir: Path,
    phase: Optional[str],
    total_jobs: int,
    done_jobs: int,
    failed_jobs: int,
    skipped_jobs: int,
    elapsed_sec: float,
    gpu_info: Dict[str, str],
    started_at_utc: str,
) -> None:
    payload = {
        "version": version,
        "final_status": final_status,
        "manifest_path": str(manifest_path),
        "outdir": str(outdir),
        "phase": phase,
        "single_shard_only": True,
        "total_jobs": total_jobs,
        "done_jobs": done_jobs,
        "failed_jobs": failed_jobs,
        "skipped_jobs": skipped_jobs,
        "elapsed_sec": round(elapsed_sec, 6),
        "gpu": gpu_info,
        "started_at_utc": started_at_utc,
        "finished_at_utc": utc_now_iso(),
    }
    atomic_write_json(path, payload)


def write_command_snapshot(path: Path, argv: Sequence[str], command_text: str, job: Job) -> None:
    payload = {
        "job_id": job.job_id,
        "row_index": job.row_index,
        "phase": job.phase,
        "script_path": job.script_path,
        "command_text": command_text,
        "command_args": list(argv),
        "symbol": job.symbol,
        "timeframe": job.timeframe,
    }
    atomic_write_json(path, payload)


def run_job(job: Job, *, logs_dir: Path, child_env: Dict[str, str], runner_log_path: Path) -> JobResult:
    start_perf = time.perf_counter()
    start_ts = utc_now_iso()
    job_slug = sanitize_filename(job.job_id)
    stdout_path = logs_dir / f"{job_slug}.stdout.log"
    stderr_path = logs_dir / f"{job_slug}.stderr.log"
    command_json_path = logs_dir / f"{job_slug}.command.json"
    write_command_snapshot(command_json_path, job.command_args, job.command_text, job)

    write_bootstrap_log(
        runner_log_path,
        f"JOB_START job_id={job.job_id} row_index={job.row_index} script={job.script_path}",
    )

    error = ""
    returncode = -1
    status = STATE_FAILED

    with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("w", encoding="utf-8") as err_f:
        try:
            completed = subprocess.run(
                job.command_args,
                shell=False,
                stdout=out_f,
                stderr=err_f,
                env=child_env,
                check=False,
            )
            returncode = int(completed.returncode)
            status = STATE_DONE if returncode == 0 else STATE_FAILED
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            err_f.write(error + "\n")
            err_f.write(traceback.format_exc())

    duration = max(0.0, time.perf_counter() - start_perf)
    end_ts = utc_now_iso()

    if status == STATE_FAILED and not error and returncode != 0:
        error = f"Child process failed with returncode={returncode}"

    write_bootstrap_log(
        runner_log_path,
        f"JOB_END job_id={job.job_id} status={status} returncode={returncode} duration_sec={duration:.4f}",
    )

    return JobResult(
        job_id=job.job_id,
        status=status,
        returncode=returncode,
        start_ts_utc=start_ts,
        end_ts_utc=end_ts,
        duration_sec=duration,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        command_json_path=str(command_json_path),
        command_text=job.command_text,
        error=error,
        row_index=job.row_index,
        phase=job.phase,
        symbol=job.symbol,
        timeframe=job.timeframe,
    )


def select_jobs(
    manifest_path: Path,
    *,
    encodings: Sequence[str],
    phase: Optional[str],
    bootstrap_log_path: Path,
    progress_every_chunks: int,
    completed_ids: set[str],
    last_states: Dict[str, str],
    rerun_failed: bool,
    force_rerun_done: bool,
    args: argparse.Namespace,
    root_outdir: Path,
) -> Tuple[List[Job], str, str, int, int]:
    encoding, delimiter, headers = sniff_encoding_and_header(manifest_path, encodings)
    write_bootstrap_log(
        bootstrap_log_path,
        f"LOAD_MANIFEST_HEADER_OK encoding={encoding} delimiter={repr(delimiter)} columns={len(headers)} header={headers[:60]}",
    )

    selected_jobs: List[Job] = []
    rows_seen = 0
    rows_selected = 0
    chunk_counter = 0
    samples_logged = 0

    for row_index, row in iter_manifest_rows(manifest_path, encoding, delimiter):
        rows_seen += 1
        if phase and not matches_phase(row, phase):
            continue

        if samples_logged < max(args.log_row_samples, 0):
            phase_guess = choose_phase(row, phase)
            try:
                preview_args, preview_worker = extract_command(
                    row_index=row_index,
                    row=row,
                    phase=phase_guess,
                    args=args,
                    root_outdir=root_outdir,
                )
                preview_status = "OK"
                preview_command = argv_to_text(preview_args)
            except Exception as exc:  # noqa: BLE001
                preview_command = ""
                preview_worker = ""
                preview_status = f"ERROR {type(exc).__name__}: {exc}"
            sample_payload = {
                "row_index": row_index,
                "phase_guess": phase_guess,
                "preview_status": preview_status,
                "preview_worker": preview_worker,
                "row": {k: row[k] for k in list(row.keys())[: args.max_sample_columns]},
                "command_preview": preview_command[: args.command_preview_chars],
            }
            write_bootstrap_log(bootstrap_log_path, f"ROW_SAMPLE {json.dumps(sample_payload, ensure_ascii=False)}")
            samples_logged += 1

        try:
            job = build_job(
                row_index=row_index,
                row=row,
                explicit_phase=phase,
                args=args,
                root_outdir=root_outdir,
            )
        except Exception as exc:  # noqa: BLE001
            write_bootstrap_log(bootstrap_log_path, f"ROW_SKIP row_index={row_index} reason={type(exc).__name__}: {exc}")
            continue

        prior_state = last_states.get(job.job_id, "")
        already_completed = job.job_id in completed_ids or prior_state == STATE_DONE
        already_failed = prior_state == STATE_FAILED

        if already_completed and not force_rerun_done:
            continue
        if already_failed and not rerun_failed:
            continue

        selected_jobs.append(job)
        rows_selected += 1

        if rows_seen % 200000 == 0:
            chunk_counter += 1
            if chunk_counter % max(progress_every_chunks, 1) == 0:
                write_bootstrap_log(
                    bootstrap_log_path,
                    f"SELECT_JOBS_PROGRESS chunks={chunk_counter} rows_seen={rows_seen} rows_selected={rows_selected}",
                )

    write_bootstrap_log(
        bootstrap_log_path,
        f"SELECT_JOBS_OK rows_seen={rows_seen} rows_selected={rows_selected} encoding={encoding}",
    )
    return selected_jobs, encoding, delimiter, rows_seen, rows_selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stable single-shard VectorBT coverage runner with Windows-safe argv execution."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--feature-root")
    parser.add_argument("--outdir")
    parser.add_argument("--outroot")
    parser.add_argument("--phase")
    parser.add_argument("--single-shard-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--force-rerun-done", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--sleep-after-job-ms", type=int, default=0)
    parser.add_argument("--gpu-mode", choices=["auto", "off"], default="auto")
    parser.add_argument("--encoding-order", default=",".join(DEFAULT_ENCODINGS))
    parser.add_argument("--dry-run-preview", type=int, default=0)
    parser.add_argument("--worker-script-override")
    parser.add_argument("--prefer-manifest-script", action="store_true")
    parser.add_argument("--child-progress-every", type=int, default=25)
    parser.add_argument("--child-gpu-mode", default="auto")
    parser.add_argument("--child-continue-on-error", action="store_true")
    parser.add_argument("--cuda-path")
    parser.add_argument("--log-row-samples", type=int, default=3)
    parser.add_argument("--max-sample-columns", type=int, default=40)
    parser.add_argument("--command-preview-chars", type=int, default=800)
    args = parser.parse_args()
    if not args.single_shard_only:
        args.single_shard_only = True
    return args


def resolve_outdir(args: argparse.Namespace) -> Path:
    if args.outdir:
        return Path(args.outdir)
    if args.outroot:
        return Path(args.outroot) / "single_shard_run"
    raise SystemExit("Either --outdir or --outroot is required.")


def main() -> None:
    started_at_utc = utc_now_iso()
    main_perf = time.perf_counter()
    args = parse_args()

    manifest_path = Path(args.manifest)
    outdir = resolve_outdir(args)
    ensure_dir(outdir)
    logs_dir = outdir / "job_logs"
    ensure_dir(logs_dir)

    bootstrap_log_path = outdir / "bootstrap.log"
    state_path = outdir / "state.jsonl"
    completed_ids_path = outdir / "completed_ids.txt"
    live_progress_path = outdir / "live_progress.json"
    summary_path = outdir / "summary.json"

    write_bootstrap_log(bootstrap_log_path, f"START version={VERSION}")
    write_bootstrap_log(
        bootstrap_log_path,
        f"ARGS manifest={manifest_path} data_root={args.data_root} feature_root={args.feature_root} outdir={outdir} phase={args.phase} single_shard_only=1 gpu_mode={args.gpu_mode} worker_override={args.worker_script_override} prefer_manifest_script={int(args.prefer_manifest_script)}",
    )
    print(
        f"[RUN] phase={args.phase} | manifest={manifest_path} | outdir={outdir} | bootstrap_log={bootstrap_log_path}"
    )

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    encodings = [enc.strip() for enc in args.encoding_order.split(",") if enc.strip()]
    completed_ids = load_completed_ids(completed_ids_path)
    last_states = load_last_states(state_path)
    gpu_info = probe_gpu(args.gpu_mode, bootstrap_log_path)
    child_env = build_child_env(args, gpu_info)

    flush_progress(
        live_progress_path,
        version=VERSION,
        status="BOOTSTRAP_START",
        manifest_path=manifest_path,
        outdir=outdir,
        phase=args.phase,
        total_jobs=0,
        done_jobs=0,
        failed_jobs=0,
        skipped_jobs=0,
        current_job_id="",
        current_row_index=-1,
        gpu_info=gpu_info,
        started_at_utc=started_at_utc,
    )

    write_bootstrap_log(bootstrap_log_path, "LOAD_MANIFEST_BEGIN")
    jobs, encoding, delimiter, rows_seen, rows_selected = select_jobs(
        manifest_path,
        encodings=encodings,
        phase=args.phase,
        bootstrap_log_path=bootstrap_log_path,
        progress_every_chunks=10,
        completed_ids=completed_ids,
        last_states=last_states,
        rerun_failed=args.rerun_failed,
        force_rerun_done=args.force_rerun_done,
        args=args,
        root_outdir=outdir,
    )
    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]
    total_jobs = len(jobs)

    write_bootstrap_log(
        bootstrap_log_path,
        f"LOAD_MANIFEST_OK rows_seen={rows_seen} selected={rows_selected} encoding={encoding} delimiter={repr(delimiter)} total_jobs={total_jobs}",
    )

    if args.dry_run_preview > 0:
        preview_payload = []
        for job in jobs[: args.dry_run_preview]:
            preview_payload.append(
                {
                    "row_index": job.row_index,
                    "job_id": job.job_id,
                    "phase": job.phase,
                    "script_path": job.script_path,
                    "command_text": job.command_text,
                    "command_args": job.command_args,
                }
            )
        atomic_write_json(outdir / "dry_run_preview.json", {"version": VERSION, "preview_jobs": preview_payload})
        write_bootstrap_log(bootstrap_log_path, f"DRY_RUN_PREVIEW_WRITTEN count={len(preview_payload)}")
        print(
            f"[DRY_RUN] dry_run_preview={args.dry_run_preview} | generated_preview_jobs={len(preview_payload)} | no_backtest_executed"
        )
        flush_summary(
            summary_path,
            version=VERSION,
            final_status="DRY_RUN",
            manifest_path=manifest_path,
            outdir=outdir,
            phase=args.phase,
            total_jobs=total_jobs,
            done_jobs=0,
            failed_jobs=0,
            skipped_jobs=0,
            elapsed_sec=max(0.0, time.perf_counter() - main_perf),
            gpu_info=gpu_info,
            started_at_utc=started_at_utc,
        )
        return

    flush_progress(
        live_progress_path,
        version=VERSION,
        status="RUN_JOBS_BEGIN",
        manifest_path=manifest_path,
        outdir=outdir,
        phase=args.phase,
        total_jobs=total_jobs,
        done_jobs=0,
        failed_jobs=0,
        skipped_jobs=0,
        current_job_id="",
        current_row_index=-1,
        gpu_info=gpu_info,
        started_at_utc=started_at_utc,
    )

    done_jobs = 0
    failed_jobs = 0
    skipped_jobs = 0
    encountered_error = False

    write_bootstrap_log(bootstrap_log_path, "RUN_JOBS_BEGIN")

    for idx, job in enumerate(jobs, start=1):
        save_state_record(
            state_path,
            {
                "ts_utc": utc_now_iso(),
                "version": VERSION,
                "job_id": job.job_id,
                "status": STATE_RUNNING,
                "row_index": job.row_index,
                "phase": job.phase,
                "symbol": job.symbol,
                "timeframe": job.timeframe,
                "command_text": job.command_text,
                "command_args": job.command_args,
                "script_path": job.script_path,
            },
        )

        flush_progress(
            live_progress_path,
            version=VERSION,
            status=STATE_RUNNING,
            manifest_path=manifest_path,
            outdir=outdir,
            phase=args.phase,
            total_jobs=total_jobs,
            done_jobs=done_jobs,
            failed_jobs=failed_jobs,
            skipped_jobs=skipped_jobs,
            current_job_id=job.job_id,
            current_row_index=job.row_index,
            gpu_info=gpu_info,
            started_at_utc=started_at_utc,
        )

        result = run_job(job, logs_dir=logs_dir, child_env=child_env, runner_log_path=bootstrap_log_path)
        save_state_record(
            state_path,
            {
                "ts_utc": utc_now_iso(),
                "version": VERSION,
                "job_id": result.job_id,
                "status": result.status,
                "returncode": result.returncode,
                "start_ts_utc": result.start_ts_utc,
                "end_ts_utc": result.end_ts_utc,
                "duration_sec": result.duration_sec,
                "stdout_path": result.stdout_path,
                "stderr_path": result.stderr_path,
                "command_json_path": result.command_json_path,
                "phase": result.phase,
                "row_index": result.row_index,
                "symbol": result.symbol,
                "timeframe": result.timeframe,
                "command_text": result.command_text,
                "error": result.error,
            },
        )

        if result.status == STATE_DONE:
            append_line(completed_ids_path, result.job_id)
            completed_ids.add(result.job_id)
            done_jobs += 1
        else:
            failed_jobs += 1
            encountered_error = True
            if not args.continue_on_error:
                write_bootstrap_log(bootstrap_log_path, f"STOP_ON_ERROR job_id={job.job_id}")
                flush_progress(
                    live_progress_path,
                    version=VERSION,
                    status=FINAL_FAILED,
                    manifest_path=manifest_path,
                    outdir=outdir,
                    phase=args.phase,
                    total_jobs=total_jobs,
                    done_jobs=done_jobs,
                    failed_jobs=failed_jobs,
                    skipped_jobs=skipped_jobs,
                    current_job_id=job.job_id,
                    current_row_index=job.row_index,
                    gpu_info=gpu_info,
                    started_at_utc=started_at_utc,
                )
                flush_summary(
                    summary_path,
                    version=VERSION,
                    final_status=FINAL_FAILED,
                    manifest_path=manifest_path,
                    outdir=outdir,
                    phase=args.phase,
                    total_jobs=total_jobs,
                    done_jobs=done_jobs,
                    failed_jobs=failed_jobs,
                    skipped_jobs=skipped_jobs,
                    elapsed_sec=max(0.0, time.perf_counter() - main_perf),
                    gpu_info=gpu_info,
                    started_at_utc=started_at_utc,
                )
                raise SystemExit(result.returncode if result.returncode != 0 else 1)

        if idx % max(args.progress_every, 1) == 0 or idx == total_jobs:
            flush_progress(
                live_progress_path,
                version=VERSION,
                status=STATE_RUNNING,
                manifest_path=manifest_path,
                outdir=outdir,
                phase=args.phase,
                total_jobs=total_jobs,
                done_jobs=done_jobs,
                failed_jobs=failed_jobs,
                skipped_jobs=skipped_jobs,
                current_job_id=job.job_id,
                current_row_index=job.row_index,
                gpu_info=gpu_info,
                started_at_utc=started_at_utc,
            )
            flush_summary(
                summary_path,
                version=VERSION,
                final_status=STATE_RUNNING,
                manifest_path=manifest_path,
                outdir=outdir,
                phase=args.phase,
                total_jobs=total_jobs,
                done_jobs=done_jobs,
                failed_jobs=failed_jobs,
                skipped_jobs=skipped_jobs,
                elapsed_sec=max(0.0, time.perf_counter() - main_perf),
                gpu_info=gpu_info,
                started_at_utc=started_at_utc,
            )

        if args.sleep_after_job_ms > 0:
            time.sleep(args.sleep_after_job_ms / 1000.0)

    final_status = FINAL_DONE_WITH_ERRORS if encountered_error else FINAL_DONE
    flush_progress(
        live_progress_path,
        version=VERSION,
        status=final_status,
        manifest_path=manifest_path,
        outdir=outdir,
        phase=args.phase,
        total_jobs=total_jobs,
        done_jobs=done_jobs,
        failed_jobs=failed_jobs,
        skipped_jobs=skipped_jobs,
        current_job_id="",
        current_row_index=-1,
        gpu_info=gpu_info,
        started_at_utc=started_at_utc,
    )
    flush_summary(
        summary_path,
        version=VERSION,
        final_status=final_status,
        manifest_path=manifest_path,
        outdir=outdir,
        phase=args.phase,
        total_jobs=total_jobs,
        done_jobs=done_jobs,
        failed_jobs=failed_jobs,
        skipped_jobs=skipped_jobs,
        elapsed_sec=max(0.0, time.perf_counter() - main_perf),
        gpu_info=gpu_info,
        started_at_utc=started_at_utc,
    )
    write_bootstrap_log(
        bootstrap_log_path,
        f"FINISH final_status={final_status} total_jobs={total_jobs} done_jobs={done_jobs} failed_jobs={failed_jobs}",
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        err = f"FATAL {type(exc).__name__}: {exc}"
        print(err, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise

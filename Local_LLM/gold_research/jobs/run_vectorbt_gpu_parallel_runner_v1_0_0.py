"""
Code Name : run_vectorbt_gpu_parallel_runner_v1_0_0.py
Version   : v1.0.0
Path      : C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_gpu_parallel_runner_v1_0_0.py
Run       : python C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_gpu_parallel_runner_v1_0_0.py --manifest C:\Data\Bot\central_backtest_results\research_coverage_master_v1_0_0\research_coverage_master_manifest.csv --data-root C:\Data\Bot\central_market_data\parquet --feature-root C:\Data\Bot\central_feature_cache --outroot C:\Data\Bot\central_backtest_results\coverage_master_runs_v1_0_0\gpu_parallel_micro_exit --phase micro_exit_expansion --runner-script C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_coverage_master_v1_0_0.py --worker-script-override C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_micro_exit_matrix_v1_0_0.py --max-parallel-runners 3 --target-gpu-util-pct 65 --max-gpu-mem-pct 80 --poll-seconds 20 --progress-every 25 --continue-on-error

Production notes
----------------
1) This file does NOT rewrite the existing worker logic. It scales throughput by launching multiple isolated shard runners in parallel.
2) Each shard gets its own manifest and outdir, so resume/state/completed_ids remain independent and durable.
3) Scheduler is GPU-aware: it starts more shard runners only when GPU utilization and VRAM are below user-defined thresholds.
4) Best use case: existing worker can access CuPy/CUDA but one process under-utilizes the GPU. Parallel shard runners can raise total GPU occupancy.
5) This file is production-safe for Windows because every child process is launched with argv list and shell=False.

Changelog v1.0.0
----------------
- Create GPU-aware parallel orchestrator for coverage master workloads.
- Split one CSV manifest into deterministic shard CSV files without editing the source manifest.
- Launch multiple existing single-shard runners concurrently with separate outdirs.
- Read live progress from child live_progress.json files and aggregate master progress.
- Use nvidia-smi polling to gate new launches by GPU utilization and GPU memory thresholds.
- Preserve resumability because each shard reuses its own state/completed_ids files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


VERSION = "v1.0.0"
DEFAULT_ENCODINGS: Tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
LIVE_STATUS_RUNNING = "RUNNING"
LIVE_STATUS_DONE = "DONE"
LIVE_STATUS_FAILED = "FAILED"
LIVE_STATUS_WAITING = "WAITING"
LIVE_STATUS_LAUNCHING = "LAUNCHING"
LIVE_STATUS_COMPLETE = "COMPLETE"


@dataclass
class GpuSnapshot:
    timestamp_utc: str
    util_pct: int
    mem_used_mib: int
    mem_total_mib: int
    mem_pct: float
    temperature_c: Optional[int]
    power_w: Optional[float]
    power_limit_w: Optional[float]
    raw_line: str


@dataclass
class ShardRuntime:
    shard_index: int
    shard_manifest_path: str
    shard_outdir: str
    total_jobs: int
    process_pid: int = 0
    process_returncode: Optional[int] = None
    launch_ts_utc: str = ""
    end_ts_utc: str = ""
    status: str = LIVE_STATUS_WAITING


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


def write_log(log_path: Path, message: str) -> None:
    append_line(log_path, f"{utc_now_iso()} {message}")


def read_json_safe(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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
                    raise ValueError("manifest is empty")
                delimiter = detect_csv_delimiter(header_line)
                reader = csv.reader([header_line], delimiter=delimiter)
                headers = next(reader)
                if len(headers) < 2:
                    raise ValueError("unable to parse manifest header")
                return enc, delimiter, headers
        except Exception as exc:  # noqa: BLE001
            errors.append(f"encoding={enc} error={type(exc).__name__}: {exc}")
    raise RuntimeError("Unable to read manifest header. " + " | ".join(errors))


def iter_manifest_rows(manifest_path: Path, encoding: str, delimiter: str):
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


def row_matches_phase(row: Dict[str, str], phase: Optional[str]) -> bool:
    if not phase:
        return True
    phase_keys = ("phase", "research_phase", "family", "category")
    target = phase.strip().lower()
    for key in phase_keys:
        if (row.get(key) or "").strip().lower() == target:
            return True
    return False


def create_shard_manifests(
    manifest_path: Path,
    *,
    phase: Optional[str],
    shard_count: int,
    outroot: Path,
    encodings: Sequence[str],
    log_path: Path,
) -> List[ShardRuntime]:
    encoding, delimiter, headers = sniff_encoding_and_header(manifest_path, encodings)
    manifests_dir = outroot / "generated_manifests"
    ensure_dir(manifests_dir)

    shard_paths = [manifests_dir / f"coverage_master_shard_{i:02d}.csv" for i in range(shard_count)]
    shard_outdirs = [outroot / f"shard_{i:02d}" for i in range(shard_count)]
    counts = [0 for _ in range(shard_count)]

    writers = []
    files = []
    try:
        for shard_path in shard_paths:
            ensure_dir(shard_path.parent)
            fh = shard_path.open("w", encoding="utf-8", newline="")
            files.append(fh)
            writer = csv.DictWriter(fh, fieldnames=headers, delimiter=delimiter)
            writer.writeheader()
            writers.append(writer)

        rows_seen = 0
        rows_selected = 0
        for row_index, row in iter_manifest_rows(manifest_path, encoding, delimiter):
            rows_seen += 1
            if not row_matches_phase(row, phase):
                continue
            shard_index = row_index % shard_count
            writers[shard_index].writerow(row)
            counts[shard_index] += 1
            rows_selected += 1
            if rows_seen % 50000 == 0:
                write_log(
                    log_path,
                    f"CREATE_SHARDS_PROGRESS rows_seen={rows_seen} rows_selected={rows_selected}",
                )
    finally:
        for fh in files:
            fh.close()

    runtimes: List[ShardRuntime] = []
    for shard_index in range(shard_count):
        runtimes.append(
            ShardRuntime(
                shard_index=shard_index,
                shard_manifest_path=str(shard_paths[shard_index]),
                shard_outdir=str(shard_outdirs[shard_index]),
                total_jobs=counts[shard_index],
            )
        )

    write_log(
        log_path,
        f"CREATE_SHARDS_DONE encoding={encoding} delimiter={repr(delimiter)} shard_count={shard_count} totals={counts}",
    )
    return runtimes


def query_gpu_snapshot(log_path: Path, gpu_index: int = 0) -> GpuSnapshot:
    cmd = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        stdout = (completed.stdout or "").strip()
        if completed.returncode != 0 or not stdout:
            raise RuntimeError(completed.stderr.strip() or "nvidia-smi returned no data")
        line = stdout.splitlines()[0].strip()
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            raise RuntimeError(f"Unexpected nvidia-smi output: {line}")
        util = int(float(parts[0]))
        mem_used = int(float(parts[1]))
        mem_total = int(float(parts[2]))
        temp = int(float(parts[3])) if parts[3] else None
        power_w = float(parts[4]) if parts[4] else None
        power_limit_w = float(parts[5]) if parts[5] else None
        mem_pct = 0.0 if mem_total <= 0 else round((mem_used / mem_total) * 100.0, 4)
        return GpuSnapshot(
            timestamp_utc=utc_now_iso(),
            util_pct=util,
            mem_used_mib=mem_used,
            mem_total_mib=mem_total,
            mem_pct=mem_pct,
            temperature_c=temp,
            power_w=power_w,
            power_limit_w=power_limit_w,
            raw_line=line,
        )
    except Exception as exc:  # noqa: BLE001
        write_log(log_path, f"GPU_QUERY_FAILED error={type(exc).__name__}: {exc}")
        return GpuSnapshot(
            timestamp_utc=utc_now_iso(),
            util_pct=0,
            mem_used_mib=0,
            mem_total_mib=0,
            mem_pct=0.0,
            temperature_c=None,
            power_w=None,
            power_limit_w=None,
            raw_line="",
        )


def can_launch_more(
    *,
    snapshot: GpuSnapshot,
    active_count: int,
    max_parallel_runners: int,
    target_gpu_util_pct: int,
    max_gpu_mem_pct: int,
) -> bool:
    if active_count >= max_parallel_runners:
        return False
    if snapshot.mem_total_mib > 0 and snapshot.mem_pct >= float(max_gpu_mem_pct):
        return False
    if snapshot.util_pct >= target_gpu_util_pct:
        return False
    return True


def build_child_command(args: argparse.Namespace, shard: ShardRuntime) -> List[str]:
    cmd = [
        sys.executable,
        str(Path(args.runner_script)),
        "--manifest",
        shard.shard_manifest_path,
        "--data-root",
        str(Path(args.data_root)),
        "--feature-root",
        str(Path(args.feature_root)),
        "--outdir",
        shard.shard_outdir,
        "--single-shard-only",
        "--progress-every",
        str(args.progress_every),
        "--gpu-mode",
        args.child_gpu_mode,
    ]
    if args.phase:
        cmd.extend(["--phase", args.phase])
    if args.continue_on_error:
        cmd.append("--continue-on-error")
    if args.rerun_failed:
        cmd.append("--rerun-failed")
    if args.force_rerun_done:
        cmd.append("--force-rerun-done")
    if args.worker_script_override:
        cmd.extend(["--worker-script-override", str(Path(args.worker_script_override))])
    if args.prefer_manifest_script:
        cmd.append("--prefer-manifest-script")
    if args.cuda_path:
        cmd.extend(["--cuda-path", str(Path(args.cuda_path))])
    if args.child_progress_every > 0:
        cmd.extend(["--child-progress-every", str(args.child_progress_every)])
    if args.child_continue_on_error:
        cmd.append("--child-continue-on-error")
    if args.max_jobs_per_shard > 0:
        cmd.extend(["--max-jobs", str(args.max_jobs_per_shard)])
    if args.sleep_after_job_ms > 0:
        cmd.extend(["--sleep-after-job-ms", str(args.sleep_after_job_ms)])
    return cmd


def start_shard(
    *,
    args: argparse.Namespace,
    shard: ShardRuntime,
    proc_map: Dict[int, subprocess.Popen],
    parent_log_path: Path,
) -> None:
    ensure_dir(Path(shard.shard_outdir))
    stdout_path = Path(shard.shard_outdir) / "parent_launch.stdout.log"
    stderr_path = Path(shard.shard_outdir) / "parent_launch.stderr.log"
    cmd = build_child_command(args, shard)
    with stdout_path.open("a", encoding="utf-8") as out_f, stderr_path.open("a", encoding="utf-8") as err_f:
        process = subprocess.Popen(  # noqa: S603
            cmd,
            shell=False,
            stdout=out_f,
            stderr=err_f,
            cwd=str(Path(shard.shard_outdir)),
            env=os.environ.copy(),
        )
    shard.process_pid = int(process.pid)
    shard.launch_ts_utc = utc_now_iso()
    shard.status = LIVE_STATUS_RUNNING
    proc_map[shard.shard_index] = process
    write_log(
        parent_log_path,
        f"LAUNCH shard={shard.shard_index} pid={process.pid} total_jobs={shard.total_jobs} cmd={subprocess.list2cmdline(cmd)}",
    )


def summarize_child_progress(shard: ShardRuntime) -> Dict:
    child_live_path = Path(shard.shard_outdir) / "live_progress.json"
    child_summary_path = Path(shard.shard_outdir) / "summary.json"
    child_live = read_json_safe(child_live_path)
    child_summary = read_json_safe(child_summary_path)

    total_jobs = int(child_live.get("total_jobs", shard.total_jobs) or shard.total_jobs)
    done_jobs = int(child_live.get("done_jobs", 0) or 0)
    failed_jobs = int(child_live.get("failed_jobs", 0) or 0)
    skipped_jobs = int(child_live.get("skipped_jobs", 0) or 0)

    status = shard.status
    if child_live:
        status = str(child_live.get("status", status) or status)
    if shard.process_returncode is not None:
        status = LIVE_STATUS_DONE if shard.process_returncode == 0 else LIVE_STATUS_FAILED
    elif child_summary.get("final_status") in {"DONE", "DONE_WITH_ERRORS"}:
        status = LIVE_STATUS_DONE
    elif child_summary.get("final_status") == "FAILED":
        status = LIVE_STATUS_FAILED

    return {
        "shard_index": shard.shard_index,
        "status": status,
        "pid": shard.process_pid,
        "returncode": shard.process_returncode,
        "launch_ts_utc": shard.launch_ts_utc,
        "end_ts_utc": shard.end_ts_utc,
        "manifest_path": shard.shard_manifest_path,
        "outdir": shard.shard_outdir,
        "total_jobs": total_jobs,
        "done_jobs": done_jobs,
        "failed_jobs": failed_jobs,
        "skipped_jobs": skipped_jobs,
        "progress_pct": 0.0 if total_jobs <= 0 else round((done_jobs / total_jobs) * 100.0, 4),
        "current_job_id": child_live.get("current_job_id", ""),
        "current_row_index": child_live.get("current_row_index", -1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-aware parallel orchestrator for VectorBT coverage runners.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--outroot", required=True)
    parser.add_argument("--phase")
    parser.add_argument("--runner-script", required=True)
    parser.add_argument("--worker-script-override")
    parser.add_argument("--prefer-manifest-script", action="store_true")
    parser.add_argument("--max-parallel-runners", type=int, default=3)
    parser.add_argument("--shard-count", type=int, default=6)
    parser.add_argument("--target-gpu-util-pct", type=int, default=65)
    parser.add_argument("--max-gpu-mem-pct", type=int, default=80)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--child-progress-every", type=int, default=25)
    parser.add_argument("--child-gpu-mode", choices=["auto", "off"], default="auto")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--child-continue-on-error", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--force-rerun-done", action="store_true")
    parser.add_argument("--max-jobs-per-shard", type=int, default=0)
    parser.add_argument("--sleep-after-job-ms", type=int, default=0)
    parser.add_argument("--cuda-path")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--encoding-order", default=",".join(DEFAULT_ENCODINGS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at_utc = utc_now_iso()

    outroot = Path(args.outroot)
    ensure_dir(outroot)

    log_path = outroot / "gpu_parallel_runner.log"
    live_path = outroot / "gpu_parallel_live_status.json"
    summary_path = outroot / "gpu_parallel_summary.json"

    write_log(log_path, f"START version={VERSION}")
    write_log(
        log_path,
        f"ARGS manifest={args.manifest} outroot={args.outroot} phase={args.phase} shard_count={args.shard_count} max_parallel_runners={args.max_parallel_runners}",
    )

    encodings = [enc.strip() for enc in args.encoding_order.split(",") if enc.strip()]
    shard_runtimes = create_shard_manifests(
        Path(args.manifest),
        phase=args.phase,
        shard_count=args.shard_count,
        outroot=outroot,
        encodings=encodings,
        log_path=log_path,
    )

    shard_queue = [shard for shard in shard_runtimes if shard.total_jobs > 0]
    proc_map: Dict[int, subprocess.Popen] = {}

    if not shard_queue:
        payload = {
            "version": VERSION,
            "status": LIVE_STATUS_COMPLETE,
            "started_at_utc": started_at_utc,
            "finished_at_utc": utc_now_iso(),
            "message": "No jobs matched the selected phase.",
            "shards": [asdict(x) for x in shard_runtimes],
        }
        atomic_write_json(live_path, payload)
        atomic_write_json(summary_path, payload)
        return

    while True:
        gpu_snapshot = query_gpu_snapshot(log_path, gpu_index=args.gpu_index)

        for shard_index, process in list(proc_map.items()):
            rc = process.poll()
            if rc is None:
                continue
            shard = shard_runtimes[shard_index]
            shard.process_returncode = int(rc)
            shard.end_ts_utc = utc_now_iso()
            shard.status = LIVE_STATUS_DONE if rc == 0 else LIVE_STATUS_FAILED
            write_log(log_path, f"CHILD_EXIT shard={shard_index} pid={shard.process_pid} returncode={rc}")
            proc_map.pop(shard_index, None)

        active_count = len(proc_map)
        while shard_queue and can_launch_more(
            snapshot=gpu_snapshot,
            active_count=active_count,
            max_parallel_runners=args.max_parallel_runners,
            target_gpu_util_pct=args.target_gpu_util_pct,
            max_gpu_mem_pct=args.max_gpu_mem_pct,
        ):
            next_shard = shard_queue.pop(0)
            next_shard.status = LIVE_STATUS_LAUNCHING
            start_shard(
                args=args,
                shard=next_shard,
                proc_map=proc_map,
                parent_log_path=log_path,
            )
            active_count = len(proc_map)
            time.sleep(2)
            gpu_snapshot = query_gpu_snapshot(log_path, gpu_index=args.gpu_index)

        shard_status_rows = [summarize_child_progress(shard) for shard in shard_runtimes]
        agg_total = sum(row["total_jobs"] for row in shard_status_rows)
        agg_done = sum(row["done_jobs"] for row in shard_status_rows)
        agg_failed = sum(row["failed_jobs"] for row in shard_status_rows)
        agg_skipped = sum(row["skipped_jobs"] for row in shard_status_rows)

        all_finished = (not shard_queue) and (len(proc_map) == 0)
        final_status = LIVE_STATUS_COMPLETE if all_finished else LIVE_STATUS_RUNNING

        live_payload = {
            "version": VERSION,
            "status": final_status,
            "started_at_utc": started_at_utc,
            "updated_at_utc": utc_now_iso(),
            "manifest": str(Path(args.manifest)),
            "outroot": str(outroot),
            "phase": args.phase,
            "gpu": asdict(gpu_snapshot),
            "scheduler": {
                "shard_count": args.shard_count,
                "max_parallel_runners": args.max_parallel_runners,
                "active_runners": len(proc_map),
                "queued_shards": len(shard_queue),
                "target_gpu_util_pct": args.target_gpu_util_pct,
                "max_gpu_mem_pct": args.max_gpu_mem_pct,
                "poll_seconds": args.poll_seconds,
            },
            "aggregate": {
                "total_jobs": agg_total,
                "done_jobs": agg_done,
                "failed_jobs": agg_failed,
                "skipped_jobs": agg_skipped,
                "remaining_jobs": max(agg_total - agg_done - agg_skipped, 0),
                "progress_pct": 0.0 if agg_total <= 0 else round((agg_done / agg_total) * 100.0, 4),
            },
            "shards": shard_status_rows,
        }
        atomic_write_json(live_path, live_payload)

        if all_finished:
            summary_payload = dict(live_payload)
            summary_payload["finished_at_utc"] = utc_now_iso()
            atomic_write_json(summary_path, summary_payload)
            write_log(log_path, f"FINISH total_jobs={agg_total} done_jobs={agg_done} failed_jobs={agg_failed}")
            break

        time.sleep(max(args.poll_seconds, 3))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL {type(exc).__name__}: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise

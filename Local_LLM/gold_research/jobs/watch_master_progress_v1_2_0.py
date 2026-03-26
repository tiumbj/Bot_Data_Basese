#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ชื่อโค้ด: watch_master_progress_v1_2_0.py
ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\watch_master_progress_v1_2_0.py
คำสั่งรัน:
    python C:\Data\Bot\Local_LLM\gold_research\jobs\watch_master_progress_v1_2_0.py --master-dir C:\Data\Bot\central_backtest_results\master_research_orchestrator_v1_0_1 --interval 2
เวอร์ชัน: v1.2.0

เป้าหมาย:
- ดู % progression ของ master orchestrator แบบ real-time
- อ่าน phase_statuses ได้จริง
- แยกสถานะว่า progress file ยัง heartbeat อยู่หรือ frozen
- ดู worker/state/progress JSON ใต้ master-dir
- ไม่แก้ไฟล์งาน เป็น monitor อย่างเดียว

สิ่งที่เพิ่มจาก v1.1.0:
- รองรับ phase_statuses
- รองรับ checked_at_utc
- รองรับ elapsed_min / overall_eta_remaining_min
- แสดง diagnostics ว่า stale ที่ master layer
- สรุป phase counters ได้ถูกต้อง
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


VERSION = "v1.2.0"
PROGRESS_BAR_WIDTH = 44
DEFAULT_STALE_SECONDS = 180
MAX_WORKER_FILES = 32
MAX_RAW_PREVIEW = 2200


@dataclass
class Snapshot:
    path: Path
    data: Optional[Dict[str, Any]]
    mtime: Optional[datetime]
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch master research orchestrator progress and worker progress in real time."
    )
    parser.add_argument(
        "--master-dir",
        required=True,
        help="Path to master orchestrator output directory",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Refresh interval in seconds. Default=2",
    )
    parser.add_argument(
        "--stale-seconds",
        type=int,
        default=DEFAULT_STALE_SECONDS,
        help="Seconds before progress is considered stale. Default=180",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one snapshot and exit",
    )
    return parser.parse_args()


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def now_local() -> datetime:
    return datetime.now()


def file_mtime(path: Path) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except Exception:
        return None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_json_text(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        return {"_root": obj}
    except Exception:
        return None


def safe_read_json(path: Path) -> Snapshot:
    if not path.exists():
        return Snapshot(path=path, data=None, mtime=None, error="missing")

    try:
        text = read_text(path)
        data = parse_json_text(text)
        if data is not None:
            return Snapshot(path=path, data=data, mtime=file_mtime(path), error=None)

        stripped = text.strip()
        for cut in range(1, min(len(stripped), 400) + 1):
            candidate = stripped[:-cut].rstrip()
            if not candidate:
                break
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return Snapshot(
                        path=path,
                        data=obj,
                        mtime=file_mtime(path),
                        error="partial_json_recovered",
                    )
                return Snapshot(
                    path=path,
                    data={"_root": obj},
                    mtime=file_mtime(path),
                    error="partial_json_recovered",
                )
            except Exception:
                continue

        return Snapshot(path=path, data=None, mtime=file_mtime(path), error="invalid_json")
    except Exception as exc:
        return Snapshot(path=path, data=None, mtime=file_mtime(path), error=f"read_error: {exc}")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def textify(value: Any, default: str = "-") -> str:
    if value is None or value == "":
        return default
    return str(value)


def first_present(data: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return default


def parse_ts(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None

    if isinstance(value, (int, float)):
        try:
            ts = float(value)
            if ts > 1e12:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts)
        except Exception:
            return None

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None

        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"

        try:
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is not None:
                return dt.astimezone().replace(tzinfo=None)
            return dt.replace(tzinfo=None)
        except Exception:
            pass

        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%d/%m/%Y %H:%M:%S",
        ):
            try:
                return datetime.strptime(raw, fmt)
            except Exception:
                continue

    return None


def fmt_dt(dt: Optional[datetime]) -> str:
    if dt is None:
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def fmt_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    if math.isnan(seconds) or math.isinf(seconds):
        return "-"
    if seconds < 0:
        seconds = 0
    total = int(round(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def age_seconds(dt: Optional[datetime], ref_now: datetime) -> Optional[float]:
    if dt is None:
        return None
    return max(0.0, (ref_now - dt).total_seconds())


def build_bar(pct: float, width: int = PROGRESS_BAR_WIDTH) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(round(width * pct / 100.0))
    filled = max(0, min(width, filled))
    return "#" * filled + "-" * (width - filled)


def find_progress_pct(data: Dict[str, Any]) -> float:
    for key in (
        "overall_progress_pct",
        "progress_pct",
        "pct",
        "percent",
        "progress_percent",
        "completion_pct",
    ):
        if key in data:
            value = safe_float(data.get(key), -1.0)
            if 0.0 <= value <= 100.0:
                return value

    done_jobs = safe_float(first_present(data, ("done_jobs", "completed_jobs", "processed_jobs"), 0), 0)
    total_jobs = safe_float(first_present(data, ("total_jobs", "expected_jobs", "job_count"), 0), 0)
    if total_jobs > 0:
        return max(0.0, min(100.0, (done_jobs / total_jobs) * 100.0))

    phases_done = safe_float(first_present(data, ("phases_done", "done_phases"), 0), 0)
    phases_total = safe_float(first_present(data, ("phases_total", "total_phases"), 0), 0)
    if phases_total > 0:
        return max(0.0, min(100.0, (phases_done / phases_total) * 100.0))

    return 0.0


def find_started_at(data: Dict[str, Any], mtime: Optional[datetime]) -> Optional[datetime]:
    started = parse_ts(first_present(data, ("started_at", "start_time", "run_started_at", "created_at")))
    if started is not None:
        return started

    elapsed_min = safe_float(data.get("elapsed_min"), -1.0)
    checked_at = parse_ts(first_present(data, ("checked_at_utc", "checked_at", "updated_at", "last_update")))
    if checked_at is not None and elapsed_min >= 0:
        return checked_at - timedelta(minutes=elapsed_min)

    return mtime


def find_updated_at(data: Dict[str, Any], mtime: Optional[datetime]) -> Optional[datetime]:
    for key in (
        "checked_at_utc",
        "checked_at",
        "updated_at",
        "last_update",
        "last_updated_at",
        "heartbeat_at",
        "written_at",
        "timestamp",
        "ts",
    ):
        dt = parse_ts(data.get(key))
        if dt is not None:
            return dt
    return mtime


def estimate_eta(data: Dict[str, Any], started_at: Optional[datetime], pct: float, ref_now: datetime) -> Tuple[Optional[float], Optional[datetime]]:
    eta_min = safe_float(data.get("overall_eta_remaining_min"), -1.0)
    if eta_min >= 0:
        eta_seconds = eta_min * 60.0
        return eta_seconds, ref_now + timedelta(seconds=eta_seconds)

    if started_at is None:
        return None, None
    if pct <= 0.0 or pct >= 100.0:
        return None, None

    elapsed = (ref_now - started_at).total_seconds()
    if elapsed <= 0:
        return None, None

    rate = pct / elapsed
    if rate <= 0:
        return None, None

    remaining = (100.0 - pct) / rate
    return remaining, ref_now + timedelta(seconds=remaining)


def infer_phase_rows(data: Dict[str, Any]) -> List[Tuple[str, str, bool]]:
    rows: List[Tuple[str, str, bool]] = []
    current_phase = textify(first_present(data, ("current_phase", "running_phase", "phase")), "-")

    phase_statuses = data.get("phase_statuses")
    if isinstance(phase_statuses, dict) and phase_statuses:
        for phase_name, status in phase_statuses.items():
            name = textify(phase_name)
            st = textify(status).upper()
            rows.append((name, st, name == current_phase or st == "RUNNING"))
        rows.sort(key=lambda x: x[0])
        return rows

    phase_order = data.get("phase_order")
    phase_status = data.get("phase_status")
    if isinstance(phase_order, list) and isinstance(phase_status, dict):
        for phase_name in phase_order:
            name = textify(phase_name)
            st = textify(phase_status.get(name, "UNKNOWN")).upper()
            rows.append((name, st, name == current_phase or st == "RUNNING"))
        return rows

    phases = data.get("phases")
    if isinstance(phases, list):
        for item in phases:
            if isinstance(item, dict):
                name = textify(first_present(item, ("name", "phase_name", "phase")), "UNKNOWN_PHASE")
                st = textify(first_present(item, ("status", "state")), "UNKNOWN").upper()
                rows.append((name, st, name == current_phase or st == "RUNNING"))
        if rows:
            return rows

    for key in sorted(data.keys()):
        if key.startswith("phase_") and isinstance(data.get(key), (str, int, float)):
            st = textify(data.get(key)).upper()
            rows.append((key, st, key == current_phase or st == "RUNNING"))

    return rows


def summarize_phase_counts(rows: List[Tuple[str, str, bool]]) -> Tuple[int, int, int, int]:
    done = running = pending = error = 0
    for _, status, _ in rows:
        s = status.upper()
        if any(x in s for x in ("DONE", "SUCCESS", "COMPLETE", "COMPLETED", "FINISHED")):
            done += 1
        elif any(x in s for x in ("RUNNING", "IN_PROGRESS", "ACTIVE")):
            running += 1
        elif any(x in s for x in ("ERROR", "FAIL", "FAILED", "CRASH")):
            error += 1
        else:
            pending += 1
    return done, running, pending, error


def classify_health(updated_at: Optional[datetime], ref_now: datetime, stale_seconds: int) -> str:
    age = age_seconds(updated_at, ref_now)
    if age is None:
        return "NO_TS"
    if age > stale_seconds:
        return f"STALE {int(age)}s"
    return f"OK {int(age)}s"


def worker_kind(path: Path) -> str:
    low = str(path).lower()
    if "uncovered" in low:
        return "uncovered_matrix"
    if "pending" in low:
        return "pending_logic_matrix"
    if "micro" in low:
        return "micro_exit_matrix"
    if "aggregate" in low or "registry" in low:
        return "registry_aggregation"
    if "feature" in low and "cache" in low:
        return "feature_cache"
    if "orchestrator_state" in low:
        return "orchestrator_state"
    return path.stem


def find_worker_files(master_dir: Path) -> List[Path]:
    patterns = [
        "*progress*.json",
        "*live*.json",
        "*state*.json",
        "*status*.json",
        "*summary*.json",
    ]
    files: List[Path] = []
    seen = set()

    for pattern in patterns:
        for path in master_dir.rglob(pattern):
            try:
                resolved = str(path.resolve())
            except Exception:
                resolved = str(path)
            if resolved in seen:
                continue
            if path.name == "live_progress.json" and path.parent == master_dir:
                continue
            seen.add(resolved)
            files.append(path)

    def sort_key(path: Path) -> Tuple[int, float]:
        low = str(path).lower()
        priority = 100
        if "orchestrator_state" in low:
            priority = 1
        elif "uncovered" in low:
            priority = 2
        elif "pending" in low:
            priority = 3
        elif "micro" in low:
            priority = 4
        elif "aggregate" in low or "registry" in low:
            priority = 5
        elif "feature" in low and "cache" in low:
            priority = 6
        mt = path.stat().st_mtime if path.exists() else 0.0
        return (priority, -mt)

    files.sort(key=sort_key)
    return files[:MAX_WORKER_FILES]


def extract_worker_row(snapshot: Snapshot) -> Dict[str, Any]:
    data = snapshot.data or {}
    pct = find_progress_pct(data)
    status = textify(first_present(data, ("status", "state")), "UNKNOWN").upper()

    if status == "UNKNOWN":
        if pct >= 100.0:
            status = "DONE"
        elif pct > 0.0:
            status = "RUNNING"

    started_at = find_started_at(data, snapshot.mtime)
    updated_at = find_updated_at(data, snapshot.mtime)

    return {
        "worker": worker_kind(snapshot.path),
        "status": status,
        "progress_pct": pct,
        "done_jobs": safe_int(first_present(data, ("done_jobs", "completed_jobs", "processed_jobs"), 0), 0),
        "running_jobs": safe_int(first_present(data, ("running_jobs", "active_jobs"), 0), 0),
        "error_jobs": safe_int(first_present(data, ("error_jobs", "failed_jobs"), 0), 0),
        "total_jobs": safe_int(first_present(data, ("total_jobs", "expected_jobs", "job_count"), 0), 0),
        "tf": textify(first_present(data, ("current_timeframe", "timeframe", "tf")), "-"),
        "step": textify(first_present(data, ("current_step", "step", "current_phase")), "-"),
        "started_at": started_at,
        "updated_at": updated_at,
        "note": snapshot.error or "-",
        "path": str(snapshot.path),
    }


def collect_worker_rows(master_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in find_worker_files(master_dir):
        snap = safe_read_json(path)
        if snap.data is None and snap.error == "missing":
            continue
        rows.append(extract_worker_row(snap))
    return rows


def render_master(snapshot: Snapshot, ref_now: datetime, stale_seconds: int) -> List[str]:
    lines: List[str] = []
    lines.append("=" * 136)
    lines.append(f"MASTER RESEARCH ORCHESTRATOR WATCHER | version={VERSION}")
    lines.append("=" * 136)
    lines.append(f"now                : {fmt_dt(ref_now)}")
    lines.append(f"master_progress    : {snapshot.path}")
    lines.append(f"master_file_status : {'READ_OK' if snapshot.data is not None else 'NOT_READY'}")
    if snapshot.error:
        lines.append(f"master_read_note   : {snapshot.error}")

    if snapshot.data is None:
        lines.append("status             : waiting for live_progress.json")
        lines.append("=" * 136)
        return lines

    data = snapshot.data
    pct = find_progress_pct(data)
    current_phase = textify(first_present(data, ("current_phase", "running_phase", "phase")), "-")
    status = textify(first_present(data, ("status", "state")), "RUNNING").upper()
    started_at = find_started_at(data, snapshot.mtime)
    updated_at = find_updated_at(data, snapshot.mtime)
    elapsed_seconds = None
    elapsed_min = safe_float(data.get("elapsed_min"), -1.0)
    if elapsed_min >= 0:
        elapsed_seconds = elapsed_min * 60.0
    elif started_at is not None:
        elapsed_seconds = max(0.0, (ref_now - started_at).total_seconds())

    eta_seconds, eta_finish = estimate_eta(data, started_at, pct, ref_now)
    health = classify_health(updated_at, ref_now, stale_seconds)

    phases_total = safe_int(first_present(data, ("phases_total", "total_phases")), 0)
    phases_done = safe_int(first_present(data, ("phases_done", "done_phases")), 0)
    phases_running = safe_int(first_present(data, ("phases_running", "running_phases")), 0)
    phases_failed = safe_int(first_present(data, ("phases_failed", "failed_phases")), 0)

    lines.append(f"status             : {status}")
    lines.append(f"current_phase      : {current_phase}")
    lines.append(f"health             : {health}")
    lines.append(f"started_at         : {fmt_dt(started_at)}")
    lines.append(f"updated_at         : {fmt_dt(updated_at)}")
    lines.append(f"elapsed            : {fmt_seconds(elapsed_seconds)}")
    lines.append(f"eta_remaining      : {fmt_seconds(eta_seconds)}")
    lines.append(f"eta_finish         : {fmt_dt(eta_finish)}")
    lines.append("-" * 136)
    lines.append(f"overall_progress   : [{build_bar(pct)}] {pct:6.2f}%")
    if phases_total > 0:
        lines.append(
            f"phase_summary      : done={phases_done} running={phases_running} failed={phases_failed} total={phases_total}"
        )

    phase_rows = infer_phase_rows(data)
    done_count, running_count, pending_count, error_count = summarize_phase_counts(phase_rows)
    lines.append(
        f"phase_counters     : done={done_count} running={running_count} pending={pending_count} error={error_count}"
    )

    if age_seconds(updated_at, ref_now) is not None and age_seconds(updated_at, ref_now) > stale_seconds:
        lines.append("-" * 136)
        lines.append("ALERT              : MASTER PROGRESS FILE FROZEN")
        lines.append("diagnostic         : orchestrator อยู่ที่ phase_2_uncovered_matrix แต่ live_progress.json ไม่ heartbeat ต่อ")
        lines.append("impact             : progress %, ETA, elapsed ที่เห็นอาจไม่สะท้อนงานจริงปัจจุบัน")

    lines.append("-" * 136)
    lines.append("MASTER PHASES")
    if phase_rows:
        for idx, (name, st, is_current) in enumerate(phase_rows, start=1):
            marker = ">>" if is_current else "  "
            lines.append(f"{marker} {idx:02d}. {name:<54} {st}")
    else:
        lines.append("  phase detail not found")

    return lines


def render_workers(worker_rows: List[Dict[str, Any]], ref_now: datetime, stale_seconds: int) -> List[str]:
    lines: List[str] = []
    done = running = pending = error = 0

    for row in worker_rows:
        st = row["status"].upper()
        if any(x in st for x in ("DONE", "SUCCESS", "COMPLETED", "FINISHED")):
            done += 1
        elif any(x in st for x in ("RUNNING", "IN_PROGRESS", "ACTIVE")):
            running += 1
        elif any(x in st for x in ("ERROR", "FAIL", "FAILED", "CRASH")):
            error += 1
        else:
            pending += 1

    lines.append("-" * 136)
    lines.append(f"WORKER COUNTERS    : done={done} running={running} pending={pending} error={error}")
    lines.append("-" * 136)
    lines.append("WORKER SNAPSHOT")

    if not worker_rows:
        lines.append("  no worker progress/state files found")
        return lines

    header = (
        f"{'worker':<24} {'status':<14} {'progress':<11} {'health':<14} "
        f"{'jobs(d/r/e/t)':<16} {'tf':<8} {'step':<20} path"
    )
    lines.append(header)
    lines.append("-" * 136)

    for row in worker_rows:
        health = classify_health(row["updated_at"], ref_now, stale_seconds)
        jobs_text = f"{row['done_jobs']}/{row['running_jobs']}/{row['error_jobs']}/{row['total_jobs']}"
        lines.append(
            f"{row['worker']:<24} "
            f"{row['status']:<14} "
            f"{row['progress_pct']:6.2f}%    "
            f"{health:<14} "
            f"{jobs_text:<16} "
            f"{row['tf']:<8} "
            f"{row['step'][:20]:<20} "
            f"{row['path']}"
        )

    return lines


def render_raw_preview(snapshot: Snapshot) -> List[str]:
    lines: List[str] = []
    if snapshot.data is None:
        return lines

    lines.append("-" * 136)
    lines.append("MASTER RAW PREVIEW")
    try:
        preview = json.dumps(snapshot.data, ensure_ascii=False, indent=2)
    except Exception:
        preview = str(snapshot.data)

    if len(preview) > MAX_RAW_PREVIEW:
        preview = preview[:MAX_RAW_PREVIEW] + "\n... (truncated)"
    lines.append(preview)
    return lines


def render_screen(master_snapshot: Snapshot, worker_rows: List[Dict[str, Any]], ref_now: datetime, stale_seconds: int) -> str:
    parts: List[str] = []
    parts.extend(render_master(master_snapshot, ref_now, stale_seconds))
    parts.extend(render_workers(worker_rows, ref_now, stale_seconds))
    parts.extend(render_raw_preview(master_snapshot))
    parts.append("=" * 136)
    parts.append("Ctrl+C = stop watcher")
    parts.append("=" * 136)
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    master_dir = Path(args.master_dir)
    master_progress = master_dir / "live_progress.json"

    while True:
        ref_now = now_local()
        master_snapshot = safe_read_json(master_progress)
        worker_rows = collect_worker_rows(master_dir)

        clear_screen()
        print(render_screen(master_snapshot, worker_rows, ref_now, args.stale_seconds))

        if args.once:
            break

        try:
            time.sleep(max(0.5, args.interval))
        except KeyboardInterrupt:
            print("\n[STOPPED] watcher terminated by user")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] watcher terminated by user")
        sys.exit(0)
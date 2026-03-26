#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ชื่อโค้ด: watch_master_progress_v1_1_0.py
ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\watch_master_progress_v1_1_0.py
คำสั่งรัน:
    python C:\Data\Bot\Local_LLM\gold_research\jobs\watch_master_progress_v1_1_0.py --master-dir C:\Data\Bot\central_backtest_results\master_research_orchestrator_v1_0_1
เวอร์ชัน: v1.1.0

Production purpose:
- ดู progress ของ master orchestrator แบบ real-time
- ดูภาพรวม pipeline ทั้งชุดในจอเดียว
- พยายามอ่าน worker progress ที่เกี่ยวข้องให้มากที่สุด
- ทนต่อกรณี JSON กำลังถูกเขียน / field name ไม่สม่ำเสมอ
- ไม่แก้ผลลัพธ์ต้นทาง เป็น monitor อย่างเดียว

รองรับ:
- master live_progress.json
- phase status แบบ phase_order + phase_status
- phase status แบบ key เริ่มด้วย phase_
- worker progress files ที่อยู่ใต้ master-dir และ subfolders
- worker ที่ชื่อเกี่ยวกับ uncovered / pending / micro / registry / aggregate / feature_cache
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


VERSION = "v1.1.0"
MAX_JSON_PREVIEW = 1800
PROGRESS_BAR_WIDTH = 42
STALE_WARNING_SECONDS = 180


@dataclass
class Snapshot:
    path: Path
    data: Optional[Dict[str, Any]]
    mtime: Optional[datetime]
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch master research orchestrator and worker progress in real time."
    )
    parser.add_argument(
        "--master-dir",
        required=True,
        help="Path to master orchestrator output directory, e.g. C:\\Data\\Bot\\central_backtest_results\\master_research_orchestrator_v1_0_1",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Refresh interval in seconds. Default=3",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one snapshot and exit.",
    )
    parser.add_argument(
        "--max-worker-files",
        type=int,
        default=24,
        help="Maximum worker progress JSON files to inspect. Default=24",
    )
    return parser.parse_args()


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def now_local() -> datetime:
    return datetime.now().astimezone().replace(tzinfo=None)


def file_mtime(path: Path) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except Exception:
        return None


def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
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
        text = safe_read_text(path)
        data = try_parse_json(text)
        if data is not None:
            return Snapshot(path=path, data=data, mtime=file_mtime(path), error=None)

        # กรณีไฟล์กำลังถูกเขียน ลองตัดท้ายทีละนิดเผื่อมี JSON สมบูรณ์อยู่ด้านหน้า
        stripped = text.strip()
        for cut in range(1, min(200, len(stripped)) + 1):
            candidate = stripped[:-cut].rstrip()
            if not candidate:
                break
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return Snapshot(path=path, data=obj, mtime=file_mtime(path), error="partial_json_recovered")
                return Snapshot(path=path, data={"_root": obj}, mtime=file_mtime(path), error="partial_json_recovered")
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


def normalize_text(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


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
            return dt
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


def age_seconds(dt: Optional[datetime], ref: Optional[datetime] = None) -> Optional[float]:
    if dt is None:
        return None
    ref = ref or now_local()
    return max(0.0, (ref - dt).total_seconds())


def build_bar(pct: float, width: int = PROGRESS_BAR_WIDTH) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(round(width * pct / 100.0))
    filled = max(0, min(width, filled))
    return "#" * filled + "-" * (width - filled)


def first_present(data: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return default


def find_numeric_progress(data: Dict[str, Any]) -> float:
    candidates = [
        "overall_progress_pct",
        "progress_pct",
        "pct",
        "percent",
        "progress_percent",
        "completion_pct",
    ]
    for key in candidates:
        if key in data:
            value = safe_float(data.get(key), -1.0)
            if 0.0 <= value <= 100.0:
                return value

    done = safe_float(first_present(data, ["done_jobs", "completed_jobs", "processed_jobs", "finished_jobs"], 0), 0)
    total = safe_float(first_present(data, ["total_jobs", "expected_jobs", "all_jobs", "job_count"], 0), 0)
    if total > 0:
        pct = (done / total) * 100.0
        return max(0.0, min(100.0, pct))

    done_phase = safe_float(first_present(data, ["done_phases", "completed_phases"], 0), 0)
    total_phase = safe_float(first_present(data, ["total_phases"], 0), 0)
    if total_phase > 0:
        pct = (done_phase / total_phase) * 100.0
        return max(0.0, min(100.0, pct))

    return 0.0


def find_started_at(data: Dict[str, Any], mtime: Optional[datetime]) -> Optional[datetime]:
    keys = [
        "started_at",
        "start_time",
        "run_started_at",
        "pipeline_started_at",
        "created_at",
    ]
    dt = parse_ts(first_present(data, keys))
    return dt or mtime


def find_updated_at(data: Dict[str, Any], mtime: Optional[datetime]) -> Optional[datetime]:
    keys = [
        "updated_at",
        "last_update",
        "last_updated_at",
        "heartbeat_at",
        "written_at",
        "ts",
        "timestamp",
    ]
    dt = parse_ts(first_present(data, keys))
    return dt or mtime


def estimate_eta(started_at: Optional[datetime], pct: float, ref_now: Optional[datetime] = None) -> Tuple[Optional[float], Optional[datetime]]:
    ref_now = ref_now or now_local()
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


def infer_phase_table(data: Dict[str, Any]) -> List[Tuple[str, str, bool]]:
    """
    return: [(phase_name, status, is_current)]
    """
    rows: List[Tuple[str, str, bool]] = []
    current_phase = normalize_text(first_present(data, ["current_phase", "running_phase", "phase"], "-"))

    phase_order = data.get("phase_order")
    phase_status = data.get("phase_status")

    if isinstance(phase_order, list) and isinstance(phase_status, dict):
        for phase in phase_order:
            phase_name = normalize_text(phase)
            status = normalize_text(phase_status.get(phase_name, "UNKNOWN")).upper()
            rows.append((phase_name, status, phase_name == current_phase))
        return rows

    phases = data.get("phases")
    if isinstance(phases, list):
        for item in phases:
            if isinstance(item, dict):
                name = normalize_text(first_present(item, ["name", "phase_name", "phase"], "UNKNOWN_PHASE"))
                status = normalize_text(first_present(item, ["status", "state"], "UNKNOWN")).upper()
                is_current = name == current_phase or status == "RUNNING"
                rows.append((name, status, is_current))
        if rows:
            return rows

    phase_keys = [k for k in data.keys() if k.startswith("phase_") and isinstance(data.get(k), (str, int, float))]
    if phase_keys:
        for k in sorted(phase_keys):
            status = normalize_text(data.get(k)).upper()
            rows.append((k, status, k == current_phase))
        return rows

    return rows


def summarize_phase_counts(rows: List[Tuple[str, str, bool]]) -> Tuple[int, int, int, int]:
    done = running = pending = error = 0
    for _, status, _ in rows:
        s = status.upper()
        if any(x in s for x in ["DONE", "SUCCESS", "COMPLETE", "COMPLETED", "FINISHED"]):
            done += 1
        elif any(x in s for x in ["RUNNING", "IN_PROGRESS", "ACTIVE"]):
            running += 1
        elif any(x in s for x in ["ERROR", "FAILED", "FAIL", "CRASH"]):
            error += 1
        else:
            pending += 1
    return done, running, pending, error


def find_worker_candidates(master_dir: Path, max_files: int) -> List[Path]:
    """
    หา progress json ที่เกี่ยวข้องกับ worker ให้มากพอ แต่ไม่กว้างจนช้า
    """
    patterns = [
        "*progress*.json",
        "*live*.json",
        "*status*.json",
        "*state*.json",
        "*summary*.json",
    ]

    candidates: List[Path] = []
    seen = set()

    direct = master_dir / "live_progress.json"
    if direct.exists():
        seen.add(str(direct.resolve()))

    for pattern in patterns:
        for path in master_dir.rglob(pattern):
            try:
                rp = str(path.resolve())
            except Exception:
                rp = str(path)
            if rp in seen:
                continue

            low = path.name.lower()
            text_hit = any(
                token in low
                for token in [
                    "progress",
                    "live",
                    "status",
                    "state",
                    "summary",
                    "uncovered",
                    "pending",
                    "micro",
                    "aggregate",
                    "registry",
                    "feature",
                ]
            )
            if not text_hit:
                continue

            seen.add(rp)
            candidates.append(path)

    def score(p: Path) -> Tuple[int, float]:
        low = str(p).lower()
        priority = 0
        if "live_progress" in low:
            priority -= 100
        if "uncovered" in low:
            priority -= 40
        if "pending" in low:
            priority -= 35
        if "micro" in low:
            priority -= 30
        if "aggregate" in low or "registry" in low:
            priority -= 20
        if "feature" in low:
            priority -= 10
        mt = p.stat().st_mtime if p.exists() else 0.0
        return (priority, -mt)

    candidates.sort(key=score)
    return candidates[:max_files]


def classify_worker_name(path: Path) -> str:
    low = str(path).lower()
    if "uncovered" in low:
        return "uncovered_matrix"
    if "pending" in low:
        return "pending_logic_matrix"
    if "micro" in low:
        return "micro_exit_matrix"
    if "aggregate" in low or "registry" in low:
        return "aggregate_registry"
    if "feature" in low and ("cache" in low or "base" in low):
        return "feature_cache"
    return path.stem


def extract_worker_status(snapshot: Snapshot) -> Dict[str, Any]:
    data = snapshot.data or {}
    started_at = find_started_at(data, snapshot.mtime)
    updated_at = find_updated_at(data, snapshot.mtime)
    progress_pct = find_numeric_progress(data)

    status = normalize_text(first_present(data, ["status", "state"], "UNKNOWN")).upper()
    if status == "UNKNOWN":
        if progress_pct >= 100.0:
            status = "DONE"
        elif progress_pct > 0.0:
            status = "RUNNING"

    total_jobs = safe_int(first_present(data, ["total_jobs", "expected_jobs", "job_count"], 0), 0)
    done_jobs = safe_int(first_present(data, ["done_jobs", "completed_jobs", "processed_jobs", "finished_jobs"], 0), 0)
    error_jobs = safe_int(first_present(data, ["error_jobs", "failed_jobs"], 0), 0)
    running_jobs = safe_int(first_present(data, ["running_jobs", "active_jobs"], 0), 0)
    current_tf = normalize_text(first_present(data, ["current_timeframe", "timeframe", "tf"], "-"))
    current_step = normalize_text(first_present(data, ["current_step", "step", "current_phase"], "-"))

    return {
        "name": classify_worker_name(snapshot.path),
        "path": snapshot.path,
        "status": status,
        "progress_pct": progress_pct,
        "started_at": started_at,
        "updated_at": updated_at,
        "total_jobs": total_jobs,
        "done_jobs": done_jobs,
        "running_jobs": running_jobs,
        "error_jobs": error_jobs,
        "current_tf": current_tf,
        "current_step": current_step,
        "read_error": snapshot.error,
    }


def format_health(updated_at: Optional[datetime], ref_now: datetime) -> str:
    age = age_seconds(updated_at, ref_now)
    if age is None:
        return "NO_TS"
    if age > STALE_WARNING_SECONDS:
        return f"STALE {int(age)}s"
    return f"OK {int(age)}s"


def render_master(snapshot: Snapshot, ref_now: datetime) -> List[str]:
    lines: List[str] = []
    lines.append("=" * 128)
    lines.append(f"MASTER RESEARCH ORCHESTRATOR WATCHER | version={VERSION}")
    lines.append("=" * 128)
    lines.append(f"now                : {fmt_dt(ref_now)}")
    lines.append(f"master_progress    : {snapshot.path}")
    lines.append(f"master_file_status : {'READ_OK' if snapshot.data is not None else 'NOT_READY'}")
    if snapshot.error:
        lines.append(f"master_read_note   : {snapshot.error}")

    if snapshot.data is None:
        lines.append("-" * 128)
        lines.append("status             : waiting for live_progress.json or file currently incomplete")
        lines.append("=" * 128)
        return lines

    data = snapshot.data
    pct = find_numeric_progress(data)
    started_at = find_started_at(data, snapshot.mtime)
    updated_at = find_updated_at(data, snapshot.mtime)
    eta_sec, eta_dt = estimate_eta(started_at, pct, ref_now)
    elapsed = None if started_at is None else (ref_now - started_at).total_seconds()

    current_phase = normalize_text(first_present(data, ["current_phase", "running_phase", "phase"], "-"))
    status = normalize_text(first_present(data, ["status", "state"], "RUNNING")).upper()
    total_jobs = safe_int(first_present(data, ["total_jobs", "expected_jobs", "all_jobs", "job_count"], 0), 0)
    done_jobs = safe_int(first_present(data, ["done_jobs", "completed_jobs", "processed_jobs", "finished_jobs"], 0), 0)
    running_jobs = safe_int(first_present(data, ["running_jobs", "active_jobs"], 0), 0)
    error_jobs = safe_int(first_present(data, ["error_jobs", "failed_jobs"], 0), 0)
    health = format_health(updated_at, ref_now)

    lines.append(f"status             : {status}")
    lines.append(f"current_phase      : {current_phase}")
    lines.append(f"health             : {health}")
    lines.append(f"started_at         : {fmt_dt(started_at)}")
    lines.append(f"updated_at         : {fmt_dt(updated_at)}")
    lines.append(f"elapsed            : {fmt_seconds(elapsed)}")
    lines.append(f"eta_remaining      : {fmt_seconds(eta_sec)}")
    lines.append(f"eta_finish         : {fmt_dt(eta_dt)}")
    lines.append("-" * 128)
    lines.append(f"overall_progress   : [{build_bar(pct)}] {pct:6.2f}%")

    if total_jobs > 0:
        lines.append(
            f"job_counters       : done={done_jobs} running={running_jobs} error={error_jobs} total={total_jobs}"
        )

    phase_rows = infer_phase_table(data)
    done_count, running_count, pending_count, error_count = summarize_phase_counts(phase_rows)
    if phase_rows:
        lines.append(
            f"phase_counters     : done={done_count} running={running_count} pending={pending_count} error={error_count}"
        )
        lines.append("-" * 128)
        lines.append("MASTER PHASES")
        for idx, (name, st, is_current) in enumerate(phase_rows, start=1):
            marker = ">>" if is_current else "  "
            lines.append(f"{marker} {idx:02d}. {name:<52} {st}")
    else:
        lines.append("phase_counters     : phase detail not found")

    return lines


def render_workers(worker_rows: List[Dict[str, Any]], ref_now: datetime) -> List[str]:
    lines: List[str] = []
    lines.append("-" * 128)
    lines.append("WORKER SNAPSHOT")

    if not worker_rows:
        lines.append("  no worker progress files found under master-dir")
        return lines

    header = (
        f"{'worker':<24} {'status':<12} {'progress':<11} {'health':<12} "
        f"{'jobs(done/run/err/total)':<26} {'tf':<8} {'step':<20} path"
    )
    lines.append(header)
    lines.append("-" * 128)

    for row in worker_rows:
        pct = row["progress_pct"]
        health = format_health(row["updated_at"], ref_now)
        jobs_text = f"{row['done_jobs']}/{row['running_jobs']}/{row['error_jobs']}/{row['total_jobs']}"
        lines.append(
            f"{row['name']:<24} "
            f"{row['status']:<12} "
            f"{pct:6.2f}%     "
            f"{health:<12} "
            f"{jobs_text:<26} "
            f"{row['current_tf']:<8} "
            f"{row['current_step'][:20]:<20} "
            f"{row['path']}"
        )
    return lines


def render_summary(worker_rows: List[Dict[str, Any]]) -> List[str]:
    done = running = pending = error = 0
    for row in worker_rows:
        st = row["status"].upper()
        if any(x in st for x in ["DONE", "SUCCESS", "COMPLETED", "FINISHED"]):
            done += 1
        elif any(x in st for x in ["RUNNING", "IN_PROGRESS", "ACTIVE"]):
            running += 1
        elif any(x in st for x in ["ERROR", "FAILED", "FAIL", "CRASH"]):
            error += 1
        else:
            pending += 1

    lines = []
    lines.append("-" * 128)
    lines.append(
        f"WORKER COUNTERS    : done={done} running={running} pending={pending} error={error}"
    )
    return lines


def render_raw_preview(snapshot: Snapshot) -> List[str]:
    lines: List[str] = []
    if snapshot.data is None:
        return lines

    lines.append("-" * 128)
    lines.append("MASTER RAW PREVIEW")
    try:
        preview = json.dumps(snapshot.data, ensure_ascii=False, indent=2)
    except Exception:
        preview = str(snapshot.data)

    if len(preview) > MAX_JSON_PREVIEW:
        preview = preview[:MAX_JSON_PREVIEW] + "\n... (truncated)"
    lines.append(preview)
    return lines


def collect_worker_rows(master_dir: Path, max_worker_files: int) -> List[Dict[str, Any]]:
    files = find_worker_candidates(master_dir, max_worker_files)
    rows: List[Dict[str, Any]] = []

    for path in files:
        if path.name == "live_progress.json" and path.parent == master_dir:
            continue
        snap = safe_read_json(path)
        if snap.data is None and snap.error == "missing":
            continue
        row = extract_worker_status(snap)
        rows.append(row)

    def sort_key(row: Dict[str, Any]) -> Tuple[int, float]:
        name = row["name"]
        priority = 100
        if name == "feature_cache":
            priority = 1
        elif name == "uncovered_matrix":
            priority = 2
        elif name == "pending_logic_matrix":
            priority = 3
        elif name == "micro_exit_matrix":
            priority = 4
        elif name == "aggregate_registry":
            priority = 5

        upd = row["updated_at"]
        stamp = upd.timestamp() if upd else 0.0
        return (priority, -stamp)

    rows.sort(key=sort_key)
    return rows


def render_screen(master_snapshot: Snapshot, worker_rows: List[Dict[str, Any]], ref_now: datetime) -> str:
    parts: List[str] = []
    parts.extend(render_master(master_snapshot, ref_now))
    parts.extend(render_summary(worker_rows))
    parts.extend(render_workers(worker_rows, ref_now))
    parts.extend(render_raw_preview(master_snapshot))
    parts.append("=" * 128)
    parts.append("Ctrl+C = stop watcher")
    parts.append("=" * 128)
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    master_dir = Path(args.master_dir)
    master_progress_path = master_dir / "live_progress.json"

    while True:
        ref_now = now_local()
        master_snapshot = safe_read_json(master_progress_path)
        worker_rows = collect_worker_rows(master_dir, args.max_worker_files)

        clear_screen()
        print(render_screen(master_snapshot, worker_rows, ref_now))

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
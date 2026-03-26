#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ชื่อโค้ด: watch_master_progress_v1_3_0.py
ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\watch_master_progress_v1_3_0.py
คำสั่งรัน:
    python C:\Data\Bot\Local_LLM\gold_research\jobs\watch_master_progress_v1_3_0.py --master-dir C:\Data\Bot\central_backtest_results\master_research_orchestrator_v1_0_1 --interval 2
เวอร์ชัน: v1.3.0

หน้าที่:
- ดู master progress แบบ real-time
- ถ้า live_progress.json stale ให้ fallback ไปอ่าน phase log ล่าสุด
- พยายามดึง progress/counters/ETA จาก log โดยอัตโนมัติ
- แสดงว่า source ที่เชื่อถือได้ตอนนี้คืออะไร
- ไม่แก้ไฟล์ใด ๆ เป็น monitor only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


VERSION = "v1.3.0"
DEFAULT_INTERVAL = 2.0
DEFAULT_STALE_SECONDS = 180
PROGRESS_BAR_WIDTH = 44
TAIL_LINE_COUNT = 120
MAX_PREVIEW_LINES = 20
MAX_WORKER_FILES = 32


@dataclass
class Snapshot:
    path: Path
    data: Optional[Dict[str, Any]]
    mtime: Optional[datetime]
    error: Optional[str] = None


@dataclass
class LogProbe:
    path: Optional[Path]
    exists: bool
    mtime: Optional[datetime]
    size_bytes: int
    total_lines: int
    added_lines_since_last: int
    added_bytes_since_last: int
    parsed_pct: Optional[float]
    parsed_done: Optional[int]
    parsed_total: Optional[int]
    parsed_running: Optional[int]
    parsed_error: Optional[int]
    parsed_eta_seconds: Optional[float]
    detected_timeframe: Optional[str]
    detected_step: Optional[str]
    last_status_line: Optional[str]
    tail_lines: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch master orchestrator progress with log fallback."
    )
    parser.add_argument(
        "--master-dir",
        required=True,
        help="Path to master orchestrator output directory",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help="Refresh interval in seconds",
    )
    parser.add_argument(
        "--stale-seconds",
        type=int,
        default=DEFAULT_STALE_SECONDS,
        help="Heartbeat stale threshold in seconds",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print once and exit",
    )
    return parser.parse_args()


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def now_local() -> datetime:
    return datetime.now()


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


def first_present(data: Dict[str, Any], keys: Tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return default


def file_mtime(path: Path) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except Exception:
        return None


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


def age_seconds(dt: Optional[datetime], ref_now: datetime) -> Optional[float]:
    if dt is None:
        return None
    return max(0.0, (ref_now - dt).total_seconds())


def classify_health(updated_at: Optional[datetime], ref_now: datetime, stale_seconds: int) -> str:
    age = age_seconds(updated_at, ref_now)
    if age is None:
        return "NO_TS"
    if age > stale_seconds:
        return f"STALE {int(age)}s"
    return f"OK {int(age)}s"


def build_bar(pct: float, width: int = PROGRESS_BAR_WIDTH) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(round(width * pct / 100.0))
    filled = max(0, min(width, filled))
    return "#" * filled + "-" * (width - filled)


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
                    return Snapshot(path=path, data=obj, mtime=file_mtime(path), error="partial_json_recovered")
                return Snapshot(path=path, data={"_root": obj}, mtime=file_mtime(path), error="partial_json_recovered")
            except Exception:
                continue

        return Snapshot(path=path, data=None, mtime=file_mtime(path), error="invalid_json")
    except Exception as exc:
        return Snapshot(path=path, data=None, mtime=file_mtime(path), error=f"read_error: {exc}")


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


def estimate_eta_from_master(data: Dict[str, Any], started_at: Optional[datetime], pct: float, ref_now: datetime) -> Tuple[Optional[float], Optional[datetime]]:
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
        for name, status in phase_statuses.items():
            phase_name = textify(name)
            phase_status = textify(status).upper()
            rows.append((phase_name, phase_status, phase_name == current_phase or phase_status == "RUNNING"))
        rows.sort(key=lambda x: x[0])
        return rows

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


def extract_worker_row(snapshot: Snapshot) -> Dict[str, Any]:
    data = snapshot.data or {}
    pct = find_progress_pct(data)
    status = textify(first_present(data, ("status", "state")), "UNKNOWN").upper()

    if status == "UNKNOWN":
        if pct >= 100.0:
            status = "DONE"
        elif pct > 0.0:
            status = "RUNNING"

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
        "updated_at": find_updated_at(data, snapshot.mtime),
        "path": str(snapshot.path),
        "note": snapshot.error or "-",
    }


def collect_worker_rows(master_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in find_worker_files(master_dir):
        snap = safe_read_json(path)
        if snap.data is None and snap.error == "missing":
            continue
        rows.append(extract_worker_row(snap))
    return rows


def newest_phase_log(master_dir: Path) -> Optional[Path]:
    phase_log_dir = master_dir / "logs" / "phase_logs"
    if not phase_log_dir.exists():
        return None

    files = [p for p in phase_log_dir.glob("*.log") if p.is_file()]
    if not files:
        return None

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def read_tail_lines(path: Path, max_lines: int = TAIL_LINE_COUNT) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return lines
    return lines[-max_lines:]


def parse_eta_seconds_from_text(text: str) -> Optional[float]:
    m = re.search(r'ETA\s*[:=]\s*(\d+):(\d{2}):(\d{2})', text, re.IGNORECASE)
    if m:
        h, mi, s = map(int, m.groups())
        return h * 3600 + mi * 60 + s

    m = re.search(r'eta(?:_remaining)?(?:_sec|_seconds)?\s*[:=]\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if m:
        return float(m.group(1))

    m = re.search(r'overall_eta_remaining_min\s*[:=]\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 60.0

    return None


def probe_phase_log(log_path: Optional[Path], previous_meta: Optional[Dict[str, int]]) -> LogProbe:
    if log_path is None or not log_path.exists():
        return LogProbe(
            path=log_path,
            exists=False,
            mtime=None,
            size_bytes=0,
            total_lines=0,
            added_lines_since_last=0,
            added_bytes_since_last=0,
            parsed_pct=None,
            parsed_done=None,
            parsed_total=None,
            parsed_running=None,
            parsed_error=None,
            parsed_eta_seconds=None,
            detected_timeframe=None,
            detected_step=None,
            last_status_line=None,
            tail_lines=[],
        )

    stat = log_path.stat()
    size_bytes = int(stat.st_size)
    mtime = datetime.fromtimestamp(stat.st_mtime)
    tail_lines = read_tail_lines(log_path, TAIL_LINE_COUNT)
    total_lines = 0
    try:
        total_lines = sum(1 for _ in log_path.open("r", encoding="utf-8", errors="replace"))
    except Exception:
        total_lines = len(tail_lines)

    prev_size = 0
    prev_lines = 0
    if previous_meta and previous_meta.get("path") == str(log_path):
        prev_size = int(previous_meta.get("size_bytes", 0))
        prev_lines = int(previous_meta.get("total_lines", 0))

    added_bytes = max(0, size_bytes - prev_size)
    added_lines = max(0, total_lines - prev_lines)

    parsed_pct = None
    parsed_done = None
    parsed_total = None
    parsed_running = None
    parsed_error = None
    parsed_eta_seconds = None
    detected_timeframe = None
    detected_step = None
    last_status_line = None

    patterns_pct = [
        re.compile(r'(\d+(?:\.\d+)?)\s*%'),
        re.compile(r'progress(?:_pct|_percent)?\s*[:=]\s*(\d+(?:\.\d+)?)', re.IGNORECASE),
    ]
    patterns_done_total = [
        re.compile(r'(?:done|processed|completed|finished|progress)\s*[:=]\s*(\d+)\s*/\s*(\d+)', re.IGNORECASE),
        re.compile(r'(\d+)\s*/\s*(\d+)\s*(?:done|processed|completed|finished)', re.IGNORECASE),
        re.compile(r'(?:job|jobs|task|tasks|shard|shards)\s*[:=]?\s*(\d+)\s*/\s*(\d+)', re.IGNORECASE),
        re.compile(r'(?:done|processed|completed|finished)_jobs\s*[:=]\s*(\d+).*?(?:total|expected|all)_jobs\s*[:=]\s*(\d+)', re.IGNORECASE),
    ]
    patterns_running = [
        re.compile(r'(?:running|active)_jobs\s*[:=]\s*(\d+)', re.IGNORECASE),
    ]
    patterns_error = [
        re.compile(r'(?:error|failed)_jobs\s*[:=]\s*(\d+)', re.IGNORECASE),
        re.compile(r'errors?\s*[:=]\s*(\d+)', re.IGNORECASE),
    ]
    patterns_tf = [
        re.compile(r'\b(M1|M2|M3|M4|M5|M6|M10|M15|M30|H1|H4|D1)\b'),
    ]

    for line in reversed(tail_lines):
        stripped = line.strip()
        if not stripped:
            continue

        if last_status_line is None:
            last_status_line = stripped

        if parsed_pct is None:
            for pat in patterns_pct:
                m = pat.search(stripped)
                if m:
                    value = safe_float(m.group(1), -1.0)
                    if 0.0 <= value <= 100.0:
                        parsed_pct = value
                        break

        if parsed_done is None or parsed_total is None:
            for pat in patterns_done_total:
                m = pat.search(stripped)
                if m:
                    d = safe_int(m.group(1), -1)
                    t = safe_int(m.group(2), -1)
                    if d >= 0 and t > 0:
                        parsed_done = d
                        parsed_total = t
                        if parsed_pct is None:
                            parsed_pct = max(0.0, min(100.0, (d / t) * 100.0))
                        break

        if parsed_running is None:
            for pat in patterns_running:
                m = pat.search(stripped)
                if m:
                    parsed_running = safe_int(m.group(1), 0)
                    break

        if parsed_error is None:
            for pat in patterns_error:
                m = pat.search(stripped)
                if m:
                    parsed_error = safe_int(m.group(1), 0)
                    break

        if parsed_eta_seconds is None:
            parsed_eta_seconds = parse_eta_seconds_from_text(stripped)

        if detected_timeframe is None:
            for pat in patterns_tf:
                m = pat.search(stripped)
                if m:
                    detected_timeframe = m.group(1)
                    break

        if detected_step is None:
            if "uncovered" in stripped.lower():
                detected_step = "uncovered_matrix"
            elif "pending" in stripped.lower():
                detected_step = "pending_logic_matrix"
            elif "micro" in stripped.lower():
                detected_step = "micro_exit_matrix"
            elif "aggregate" in stripped.lower() or "registry" in stripped.lower():
                detected_step = "registry_aggregation"

        if all(
            value is not None
            for value in [parsed_pct, parsed_done, parsed_total, parsed_eta_seconds, detected_timeframe, detected_step]
        ):
            break

    return LogProbe(
        path=log_path,
        exists=True,
        mtime=mtime,
        size_bytes=size_bytes,
        total_lines=total_lines,
        added_lines_since_last=added_lines,
        added_bytes_since_last=added_bytes,
        parsed_pct=parsed_pct,
        parsed_done=parsed_done,
        parsed_total=parsed_total,
        parsed_running=parsed_running,
        parsed_error=parsed_error,
        parsed_eta_seconds=parsed_eta_seconds,
        detected_timeframe=detected_timeframe,
        detected_step=detected_step,
        last_status_line=last_status_line,
        tail_lines=tail_lines[-MAX_PREVIEW_LINES:],
    )


def choose_truth_source(master_snapshot: Snapshot, log_probe: LogProbe, ref_now: datetime, stale_seconds: int) -> str:
    if master_snapshot.data is not None:
        updated_at = find_updated_at(master_snapshot.data, master_snapshot.mtime)
        if updated_at is not None and age_seconds(updated_at, ref_now) is not None and age_seconds(updated_at, ref_now) <= stale_seconds:
            return "master_live_progress"
    if log_probe.exists and log_probe.mtime is not None and age_seconds(log_probe.mtime, ref_now) is not None and age_seconds(log_probe.mtime, ref_now) <= stale_seconds:
        return "phase_log_fallback"
    return "stale_sources_only"


def render_master(snapshot: Snapshot, log_probe: LogProbe, ref_now: datetime, stale_seconds: int) -> List[str]:
    lines: List[str] = []
    lines.append("=" * 144)
    lines.append(f"MASTER RESEARCH ORCHESTRATOR WATCHER | version={VERSION}")
    lines.append("=" * 144)

    lines.append(f"now                  : {fmt_dt(ref_now)}")
    lines.append(f"master_progress      : {snapshot.path}")
    lines.append(f"master_file_status   : {'READ_OK' if snapshot.data is not None else 'NOT_READY'}")
    if snapshot.error:
        lines.append(f"master_read_note     : {snapshot.error}")

    truth_source = choose_truth_source(snapshot, log_probe, ref_now, stale_seconds)
    lines.append(f"truth_source         : {truth_source}")

    if snapshot.data is None:
        lines.append("status               : waiting for live_progress.json")
        lines.append("=" * 144)
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

    eta_seconds, eta_finish = estimate_eta_from_master(data, started_at, pct, ref_now)
    health = classify_health(updated_at, ref_now, stale_seconds)

    phases_total = safe_int(first_present(data, ("phases_total", "total_phases")), 0)
    phases_done = safe_int(first_present(data, ("phases_done", "done_phases")), 0)
    phases_running = safe_int(first_present(data, ("phases_running", "running_phases")), 0)
    phases_failed = safe_int(first_present(data, ("phases_failed", "failed_phases")), 0)

    lines.append(f"status               : {status}")
    lines.append(f"current_phase        : {current_phase}")
    lines.append(f"health               : {health}")
    lines.append(f"started_at           : {fmt_dt(started_at)}")
    lines.append(f"updated_at           : {fmt_dt(updated_at)}")
    lines.append(f"elapsed(master)      : {fmt_seconds(elapsed_seconds)}")
    lines.append(f"eta_remaining(master): {fmt_seconds(eta_seconds)}")
    lines.append(f"eta_finish(master)   : {fmt_dt(eta_finish)}")
    lines.append("-" * 144)
    lines.append(f"overall_progress     : [{build_bar(pct)}] {pct:6.2f}%")
    if phases_total > 0:
        lines.append(
            f"phase_summary        : done={phases_done} running={phases_running} failed={phases_failed} total={phases_total}"
        )

    phase_rows = infer_phase_rows(data)
    done_count, running_count, pending_count, error_count = summarize_phase_counts(phase_rows)
    lines.append(
        f"phase_counters       : done={done_count} running={running_count} pending={pending_count} error={error_count}"
    )

    if age_seconds(updated_at, ref_now) is not None and age_seconds(updated_at, ref_now) > stale_seconds:
        lines.append("-" * 144)
        lines.append("ALERT                : MASTER PROGRESS FILE FROZEN")
        lines.append("diagnostic           : worker ยังอาจรันต่อ แต่ live_progress.json ไม่อัปเดต")
        lines.append("impact               : master progress/ETA/elapsed ไม่ใช่สถานะจริงปัจจุบัน")

    lines.append("-" * 144)
    lines.append("MASTER PHASES")
    if phase_rows:
        for idx, (name, st, is_current) in enumerate(phase_rows, start=1):
            marker = ">>" if is_current else "  "
            lines.append(f"{marker} {idx:02d}. {name:<56} {st}")
    else:
        lines.append("  phase detail not found")

    return lines


def render_log_fallback(log_probe: LogProbe, ref_now: datetime, stale_seconds: int) -> List[str]:
    lines: List[str] = []
    lines.append("-" * 144)
    lines.append("PHASE LOG FALLBACK")

    if not log_probe.exists or log_probe.path is None:
        lines.append("log_status            : phase log not found")
        return lines

    lines.append(f"log_path              : {log_probe.path}")
    lines.append(f"log_updated_at        : {fmt_dt(log_probe.mtime)}")
    lines.append(f"log_health            : {classify_health(log_probe.mtime, ref_now, stale_seconds)}")
    lines.append(f"log_size_bytes        : {log_probe.size_bytes}")
    lines.append(f"log_total_lines       : {log_probe.total_lines}")
    lines.append(f"log_added_lines       : {log_probe.added_lines_since_last}")
    lines.append(f"log_added_bytes       : {log_probe.added_bytes_since_last}")

    if log_probe.parsed_pct is not None:
        lines.append(f"log_progress_pct      : {log_probe.parsed_pct:6.2f}%")
    else:
        lines.append("log_progress_pct      : -")

    if log_probe.parsed_done is not None or log_probe.parsed_total is not None:
        lines.append(
            f"log_jobs              : done={textify(log_probe.parsed_done)} running={textify(log_probe.parsed_running)} "
            f"error={textify(log_probe.parsed_error)} total={textify(log_probe.parsed_total)}"
        )
    else:
        lines.append(
            f"log_jobs              : done=- running={textify(log_probe.parsed_running)} error={textify(log_probe.parsed_error)} total=-"
        )

    lines.append(f"log_timeframe         : {textify(log_probe.detected_timeframe)}")
    lines.append(f"log_detected_step     : {textify(log_probe.detected_step)}")

    if log_probe.parsed_eta_seconds is not None:
        lines.append(f"log_eta_remaining     : {fmt_seconds(log_probe.parsed_eta_seconds)}")
        lines.append(f"log_eta_finish        : {fmt_dt(ref_now + timedelta(seconds=log_probe.parsed_eta_seconds))}")
    else:
        lines.append("log_eta_remaining     : -")
        lines.append("log_eta_finish        : -")

    lines.append(f"log_last_status_line  : {textify(log_probe.last_status_line)}")
    lines.append("-" * 144)
    lines.append("LOG TAIL PREVIEW")
    if log_probe.tail_lines:
        for line in log_probe.tail_lines[-MAX_PREVIEW_LINES:]:
            lines.append(line[-220:])
    else:
        lines.append("  no lines")
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

    lines.append("-" * 144)
    lines.append(f"WORKER COUNTERS       : done={done} running={running} pending={pending} error={error}")
    lines.append("-" * 144)
    lines.append("WORKER SNAPSHOT")

    if not worker_rows:
        lines.append("  no worker progress/state files found")
        return lines

    header = (
        f"{'worker':<24} {'status':<14} {'progress':<11} {'health':<14} "
        f"{'jobs(d/r/e/t)':<16} {'tf':<8} {'step':<20} path"
    )
    lines.append(header)
    lines.append("-" * 144)

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


def render_screen(master_snapshot: Snapshot, worker_rows: List[Dict[str, Any]], log_probe: LogProbe, ref_now: datetime, stale_seconds: int) -> str:
    parts: List[str] = []
    parts.extend(render_master(master_snapshot, log_probe, ref_now, stale_seconds))
    parts.extend(render_log_fallback(log_probe, ref_now, stale_seconds))
    parts.extend(render_workers(worker_rows, ref_now, stale_seconds))
    parts.append("=" * 144)
    parts.append("Ctrl+C = stop watcher")
    parts.append("=" * 144)
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    master_dir = Path(args.master_dir)
    master_progress = master_dir / "live_progress.json"

    previous_log_meta: Dict[str, int] = {}

    while True:
        ref_now = now_local()
        master_snapshot = safe_read_json(master_progress)
        worker_rows = collect_worker_rows(master_dir)

        latest_log = newest_phase_log(master_dir)
        log_probe = probe_phase_log(latest_log, previous_log_meta if previous_log_meta else None)

        if log_probe.path is not None:
            previous_log_meta = {
                "path": str(log_probe.path),
                "size_bytes": log_probe.size_bytes,
                "total_lines": log_probe.total_lines,
            }

        clear_screen()
        print(render_screen(master_snapshot, worker_rows, log_probe, ref_now, args.stale_seconds))

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
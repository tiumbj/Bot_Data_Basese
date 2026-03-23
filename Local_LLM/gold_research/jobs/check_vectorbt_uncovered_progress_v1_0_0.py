#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
name: check_vectorbt_uncovered_progress_v1_0_0.py
path: C:\Data\Bot\Local_LLM\gold_research\jobs\check_vectorbt_uncovered_progress_v1_0_0.py
version: v1.0.0

Purpose:
- Check progress of run_vectorbt_uncovered_parallel_backtests outputs.
- Read per-timeframe state JSON files, result CSV files, summary/manifest if present.
- Print overall progress %, per-timeframe progress, current active timeframe, and ETA hints
  based on observed elapsed time from file timestamps when possible.

Notes:
- Safe to run repeatedly while the main backtest is running.
- Does not modify any result files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_OUTDIR = r"C:\Data\Bot\central_backtest_results\vectorbt_uncovered_parallel_v1_1_2"
DEFAULT_TIMEFRAMES = ["M2", "M3", "M4", "M6", "M30", "D1"]


@dataclass
class TfProgress:
    timeframe: str
    state_path: str
    result_csv_path: str
    state_exists: bool
    result_exists: bool
    jobs_total: int
    jobs_completed: int
    jobs_remaining: int
    progress_pct: float
    status: str
    last_completed_job_id: str
    result_rows: int
    state_mtime_utc: str
    csv_mtime_utc: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_utc_iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def safe_read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            row_count = -1  # exclude header if present
            for row_count, _ in enumerate(reader):
                pass
            if row_count < 0:
                return 0
            # row_count is zero-based count including header if file has rows
            # subtract one header row when file is non-empty
            return max(0, row_count)
    except Exception:
        return 0


def detect_status(jobs_total: int, jobs_completed: int, state_exists: bool, result_exists: bool) -> str:
    if jobs_total > 0 and jobs_completed >= jobs_total:
        return "DONE"
    if jobs_completed > 0:
        return "RUNNING"
    if state_exists or result_exists:
        return "STARTED"
    return "PENDING"


def load_tf_progress(outdir: Path, timeframe: str) -> TfProgress:
    state_path = outdir / "state" / f"XAUUSD_{timeframe}_state.json"
    result_csv_path = outdir / "per_timeframe" / f"XAUUSD_{timeframe}_vectorbt_uncovered_results.csv"

    state_exists = state_path.exists()
    result_exists = result_csv_path.exists()

    jobs_total = 0
    jobs_completed = 0
    jobs_remaining = 0
    progress_pct = 0.0
    last_completed_job_id = ""

    state_json = safe_read_json(state_path) if state_exists else None
    if isinstance(state_json, dict):
        jobs_total = int(state_json.get("jobs_total", 0) or 0)
        jobs_completed = int(state_json.get("completed_jobs", 0) or 0)
        jobs_remaining = int(state_json.get("remaining_jobs", max(0, jobs_total - jobs_completed)) or 0)
        progress_pct = float(state_json.get("progress_pct", 0.0) or 0.0)
        last_completed_job_id = str(state_json.get("last_completed_job_id", "") or "")

    result_rows = count_csv_rows(result_csv_path)

    # Fallback if state doesn't exist or has zeros but CSV exists
    if jobs_completed == 0 and result_rows > 0:
        jobs_completed = result_rows
        if jobs_total > 0:
            jobs_remaining = max(0, jobs_total - jobs_completed)
            progress_pct = round((jobs_completed / jobs_total) * 100.0, 6)

    status = detect_status(jobs_total, jobs_completed, state_exists, result_exists)

    state_mtime_utc = to_utc_iso_from_ts(state_path.stat().st_mtime) if state_exists else ""
    csv_mtime_utc = to_utc_iso_from_ts(result_csv_path.stat().st_mtime) if result_exists else ""

    return TfProgress(
        timeframe=timeframe,
        state_path=str(state_path),
        result_csv_path=str(result_csv_path),
        state_exists=state_exists,
        result_exists=result_exists,
        jobs_total=jobs_total,
        jobs_completed=jobs_completed,
        jobs_remaining=jobs_remaining,
        progress_pct=round(progress_pct, 6),
        status=status,
        last_completed_job_id=last_completed_job_id,
        result_rows=result_rows,
        state_mtime_utc=state_mtime_utc,
        csv_mtime_utc=csv_mtime_utc,
    )


def infer_current_timeframe(tf_rows: List[TfProgress]) -> str:
    running = [r for r in tf_rows if r.status == "RUNNING"]
    if running:
        running.sort(key=lambda x: (x.state_mtime_utc or "", x.csv_mtime_utc or ""), reverse=True)
        return running[0].timeframe
    started = [r for r in tf_rows if r.status == "STARTED"]
    if started:
        started.sort(key=lambda x: (x.state_mtime_utc or "", x.csv_mtime_utc or ""), reverse=True)
        return started[0].timeframe
    done = [r for r in tf_rows if r.status == "DONE"]
    if done:
        done.sort(key=lambda x: x.timeframe)
        return done[-1].timeframe
    return ""


def format_hours(hours: Optional[float]) -> str:
    if hours is None or math.isinf(hours) or math.isnan(hours):
        return ""
    if hours < 1:
        minutes = hours * 60.0
        return f"{minutes:.1f} min"
    return f"{hours:.2f} hr"


def infer_eta(outdir: Path, tf_rows: List[TfProgress]) -> Tuple[str, str]:
    """
    Rough ETA based on first existing file mtime in outdir and current completion count.
    This is approximate only.
    Returns (elapsed_str, eta_str)
    """
    existing_paths: List[Path] = []
    for r in tf_rows:
        sp = Path(r.state_path)
        cp = Path(r.result_csv_path)
        if sp.exists():
            existing_paths.append(sp)
        if cp.exists():
            existing_paths.append(cp)

    if not existing_paths:
        return "", ""

    first_ts = min(p.stat().st_mtime for p in existing_paths)
    last_ts = max(p.stat().st_mtime for p in existing_paths)
    now_ts = datetime.now().timestamp()

    elapsed_sec = max(0.0, now_ts - first_ts)
    overall_total = sum(r.jobs_total for r in tf_rows if r.jobs_total > 0)
    overall_done = sum(r.jobs_completed for r in tf_rows)

    if overall_done <= 0 or overall_total <= 0:
        return format_hours(elapsed_sec / 3600.0), ""

    sec_per_job = elapsed_sec / overall_done
    remaining_jobs = max(0, overall_total - overall_done)
    eta_sec = remaining_jobs * sec_per_job
    return format_hours(elapsed_sec / 3600.0), format_hours(eta_sec / 3600.0)


def print_console_report(outdir: Path, tf_rows: List[TfProgress]) -> None:
    overall_total = sum(r.jobs_total for r in tf_rows if r.jobs_total > 0)
    overall_done = sum(r.jobs_completed for r in tf_rows)
    overall_remaining = max(0, overall_total - overall_done)
    overall_pct = (overall_done / overall_total * 100.0) if overall_total > 0 else 0.0
    current_tf = infer_current_timeframe(tf_rows)
    elapsed_str, eta_str = infer_eta(outdir, tf_rows)

    print("=" * 120)
    print("[DONE] version=v1.0.0")
    print(f"[DONE] checked_at_utc={utc_now_iso()}")
    print(f"[DONE] outdir={outdir}")
    print(f"[DONE] current_timeframe={current_tf or 'UNKNOWN'}")
    print(f"[DONE] overall_jobs_total={overall_total}")
    print(f"[DONE] overall_jobs_completed={overall_done}")
    print(f"[DONE] overall_jobs_remaining={overall_remaining}")
    print(f"[DONE] overall_progress_pct={overall_pct:.6f}")
    if elapsed_str:
        print(f"[DONE] observed_elapsed={elapsed_str}")
    if eta_str:
        print(f"[DONE] rough_eta_remaining={eta_str}")
    print("-" * 120)

    header = (
        f"{'TF':<6} {'STATUS':<10} {'DONE':>8} {'TOTAL':>8} {'LEFT':>8} "
        f"{'%':>10} {'ROWS':>8} {'LAST_JOB_ID':<48}"
    )
    print(header)
    print("-" * 120)
    for r in tf_rows:
        print(
            f"{r.timeframe:<6} {r.status:<10} {r.jobs_completed:>8} {r.jobs_total:>8} "
            f"{r.jobs_remaining:>8} {r.progress_pct:>10.4f} {r.result_rows:>8} "
            f"{r.last_completed_job_id[:48]:<48}"
        )
    print("=" * 120)


def write_json_report(outdir: Path, tf_rows: List[TfProgress]) -> Path:
    overall_total = sum(r.jobs_total for r in tf_rows if r.jobs_total > 0)
    overall_done = sum(r.jobs_completed for r in tf_rows)
    overall_remaining = max(0, overall_total - overall_done)
    overall_pct = (overall_done / overall_total * 100.0) if overall_total > 0 else 0.0
    current_tf = infer_current_timeframe(tf_rows)
    elapsed_str, eta_str = infer_eta(outdir, tf_rows)

    report = {
        "version": "v1.0.0",
        "checked_at_utc": utc_now_iso(),
        "outdir": str(outdir),
        "current_timeframe": current_tf,
        "overall_jobs_total": overall_total,
        "overall_jobs_completed": overall_done,
        "overall_jobs_remaining": overall_remaining,
        "overall_progress_pct": round(overall_pct, 6),
        "observed_elapsed": elapsed_str,
        "rough_eta_remaining": eta_str,
        "timeframes": [asdict(r) for r in tf_rows],
    }

    out_path = outdir / "progress_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check progress of uncovered VectorBT backtests.")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output directory of the VectorBT runner.")
    parser.add_argument(
        "--timeframes",
        nargs="*",
        default=DEFAULT_TIMEFRAMES,
        help="Timeframes to inspect. Default: M2 M3 M4 M6 M30 D1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)

    tf_rows = [load_tf_progress(outdir=outdir, timeframe=tf) for tf in args.timeframes]
    print_console_report(outdir=outdir, tf_rows=tf_rows)
    report_path = write_json_report(outdir=outdir, tf_rows=tf_rows)
    print(f"[DONE] progress_report_json={report_path}")


if __name__ == "__main__":
    main()

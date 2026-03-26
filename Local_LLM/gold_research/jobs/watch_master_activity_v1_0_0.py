#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ชื่อโค้ด: watch_master_activity_v1_0_0.py
ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\watch_master_activity_v1_0_0.py
คำสั่งรัน:
    python C:\Data\Bot\Local_LLM\gold_research\jobs\watch_master_activity_v1_0_0.py --root C:\Data\Bot\central_backtest_results\master_research_orchestrator_v1_0_1
เวอร์ชัน: v1.0.0

หน้าที่:
- ดู activity จริงของไฟล์ทั้งหมดใต้ root แบบ real-time
- บอกว่าไฟล์ไหนเพิ่งถูกแก้ล่าสุด
- บอกว่า tree นี้ยังมีการเขียนไฟล์อยู่หรือไม่
- ใช้แยกปัญหา:
  1) progress layer ค้าง แต่ worker ยังวิ่ง
  2) ทั้ง pipeline ค้างจริง

หมายเหตุ:
- monitor only
- ไม่แก้ไฟล์ใด ๆ
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


VERSION = "v1.0.0"
DEFAULT_INTERVAL = 3.0
DEFAULT_TOP = 25
DEFAULT_STALE_SECONDS = 180


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch real file activity under master output tree.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory to watch",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help="Refresh interval in seconds (default: 3)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help="Number of newest files to show (default: 25)",
    )
    parser.add_argument(
        "--stale-seconds",
        type=int,
        default=DEFAULT_STALE_SECONDS,
        help="Age in seconds after which activity is considered stale (default: 180)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print once and exit",
    )
    return parser.parse_args()


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def fmt_dt(dt: datetime | None) -> str:
    if dt is None:
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
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


def safe_stat(path: Path) -> Tuple[float, int] | None:
    try:
        st = path.stat()
        return st.st_mtime, st.st_size
    except Exception:
        return None


def collect_files(root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        stat = safe_stat(path)
        if stat is None:
            continue
        mtime, size = stat
        rows.append(
            {
                "path": path,
                "mtime_ts": float(mtime),
                "mtime_dt": datetime.fromtimestamp(mtime),
                "size": int(size),
            }
        )
    rows.sort(key=lambda x: x["mtime_ts"], reverse=True)
    return rows


def classify_activity(age_sec: float, stale_seconds: int) -> str:
    if age_sec <= 10:
        return "LIVE"
    if age_sec <= stale_seconds:
        return "ACTIVE"
    return "STALE"


def detect_changes(
    current_rows: List[Dict],
    previous_map: Dict[str, Tuple[float, int]],
) -> Tuple[int, int, Dict[str, Tuple[float, int]]]:
    changed = 0
    new_files = 0
    next_map: Dict[str, Tuple[float, int]] = {}

    for row in current_rows:
        key = str(row["path"])
        value = (row["mtime_ts"], row["size"])
        next_map[key] = value

        if key not in previous_map:
            new_files += 1
            continue

        old_mtime, old_size = previous_map[key]
        if row["mtime_ts"] != old_mtime or row["size"] != old_size:
            changed += 1

    return changed, new_files, next_map


def render(
    root: Path,
    rows: List[Dict],
    now_dt: datetime,
    stale_seconds: int,
    changed_count: int,
    new_files_count: int,
    top_n: int,
) -> str:
    lines: List[str] = []
    lines.append("=" * 150)
    lines.append(f"MASTER FILE ACTIVITY WATCHER | version={VERSION}")
    lines.append("=" * 150)
    lines.append(f"now                  : {fmt_dt(now_dt)}")
    lines.append(f"root                 : {root}")
    lines.append(f"total_files          : {len(rows)}")
    lines.append(f"changed_since_last   : {changed_count}")
    lines.append(f"new_files_since_last : {new_files_count}")

    if rows:
        newest = rows[0]
        newest_age = max(0.0, (now_dt - newest["mtime_dt"]).total_seconds())
        lines.append(f"latest_file_time     : {fmt_dt(newest['mtime_dt'])}")
        lines.append(f"latest_file_age      : {fmt_seconds(newest_age)}")
        lines.append(f"tree_activity        : {classify_activity(newest_age, stale_seconds)}")
    else:
        lines.append("latest_file_time     : -")
        lines.append("latest_file_age      : -")
        lines.append("tree_activity        : EMPTY")

    lines.append("-" * 150)
    lines.append(f"{'status':<8} {'age':<12} {'mtime':<19} {'size(bytes)':<12} path")
    lines.append("-" * 150)

    for row in rows[:top_n]:
        age_sec = max(0.0, (now_dt - row["mtime_dt"]).total_seconds())
        status = classify_activity(age_sec, stale_seconds)
        lines.append(
            f"{status:<8} {fmt_seconds(age_sec):<12} {fmt_dt(row['mtime_dt']):<19} {row['size']:<12} {row['path']}"
        )

    lines.append("-" * 150)
    lines.append("การแปลผล:")
    lines.append("LIVE/ACTIVE = ยังมีไฟล์ใน tree นี้ถูกเขียนล่าสุด")
    lines.append("STALE       = tree นี้ไม่ค่อยมีไฟล์ถูกเขียนแล้ว")
    lines.append("=" * 150)
    lines.append("Ctrl+C = stop watcher")
    lines.append("=" * 150)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    previous_map: Dict[str, Tuple[float, int]] = {}

    while True:
        now_dt = datetime.now()
        rows = collect_files(root)
        changed_count, new_files_count, previous_map = detect_changes(rows, previous_map)

        clear_screen()
        print(
            render(
                root=root,
                rows=rows,
                now_dt=now_dt,
                stale_seconds=args.stale_seconds,
                changed_count=changed_count,
                new_files_count=new_files_count,
                top_n=args.top,
            )
        )

        if args.once:
            break

        try:
            time.sleep(max(0.5, args.interval))
        except KeyboardInterrupt:
            print("\n[STOPPED] activity watcher terminated by user")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] activity watcher terminated by user")
        sys.exit(0)
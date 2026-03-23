# ============================================================
# ชื่อโค้ด: cpu_isolated_soak.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\cpu_isolated_soak.py
# คำสั่งรัน: python C:\Data\Bot\Local_LLM\gold_research\jobs\cpu_isolated_soak.py --hours 8 --workers 8 --matrix-size 2048 --tag overnight_cpu_soak
# เวอร์ชัน: v1.0.0
# ============================================================

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import numpy as np


VERSION = "v1.0.0"
DEFAULT_OUTDIR = Path(r"C:\Data\Bot\Local_LLM\gold_research\runs\cpu_isolated_soak")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def worker_loop(worker_id: int, matrix_size: int, end_time: float, heartbeat_every: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed=worker_id + 20260320)
    a = rng.standard_normal((matrix_size, matrix_size), dtype=np.float32)
    b = rng.standard_normal((matrix_size, matrix_size), dtype=np.float32)

    iterations = 0
    checksum = 0.0
    last_mark = time.time()

    while time.time() < end_time:
        c = a @ b
        d = b @ a
        a = np.tanh(c + d, dtype=np.float32)
        b = np.maximum(c - d, 0.0, dtype=np.float32)
        checksum += float(a[0, 0]) + float(b[0, 0])
        iterations += 2

        if iterations % heartbeat_every == 0:
            now = time.time()
            _ = now - last_mark
            last_mark = now

    return {
        "worker_id": worker_id,
        "iterations": iterations,
        "checksum": checksum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone CPU soak job for overnight isolated load.")
    parser.add_argument("--hours", type=float, default=8.0, help="ระยะเวลารวมเป็นชั่วโมง")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2), help="จำนวน process")
    parser.add_argument("--matrix-size", type=int, default=2048, help="ขนาด matrix square")
    parser.add_argument("--tag", type=str, default="overnight_cpu_soak", help="tag สำหรับแยก output")
    args = parser.parse_args()

    outdir = DEFAULT_OUTDIR / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    meta = {
        "version": VERSION,
        "tag": args.tag,
        "started_at_utc": utc_now(),
        "hours": args.hours,
        "workers": args.workers,
        "matrix_size": args.matrix_size,
        "cpu_count": os.cpu_count(),
        "outdir": str(outdir),
    }
    write_json(outdir / "run_meta.json", meta)

    print("=" * 100)
    print(f"[INFO] version={VERSION}")
    print(f"[INFO] tag={args.tag}")
    print(f"[INFO] workers={args.workers}")
    print(f"[INFO] matrix_size={args.matrix_size}")
    print(f"[INFO] hours={args.hours}")
    print(f"[INFO] outdir={outdir}")
    print("=" * 100)

    start_ts = time.time()
    end_ts = start_ts + (args.hours * 3600.0)

    with mp.Pool(processes=args.workers) as pool:
        async_results = [
            pool.apply_async(worker_loop, kwds={
                "worker_id": i,
                "matrix_size": args.matrix_size,
                "end_time": end_ts,
                "heartbeat_every": 10,
            })
            for i in range(args.workers)
        ]

        while True:
            ready = sum(1 for r in async_results if r.ready())
            elapsed = time.time() - start_ts
            heartbeat = {
                "ts_utc": utc_now(),
                "elapsed_sec": round(elapsed, 3),
                "workers_total": args.workers,
                "workers_done": ready,
                "workers_running": args.workers - ready,
            }
            append_jsonl(outdir / "heartbeat.jsonl", heartbeat)
            write_json(outdir / "latest_status.json", heartbeat)
            print(
                f"[HEARTBEAT] elapsed_sec={heartbeat['elapsed_sec']} "
                f"workers_done={ready}/{args.workers}"
            )

            if ready == args.workers:
                break
            time.sleep(60)

        results = [r.get() for r in async_results]

    total_elapsed = time.time() - start_ts
    total_iterations = int(sum(x["iterations"] for x in results))
    total_checksum = float(sum(x["checksum"] for x in results))

    summary = {
        "version": VERSION,
        "tag": args.tag,
        "finished_at_utc": utc_now(),
        "elapsed_sec": round(total_elapsed, 3),
        "workers": args.workers,
        "matrix_size": args.matrix_size,
        "total_iterations": total_iterations,
        "total_checksum": total_checksum,
        "worker_results": results,
        "outdir": str(outdir),
    }
    write_json(outdir / "summary.json", summary)

    print("=" * 100)
    print("[DONE] cpu isolated soak completed")
    print(f"[DONE] summary={outdir / 'summary.json'}")
    print("=" * 100)


if __name__ == "__main__":
    mp.freeze_support()
    main()
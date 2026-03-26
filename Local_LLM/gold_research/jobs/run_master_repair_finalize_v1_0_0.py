#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ชื่อโค้ด: run_master_repair_finalize_v1_0_0.py
ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\run_master_repair_finalize_v1_0_0.py
คำสั่งรัน:
    python C:\Data\Bot\Local_LLM\gold_research\jobs\run_master_repair_finalize_v1_0_0.py
เวอร์ชัน: v1.0.0

Production purpose:
- ปิด pipeline ช่วงท้ายให้จบจริงจากของที่มีอยู่แล้ว
- auto-detect uncovered TF ที่มีจริง
- run pending -> micro exit -> aggregate ด้วย CLI contract ที่ถูกต้อง
- auto-save / auto-resume / progress / ETA / logs

หมายเหตุ:
- ไม่แตะไฟล์ orchestrator เดิม
- ใช้สำหรับ repair/finalize รอบนี้แบบ production-first
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


VERSION = "v1.0.0"
RESEARCH_TFS = ["M1", "M2", "M3", "M4", "M5", "M6", "M10", "M15", "M30", "H1", "H4", "D1"]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def fmt_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    if seconds < 0 or math.isnan(seconds) or math.isinf(seconds):
        return "-"
    total = int(round(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def run_subprocess(cmd: List[str], log_path: Path, cwd: Optional[Path] = None) -> int:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8", errors="replace") as logf:
        logf.write("=" * 120 + "\n")
        logf.write(f"[START] ts={now_utc_iso()} cmd={' '.join(cmd)}\n")
        logf.flush()

        process = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            logf.write(line)
            logf.flush()

        return_code = process.wait()
        logf.write(f"[END] ts={now_utc_iso()} returncode={return_code}\n")
        logf.write("=" * 120 + "\n")
        logf.flush()
        return return_code


def detect_feature_root(candidates: List[Path], symbol: str = "XAUUSD") -> Path:
    # ใช้ root ที่มีไฟล์จริงอย่างน้อยหนึ่ง TF
    for root in candidates:
        for tf in RESEARCH_TFS:
            fp = root / f"{symbol}_{tf}_base_features.parquet"
            if fp.exists():
                return root
    raise FileNotFoundError(
        "ไม่พบ feature root ที่มีไฟล์ *_base_features.parquet จริง"
    )


def detect_uncovered_timeframes(uncovered_dir: Path, symbol: str = "XAUUSD") -> List[str]:
    per_tf = uncovered_dir / "per_timeframe"
    found: List[str] = []
    if not per_tf.exists():
        return found

    for tf in RESEARCH_TFS:
        path = per_tf / f"{symbol}_{tf}_uncovered_results.csv"
        if path.exists() and path.stat().st_size > 0:
            found.append(tf)
    return found


def detect_pending_timeframes(pending_dir: Path, symbol: str = "XAUUSD") -> List[str]:
    per_tf = pending_dir / "per_timeframe"
    found: List[str] = []
    if not per_tf.exists():
        return found

    for tf in RESEARCH_TFS:
        path = per_tf / f"{symbol}_{tf}_pending_logic_results.csv"
        if path.exists() and path.stat().st_size > 0:
            found.append(tf)
    return found


def estimate_eta(start_ts: float, completed_phases: int, total_phases: int) -> Optional[float]:
    if completed_phases <= 0:
        return None
    elapsed = time.time() - start_ts
    rate = completed_phases / elapsed if elapsed > 0 else 0.0
    if rate <= 0:
        return None
    remaining = total_phases - completed_phases
    return remaining / rate


def write_progress(
    progress_path: Path,
    state_path: Path,
    current_phase: str,
    phase_statuses: Dict[str, str],
    started_at_iso: str,
    details: Dict[str, Any],
) -> None:
    done_count = sum(1 for s in phase_statuses.values() if s == "DONE")
    failed_count = sum(1 for s in phase_statuses.values() if s == "FAILED")
    running_count = sum(1 for s in phase_statuses.values() if s == "RUNNING")
    total = len(phase_statuses)
    pct = (done_count / total) * 100.0 if total > 0 else 0.0

    progress_payload = {
        "version": VERSION,
        "checked_at_utc": now_utc_iso(),
        "started_at_utc": started_at_iso,
        "current_phase": current_phase,
        "phases_total": total,
        "phases_done": done_count,
        "phases_failed": failed_count,
        "phases_running": running_count,
        "overall_progress_pct": round(pct, 2),
        "phase_statuses": phase_statuses,
        "details": details,
    }
    save_json(progress_path, progress_payload)

    state_payload = {
        "version": VERSION,
        "updated_at_utc": now_utc_iso(),
        "current_phase": current_phase,
        "phase_statuses": phase_statuses,
        "details": details,
    }
    save_json(state_path, state_payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair/finalize master pipeline tail phases")
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable path",
    )
    parser.add_argument(
        "--jobs-root",
        type=Path,
        default=Path(r"C:\Data\Bot\Local_LLM\gold_research\jobs"),
        help="Jobs directory",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=None,
        help="Optional explicit feature root",
    )
    parser.add_argument(
        "--feature-root-candidates",
        default=r"C:\Data\Bot\central_feature_cache;C:\Data\Bot\central_feature_cache\base_features_v1_0_0",
        help="Semicolon-separated candidate feature roots",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Data\Bot\central_market_data\parquet"),
        help="Parquet data root",
    )
    parser.add_argument(
        "--uncovered-dir",
        type=Path,
        default=Path(r"C:\Data\Bot\central_backtest_results\vectorbt_uncovered_matrix_v1_0_0"),
        help="Uncovered results directory",
    )
    parser.add_argument(
        "--pending-dir",
        type=Path,
        default=Path(r"C:\Data\Bot\central_backtest_results\vectorbt_pending_logic_matrix_v1_0_0"),
        help="Pending logic directory",
    )
    parser.add_argument(
        "--micro-exit-dir",
        type=Path,
        default=Path(r"C:\Data\Bot\central_backtest_results\vectorbt_micro_exit_matrix_v1_0_0"),
        help="Micro exit directory",
    )
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=Path(r"C:\Data\Bot\central_backtest_results\winner_registry_v1_0_0"),
        help="Winner registry root",
    )
    parser.add_argument(
        "--repair-root",
        type=Path,
        default=Path(r"C:\Data\Bot\central_backtest_results\master_repair_finalize_v1_0_0"),
        help="Repair runner state/log output root",
    )
    parser.add_argument(
        "--symbol",
        default="XAUUSD",
        help="Trading symbol for research files",
    )
    parser.add_argument(
        "--top-n-ema",
        type=int,
        default=12,
        help="Pending phase top EMA count",
    )
    parser.add_argument(
        "--pending-batch-size",
        type=int,
        default=24,
        help="Pending phase batch size",
    )
    parser.add_argument(
        "--top-n-candidates",
        type=int,
        default=100,
        help="Micro exit top pending candidates per TF",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=16,
        help="Micro exit batch size",
    )
    parser.add_argument(
        "--top-per-timeframe",
        type=int,
        default=50,
        help="Aggregate winners top per timeframe",
    )
    parser.add_argument(
        "--force-rerun-pending",
        action="store_true",
        help="Force rerun pending phase",
    )
    parser.add_argument(
        "--force-rerun-micro",
        action="store_true",
        help="Force rerun micro phase",
    )
    parser.add_argument(
        "--force-rerun-aggregate",
        action="store_true",
        help="Force rerun aggregate phase",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    ensure_dir(args.repair_root)
    ensure_dir(args.repair_root / "logs")
    ensure_dir(args.repair_root / "state")

    progress_path = args.repair_root / "live_progress.json"
    state_path = args.repair_root / "state" / "finalize_state.json"
    summary_path = args.repair_root / "summary.json"

    started_at_iso = now_utc_iso()
    start_ts = time.time()

    phase_order = [
        "phase_1_detect_inputs",
        "phase_2_pending_logic_repair",
        "phase_3_micro_exit_repair",
        "phase_4_registry_aggregate",
    ]
    phase_statuses = {phase: "PENDING" for phase in phase_order}
    details: Dict[str, Any] = {}

    write_progress(progress_path, state_path, "phase_1_detect_inputs", phase_statuses, started_at_iso, details)

    # Phase 1: detect inputs
    phase_statuses["phase_1_detect_inputs"] = "RUNNING"
    write_progress(progress_path, state_path, "phase_1_detect_inputs", phase_statuses, started_at_iso, details)

    if args.feature_root is not None:
        feature_root = args.feature_root
    else:
        candidates = [Path(x.strip()) for x in args.feature_root_candidates.split(";") if x.strip()]
        feature_root = detect_feature_root(candidates, symbol=args.symbol)

    uncovered_tfs = detect_uncovered_timeframes(args.uncovered_dir, symbol=args.symbol)
    if not uncovered_tfs:
        raise RuntimeError("ไม่พบ uncovered per_timeframe CSV ที่ใช้ต่อ downstream ได้")

    details["feature_root"] = str(feature_root)
    details["uncovered_detected_timeframes"] = uncovered_tfs
    details["uncovered_count"] = len(uncovered_tfs)

    phase_statuses["phase_1_detect_inputs"] = "DONE"
    write_progress(progress_path, state_path, "phase_2_pending_logic_repair", phase_statuses, started_at_iso, details)

    # Phase 2: pending
    phase_statuses["phase_2_pending_logic_repair"] = "RUNNING"
    write_progress(progress_path, state_path, "phase_2_pending_logic_repair", phase_statuses, started_at_iso, details)

    pending_script = args.jobs_root / "run_vectorbt_pending_logic_matrix_v1_0_0.py"
    pending_summary = args.pending_dir / "summary.json"
    pending_existing_tfs = detect_pending_timeframes(args.pending_dir, symbol=args.symbol)
    need_run_pending = args.force_rerun_pending or len(pending_existing_tfs) == 0

    if need_run_pending:
        pending_cmd = [
            args.python_exe,
            str(pending_script),
            "--data-root", str(args.data_root),
            "--feature-root", str(feature_root),
            "--ema-source-dir", str(args.uncovered_dir),
            "--outdir", str(args.pending_dir),
            "--symbol", args.symbol,
            "--timeframes", ",".join(uncovered_tfs),
            "--top-n-ema", str(args.top_n_ema),
            "--batch-size", str(args.pending_batch_size),
        ]
        rc = run_subprocess(pending_cmd, args.repair_root / "logs" / "phase_2_pending_logic_repair.log")
        if rc != 0:
            phase_statuses["phase_2_pending_logic_repair"] = "FAILED"
            details["phase_2_returncode"] = rc
            write_progress(progress_path, state_path, "phase_2_pending_logic_repair", phase_statuses, started_at_iso, details)
            raise SystemExit(rc)

    pending_tfs = detect_pending_timeframes(args.pending_dir, symbol=args.symbol)
    details["pending_detected_timeframes"] = pending_tfs
    details["pending_count"] = len(pending_tfs)
    details["pending_summary_path"] = str(pending_summary)

    if len(pending_tfs) == 0:
        phase_statuses["phase_2_pending_logic_repair"] = "FAILED"
        details["phase_2_reason"] = "pending per_timeframe results not created"
        write_progress(progress_path, state_path, "phase_2_pending_logic_repair", phase_statuses, started_at_iso, details)
        raise RuntimeError("pending phase ไม่สร้าง per_timeframe pending_logic_results.csv")

    phase_statuses["phase_2_pending_logic_repair"] = "DONE"
    write_progress(progress_path, state_path, "phase_3_micro_exit_repair", phase_statuses, started_at_iso, details)

    # Phase 3: micro exit
    phase_statuses["phase_3_micro_exit_repair"] = "RUNNING"
    write_progress(progress_path, state_path, "phase_3_micro_exit_repair", phase_statuses, started_at_iso, details)

    micro_script = args.jobs_root / "run_vectorbt_micro_exit_matrix_v1_0_0.py"
    micro_summary = args.micro_exit_dir / "summary.json"
    micro_need_run = args.force_rerun_micro or not micro_summary.exists()

    if micro_need_run:
        micro_cmd = [
            args.python_exe,
            str(micro_script),
            "--candidate-source-dir", str(args.pending_dir),
            "--data-root", str(args.data_root),
            "--feature-root", str(feature_root),
            "--outdir", str(args.micro_exit_dir),
            "--symbol", args.symbol,
            "--timeframes", ",".join(pending_tfs),
            "--top-n-candidates", str(args.top_n_candidates),
            "--batch-size", str(args.micro_batch_size),
        ]
        rc = run_subprocess(micro_cmd, args.repair_root / "logs" / "phase_3_micro_exit_repair.log")
        if rc != 0:
            phase_statuses["phase_3_micro_exit_repair"] = "FAILED"
            details["phase_3_returncode"] = rc
            write_progress(progress_path, state_path, "phase_3_micro_exit_repair", phase_statuses, started_at_iso, details)
            raise SystemExit(rc)

    micro_summary_json = load_json(micro_summary)
    details["micro_summary_path"] = str(micro_summary)
    details["micro_summary"] = micro_summary_json

    phase_statuses["phase_3_micro_exit_repair"] = "DONE"
    write_progress(progress_path, state_path, "phase_4_registry_aggregate", phase_statuses, started_at_iso, details)

    # Phase 4: aggregate
    phase_statuses["phase_4_registry_aggregate"] = "RUNNING"
    write_progress(progress_path, state_path, "phase_4_registry_aggregate", phase_statuses, started_at_iso, details)

    aggregate_script = args.jobs_root / "aggregate_winners_into_registry_v1_0_0.py"
    aggregate_summary = args.registry_root / "summary.json"
    need_run_aggregate = args.force_rerun_aggregate or True

    if need_run_aggregate:
        aggregate_cmd = [
            args.python_exe,
            str(aggregate_script),
            "--feature-root", str(feature_root),
            "--uncovered-dir", str(args.uncovered_dir),
            "--pending-dir", str(args.pending_dir),
            "--micro-exit-dir", str(args.micro_exit_dir),
            "--registry-root", str(args.registry_root),
            "--top-per-timeframe", str(args.top_per_timeframe),
        ]
        rc = run_subprocess(aggregate_cmd, args.repair_root / "logs" / "phase_4_registry_aggregate.log")
        if rc != 0:
            phase_statuses["phase_4_registry_aggregate"] = "FAILED"
            details["phase_4_returncode"] = rc
            write_progress(progress_path, state_path, "phase_4_registry_aggregate", phase_statuses, started_at_iso, details)
            raise SystemExit(rc)

    aggregate_summary_json = load_json(aggregate_summary)
    details["aggregate_summary_path"] = str(aggregate_summary)
    details["aggregate_summary"] = aggregate_summary_json

    phase_statuses["phase_4_registry_aggregate"] = "DONE"
    write_progress(progress_path, state_path, "-", phase_statuses, started_at_iso, details)

    elapsed = time.time() - start_ts
    eta = estimate_eta(start_ts, completed_phases=4, total_phases=4)

    final_summary = {
        "version": VERSION,
        "finished_at_utc": now_utc_iso(),
        "started_at_utc": started_at_iso,
        "elapsed_seconds": round(elapsed, 2),
        "elapsed_hms": fmt_seconds(elapsed),
        "eta_remaining": fmt_seconds(eta),
        "status": "DONE",
        "phase_statuses": phase_statuses,
        "details": details,
    }
    save_json(summary_path, final_summary)

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] repair_root={args.repair_root}")
    print(f"[DONE] feature_root={feature_root}")
    print(f"[DONE] uncovered_detected_timeframes={','.join(uncovered_tfs)}")
    print(f"[DONE] pending_detected_timeframes={','.join(pending_tfs)}")
    print(f"[DONE] live_progress={progress_path}")
    print(f"[DONE] summary_json={summary_path}")
    print(f"[DONE] registry_summary={aggregate_summary}")
    print("=" * 120)


if __name__ == "__main__":
    main()
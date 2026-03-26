
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_master_research_orchestrator_v1_0_1.py
Version: v1.0.1

Purpose:
- Orchestrate the full research pipeline in phases
- Auto-resume from saved state
- Auto-save progress, summary, and logs
- Continue-on-error optional
- Correct worker argument mapping for current worker scripts

Phases:
1. build_base_feature_cache_v1_0_0.py
2. run_vectorbt_uncovered_matrix_v1_0_0.py
3. run_vectorbt_pending_logic_matrix_v1_0_0.py
4. run_vectorbt_micro_exit_matrix_v1_0_0.py
5. aggregate_winners_into_registry_v1_0_0.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


VERSION = "v1.0.1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class PhaseSpec:
    order: int
    phase_id: str
    title: str
    script_name: str
    required: bool
    args: List[str]


def build_phase_specs(jobs_dir: Path, args: argparse.Namespace) -> List[PhaseSpec]:
    return [
        PhaseSpec(
            order=1,
            phase_id="phase_1_feature_cache",
            title="Build base feature cache",
            script_name="build_base_feature_cache_v1_0_0.py",
            required=False,
            args=[
                "--data-root", str(args.data_root),
                "--outdir", str(args.feature_outdir),
            ],
        ),
        PhaseSpec(
            order=2,
            phase_id="phase_2_uncovered_matrix",
            title="Run VectorBT uncovered matrix",
            script_name="run_vectorbt_uncovered_matrix_v1_0_0.py",
            required=True,
            args=[
                "--data-root", str(args.data_root),
                "--outdir", str(args.uncovered_outdir),
                "--batch-size", str(args.uncovered_batch_size),
            ],
        ),
        PhaseSpec(
            order=3,
            phase_id="phase_3_pending_logic",
            title="Run VectorBT pending logic matrix",
            script_name="run_vectorbt_pending_logic_matrix_v1_0_0.py",
            required=True,
            args=[
                "--ema-source-dir", str(args.uncovered_outdir),
                "--data-root", str(args.data_root),
                "--outdir", str(args.pending_outdir),
                "--top-n-ema", str(args.pending_top_n_ema),
                "--batch-size", str(args.pending_batch_size),
            ],
        ),
        PhaseSpec(
            order=4,
            phase_id="phase_4_micro_exit",
            title="Run VectorBT micro exit matrix",
            script_name="run_vectorbt_micro_exit_matrix_v1_0_0.py",
            required=True,
            args=[
                "--pending-source-dir", str(args.pending_outdir),
                "--data-root", str(args.data_root),
                "--outdir", str(args.micro_exit_outdir),
                "--top-n-candidates", str(args.micro_exit_top_n_candidates),
                "--batch-size", str(args.micro_exit_batch_size),
            ],
        ),
        PhaseSpec(
            order=5,
            phase_id="phase_5_registry_aggregation",
            title="Aggregate winners into registry",
            script_name="aggregate_winners_into_registry_v1_0_0.py",
            required=True,
            args=[
                "--feature-root", str(args.feature_outdir),
                "--uncovered-dir", str(args.uncovered_outdir),
                "--pending-dir", str(args.pending_outdir),
                "--micro-exit-dir", str(args.micro_exit_outdir),
                "--registry-root", str(args.registry_outdir),
            ],
        ),
    ]


def parse_args() -> argparse.Namespace:
    default_jobs_dir = Path(r"C:\Data\Bot\Local_LLM\gold_research\jobs")
    default_data_root = Path(r"C:\Data\Bot\central_market_data\parquet")
    base_out = Path(r"C:\Data\Bot\central_backtest_results")
    orchestrator_root = base_out / "master_research_orchestrator_v1_0_1"

    parser = argparse.ArgumentParser(description="Master research orchestrator with auto-resume")
    parser.add_argument("--jobs-dir", type=Path, default=default_jobs_dir)
    parser.add_argument("--data-root", type=Path, default=default_data_root)

    parser.add_argument("--feature-outdir", type=Path, default=base_out / "base_feature_cache_v1_0_0")
    parser.add_argument("--uncovered-outdir", type=Path, default=base_out / "vectorbt_uncovered_matrix_v1_0_0")
    parser.add_argument("--pending-outdir", type=Path, default=base_out / "vectorbt_pending_logic_matrix_v1_0_0")
    parser.add_argument("--micro-exit-outdir", type=Path, default=base_out / "vectorbt_micro_exit_matrix_v1_0_0")
    parser.add_argument("--registry-outdir", type=Path, default=base_out / "winner_registry_v1_0_0")

    parser.add_argument("--orchestrator-root", type=Path, default=orchestrator_root)
    parser.add_argument("--continue-on-error", action="store_true")

    parser.add_argument("--uncovered-batch-size", type=int, default=16)
    parser.add_argument("--pending-top-n-ema", type=int, default=20)
    parser.add_argument("--pending-batch-size", type=int, default=12)
    parser.add_argument("--micro-exit-top-n-candidates", type=int, default=100)
    parser.add_argument("--micro-exit-batch-size", type=int, default=16)

    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def detect_phase_completion(phase: PhaseSpec, args: argparse.Namespace) -> bool:
    # Conservative completion detection based on summary or expected key outputs
    if phase.phase_id == "phase_1_feature_cache":
        summary = args.feature_outdir / "summary.json"
        return summary.exists()
    if phase.phase_id == "phase_2_uncovered_matrix":
        summary = args.uncovered_outdir / "summary.json"
        return summary.exists()
    if phase.phase_id == "phase_3_pending_logic":
        summary = args.pending_outdir / "summary.json"
        return summary.exists()
    if phase.phase_id == "phase_4_micro_exit":
        summary = args.micro_exit_outdir / "summary.json"
        return summary.exists()
    if phase.phase_id == "phase_5_registry_aggregation":
        active_csv = args.registry_outdir / "active_winner_registry.csv"
        return active_csv.exists()
    return False


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        return read_json(state_path)
    return {
        "version": VERSION,
        "created_at_utc": utc_now_iso(),
        "updated_at_utc": utc_now_iso(),
        "status": "INITIALIZED",
        "phases": {},
    }


def save_state(state_path: Path, state: dict) -> None:
    state["updated_at_utc"] = utc_now_iso()
    write_json(state_path, state)


def build_live_progress(state: dict, phase_specs: List[PhaseSpec], current_phase_id: Optional[str], start_ts: float) -> dict:
    total = len(phase_specs)
    done = 0
    failed = 0
    running = 0

    for phase in phase_specs:
        p = state["phases"].get(phase.phase_id, {})
        status = p.get("status", "PENDING")
        if status == "DONE":
            done += 1
        elif status == "FAILED":
            failed += 1
        elif status == "RUNNING":
            running += 1

    elapsed_sec = max(0.0, time.time() - start_ts)
    progress_pct = round((done / total) * 100.0, 4) if total else 0.0

    eta_remaining_min = None
    if done > 0 and done < total:
        sec_per_phase = elapsed_sec / done
        eta_remaining_min = round(((total - done) * sec_per_phase) / 60.0, 2)
    elif done == total:
        eta_remaining_min = 0.0

    return {
        "version": VERSION,
        "checked_at_utc": utc_now_iso(),
        "current_phase": current_phase_id,
        "phases_total": total,
        "phases_done": done,
        "phases_failed": failed,
        "phases_running": running,
        "overall_progress_pct": progress_pct,
        "elapsed_min": round(elapsed_sec / 60.0, 2),
        "overall_eta_remaining_min": eta_remaining_min,
        "phase_statuses": {
            phase.phase_id: state["phases"].get(phase.phase_id, {}).get("status", "PENDING")
            for phase in phase_specs
        },
    }


def run_phase(phase: PhaseSpec, args: argparse.Namespace, jobs_dir: Path, log_dir: Path) -> subprocess.CompletedProcess:
    script_path = jobs_dir / phase.script_name
    if not script_path.exists():
        raise FileNotFoundError(f"worker_script_missing={script_path}")

    cmd = [sys.executable, str(script_path)] + phase.args
    log_path = log_dir / f"{phase.order:02d}_{phase.phase_id}.log"
    ensure_dir(log_dir)

    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"\n{'=' * 120}\n")
        lf.write(f"[START] ts={utc_now_iso()} version={VERSION} phase={phase.phase_id}\n")
        lf.write(f"[CMD] {' '.join(cmd)}\n")
        lf.flush()

        proc = subprocess.run(
            cmd,
            stdout=lf,
            stderr=lf,
            text=True,
            check=False,
        )

        lf.write(f"[END] ts={utc_now_iso()} returncode={proc.returncode} phase={phase.phase_id}\n")
        lf.flush()

    return proc


def main() -> None:
    args = parse_args()

    ensure_dir(args.orchestrator_root)
    state_dir = args.orchestrator_root / "state"
    logs_dir = args.orchestrator_root / "logs" / "phase_logs"
    ensure_dir(state_dir)
    ensure_dir(logs_dir)

    run_manifest_path = args.orchestrator_root / "run_manifest.json"
    state_path = state_dir / "orchestrator_state.json"
    live_progress_path = args.orchestrator_root / "live_progress.json"
    summary_path = args.orchestrator_root / "summary.json"

    phase_specs = build_phase_specs(args.jobs_dir, args)

    run_manifest = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "jobs_dir": str(args.jobs_dir),
        "data_root": str(args.data_root),
        "feature_outdir": str(args.feature_outdir),
        "uncovered_outdir": str(args.uncovered_outdir),
        "pending_outdir": str(args.pending_outdir),
        "micro_exit_outdir": str(args.micro_exit_outdir),
        "registry_outdir": str(args.registry_outdir),
        "continue_on_error": args.continue_on_error,
        "no_resume": args.no_resume,
        "phases": [asdict(p) for p in phase_specs],
    }
    write_json(run_manifest_path, run_manifest)

    state = load_state(state_path) if not args.no_resume else {
        "version": VERSION,
        "created_at_utc": utc_now_iso(),
        "updated_at_utc": utc_now_iso(),
        "status": "INITIALIZED",
        "phases": {},
    }

    start_ts = time.time()
    current_phase_id = None

    for phase in phase_specs:
        current_phase_id = phase.phase_id

        existing = state["phases"].get(phase.phase_id, {})
        prior_status = existing.get("status")
        already_complete = detect_phase_completion(phase, args)

        if prior_status == "DONE" and not args.no_resume:
            continue

        if already_complete and not args.no_resume:
            state["phases"][phase.phase_id] = {
                "order": phase.order,
                "title": phase.title,
                "script_name": phase.script_name,
                "status": "DONE",
                "started_at_utc": existing.get("started_at_utc", utc_now_iso()),
                "ended_at_utc": utc_now_iso(),
                "returncode": 0,
                "auto_detected_complete": True,
            }
            save_state(state_path, state)
            write_json(live_progress_path, build_live_progress(state, phase_specs, current_phase_id, start_ts))
            continue

        state["phases"][phase.phase_id] = {
            "order": phase.order,
            "title": phase.title,
            "script_name": phase.script_name,
            "status": "RUNNING",
            "started_at_utc": utc_now_iso(),
            "args": phase.args,
        }
        save_state(state_path, state)
        write_json(live_progress_path, build_live_progress(state, phase_specs, current_phase_id, start_ts))

        try:
            proc = run_phase(phase, args, args.jobs_dir, logs_dir)
            if proc.returncode == 0:
                state["phases"][phase.phase_id].update({
                    "status": "DONE",
                    "ended_at_utc": utc_now_iso(),
                    "returncode": 0,
                })
            else:
                state["phases"][phase.phase_id].update({
                    "status": "FAILED",
                    "ended_at_utc": utc_now_iso(),
                    "returncode": proc.returncode,
                })
                save_state(state_path, state)
                write_json(live_progress_path, build_live_progress(state, phase_specs, current_phase_id, start_ts))
                if phase.required and not args.continue_on_error:
                    state["status"] = "FAILED"
                    save_state(state_path, state)
                    write_json(summary_path, {
                        "version": VERSION,
                        "ended_at_utc": utc_now_iso(),
                        "status": "FAILED",
                        "failed_phase": phase.phase_id,
                        "state_path": str(state_path),
                        "live_progress_path": str(live_progress_path),
                    })
                    print(f"[FAILED] phase={phase.phase_id} returncode={proc.returncode}")
                    return
        except Exception as exc:
            state["phases"][phase.phase_id].update({
                "status": "FAILED",
                "ended_at_utc": utc_now_iso(),
                "returncode": None,
                "error": repr(exc),
            })
            save_state(state_path, state)
            write_json(live_progress_path, build_live_progress(state, phase_specs, current_phase_id, start_ts))
            if phase.required and not args.continue_on_error:
                state["status"] = "FAILED"
                save_state(state_path, state)
                write_json(summary_path, {
                    "version": VERSION,
                    "ended_at_utc": utc_now_iso(),
                    "status": "FAILED",
                    "failed_phase": phase.phase_id,
                    "error": repr(exc),
                    "state_path": str(state_path),
                    "live_progress_path": str(live_progress_path),
                })
                print(f"[FAILED] phase={phase.phase_id} error={exc!r}")
                return

        save_state(state_path, state)
        write_json(live_progress_path, build_live_progress(state, phase_specs, current_phase_id, start_ts))

    state["status"] = "DONE"
    save_state(state_path, state)
    final_progress = build_live_progress(state, phase_specs, None, start_ts)
    write_json(live_progress_path, final_progress)
    write_json(summary_path, {
        "version": VERSION,
        "ended_at_utc": utc_now_iso(),
        "status": "DONE",
        "phases_total": len(phase_specs),
        "phases_done": sum(1 for p in state["phases"].values() if p.get("status") == "DONE"),
        "phases_failed": sum(1 for p in state["phases"].values() if p.get("status") == "FAILED"),
        "elapsed_min": round((time.time() - start_ts) / 60.0, 2),
        "state_path": str(state_path),
        "live_progress_path": str(live_progress_path),
        "summary_path": str(summary_path),
    })
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] orchestrator_root={args.orchestrator_root}")
    print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()

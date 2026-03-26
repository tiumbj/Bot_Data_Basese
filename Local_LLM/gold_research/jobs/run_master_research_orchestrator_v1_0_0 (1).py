
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_master_research_orchestrator_v1_0_0.py
Version: v1.0.0

Master research orchestrator for:
1) build_base_feature_cache
2) run_vectorbt_uncovered_matrix
3) run_vectorbt_pending_logic_matrix
4) run_vectorbt_micro_exit_matrix
5) aggregate_winners_into_registry

Design goals:
- auto run all phases
- auto resume via orchestrator_state.json
- auto save progress/summary
- optional continue-on-error mode
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


VERSION = "v1.0.0"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PhaseSpec:
    phase_id: str
    script_name: str
    args: List[str]
    required: bool


def load_json(path: Path, default):
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return default


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def build_phase_specs(args: argparse.Namespace) -> List[PhaseSpec]:
    return [
        PhaseSpec(
            phase_id="build_base_feature_cache",
            script_name="build_base_feature_cache_v1_0_0.py",
            args=[
                "--data-root", str(args.data_root),
                "--outdir", str(args.feature_outdir),
                "--timeframes", ",".join(args.all_timeframes),
            ],
            required=True,
        ),
        PhaseSpec(
            phase_id="run_vectorbt_uncovered_matrix",
            script_name="run_vectorbt_uncovered_matrix_v1_0_0.py",
            args=[
                "--data-root", str(args.data_root),
                "--feature-root", str(args.feature_outdir),
                "--outdir", str(args.uncovered_outdir),
                "--timeframes", ",".join(args.uncovered_timeframes),
                "--batch-size", str(args.uncovered_batch_size),
            ],
            required=True,
        ),
        PhaseSpec(
            phase_id="run_vectorbt_pending_logic_matrix",
            script_name="run_vectorbt_pending_logic_matrix_v1_0_0.py",
            args=[
                "--data-root", str(args.data_root),
                "--feature-root", str(args.feature_outdir),
                "--ema-source-dir", str(args.uncovered_outdir),
                "--outdir", str(args.pending_outdir),
                "--timeframes", ",".join(args.all_timeframes),
                "--top-n-ema", str(args.top_n_ema),
                "--batch-size", str(args.pending_batch_size),
            ],
            required=True,
        ),
        PhaseSpec(
            phase_id="run_vectorbt_micro_exit_matrix",
            script_name="run_vectorbt_micro_exit_matrix_v1_0_0.py",
            args=[
                "--data-root", str(args.data_root),
                "--feature-root", str(args.feature_outdir),
                "--candidate-source-dir", str(args.pending_outdir),
                "--outdir", str(args.micro_exit_outdir),
                "--timeframes", ",".join(args.all_timeframes),
                "--top-n-candidates", str(args.top_n_candidates),
                "--batch-size", str(args.micro_batch_size),
            ],
            required=True,
        ),
        PhaseSpec(
            phase_id="aggregate_winners_into_registry",
            script_name="aggregate_winners_into_registry_v1_0_0.py",
            args=[
                "--feature-root", str(args.feature_outdir),
                "--uncovered-dir", str(args.uncovered_outdir),
                "--pending-dir", str(args.pending_outdir),
                "--micro-exit-dir", str(args.micro_exit_outdir),
                "--registry-root", str(args.registry_root),
                "--top-per-timeframe", str(args.registry_top_per_timeframe),
            ],
            required=True,
        ),
    ]


def discover_script_path(script_dir: Path, script_name: str) -> Optional[Path]:
    script_path = script_dir / script_name
    if script_path.exists():
        return script_path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Master research orchestrator")
    parser.add_argument("--script-dir", type=Path, default=Path(r"C:\Data\Bot\Local_LLM\gold_research\jobs"))
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\Data\Bot\central_market_data\parquet"))
    parser.add_argument("--master-outdir", type=Path, default=Path(r"C:\Data\Bot\central_backtest_results\master_research_orchestrator_v1_0_0"))
    parser.add_argument("--feature-outdir", type=Path, default=Path(r"C:\Data\Bot\central_feature_cache\base_features_v1_0_0"))
    parser.add_argument("--uncovered-outdir", type=Path, default=Path(r"C:\Data\Bot\central_backtest_results\vectorbt_uncovered_matrix_v1_0_0"))
    parser.add_argument("--pending-outdir", type=Path, default=Path(r"C:\Data\Bot\central_backtest_results\vectorbt_pending_logic_matrix_v1_0_0"))
    parser.add_argument("--micro-exit-outdir", type=Path, default=Path(r"C:\Data\Bot\central_backtest_results\vectorbt_micro_exit_matrix_v1_0_0"))
    parser.add_argument("--registry-root", type=Path, default=Path(r"C:\Data\Bot\central_strategy_registry\winner_registry_v1_0_0"))
    parser.add_argument("--all-timeframes", default="M1,M2,M3,M4,M5,M6,M10,M15,M30,H1,H4,D1")
    parser.add_argument("--uncovered-timeframes", default="M2,M3,M4,M6,M30,D1")
    parser.add_argument("--uncovered-batch-size", type=int, default=16)
    parser.add_argument("--pending-batch-size", type=int, default=24)
    parser.add_argument("--micro-batch-size", type=int, default=24)
    parser.add_argument("--top-n-ema", type=int, default=12)
    parser.add_argument("--top-n-candidates", type=int, default=10)
    parser.add_argument("--registry-top-per-timeframe", type=int, default=5)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    args.all_timeframes = [tf.strip() for tf in args.all_timeframes.split(",") if tf.strip()]
    args.uncovered_timeframes = [tf.strip() for tf in args.uncovered_timeframes.split(",") if tf.strip()]

    outdir = args.master_outdir
    logs_dir = outdir / "logs" / "phase_logs"
    state_path = outdir / "state" / "orchestrator_state.json"
    live_progress_path = outdir / "live_progress.json"
    summary_path = outdir / "summary.json"
    manifest_path = outdir / "run_manifest.json"
    logs_dir.mkdir(parents=True, exist_ok=True)

    phase_specs = build_phase_specs(args)
    manifest = {
        "version": VERSION,
        "created_at_utc": now_utc_iso(),
        "script_dir": str(args.script_dir),
        "data_root": str(args.data_root),
        "master_outdir": str(outdir),
        "phases": [asdict(p) for p in phase_specs],
    }
    save_json(manifest_path, manifest)

    state = load_json(state_path, {
        "version": VERSION,
        "started_at_utc": now_utc_iso(),
        "phases": {},
    })
    if args.no_resume:
        state = {
            "version": VERSION,
            "started_at_utc": now_utc_iso(),
            "phases": {},
        }

    overall_start = time.time()
    total_phases = len(phase_specs)
    completed_phases = 0

    for idx, phase in enumerate(phase_specs, start=1):
        phase_state = state["phases"].get(phase.phase_id, {})
        if phase_state.get("status") == "DONE":
            completed_phases += 1
            continue

        script_path = discover_script_path(args.script_dir, phase.script_name)
        if script_path is None:
            status = "SKIPPED" if not phase.required else "FAILED"
            message = f"script_not_found: {phase.script_name}"
            state["phases"][phase.phase_id] = {
                "status": status,
                "updated_at_utc": now_utc_iso(),
                "message": message,
            }
            save_json(state_path, state)
            if status == "FAILED" and not args.continue_on_error:
                raise FileNotFoundError(message)
            continue

        log_path = logs_dir / f"{idx:02d}_{phase.phase_id}.log"
        command = [sys.executable, str(script_path)] + phase.args
        live_progress = {
            "version": VERSION,
            "updated_at_utc": now_utc_iso(),
            "current_phase": phase.phase_id,
            "phase_index": idx,
            "phase_total": total_phases,
            "phase_progress_pct": round(((idx - 1) / total_phases) * 100.0, 4),
            "completed_phases": completed_phases,
            "remaining_phases": total_phases - completed_phases,
            "status": "RUNNING",
            "command": command,
        }
        save_json(live_progress_path, live_progress)

        state["phases"][phase.phase_id] = {
            "status": "RUNNING",
            "started_at_utc": now_utc_iso(),
            "script_path": str(script_path),
            "log_path": str(log_path),
            "command": command,
        }
        save_json(state_path, state)

        with log_path.open("a", encoding="utf-8") as log_fh:
            log_fh.write(f"[{now_utc_iso()}] START {phase.phase_id}\n")
            log_fh.write("COMMAND: " + " ".join(command) + "\n")
            log_fh.flush()
            proc = subprocess.run(command, stdout=log_fh, stderr=log_fh, text=True)

        if proc.returncode == 0:
            completed_phases += 1
            state["phases"][phase.phase_id].update({
                "status": "DONE",
                "updated_at_utc": now_utc_iso(),
                "return_code": proc.returncode,
            })
            save_json(state_path, state)
        else:
            state["phases"][phase.phase_id].update({
                "status": "FAILED",
                "updated_at_utc": now_utc_iso(),
                "return_code": proc.returncode,
            })
            save_json(state_path, state)
            if not args.continue_on_error:
                break

    done_count = sum(1 for p in state["phases"].values() if p.get("status") == "DONE")
    failed_count = sum(1 for p in state["phases"].values() if p.get("status") == "FAILED")
    skipped_count = sum(1 for p in state["phases"].values() if p.get("status") == "SKIPPED")
    total_elapsed = time.time() - overall_start
    summary = {
        "version": VERSION,
        "checked_at_utc": now_utc_iso(),
        "master_outdir": str(outdir),
        "phase_total": total_phases,
        "phase_done": done_count,
        "phase_failed": failed_count,
        "phase_skipped": skipped_count,
        "overall_progress_pct": round((done_count / total_phases) * 100.0, 4) if total_phases else 100.0,
        "elapsed_sec_this_session": round(total_elapsed, 4),
        "state_path": str(state_path),
        "manifest_path": str(manifest_path),
    }
    save_json(summary_path, summary)
    save_json(live_progress_path, {
        "version": VERSION,
        "updated_at_utc": now_utc_iso(),
        "status": "DONE" if failed_count == 0 else "PARTIAL",
        "overall_progress_pct": summary["overall_progress_pct"],
        "summary_path": str(summary_path),
    })

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] master_outdir={outdir}")
    print(f"[DONE] phase_total={total_phases}")
    print(f"[DONE] phase_done={done_count}")
    print(f"[DONE] phase_failed={failed_count}")
    print(f"[DONE] phase_skipped={skipped_count}")
    print(f"[DONE] overall_progress_pct={summary['overall_progress_pct']}")
    print(f"[DONE] summary_json={summary_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()

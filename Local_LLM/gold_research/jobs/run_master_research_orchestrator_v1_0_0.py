
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code Name  : run_master_research_orchestrator_v1_0_0.py
File Path  : C:\Data\Bot\Local_LLM\gold_research\jobs\run_master_research_orchestrator_v1_0_0.py
Run Command:
    python C:\Data\Bot\Local_LLM\gold_research\jobs\run_master_research_orchestrator_v1_0_0.py

Version    : v1.0.0

Changelog
---------
- v1.0.0
  - Added production-style master research orchestrator for the locked project direction.
  - Runs research phases automatically in sequence with resume-friendly behavior.
  - Records phase state, progress, ETA, logs, and summary JSON files.
  - Supports skip-if-done, continue-on-error, and deterministic phase manifest.
  - Designed to coordinate current VectorBT research scripts and future additions without manual babysitting.

Purpose
-------
This orchestrator is the control plane for the research factory.
It does not replace the existing backtest scripts.
It runs them in the correct order, tracks their status, and saves enough state
so the project can continue after interruption.

Locked Direction Alignment
--------------------------
- Parallel uncovered backtests remain part of the research pipeline.
- VectorBT is used for fast research layers.
- Results are intended to accumulate toward an intelligent database / winner registry pipeline.
- The orchestrator is built to be resume-friendly and production-minded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


VERSION = "v1.0.0"
DEFAULT_PROJECT_ROOT = Path(r"C:\Data\Bot\Local_LLM\gold_research")
DEFAULT_RESULTS_ROOT = Path(r"C:\Data\Bot\central_backtest_results")
DEFAULT_MARKET_DATA_ROOT = Path(r"C:\Data\Bot\central_market_data\parquet")
DEFAULT_JOBS_DIR = DEFAULT_PROJECT_ROOT / "jobs"

PHASE_STATUS_PENDING = "PENDING"
PHASE_STATUS_RUNNING = "RUNNING"
PHASE_STATUS_DONE = "DONE"
PHASE_STATUS_FAILED = "FAILED"
PHASE_STATUS_SKIPPED = "SKIPPED"

UTC = timezone.utc


def now_utc_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: dict) -> None:
    safe_mkdir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def human_minutes(minutes: Optional[float]) -> str:
    if minutes is None or math.isinf(minutes) or math.isnan(minutes):
        return "unknown"
    if minutes < 60:
        return f"{minutes:.2f} min"
    return f"{minutes / 60.0:.2f} hr"


def file_has_nonzero_size(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


@dataclass
class PhaseSpec:
    phase_id: str
    title: str
    script_name: str
    args: List[str]
    expected_outputs: List[str]
    soft_required: bool = True
    allow_skip_if_done: bool = True
    description: str = ""


@dataclass
class PhaseState:
    phase_id: str
    title: str
    script_name: str
    status: str
    started_at_utc: Optional[str]
    finished_at_utc: Optional[str]
    elapsed_sec: float
    expected_outputs: List[str]
    found_outputs: List[str]
    return_code: Optional[int]
    log_path: str
    error_message: str
    description: str


class ResearchOrchestrator:
    def __init__(
        self,
        jobs_dir: Path,
        market_data_root: Path,
        results_root: Path,
        outdir: Path,
        continue_on_error: bool,
        dry_run: bool,
    ) -> None:
        self.jobs_dir = jobs_dir
        self.market_data_root = market_data_root
        self.results_root = results_root
        self.outdir = outdir
        self.continue_on_error = continue_on_error
        self.dry_run = dry_run

        self.logs_dir = self.outdir / "logs"
        self.state_dir = self.outdir / "state"
        self.phase_logs_dir = self.logs_dir / "phase_logs"
        self.live_progress_json = self.outdir / "live_progress.json"
        self.orchestrator_state_json = self.state_dir / "orchestrator_state.json"
        self.manifest_json = self.outdir / "run_manifest.json"
        self.summary_json = self.outdir / "summary.json"

        safe_mkdir(self.logs_dir)
        safe_mkdir(self.state_dir)
        safe_mkdir(self.phase_logs_dir)

        self.started_at_ts = time.time()
        self.started_at_utc = now_utc_iso()
        self.phases: List[PhaseSpec] = self._build_phases()
        self.state: Dict[str, dict] = read_json(self.orchestrator_state_json, default={})

    def _build_phases(self) -> List[PhaseSpec]:
        # This phase plan is intentionally practical:
        # 1) Finish uncovered TF baseline fast research
        # 2) Expand pending logic + strategy coverage
        # 3) Reserve slot for micro-exit full-family runner (to be plugged in when ready)
        # 4) Aggregate outputs into a registry-ready summary
        return [
            PhaseSpec(
                phase_id="phase_01_uncovered_vbt",
                title="VectorBT uncovered baseline coverage",
                script_name="run_vectorbt_uncovered_parallel_backtests_v1_1_2.py",
                args=[
                    "--data-root", str(self.market_data_root),
                    "--outdir", str(self.results_root / "vectorbt_uncovered_parallel_v1_1_2"),
                ],
                expected_outputs=[
                    str(self.results_root / "vectorbt_uncovered_parallel_v1_1_2" / "summary.json"),
                    str(self.results_root / "vectorbt_uncovered_parallel_v1_1_2" / "live_progress.json"),
                ],
                description="Fast baseline sweep for uncovered timeframes using VectorBT.",
            ),
            PhaseSpec(
                phase_id="phase_02_pending_logic_vbt",
                title="VectorBT pending logic and strategy coverage",
                script_name="run_vectorbt_pending_master_coverage_v1_0_0.py",
                args=[
                    "--ema-source-dir", str(self.results_root / "vectorbt_uncovered_parallel_v1_1_2"),
                    "--data-root", str(self.market_data_root),
                    "--outdir", str(self.results_root / "vectorbt_pending_master_coverage_v1_0_0"),
                ],
                expected_outputs=[
                    str(self.results_root / "vectorbt_pending_master_coverage_v1_0_0" / "summary.json"),
                    str(self.results_root / "vectorbt_pending_master_coverage_v1_0_0" / "live_progress.json"),
                ],
                description="Covers pending strategy families and logic variants using top EMA winners.",
            ),
            PhaseSpec(
                phase_id="phase_03_micro_exit_full_family",
                title="VectorBT micro-exit full-family coverage",
                script_name="run_vectorbt_micro_exit_full_family_v1_0_0.py",
                args=[
                    "--source-dir", str(self.results_root / "vectorbt_pending_master_coverage_v1_0_0"),
                    "--data-root", str(self.market_data_root),
                    "--outdir", str(self.results_root / "vectorbt_micro_exit_full_family_v1_0_0"),
                ],
                expected_outputs=[
                    str(self.results_root / "vectorbt_micro_exit_full_family_v1_0_0" / "summary.json"),
                ],
                soft_required=True,
                description="Reserved next-phase micro-exit family coverage across research TF.",
            ),
            PhaseSpec(
                phase_id="phase_04_registry_ready_aggregation",
                title="Aggregate winners into registry-ready outputs",
                script_name="aggregate_winners_into_registry_v1_0_0.py",
                args=[
                    "--source-uncovered", str(self.results_root / "vectorbt_uncovered_parallel_v1_1_2"),
                    "--source-pending", str(self.results_root / "vectorbt_pending_master_coverage_v1_0_0"),
                    "--source-micro-exit", str(self.results_root / "vectorbt_micro_exit_full_family_v1_0_0"),
                    "--outdir", str(self.results_root / "registry_ready_aggregation_v1_0_0"),
                ],
                expected_outputs=[
                    str(self.results_root / "registry_ready_aggregation_v1_0_0" / "summary.json"),
                ],
                soft_required=True,
                description="Reserved aggregation phase for winner registry / intelligent DB preparation.",
            ),
        ]

    def save_manifest(self) -> None:
        payload = {
            "version": VERSION,
            "generated_at_utc": now_utc_iso(),
            "jobs_dir": str(self.jobs_dir),
            "market_data_root": str(self.market_data_root),
            "results_root": str(self.results_root),
            "outdir": str(self.outdir),
            "continue_on_error": self.continue_on_error,
            "dry_run": self.dry_run,
            "phase_count": len(self.phases),
            "phases": [asdict(p) for p in self.phases],
        }
        write_json(self.manifest_json, payload)

    def _phase_script_path(self, spec: PhaseSpec) -> Path:
        return self.jobs_dir / spec.script_name

    def _expected_output_paths(self, spec: PhaseSpec) -> List[Path]:
        return [Path(x) for x in spec.expected_outputs]

    def _is_phase_done_from_outputs(self, spec: PhaseSpec) -> bool:
        outputs = self._expected_output_paths(spec)
        if not outputs:
            return False
        return all(file_has_nonzero_size(p) for p in outputs)

    def _load_phase_state(self, phase_id: str) -> Optional[dict]:
        return self.state.get(phase_id)

    def _save_phase_state(self, phase_state: PhaseState) -> None:
        self.state[phase_state.phase_id] = asdict(phase_state)
        write_json(self.orchestrator_state_json, self.state)
        self._write_live_progress()

    def _write_live_progress(self) -> None:
        completed = 0
        failed = 0
        running = 0
        skipped = 0
        pending = 0

        phase_items = []
        for spec in self.phases:
            item = self.state.get(spec.phase_id)
            status = item["status"] if item else PHASE_STATUS_PENDING
            if status == PHASE_STATUS_DONE:
                completed += 1
            elif status == PHASE_STATUS_FAILED:
                failed += 1
            elif status == PHASE_STATUS_RUNNING:
                running += 1
            elif status == PHASE_STATUS_SKIPPED:
                skipped += 1
            else:
                pending += 1
            phase_items.append(item if item else {
                "phase_id": spec.phase_id,
                "title": spec.title,
                "status": PHASE_STATUS_PENDING,
            })

        total = len(self.phases)
        progress_pct = (completed / total * 100.0) if total > 0 else 0.0
        elapsed_sec = max(0.0, time.time() - self.started_at_ts)

        eta_min = None
        if completed > 0 and completed < total:
            avg_sec_per_completed_phase = elapsed_sec / completed
            eta_min = (avg_sec_per_completed_phase * (total - completed)) / 60.0

        payload = {
            "version": VERSION,
            "updated_at_utc": now_utc_iso(),
            "orchestrator_started_at_utc": self.started_at_utc,
            "outdir": str(self.outdir),
            "phase_total": total,
            "phase_completed": completed,
            "phase_failed": failed,
            "phase_running": running,
            "phase_skipped": skipped,
            "phase_pending": pending,
            "progress_pct": round(progress_pct, 6),
            "elapsed_min": round(elapsed_sec / 60.0, 6),
            "eta_remaining_min": round(eta_min, 6) if eta_min is not None else None,
            "phases": phase_items,
        }
        write_json(self.live_progress_json, payload)

    def _write_summary(self) -> None:
        items = list(self.state.values())
        payload = {
            "version": VERSION,
            "generated_at_utc": now_utc_iso(),
            "outdir": str(self.outdir),
            "phase_total": len(self.phases),
            "phase_done": sum(1 for x in items if x["status"] == PHASE_STATUS_DONE),
            "phase_failed": sum(1 for x in items if x["status"] == PHASE_STATUS_FAILED),
            "phase_skipped": sum(1 for x in items if x["status"] == PHASE_STATUS_SKIPPED),
            "phase_running": sum(1 for x in items if x["status"] == PHASE_STATUS_RUNNING),
            "phase_pending": sum(1 for x in items if x["status"] == PHASE_STATUS_PENDING),
            "total_elapsed_min": round((time.time() - self.started_at_ts) / 60.0, 6),
            "state_json": str(self.orchestrator_state_json),
            "live_progress_json": str(self.live_progress_json),
            "manifest_json": str(self.manifest_json),
            "phases": items,
        }
        write_json(self.summary_json, payload)

    def _print_progress_line(self, current_phase_idx: int, spec: PhaseSpec, status: str) -> None:
        total = len(self.phases)
        completed = sum(
            1 for item in self.state.values()
            if item.get("status") in {PHASE_STATUS_DONE, PHASE_STATUS_SKIPPED}
        )
        pct = (completed / total * 100.0) if total > 0 else 0.0
        elapsed_min = (time.time() - self.started_at_ts) / 60.0
        print(
            f"[MASTER] version={VERSION} phase={current_phase_idx}/{total} "
            f"phase_id={spec.phase_id} status={status} overall_progress_pct={pct:.2f} "
            f"elapsed={human_minutes(elapsed_min)}",
            flush=True,
        )

    def _run_subprocess(self, cmd: List[str], log_path: Path) -> Tuple[int, Optional[str]]:
        safe_mkdir(log_path.parent)
        if self.dry_run:
            log_path.write_text("[DRY_RUN] " + " ".join(cmd), encoding="utf-8")
            return 0, None

        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n{'=' * 120}\n")
            log_file.write(f"[START] {now_utc_iso()} cmd={' '.join(cmd)}\n")
            log_file.flush()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            try:
                assert process.stdout is not None
                for line in process.stdout:
                    sys.stdout.write(line)
                    log_file.write(line)
                    log_file.flush()
                process.wait()
            except KeyboardInterrupt:
                log_file.write(f"[INTERRUPTED] {now_utc_iso()}\n")
                log_file.flush()
                try:
                    process.terminate()
                except Exception:
                    pass
                raise
            except Exception as exc:
                log_file.write(f"[EXCEPTION] {now_utc_iso()} error={exc}\n")
                log_file.write(traceback.format_exc())
                log_file.flush()
                try:
                    process.terminate()
                except Exception:
                    pass
                return 999, str(exc)

        return process.returncode, None

    def run(self) -> int:
        self.save_manifest()
        self._write_live_progress()

        for idx, spec in enumerate(self.phases, start=1):
            self._print_progress_line(idx, spec, "CHECK")

            script_path = self._phase_script_path(spec)
            expected_outputs = self._expected_output_paths(spec)
            found_outputs_before = [str(p) for p in expected_outputs if file_has_nonzero_size(p)]

            if not script_path.exists():
                phase_state = PhaseState(
                    phase_id=spec.phase_id,
                    title=spec.title,
                    script_name=spec.script_name,
                    status=PHASE_STATUS_SKIPPED if spec.soft_required else PHASE_STATUS_FAILED,
                    started_at_utc=None,
                    finished_at_utc=now_utc_iso(),
                    elapsed_sec=0.0,
                    expected_outputs=[str(x) for x in expected_outputs],
                    found_outputs=found_outputs_before,
                    return_code=None,
                    log_path=str(self.phase_logs_dir / f"{spec.phase_id}.log"),
                    error_message=f"Script not found: {script_path}",
                    description=spec.description,
                )
                self._save_phase_state(phase_state)
                self._print_progress_line(idx, spec, phase_state.status)
                if not spec.soft_required and not self.continue_on_error:
                    self._write_summary()
                    return 1
                continue

            if spec.allow_skip_if_done and self._is_phase_done_from_outputs(spec):
                phase_state = PhaseState(
                    phase_id=spec.phase_id,
                    title=spec.title,
                    script_name=spec.script_name,
                    status=PHASE_STATUS_SKIPPED,
                    started_at_utc=None,
                    finished_at_utc=now_utc_iso(),
                    elapsed_sec=0.0,
                    expected_outputs=[str(x) for x in expected_outputs],
                    found_outputs=[str(p) for p in expected_outputs if file_has_nonzero_size(p)],
                    return_code=0,
                    log_path=str(self.phase_logs_dir / f"{spec.phase_id}.log"),
                    error_message="Skipped because outputs already exist.",
                    description=spec.description,
                )
                self._save_phase_state(phase_state)
                self._print_progress_line(idx, spec, PHASE_STATUS_SKIPPED)
                continue

            started_at_utc = now_utc_iso()
            phase_log_path = self.phase_logs_dir / f"{spec.phase_id}.log"
            running_state = PhaseState(
                phase_id=spec.phase_id,
                title=spec.title,
                script_name=spec.script_name,
                status=PHASE_STATUS_RUNNING,
                started_at_utc=started_at_utc,
                finished_at_utc=None,
                elapsed_sec=0.0,
                expected_outputs=[str(x) for x in expected_outputs],
                found_outputs=found_outputs_before,
                return_code=None,
                log_path=str(phase_log_path),
                error_message="",
                description=spec.description,
            )
            self._save_phase_state(running_state)
            self._print_progress_line(idx, spec, PHASE_STATUS_RUNNING)

            cmd = [sys.executable, str(script_path)] + spec.args
            phase_started_ts = time.time()
            return_code, subprocess_error = self._run_subprocess(cmd=cmd, log_path=phase_log_path)
            elapsed_sec = max(0.0, time.time() - phase_started_ts)
            found_outputs_after = [str(p) for p in expected_outputs if file_has_nonzero_size(p)]

            if return_code == 0:
                final_status = PHASE_STATUS_DONE
                err_msg = ""
            else:
                final_status = PHASE_STATUS_FAILED
                err_msg = subprocess_error or f"Subprocess exited with return code {return_code}"

            phase_state = PhaseState(
                phase_id=spec.phase_id,
                title=spec.title,
                script_name=spec.script_name,
                status=final_status,
                started_at_utc=started_at_utc,
                finished_at_utc=now_utc_iso(),
                elapsed_sec=elapsed_sec,
                expected_outputs=[str(x) for x in expected_outputs],
                found_outputs=found_outputs_after,
                return_code=return_code,
                log_path=str(phase_log_path),
                error_message=err_msg,
                description=spec.description,
            )
            self._save_phase_state(phase_state)
            self._print_progress_line(idx, spec, final_status)

            if final_status == PHASE_STATUS_FAILED and not self.continue_on_error:
                self._write_summary()
                return 1

        self._write_summary()
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Master research orchestrator for the locked gold research pipeline."
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=DEFAULT_JOBS_DIR,
        help="Directory containing runnable research job scripts.",
    )
    parser.add_argument(
        "--market-data-root",
        type=Path,
        default=DEFAULT_MARKET_DATA_ROOT,
        help="Canonical market data root directory.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Central backtest results root directory.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "master_research_orchestrator_v1_0_0",
        help="Directory for orchestrator state, logs, manifest, and summary.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining phases even if one phase fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write manifest and logs without launching child scripts.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    safe_mkdir(args.outdir)

    orchestrator = ResearchOrchestrator(
        jobs_dir=args.jobs_dir,
        market_data_root=args.market_data_root,
        results_root=args.results_root,
        outdir=args.outdir,
        continue_on_error=args.continue_on_error,
        dry_run=args.dry_run,
    )
    exit_code = orchestrator.run()

    print("=" * 120, flush=True)
    print(f"[DONE] version={VERSION}", flush=True)
    print(f"[DONE] outdir={args.outdir}", flush=True)
    print(f"[DONE] manifest_json={orchestrator.manifest_json}", flush=True)
    print(f"[DONE] state_json={orchestrator.orchestrator_state_json}", flush=True)
    print(f"[DONE] progress_json={orchestrator.live_progress_json}", flush=True)
    print(f"[DONE] summary_json={orchestrator.summary_json}", flush=True)
    print(f"[DONE] exit_code={exit_code}", flush=True)
    print("=" * 120, flush=True)

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

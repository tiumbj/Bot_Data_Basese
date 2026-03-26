# ============================================================
# ชื่อโค้ด: run_seed_execution_watchdog_manager_v1_0_1.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\run_seed_execution_watchdog_manager_v1_0_1.py
# คำสั่งรัน:
# python C:\Data\Bot\Local_LLM\gold_research\jobs\run_seed_execution_watchdog_manager_v1_0_1.py --manifest C:\Data\Bot\central_backtest_results\truth_gate_seed_manifest_v1_0_0\truth_gate_seed_manifest.csv --data-root C:\Data\Bot\central_market_data\parquet --feature-root C:\Data\Bot\central_feature_cache --base-out C:\Data\Bot\central_backtest_results\truth_gate_seed_managed_run_v1_0_1
# เวอร์ชัน: v1.0.1
# ============================================================

from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


VERSION = "v1.0.1"
DEFAULT_RUNNER = r"C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_micro_exit_coverage_batched_v1_0_8.py"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


class StateDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        ensure_dir(db_path.parent)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS manager_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                runner_script TEXT NOT NULL,
                manifest TEXT NOT NULL,
                data_root TEXT NOT NULL,
                feature_root TEXT NOT NULL,
                base_out TEXT NOT NULL,
                shard_count INTEGER NOT NULL,
                stuck_timeout_sec INTEGER NOT NULL,
                poll_sec INTEGER NOT NULL,
                max_retries INTEGER NOT NULL,
                phase TEXT NOT NULL,
                portfolio_chunk_size INTEGER NOT NULL,
                preflight_flush_size INTEGER NOT NULL,
                progress_every_groups INTEGER NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS shard_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                manager_run_id INTEGER NOT NULL,
                shard_index INTEGER NOT NULL,
                attempt_no INTEGER NOT NULL,
                outdir TEXT NOT NULL,
                started_at_utc TEXT NOT NULL,
                ended_at_utc TEXT,
                status TEXT NOT NULL,
                exit_code INTEGER,
                last_heartbeat_utc TEXT,
                last_progress_mtime_utc TEXT,
                last_timeframe TEXT,
                last_done INTEGER DEFAULT 0,
                last_total INTEGER DEFAULT 0,
                last_groups_completed INTEGER DEFAULT 0,
                last_groups_total INTEGER DEFAULT 0,
                last_rate REAL DEFAULT 0.0,
                note TEXT,
                UNIQUE(manager_run_id, shard_index, attempt_no)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS shard_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                manager_run_id INTEGER NOT NULL,
                shard_index INTEGER NOT NULL,
                attempt_no INTEGER NOT NULL,
                created_at_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_message TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def insert_manager_run(self, payload: Dict[str, Any]) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO manager_runs (
                version, created_at_utc, runner_script, manifest, data_root, feature_root, base_out,
                shard_count, stuck_timeout_sec, poll_sec, max_retries, phase,
                portfolio_chunk_size, preflight_flush_size, progress_every_groups
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["version"],
                payload["created_at_utc"],
                payload["runner_script"],
                payload["manifest"],
                payload["data_root"],
                payload["feature_root"],
                payload["base_out"],
                payload["shard_count"],
                payload["stuck_timeout_sec"],
                payload["poll_sec"],
                payload["max_retries"],
                payload["phase"],
                payload["portfolio_chunk_size"],
                payload["preflight_flush_size"],
                payload["progress_every_groups"],
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def insert_attempt(self, manager_run_id: int, shard_index: int, attempt_no: int, outdir: Path) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO shard_attempts (
                manager_run_id, shard_index, attempt_no, outdir, started_at_utc, status
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                manager_run_id,
                shard_index,
                attempt_no,
                str(outdir),
                utc_now_iso(),
                "RUNNING",
            ),
        )
        self.conn.commit()

    def update_attempt(self, manager_run_id: int, shard_index: int, attempt_no: int, **kwargs: Any) -> None:
        allowed = {
            "ended_at_utc",
            "status",
            "exit_code",
            "last_heartbeat_utc",
            "last_progress_mtime_utc",
            "last_timeframe",
            "last_done",
            "last_total",
            "last_groups_completed",
            "last_groups_total",
            "last_rate",
            "note",
        }
        parts = []
        values = []
        for key, value in kwargs.items():
            if key not in allowed:
                continue
            parts.append(f"{key} = ?")
            values.append(value)
        if not parts:
            return
        values.extend([manager_run_id, shard_index, attempt_no])
        sql = f"""
            UPDATE shard_attempts
            SET {", ".join(parts)}
            WHERE manager_run_id = ? AND shard_index = ? AND attempt_no = ?
        """
        self.conn.execute(sql, values)
        self.conn.commit()

    def insert_event(self, manager_run_id: int, shard_index: int, attempt_no: int, event_type: str, event_message: str) -> None:
        self.conn.execute(
            """
            INSERT INTO shard_events (
                manager_run_id, shard_index, attempt_no, created_at_utc, event_type, event_message
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                manager_run_id,
                shard_index,
                attempt_no,
                utc_now_iso(),
                event_type,
                event_message,
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


@dataclass
class ShardRuntime:
    shard_index: int
    attempt_no: int = 0
    process: Optional[subprocess.Popen] = None
    outdir: Optional[Path] = None
    stdout_path: Optional[Path] = None
    stderr_path: Optional[Path] = None
    last_progress_payload: Dict[str, Any] = field(default_factory=dict)
    last_progress_mtime: Optional[float] = None
    last_heartbeat_seen_at: Optional[float] = None
    completed: bool = False
    failed: bool = False
    restarts: int = 0
    status: str = "PENDING"

    @property
    def live_progress_path(self) -> Optional[Path]:
        if self.outdir is None:
            return None
        return self.outdir / "live_progress.json"


class SeedExecutionWatchdogManager:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        ensure_dir(self.args.base_out)
        self.state_db = StateDB(self.args.base_out / "execution_manager_state.sqlite")
        self.manager_run_id = self.state_db.insert_manager_run(
            {
                "version": VERSION,
                "created_at_utc": utc_now_iso(),
                "runner_script": str(self.args.runner_script),
                "manifest": str(self.args.manifest),
                "data_root": str(self.args.data_root),
                "feature_root": str(self.args.feature_root),
                "base_out": str(self.args.base_out),
                "shard_count": self.args.shard_count,
                "stuck_timeout_sec": self.args.stuck_timeout_sec,
                "poll_sec": self.args.poll_sec,
                "max_retries": self.args.max_retries,
                "phase": self.args.phase,
                "portfolio_chunk_size": self.args.portfolio_chunk_size,
                "preflight_flush_size": self.args.preflight_flush_size,
                "progress_every_groups": self.args.progress_every_groups,
            }
        )
        self.shards: Dict[int, ShardRuntime] = {
            idx: ShardRuntime(shard_index=idx) for idx in range(self.args.shard_count)
        }
        self.stop_requested = False
        self._register_signal_handlers()

    def _register_signal_handlers(self) -> None:
        def handle_stop(signum: int, _frame: Any) -> None:
            self.stop_requested = True
            print(f"[MANAGER] stop_requested signal={signum}")

        signal.signal(signal.SIGINT, handle_stop)
        signal.signal(signal.SIGTERM, handle_stop)

    def _build_attempt_outdir(self, shard_index: int, attempt_no: int) -> Path:
        return self.args.base_out / f"shard_{shard_index}" / f"attempt_{attempt_no:02d}"

    def _build_runner_args(self, shard_index: int, outdir: Path) -> List[str]:
        return [
            self.args.python_exe,
            str(self.args.runner_script),
            "--manifest",
            str(self.args.manifest),
            "--data-root",
            str(self.args.data_root),
            "--feature-root",
            str(self.args.feature_root),
            "--outdir",
            str(outdir),
            "--phase",
            self.args.phase,
            "--portfolio-chunk-size",
            str(self.args.portfolio_chunk_size),
            "--preflight-flush-size",
            str(self.args.preflight_flush_size),
            "--progress-every-groups",
            str(self.args.progress_every_groups),
            "--shard-index",
            str(shard_index),
            "--shard-count",
            str(self.args.shard_count),
            "--continue-on-error",
        ]

    def start_all(self) -> None:
        for shard_index in range(self.args.shard_count):
            self.start_shard(shard_index)

    def start_shard(self, shard_index: int) -> None:
        shard = self.shards[shard_index]
        shard.attempt_no += 1
        shard.status = "STARTING"
        shard.outdir = self._build_attempt_outdir(shard_index, shard.attempt_no)
        ensure_dir(shard.outdir)
        shard.stdout_path = shard.outdir / "stdout.log"
        shard.stderr_path = shard.outdir / "stderr.log"

        cmd = self._build_runner_args(shard_index, shard.outdir)

        stdout_handle = open(shard.stdout_path, "w", encoding="utf-8")
        stderr_handle = open(shard.stderr_path, "w", encoding="utf-8")

        shard.process = subprocess.Popen(
            cmd,
            stdout=stdout_handle,
            stderr=stderr_handle,
            cwd=str(self.args.base_out),
        )
        shard.last_progress_payload = {}
        shard.last_progress_mtime = None
        shard.last_heartbeat_seen_at = time.time()
        shard.status = "RUNNING"

        self.state_db.insert_attempt(self.manager_run_id, shard_index, shard.attempt_no, shard.outdir)
        self.state_db.insert_event(
            self.manager_run_id,
            shard_index,
            shard.attempt_no,
            "START",
            f"Started shard {shard_index} attempt {shard.attempt_no} pid={shard.process.pid} outdir={shard.outdir}",
        )

    def terminate_shard(self, shard_index: int, reason: str) -> None:
        shard = self.shards[shard_index]
        if shard.process is None:
            return
        try:
            shard.process.terminate()
            try:
                shard.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                shard.process.kill()
                shard.process.wait(timeout=15)
        except Exception:
            pass
        self.state_db.insert_event(
            self.manager_run_id,
            shard_index,
            shard.attempt_no,
            "TERMINATE",
            reason,
        )
        self.state_db.update_attempt(
            self.manager_run_id,
            shard_index,
            shard.attempt_no,
            ended_at_utc=utc_now_iso(),
            status="TERMINATED",
            exit_code=shard.process.returncode if shard.process else None,
            note=reason,
        )
        shard.process = None
        shard.status = "TERMINATED"

    def _update_progress_snapshot(self, shard_index: int) -> None:
        shard = self.shards[shard_index]
        progress_path = shard.live_progress_path
        if progress_path is None or not progress_path.exists():
            return

        mtime = progress_path.stat().st_mtime
        if shard.last_progress_mtime is None or mtime > shard.last_progress_mtime:
            payload = read_json(progress_path)
            if payload is not None:
                shard.last_progress_payload = payload
                shard.last_progress_mtime = mtime
                shard.last_heartbeat_seen_at = time.time()
                self.state_db.update_attempt(
                    self.manager_run_id,
                    shard_index,
                    shard.attempt_no,
                    last_heartbeat_utc=utc_now_iso(),
                    last_progress_mtime_utc=datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
                    last_timeframe=payload.get("current_timeframe"),
                    last_done=safe_int(payload.get("execution_done")),
                    last_total=safe_int(payload.get("execution_total")),
                    last_groups_completed=safe_int(payload.get("groups_completed")),
                    last_groups_total=safe_int(payload.get("groups_total")),
                    last_rate=safe_float(payload.get("observed_execution_rate_jobs_per_min")),
                )

    def _mark_completed(self, shard_index: int, exit_code: int) -> None:
        shard = self.shards[shard_index]
        shard.completed = True
        shard.status = "COMPLETED"
        self.state_db.update_attempt(
            self.manager_run_id,
            shard_index,
            shard.attempt_no,
            ended_at_utc=utc_now_iso(),
            status="COMPLETED",
            exit_code=exit_code,
            note="Shard completed successfully",
        )
        self.state_db.insert_event(
            self.manager_run_id,
            shard_index,
            shard.attempt_no,
            "COMPLETE",
            f"Shard {shard_index} completed exit_code={exit_code}",
        )

    def _mark_failed(self, shard_index: int, exit_code: int, reason: str) -> None:
        shard = self.shards[shard_index]
        shard.failed = True
        shard.status = "FAILED"
        self.state_db.update_attempt(
            self.manager_run_id,
            shard_index,
            shard.attempt_no,
            ended_at_utc=utc_now_iso(),
            status="FAILED",
            exit_code=exit_code,
            note=reason,
        )
        self.state_db.insert_event(
            self.manager_run_id,
            shard_index,
            shard.attempt_no,
            "FAIL",
            f"Shard {shard_index} failed exit_code={exit_code} reason={reason}",
        )

    def _should_restart(self, shard_index: int) -> bool:
        shard = self.shards[shard_index]
        return shard.restarts < self.args.max_retries

    def _restart_shard(self, shard_index: int, reason: str) -> None:
        shard = self.shards[shard_index]
        shard.restarts += 1
        self.state_db.insert_event(
            self.manager_run_id,
            shard_index,
            shard.attempt_no,
            "RESTART_REQUEST",
            f"Restarting shard {shard_index} restart_no={shard.restarts} reason={reason}",
        )
        self.start_shard(shard_index)

    def check_shard(self, shard_index: int) -> None:
        shard = self.shards[shard_index]
        if shard.completed or shard.failed:
            return

        self._update_progress_snapshot(shard_index)

        if shard.process is None:
            return

        exit_code = shard.process.poll()
        if exit_code is not None:
            if exit_code == 0:
                self._mark_completed(shard_index, exit_code)
            else:
                reason = f"Process exited non-zero exit_code={exit_code}"
                self._mark_failed(shard_index, exit_code, reason)
                if self._should_restart(shard_index):
                    self._restart_shard(shard_index, reason)
                else:
                    shard.status = "FAILED_MAX_RETRIES"
            return

        now = time.time()
        last_seen = shard.last_heartbeat_seen_at or now
        age_sec = now - last_seen

        if age_sec > self.args.stuck_timeout_sec:
            reason = (
                f"Heartbeat timeout shard={shard_index} "
                f"age_sec={round(age_sec, 2)} "
                f"tf={shard.last_progress_payload.get('current_timeframe')} "
                f"done={safe_int(shard.last_progress_payload.get('execution_done'))} "
                f"groups={safe_int(shard.last_progress_payload.get('groups_completed'))}/"
                f"{safe_int(shard.last_progress_payload.get('groups_total'))}"
            )
            self.terminate_shard(shard_index, reason)
            if self._should_restart(shard_index):
                self._restart_shard(shard_index, reason)
            else:
                shard.failed = True
                shard.status = "FAILED_MAX_RETRIES"

    def render_status(self) -> None:
        os.system("cls" if os.name == "nt" else "clear")
        print("=" * 140)
        print(f"[MANAGER] version={VERSION} now_utc={utc_now_iso()} base_out={self.args.base_out}")
        print(
            f"[MANAGER] runner={self.args.runner_script} shard_count={self.args.shard_count} "
            f"stuck_timeout_sec={self.args.stuck_timeout_sec} poll_sec={self.args.poll_sec} "
            f"max_retries={self.args.max_retries}"
        )
        print("=" * 140)
        print(f"{'shard':<7} {'attempt':<8} {'status':<18} {'tf':<5} {'done':<8} {'total':<8} {'groups':<12} {'rate':<10} {'restarts':<8} {'heartbeat_age_s':<16} {'outdir'}")

        total_done = 0
        total_total = 0
        total_rate = 0.0

        for shard_index in range(self.args.shard_count):
            shard = self.shards[shard_index]
            payload = shard.last_progress_payload
            done = safe_int(payload.get("execution_done"))
            total = safe_int(payload.get("execution_total"))
            groups_done = safe_int(payload.get("groups_completed"))
            groups_total = safe_int(payload.get("groups_total"))
            rate = safe_float(payload.get("observed_execution_rate_jobs_per_min"))
            tf = str(payload.get("current_timeframe") or "-")
            age = "-"
            if shard.last_heartbeat_seen_at is not None:
                age = f"{round(time.time() - shard.last_heartbeat_seen_at, 1)}"

            total_done += done
            total_total += total
            total_rate += rate

            outdir_str = str(shard.outdir) if shard.outdir else "-"
            print(
                f"{shard_index:<7} {shard.attempt_no:<8} {shard.status:<18} {tf:<5} {done:<8} {total:<8} "
                f"{f'{groups_done}/{groups_total}':<12} {round(rate, 2):<10} {shard.restarts:<8} {age:<16} {outdir_str}"
            )

        print("-" * 140)
        print(f"TOTAL_DONE={total_done} TOTAL_TOTAL={total_total} TOTAL_RATE_JOBS_PER_MIN={round(total_rate, 2)}")
        print("=" * 140)

    def all_finished(self) -> bool:
        for shard in self.shards.values():
            if not shard.completed and not shard.failed:
                return False
        return True

    def stop_all(self) -> None:
        for shard_index, shard in self.shards.items():
            if shard.process is not None and shard.process.poll() is None:
                self.terminate_shard(shard_index, "Manager shutdown requested")

    def run(self) -> int:
        self.start_all()
        while True:
            if self.stop_requested:
                self.stop_all()
                break

            for shard_index in range(self.args.shard_count):
                self.check_shard(shard_index)

            self.render_status()

            if self.all_finished():
                break

            time.sleep(self.args.poll_sec)

        completed = sum(1 for shard in self.shards.values() if shard.completed)
        failed = sum(1 for shard in self.shards.values() if shard.failed)
        print(f"[MANAGER DONE] completed={completed} failed={failed}")

        self.state_db.close()
        return 0 if failed == 0 else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Managed seed execution watchdog for v1.0.8 runner.")
    parser.add_argument("--runner-script", type=Path, default=Path(DEFAULT_RUNNER))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--base-out", type=Path, required=True)
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--stuck-timeout-sec", type=int, default=180)
    parser.add_argument("--poll-sec", type=int, default=15)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--phase", type=str, default="micro_exit_expansion")
    parser.add_argument("--portfolio-chunk-size", type=int, default=16)
    parser.add_argument("--preflight-flush-size", type=int, default=1000)
    parser.add_argument("--progress-every-groups", type=int, default=1)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.runner_script.exists():
        print(f"[ERROR] runner_script not found: {args.runner_script}")
        return 2
    if not args.manifest.exists():
        print(f"[ERROR] manifest not found: {args.manifest}")
        return 2
    if not args.data_root.exists():
        print(f"[ERROR] data_root not found: {args.data_root}")
        return 2
    if not args.feature_root.exists():
        print(f"[ERROR] feature_root not found: {args.feature_root}")
        return 2

    ensure_dir(args.base_out)

    manager = SeedExecutionWatchdogManager(args)
    return manager.run()


if __name__ == "__main__":
    raise SystemExit(main())
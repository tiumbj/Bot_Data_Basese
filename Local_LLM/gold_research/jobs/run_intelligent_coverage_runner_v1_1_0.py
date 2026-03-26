#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_intelligent_coverage_runner_v1_1_0.py

Version: v1.1.0
Path   : C:/Data/Bot/Local_LLM/gold_research/jobs/run_intelligent_coverage_runner_v1_1_0.py

Production-grade intelligent coverage runner foundation.

What this file fixes:
1. Deterministic preflight classification.
2. Non-empty status_reason for every rejected row.
3. Canonical support registry in one place.
4. Resume-safe group execution using SQLite checkpoints.
5. Honest progress metrics separated into manifest/preflight/execution views.
6. Queryable state store instead of giant CSV-only orchestration.

Execution modes:
- preflight_only : build classified state + summaries only.
- noop_executor  : run valid groups through a checkpointed no-op executor.

Notes:
- This is the orchestration/reliability layer.
- It is intentionally executor-agnostic so it can be wired to family-specific
  vectorbt executors in the next integration step without redesigning the runner.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import signal
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

VERSION = "v1.1.0"
DB_NAME = "runner_state.sqlite"
BOOTSTRAP_LOG_NAME = "bootstrap.log"
LIVE_PROGRESS_NAME = "live_progress.json"
PRECHECK_SUMMARY_NAME = "preflight_summary.json"
GROUP_SUMMARY_CSV_NAME = "summary_by_group.csv"
REASON_SUMMARY_CSV_NAME = "summary_by_reason.csv"
FAMILY_SUMMARY_CSV_NAME = "summary_by_family.csv"
RUN_SUMMARY_JSON_NAME = "run_summary.json"

REQUIRED_COLUMNS = (
    "manifest_id",
    "phase",
    "symbol",
    "timeframe",
    "strategy_family",
    "logic_strictness",
    "swing_variant",
    "micro_exit_variant",
)

DEFAULT_ALLOWED_TIMEFRAMES = (
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M10",
    "M15",
    "M30",
    "H1",
    "H4",
    "D1",
)

KNOWN_FAMILIES = (
    "adx_ema_trend_continuation",
    "bos_continuation",
    "ema_trend_continuation",
    "market_structure_continuation",
    "swing_pullback",
    "breakout_retest",
    "choch_reversal",
    "failed_breakout_reversal",
    "session_conditioned_trend",
    "volatility_expansion_contraction",
)


@dataclass(frozen=True)
class FamilySupportSpec:
    family: str
    allowed_timeframes: Tuple[str, ...] = DEFAULT_ALLOWED_TIMEFRAMES
    allowed_logic_strictness: Tuple[str, ...] = ("strict", "medium", "relaxed")
    allowed_swing_variants: Tuple[str, ...] = ("short", "medium", "long")
    allowed_micro_exit_variants: Tuple[str, ...] = ("*",)
    allowed_sides: Tuple[str, ...] = ("", "long", "short", "both")
    enabled: bool = True

    def supports_micro_exit(self, value: str) -> bool:
        if not self.allowed_micro_exit_variants:
            return False
        if "*" in self.allowed_micro_exit_variants:
            return True
        return value in self.allowed_micro_exit_variants


@dataclass
class SupportRegistry:
    families: Dict[str, FamilySupportSpec] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "SupportRegistry":
        specs = {
            family: FamilySupportSpec(family=family)
            for family in KNOWN_FAMILIES
        }
        return cls(families=specs)

    @classmethod
    def from_json(cls, path: Optional[Path]) -> "SupportRegistry":
        if path is None:
            return cls.default()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        families: Dict[str, FamilySupportSpec] = {}
        for row in payload.get("families", []):
            spec = FamilySupportSpec(
                family=str(row["family"]),
                allowed_timeframes=tuple(row.get("allowed_timeframes", list(DEFAULT_ALLOWED_TIMEFRAMES))),
                allowed_logic_strictness=tuple(row.get("allowed_logic_strictness", ["strict", "medium", "relaxed"])),
                allowed_swing_variants=tuple(row.get("allowed_swing_variants", ["short", "medium", "long"])),
                allowed_micro_exit_variants=tuple(row.get("allowed_micro_exit_variants", ["*"])),
                allowed_sides=tuple(row.get("allowed_sides", ["", "long", "short", "both"])),
                enabled=bool(row.get("enabled", True)),
            )
            families[spec.family] = spec
        return cls(families=families)

    def classify_support(self, row: Dict[str, str]) -> Tuple[bool, str]:
        family = clean_value(row.get("strategy_family"))
        timeframe = clean_value(row.get("timeframe"))
        logic_strictness = clean_value(row.get("logic_strictness"))
        swing_variant = clean_value(row.get("swing_variant"))
        micro_exit_variant = clean_value(row.get("micro_exit_variant"))
        side = clean_value(row.get("side"))

        if family not in self.families:
            return False, f"unsupported strategy_family={family or '<EMPTY>'}"

        spec = self.families[family]
        if not spec.enabled:
            return False, f"disabled strategy_family={family}"
        if timeframe not in spec.allowed_timeframes:
            return False, f"unsupported timeframe={timeframe or '<EMPTY>'} for family={family}"
        if logic_strictness not in spec.allowed_logic_strictness:
            return False, f"unsupported logic_strictness={logic_strictness or '<EMPTY>'} for family={family}"
        if swing_variant not in spec.allowed_swing_variants:
            return False, f"unsupported swing_variant={swing_variant or '<EMPTY>'} for family={family}"
        if not spec.supports_micro_exit(micro_exit_variant):
            return False, f"unsupported micro_exit_variant={micro_exit_variant or '<EMPTY>'} for family={family}"
        if side not in spec.allowed_sides:
            return False, f"unsupported side={side or '<EMPTY>'} for family={family}"
        return True, "VALID"


@dataclass
class Counters:
    manifest_total: int = 0
    preflight_valid: int = 0
    preflight_invalid: int = 0
    preflight_unsupported: int = 0
    preflight_skipped: int = 0
    execution_total_groups: int = 0
    execution_done_groups: int = 0
    execution_total_jobs: int = 0
    execution_done_jobs: int = 0
    execution_error_groups: int = 0
    execution_resumed_groups: int = 0

    def as_dict(self) -> Dict[str, int]:
        return dataclasses.asdict(self)


class GracefulShutdown:
    def __init__(self) -> None:
        self.stop_requested = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum: int, _frame) -> None:
        self.stop_requested = True
        sys.stderr.write(f"\n[WARN] received_signal={signum} stop_requested=true\n")
        sys.stderr.flush()


class IntelligentCoverageRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.manifest_path = Path(args.manifest)
        self.outdir = Path(args.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.outdir / DB_NAME
        self.bootstrap_log_path = self.outdir / BOOTSTRAP_LOG_NAME
        self.live_progress_path = self.outdir / LIVE_PROGRESS_NAME
        self.preflight_summary_path = self.outdir / PRECHECK_SUMMARY_NAME
        self.group_summary_csv_path = self.outdir / GROUP_SUMMARY_CSV_NAME
        self.reason_summary_csv_path = self.outdir / REASON_SUMMARY_CSV_NAME
        self.family_summary_csv_path = self.outdir / FAMILY_SUMMARY_CSV_NAME
        self.run_summary_json_path = self.outdir / RUN_SUMMARY_JSON_NAME
        self.registry = SupportRegistry.from_json(Path(args.support_registry_json) if args.support_registry_json else None)
        self.counters = Counters()
        self.start_monotonic = time.monotonic()
        self.shutdown = GracefulShutdown()
        self.last_progress_flush = 0.0
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self._init_db()

    def close(self) -> None:
        try:
            self.conn.commit()
        finally:
            self.conn.close()

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        with self.bootstrap_log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line)

    def _init_db(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS run_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS preflight_results (
                job_id TEXT PRIMARY KEY,
                manifest_id TEXT,
                row_index INTEGER,
                phase TEXT,
                symbol TEXT,
                timeframe TEXT,
                strategy_family TEXT,
                logic_variant TEXT,
                logic_strictness TEXT,
                swing_variant TEXT,
                pullback_zone_variant TEXT,
                entry_variant TEXT,
                micro_exit_variant TEXT,
                management_variant TEXT,
                regime_variant TEXT,
                robustness_variant TEXT,
                side TEXT,
                status TEXT NOT NULL,
                status_reason TEXT NOT NULL,
                shard_index INTEGER NOT NULL,
                group_key TEXT,
                raw_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS group_execution (
                group_key TEXT PRIMARY KEY,
                shard_index INTEGER NOT NULL,
                timeframe TEXT,
                strategy_family TEXT,
                status TEXT NOT NULL,
                job_count INTEGER NOT NULL,
                completed_job_count INTEGER NOT NULL,
                last_error TEXT,
                updated_at_utc TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_preflight_status ON preflight_results(status);
            CREATE INDEX IF NOT EXISTS idx_preflight_reason ON preflight_results(status_reason);
            CREATE INDEX IF NOT EXISTS idx_preflight_family ON preflight_results(strategy_family);
            CREATE INDEX IF NOT EXISTS idx_preflight_timeframe ON preflight_results(timeframe);
            CREATE INDEX IF NOT EXISTS idx_preflight_shard ON preflight_results(shard_index);
            CREATE INDEX IF NOT EXISTS idx_group_status ON group_execution(status);
            """
        )
        self.conn.commit()

    def run(self) -> int:
        self.log(
            f"START version={VERSION} execution_mode={self.args.execution_mode} "
            f"manifest={self.manifest_path} outdir={self.outdir} shard_index={self.args.shard_index} shard_count={self.args.shard_count}"
        )
        self.write_run_meta()
        self.preflight_manifest()
        self.persist_group_plan()
        self.write_summaries()
        self.flush_progress(force=True)

        if self.args.execution_mode == "preflight_only":
            self.log("DONE mode=preflight_only")
            return 0

        self.execute_groups_noop()
        self.write_summaries()
        self.flush_progress(force=True)
        self.write_run_summary()
        self.log("DONE mode=noop_executor")
        return 0

    def write_run_meta(self) -> None:
        now_utc = utc_now_iso()
        items = {
            "version": VERSION,
            "manifest": str(self.manifest_path),
            "outdir": str(self.outdir),
            "execution_mode": self.args.execution_mode,
            "phase": self.args.phase,
            "shard_index": str(self.args.shard_index),
            "shard_count": str(self.args.shard_count),
            "started_at_utc": now_utc,
        }
        self.conn.executemany(
            "INSERT OR REPLACE INTO run_meta(key, value) VALUES(?, ?)",
            list(items.items()),
        )
        self.conn.commit()

    def preflight_manifest(self) -> None:
        self.log("PREFLIGHT begin")
        insert_sql = (
            "INSERT OR REPLACE INTO preflight_results("
            "job_id, manifest_id, row_index, phase, symbol, timeframe, strategy_family, logic_variant, logic_strictness, "
            "swing_variant, pullback_zone_variant, entry_variant, micro_exit_variant, management_variant, regime_variant, "
            "robustness_variant, side, status, status_reason, shard_index, group_key, raw_json"
            ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )

        batch: List[Tuple[object, ...]] = []
        with self.manifest_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            self._validate_manifest_columns(reader.fieldnames or [])

            for row in reader:
                if self.shutdown.stop_requested:
                    self.log("PREFLIGHT interrupted_by_signal=true")
                    break

                self.counters.manifest_total += 1
                normalized = {k: clean_value(v) for k, v in row.items()}
                job_id = clean_value(normalized.get("job_id")) or self._hash_row(normalized)
                manifest_id = clean_value(normalized.get("manifest_id"))
                row_index = to_int(normalized.get("row_index"), default=self.counters.manifest_total - 1)
                decision_status, status_reason, group_key = self.classify_row(normalized)
                shard_index = self.compute_shard_index(normalized)

                if decision_status == "VALID":
                    self.counters.preflight_valid += 1
                elif decision_status == "INVALID_ROW":
                    self.counters.preflight_invalid += 1
                elif decision_status == "UNSUPPORTED_CONFIG":
                    self.counters.preflight_unsupported += 1
                elif decision_status == "SKIPPED_ROW":
                    self.counters.preflight_skipped += 1
                else:
                    raise RuntimeError(f"unknown_decision_status={decision_status}")

                batch.append(
                    (
                        job_id,
                        manifest_id,
                        row_index,
                        clean_value(normalized.get("phase")),
                        clean_value(normalized.get("symbol")),
                        clean_value(normalized.get("timeframe")),
                        clean_value(normalized.get("strategy_family")),
                        clean_value(normalized.get("logic_variant")),
                        clean_value(normalized.get("logic_strictness")),
                        clean_value(normalized.get("swing_variant")),
                        clean_value(normalized.get("pullback_zone_variant")),
                        clean_value(normalized.get("entry_variant")),
                        clean_value(normalized.get("micro_exit_variant")),
                        clean_value(normalized.get("management_variant")),
                        clean_value(normalized.get("regime_variant")),
                        clean_value(normalized.get("robustness_variant")),
                        clean_value(normalized.get("side")),
                        decision_status,
                        status_reason,
                        shard_index,
                        group_key,
                        json.dumps(normalized, ensure_ascii=False, separators=(",", ":")),
                    )
                )

                if len(batch) >= self.args.preflight_flush_size:
                    self.conn.executemany(insert_sql, batch)
                    self.conn.commit()
                    batch.clear()
                    self.flush_progress(force=False)

            if batch:
                self.conn.executemany(insert_sql, batch)
                self.conn.commit()
        self.log(
            f"PREFLIGHT done manifest_total={self.counters.manifest_total} "
            f"valid={self.counters.preflight_valid} invalid={self.counters.preflight_invalid} "
            f"unsupported={self.counters.preflight_unsupported} skipped={self.counters.preflight_skipped}"
        )
        self.write_preflight_summary()

    def _validate_manifest_columns(self, fieldnames: Sequence[str]) -> None:
        missing = [col for col in REQUIRED_COLUMNS if col not in fieldnames]
        if missing:
            raise ValueError(f"manifest_missing_required_columns={missing}")

    def classify_row(self, row: Dict[str, str]) -> Tuple[str, str, Optional[str]]:
        missing = [col for col in REQUIRED_COLUMNS if not clean_value(row.get(col))]
        if missing:
            return "INVALID_ROW", f"missing required columns={','.join(missing)}", None

        if self.args.phase and clean_value(row.get("phase")) and clean_value(row.get("phase")) != self.args.phase:
            return "SKIPPED_ROW", f"phase mismatch row_phase={clean_value(row.get('phase'))} runner_phase={self.args.phase}", None

        shard_index = self.compute_shard_index(row)
        if shard_index != self.args.shard_index:
            return "SKIPPED_ROW", f"assigned to shard_index={shard_index}", None

        supported, reason = self.registry.classify_support(row)
        if not supported:
            return "UNSUPPORTED_CONFIG", reason, None

        group_key = self.make_group_key(row)
        return "VALID", "VALID", group_key

    def compute_shard_index(self, row: Dict[str, str]) -> int:
        identity = "|".join(
            [
                clean_value(row.get("symbol")),
                clean_value(row.get("timeframe")),
                clean_value(row.get("strategy_family")),
                clean_value(row.get("logic_variant")),
                clean_value(row.get("logic_strictness")),
                clean_value(row.get("swing_variant")),
                clean_value(row.get("entry_variant")),
                clean_value(row.get("micro_exit_variant")),
                clean_value(row.get("side")),
                clean_value(row.get("job_id")),
            ]
        )
        digest = hashlib.md5(identity.encode("utf-8")).hexdigest()
        value = int(digest[:12], 16)
        return value % self.args.shard_count

    def make_group_key(self, row: Dict[str, str]) -> str:
        parts = [
            clean_value(row.get("phase")),
            clean_value(row.get("symbol")),
            clean_value(row.get("timeframe")),
            clean_value(row.get("strategy_family")),
            clean_value(row.get("logic_variant")),
            clean_value(row.get("logic_strictness")),
            clean_value(row.get("swing_variant")),
            clean_value(row.get("entry_variant")),
            clean_value(row.get("micro_exit_variant")),
            clean_value(row.get("management_variant")),
            clean_value(row.get("regime_variant")),
            clean_value(row.get("robustness_variant")),
            clean_value(row.get("side")),
            clean_value(row.get("ema_fast")),
            clean_value(row.get("ema_slow")),
        ]
        raw = "|".join(parts)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def persist_group_plan(self) -> None:
        self.log("GROUP_PLAN begin")
        rows = self.conn.execute(
            """
            SELECT group_key, shard_index, MIN(timeframe) AS timeframe, MIN(strategy_family) AS strategy_family, COUNT(*) AS job_count
            FROM preflight_results
            WHERE status = 'VALID'
            GROUP BY group_key, shard_index
            """
        ).fetchall()

        self.counters.execution_total_groups = len(rows)
        self.counters.execution_total_jobs = sum(int(row["job_count"]) for row in rows)

        upserts: List[Tuple[object, ...]] = []
        resumed_groups = 0
        for row in rows:
            existing = self.conn.execute(
                "SELECT status, completed_job_count FROM group_execution WHERE group_key = ?",
                (row["group_key"],),
            ).fetchone()
            if existing and existing["status"] == "DONE":
                resumed_groups += 1
                self.counters.execution_done_groups += 1
                self.counters.execution_done_jobs += int(existing["completed_job_count"])
                continue

            upserts.append(
                (
                    row["group_key"],
                    int(row["shard_index"]),
                    clean_value(row["timeframe"]),
                    clean_value(row["strategy_family"]),
                    "PENDING",
                    int(row["job_count"]),
                    0,
                    None,
                    utc_now_iso(),
                )
            )

        if upserts:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO group_execution(
                    group_key, shard_index, timeframe, strategy_family, status, job_count, completed_job_count, last_error, updated_at_utc
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                upserts,
            )
            self.conn.commit()

        self.counters.execution_resumed_groups = resumed_groups
        self.log(
            f"GROUP_PLAN done total_groups={self.counters.execution_total_groups} total_jobs={self.counters.execution_total_jobs} resumed_groups={resumed_groups}"
        )

    def execute_groups_noop(self) -> None:
        self.log("EXECUTION begin mode=noop_executor")
        rows = self.conn.execute(
            """
            SELECT group_key, timeframe, strategy_family, job_count
            FROM group_execution
            WHERE status != 'DONE'
            ORDER BY timeframe, strategy_family, group_key
            """
        ).fetchall()

        for row in rows:
            if self.shutdown.stop_requested:
                self.log("EXECUTION interrupted_by_signal=true")
                break

            group_key = row["group_key"]
            job_count = int(row["job_count"])
            timeframe = clean_value(row["timeframe"])
            strategy_family = clean_value(row["strategy_family"])

            try:
                self.conn.execute(
                    "UPDATE group_execution SET status = ?, updated_at_utc = ? WHERE group_key = ?",
                    ("RUNNING", utc_now_iso(), group_key),
                )
                self.conn.commit()

                # No-op executor: proves deterministic routing + checkpoint flow without touching strategy logic.
                time.sleep(self.args.noop_group_sleep_ms / 1000.0)

                self.conn.execute(
                    """
                    UPDATE group_execution
                    SET status = ?, completed_job_count = ?, last_error = NULL, updated_at_utc = ?
                    WHERE group_key = ?
                    """,
                    ("DONE", job_count, utc_now_iso(), group_key),
                )
                self.conn.commit()
                self.counters.execution_done_groups += 1
                self.counters.execution_done_jobs += job_count
            except Exception as exc:  # pragma: no cover - defensive
                self.conn.execute(
                    """
                    UPDATE group_execution
                    SET status = ?, last_error = ?, updated_at_utc = ?
                    WHERE group_key = ?
                    """,
                    ("ERROR", str(exc), utc_now_iso(), group_key),
                )
                self.conn.commit()
                self.counters.execution_error_groups += 1

            self.flush_progress(force=False)

        self.log(
            f"EXECUTION done done_groups={self.counters.execution_done_groups} done_jobs={self.counters.execution_done_jobs} errors={self.counters.execution_error_groups}"
        )

    def write_preflight_summary(self) -> None:
        payload = {
            "version": VERSION,
            "phase": self.args.phase,
            "manifest": str(self.manifest_path),
            "outdir": str(self.outdir),
            "manifest_total": self.counters.manifest_total,
            "preflight_valid": self.counters.preflight_valid,
            "preflight_invalid": self.counters.preflight_invalid,
            "preflight_unsupported": self.counters.preflight_unsupported,
            "preflight_skipped": self.counters.preflight_skipped,
            "updated_at_utc": utc_now_iso(),
        }
        self.preflight_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_summaries(self) -> None:
        self._write_reason_summary_csv()
        self._write_family_summary_csv()
        self._write_group_summary_csv()
        self.write_run_summary()

    def _write_reason_summary_csv(self) -> None:
        rows = self.conn.execute(
            """
            SELECT status, status_reason, COUNT(*) AS row_count
            FROM preflight_results
            GROUP BY status, status_reason
            ORDER BY row_count DESC, status, status_reason
            """
        ).fetchall()
        with self.reason_summary_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["status", "status_reason", "row_count"])
            for row in rows:
                writer.writerow([row["status"], row["status_reason"], row["row_count"]])

    def _write_family_summary_csv(self) -> None:
        rows = self.conn.execute(
            """
            SELECT strategy_family, status, COUNT(*) AS row_count
            FROM preflight_results
            GROUP BY strategy_family, status
            ORDER BY strategy_family, status
            """
        ).fetchall()
        with self.family_summary_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["strategy_family", "status", "row_count"])
            for row in rows:
                writer.writerow([row["strategy_family"], row["status"], row["row_count"]])

    def _write_group_summary_csv(self) -> None:
        rows = self.conn.execute(
            """
            SELECT group_key, shard_index, timeframe, strategy_family, status, job_count, completed_job_count, last_error, updated_at_utc
            FROM group_execution
            ORDER BY timeframe, strategy_family, group_key
            """
        ).fetchall()
        with self.group_summary_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "group_key",
                "shard_index",
                "timeframe",
                "strategy_family",
                "status",
                "job_count",
                "completed_job_count",
                "last_error",
                "updated_at_utc",
            ])
            for row in rows:
                writer.writerow(
                    [
                        row["group_key"],
                        row["shard_index"],
                        row["timeframe"],
                        row["strategy_family"],
                        row["status"],
                        row["job_count"],
                        row["completed_job_count"],
                        row["last_error"],
                        row["updated_at_utc"],
                    ]
                )

    def write_run_summary(self) -> None:
        payload = self.build_live_progress_payload()
        self.run_summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def build_live_progress_payload(self) -> Dict[str, object]:
        elapsed_min = max((time.monotonic() - self.start_monotonic) / 60.0, 1e-9)
        execution_remaining_jobs = max(self.counters.execution_total_jobs - self.counters.execution_done_jobs, 0)
        execution_rate_jobs_per_min = self.counters.execution_done_jobs / elapsed_min
        eta_remaining_min = (
            execution_remaining_jobs / execution_rate_jobs_per_min
            if execution_rate_jobs_per_min > 0
            else None
        )
        manifest_supported_pct = (
            (self.counters.preflight_valid / self.counters.manifest_total) * 100.0
            if self.counters.manifest_total > 0
            else 0.0
        )
        execution_progress_pct = (
            (self.counters.execution_done_jobs / self.counters.execution_total_jobs) * 100.0
            if self.counters.execution_total_jobs > 0
            else 0.0
        )

        active_row = self.conn.execute(
            """
            SELECT timeframe, strategy_family, status, updated_at_utc
            FROM group_execution
            WHERE status IN ('RUNNING', 'PENDING')
            ORDER BY CASE status WHEN 'RUNNING' THEN 0 ELSE 1 END, updated_at_utc ASC
            LIMIT 1
            """
        ).fetchone()

        return {
            "version": VERSION,
            "phase": self.args.phase,
            "execution_mode": self.args.execution_mode,
            "manifest_total": self.counters.manifest_total,
            "preflight_valid": self.counters.preflight_valid,
            "preflight_invalid": self.counters.preflight_invalid,
            "preflight_unsupported": self.counters.preflight_unsupported,
            "preflight_skipped": self.counters.preflight_skipped,
            "manifest_supported_pct": round(manifest_supported_pct, 4),
            "execution_total_groups": self.counters.execution_total_groups,
            "execution_done_groups": self.counters.execution_done_groups,
            "execution_error_groups": self.counters.execution_error_groups,
            "execution_resumed_groups": self.counters.execution_resumed_groups,
            "execution_total_jobs": self.counters.execution_total_jobs,
            "execution_done_jobs": self.counters.execution_done_jobs,
            "execution_remaining_jobs": execution_remaining_jobs,
            "execution_progress_pct": round(execution_progress_pct, 4),
            "observed_elapsed_min": round(elapsed_min, 4),
            "observed_execution_rate_jobs_per_min": round(execution_rate_jobs_per_min, 4),
            "execution_eta_remaining_min": round(eta_remaining_min, 4) if eta_remaining_min is not None else None,
            "current_timeframe": clean_value(active_row["timeframe"]) if active_row else None,
            "current_strategy_family": clean_value(active_row["strategy_family"]) if active_row else None,
            "current_group_status": clean_value(active_row["status"]) if active_row else None,
            "outdir": str(self.outdir),
            "updated_at_utc": utc_now_iso(),
        }

    def flush_progress(self, force: bool) -> None:
        now = time.monotonic()
        if (not force) and (now - self.last_progress_flush < self.args.progress_update_every_sec):
            return
        payload = self.build_live_progress_payload()
        self.live_progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.last_progress_flush = now

    def _hash_row(self, row: Dict[str, str]) -> str:
        body = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha1(body.encode("utf-8")).hexdigest()[:20]


def clean_value(value: Optional[object]) -> str:
    if value is None:
        return ""
    return str(value).strip()


def to_int(value: Optional[object], default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(str(value).strip()))
    except Exception:
        return default


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Intelligent coverage runner with deterministic preflight and resume-safe group checkpoints.")
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--phase", default="", help="Optional phase filter.")
    parser.add_argument("--execution-mode", choices=("preflight_only", "noop_executor"), default="preflight_only")
    parser.add_argument("--support-registry-json", default="", help="Optional JSON file that overrides the embedded support registry.")
    parser.add_argument("--shard-index", type=int, default=0, help="Current shard index.")
    parser.add_argument("--shard-count", type=int, default=1, help="Total shard count.")
    parser.add_argument("--preflight-flush-size", type=int, default=5000, help="Rows per SQLite commit during preflight.")
    parser.add_argument("--progress-update-every-sec", type=float, default=5.0, help="Progress JSON flush cadence in seconds.")
    parser.add_argument("--noop-group-sleep-ms", type=int, default=0, help="Optional sleep per group for checkpoint smoke tests.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.shard_index < 0:
        raise ValueError("--shard-index must be >= 0")
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be > 0")
    if args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must be < --shard-count")

    runner = IntelligentCoverageRunner(args)
    try:
        return runner.run()
    finally:
        runner.close()


if __name__ == "__main__":
    raise SystemExit(main())

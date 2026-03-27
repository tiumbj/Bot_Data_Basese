"""
standardized_result_pack_v1_0_0.py
Version: v1.0.0
Purpose:
    Standardized result writing for Gate 4 runner.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


RESULT_COLUMNS = [
    "result_key",
    "run_id",
    "job_id",
    "group_id",
    "timeframe",
    "strategy",
    "entry",
    "micro_exit",
    "status",
    "reject_reason",
    "error_reason",
    "missing_feature_reason",
    "recovery_status",
    "started_at_utc",
    "finished_at_utc",
    "duration_sec",
    "metrics_json",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_replace(src: Path, dst: Path) -> None:
    os.replace(src, dst)


def deterministic_result_key(job_payload: Dict[str, Any]) -> str:
    if job_payload.get("job_id"):
        base = str(job_payload["job_id"])
    else:
        base = "|".join(
            [
                str(job_payload.get("timeframe", "")),
                str(job_payload.get("strategy", "")),
                str(job_payload.get("entry", "")),
                str(job_payload.get("micro_exit", "")),
                str(job_payload.get("group_id", "")),
            ]
        )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]


def build_standardized_result(
    *,
    run_id: str,
    job_payload: Dict[str, Any],
    executor_result: Dict[str, Any],
    recovery_status: str,
    started_at_utc: str,
    finished_at_utc: str,
    duration_sec: float,
) -> Dict[str, Any]:
    status = str(executor_result.get("status", "failed")).strip().lower()
    if status not in {"success", "failed", "rejected", "missing_feature"}:
        status = "failed"

    return {
        "result_key": deterministic_result_key(job_payload),
        "run_id": run_id,
        "job_id": str(job_payload.get("job_id", "")),
        "group_id": str(job_payload.get("group_id", "")),
        "timeframe": str(job_payload.get("timeframe", "")),
        "strategy": str(job_payload.get("strategy", "")),
        "entry": str(job_payload.get("entry", "")),
        "micro_exit": str(job_payload.get("micro_exit", "")),
        "status": status,
        "reject_reason": str(executor_result.get("reject_reason", "")),
        "error_reason": str(executor_result.get("error_reason", "")),
        "missing_feature_reason": str(executor_result.get("missing_feature_reason", "")),
        "recovery_status": recovery_status,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "duration_sec": round(float(duration_sec), 6),
        "metrics_json": json.dumps(executor_result.get("metrics", {}), ensure_ascii=False, sort_keys=True),
    }


class StandardizedResultPackWriter:
    def __init__(self, outdir: str | Path) -> None:
        self.outdir = Path(outdir)
        self.results_dir = self.outdir / "results"
        self.summary_dir = self.outdir / "summary"
        self.results_jsonl = self.results_dir / "unit_results.jsonl"
        self.results_csv = self.results_dir / "unit_results.csv"
        self.execution_summary_json = self.summary_dir / "execution_summary.json"
        self.recovery_summary_json = self.summary_dir / "recovery_summary.json"
        self.reject_reason_summary_json = self.summary_dir / "reject_reason_summary.json"
        self.error_reason_summary_json = self.summary_dir / "error_reason_summary.json"
        self.missing_feature_summary_json = self.summary_dir / "missing_feature_summary.json"

    def ensure_layout(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        if not self.results_csv.exists():
            self._rewrite_csv([])

    def append_result(self, row: Dict[str, Any]) -> None:
        self.ensure_layout()
        with self.results_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        rows = self.read_all_results()
        rows.append(row)
        self._rewrite_csv(rows)

    def read_all_results(self) -> List[Dict[str, Any]]:
        if not self.results_jsonl.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with self.results_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def write_summaries(self, state: Dict[str, Any], rows: Iterable[Dict[str, Any]]) -> None:
        rows = list(rows)
        reject_reason_counts: Dict[str, int] = {}
        error_reason_counts: Dict[str, int] = {}
        missing_feature_counts: Dict[str, int] = {}

        for row in rows:
            if row["reject_reason"]:
                reject_reason_counts[row["reject_reason"]] = reject_reason_counts.get(row["reject_reason"], 0) + 1
            if row["error_reason"]:
                error_reason_counts[row["error_reason"]] = error_reason_counts.get(row["error_reason"], 0) + 1
            if row["missing_feature_reason"]:
                missing_feature_counts[row["missing_feature_reason"]] = (
                    missing_feature_counts.get(row["missing_feature_reason"], 0) + 1
                )

        self._write_json(self.execution_summary_json, state.get("progress", {}))
        self._write_json(self.recovery_summary_json, state.get("recovery", {}))
        self._write_json(self.reject_reason_summary_json, reject_reason_counts)
        self._write_json(self.error_reason_summary_json, error_reason_counts)
        self._write_json(self.missing_feature_summary_json, missing_feature_counts)

    def _rewrite_csv(self, rows: List[Dict[str, Any]]) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(self.results_dir), encoding="utf-8", newline="") as tmp:
            writer = csv.DictWriter(tmp, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({col: row.get(col, "") for col in RESULT_COLUMNS})
            tmp_path = Path(tmp.name)
        atomic_replace(tmp_path, self.results_csv)

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)
        atomic_replace(tmp_path, path)

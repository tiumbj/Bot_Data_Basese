If acceptance test fails, Gate 4 remains OPEN.
"""

execution_state_store = '''"""
execution_state_store_v1_0_0.py
Version: v1.0.0
Purpose:
Persistent execution state for resume-safe Gate 4 runner.
"""

from future import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

SCHEMA_VERSION = "v1.0.0"

def utc_now_iso() -> str:
return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding=encoding) as tmp:
tmp.write(text)
tmp_path = Path(tmp.name)
os.replace(tmp_path, path)

def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))

def load_json_or_default(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
if not path.exists():
return default
return json.loads(path.read_text(encoding="utf-8"))

def file_sha256(path: Path) -> str:
h = hashlib.sha256()
with path.open("rb") as f:
for chunk in iter(lambda: f.read(1024 * 1024), b""):
h.update(chunk)
return h.hexdigest()

@dataclass(frozen=True)
class RunIdentity:
runner_version: str
manifest_path: str
manifest_hash: str
outdir: str
shard_index: int
shard_count: int
executor_ref: str

class ExecutionStateStore:
def init(self, outdir: str | Path) -> None:
self.outdir = Path(outdir)
self.state_dir = self.outdir / "state"
self.state_path = self.state_dir / "execution_state.json"
self.write_audit_path = self.state_dir / "write_audit.json"
self.completed_job_ids_path = self.state_dir / "completed_job_ids.jsonl"
self.completed_group_ids_path = self.state_dir / "completed_group_ids.jsonl"

@staticmethod
def build_default_state(run_identity: RunIdentity, total_jobs: int) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_meta": {
            "run_id": hashlib.sha256(
                (
                    f"{run_identity.runner_version}|{run_identity.manifest_hash}|"
                    f"{run_identity.outdir}|{run_identity.shard_index}|{run_identity.shard_count}|"
                    f"{run_identity.executor_ref}"
                ).encode("utf-8")
            ).hexdigest()[:24],
            "runner_version": run_identity.runner_version,
            "manifest_path": run_identity.manifest_path,
            "manifest_hash": run_identity.manifest_hash,
            "outdir": run_identity.outdir,
            "shard_index": run_identity.shard_index,
            "shard_count": run_identity.shard_count,
            "executor_ref": run_identity.executor_ref,
            "created_at_utc": utc_now_iso(),
            "last_checkpoint_at_utc": utc_now_iso(),
        },
        "recovery": {
            "recovery_status": "fresh_start",
            "recovery_start_time_utc": utc_now_iso(),
            "resumed_jobs": 0,
            "resumed_groups": 0,
            "skipped_completed_jobs": 0,
            "skipped_completed_groups": 0,
        },
        "progress": {
            "total_jobs": total_jobs,
            "completed_jobs": 0,
            "success_jobs": 0,
            "failed_jobs": 0,
            "rejected_jobs": 0,
            "missing_feature_jobs": 0,
        },
        "completed_sets_counts": {
            "completed_job_ids": 0,
            "completed_group_ids": 0,
            "written_result_keys": 0,
        },
    }

def initialize_or_resume(self, run_identity: RunIdentity, total_jobs: int) -> Dict[str, Any]:
    state = load_json_or_default(
        self.state_path,
        self.build_default_state(run_identity=run_identity, total_jobs=total_jobs),
    )
    if self.state_path.exists():
        self._validate_run_identity(state, run_identity)
        state["recovery"]["recovery_status"] = "resumed"
        state["recovery"]["recovery_start_time_utc"] = utc_now_iso()
        state["recovery"]["resumed_jobs"] = self._count_jsonl_lines(self.completed_job_ids_path)
        state["recovery"]["resumed_groups"] = self._count_jsonl_lines(self.completed_group_ids_path)
    self._touch_checkpoint(state)
    self.save_state(state)
    self._ensure_write_audit()
    return state

def save_state(self, state: Dict[str, Any]) -> None:
    self._touch_checkpoint(state)
    atomic_write_json(self.state_path, state)

def load_state(self) -> Dict[str, Any]:
    return load_json_or_default(self.state_path, {})

def _ensure_write_audit(self) -> None:
    if not self.write_audit_path.exists():
        atomic_write_json(self.write_audit_path, {"written_result_keys": []})

def load_written_result_keys(self) -> Set[str]:
    self._ensure_write_audit()
    payload = load_json_or_default(self.write_audit_path, {"written_result_keys": []})
    return set(payload.get("written_result_keys", []))

def save_written_result_keys(self, keys: Set[str]) -> None:
    atomic_write_json(self.write_audit_path, {"written_result_keys": sorted(keys)})

def load_completed_job_ids(self) -> Set[str]:
    return self._load_jsonl_set(self.completed_job_ids_path)

def load_completed_group_ids(self) -> Set[str]:
    return self._load_jsonl_set(self.completed_group_ids_path)

def append_completed_job_id(self, job_id: str) -> None:
    self._append_jsonl_unique(self.completed_job_ids_path, job_id)

def append_completed_group_id(self, group_id: str) -> None:
    self._append_jsonl_unique(self.completed_group_ids_path, group_id)

def update_progress_counters(
    self,
    state: Dict[str, Any],
    status: str,
    completed_job_ids_count: int,
    completed_group_ids_count: int,
    written_result_keys_count: int,
    skipped_completed_jobs: Optional[int] = None,
    skipped_completed_groups: Optional[int] = None,
) -> Dict[str, Any]:
    state["progress"]["completed_jobs"] = completed_job_ids_count
    state["completed_sets_counts"]["completed_job_ids"] = completed_job_ids_count
    state["completed_sets_counts"]["completed_group_ids"] = completed_group_ids_count
    state["completed_sets_counts"]["written_result_keys"] = written_result_keys_count

    if skipped_completed_jobs is not None:
        state["recovery"]["skipped_completed_jobs"] = skipped_completed_jobs
    if skipped_completed_groups is not None:
        state["recovery"]["skipped_completed_groups"] = skipped_completed_groups

    status_field_map = {
        "success": "success_jobs",
        "failed": "failed_jobs",
        "rejected": "rejected_jobs",
        "missing_feature": "missing_feature_jobs",
    }
    if status in status_field_map:
        state["progress"][status_field_map[status]] += 1
    return state

def _validate_run_identity(self, state: Dict[str, Any], run_identity: RunIdentity) -> None:
    run_meta = state.get("run_meta", {})
    checks = {
        "manifest_hash": run_identity.manifest_hash,
        "outdir": run_identity.outdir,
        "shard_index": run_identity.shard_index,
        "shard_count": run_identity.shard_count,
        "executor_ref": run_identity.executor_ref,
    }
    mismatches = []
    for key, expected in checks.items():
        actual = run_meta.get(key)
        if actual != expected:
            mismatches.append(f"{key}: expected={expected!r}, actual={actual!r}")
    if mismatches:
        raise RuntimeError(
            "Resume-safe validation failed. Same-outdir restart requires same run identity. "
            + "; ".join(mismatches)
        )

def _touch_checkpoint(self, state: Dict[str, Any]) -> None:
    state.setdefault("run_meta", {})
    state["run_meta"]["last_checkpoint_at_utc"] = utc_now_iso()

@staticmethod
def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

@staticmethod
def _load_jsonl_set(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    items: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.add(json.loads(line)["value"])
    return items

@staticmethod
def _append_jsonl_unique(path: Path, value: str) -> None:
    existing = ExecutionStateStore._load_jsonl_set(path)
    if value in existing:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"value": value}, ensure_ascii=False) + "\\n")

        
# ============================================================
# ชื่อโค้ด: acceptance_test_gate4_multistatus_v1_0_0.py
# เวอร์ชัน: v1.0.0
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\acceptance_test_gate4_multistatus_v1_0_0.py
# คำสั่งรัน:
# python C:\Data\Bot\Local_LLM\gold_research\jobs\acceptance_test_gate4_multistatus_v1_0_0.py
# ============================================================

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


VERSION = "v1.0.0"
JOBS_DIR = Path(r"C:\Data\Bot\Local_LLM\gold_research\jobs")
OUTROOT = Path(r"C:\Data\Bot\central_backtest_results\gate4_multistatus_acceptance_v1_0_0")
MANIFEST_PATH = OUTROOT / "gate4_multistatus_manifest.jsonl"
OUTDIR = OUTROOT / "run_outdir"
REPORT_PATH = OUTROOT / "acceptance_report.json"

RUNNER_PATH = JOBS_DIR / "resume_safe_execution_runner_v1_0_0.py"
STATE_STORE_PATH = JOBS_DIR / "execution_state_store_v1_0_0.py"
RESULT_PACK_PATH = JOBS_DIR / "standardized_result_pack_v1_0_0.py"

MOCK_EXECUTOR_FILENAME = "mock_gate4_multistatus_executor_v1_0_0.py"
MOCK_EXECUTOR_PATH = JOBS_DIR / MOCK_EXECUTOR_FILENAME
MOCK_EXECUTOR_REF = "mock_gate4_multistatus_executor_v1_0_0:run_job"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _assert_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def _build_manifest() -> None:
    OUTROOT.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "job_id": "gate4_success_001",
            "group_id": "group_success",
            "timeframe": "M1",
            "strategy": "pullback_deep",
            "entry": "bos_choch_atr_adx_ema",
            "micro_exit": "micro_exit_v2_fast_invalidation",
            "mock_mode": "success",
        },
        {
            "job_id": "gate4_failed_001",
            "group_id": "group_failed",
            "timeframe": "M2",
            "strategy": "pullback_deep",
            "entry": "bos_choch_atr_adx_ema",
            "micro_exit": "micro_exit_v2_fast_invalidation",
            "mock_mode": "failed",
        },
        {
            "job_id": "gate4_rejected_001",
            "group_id": "group_rejected",
            "timeframe": "M3",
            "strategy": "pullback_deep",
            "entry": "bos_choch_atr_adx_ema",
            "micro_exit": "micro_exit_v2_fast_invalidation",
            "mock_mode": "rejected",
        },
        {
            "job_id": "gate4_missing_feature_001",
            "group_id": "group_missing_feature",
            "timeframe": "M4",
            "strategy": "pullback_deep",
            "entry": "bos_choch_atr_adx_ema",
            "micro_exit": "micro_exit_v2_fast_invalidation",
            "mock_mode": "missing_feature",
        },
    ]

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        for row in jobs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_mock_executor() -> None:
    mock_code = '''
from __future__ import annotations

from typing import Any, Dict

def run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(job.get("mock_mode", "success")).strip().lower()

    if mode == "success":
        return {
            "status": "success",
            "metrics": {
                "trade_count": 10,
                "pnl_sum": 25.5,
                "win_rate_pct": 60.0,
            },
            "reject_reason": "",
            "error_reason": "",
            "missing_feature_reason": "",
        }

    if mode == "failed":
        return {
            "status": "failed",
            "metrics": {},
            "reject_reason": "",
            "error_reason": "SIMULATED_FAILURE",
            "missing_feature_reason": "",
        }

    if mode == "rejected":
        return {
            "status": "rejected",
            "metrics": {},
            "reject_reason": "SIMULATED_RULE_REJECT",
            "error_reason": "",
            "missing_feature_reason": "",
        }

    if mode == "missing_feature":
        return {
            "status": "missing_feature",
            "metrics": {},
            "reject_reason": "",
            "error_reason": "",
            "missing_feature_reason": "SIMULATED_FEATURE_CACHE_MISSING",
        }

    return {
        "status": "failed",
        "metrics": {},
        "reject_reason": "",
        "error_reason": f"UNKNOWN_MODE:{mode}",
        "missing_feature_reason": "",
    }
'''
    MOCK_EXECUTOR_PATH.write_text(mock_code.strip() + "\n", encoding="utf-8")


def _run_cmd(cmd: List[str]) -> Dict[str, Any]:
    result = subprocess.run(cmd, cwd=str(JOBS_DIR), capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _validate_runner_files() -> None:
    _assert_file_exists(RUNNER_PATH)
    _assert_file_exists(STATE_STORE_PATH)
    _assert_file_exists(RESULT_PACK_PATH)


def _validate_first_run(state: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, bool]:
    progress = state.get("progress", {})
    recovery = state.get("recovery", {})
    return {
        "first_run_completed_all_4_jobs": progress.get("completed_jobs") == 4,
        "first_run_success_count_ok": progress.get("success_jobs") == 1,
        "first_run_failed_count_ok": progress.get("failed_jobs") == 1,
        "first_run_rejected_count_ok": progress.get("rejected_jobs") == 1,
        "first_run_missing_feature_count_ok": progress.get("missing_feature_jobs") == 1,
        "first_run_fresh_start": recovery.get("recovery_status") == "fresh_start",
        "first_run_resumed_jobs_zero": recovery.get("resumed_jobs") == 0,
        "first_run_skipped_jobs_zero": recovery.get("skipped_completed_jobs") == 0,
        "first_run_result_rows_4": len(rows) == 4,
    }


def _validate_second_run(state: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, bool]:
    progress = state.get("progress", {})
    recovery = state.get("recovery", {})
    return {
        "second_run_completed_still_4_jobs": progress.get("completed_jobs") == 4,
        "second_run_counts_not_incremented_wrongly": (
            progress.get("success_jobs") == 1
            and progress.get("failed_jobs") == 1
            and progress.get("rejected_jobs") == 1
            and progress.get("missing_feature_jobs") == 1
        ),
        "second_run_resumed_mode": recovery.get("recovery_status") == "resumed",
        "second_run_resumed_jobs_4": recovery.get("resumed_jobs") == 4,
        "second_run_skipped_jobs_4": recovery.get("skipped_completed_jobs") == 4,
        "second_run_result_rows_still_4": len(rows) == 4,
    }


def _validate_reason_summaries() -> Dict[str, bool]:
    error_summary = _read_json(OUTDIR / "summary" / "error_reason_summary.json")
    reject_summary = _read_json(OUTDIR / "summary" / "reject_reason_summary.json")
    missing_feature_summary = _read_json(OUTDIR / "summary" / "missing_feature_summary.json")
    return {
        "error_reason_summary_ok": error_summary.get("SIMULATED_FAILURE") == 1,
        "reject_reason_summary_ok": reject_summary.get("SIMULATED_RULE_REJECT") == 1,
        "missing_feature_summary_ok": missing_feature_summary.get("SIMULATED_FEATURE_CACHE_MISSING") == 1,
    }


def main() -> int:
    _validate_runner_files()

    if OUTROOT.exists():
        shutil.rmtree(OUTROOT)
    OUTROOT.mkdir(parents=True, exist_ok=True)

    _build_manifest()
    _build_mock_executor()

    first_cmd = [
        sys.executable,
        str(RUNNER_PATH),
        "--manifest",
        str(MANIFEST_PATH),
        "--outdir",
        str(OUTDIR),
        "--executor",
        MOCK_EXECUTOR_REF,
        "--shard-index",
        "0",
        "--shard-count",
        "1",
    ]
    first_run = _run_cmd(first_cmd)
    if first_run["returncode"] != 0:
        _write_json(REPORT_PATH, {"version": VERSION, "status": "FAILED_FIRST_RUN", "first_run": first_run})
        print(first_run["stdout"])
        print(first_run["stderr"])
        return 1

    state_after_first = _read_json(OUTDIR / "state" / "execution_state.json")
    rows_after_first = _load_jsonl(OUTDIR / "results" / "unit_results.jsonl")

    second_run = _run_cmd(first_cmd)
    if second_run["returncode"] != 0:
        _write_json(
            REPORT_PATH,
            {"version": VERSION, "status": "FAILED_SECOND_RUN", "first_run": first_run, "second_run": second_run},
        )
        print(second_run["stdout"])
        print(second_run["stderr"])
        return 1

    state_after_second = _read_json(OUTDIR / "state" / "execution_state.json")
    rows_after_second = _load_jsonl(OUTDIR / "results" / "unit_results.jsonl")

    checks: Dict[str, bool] = {}
    checks.update(_validate_first_run(state_after_first, rows_after_first))
    checks.update(_validate_second_run(state_after_second, rows_after_second))
    checks.update(_validate_reason_summaries())

    execution_summary = _read_json(OUTDIR / "summary" / "execution_summary.json")
    recovery_summary = _read_json(OUTDIR / "summary" / "recovery_summary.json")
    write_audit = _read_json(OUTDIR / "state" / "write_audit.json")

    checks["execution_summary_completed_jobs_ok"] = execution_summary.get("completed_jobs") == 4
    checks["recovery_summary_skipped_jobs_ok"] = recovery_summary.get("skipped_completed_jobs") == 4
    checks["write_audit_key_count_ok"] = len(write_audit.get("written_result_keys", [])) == 4
    checks["completed_job_ids_count_ok"] = _count_jsonl(OUTDIR / "state" / "completed_job_ids.jsonl") == 4

    report = {
        "version": VERSION,
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "manifest_path": str(MANIFEST_PATH),
        "outdir": str(OUTDIR),
        "first_run": first_run,
        "second_run": second_run,
        "state_after_first": state_after_first,
        "state_after_second": state_after_second,
    }
    _write_json(REPORT_PATH, report)

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] manifest={MANIFEST_PATH}")
    print(f"[DONE] outdir={OUTDIR}")
    print(f"[DONE] report={REPORT_PATH}")
    print(f"[DONE] status={report['status']}")
    for key, value in checks.items():
        print(f"[CHECK] {key}={value}")
    print("=" * 120)

    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

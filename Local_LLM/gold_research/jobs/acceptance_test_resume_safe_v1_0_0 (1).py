"""
acceptance_test_resume_safe_v1_0_0.py
Version: v1.0.0
Purpose:
    Gate 4 acceptance test for same-outdir resume-safe runner.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


TEST_VERSION = "v1.0.0"


def build_test_manifest(path: Path, count: int) -> None:
    rows: List[Dict[str, Any]] = []
    for i in range(count):
        rows.append(
            {
                "job_id": f"job_{i:04d}",
                "group_id": f"group_{i//2:03d}",
                "timeframe": "M1",
                "strategy": "pullback_deep",
                "entry": "bos_choch_atr_adx_ema",
                "micro_exit": "micro_exit_v2_fast_invalidation",
                "payload_index": i,
            }
        )
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_mock_executor_file(path: Path) -> None:
    code = """
from __future__ import annotations

import time
from typing import Any, Dict

def run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    idx = int(job.get("payload_index", 0))
    time.sleep(0.01)
    if idx % 11 == 0 and idx > 0:
        return {
            "status": "rejected",
            "metrics": {"payload_index": idx},
            "reject_reason": "RULE_REJECT_SAMPLE",
            "error_reason": "",
            "missing_feature_reason": "",
        }
    if idx % 13 == 0 and idx > 0:
        return {
            "status": "missing_feature",
            "metrics": {"payload_index": idx},
            "reject_reason": "",
            "error_reason": "",
            "missing_feature_reason": "MISSING_FEATURE_SAMPLE",
        }
    return {
        "status": "success",
        "metrics": {"payload_index": idx, "pnl_sum": idx * 1.25},
        "reject_reason": "",
        "error_reason": "",
        "missing_feature_reason": "",
    }
"""
    path.write_text(code.strip() + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run_cmd(cmd: List[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Acceptance test for resume-safe runner")
    parser.add_argument("--workdir", required=True, help="Temporary test directory")
    parser.add_argument("--job-count", type=int, default=12, help="Number of test jobs")
    parser.add_argument("--partial-stop", type=int, default=5, help="First pass intentional partial stop")
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    manifest_path = workdir / "test_manifest.jsonl"
    outdir = workdir / "run_outdir"
    mock_executor_path = workdir / "mock_executor_for_gate4_v1_0_0.py"
    runner_src = Path(__file__).resolve().parent / "resume_safe_execution_runner_v1_0_0.py"
    state_store_src = Path(__file__).resolve().parent / "execution_state_store_v1_0_0.py"
    result_pack_src = Path(__file__).resolve().parent / "standardized_result_pack_v1_0_0.py"

    build_test_manifest(manifest_path, args.job_count)
    build_mock_executor_file(mock_executor_path)

    shutil.copy2(runner_src, workdir / runner_src.name)
    shutil.copy2(state_store_src, workdir / state_store_src.name)
    shutil.copy2(result_pack_src, workdir / result_pack_src.name)

    first_cmd = [
        sys.executable,
        str(workdir / "resume_safe_execution_runner_v1_0_0.py"),
        "--manifest", str(manifest_path),
        "--outdir", str(outdir),
        "--executor", "mock_executor_for_gate4_v1_0_0:run_job",
        "--shard-index", "0",
        "--shard-count", "1",
        "--stop-after-n", str(args.partial_stop),
    ]
    run_cmd(first_cmd, cwd=workdir)

    state_after_first = read_json(outdir / "state" / "execution_state.json")
    first_result_count = count_jsonl(outdir / "results" / "unit_results.jsonl")

    second_cmd = [
        sys.executable,
        str(workdir / "resume_safe_execution_runner_v1_0_0.py"),
        "--manifest", str(manifest_path),
        "--outdir", str(outdir),
        "--executor", "mock_executor_for_gate4_v1_0_0:run_job",
        "--shard-index", "0",
        "--shard-count", "1",
    ]
    run_cmd(second_cmd, cwd=workdir)

    state_after_second = read_json(outdir / "state" / "execution_state.json")
    second_result_count = count_jsonl(outdir / "results" / "unit_results.jsonl")

    checks = {
        "partial_run_completed_some_jobs": state_after_first["progress"]["completed_jobs"] > 0,
        "first_pass_result_rows_match_completed_jobs": first_result_count == state_after_first["progress"]["completed_jobs"],
        "resumed_mode_detected": state_after_second["recovery"]["recovery_status"] == "resumed",
        "resumed_jobs_reported": state_after_second["recovery"]["resumed_jobs"] >= state_after_first["progress"]["completed_jobs"],
        "progress_did_not_reset": state_after_second["progress"]["completed_jobs"] >= state_after_first["progress"]["completed_jobs"],
        "final_completed_jobs_match_job_count": state_after_second["progress"]["completed_jobs"] == args.job_count,
        "result_rows_not_duplicated": second_result_count == args.job_count,
        "skipped_completed_jobs_reported": state_after_second["recovery"]["skipped_completed_jobs"] >= state_after_first["progress"]["completed_jobs"],
    }

    report_path = workdir / "acceptance_report.json"
    report = {
        "version": TEST_VERSION,
        "manifest_path": str(manifest_path),
        "outdir": str(outdir),
        "checks": checks,
        "passed": all(checks.values()),
        "state_after_first": state_after_first,
        "state_after_second": state_after_second,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 100)
    print(f"[DONE] version={TEST_VERSION}")
    print(f"[DONE] report={report_path}")
    print(f"[DONE] passed={report['passed']}")
    for key, value in checks.items():
        print(f"[CHECK] {key}={value}")
    print("=" * 100)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

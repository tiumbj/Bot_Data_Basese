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

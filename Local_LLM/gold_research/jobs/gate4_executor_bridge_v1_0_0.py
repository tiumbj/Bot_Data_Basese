# ============================================================
# ชื่อโค้ด: gate4_executor_bridge_v1_0_0.py
# เวอร์ชัน: v1.0.0
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\gate4_executor_bridge_v1_0_0.py
# คำสั่งรันร่วมกับ runner:
# python C:\Data\Bot\Local_LLM\gold_research\jobs\resume_safe_execution_runner_v1_0_0.py ^
#   --manifest C:\Data\Bot\central_backtest_results\truth_gate_seed_manifest.jsonl ^
#   --outdir C:\Data\Bot\central_backtest_results\gate4_promoted_runner_run ^
#   --executor gate4_executor_bridge_v1_0_0:run_job ^
#   --shard-index 0 ^
#   --shard-count 1
#
# เป้าหมาย:
# - ใช้ execution logic เดิมจาก run_research_jobs.py ต่อ
# - แปลงของเดิมให้เป็น executor มาตรฐานแบบ module:function
# - ให้ runner ใหม่เรียกใช้ได้โดยไม่ต้องเขียน backtest logic ใหม่ทั้งก้อน
#
# changelog:
# - v1.0.0
#   - เพิ่ม bridge executor มาตรฐานสำหรับ Gate 4
#   - reuse run_research_jobs.execute_job(...) เป็นแกนหลัก
#   - เติม default artifact paths เมื่อ job ไม่มีให้
#   - เติม default dataset path จาก environment ได้
#   - map summary/status เดิม -> success / failed / rejected / missing_feature
# ============================================================

from __future__ import annotations

import copy
import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict

from run_research_jobs import (
    VERSION as LEGACY_RUNNER_VERSION,
    execute_job,
    normalize_job_for_executor,
)


BRIDGE_VERSION = "v1.0.0"
DEFAULT_SYMBOL = "XAUUSD"
DEFAULT_ALLOW_STUB_FAMILIES = False

# ใช้ environment variable นี้เพื่อช่วยเติม dataset path ให้ job ที่ยังไม่มี dataset.ohlc_csv
# ตัวอย่าง:
# set GATE4_DEFAULT_OHLC_ROOT=C:\Data\Bot\central_market_data\tf
DEFAULT_OHLC_ROOT_ENV = "GATE4_DEFAULT_OHLC_ROOT"

# ใช้ environment variable นี้เพื่อกำหนด root ของ artifact ถ้า manifest job ยังไม่มี artifact_paths
# ตัวอย่าง:
# set GATE4_BRIDGE_ARTIFACT_ROOT=C:\Data\Bot\central_backtest_results\gate4_bridge_artifacts
DEFAULT_ARTIFACT_ROOT_ENV = "GATE4_BRIDGE_ARTIFACT_ROOT"


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _derive_result_key(job: Dict[str, Any]) -> str:
    if _safe_str(job.get("job_id")):
        base = _safe_str(job["job_id"])
    else:
        base = "|".join(
            [
                _safe_str(job.get("timeframe")),
                _safe_str(job.get("stage")),
                _safe_str(job.get("family_id")),
                _safe_str(job.get("entry_style")),
                _safe_str(job.get("strategy_family")),
                _safe_str(job.get("entry_logic")),
                _safe_str(job.get("micro_exit")),
            ]
        )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]


def _default_artifact_root() -> Path:
    env_value = _safe_str(os.environ.get(DEFAULT_ARTIFACT_ROOT_ENV))
    if env_value:
        return Path(env_value)
    return Path.cwd() / "_gate4_bridge_artifacts"


def _build_default_artifact_paths(job: Dict[str, Any]) -> Dict[str, str]:
    job_id = _safe_str(job.get("job_id")) or _derive_result_key(job)
    result_key = _derive_result_key(job)

    root = _default_artifact_root()
    job_result_dir = root / result_key
    _ensure_dir(job_result_dir)

    return {
        "job_result_dir": str(job_result_dir),
        "summary_json": str(job_result_dir / "summary.json"),
        "trades_csv": str(job_result_dir / "trades.csv"),
        "equity_curve_csv": str(job_result_dir / "equity_curve.csv"),
        "debug_json": str(job_result_dir / "debug.json"),
        "job_id": job_id,
    }


def _merge_artifact_paths(job: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(job)
    default_paths = _build_default_artifact_paths(merged)

    artifact_paths = merged.get("artifact_paths")
    if not isinstance(artifact_paths, dict):
        artifact_paths = {}

    merged_paths = dict(default_paths)
    for key, value in artifact_paths.items():
        if _safe_str(value):
            merged_paths[key] = str(value)

    merged["artifact_paths"] = merged_paths
    return merged


def _try_fill_default_dataset(job: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(job)
    dataset = merged.get("dataset")
    if not isinstance(dataset, dict):
        dataset = {}
    merged["dataset"] = dataset

    if _safe_str(dataset.get("ohlc_csv")):
        return merged

    tf = _safe_str(merged.get("timeframe")).upper()
    env_root = _safe_str(os.environ.get(DEFAULT_OHLC_ROOT_ENV))
    if not env_root or not tf:
        return merged

    candidate = Path(env_root) / f"{DEFAULT_SYMBOL}_{tf}.csv"
    if candidate.exists():
        dataset["ohlc_csv"] = str(candidate)
    return merged


def _prepare_job(job: Dict[str, Any]) -> Dict[str, Any]:
    prepared = copy.deepcopy(job)

    if not _safe_str(prepared.get("job_id")):
        prepared["job_id"] = _derive_result_key(prepared)

    if not _safe_str(prepared.get("symbol")):
        prepared["symbol"] = DEFAULT_SYMBOL

    if "stage" not in prepared and "stage_name" in prepared:
        prepared["stage"] = prepared["stage_name"]

    prepared = _merge_artifact_paths(prepared)
    prepared = _try_fill_default_dataset(prepared)

    return prepared


def _collect_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "trade_count",
        "total_trades",
        "wins",
        "losses",
        "pnl_sum",
        "net_profit",
        "profit_factor",
        "expectancy",
        "expectancy_r",
        "max_drawdown",
        "max_drawdown_pct",
        "win_rate_pct",
        "avg_win",
        "avg_loss",
        "payoff_ratio",
        "sharpe",
        "sortino",
    ]
    metrics: Dict[str, Any] = {}

    for key in keys:
        if key in summary:
            metrics[key] = summary.get(key)

    if "job_id" in summary:
        metrics["job_id"] = summary.get("job_id")
    if "status" in summary:
        metrics["legacy_status"] = summary.get("status")
    if "summary_json" in summary:
        metrics["summary_json"] = summary.get("summary_json")
    return metrics


def _detect_missing_feature_from_message(message: str) -> bool:
    text = _safe_str(message).lower()
    signals = [
        "missing_feature",
        "feature cache",
        "feature_cache",
        "not found",
        "no such file",
        "ohlc_csv",
        "dataset",
    ]
    return any(token in text for token in signals)


def _translate_summary(prepared_job: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    raw_status = _safe_str(summary.get("status", "done")).lower()
    skip_reason = _safe_str(summary.get("skip_reason"))
    reject_reason = _safe_str(summary.get("reject_reason"))
    error_reason = _safe_str(summary.get("error_reason"))
    missing_feature_reason = _safe_str(summary.get("missing_feature_reason"))

    translated_status = "success"

    if raw_status in {"done", "success", "completed"}:
        translated_status = "success"
    elif raw_status in {"missing_feature", "missing-feature"}:
        translated_status = "missing_feature"
    elif raw_status in {"failed", "error"}:
        translated_status = "failed"
    elif raw_status in {"skipped", "rejected", "reject"}:
        if _detect_missing_feature_from_message(skip_reason):
            translated_status = "missing_feature"
            if not missing_feature_reason:
                missing_feature_reason = skip_reason or "missing_feature_detected_from_skip_reason"
        else:
            translated_status = "rejected"
            if not reject_reason:
                reject_reason = skip_reason or "job_skipped_by_legacy_executor"
    else:
        if missing_feature_reason:
            translated_status = "missing_feature"
        elif error_reason:
            translated_status = "failed"
        elif skip_reason:
            translated_status = "rejected"
            reject_reason = reject_reason or skip_reason
        else:
            translated_status = "success"

    summary_json = prepared_job["artifact_paths"]["summary_json"]
    metrics = _collect_metrics(summary)
    metrics["bridge_version"] = BRIDGE_VERSION
    metrics["legacy_runner_version"] = LEGACY_RUNNER_VERSION
    metrics["summary_json"] = summary_json

    return {
        "status": translated_status,
        "metrics": metrics,
        "reject_reason": reject_reason,
        "error_reason": error_reason,
        "missing_feature_reason": missing_feature_reason,
    }


def run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    มาตรฐานที่ runner ใหม่ต้องการ:
        input  : job 1 dict
        output : {
            "status": "success" | "failed" | "rejected" | "missing_feature",
            "metrics": {...},
            "reject_reason": "",
            "error_reason": "",
            "missing_feature_reason": "",
        }

    การทำงาน:
    1) เตรียม job ให้พร้อมสำหรับ legacy executor
    2) เรียก run_research_jobs.execute_job(...)
    3) แปลงผลลัพธ์เก่าให้เป็นมาตรฐานใหม่
    """
    try:
        prepared_job = _prepare_job(job)
        normalized_job = normalize_job_for_executor(prepared_job)
        summary = execute_job(normalized_job, allow_stub_families=DEFAULT_ALLOW_STUB_FAMILIES)

        if not isinstance(summary, dict):
            raise TypeError("Legacy executor returned non-dict summary")

        return _translate_summary(prepared_job=normalized_job, summary=summary)

    except FileNotFoundError as exc:
        return {
            "status": "missing_feature",
            "metrics": {
                "bridge_version": BRIDGE_VERSION,
                "legacy_runner_version": LEGACY_RUNNER_VERSION,
            },
            "reject_reason": "",
            "error_reason": "",
            "missing_feature_reason": f"FileNotFoundError: {exc}",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "metrics": {
                "bridge_version": BRIDGE_VERSION,
                "legacy_runner_version": LEGACY_RUNNER_VERSION,
                "traceback": traceback.format_exc(),
            },
            "reject_reason": "",
            "error_reason": f"{type(exc).__name__}: {exc}",
            "missing_feature_reason": "",
        }


if __name__ == "__main__":
    sample_job = {
        "job_id": "sample_job_001",
        "stage": "benchmark",
        "timeframe": "M30",
        "symbol": "XAUUSD",
        "artifact_paths": {
            "job_result_dir": str((_default_artifact_root() / "sample_job_001").resolve()),
            "summary_json": str((_default_artifact_root() / "sample_job_001" / "summary.json").resolve()),
        },
        "dataset": {},
    }
    print(json.dumps(run_job(sample_job), indent=2, ensure_ascii=False))

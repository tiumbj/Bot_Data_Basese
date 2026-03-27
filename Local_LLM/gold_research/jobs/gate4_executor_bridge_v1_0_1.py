# ============================================================
# ชื่อโค้ด: gate4_executor_bridge_v1_0_1.py
# เวอร์ชัน: v1.0.1
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\gate4_executor_bridge_v1_0_1.py
# คำสั่งรันร่วมกับ runner:
# python C:\Data\Bot\Local_LLM\gold_research\jobs\resume_safe_execution_runner_v1_0_0.py ^
#   --manifest C:\Data\Bot\central_backtest_results\gate4_real_pilot_manifest_v1_0_0\real_pilot_manifest_24.jsonl ^
#   --outdir C:\Data\Bot\central_backtest_results\gate4_real_pilot_run_24_v1_0_1 ^
#   --executor gate4_executor_bridge_v1_0_1:run_job ^
#   --shard-index 0 ^
#   --shard-count 1
#
# เป้าหมาย:
# - ใช้ execution logic เดิมจาก run_research_jobs.py ต่อ
# - แปลง real pilot schema -> legacy schema ให้ครบก่อนส่งเข้า execute_job(...)
# - แก้ root cause ของ v1.0.0 ที่ล้มด้วย KeyError: 'stage'
#
# changelog:
# - v1.0.1
#   - map real job fields: strategy_family, entry_logic, regime_filter, cooldown_bars
#   - เติม stage, family_id, entry_style, dataset, variant ให้ครบ
#   - map regime_filter -> legacy regime_filter_id
#   - map timeframe -> canonical OHLC csv path อัตโนมัติ
#   - preserve artifact_paths และเติม default เมื่อจำเป็น
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


BRIDGE_VERSION = "v1.0.1"
DEFAULT_SYMBOL = "XAUUSD"
DEFAULT_ALLOW_STUB_FAMILIES = False

DEFAULT_OHLC_ROOT_ENV = "GATE4_DEFAULT_OHLC_ROOT"
DEFAULT_ARTIFACT_ROOT_ENV = "GATE4_BRIDGE_ARTIFACT_ROOT"

SUPPORTED_TFS = {"M1", "M2", "M3", "M4", "M5", "M6", "M10", "M15", "M30", "H1", "H4", "D1"}

REGIME_FILTER_MAP = {
    "off": "regime_filter_off",
    "none": "regime_filter_off",
    "trend_only": "regime_filter_trend_only",
    "trend_or_neutral": "regime_filter_trend_or_neutral",
    "regime_filter_off": "regime_filter_off",
    "regime_filter_trend_only": "regime_filter_trend_only",
    "regime_filter_trend_or_neutral": "regime_filter_trend_or_neutral",
}

STRATEGY_FAMILY_MAP = {
    "pullback_deep": "deep_pullback_continuation",
}

ENTRY_STYLE_MAP = {
    "pullback_deep": "deep",
}


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return int(default)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _derive_result_key(job: Dict[str, Any]) -> str:
    if _safe_str(job.get("job_id")):
        base = _safe_str(job["job_id"])
    else:
        base = "|".join(
            [
                _safe_str(job.get("timeframe")),
                _safe_str(job.get("strategy_family")),
                _safe_str(job.get("entry_logic")),
                _safe_str(job.get("micro_exit")),
                _safe_str(job.get("parameter_fingerprint")),
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


def _default_ohlc_root() -> Path:
    env_value = _safe_str(os.environ.get(DEFAULT_OHLC_ROOT_ENV))
    if env_value:
        return Path(env_value)
    return Path(r"C:\Data\Bot\central_market_data\tf")


def _fill_dataset(job: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(job)
    dataset = merged.get("dataset")
    if not isinstance(dataset, dict):
        dataset = {}
    merged["dataset"] = dataset

    if _safe_str(dataset.get("ohlc_csv")):
        return merged

    tf = _safe_str(merged.get("timeframe")).upper()
    symbol = _safe_str(merged.get("symbol")) or DEFAULT_SYMBOL
    if tf not in SUPPORTED_TFS:
        return merged

    candidate = _default_ohlc_root() / f"{symbol}_{tf}.csv"
    dataset["ohlc_csv"] = str(candidate)
    return merged


def _map_regime_filter_id(raw_value: Any) -> str:
    key = _safe_str(raw_value).lower()
    return REGIME_FILTER_MAP.get(key, "regime_filter_off")


def _build_variant(job: Dict[str, Any]) -> Dict[str, Any]:
    variant = job.get("variant")
    if not isinstance(variant, dict):
        variant = {}

    if not _safe_str(variant.get("micro_exit_id")):
        variant["micro_exit_id"] = _safe_str(job.get("micro_exit"))

    if variant.get("cooldown_bars") is None or _safe_str(variant.get("cooldown_bars")) == "":
        variant["cooldown_bars"] = _safe_int(job.get("cooldown_bars"), 0)

    if not _safe_str(variant.get("regime_filter_id")):
        variant["regime_filter_id"] = _map_regime_filter_id(job.get("regime_filter"))

    return variant


def _map_stage(job: Dict[str, Any]) -> str:
    stage = _safe_str(job.get("stage"))
    if stage:
        return stage

    stage_name = _safe_str(job.get("stage_name"))
    if stage_name:
        return stage_name

    # real pilot jobs are family execution jobs, not benchmark jobs
    if _safe_str(job.get("strategy_family")):
        return "family_research"

    return "family_research"


def _map_family_id(job: Dict[str, Any]) -> str:
    family_id = _safe_str(job.get("family_id"))
    if family_id:
        return family_id

    strategy_family = _safe_str(job.get("strategy_family")).lower()
    if strategy_family:
        return STRATEGY_FAMILY_MAP.get(strategy_family, strategy_family)

    return ""


def _map_entry_style(job: Dict[str, Any]) -> str:
    entry_style = _safe_str(job.get("entry_style"))
    if entry_style:
        return entry_style

    strategy_family = _safe_str(job.get("strategy_family")).lower()
    return ENTRY_STYLE_MAP.get(strategy_family, "")


def _prepare_job(job: Dict[str, Any]) -> Dict[str, Any]:
    prepared = copy.deepcopy(job)

    if not _safe_str(prepared.get("job_id")):
        prepared["job_id"] = _derive_result_key(prepared)

    if not _safe_str(prepared.get("symbol")):
        prepared["symbol"] = DEFAULT_SYMBOL

    prepared["stage"] = _map_stage(prepared)
    prepared["family_id"] = _map_family_id(prepared)
    prepared["entry_style"] = _map_entry_style(prepared)
    prepared["variant"] = _build_variant(prepared)

    # preserve real schema fields and also expose normalized aliases
    if not _safe_str(prepared.get("entry_logic")) and _safe_str(prepared.get("entry")):
        prepared["entry_logic"] = _safe_str(prepared.get("entry"))

    if not _safe_str(prepared.get("strategy_family")) and _safe_str(prepared.get("strategy")):
        prepared["strategy_family"] = _safe_str(prepared.get("strategy"))

    prepared = _merge_artifact_paths(prepared)
    prepared = _fill_dataset(prepared)

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
        "summary_json",
    ]
    metrics: Dict[str, Any] = {}

    for key in keys:
        if key in summary:
            metrics[key] = summary.get(key)

    if "job_id" in summary:
        metrics["job_id"] = summary.get("job_id")
    if "status" in summary:
        metrics["legacy_status"] = summary.get("status")

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
        "parquet",
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
    metrics["mapped_stage"] = prepared_job.get("stage", "")
    metrics["mapped_family_id"] = prepared_job.get("family_id", "")
    metrics["mapped_entry_style"] = prepared_job.get("entry_style", "")

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
        "strategy_family": "pullback_deep",
        "entry_logic": "bos_choch_atr_adx_ema",
        "micro_exit": "micro_exit_v2_fast_invalidation",
        "regime_filter": "trend_only",
        "cooldown_bars": "0",
        "timeframe": "M1",
        "symbol": "XAUUSD",
    }
    print(json.dumps(run_job(sample_job), indent=2, ensure_ascii=False))

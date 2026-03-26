# ============================================================
# ชื่อโค้ด: build_research_evidence_truth_gate_v1_0_0.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\build_research_evidence_truth_gate_v1_0_0.py
# คำสั่งรัน: python C:\Data\Bot\Local_LLM\gold_research\jobs\build_research_evidence_truth_gate_v1_0_0.py
# เวอร์ชัน: v1.0.0
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

VERSION = "v1.0.0"

ROOT = Path(r"C:\Data\Bot")
CENTRAL_BACKTEST_RESULTS = ROOT / "central_backtest_results"
DEFAULT_OUTDIR = CENTRAL_BACKTEST_RESULTS / "research_evidence_truth_gate_v1_0_0"

LOCKED_RESEARCH_TIMEFRAMES = [
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
]

TRUTH_GATE_OUTPUTS = {
    "coverage_by_timeframe_csv": "coverage_by_timeframe.csv",
    "micro_exit_coverage_matrix_csv": "micro_exit_coverage_matrix.csv",
    "kill_stage_summary_csv": "kill_stage_summary.csv",
    "standardized_metrics_pack_csv": "standardized_metrics_pack.csv",
    "truth_gate_summary_json": "truth_gate_summary.json",
    "truth_gate_recommendation_txt": "truth_gate_recommendation.txt",
}

TIMEFRAME_PATTERN = re.compile(r"(?<![A-Z0-9])(M1|M2|M3|M4|M5|M6|M10|M15|M30|H1|H4|D1)(?![A-Z0-9])", re.IGNORECASE)

MICRO_EXIT_KEYWORDS = [
    "micro_exit",
    "fast_invalidation",
    "momentum_fade",
    "time_stop",
    "short_ema_weakness",
    "base",
]

KILL_STAGE_RELAX_HINTS = {
    "FILTER_INTERSECTION": "RELAX_FILTER_INTERSECTION",
    "REGIME_FILTER": "RELAX_REGIME_FILTER",
    "TIMEFRAME_FILTER": "RELAX_TIMEFRAME_FILTER",
    "ENTRY_FILTER": "RELAX_ENTRY_FILTER",
    "EXIT_FILTER": "RELAX_EXIT_FILTER",
    "NO_SIGNAL": "RELAX_SIGNAL_GENERATION",
    "UNSPECIFIED": "MANUAL_REVIEW_REQUIRED",
}


@dataclass
class SignalDebugSummary:
    path: Path
    debug_dir: str
    total_jobs: int
    survived_jobs: int
    survival_pct: float
    top_kill_stage: str
    stage_summary: Dict[str, Any]
    layer_failure_counts: Dict[str, Any]


@dataclass
class ProgressSnapshot:
    path: Path
    runner_type: str
    timeframe: Optional[str]
    phase: Optional[str]
    status: str
    metrics: Dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    try:
        if isinstance(value, str) and value.strip() == "":
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
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


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def normalize_tf(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in LOCKED_RESEARCH_TIMEFRAMES:
        return text
    return None


def find_timeframe_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    match = TIMEFRAME_PATTERN.search(text.upper())
    if not match:
        return None
    candidate = match.group(1).upper()
    return candidate if candidate in LOCKED_RESEARCH_TIMEFRAMES else None


def extract_timeframe_from_path(path: Path) -> Optional[str]:
    parts = [str(path)] + list(path.parts)
    for part in parts:
        tf = find_timeframe_from_text(str(part))
        if tf:
            return tf
    return None


def contains_micro_exit_text(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in MICRO_EXIT_KEYWORDS)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Research Evidence Truth Gate report pack.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=CENTRAL_BACKTEST_RESULTS,
        help="Root directory to scan for research/backtest/debug outputs.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Output directory for truth gate artifacts.",
    )
    parser.add_argument(
        "--signal-debug-root",
        type=Path,
        default=CENTRAL_BACKTEST_RESULTS / "signal_debug",
        help="Optional root directory for signal debug results.",
    )
    parser.add_argument(
        "--max-csv-rows",
        type=int,
        default=250000,
        help="Maximum CSV rows to read per file for defensive scanning.",
    )
    return parser


def scan_signal_debug_summaries(results_root: Path, signal_debug_root: Path) -> List[SignalDebugSummary]:
    candidate_paths: List[Path] = []

    if signal_debug_root.exists():
        candidate_paths.extend(signal_debug_root.rglob("signal_debug_decision_summary.json"))
    candidate_paths.extend(results_root.rglob("signal_debug_decision_summary.json"))

    unique_paths: List[Path] = []
    seen: set[str] = set()
    for path in candidate_paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)

    summaries: List[SignalDebugSummary] = []
    for path in unique_paths:
        payload = read_json(path)
        if not payload:
            continue
        summaries.append(
            SignalDebugSummary(
                path=path,
                debug_dir=str(payload.get("debug_dir", "")),
                total_jobs=safe_int(payload.get("total_jobs")),
                survived_jobs=safe_int(payload.get("survived_jobs")),
                survival_pct=safe_float(payload.get("survival_pct")),
                top_kill_stage=str(payload.get("top_kill_stage") or "UNSPECIFIED"),
                stage_summary=payload.get("stage_summary") or {},
                layer_failure_counts=payload.get("layer_failure_counts") or {},
            )
        )
    return summaries


def try_read_csv_columns(path: Path, max_rows: int) -> pd.DataFrame:
    try:
        return pd.read_csv(path, nrows=max_rows)
    except UnicodeDecodeError:
        return pd.read_csv(path, nrows=max_rows, encoding="latin1")
    except Exception:
        return pd.DataFrame()


def collect_timeframes_from_dataframe(df: pd.DataFrame, path: Path) -> List[str]:
    if df.empty:
        tf = extract_timeframe_from_path(path)
        return [tf] if tf else []

    candidate_columns = [
        "timeframe",
        "tf",
        "current_timeframe",
        "entry_timeframe",
        "deployment_timeframe",
        "research_timeframe",
        "target_timeframe",
        "window_timeframe",
    ]
    found: List[str] = []
    for col in candidate_columns:
        if col in df.columns:
            for value in df[col].dropna().astype(str).unique().tolist():
                tf = normalize_tf(value)
                if tf:
                    found.append(tf)

    if not found:
        tf = extract_timeframe_from_path(path)
        if tf:
            found.append(tf)

    return sorted(set(found))


def infer_micro_exit_presence(df: pd.DataFrame, path: Path) -> bool:
    path_text = str(path).lower()
    if contains_micro_exit_text(path_text):
        return True

    if df.empty:
        return False

    candidate_columns = [
        "exit_name",
        "exit_variant",
        "micro_exit_name",
        "micro_exit_variant",
        "winner_exit",
        "strategy_id",
        "package_id",
        "component_name",
    ]
    for col in candidate_columns:
        if col in df.columns:
            series = df[col].dropna().astype(str)
            for value in series.head(1000).tolist():
                if contains_micro_exit_text(value):
                    return True
    return False


def scan_progress_snapshots(results_root: Path) -> List[ProgressSnapshot]:
    snapshots: List[ProgressSnapshot] = []
    for path in results_root.rglob("live_progress.json"):
        payload = read_json(path)
        if not payload:
            continue
        outdir = str(payload.get("outdir", ""))
        timeframe = normalize_tf(payload.get("current_timeframe")) or extract_timeframe_from_path(path)
        runner_type = "unknown"
        lower_path = str(path).lower()
        if "intelligent_runner" in lower_path:
            runner_type = "intelligent_runner"
        elif "vectorbt" in lower_path or "coverage_master_runs" in lower_path:
            runner_type = "vectorbt_runner"
        elif "weekly" in lower_path:
            runner_type = "weekly_runner"

        status = "unknown"
        if safe_int(payload.get("execution_error")) > 0 or safe_int(payload.get("execution_error_groups")) > 0:
            status = "error"
        elif safe_int(payload.get("execution_done")) > 0 or safe_int(payload.get("execution_done_jobs")) > 0:
            status = "active_execution"
        elif safe_int(payload.get("preflight_valid")) > 0 or safe_int(payload.get("manifest_total")) > 0:
            status = "preflight_or_summary"

        snapshots.append(
            ProgressSnapshot(
                path=path,
                runner_type=runner_type,
                timeframe=timeframe,
                phase=payload.get("phase"),
                status=status,
                metrics=payload,
            )
        )
    return snapshots


def scan_summary_csvs(results_root: Path, max_rows: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    coverage_rows: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []

    target_names = {
        "summary_by_group.csv",
        "summary_by_family.csv",
        "top_surviving_jobs.csv",
        "top_surviving_groups.csv",
        "kill_stage_by_timeframe.csv",
        "coverage_results_all.csv",
        "preflight_results.csv",
        "leaderboard_done_only.jsonl.csv",
    }

    for path in results_root.rglob("*.csv"):
        if path.name not in target_names and "summary" not in path.name.lower() and "surviv" not in path.name.lower():
            continue

        df = try_read_csv_columns(path, max_rows=max_rows)
        tfs = collect_timeframes_from_dataframe(df, path)
        has_micro_exit = infer_micro_exit_presence(df, path)

        if not tfs:
            tf = extract_timeframe_from_path(path)
            tfs = [tf] if tf else []

        jobs = 0
        survived_jobs = 0
        trades = 0
        pnl_sum = 0.0
        win_rate = None

        numeric_candidates = {
            "jobs": ["jobs", "total_jobs", "execution_total", "count"],
            "survived_jobs": ["survived_jobs", "survivors", "valid_jobs", "execution_done"],
            "trades": ["trades", "trade_count", "estimated_trade_count"],
            "pnl_sum": ["pnl_sum", "net_profit", "pnl", "profit_sum"],
            "win_rate": ["win_rate_pct", "win_rate", "win_pct"],
        }

        for target_key, source_columns in numeric_candidates.items():
            value = None
            for col in source_columns:
                if col in df.columns:
                    if target_key == "win_rate":
                        value = safe_float(pd.to_numeric(df[col], errors="coerce").dropna().mean())
                    else:
                        value = safe_float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
                    break
            if target_key == "jobs":
                jobs = safe_int(value)
            elif target_key == "survived_jobs":
                survived_jobs = safe_int(value)
            elif target_key == "trades":
                trades = safe_int(value)
            elif target_key == "pnl_sum":
                pnl_sum = safe_float(value)
            elif target_key == "win_rate":
                win_rate = None if value is None else safe_float(value)

        for tf in tfs:
            if not tf:
                continue
            coverage_rows.append(
                {
                    "timeframe": tf,
                    "source_path": str(path),
                    "source_name": path.name,
                    "has_micro_exit_text": has_micro_exit,
                    "jobs": jobs,
                    "survived_jobs": survived_jobs,
                    "trades": trades,
                    "pnl_sum": pnl_sum,
                    "win_rate_pct_mean": win_rate,
                }
            )

        metrics_rows.append(
            {
                "source_path": str(path),
                "source_name": path.name,
                "timeframes_detected": ",".join([tf for tf in tfs if tf]),
                "has_micro_exit_text": has_micro_exit,
                "jobs": jobs,
                "survived_jobs": survived_jobs,
                "trades": trades,
                "pnl_sum": pnl_sum,
                "win_rate_pct_mean": win_rate,
            }
        )

    return coverage_rows, metrics_rows


def scan_json_summaries(results_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    target_names = {
        "run_summary.json",
        "preflight_summary.json",
        "summary.json",
        "decision_summary.json",
        "signal_debug_decision_summary.json",
        "progress_summary.json",
        "reject_summary.json",
    }

    for path in results_root.rglob("*.json"):
        if path.name not in target_names and "summary" not in path.name.lower():
            continue

        payload = read_json(path)
        if not payload:
            continue

        tf = normalize_tf(payload.get("timeframe")) or normalize_tf(payload.get("current_timeframe")) or extract_timeframe_from_path(path)

        rows.append(
            {
                "source_path": str(path),
                "source_name": path.name,
                "timeframe": tf,
                "phase": payload.get("phase"),
                "version": payload.get("version"),
                "manifest_total": safe_int(payload.get("manifest_total")),
                "preflight_valid": safe_int(payload.get("preflight_valid")),
                "preflight_invalid": safe_int(payload.get("preflight_invalid")),
                "preflight_unsupported": safe_int(payload.get("preflight_unsupported")),
                "preflight_skipped": safe_int(payload.get("preflight_skipped")),
                "execution_total_jobs": safe_int(payload.get("execution_total_jobs") or payload.get("execution_total")),
                "execution_done_jobs": safe_int(payload.get("execution_done_jobs") or payload.get("execution_done")),
                "execution_error_jobs": safe_int(payload.get("execution_error_jobs") or payload.get("execution_error")),
                "observed_execution_rate_jobs_per_min": safe_float(payload.get("observed_execution_rate_jobs_per_min")),
                "micro_exit_detected": contains_micro_exit_text(str(path)) or contains_micro_exit_text(json.dumps(payload, ensure_ascii=False)),
            }
        )
    return rows


def aggregate_coverage(
    coverage_rows: List[Dict[str, Any]],
    json_summary_rows: List[Dict[str, Any]],
    progress_snapshots: List[ProgressSnapshot],
) -> pd.DataFrame:
    per_tf: Dict[str, Dict[str, Any]] = {
        tf: {
            "timeframe": tf,
            "csv_sources": 0,
            "json_sources": 0,
            "progress_sources": 0,
            "jobs_sum": 0,
            "survived_jobs_sum": 0,
            "trades_sum": 0,
            "pnl_sum": 0.0,
            "mean_win_rate_pct": None,
            "micro_exit_evidence_count": 0,
            "has_any_evidence": False,
        }
        for tf in LOCKED_RESEARCH_TIMEFRAMES
    }

    win_rates_by_tf: Dict[str, List[float]] = defaultdict(list)

    for row in coverage_rows:
        tf = row.get("timeframe")
        if tf not in per_tf:
            continue
        bucket = per_tf[tf]
        bucket["csv_sources"] += 1
        bucket["jobs_sum"] += safe_int(row.get("jobs"))
        bucket["survived_jobs_sum"] += safe_int(row.get("survived_jobs"))
        bucket["trades_sum"] += safe_int(row.get("trades"))
        bucket["pnl_sum"] += safe_float(row.get("pnl_sum"))
        bucket["has_any_evidence"] = True
        if row.get("has_micro_exit_text"):
            bucket["micro_exit_evidence_count"] += 1
        if row.get("win_rate_pct_mean") is not None and not math.isnan(float(row.get("win_rate_pct_mean"))):
            win_rates_by_tf[tf].append(safe_float(row.get("win_rate_pct_mean")))

    for row in json_summary_rows:
        tf = row.get("timeframe")
        if tf not in per_tf:
            continue
        bucket = per_tf[tf]
        bucket["json_sources"] += 1
        bucket["jobs_sum"] += safe_int(row.get("execution_total_jobs"))
        bucket["survived_jobs_sum"] += safe_int(row.get("execution_done_jobs")) + safe_int(row.get("preflight_valid"))
        bucket["has_any_evidence"] = True
        if row.get("micro_exit_detected"):
            bucket["micro_exit_evidence_count"] += 1

    for snap in progress_snapshots:
        tf = snap.timeframe
        if tf not in per_tf:
            continue
        bucket = per_tf[tf]
        bucket["progress_sources"] += 1
        bucket["has_any_evidence"] = True
        if contains_micro_exit_text(str(snap.path)):
            bucket["micro_exit_evidence_count"] += 1

    rows: List[Dict[str, Any]] = []
    for tf in LOCKED_RESEARCH_TIMEFRAMES:
        bucket = per_tf[tf]
        wins = win_rates_by_tf.get(tf, [])
        mean_win = round(sum(wins) / len(wins), 4) if wins else None
        coverage_status = "MISSING"
        if bucket["has_any_evidence"]:
            coverage_status = "PRESENT"
        if bucket["survived_jobs_sum"] > 0 or bucket["trades_sum"] > 0:
            coverage_status = "STRONG"

        micro_exit_status = "MISSING"
        if bucket["micro_exit_evidence_count"] > 0:
            micro_exit_status = "PRESENT"
        if bucket["micro_exit_evidence_count"] > 1 or bucket["trades_sum"] > 0:
            micro_exit_status = "STRONG"

        rows.append(
            {
                "timeframe": tf,
                "coverage_status": coverage_status,
                "micro_exit_status": micro_exit_status,
                "csv_sources": bucket["csv_sources"],
                "json_sources": bucket["json_sources"],
                "progress_sources": bucket["progress_sources"],
                "jobs_sum": bucket["jobs_sum"],
                "survived_jobs_sum": bucket["survived_jobs_sum"],
                "trades_sum": bucket["trades_sum"],
                "pnl_sum": round(bucket["pnl_sum"], 6),
                "mean_win_rate_pct": mean_win,
                "micro_exit_evidence_count": bucket["micro_exit_evidence_count"],
                "has_any_evidence": bucket["has_any_evidence"],
            }
        )

    return pd.DataFrame(rows)


def build_micro_exit_matrix(coverage_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in coverage_df.iterrows():
        rows.append(
            {
                "timeframe": row["timeframe"],
                "micro_exit_required": True,
                "micro_exit_status": row["micro_exit_status"],
                "evidence_count": safe_int(row["micro_exit_evidence_count"]),
                "jobs_sum": safe_int(row["jobs_sum"]),
                "trades_sum": safe_int(row["trades_sum"]),
                "gate_pass": row["micro_exit_status"] != "MISSING",
            }
        )
    return pd.DataFrame(rows)


def build_kill_stage_summary(signal_summaries: List[SignalDebugSummary]) -> pd.DataFrame:
    stage_counter: Counter[str] = Counter()
    total_jobs_by_stage: Dict[str, int] = defaultdict(int)
    survived_jobs_by_stage: Dict[str, int] = defaultdict(int)
    sources_by_stage: Dict[str, List[str]] = defaultdict(list)

    if not signal_summaries:
        return pd.DataFrame(
            [
                {
                    "top_kill_stage": "NO_SIGNAL_DEBUG_FOUND",
                    "signal_debug_sources": 0,
                    "total_jobs_sum": 0,
                    "survived_jobs_sum": 0,
                    "mean_survival_pct": 0.0,
                    "recommended_action": "MANUAL_REVIEW_REQUIRED",
                }
            ]
        )

    survival_pct_by_stage: Dict[str, List[float]] = defaultdict(list)

    for summary in signal_summaries:
        stage = summary.top_kill_stage or "UNSPECIFIED"
        stage_counter[stage] += 1
        total_jobs_by_stage[stage] += summary.total_jobs
        survived_jobs_by_stage[stage] += summary.survived_jobs
        survival_pct_by_stage[stage].append(summary.survival_pct)
        sources_by_stage[stage].append(str(summary.path))

    rows: List[Dict[str, Any]] = []
    for stage, count in stage_counter.most_common():
        surv_list = survival_pct_by_stage.get(stage, [])
        mean_survival = round(sum(surv_list) / len(surv_list), 4) if surv_list else 0.0
        rows.append(
            {
                "top_kill_stage": stage,
                "signal_debug_sources": count,
                "total_jobs_sum": total_jobs_by_stage[stage],
                "survived_jobs_sum": survived_jobs_by_stage[stage],
                "mean_survival_pct": mean_survival,
                "recommended_action": KILL_STAGE_RELAX_HINTS.get(stage, "MANUAL_REVIEW_REQUIRED"),
                "source_paths": " | ".join(sources_by_stage[stage][:10]),
            }
        )

    return pd.DataFrame(rows)


def build_standardized_metrics_pack(
    coverage_df: pd.DataFrame,
    micro_exit_df: pd.DataFrame,
    kill_stage_df: pd.DataFrame,
    signal_summaries: List[SignalDebugSummary],
    json_summary_rows: List[Dict[str, Any]],
    progress_snapshots: List[ProgressSnapshot],
) -> pd.DataFrame:
    metrics: List[Dict[str, Any]] = []

    total_tfs = len(LOCKED_RESEARCH_TIMEFRAMES)
    coverage_present = safe_int((coverage_df["coverage_status"] != "MISSING").sum())
    micro_exit_present = safe_int((micro_exit_df["micro_exit_status"] != "MISSING").sum())
    strong_tfs = safe_int((coverage_df["coverage_status"] == "STRONG").sum())

    metrics.append({"metric_group": "coverage", "metric_name": "locked_timeframes_total", "metric_value": total_tfs, "unit": "count"})
    metrics.append({"metric_group": "coverage", "metric_name": "timeframes_with_any_evidence", "metric_value": coverage_present, "unit": "count"})
    metrics.append({"metric_group": "coverage", "metric_name": "timeframes_with_strong_evidence", "metric_value": strong_tfs, "unit": "count"})
    metrics.append({"metric_group": "micro_exit", "metric_name": "timeframes_with_micro_exit_evidence", "metric_value": micro_exit_present, "unit": "count"})
    metrics.append(
        {
            "metric_group": "coverage",
            "metric_name": "coverage_pct_of_locked_tf",
            "metric_value": round((coverage_present / total_tfs) * 100.0 if total_tfs > 0 else 0.0, 4),
            "unit": "pct",
        }
    )
    metrics.append(
        {
            "metric_group": "micro_exit",
            "metric_name": "micro_exit_pct_of_locked_tf",
            "metric_value": round((micro_exit_present / total_tfs) * 100.0 if total_tfs > 0 else 0.0, 4),
            "unit": "pct",
        }
    )

    total_signal_debug = len(signal_summaries)
    total_debug_jobs = sum(item.total_jobs for item in signal_summaries)
    total_debug_survivors = sum(item.survived_jobs for item in signal_summaries)
    overall_survival_pct = round((total_debug_survivors / total_debug_jobs) * 100.0, 6) if total_debug_jobs > 0 else 0.0

    metrics.append({"metric_group": "signal_debug", "metric_name": "signal_debug_sources", "metric_value": total_signal_debug, "unit": "count"})
    metrics.append({"metric_group": "signal_debug", "metric_name": "signal_debug_total_jobs", "metric_value": total_debug_jobs, "unit": "count"})
    metrics.append({"metric_group": "signal_debug", "metric_name": "signal_debug_survived_jobs", "metric_value": total_debug_survivors, "unit": "count"})
    metrics.append({"metric_group": "signal_debug", "metric_name": "signal_debug_survival_pct", "metric_value": overall_survival_pct, "unit": "pct"})

    if not kill_stage_df.empty:
        top_row = kill_stage_df.iloc[0].to_dict()
        metrics.append({"metric_group": "kill_stage", "metric_name": "top_kill_stage", "metric_value": top_row.get("top_kill_stage"), "unit": "text"})
        metrics.append({"metric_group": "kill_stage", "metric_name": "top_kill_stage_sources", "metric_value": top_row.get("signal_debug_sources"), "unit": "count"})
        metrics.append({"metric_group": "kill_stage", "metric_name": "top_kill_stage_mean_survival_pct", "metric_value": top_row.get("mean_survival_pct"), "unit": "pct"})

    total_json_sources = len(json_summary_rows)
    total_progress_sources = len(progress_snapshots)
    total_execution_done = sum(safe_int(row.get("execution_done_jobs")) for row in json_summary_rows)
    total_execution_error = sum(safe_int(row.get("execution_error_jobs")) for row in json_summary_rows)
    total_preflight_valid = sum(safe_int(row.get("preflight_valid")) for row in json_summary_rows)

    metrics.append({"metric_group": "runtime", "metric_name": "json_summary_sources", "metric_value": total_json_sources, "unit": "count"})
    metrics.append({"metric_group": "runtime", "metric_name": "progress_snapshot_sources", "metric_value": total_progress_sources, "unit": "count"})
    metrics.append({"metric_group": "runtime", "metric_name": "execution_done_jobs_sum", "metric_value": total_execution_done, "unit": "count"})
    metrics.append({"metric_group": "runtime", "metric_name": "execution_error_jobs_sum", "metric_value": total_execution_error, "unit": "count"})
    metrics.append({"metric_group": "runtime", "metric_name": "preflight_valid_sum", "metric_value": total_preflight_valid, "unit": "count"})

    return pd.DataFrame(metrics)


def build_recommendation(
    coverage_df: pd.DataFrame,
    micro_exit_df: pd.DataFrame,
    kill_stage_df: pd.DataFrame,
    signal_summaries: List[SignalDebugSummary],
) -> Tuple[str, Dict[str, Any]]:
    missing_tf = coverage_df.loc[coverage_df["coverage_status"] == "MISSING", "timeframe"].tolist()
    missing_micro_exit_tf = micro_exit_df.loc[micro_exit_df["micro_exit_status"] == "MISSING", "timeframe"].tolist()

    total_debug_jobs = sum(item.total_jobs for item in signal_summaries)
    total_debug_survivors = sum(item.survived_jobs for item in signal_summaries)
    overall_survival_pct = round((total_debug_survivors / total_debug_jobs) * 100.0, 6) if total_debug_jobs > 0 else 0.0

    top_kill_stage = "UNSPECIFIED"
    recommended_action = "MANUAL_REVIEW_REQUIRED"
    if not kill_stage_df.empty:
        first_row = kill_stage_df.iloc[0].to_dict()
        top_kill_stage = str(first_row.get("top_kill_stage") or "UNSPECIFIED")
        recommended_action = str(first_row.get("recommended_action") or "MANUAL_REVIEW_REQUIRED")

    coverage_complete = len(missing_tf) == 0
    micro_exit_complete = len(missing_micro_exit_tf) == 0
    allow_full_run = False
    decision_code = "STOP_FULL_RUN_AND_RELAX_LAYER"

    if total_debug_jobs > 0 and total_debug_survivors == 0:
        decision_code = f"STOP_FULL_RUN_AND_RELAX_LAYER_{top_kill_stage}"
        allow_full_run = False
    elif not coverage_complete:
        decision_code = "STOP_FULL_RUN_AND_FILL_TIMEFRAME_COVERAGE"
        allow_full_run = False
    elif not micro_exit_complete:
        decision_code = "STOP_FULL_RUN_AND_FILL_MICRO_EXIT_COVERAGE"
        allow_full_run = False
    else:
        decision_code = "ALLOW_VT_FULL_RUN_ON_SEED_SET"
        allow_full_run = True

    lines: List[str] = []
    lines.append("=" * 120)
    lines.append(f"[TRUTH GATE] version={VERSION}")
    lines.append(f"[TRUTH GATE] generated_at_utc={utc_now_iso()}")
    lines.append(f"[TRUTH GATE] decision_code={decision_code}")
    lines.append(f"[TRUTH GATE] allow_full_run={allow_full_run}")
    lines.append(f"[TRUTH GATE] top_kill_stage={top_kill_stage}")
    lines.append(f"[TRUTH GATE] recommended_action={recommended_action}")
    lines.append(f"[TRUTH GATE] locked_timeframes_total={len(LOCKED_RESEARCH_TIMEFRAMES)}")
    lines.append(f"[TRUTH GATE] coverage_complete={coverage_complete}")
    lines.append(f"[TRUTH GATE] micro_exit_complete={micro_exit_complete}")
    lines.append(f"[TRUTH GATE] signal_debug_total_jobs={total_debug_jobs}")
    lines.append(f"[TRUTH GATE] signal_debug_total_survivors={total_debug_survivors}")
    lines.append(f"[TRUTH GATE] signal_debug_survival_pct={overall_survival_pct}")
    lines.append(f"[TRUTH GATE] missing_timeframes={missing_tf if missing_tf else 'NONE'}")
    lines.append(f"[TRUTH GATE] missing_micro_exit_timeframes={missing_micro_exit_tf if missing_micro_exit_tf else 'NONE'}")
    lines.append("=" * 120)

    if decision_code == "ALLOW_VT_FULL_RUN_ON_SEED_SET":
        lines.append("คำตัดสิน: อนุญาตให้เปิด full-run ได้เฉพาะ seed set ที่ผ่าน gate นี้แล้ว")
    else:
        lines.append("คำตัดสิน: ห้ามเปิด full-run ใหม่ทั้งก้อน จนกว่าจะปิด gap ตาม decision_code")

    if top_kill_stage != "UNSPECIFIED":
        lines.append(f"ชั้นที่ต้องแก้ก่อน: {top_kill_stage}")
    if missing_tf:
        lines.append(f"TF ที่ยังไม่มี evidence: {', '.join(missing_tf)}")
    if missing_micro_exit_tf:
        lines.append(f"TF ที่ยังไม่มี micro exit evidence: {', '.join(missing_micro_exit_tf)}")

    detail = {
        "decision_code": decision_code,
        "allow_full_run": allow_full_run,
        "top_kill_stage": top_kill_stage,
        "recommended_action": recommended_action,
        "coverage_complete": coverage_complete,
        "micro_exit_complete": micro_exit_complete,
        "missing_timeframes": missing_tf,
        "missing_micro_exit_timeframes": missing_micro_exit_tf,
        "signal_debug_total_jobs": total_debug_jobs,
        "signal_debug_total_survivors": total_debug_survivors,
        "signal_debug_survival_pct": overall_survival_pct,
    }
    return "\n".join(lines) + "\n", detail


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    results_root: Path = args.results_root
    outdir: Path = args.outdir
    signal_debug_root: Path = args.signal_debug_root
    max_csv_rows: int = args.max_csv_rows

    ensure_dir(outdir)

    signal_summaries = scan_signal_debug_summaries(results_root=results_root, signal_debug_root=signal_debug_root)
    coverage_rows, metrics_rows = scan_summary_csvs(results_root=results_root, max_rows=max_csv_rows)
    json_summary_rows = scan_json_summaries(results_root=results_root)
    progress_snapshots = scan_progress_snapshots(results_root=results_root)

    coverage_df = aggregate_coverage(
        coverage_rows=coverage_rows,
        json_summary_rows=json_summary_rows,
        progress_snapshots=progress_snapshots,
    ).sort_values(["timeframe"]).reset_index(drop=True)

    micro_exit_df = build_micro_exit_matrix(coverage_df=coverage_df).sort_values(["timeframe"]).reset_index(drop=True)
    kill_stage_df = build_kill_stage_summary(signal_summaries=signal_summaries)
    metrics_df = build_standardized_metrics_pack(
        coverage_df=coverage_df,
        micro_exit_df=micro_exit_df,
        kill_stage_df=kill_stage_df,
        signal_summaries=signal_summaries,
        json_summary_rows=json_summary_rows,
        progress_snapshots=progress_snapshots,
    )

    recommendation_text, recommendation_detail = build_recommendation(
        coverage_df=coverage_df,
        micro_exit_df=micro_exit_df,
        kill_stage_df=kill_stage_df,
        signal_summaries=signal_summaries,
    )

    coverage_csv = outdir / TRUTH_GATE_OUTPUTS["coverage_by_timeframe_csv"]
    micro_exit_csv = outdir / TRUTH_GATE_OUTPUTS["micro_exit_coverage_matrix_csv"]
    kill_stage_csv = outdir / TRUTH_GATE_OUTPUTS["kill_stage_summary_csv"]
    metrics_csv = outdir / TRUTH_GATE_OUTPUTS["standardized_metrics_pack_csv"]
    summary_json = outdir / TRUTH_GATE_OUTPUTS["truth_gate_summary_json"]
    recommendation_txt = outdir / TRUTH_GATE_OUTPUTS["truth_gate_recommendation_txt"]

    save_dataframe(coverage_df, coverage_csv)
    save_dataframe(micro_exit_df, micro_exit_csv)
    save_dataframe(kill_stage_df, kill_stage_csv)
    save_dataframe(metrics_df, metrics_csv)

    summary_payload = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "results_root": str(results_root),
        "signal_debug_root": str(signal_debug_root),
        "locked_timeframes": LOCKED_RESEARCH_TIMEFRAMES,
        "signal_debug_sources_found": len(signal_summaries),
        "csv_metric_sources_found": len(metrics_rows),
        "json_summary_sources_found": len(json_summary_rows),
        "progress_snapshot_sources_found": len(progress_snapshots),
        "outputs": {k: str(outdir / v) for k, v in TRUTH_GATE_OUTPUTS.items()},
        "recommendation": recommendation_detail,
        "coverage_preview": coverage_df.head(20).to_dict(orient="records"),
        "kill_stage_preview": kill_stage_df.head(20).to_dict(orient="records"),
        "metrics_preview": metrics_df.head(50).to_dict(orient="records"),
    }
    write_json(summary_json, summary_payload)
    write_text(recommendation_txt, recommendation_text)

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] results_root={results_root}")
    print(f"[DONE] outdir={outdir}")
    print(f"[DONE] coverage_by_timeframe={coverage_csv}")
    print(f"[DONE] micro_exit_coverage_matrix={micro_exit_csv}")
    print(f"[DONE] kill_stage_summary={kill_stage_csv}")
    print(f"[DONE] standardized_metrics_pack={metrics_csv}")
    print(f"[DONE] truth_gate_summary={summary_json}")
    print(f"[DONE] truth_gate_recommendation={recommendation_txt}")
    print(f"[DONE] signal_debug_sources_found={len(signal_summaries)}")
    print(f"[DONE] json_summary_sources_found={len(json_summary_rows)}")
    print(f"[DONE] progress_snapshot_sources_found={len(progress_snapshots)}")
    print(f"[DONE] recommendation_code={recommendation_detail['decision_code']}")
    print("=" * 120)


if __name__ == "__main__":
    main()
"""
Gate 5 - Registry Promotion Script v1.0.1
Reads execution results from Gate 4 output (parsing metrics_json for trading metrics),
applies promotion criteria, and appends to candidate registry.

Usage:
    python promote_to_registry_v1_0_1.py --outdir <gate4_output_dir> [--dry-run]
"""

import json
import csv
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import re


def load_promotion_criteria(criteria_path: Path) -> Dict[str, Any]:
    with open(criteria_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_summary_path(metrics_json_str: str) -> Optional[Path]:
    try:
        data = json.loads(metrics_json_str)
        summary_path = data.get("summary_json", "")
        if summary_path:
            return Path(summary_path)
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def load_metrics_from_summary(summary_path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        return {
            "trade_count": metrics.get("trade_count", 0),
            "win_rate_pct": metrics.get("win_rate_pct", 0.0),
            "profit_factor": metrics.get("payoff_ratio", 0.0),
            "expectancy": metrics.get("pnl_sum", 0.0),
            "pnl_sum": metrics.get("pnl_sum", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "avg_win": metrics.get("avg_win", 0.0),
            "avg_loss": metrics.get("avg_loss", 0.0),
            "status": data.get("status", "done"),
            "timeframe": data.get("timeframe", ""),
            "family_id": data.get("family_id", ""),
            "entry_style": data.get("entry_style", ""),
            "micro_exit": data.get("micro_exit", {}).get("exit_id", ""),
            "side_policy": data.get("side_policy", {}).get("side_policy_id", "both"),
        }
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def load_execution_results_with_metrics(outdir: Path) -> List[Dict[str, Any]]:
    results = []
    unit_results_csv = outdir / "results" / "unit_results.csv"
    if not unit_results_csv.exists():
        return results

    with open(unit_results_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            summary_path = extract_summary_path(row.get("metrics_json", ""))
            if summary_path:
                metrics = load_metrics_from_summary(summary_path)
                if metrics:
                    row["_parsed_metrics"] = metrics
                    row["_has_metrics"] = True
                else:
                    row["_has_metrics"] = False
            else:
                row["_has_metrics"] = False
            results.append(row)
    return results


def load_candidates(candidates_csv: Path) -> List[Dict[str, Any]]:
    if not candidates_csv.exists():
        return []
    with open(candidates_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def check_criteria(row: Dict[str, Any], criteria: Dict[str, Any]) -> tuple[bool, List[str]]:
    reasons = []
    metrics = row.get("_parsed_metrics", {})

    if "min_trade_count" in criteria:
        trade_count = int(metrics.get("trade_count", 0))
        if trade_count < criteria["min_trade_count"]:
            reasons.append(f"trade_count_below_{criteria['min_trade_count']}")

    if "min_profit_factor" in criteria:
        pf = float(metrics.get("profit_factor", 0))
        if pf < criteria["min_profit_factor"]:
            reasons.append(f"profit_factor_below_{criteria['min_profit_factor']}")

    if "min_expectancy" in criteria:
        expectancy = float(metrics.get("expectancy", 0))
        if expectancy < criteria["min_expectancy"]:
            reasons.append(f"expectancy_below_{criteria['min_expectancy']}")

    if "execution_success" in criteria:
        status = row.get("status", "").strip().lower()
        if status != "success":
            reasons.append("execution_not_success")

    if "no_missing_features" in criteria:
        missing = row.get("missing_feature_reason", "").strip()
        if missing:
            reasons.append("missing_feature")

    passed = len(reasons) == 0
    return passed, reasons


def compute_score(metrics: Dict[str, Any], criteria: Dict[str, Any]) -> float:
    pf = float(metrics.get("profit_factor", 0))
    expectancy = float(metrics.get("expectancy", 0))
    trade_count = int(metrics.get("trade_count", 1))
    win_rate = float(metrics.get("win_rate_pct", 0))
    dd = abs(float(metrics.get("max_drawdown", 0)))

    scoring_cfg = criteria.get("scoring", {})
    pf_w = scoring_cfg.get("profit_factor_weight", 0.30)
    exp_w = scoring_cfg.get("expectancy_weight", 0.25)
    tc_w = scoring_cfg.get("trade_count_weight", 0.20)
    wr_w = scoring_cfg.get("win_rate_weight", 0.15)
    dd_w = scoring_cfg.get("drawdown_penalty_weight", 0.10)

    pf_norm = min(pf / 3.0, 1.0)
    exp_norm = min(expectancy / 1000.0, 1.0)
    tc_norm = min(trade_count / 500.0, 1.0)
    wr_norm = min(win_rate / 50.0, 1.0)
    dd_norm = min(dd / 30.0, 1.0)

    score = (pf_w * pf_norm + exp_w * exp_norm + tc_w * tc_norm +
             wr_w * wr_norm + dd_w * dd_norm) * 5.0
    return round(score, 4)


def promote(
    outdir: Path,
    registry_dir: Path,
    criteria_path: Path,
    dry_run: bool = True
) -> Dict[str, Any]:
    criteria = load_promotion_criteria(criteria_path)
    exec_results = load_execution_results_with_metrics(outdir)

    if not exec_results:
        return {"status": "no_results", "promoted": 0, "rejected": 0, "audit": []}

    candidates_csv = registry_dir / "registry_candidates.csv"
    existing = load_candidates(candidates_csv)
    existing_job_ids = {r["job_id"] for r in existing}

    exec_criteria = criteria["gate_4_execution_promotion"]["criteria"]
    promoted_rows = []
    rejected_rows = []
    audit_rows = []
    no_metrics = []

    for row in exec_results:
        job_id = row.get("job_id", "").strip()
        if not job_id:
            continue

        if not row.get("_has_metrics", False):
            no_metrics.append(job_id)
            continue

        if job_id in existing_job_ids:
            audit_rows.append({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "job_id": job_id,
                "from_stage": "execution_result",
                "to_stage": "skipped_already_exists",
                "criteria_version": criteria["promotion_criteria_version"],
                "promotion_result": "skipped",
                "rejection_reason": "already_in_registry"
            })
            continue

        passed, reasons = check_criteria(row, exec_criteria)
        timestamp = datetime.now(timezone.utc).isoformat()
        metrics = row["_parsed_metrics"]

        if passed:
            score = compute_score(metrics, criteria["promotion_stages"]["stage_2_candidate_to_winner"]["criteria"])
            new_row = {
                "job_id": job_id,
                "timeframe": metrics.get("timeframe", ""),
                "strategy_family": metrics.get("family_id", ""),
                "logic_variant": metrics.get("entry_style", ""),
                "micro_exit_variant": metrics.get("micro_exit", ""),
                "trade_count": str(metrics.get("trade_count", 0)),
                "win_rate_pct": str(metrics.get("win_rate_pct", 0)),
                "profit_factor": str(metrics.get("profit_factor", 0)),
                "expectancy": str(metrics.get("expectancy", 0)),
                "pnl_sum": str(metrics.get("pnl_sum", 0)),
                "max_drawdown": str(metrics.get("max_drawdown", 0)),
                "avg_win": str(metrics.get("avg_win", 0)),
                "avg_loss": str(metrics.get("avg_loss", 0)),
                "status": metrics.get("status", "DONE"),
                "created_at_utc": timestamp,
                "source_file": str(outdir),
                "result_stage": "gate4_execution",
                "regime_summary": "",
                "side": metrics.get("side_policy", "both"),
                "score": str(score),
                "registry_version": criteria["registry_governance"]["current_version"]
            }
            promoted_rows.append(new_row)
            audit_rows.append({
                "timestamp_utc": timestamp,
                "job_id": job_id,
                "from_stage": "execution_result",
                "to_stage": "registry_candidates",
                "criteria_version": criteria["promotion_criteria_version"],
                "promotion_result": "promoted",
                "rejection_reason": ""
            })
        else:
            rejected_rows.append({"job_id": job_id, "reasons": reasons})
            audit_rows.append({
                "timestamp_utc": timestamp,
                "job_id": job_id,
                "from_stage": "execution_result",
                "to_stage": "rejected",
                "criteria_version": criteria["promotion_criteria_version"],
                "promotion_result": "rejected",
                "rejection_reason": "; ".join(reasons)
            })

    if dry_run:
        return {
            "status": "dry_run",
            "promoted": len(promoted_rows),
            "rejected": len(rejected_rows),
            "no_metrics": no_metrics,
            "would_be_promoted": promoted_rows[:5],
            "would_be_rejected": rejected_rows[:5],
            "audit": audit_rows
        }

    if promoted_rows:
        fieldnames = list(promoted_rows[0].keys())
        write_header = not candidates_csv.exists()
        with open(candidates_csv, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(promoted_rows)

    audit_csv = registry_dir / "promotion_audit_log.csv"
    if audit_rows:
        audit_fields = list(audit_rows[0].keys())
        audit_header = not audit_csv.exists()
        with open(audit_csv, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=audit_fields)
            if audit_header:
                writer.writeheader()
            writer.writerows(audit_rows)

    return {
        "status": "complete",
        "promoted": len(promoted_rows),
        "rejected": len(rejected_rows),
        "no_metrics": no_metrics,
        "audit": audit_rows
    }


def main():
    parser = argparse.ArgumentParser(description="Gate 5 - Promote execution results to registry")
    parser.add_argument("--outdir", required=True, type=Path, help="Gate 4 execution output directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be promoted without writing")
    args = parser.parse_args()

    registry_dir = Path("C:/Data/Bot/central_backtest_results/winner_registry")
    criteria_path = registry_dir / "promotion_criteria_v1_0_0.json"

    result = promote(args.outdir, registry_dir, criteria_path, dry_run=args.dry_run)

    print(f"[PROMOTION] status={result['status']}")
    print(f"[PROMOTION] promoted={result['promoted']}")
    print(f"[PROMOTION] rejected={result['rejected']}")
    print(f"[PROMOTION] no_metrics={len(result.get('no_metrics', []))}")

    if result.get("would_be_promoted"):
        print("\n[SAMPLE] Would be promoted (first 5):")
        for row in result["would_be_promoted"]:
            print(f"  job_id={row['job_id']} score={row['score']} pf={row['profit_factor']} tc={row['trade_count']}")

    if result.get("would_be_rejected"):
        print("\n[SAMPLE] Would be rejected (first 5):")
        for row in result["would_be_rejected"]:
            print(f"  job_id={row['job_id']} reasons={row['reasons']}")

    if result.get("no_metrics"):
        print(f"\n[WARNING] {len(result['no_metrics'])} jobs have no parseable metrics")


if __name__ == "__main__":
    main()

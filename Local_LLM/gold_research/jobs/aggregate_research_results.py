# version: v1.0.0
# file: C:\Data\Bot\Local_LLM\gold_research\jobs\aggregate_research_results.py
# run:
# python C:\Data\Bot\Local_LLM\gold_research\jobs\aggregate_research_results.py --manifest C:\Data\Bot\central_backtest_results\research_jobs\research_job_manifest.jsonl --state C:\Data\Bot\central_backtest_results\research_jobs\job_state.jsonl --outdir C:\Data\Bot\central_backtest_results\research_reports

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "v1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate research job results into leaderboard and progress reports")
    parser.add_argument("--manifest", required=True, help="Path to research_job_manifest.jsonl")
    parser.add_argument("--state", required=True, help="Path to job_state.jsonl")
    parser.add_argument("--outdir", required=True, help="Directory to write reports")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def latest_state_map(state_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in state_rows:
        latest[row["job_id"]] = row
    return latest


def load_summary(summary_path: str) -> dict[str, Any] | None:
    if not summary_path:
        return None
    path = Path(summary_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def score_row(row: dict[str, Any]) -> float:
    metrics = row.get("metrics", {})
    pnl_sum = metrics.get("pnl_sum")
    payoff_ratio = metrics.get("payoff_ratio")
    max_consecutive_losses = metrics.get("max_consecutive_losses")
    trade_count = metrics.get("trade_count")

    score = 0.0
    if isinstance(pnl_sum, (int, float)):
        score += float(pnl_sum) * 0.25
    if isinstance(payoff_ratio, (int, float)):
        score += float(payoff_ratio) * 100.0
    if isinstance(max_consecutive_losses, (int, float)):
        score -= float(max_consecutive_losses) * 20.0
    if isinstance(trade_count, (int, float)):
        score += min(float(trade_count), 1000.0) * 0.01
    return round(score, 4)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    state_path = Path(args.state)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_jsonl(manifest_path)
    state_rows = read_jsonl(state_path)
    latest_state = latest_state_map(state_rows)

    leaderboard_all: list[dict[str, Any]] = []
    done_only: list[dict[str, Any]] = []
    reject_counter: Counter[str] = Counter()

    for job in manifest_rows:
        state = latest_state.get(job["job_id"], {})
        status = state.get("status", "pending")
        summary_json = state.get("summary_json", "")
        summary = load_summary(summary_json)

        row = {
            "job_id": job["job_id"],
            "stage": job.get("stage", ""),
            "family_id": job.get("family_id", job.get("family", "")),
            "timeframe": job.get("timeframe", ""),
            "entry_style": job.get("entry_style", ""),
            "micro_exit": job.get("micro_exit", {}),
            "cooldown": job.get("cooldown", {}),
            "regime_filter": job.get("regime_filter", {}),
            "status": status,
            "summary_json": summary_json,
            "metrics": {},
            "notes": [],
            "score": None,
        }

        if summary is not None:
            if "metrics" in summary:
                row["metrics"] = summary["metrics"]
            else:
                # benchmark or future real-family summaries may not nest metrics
                row["metrics"] = {
                    "pnl_sum": summary.get("pnl_sum"),
                    "payoff_ratio": summary.get("payoff_ratio"),
                    "max_consecutive_losses": summary.get("max_consecutive_losses"),
                    "trade_count": summary.get("trades"),
                }
            row["notes"] = summary.get("notes", [])
            row["score"] = score_row(row)

        if status == "skipped":
            reject_reason = "unknown_skip_reason"
            if summary and "skip_reason" in summary:
                reject_reason = summary["skip_reason"]
            reject_counter[reject_reason] += 1

        if status == "failed":
            reject_counter["failed"] += 1

        leaderboard_all.append(row)
        if status == "done":
            done_only.append(row)

    leaderboard_all_sorted = sorted(
        leaderboard_all,
        key=lambda x: (
            {"done": 0, "pending": 1, "running": 2, "skipped": 3, "failed": 4}.get(x["status"], 9),
            -(x["score"] if isinstance(x["score"], (int, float)) else -999999),
        ),
    )

    done_only_sorted = sorted(
        done_only,
        key=lambda x: -(x["score"] if isinstance(x["score"], (int, float)) else -999999),
    )

    progress_summary = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "manifest_path": str(manifest_path),
        "state_path": str(state_path),
        "total_jobs": len(manifest_rows),
        "status_counts": dict(Counter(row["status"] for row in leaderboard_all)),
        "done_by_stage": dict(Counter(row["stage"] for row in done_only)),
        "done_by_family": dict(Counter(row["family_id"] for row in done_only)),
        "done_by_timeframe": dict(Counter(row["timeframe"] for row in done_only if row["timeframe"])),
    }

    reject_summary = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "reject_reasons": dict(reject_counter),
    }

    write_jsonl(outdir / "leaderboard_all.jsonl", leaderboard_all_sorted)
    write_jsonl(outdir / "leaderboard_done_only.jsonl", done_only_sorted)
    write_json(outdir / "progress_summary.json", progress_summary)
    write_json(outdir / "reject_summary.json", reject_summary)

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] total_jobs={len(manifest_rows)}")
    print(f"[DONE] done_jobs={len(done_only)}")
    print(f"[DONE] leaderboard_all={outdir / 'leaderboard_all.jsonl'}")
    print(f"[DONE] leaderboard_done_only={outdir / 'leaderboard_done_only.jsonl'}")
    print(f"[DONE] progress_summary={outdir / 'progress_summary.json'}")
    print(f"[DONE] reject_summary={outdir / 'reject_summary.json'}")
    print("=" * 120)


if __name__ == "__main__":
    main()
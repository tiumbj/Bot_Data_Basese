from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

VERSION = "v1.0.0"


@dataclass
class ShardStatus:
    name: str
    start: int
    end: int
    exists: bool
    results_count: int
    state_count: int
    summary_exists: bool
    processed_jobs: Optional[int]
    results_count_summary: Optional[int]
    no_trade_jobs: Optional[int]
    error_jobs: Optional[int]
    total_elapsed_sec: Optional[float]
    is_complete: bool
    last_write_time: Optional[str]


def iso_local(timestamp_value: float) -> str:
    return datetime.fromtimestamp(timestamp_value).isoformat(timespec="seconds")


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_expected_shards(max_end: int, shard_size: int) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    start = 0
    while start < max_end:
        end = min(start + shard_size, max_end)
        ranges.append((start, end))
        start = end
    return ranges


def inspect_one_shard(shard_dir: Path, start: int, end: int) -> ShardStatus:
    results_path = shard_dir / "results.jsonl"
    state_path = shard_dir / "state.jsonl"
    summary_path = shard_dir / "summary.json"

    summary = load_json(summary_path)

    results_count = count_lines(results_path)
    state_count = count_lines(state_path)

    processed_jobs = summary.get("processed_jobs")
    results_count_summary = summary.get("results_count")
    no_trade_jobs = summary.get("no_trade_jobs")
    error_jobs = summary.get("error_jobs")
    total_elapsed_sec = summary.get("total_elapsed_sec")

    expected_jobs = max(0, end - start)
    is_complete = False
    if summary_path.exists():
        if processed_jobs == expected_jobs or results_count_summary == expected_jobs:
            is_complete = True

    last_write_time = None
    if shard_dir.exists():
        last_write_time = iso_local(shard_dir.stat().st_mtime)

    return ShardStatus(
        name=shard_dir.name,
        start=start,
        end=end,
        exists=shard_dir.exists(),
        results_count=results_count,
        state_count=state_count,
        summary_exists=summary_path.exists(),
        processed_jobs=processed_jobs,
        results_count_summary=results_count_summary,
        no_trade_jobs=no_trade_jobs,
        error_jobs=error_jobs,
        total_elapsed_sec=total_elapsed_sec,
        is_complete=is_complete,
        last_write_time=last_write_time,
    )


def summarize_status(statuses: List[ShardStatus]) -> Dict:
    existing = [x for x in statuses if x.exists]
    complete = [x for x in statuses if x.is_complete]
    active = [
        x for x in statuses
        if x.exists and not x.is_complete and (x.results_count > 0 or x.state_count > 0 or x.summary_exists)
    ]
    missing = [x for x in statuses if not x.exists]

    return {
        "version": VERSION,
        "expected_shards": len(statuses),
        "existing_shards": len(existing),
        "complete_shards": len(complete),
        "active_shards": len(active),
        "missing_shards": len(missing),
        "farthest_complete_end": max((x.end for x in complete), default=0),
        "latest_existing_end": max((x.end for x in existing), default=0),
        "total_results_lines": sum(x.results_count for x in existing),
        "total_state_lines": sum(x.state_count for x in existing),
        "total_processed_jobs_from_summary": sum(int(x.processed_jobs or 0) for x in existing),
        "total_no_trade_jobs_from_summary": sum(int(x.no_trade_jobs or 0) for x in existing),
        "total_error_jobs_from_summary": sum(int(x.error_jobs or 0) for x in existing),
    }


def write_report(
    report_path: Path,
    summary_payload: Dict,
    statuses: List[ShardStatus],
    campaign_state: Dict,
    campaign_summary: Dict,
) -> None:
    lines: List[str] = []
    lines.append(f"full_discovery_status_report {VERSION}")
    lines.append("=" * 120)
    lines.append("")
    lines.append("[campaign_state.json]")
    lines.append(json.dumps(campaign_state, ensure_ascii=False, indent=2) if campaign_state else "not_found")
    lines.append("")
    lines.append("[campaign_summary.json]")
    lines.append(json.dumps(campaign_summary, ensure_ascii=False, indent=2) if campaign_summary else "not_found")
    lines.append("")
    lines.append("[aggregated_status]")
    lines.append(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("[shards]")
    lines.append(
        "name,start,end,exists,results_count,state_count,summary_exists,processed_jobs,results_count_summary,"
        "no_trade_jobs,error_jobs,total_elapsed_sec,is_complete,last_write_time"
    )
    for item in statuses:
        lines.append(
            f"{item.name},{item.start},{item.end},{item.exists},{item.results_count},{item.state_count},"
            f"{item.summary_exists},{item.processed_jobs},{item.results_count_summary},{item.no_trade_jobs},"
            f"{item.error_jobs},{item.total_elapsed_sec},{item.is_complete},{item.last_write_time}"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check full discovery status from 0 to a target end range.")
    parser.add_argument("--outroot", required=True, help="Root folder containing shard_* directories")
    parser.add_argument("--max-end", type=int, default=500000, help="Upper bound to inspect, e.g. 500000")
    parser.add_argument("--shard-size", type=int, default=10000, help="Shard size, e.g. 10000")
    parser.add_argument("--report", default=None, help="Optional output txt report path")
    args = parser.parse_args()

    outroot = Path(args.outroot)
    max_end = int(args.max_end)
    shard_size = int(args.shard_size)

    if not outroot.exists():
        print(f"[ERROR] version={VERSION} reason=outroot_not_found path={outroot}")
        return 2
    if max_end <= 0:
        print(f"[ERROR] version={VERSION} reason=max_end_must_be_positive value={max_end}")
        return 2
    if shard_size <= 0:
        print(f"[ERROR] version={VERSION} reason=shard_size_must_be_positive value={shard_size}")
        return 2

    expected_ranges = build_expected_shards(max_end=max_end, shard_size=shard_size)
    statuses: List[ShardStatus] = []
    for start, end in expected_ranges:
        shard_dir = outroot / f"shard_{start:06d}_{end:06d}"
        statuses.append(inspect_one_shard(shard_dir=shard_dir, start=start, end=end))

    campaign_state = load_json(outroot / "campaign_state.json")
    campaign_summary = load_json(outroot / "campaign_summary.json")
    summary_payload = summarize_status(statuses=statuses)

    report_path = Path(args.report) if args.report else outroot / "full_discovery_status_report.txt"
    write_report(
        report_path=report_path,
        summary_payload=summary_payload,
        statuses=statuses,
        campaign_state=campaign_state,
        campaign_summary=campaign_summary,
    )

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] outroot={outroot}")
    print(f"[DONE] max_end={max_end}")
    print(f"[DONE] shard_size={shard_size}")
    print(f"[DONE] expected_shards={summary_payload['expected_shards']}")
    print(f"[DONE] existing_shards={summary_payload['existing_shards']}")
    print(f"[DONE] complete_shards={summary_payload['complete_shards']}")
    print(f"[DONE] active_shards={summary_payload['active_shards']}")
    print(f"[DONE] missing_shards={summary_payload['missing_shards']}")
    print(f"[DONE] farthest_complete_end={summary_payload['farthest_complete_end']}")
    print(f"[DONE] latest_existing_end={summary_payload['latest_existing_end']}")
    print(f"[DONE] total_results_lines={summary_payload['total_results_lines']}")
    print(f"[DONE] total_state_lines={summary_payload['total_state_lines']}")
    print(f"[DONE] total_processed_jobs_from_summary={summary_payload['total_processed_jobs_from_summary']}")
    print(f"[DONE] total_no_trade_jobs_from_summary={summary_payload['total_no_trade_jobs_from_summary']}")
    print(f"[DONE] total_error_jobs_from_summary={summary_payload['total_error_jobs_from_summary']}")
    print(f"[DONE] report={report_path}")
    print("=" * 120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

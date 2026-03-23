# ============================================================
# ชื่อโค้ด: organize_database_assets.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\organize_database_assets.py
# คำสั่งรัน:
#   python C:\Data\Bot\Local_LLM\gold_research\jobs\organize_database_assets.py --dry-run
#   python C:\Data\Bot\Local_LLM\gold_research\jobs\organize_database_assets.py --execute
# เวอร์ชัน: v1.0.1
# ============================================================

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List


VERSION = "v1.0.1"
DEFAULT_TARGET_ROOT = Path(r"C:\Data\data_base")


@dataclass
class MovePlan:
    category: str
    source: Path
    dest: Path
    exists_at_plan_time: bool
    action: str
    note: str


@dataclass
class MoveResult:
    category: str
    source: str
    dest: str
    action: str
    status: str
    note: str


def utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dirs(target_root: Path) -> None:
    required_dirs = [
        target_root,
        target_root / "canonical_dataset",
        target_root / "backtest_results",
        target_root / "indexes",
        target_root / "manifests",
        target_root / "logs",
        target_root / "_migration",
        target_root / "_legacy_snapshots",
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def build_raw_plans(target_root: Path) -> List[MovePlan]:
    raw_specs = [
        ("canonical_dataset", r"C:\Data\data_base\canonical_dataset\dataset", target_root / "canonical_dataset" / "dataset"),
        ("canonical_dataset", r"C:\Data\Bot\central_dataset", target_root / "canonical_dataset" / "central_dataset"),
        ("backtest_results", r"C:\Data\data_base\backtest_results\central_backtest_results", target_root / "backtest_results" / "central_backtest_results"),
        ("backtest_results", r"C:\Data\data_base\backtest_results\gold_research_results", target_root / "backtest_results" / "gold_research_results"),
        ("backtest_results", r"C:\Data\Bot\Local_LLM\gold_research\artifacts", target_root / "backtest_results" / "gold_research_artifacts"),
        ("backtest_results", r"C:\Data\Bot\Local_LLM\gold_research\output", target_root / "backtest_results" / "gold_research_output"),
        ("backtest_results", r"C:\Data\data_base\backtest_results\gold_research_outputs", target_root / "backtest_results" / "gold_research_outputs"),
        ("indexes", r"C:\Data\Bot\Local_LLM\gold_research\index", target_root / "indexes" / "gold_research_index"),
        ("indexes", r"C:\Data\Bot\Local_LLM\gold_research\indexes", target_root / "indexes" / "gold_research_indexes"),
        ("manifests", r"C:\Data\Bot\Local_LLM\gold_research\manifest", target_root / "manifests" / "gold_research_manifest"),
        ("manifests", r"C:\Data\Bot\Local_LLM\gold_research\manifests", target_root / "manifests" / "gold_research_manifests"),
        ("logs", r"C:\Data\data_base\logs\gold_research_logs", target_root / "logs" / "gold_research_logs"),
        ("logs", r"C:\Data\Bot\Local_LLM\logs", target_root / "logs" / "local_llm_logs"),
        ("indexes", r"C:\Data\data_base\backtest_results\central_backtest_results\index\backtest_index.csv", target_root / "indexes" / "backtest_index.csv"),
        ("manifests", r"C:\Data\data_base\canonical_dataset\dataset\manifest\dataset_manifest.csv", target_root / "manifests" / "dataset_manifest.csv"),
        ("manifests", r"C:\Data\data_base\canonical_dataset\dataset\manifest\dataset_manifest.json", target_root / "manifests" / "dataset_manifest.json"),
    ]

    plans: List[MovePlan] = []
    for category, source_text, dest in raw_specs:
        source = Path(source_text)
        exists = source.exists()
        plans.append(
            MovePlan(
                category=category,
                source=source,
                dest=dest,
                exists_at_plan_time=exists,
                action="move" if exists else "skip",
                note="ready" if exists else "source_not_found",
            )
        )
    return plans


def is_child_of(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
        return True
    except ValueError:
        return False


def prune_nested_plans(plans: List[MovePlan]) -> List[MovePlan]:
    kept: List[MovePlan] = []
    moving_parents = [p for p in plans if p.action == "move" and p.source.exists() and p.source.is_dir()]

    for plan in plans:
        if plan.action != "move":
            kept.append(plan)
            continue

        covered_by_parent = False
        for parent_plan in moving_parents:
            if parent_plan.source == plan.source:
                continue
            if is_child_of(plan.source, parent_plan.source):
                covered_by_parent = True
                break

        if covered_by_parent:
            kept.append(
                MovePlan(
                    category=plan.category,
                    source=plan.source,
                    dest=plan.dest,
                    exists_at_plan_time=plan.exists_at_plan_time,
                    action="skip",
                    note="covered_by_parent_move",
                )
            )
        else:
            kept.append(plan)

    return kept


def sort_plans(plans: List[MovePlan]) -> List[MovePlan]:
    def sort_key(plan: MovePlan):
        move_rank = 0 if plan.action == "move" else 1
        file_rank = 0 if plan.source.is_file() else 1
        depth_rank = -len(plan.source.parts)
        return (move_rank, file_rank, depth_rank, str(plan.source).lower())

    return sorted(plans, key=sort_key)


def move_path(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"destination_exists: {dest}")
    shutil.move(str(source), str(dest))


def print_line(text: str) -> None:
    print(text, flush=True)


def save_summary(target_root: Path, payload: dict) -> Path:
    summary_path = target_root / "_migration" / f"organize_database_assets_{utc_now_text()}.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize database-related assets into C:\\Data\\data_base")
    parser.add_argument("--target-root", default=str(DEFAULT_TARGET_ROOT), help="Target root path")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions only")
    parser.add_argument("--execute", action="store_true", help="Execute move actions")
    args = parser.parse_args()

    if args.dry_run == args.execute:
        raise SystemExit("Use exactly one of --dry-run or --execute")

    target_root = Path(args.target_root)
    ensure_dirs(target_root)

    raw_plans = build_raw_plans(target_root)
    plans = sort_plans(prune_nested_plans(raw_plans))

    print_line("=" * 120)
    print_line(f"[INFO] organize_database_assets.py version={VERSION}")
    print_line(f"[INFO] target_root={target_root}")
    print_line(f"[INFO] dry_run={args.dry_run}")
    print_line("=" * 120)

    results: List[MoveResult] = []
    moved_count = 0
    skipped_count = 0
    error_count = 0

    for plan in plans:
        source_exists_now = plan.source.exists()
        effective_action = plan.action
        effective_note = plan.note

        if effective_action == "move" and not source_exists_now:
            effective_action = "skip"
            effective_note = "source_not_found_at_execution"

        print_line(
            f"[PLAN] category={plan.category} action={effective_action} exists={source_exists_now} "
            f"source={plan.source} -> dest={plan.dest} note={effective_note}"
        )

        if effective_action == "skip":
            skipped_count += 1
            results.append(
                MoveResult(
                    category=plan.category,
                    source=str(plan.source),
                    dest=str(plan.dest),
                    action=effective_action,
                    status="SKIPPED",
                    note=effective_note,
                )
            )
            continue

        if args.dry_run:
            results.append(
                MoveResult(
                    category=plan.category,
                    source=str(plan.source),
                    dest=str(plan.dest),
                    action=effective_action,
                    status="PLANNED",
                    note=effective_note,
                )
            )
            continue

        try:
            move_path(plan.source, plan.dest)
            moved_count += 1
            print_line(f"[DONE] moved: {plan.source} -> {plan.dest}")
            results.append(
                MoveResult(
                    category=plan.category,
                    source=str(plan.source),
                    dest=str(plan.dest),
                    action=effective_action,
                    status="MOVED",
                    note=effective_note,
                )
            )
        except Exception as exc:
            error_count += 1
            print_line(f"[ERROR] move_failed: {plan.source} -> {plan.dest} reason={exc}")
            results.append(
                MoveResult(
                    category=plan.category,
                    source=str(plan.source),
                    dest=str(plan.dest),
                    action=effective_action,
                    status="ERROR",
                    note=str(exc),
                )
            )

    summary_payload = {
        "version": VERSION,
        "target_root": str(target_root),
        "dry_run": args.dry_run,
        "generated_at_utc": utc_now_text(),
        "moved_count": moved_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
        "results": [asdict(item) for item in results],
    }
    summary_json = save_summary(target_root, summary_payload)

    print_line("=" * 120)
    print_line(f"[SUMMARY] moved_count={moved_count} skipped_count={skipped_count} error_count={error_count}")
    print_line(f"[SUMMARY] summary_json={summary_json}")
    print_line("=" * 120)


if __name__ == "__main__":
    main()
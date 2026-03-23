# ============================================================
# ชื่อโค้ด: migrate_tf_dataset_to_central_db.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\migrate_tf_dataset_to_central_db.py
# คำสั่งรัน:
#   python C:\Data\Bot\Local_LLM\gold_research\jobs\migrate_tf_dataset_to_central_db.py --dry-run
#   python C:\Data\Bot\Local_LLM\gold_research\jobs\migrate_tf_dataset_to_central_db.py --execute
# เวอร์ชัน: v1.0.1
# ============================================================

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List


VERSION = "v1.0.1"
SOURCE_TF_DIR = Path(r"C:\Data\data_base\canonical_dataset\dataset\tf")
TARGET_TF_DIR = Path(r"C:\Data\data_base\canonical_dataset\dataset\tf")
REPORT_DIR = Path(r"C:\Data\data_base\_migration")
ALLOWED_EXTENSIONS = {".csv", ".parquet"}


@dataclass
class FilePlan:
    source_path: str
    target_path: str
    relative_path: str
    source_exists: bool
    target_exists: bool
    action: str
    note: str
    source_size_bytes: int | None


@dataclass
class FileResult:
    source_path: str
    target_path: str
    relative_path: str
    status: str
    note: str
    source_sha256: str | None
    target_sha256: str | None
    source_size_bytes: int | None
    target_size_bytes: int | None


def utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dirs() -> None:
    TARGET_TF_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_source_files(source_root: Path) -> List[Path]:
    if not source_root.exists():
        return []
    files: List[Path] = []
    for path in source_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append(path)
    return sorted(files, key=lambda p: str(p).lower())


def build_plans(source_root: Path, target_root: Path) -> List[FilePlan]:
    plans: List[FilePlan] = []

    for source_path in collect_source_files(source_root):
        relative_path = source_path.relative_to(source_root)
        target_path = target_root / relative_path
        source_exists = source_path.exists()
        target_exists = target_path.exists()
        source_size = source_path.stat().st_size if source_exists else None

        if not source_exists:
            action = "skip"
            note = "source_not_found"
        elif not target_exists:
            action = "copy"
            note = "target_missing"
        else:
            action = "verify_or_replace"
            note = "target_exists"

        plans.append(
            FilePlan(
                source_path=str(source_path),
                target_path=str(target_path),
                relative_path=str(relative_path),
                source_exists=source_exists,
                target_exists=target_exists,
                action=action,
                note=note,
                source_size_bytes=source_size,
            )
        )

    return plans


def save_json_report(filename_prefix: str, payload: dict) -> Path:
    report_path = REPORT_DIR / f"{filename_prefix}_{utc_now_text()}.json"
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report_path


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(target))


def print_line(text: str) -> None:
    print(text, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate TF dataset files into central data_base canonical dataset."
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--execute", action="store_true", help="Execute copy/verify")
    args = parser.parse_args()

    if args.dry_run == args.execute:
        raise SystemExit("Use exactly one of --dry-run or --execute")

    ensure_dirs()
    plans = build_plans(SOURCE_TF_DIR, TARGET_TF_DIR)

    print_line("=" * 120)
    print_line(f"[INFO] migrate_tf_dataset_to_central_db.py version={VERSION}")
    print_line(f"[INFO] source_tf_dir={SOURCE_TF_DIR}")
    print_line(f"[INFO] target_tf_dir={TARGET_TF_DIR}")
    print_line(f"[INFO] dry_run={args.dry_run}")
    print_line("=" * 120)

    for plan in plans:
        print_line(
            f"[PLAN] action={plan.action} source_exists={plan.source_exists} "
            f"target_exists={plan.target_exists} relative={plan.relative_path} "
            f"note={plan.note} size={plan.source_size_bytes}"
        )

    results: List[FileResult] = []
    copied_count = 0
    verified_same_count = 0
    replaced_count = 0
    skipped_count = 0
    error_count = 0

    for plan in plans:
        source = Path(plan.source_path)
        target = Path(plan.target_path)

        if args.dry_run:
            results.append(
                FileResult(
                    source_path=plan.source_path,
                    target_path=plan.target_path,
                    relative_path=plan.relative_path,
                    status="PLANNED",
                    note=plan.note,
                    source_sha256=None,
                    target_sha256=None,
                    source_size_bytes=plan.source_size_bytes,
                    target_size_bytes=target.stat().st_size if target.exists() else None,
                )
            )
            continue

        if not source.exists():
            skipped_count += 1
            results.append(
                FileResult(
                    source_path=plan.source_path,
                    target_path=plan.target_path,
                    relative_path=plan.relative_path,
                    status="SKIPPED",
                    note="source_not_found_at_execution",
                    source_sha256=None,
                    target_sha256=None,
                    source_size_bytes=None,
                    target_size_bytes=target.stat().st_size if target.exists() else None,
                )
            )
            continue

        try:
            source_sha = sha256_of_file(source)
            source_size = source.stat().st_size

            if not target.exists():
                copy_file(source, target)
                target_sha = sha256_of_file(target)
                target_size = target.stat().st_size
                copied_count += 1
                status = "COPIED"
                note = "target_created"
                print_line(f"[DONE] copied: {source} -> {target}")
            else:
                target_sha_before = sha256_of_file(target)
                target_size_before = target.stat().st_size

                if source_sha == target_sha_before:
                    verified_same_count += 1
                    status = "VERIFIED_SAME"
                    note = "already_identical"
                    target_sha = target_sha_before
                    target_size = target_size_before
                    print_line(f"[DONE] verified_same: {target}")
                else:
                    copy_file(source, target)
                    target_sha = sha256_of_file(target)
                    target_size = target.stat().st_size
                    replaced_count += 1
                    status = "REPLACED"
                    note = "target_overwritten_with_source"
                    print_line(f"[DONE] replaced: {source} -> {target}")

            results.append(
                FileResult(
                    source_path=plan.source_path,
                    target_path=plan.target_path,
                    relative_path=plan.relative_path,
                    status=status,
                    note=note,
                    source_sha256=source_sha,
                    target_sha256=target_sha,
                    source_size_bytes=source_size,
                    target_size_bytes=target_size,
                )
            )
        except Exception as exc:
            error_count += 1
            print_line(f"[ERROR] relative={plan.relative_path} reason={exc}")
            results.append(
                FileResult(
                    source_path=plan.source_path,
                    target_path=plan.target_path,
                    relative_path=plan.relative_path,
                    status="ERROR",
                    note=str(exc),
                    source_sha256=None,
                    target_sha256=None,
                    source_size_bytes=plan.source_size_bytes,
                    target_size_bytes=target.stat().st_size if target.exists() else None,
                )
            )

    payload = {
        "version": VERSION,
        "source_tf_dir": str(SOURCE_TF_DIR),
        "target_tf_dir": str(TARGET_TF_DIR),
        "dry_run": args.dry_run,
        "generated_at_utc": utc_now_text(),
        "planned_file_count": len(plans),
        "copied_count": copied_count,
        "verified_same_count": verified_same_count,
        "replaced_count": replaced_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
        "plans": [asdict(plan) for plan in plans],
        "results": [asdict(result) for result in results],
    }

    report_path = save_json_report("migrate_tf_dataset_to_central_db", payload)

    print_line("=" * 120)
    print_line(
        f"[SUMMARY] planned_file_count={len(plans)} copied_count={copied_count} "
        f"verified_same_count={verified_same_count} replaced_count={replaced_count} "
        f"skipped_count={skipped_count} error_count={error_count}"
    )
    print_line(f"[SUMMARY] report_json={report_path}")
    print_line("=" * 120)


if __name__ == "__main__":
    main()
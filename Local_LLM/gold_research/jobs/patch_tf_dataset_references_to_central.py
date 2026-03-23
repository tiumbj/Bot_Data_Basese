# ============================================================
# ชื่อโค้ด: patch_tf_dataset_references_to_central.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\patch_tf_dataset_references_to_central.py
# คำสั่งรัน:
#   python C:\Data\Bot\Local_LLM\gold_research\jobs\patch_tf_dataset_references_to_central.py --dry-run
#   python C:\Data\Bot\Local_LLM\gold_research\jobs\patch_tf_dataset_references_to_central.py --execute
# เวอร์ชัน: v1.0.1
# ============================================================

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple


VERSION = "v1.0.1"
DEFAULT_ROOT = Path(r"C:\Data\Bot")
OUTPUT_DIR = Path(r"C:\Data\data_base\_migration")

REPLACEMENTS: List[Tuple[str, str]] = [
    (
        r"C:\Data\data_base\canonical_dataset\dataset\tf",
        r"C:\Data\data_base\canonical_dataset\dataset\tf",
    ),
]

TEXT_FILE_EXTENSIONS = {
    ".py",
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".csv",
    ".ps1",
    ".bat",
    ".cmd",
}

EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
}


@dataclass
class FilePatchPlan:
    file_path: str
    replacement_count: int
    changes: list[dict]


@dataclass
class FilePatchResult:
    file_path: str
    status: str
    replacement_count: int
    note: str
    backup_path: str | None


def utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_text_candidate(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TEXT_FILE_EXTENSIONS


def should_skip_path(path: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


def iter_candidate_files(root: Path):
    for path in root.rglob("*"):
        if should_skip_path(path):
            continue
        if is_text_candidate(path):
            yield path


def read_text_safely(file_path: Path) -> str | None:
    encodings = ["utf-8", "utf-8-sig", "cp874", "latin-1"]
    for encoding in encodings:
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception:
            return None
    return None


def build_patch_plan(root: Path) -> List[FilePatchPlan]:
    plans: List[FilePatchPlan] = []

    for file_path in iter_candidate_files(root):
        original_text = read_text_safely(file_path)
        if original_text is None:
            continue

        changes = []
        total_count = 0

        for old_text, new_text in REPLACEMENTS:
            count = original_text.count(old_text)
            if count > 0:
                changes.append(
                    {
                        "old": old_text,
                        "new": new_text,
                        "count": count,
                    }
                )
                total_count += count

        if total_count > 0:
            plans.append(
                FilePatchPlan(
                    file_path=str(file_path),
                    replacement_count=total_count,
                    changes=changes,
                )
            )

    return sorted(plans, key=lambda x: x.file_path.lower())


def apply_replacements_to_text(text: str) -> tuple[str, int]:
    total = 0
    updated = text

    for old_text, new_text in REPLACEMENTS:
        count = updated.count(old_text)
        if count > 0:
            updated = updated.replace(old_text, new_text)
            total += count

    return updated, total


def backup_file(source_path: Path, backup_root: Path, root: Path) -> Path:
    timestamp = utc_now_text()
    backup_dir = backup_root / f"patch_tf_dataset_references_to_central_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    try:
        relative_path = source_path.relative_to(root)
    except ValueError:
        relative_path = Path(source_path.name)

    backup_path = backup_dir / relative_path
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path.write_bytes(source_path.read_bytes())
    return backup_path


def save_summary(filename_prefix: str, payload: dict) -> Path:
    summary_path = OUTPUT_DIR / f"{filename_prefix}_{utc_now_text()}.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_path


def print_header(root: Path, dry_run: bool) -> None:
    print("=" * 120)
    print(f"[INFO] patch_tf_dataset_references_to_central.py version={VERSION}")
    print(f"[INFO] root={root}")
    print(f"[INFO] output_dir={OUTPUT_DIR}")
    print(f"[INFO] dry_run={dry_run}")
    print("=" * 120)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch TF dataset path references to the central canonical dataset root."
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Root folder to scan",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute patch",
    )
    args = parser.parse_args()

    if args.dry_run == args.execute:
        raise SystemExit("Use exactly one of --dry-run or --execute")

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    ensure_output_dir()
    plans = build_patch_plan(root)

    print_header(root=root, dry_run=args.dry_run)

    for plan in plans:
        print(f"[PLAN] file={plan.file_path} replacement_count={plan.replacement_count}")
        for change in plan.changes:
            print(
                f"        old={change['old']} -> new={change['new']} count={change['count']}"
            )

    results: List[FilePatchResult] = []
    patched_file_count = 0
    patched_replacement_count = 0
    error_count = 0

    if args.execute:
        backup_root = OUTPUT_DIR / "_backups"

        for plan in plans:
            file_path = Path(plan.file_path)
            original_text = read_text_safely(file_path)

            if original_text is None:
                error_count += 1
                print(f"[ERROR] read_failed file={file_path}")
                results.append(
                    FilePatchResult(
                        file_path=str(file_path),
                        status="ERROR",
                        replacement_count=0,
                        note="read_failed",
                        backup_path=None,
                    )
                )
                continue

            updated_text, replacement_count = apply_replacements_to_text(original_text)

            if replacement_count == 0 or updated_text == original_text:
                results.append(
                    FilePatchResult(
                        file_path=str(file_path),
                        status="SKIPPED",
                        replacement_count=0,
                        note="no_change_after_recompute",
                        backup_path=None,
                    )
                )
                continue

            try:
                backup_path = backup_file(
                    source_path=file_path,
                    backup_root=backup_root,
                    root=root,
                )
                file_path.write_text(updated_text, encoding="utf-8")
                patched_file_count += 1
                patched_replacement_count += replacement_count

                print(
                    f"[DONE] patched file={file_path} replacement_count={replacement_count}"
                )
                results.append(
                    FilePatchResult(
                        file_path=str(file_path),
                        status="PATCHED",
                        replacement_count=replacement_count,
                        note="success",
                        backup_path=str(backup_path),
                    )
                )
            except Exception as exc:
                error_count += 1
                print(f"[ERROR] patch_failed file={file_path} reason={exc}")
                results.append(
                    FilePatchResult(
                        file_path=str(file_path),
                        status="ERROR",
                        replacement_count=0,
                        note=str(exc),
                        backup_path=None,
                    )
                )
    else:
        for plan in plans:
            results.append(
                FilePatchResult(
                    file_path=plan.file_path,
                    status="PLANNED",
                    replacement_count=plan.replacement_count,
                    note="dry_run",
                    backup_path=None,
                )
            )

    summary_payload = {
        "version": VERSION,
        "root": str(root),
        "dry_run": args.dry_run,
        "generated_at_utc": utc_now_text(),
        "planned_file_count": len(plans),
        "patched_file_count": patched_file_count,
        "patched_replacement_count": patched_replacement_count,
        "error_count": error_count,
        "plans": [asdict(plan) for plan in plans],
        "results": [asdict(result) for result in results],
    }

    summary_path = save_summary("patch_tf_dataset_references_to_central", summary_payload)

    print("=" * 120)
    print(
        f"[SUMMARY] planned_file_count={len(plans)} "
        f"patched_file_count={patched_file_count} "
        f"patched_replacement_count={patched_replacement_count} "
        f"error_count={error_count}"
    )
    print(f"[SUMMARY] summary_json={summary_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()
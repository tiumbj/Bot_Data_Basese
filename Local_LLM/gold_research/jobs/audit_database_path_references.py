# ============================================================
# ชื่อโค้ด: audit_database_path_references.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\audit_database_path_references.py
# คำสั่งรัน:
#   python C:\Data\Bot\Local_LLM\gold_research\jobs\audit_database_path_references.py
#   python C:\Data\Bot\Local_LLM\gold_research\jobs\audit_database_path_references.py --root C:\Data\Bot
# เวอร์ชัน: v1.0.0
# ============================================================

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

VERSION = "v1.0.0"
DEFAULT_ROOT = Path(r"C:\Data\Bot")
DEFAULT_OUTPUT_DIR = Path(r"C:\Data\data_base\_migration")

OLD_TO_NEW = {
    r"C:\Data\data_base\canonical_dataset\dataset": r"C:\Data\data_base\canonical_dataset\dataset",
    r"C:\Data\data_base\backtest_results\central_backtest_results": r"C:\Data\data_base\backtest_results\central_backtest_results",
    r"C:\Data\data_base\backtest_results\gold_research_results": r"C:\Data\data_base\backtest_results\gold_research_results",
    r"C:\Data\data_base\backtest_results\gold_research_outputs": r"C:\Data\data_base\backtest_results\gold_research_outputs",
    r"C:\Data\data_base\logs\gold_research_logs": r"C:\Data\data_base\logs\gold_research_logs",
}

TEXT_FILE_SUFFIXES = {
    ".py", ".txt", ".md", ".json", ".yaml", ".yml", ".ini", ".cfg", ".conf",
    ".toml", ".csv", ".ps1", ".bat", ".cmd", ".env", ".sql", ".log",
}
SKIP_DIR_NAMES = {
    ".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache",
    ".pytest_cache", ".idea", ".vscode",
}


@dataclass
class MatchRecord:
    file_path: str
    line_number: int
    old_path: str
    suggested_new_path: str
    line_text: str


def utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def should_scan_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TEXT_FILE_SUFFIXES


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if should_scan_file(path):
            yield path


def safe_read_lines(path: Path) -> List[str] | None:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding, errors="strict").splitlines()
        except Exception:
            continue
    return None


def scan_root(root: Path) -> List[MatchRecord]:
    matches: List[MatchRecord] = []
    for file_path in iter_files(root):
        lines = safe_read_lines(file_path)
        if lines is None:
            continue
        for idx, line in enumerate(lines, start=1):
            for old_path, new_path in OLD_TO_NEW.items():
                if old_path in line:
                    matches.append(
                        MatchRecord(
                            file_path=str(file_path),
                            line_number=idx,
                            old_path=old_path,
                            suggested_new_path=new_path,
                            line_text=line.strip(),
                        )
                    )
    return matches


def write_outputs(output_dir: Path, matches: List[MatchRecord]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = utc_now_text()
    json_path = output_dir / f"database_path_audit_{ts}.json"
    txt_path = output_dir / f"database_path_audit_{ts}.txt"

    payload = {
        "version": VERSION,
        "generated_at_utc": ts,
        "match_count": len(matches),
        "matches": [asdict(m) for m in matches],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: List[str] = []
    lines.append("=" * 120)
    lines.append(f"database_path_audit version={VERSION}")
    lines.append(f"generated_at_utc={ts}")
    lines.append(f"match_count={len(matches)}")
    lines.append("=" * 120)
    for m in matches:
        lines.append(f"[MATCH] file={m.file_path} line={m.line_number}")
        lines.append(f"        old={m.old_path}")
        lines.append(f"        new={m.suggested_new_path}")
        lines.append(f"        text={m.line_text}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, txt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit old database-related path references in project files.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Root folder to scan")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to write audit reports")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)

    if not root.exists():
        raise SystemExit(f"root_not_found: {root}")

    print("=" * 120, flush=True)
    print(f"[INFO] audit_database_path_references.py version={VERSION}", flush=True)
    print(f"[INFO] root={root}", flush=True)
    print(f"[INFO] output_dir={output_dir}", flush=True)
    print("=" * 120, flush=True)

    matches = scan_root(root)
    json_path, txt_path = write_outputs(output_dir, matches)

    for m in matches:
        print(
            f"[MATCH] file={m.file_path} line={m.line_number} old={m.old_path} new={m.suggested_new_path}",
            flush=True,
        )

    print("=" * 120, flush=True)
    print(f"[SUMMARY] match_count={len(matches)}", flush=True)
    print(f"[SUMMARY] json_report={json_path}", flush=True)
    print(f"[SUMMARY] txt_report={txt_path}", flush=True)
    print("=" * 120, flush=True)


if __name__ == "__main__":
    main()

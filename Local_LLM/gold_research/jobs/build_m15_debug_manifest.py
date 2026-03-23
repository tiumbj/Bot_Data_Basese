# ==================================================================================================
# FILE: build_m15_debug_manifest.py
# PATH: C:\Data\Bot\Local_LLM\gold_research\jobs\build_m15_debug_manifest.py
# VERSION: v1.0.0
#
# CHANGELOG:
# - v1.0.0
#   1) Create M15-only debug manifest from full discovery JSONL manifest
#   2) Read JSONL correctly line-by-line
#   3) Export first 1000 M15 jobs as JSONL
# ==================================================================================================

from __future__ import annotations

import json
from pathlib import Path

VERSION = "v1.0.0"

INPUT_PATH = Path(r"C:\Data\Bot\central_backtest_results\research_jobs_full_discovery\research_job_manifest_full_discovery.jsonl")
OUTPUT_PATH = Path(r"C:\Data\Bot\central_backtest_results\research_jobs_full_discovery\debug_manifest_m15_000000_001000.jsonl")
TARGET_TIMEFRAME = "M15"
LIMIT = 1000


def get_timeframe(row: dict) -> str:
    value = row.get("timeframe", row.get("tf", ""))
    return str(value).strip().upper()


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input manifest not found: {INPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    selected = 0
    scanned = 0

    with INPUT_PATH.open("r", encoding="utf-8") as fin, OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            text = line.strip()
            if not text:
                continue

            scanned += 1
            row = json.loads(text)

            if get_timeframe(row) != TARGET_TIMEFRAME:
                continue

            fout.write(json.dumps(row, ensure_ascii=False))
            fout.write("\n")
            selected += 1

            if selected >= LIMIT:
                break

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] input={INPUT_PATH}")
    print(f"[DONE] output={OUTPUT_PATH}")
    print(f"[DONE] target_timeframe={TARGET_TIMEFRAME}")
    print(f"[DONE] selected_jobs={selected}")
    print(f"[DONE] scanned_rows={scanned}")
    print("=" * 120)


if __name__ == "__main__":
    main()
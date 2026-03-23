# ==================================================================================================
# FILE: build_tf_debug_manifest.py
# PATH: C:\Data\Bot\Local_LLM\gold_research\jobs\build_tf_debug_manifest.py
# VERSION: v1.1.0
#
# CHANGELOG:
# - v1.1.0
#   1) Generic JSONL manifest slicer for any timeframe
#   2) Reads full discovery manifest line-by-line
#   3) Exports first N jobs for the requested timeframe
#   4) Output file naming is deterministic and ready for the debug/vectorbt pipeline
# ==================================================================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path

VERSION = "v1.1.0"

INPUT_PATH = Path(
    r"C:\Data\Bot\central_backtest_results\research_jobs_full_discovery\research_job_manifest_full_discovery.jsonl"
)
OUTPUT_DIR = Path(r"C:\Data\Bot\central_backtest_results\research_jobs_full_discovery")


def normalize_tf(value: str) -> str:
    return str(value).strip().upper()


def get_timeframe(row: dict) -> str:
    return normalize_tf(row.get("timeframe", row.get("tf", "")))


def build_output_path(timeframe: str, limit: int) -> Path:
    tf = normalize_tf(timeframe)
    return OUTPUT_DIR / f"debug_manifest_{tf.lower()}_000000_{limit:06d}.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build timeframe-specific debug manifest from full discovery JSONL.")
    parser.add_argument("--timeframe", required=True, help="Target timeframe เช่น M30, H1")
    parser.add_argument("--limit", type=int, default=1000, help="Number of jobs to export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeframe = normalize_tf(args.timeframe)
    limit = max(1, int(args.limit))

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input manifest not found: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = build_output_path(timeframe, limit)

    selected = 0
    scanned = 0

    with INPUT_PATH.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            text = line.strip()
            if not text:
                continue

            scanned += 1
            row = json.loads(text)

            if get_timeframe(row) != timeframe:
                continue

            fout.write(json.dumps(row, ensure_ascii=False))
            fout.write("\n")
            selected += 1

            if selected >= limit:
                break

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] input={INPUT_PATH}")
    print(f"[DONE] output={output_path}")
    print(f"[DONE] target_timeframe={timeframe}")
    print(f"[DONE] selected_jobs={selected}")
    print(f"[DONE] scanned_rows={scanned}")
    print("=" * 120)


if __name__ == "__main__":
    main()
# ============================================================
# ชื่อโค้ด: append_paper_trade_event.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\append_paper_trade_event.py
# คำสั่งรัน:
#   python C:\Data\Bot\Local_LLM\gold_research\jobs\append_paper_trade_event.py --outdir C:\Data\Bot\central_backtest_results\paper_trade --type signal --input C:\Data\Bot\central_backtest_results\paper_trade\incoming_signal.json
# เวอร์ชัน: v1.0.0
# ============================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VERSION = "v1.0.0"

SIGNAL_REQUIRED_FIELDS = [
    "signal_id",
    "ts_utc",
    "strategy_id",
    "timeframe",
    "side",
    "trend_bucket",
    "volatility_bucket",
    "price_location_bucket",
    "approved",
]

TRADE_REQUIRED_FIELDS = [
    "trade_id",
    "signal_id",
    "entry_time_utc",
    "exit_time_utc",
    "entry_price",
    "exit_price",
    "pnl",
    "exit_reason",
    "trade_date",
    "hour_bucket",
    "trend_bucket",
    "volatility_bucket",
    "price_location_bucket",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ input | file={path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"input ต้องเป็น JSON object | file={path}")
    return payload


def ensure_required_fields(payload: dict[str, Any], required_fields: list[str], record_type: str) -> None:
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(
            f"{record_type} missing required fields | missing={missing}"
        )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Append paper-trade signal/trade event to monitor jsonl files.")
    parser.add_argument("--outdir", required=True, help="Output directory containing signals.jsonl and trades.jsonl")
    parser.add_argument("--type", required=True, choices=["signal", "trade"], help="Event type")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    input_path = Path(args.input)

    payload = load_json(input_path)

    print("=" * 120)
    print(f"[INFO] append_paper_trade_event.py version={VERSION}")
    print(f"[INFO] outdir={outdir}")
    print(f"[INFO] type={args.type}")
    print(f"[INFO] input={input_path}")
    print("=" * 120)

    if args.type == "signal":
        ensure_required_fields(payload, SIGNAL_REQUIRED_FIELDS, "signal")
        output_path = outdir / "signals.jsonl"
    else:
        ensure_required_fields(payload, TRADE_REQUIRED_FIELDS, "trade")
        output_path = outdir / "trades.jsonl"

    append_jsonl(output_path, payload)

    print(f"[DONE] appended_type={args.type}")
    print(f"[DONE] output={output_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()
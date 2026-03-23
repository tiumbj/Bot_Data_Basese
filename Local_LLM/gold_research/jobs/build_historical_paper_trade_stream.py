# ============================================================
# ชื่อโค้ด: build_historical_paper_trade_stream.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\build_historical_paper_trade_stream.py
# คำสั่งรัน:
#   split mode:
#   python C:\Data\Bot\Local_LLM\gold_research\jobs\build_historical_paper_trade_stream.py --mode split --signals-input C:\Data\Bot\YOUR_SOURCE\historical_signals.csv --trades-input C:\Data\Bot\YOUR_SOURCE\historical_trades.csv --outdir C:\Data\Bot\central_backtest_results\paper_trade_historical
#
#   merged mode:
#   python C:\Data\Bot\Local_LLM\gold_research\jobs\build_historical_paper_trade_stream.py --mode merged --merged-input C:\Data\Bot\YOUR_SOURCE\historical_stream.csv --outdir C:\Data\Bot\central_backtest_results\paper_trade_historical
# เวอร์ชัน: v1.0.1
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "v1.0.1"
TARGET_STRATEGY_ID = "locked_ms_bos_choch_pullback_atr_adx_ema_entry_v2_pullback_deep"
TARGET_TIMEFRAME = "M4"
TARGET_SIDE_POLICY = "BOTH"

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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ input | file={path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            return [dict(row) for row in reader]

    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                text = line.strip()
                if not text:
                    continue
                obj = json.loads(text)
                if not isinstance(obj, dict):
                    raise ValueError(f"JSONL row ต้องเป็น object | file={path} | line={line_no}")
                rows.append(obj)
        return rows

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            if not all(isinstance(x, dict) for x in payload):
                raise ValueError(f"JSON list ต้องเป็น list[object] | file={path}")
            return payload
        if isinstance(payload, dict):
            if "rows" in payload and isinstance(payload["rows"], list):
                if not all(isinstance(x, dict) for x in payload["rows"]):
                    raise ValueError(f"payload['rows'] ต้องเป็น list[object] | file={path}")
                return payload["rows"]
            raise ValueError(f"JSON object ต้องมี key='rows' หรือเป็น list โดยตรง | file={path}")
        raise ValueError(f"ไม่รองรับ JSON รูปแบบนี้ | file={path}")

    raise ValueError(f"รองรับเฉพาะ .csv .jsonl .json | file={path}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def normalize_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def normalize_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    candidates = [
        text,
        text.replace("Z", "+00:00") if text.endswith("Z") else text,
    ]

    for candidate in candidates:
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    return None


def to_iso_utc(value: Any, fallback: str = "") -> str:
    dt = parse_datetime(value)
    if dt is None:
        return fallback
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def trade_date_from_any(exit_time_utc: str, trade_date_value: Any) -> str:
    raw = normalize_str(trade_date_value)
    if raw:
        return raw
    dt = parse_datetime(exit_time_utc)
    if dt is None:
        return ""
    return dt.date().isoformat()


def hour_bucket_from_any(exit_time_utc: str, hour_bucket_value: Any) -> str:
    raw = normalize_str(hour_bucket_value)
    if raw:
        return raw.zfill(2)
    dt = parse_datetime(exit_time_utc)
    if dt is None:
        return ""
    return f"{dt.hour:02d}"


def first_present(row: dict[str, Any], candidates: list[str], default: Any = None) -> Any:
    for key in candidates:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def ensure_fields(row: dict[str, Any], required_fields: list[str], row_name: str) -> None:
    missing = [field for field in required_fields if field not in row or row[field] in (None, "")]
    if missing:
        raise ValueError(f"{row_name} missing required fields | missing={missing} | row={row}")


def normalize_side(value: Any) -> str:
    text = normalize_str(value).upper()
    if text in {"BUY", "LONG"}:
        return "BUY"
    if text in {"SELL", "SHORT"}:
        return "SELL"
    return text


def normalize_exit_reason(value: Any) -> str:
    text = normalize_str(value).lower()
    mapping = {
        "tp": "take_profit",
        "takeprofit": "take_profit",
        "take_profit": "take_profit",
        "sl": "stop_loss",
        "stoploss": "stop_loss",
        "stop_loss": "stop_loss",
        "manual": "manual_exit",
        "manual_exit": "manual_exit",
        "timeout": "time_exit",
        "time_exit": "time_exit",
    }
    return mapping.get(text, text)
def read_native_bucket(row: dict[str, Any], key_name: str) -> str:
    raw_value = row.get(key_name, "UNKNOWN")
    normalized = normalize_str(raw_value, "UNKNOWN")
    if not normalized:
        return "UNKNOWN"
    return normalized

def build_trade_row_from_source(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    entry_time_utc = to_iso_utc(
        first_present(row, ["entry_time_utc", "entry_time", "open_time", "entry_bar_time"]),
        fallback="",
    )
    exit_time_utc = to_iso_utc(
        first_present(row, ["exit_time_utc", "exit_time", "close_time", "exit_bar_time"]),
        fallback="",
    )

    trade_row = {
        "trade_id": normalize_str(
            first_present(row, ["trade_id", "position_id", "ticket", "id"], f"trade_{row_index:08d}")
        ),
        "signal_id": normalize_str(
            first_present(row, ["signal_id", "setup_id", "source_signal_id"], f"sig_{row_index:08d}")
        ),
        "entry_time_utc": entry_time_utc,
        "exit_time_utc": exit_time_utc,
        "entry_price": normalize_float(first_present(row, ["entry_price", "open_price"], 0.0)),
        "exit_price": normalize_float(first_present(row, ["exit_price", "close_price"], 0.0)),
        "pnl": normalize_float(first_present(row, ["pnl", "net_pnl", "profit"], 0.0)),
        "exit_reason": normalize_exit_reason(
            first_present(row, ["exit_reason", "close_reason", "reason"], "")
        ),
        "trade_date": trade_date_from_any(
            exit_time_utc,
            first_present(row, ["trade_date", "date"], ""),
        ),
        "hour_bucket": hour_bucket_from_any(
            exit_time_utc,
            first_present(row, ["hour_bucket", "exit_hour"], ""),
        ),
        "trend_bucket": read_native_bucket(row, "trend_bucket"),
        "volatility_bucket": read_native_bucket(row, "volatility_bucket"),
        "price_location_bucket": read_native_bucket(row, "price_location_bucket"),
    }

    ensure_fields(trade_row, TRADE_REQUIRED_FIELDS, f"trade_row[{row_index}]")
    return trade_row

# version: v1.0.5
# code name: build_historical_paper_trade_stream.py
# file path: C:\Data\Bot\Local_LLM\gold_research\jobs\build_historical_paper_trade_stream.py

def build_trade_row_from_source(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    entry_time_utc = to_iso_utc(
        first_present(
            row,
            ["entry_time_utc", "entry_time", "open_time", "entry_bar_time"],
        ),
        fallback="",
    )
    exit_time_utc = to_iso_utc(
        first_present(
            row,
            [
                "exit_time_utc",
                "effective_exit_time_utc",
                "exit_time",
                "close_time",
                "exit_bar_time",
            ],
        ),
        fallback="",
    )

    trade_row = {
        "trade_id": normalize_str(
            first_present(row, ["trade_id", "position_id", "ticket", "id"], f"trade_{row_index:08d}")
        ),
        "signal_id": normalize_str(
            first_present(row, ["signal_id", "setup_id", "source_signal_id"], f"sig_{row_index:08d}")
        ),
        "entry_time_utc": entry_time_utc,
        "exit_time_utc": exit_time_utc,
        "entry_price": normalize_float(first_present(row, ["entry_price", "open_price"], 0.0)),
        "exit_price": normalize_float(
            first_present(row, ["effective_exit_price", "exit_price", "close_price"], 0.0)
        ),
        "pnl": normalize_float(
            first_present(row, ["effective_pnl", "pnl", "net_pnl", "profit"], 0.0)
        ),
        "exit_reason": normalize_exit_reason(
            first_present(
                row,
                ["effective_exit_reason", "exit_reason", "close_reason", "reason"],
                "",
            )
        ),
        "trade_date": trade_date_from_any(
            exit_time_utc,
            first_present(row, ["trade_date", "date"], ""),
        ),
        "hour_bucket": hour_bucket_from_any(
            exit_time_utc,
            first_present(row, ["hour_bucket", "exit_hour"], ""),
        ),
        "trend_bucket": read_native_bucket(row, "trend_bucket"),
        "volatility_bucket": read_native_bucket(row, "volatility_bucket"),
        "price_location_bucket": read_native_bucket(row, "price_location_bucket"),
    }

    ensure_fields(trade_row, TRADE_REQUIRED_FIELDS, f"trade_row[{row_index}]")
    return trade_row


# version: v1.0.9
# code name: build_historical_paper_trade_stream.py
# file path: C:\Data\Bot\Local_LLM\gold_research\jobs\build_historical_paper_trade_stream.py

def filter_candidate_signal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if normalize_str(row["strategy_id"]) != TARGET_STRATEGY_ID:
            continue

        row_timeframe = normalize_str(row.get("timeframe", ""), "")
        if row_timeframe and row_timeframe != TARGET_TIMEFRAME:
            continue

        side = normalize_side(row["side"])
        if TARGET_SIDE_POLICY == "LONG_ONLY" and side != "BUY":
            continue
        if TARGET_SIDE_POLICY == "SHORT_ONLY" and side != "SELL":
            continue

        approved_value = normalize_bool(row.get("approved", True), True)
        if not approved_value:
            continue

        filtered.append(row)
    return filtered

# version: v1.0.6
# code name: build_historical_paper_trade_stream.py
# file path: C:\Data\Bot\Local_LLM\gold_research\jobs\build_historical_paper_trade_stream.py

def read_native_bucket(row: dict[str, Any], key_name: str) -> str:
    raw_value = row.get(key_name, "UNKNOWN")
    normalized = normalize_str(raw_value, "UNKNOWN")
    if not normalized:
        return "UNKNOWN"
    return normalized


def build_signal_row_from_source(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    ts_utc = to_iso_utc(
        first_present(
            row,
            [
                "ts_utc",
                "signal_time_utc",
                "signal_time",
                "timestamp",
                "entry_signal_time",
                "entry_time",
                "entry_bar_time",
            ],
        ),
        fallback="",
    )

    side = normalize_side(first_present(row, ["side", "signal_side", "direction"], "BUY"))

    signal_row = {
        "signal_id": normalize_str(
            first_present(row, ["signal_id", "setup_id", "id"], f"sig_{row_index:08d}")
        ),
        "ts_utc": ts_utc,
        "strategy_id": normalize_str(
            first_present(row, ["strategy_id", "strategy", "strategy_name"], TARGET_STRATEGY_ID),
            TARGET_STRATEGY_ID,
        ),
        "timeframe": normalize_str(
            first_present(row, ["timeframe", "tf"], TARGET_TIMEFRAME),
            TARGET_TIMEFRAME,
        ),
        "side": side,
        "trend_bucket": read_native_bucket(row, "trend_bucket"),
        "volatility_bucket": read_native_bucket(row, "volatility_bucket"),
        "price_location_bucket": read_native_bucket(row, "price_location_bucket"),
        "approved": normalize_bool(
            first_present(row, ["approved", "is_approved", "paper_approved"], True),
            True,
        ),
    }

    ensure_fields(signal_row, SIGNAL_REQUIRED_FIELDS, f"signal_row[{row_index}]")
    return signal_row


# version: v1.0.8
# code name: build_historical_paper_trade_stream.py
# file path: C:\Data\Bot\Local_LLM\gold_research\jobs\build_historical_paper_trade_stream.py

def build_signal_row_from_source(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    ts_utc = to_iso_utc(
        first_present(
            row,
            [
                "ts_utc",
                "entry_time_utc",
                "ts",
                "timestamp",
                "signal_time_utc",
                "signal_time",
                "entry_signal_time",
                "entry_time",
                "entry_bar_time",
                "bar_time",
                "opened_at",
                "open_time",
            ],
        ),
        fallback="",
    )

    side = normalize_side(first_present(row, ["side", "signal_side", "direction"], "BUY"))

    strategy_value = normalize_str(
        first_present(row, ["strategy_id", "strategy", "strategy_name"], TARGET_STRATEGY_ID),
        TARGET_STRATEGY_ID,
    )

    timeframe_value = normalize_str(
        first_present(row, ["timeframe", "tf"], ""),
        "",
    )
    if not timeframe_value:
        timeframe_value = "M4"

    signal_row = {
        "signal_id": normalize_str(
            first_present(row, ["signal_id", "setup_id", "id"], f"sig_{row_index:08d}")
        ),
        "ts_utc": ts_utc,
        "strategy_id": strategy_value,
        "timeframe": timeframe_value,
        "side": side,
        "trend_bucket": read_native_bucket(row, "trend_bucket"),
        "volatility_bucket": read_native_bucket(row, "volatility_bucket"),
        "price_location_bucket": read_native_bucket(row, "price_location_bucket"),
        "approved": normalize_bool(
            first_present(
                row,
                ["approved", "is_approved", "paper_approved", "included_in_variant"],
                True,
            ),
            True,
        ),
    }

    ensure_fields(signal_row, SIGNAL_REQUIRED_FIELDS, f"signal_row[{row_index}]")
    return signal_row

def filter_candidate_trade_rows(rows: list[dict[str, Any]], signal_ids: set[str]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if normalize_str(row["signal_id"]) not in signal_ids:
            continue
        filtered.append(row)
    return filtered


def deduplicate_by_key(rows: list[dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key_value = normalize_str(row.get(key_name))
        if not key_value:
            continue
        if key_value in seen:
            continue
        seen.add(key_value)
        deduped.append(row)
    return deduped


def build_from_split(signals_input: Path, trades_input: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    signal_source_rows = read_records(signals_input)
    trade_source_rows = read_records(trades_input)

    signal_rows = [
        build_signal_row_from_source(row, row_index=i + 1)
        for i, row in enumerate(signal_source_rows)
    ]
    trade_rows = [
        build_trade_row_from_source(row, row_index=i + 1)
        for i, row in enumerate(trade_source_rows)
    ]

    signal_rows = deduplicate_by_key(signal_rows, "signal_id")
    signal_rows = filter_candidate_signal_rows(signal_rows)

    signal_ids = {normalize_str(row["signal_id"]) for row in signal_rows}
    trade_rows = deduplicate_by_key(trade_rows, "trade_id")
    trade_rows = filter_candidate_trade_rows(trade_rows, signal_ids)

    return signal_rows, trade_rows


def build_from_merged(merged_input: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged_rows = read_records(merged_input)

    signal_rows = [
        build_signal_row_from_source(row, row_index=i + 1)
        for i, row in enumerate(merged_rows)
    ]
    trade_rows = [
        build_trade_row_from_source(row, row_index=i + 1)
        for i, row in enumerate(merged_rows)
    ]

    signal_rows = deduplicate_by_key(signal_rows, "signal_id")
    signal_rows = filter_candidate_signal_rows(signal_rows)

    signal_ids = {normalize_str(row["signal_id"]) for row in signal_rows}
    trade_rows = deduplicate_by_key(trade_rows, "trade_id")
    trade_rows = filter_candidate_trade_rows(trade_rows, signal_ids)

    return signal_rows, trade_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build historical paper-trade stream for monitor pipeline.")
    parser.add_argument("--mode", required=True, choices=["split", "merged"], help="Input mode")
    parser.add_argument("--signals-input", help="Historical signals source file (.csv/.jsonl/.json)")
    parser.add_argument("--trades-input", help="Historical trades source file (.csv/.jsonl/.json)")
    parser.add_argument("--merged-input", help="Merged historical source file (.csv/.jsonl/.json)")
    parser.add_argument("--outdir", required=True, help="Output directory for signals.jsonl and trades.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir)
    signals_out = outdir / "signals.jsonl"
    trades_out = outdir / "trades.jsonl"
    summary_out = outdir / "historical_stream_build_summary.json"

    print("=" * 120)
    print(f"[INFO] build_historical_paper_trade_stream.py version={VERSION}")
    print(f"[INFO] mode={args.mode}")
    print(f"[INFO] outdir={outdir}")
    print("=" * 120)

    if args.mode == "split":
        if not args.signals_input or not args.trades_input:
            raise ValueError("mode=split ต้องส่ง --signals-input และ --trades-input")
        signal_rows, trade_rows = build_from_split(
            signals_input=Path(args.signals_input),
            trades_input=Path(args.trades_input),
        )
    else:
        if not args.merged_input:
            raise ValueError("mode=merged ต้องส่ง --merged-input")
        signal_rows, trade_rows = build_from_merged(Path(args.merged_input))

    write_jsonl(signals_out, signal_rows)
    write_jsonl(trades_out, trade_rows)

    summary = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "target_strategy_id": TARGET_STRATEGY_ID,
        "target_timeframe": TARGET_TIMEFRAME,
        "target_side_policy": TARGET_SIDE_POLICY,
        "mode": args.mode,
        "signals_output": str(signals_out),
        "trades_output": str(trades_out),
        "signals_rows": len(signal_rows),
        "trades_rows": len(trade_rows),
        "notes": [
            "ไฟล์นี้สร้าง historical paper-trade stream สำหรับ monitor pipeline",
            "ยังไม่ได้คำนวณ strategy จาก OHLC ดิบโดยตรง",
            "ใช้สำหรับแปลง historical outputs ให้เข้า signals.jsonl และ trades.jsonl",
        ],
    }
    write_json(summary_out, summary)

    print(f"[DONE] signals_output={signals_out}")
    print(f"[DONE] trades_output={trades_out}")
    print(f"[DONE] signals_rows={len(signal_rows)}")
    print(f"[DONE] trades_rows={len(trade_rows)}")
    print(f"[DONE] summary={summary_out}")
    print("=" * 120)


if __name__ == "__main__":
    main()
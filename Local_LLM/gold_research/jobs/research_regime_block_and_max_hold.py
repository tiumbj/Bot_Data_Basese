# version: v1.0.0
# file: C:\Data\Bot\Local_LLM\gold_research\jobs\research_regime_block_and_max_hold.py
# run:
# python C:\Data\Bot\Local_LLM\gold_research\jobs\research_regime_block_and_max_hold.py --input C:\Data\Bot\central_backtest_results\paper_trade_historical\trades.jsonl --outdir C:\Data\Bot\central_backtest_results\research_regime_max_hold

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "v1.0.0"


@dataclass
class TradeRow:
    trade_id: str
    signal_id: str
    entry_time_utc: str
    exit_time_utc: str
    entry_price: float
    exit_price: float
    pnl: float
    exit_reason: str
    trade_date: str
    hour_bucket: str
    trend_bucket: str
    volatility_bucket: str
    price_location_bucket: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research regime-block + max-hold proxy variants from trades.jsonl")
    parser.add_argument("--input", required=True, help="Path to trades.jsonl")
    parser.add_argument("--outdir", required=True, help="Output directory")
    return parser.parse_args()


def parse_utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def hold_hours(row: TradeRow) -> float:
    return (parse_utc(row.exit_time_utc) - parse_utc(row.entry_time_utc)).total_seconds() / 3600.0


def is_blocked_regime(row: TradeRow) -> bool:
    if row.price_location_bucket == "INSIDE_EMA_STACK":
        return True
    if (
        row.trend_bucket == "BULL_TREND"
        and row.volatility_bucket == "LOW_VOL"
        and row.price_location_bucket == "ABOVE_EMA_STACK"
    ):
        return True
    return False


def load_rows(path: Path) -> list[TradeRow]:
    rows: list[TradeRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            rows.append(
                TradeRow(
                    trade_id=str(raw["trade_id"]),
                    signal_id=str(raw["signal_id"]),
                    entry_time_utc=str(raw["entry_time_utc"]),
                    exit_time_utc=str(raw["exit_time_utc"]),
                    entry_price=float(raw["entry_price"]),
                    exit_price=float(raw["exit_price"]),
                    pnl=float(raw["pnl"]),
                    exit_reason=str(raw["exit_reason"]),
                    trade_date=str(raw["trade_date"]),
                    hour_bucket=str(raw["hour_bucket"]),
                    trend_bucket=str(raw["trend_bucket"]),
                    volatility_bucket=str(raw["volatility_bucket"]),
                    price_location_bucket=str(raw["price_location_bucket"]),
                )
            )
    return rows


def max_consecutive_losses(rows: list[dict[str, Any]]) -> int:
    best = 0
    cur = 0
    for row in rows:
        if float(row["effective_pnl"]) <= 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def summarize_variant(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    trades = len(rows)
    wins = sum(1 for r in rows if float(r["effective_pnl"]) > 0)
    losses = trades - wins
    pnl_sum = round(sum(float(r["effective_pnl"]) for r in rows), 4)

    gross_profit = sum(float(r["effective_pnl"]) for r in rows if float(r["effective_pnl"]) > 0)
    gross_loss_abs = abs(sum(float(r["effective_pnl"]) for r in rows if float(r["effective_pnl"]) <= 0))
    payoff_ratio = round(gross_profit / gross_loss_abs, 4) if gross_loss_abs > 0 else 0.0

    tp_count = sum(1 for r in rows if r["effective_exit_reason"] == "take_profit")
    sl_count = sum(1 for r in rows if r["effective_exit_reason"] == "stop_loss")
    mh_count = sum(1 for r in rows if r["effective_exit_reason"] == "max_hold_proxy")

    original_tp_cut = sum(
        1 for r in rows
        if r["time_stop_candidate"] and r["original_exit_reason"] == "take_profit"
    )
    original_sl_cut = sum(
        1 for r in rows
        if r["time_stop_candidate"] and r["original_exit_reason"] == "stop_loss"
    )

    return {
        "variant": name,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round((wins * 100.0 / trades), 2) if trades > 0 else 0.0,
        "pnl_sum": pnl_sum,
        "payoff_ratio": payoff_ratio,
        "max_consecutive_losses": max_consecutive_losses(rows),
        "take_profit_count": tp_count,
        "stop_loss_count": sl_count,
        "max_hold_proxy_count": mh_count,
        "max_hold_proxy_from_original_tp": original_tp_cut,
        "max_hold_proxy_from_original_sl": original_sl_cut,
    }


def build_baseline_filtered(rows: list[TradeRow]) -> list[TradeRow]:
    return [row for row in rows if not is_blocked_regime(row)]


def build_max_hold_proxy(rows: list[TradeRow], threshold_hours: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        hh = hold_hours(row)
        candidate = hh > threshold_hours

        output.append(
            {
                "trade_id": row.trade_id,
                "signal_id": row.signal_id,
                "entry_time_utc": row.entry_time_utc,
                "exit_time_utc": row.exit_time_utc,
                "hold_hours": round(hh, 2),
                "trend_bucket": row.trend_bucket,
                "volatility_bucket": row.volatility_bucket,
                "price_location_bucket": row.price_location_bucket,
                "original_exit_reason": row.exit_reason,
                "effective_exit_reason": "max_hold_proxy" if candidate else row.exit_reason,
                "time_stop_candidate": candidate,
                "original_pnl": row.pnl,
                "effective_pnl": row.pnl,  # proxy stage: do not invent exit price
            }
        )
    return output


def build_baseline_rows(rows: list[TradeRow]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                "trade_id": row.trade_id,
                "signal_id": row.signal_id,
                "entry_time_utc": row.entry_time_utc,
                "exit_time_utc": row.exit_time_utc,
                "hold_hours": round(hold_hours(row), 2),
                "trend_bucket": row.trend_bucket,
                "volatility_bucket": row.volatility_bucket,
                "price_location_bucket": row.price_location_bucket,
                "original_exit_reason": row.exit_reason,
                "effective_exit_reason": row.exit_reason,
                "time_stop_candidate": False,
                "original_pnl": row.pnl,
                "effective_pnl": row.pnl,
            }
        )
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    baseline_filtered = build_baseline_filtered(rows)

    baseline_payload = build_baseline_rows(baseline_filtered)
    max_hold_6h_payload = build_max_hold_proxy(baseline_filtered, 6.0)
    max_hold_4h_payload = build_max_hold_proxy(baseline_filtered, 4.0)

    write_jsonl(outdir / "baseline_filtered.jsonl", baseline_payload)
    write_jsonl(outdir / "baseline_filtered_max_hold_6h_proxy.jsonl", max_hold_6h_payload)
    write_jsonl(outdir / "baseline_filtered_max_hold_4h_proxy.jsonl", max_hold_4h_payload)

    summaries = [
        summarize_variant("baseline_filtered", baseline_payload),
        summarize_variant("baseline_filtered_max_hold_6h_proxy", max_hold_6h_payload),
        summarize_variant("baseline_filtered_max_hold_4h_proxy", max_hold_4h_payload),
    ]

    with (outdir / "variant_results.jsonl").open("w", encoding="utf-8") as f:
        for row in summaries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "version": VERSION,
        "input": str(input_path),
        "output_dir": str(outdir),
        "rows_loaded": len(rows),
        "rows_after_regime_block": len(baseline_filtered),
        "variants": summaries,
    }
    write_json(outdir / "variant_report.json", report)

    print(f"[DONE] version={VERSION}")
    print(f"[DONE] rows_loaded={len(rows)}")
    print(f"[DONE] rows_after_regime_block={len(baseline_filtered)}")
    print(f"[DONE] outdir={outdir}")


if __name__ == "__main__":
    main()
# version: v1.0.0
# file: C:\Data\Bot\Local_LLM\gold_research\jobs\real_max_hold_replay.py
# run:
# python C:\Data\Bot\Local_LLM\gold_research\jobs\real_max_hold_replay.py --trades C:\Data\Bot\central_backtest_results\paper_trade_historical\trades.jsonl --ohlc C:\Data\Bot\YOUR_OHLC_FILE.csv --outdir C:\Data\Bot\central_backtest_results\real_max_hold_replay

from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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


@dataclass
class OhlcBar:
    ts: datetime
    close: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real replay for regime block + max hold exits")
    parser.add_argument("--trades", required=True, help="Path to trades.jsonl")
    parser.add_argument("--ohlc", required=True, help="Path to OHLC csv")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument(
        "--time-column",
        default="time",
        help="OHLC time column name. Common values: time, datetime, timestamp, Date",
    )
    parser.add_argument(
        "--close-column",
        default="close",
        help="OHLC close column name. Common values: close, Close",
    )
    return parser.parse_args()


def parse_utc(ts: str) -> datetime:
    ts = ts.strip()
    if ts.endswith("Z"):
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def load_trades(path: Path) -> list[TradeRow]:
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


def load_ohlc(path: Path, time_column: str, close_column: str) -> tuple[list[datetime], list[OhlcBar]]:
    bars: list[OhlcBar] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if time_column not in cols:
            raise ValueError(f"OHLC missing time column: {time_column}. Available={sorted(cols)}")
        if close_column not in cols:
            raise ValueError(f"OHLC missing close column: {close_column}. Available={sorted(cols)}")

        for row in reader:
            raw_ts = str(row[time_column]).strip()
            raw_close = str(row[close_column]).strip()
            if not raw_ts or not raw_close:
                continue
            ts = parse_utc(raw_ts)
            close = float(raw_close)
            bars.append(OhlcBar(ts=ts, close=close))

    bars.sort(key=lambda x: x.ts)
    index = [bar.ts for bar in bars]
    return index, bars


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


def classify_side(row: TradeRow) -> str:
    # ใช้ข้อมูลจริงจาก pnl และ entry/exit เพื่อเดาว่าเป็น long หรือ short
    # ถ้า exit > entry และ pnl > 0 => long
    # ถ้า exit < entry และ pnl > 0 => short
    if row.pnl > 0:
        return "LONG" if row.exit_price > row.entry_price else "SHORT"
    if row.pnl < 0:
        return "LONG" if row.exit_price < row.entry_price else "SHORT"
    # fallback ถ้า pnl == 0
    return "LONG"


def recompute_pnl(entry_price: float, exit_price: float, side: str) -> float:
    if side == "LONG":
        return exit_price - entry_price
    if side == "SHORT":
        return entry_price - exit_price
    raise ValueError(f"Unknown side: {side}")


def max_consecutive_losses(rows: list[dict[str, Any]]) -> int:
    best = 0
    cur = 0
    for row in rows:
        if float(row["effective_pnl"]) <= 0:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def find_first_bar_at_or_after(target_ts: datetime, ts_index: list[datetime], bars: list[OhlcBar]) -> OhlcBar | None:
    pos = bisect_left(ts_index, target_ts)
    if pos >= len(bars):
        return None
    return bars[pos]


def build_effective_row_baseline(row: TradeRow) -> dict[str, Any]:
    return {
        "trade_id": row.trade_id,
        "signal_id": row.signal_id,
        "entry_time_utc": row.entry_time_utc,
        "effective_exit_time_utc": row.exit_time_utc,
        "entry_price": row.entry_price,
        "effective_exit_price": row.exit_price,
        "side": classify_side(row),
        "original_exit_reason": row.exit_reason,
        "effective_exit_reason": row.exit_reason,
        "time_stop_applied": False,
        "hold_hours_original": round((parse_utc(row.exit_time_utc) - parse_utc(row.entry_time_utc)).total_seconds() / 3600.0, 2),
        "trend_bucket": row.trend_bucket,
        "volatility_bucket": row.volatility_bucket,
        "price_location_bucket": row.price_location_bucket,
        "original_pnl": row.pnl,
        "effective_pnl": row.pnl,
    }


def build_effective_row_max_hold(
    row: TradeRow,
    threshold_hours: float,
    ts_index: list[datetime],
    bars: list[OhlcBar],
) -> dict[str, Any]:
    entry_dt = parse_utc(row.entry_time_utc)
    exit_dt = parse_utc(row.exit_time_utc)
    original_hold_hours = (exit_dt - entry_dt).total_seconds() / 3600.0
    side = classify_side(row)

    if original_hold_hours <= threshold_hours:
        return build_effective_row_baseline(row)

    threshold_dt = entry_dt + timedelta(hours=threshold_hours)
    bar = find_first_bar_at_or_after(threshold_dt, ts_index, bars)

    # ถ้า OHLC ไม่พอจนหา bar ไม่เจอ ให้ fallback เป็นของเดิมและ mark reason
    if bar is None:
        out = build_effective_row_baseline(row)
        out["effective_exit_reason"] = "original_exit_no_bar_after_threshold"
        out["time_stop_applied"] = False
        return out

    effective_exit_price = float(bar.close)
    effective_exit_time_utc = iso_z(bar.ts)
    effective_pnl = recompute_pnl(row.entry_price, effective_exit_price, side)

    return {
        "trade_id": row.trade_id,
        "signal_id": row.signal_id,
        "entry_time_utc": row.entry_time_utc,
        "effective_exit_time_utc": effective_exit_time_utc,
        "entry_price": row.entry_price,
        "effective_exit_price": effective_exit_price,
        "side": side,
        "original_exit_reason": row.exit_reason,
        "effective_exit_reason": "max_hold_exit",
        "time_stop_applied": True,
        "hold_hours_original": round(original_hold_hours, 2),
        "trend_bucket": row.trend_bucket,
        "volatility_bucket": row.volatility_bucket,
        "price_location_bucket": row.price_location_bucket,
        "original_pnl": row.pnl,
        "effective_pnl": round(effective_pnl, 10),
    }


def summarize_variant(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    trades = len(rows)
    wins = sum(1 for r in rows if float(r["effective_pnl"]) > 0)
    losses = trades - wins
    pnl_sum = round(sum(float(r["effective_pnl"]) for r in rows), 4)

    gross_profit = sum(float(r["effective_pnl"]) for r in rows if float(r["effective_pnl"]) > 0)
    gross_loss_abs = abs(sum(float(r["effective_pnl"]) for r in rows if float(r["effective_pnl"]) <= 0))
    payoff_ratio = round(gross_profit / gross_loss_abs, 4) if gross_loss_abs > 0 else 0.0

    max_hold_count = sum(1 for r in rows if r["effective_exit_reason"] == "max_hold_exit")
    max_hold_from_tp = sum(
        1 for r in rows
        if r["effective_exit_reason"] == "max_hold_exit" and r["original_exit_reason"] == "take_profit"
    )
    max_hold_from_sl = sum(
        1 for r in rows
        if r["effective_exit_reason"] == "max_hold_exit" and r["original_exit_reason"] == "stop_loss"
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
        "take_profit_count_effective": sum(1 for r in rows if r["effective_exit_reason"] == "take_profit"),
        "stop_loss_count_effective": sum(1 for r in rows if r["effective_exit_reason"] == "stop_loss"),
        "max_hold_exit_count": max_hold_count,
        "max_hold_exit_from_original_tp": max_hold_from_tp,
        "max_hold_exit_from_original_sl": max_hold_from_sl,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    trades_path = Path(args.trades)
    ohlc_path = Path(args.ohlc)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    trades = load_trades(trades_path)
    ts_index, bars = load_ohlc(ohlc_path, args.time_column, args.close_column)

    filtered_trades = [row for row in trades if not is_blocked_regime(row)]

    baseline_rows = [build_effective_row_baseline(row) for row in filtered_trades]
    max_hold_6h_rows = [build_effective_row_max_hold(row, 6.0, ts_index, bars) for row in filtered_trades]
    max_hold_4h_rows = [build_effective_row_max_hold(row, 4.0, ts_index, bars) for row in filtered_trades]

    baseline_summary = summarize_variant("baseline_filtered", baseline_rows)
    max_hold_6h_summary = summarize_variant("baseline_filtered_max_hold_6h", max_hold_6h_rows)
    max_hold_4h_summary = summarize_variant("baseline_filtered_max_hold_4h", max_hold_4h_rows)

    write_jsonl(outdir / "baseline_filtered.jsonl", baseline_rows)
    write_jsonl(outdir / "baseline_filtered_max_hold_6h.jsonl", max_hold_6h_rows)
    write_jsonl(outdir / "baseline_filtered_max_hold_4h.jsonl", max_hold_4h_rows)

    with (outdir / "variant_results.jsonl").open("w", encoding="utf-8") as f:
        for row in [baseline_summary, max_hold_6h_summary, max_hold_4h_summary]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "version": VERSION,
        "trades_path": str(trades_path),
        "ohlc_path": str(ohlc_path),
        "rows_loaded": len(trades),
        "rows_after_regime_block": len(filtered_trades),
        "variants": [
            baseline_summary,
            max_hold_6h_summary,
            max_hold_4h_summary,
        ],
    }
    write_json(outdir / "variant_report.json", report)

    print(f"[DONE] version={VERSION}")
    print(f"[DONE] rows_loaded={len(trades)}")
    print(f"[DONE] rows_after_regime_block={len(filtered_trades)}")
    print(f"[DONE] outdir={outdir}")


if __name__ == "__main__":
    main()
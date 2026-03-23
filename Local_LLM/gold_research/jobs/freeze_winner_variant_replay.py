# version: v1.0.0
# file: C:\Data\Bot\Local_LLM\gold_research\jobs\freeze_winner_variant_replay.py
# run:
# python C:\Data\Bot\Local_LLM\gold_research\jobs\freeze_winner_variant_replay.py --trades C:\Data\Bot\central_backtest_results\paper_trade_historical\trades.jsonl --ohlc "C:\Data\Bot\central_market_data\tf\XAUUSD_M30.csv" --outdir C:\Data\Bot\central_backtest_results\winner_variant_freeze --time-column datetime --close-column close

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
WINNER_STRATEGY_ID = (
    "locked_ms_bos_choch_pullback_atr_adx_ema_entry_v2_pullback_deep_"
    "regime_block_conditional_max_hold_4h_cooldown_2loss_skip1"
)


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
    parser = argparse.ArgumentParser(description="Freeze winner variant replay into final auditable artifacts")
    parser.add_argument("--trades", required=True, help="Path to trades.jsonl")
    parser.add_argument("--ohlc", required=True, help="Path to OHLC csv")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--time-column", default="datetime", help="OHLC time column name")
    parser.add_argument("--close-column", default="close", help="OHLC close column name")
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


def utc_now_iso() -> str:
    return iso_z(datetime.now(timezone.utc))


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
    rows.sort(key=lambda r: parse_utc(r.entry_time_utc))
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
            bars.append(OhlcBar(ts=parse_utc(raw_ts), close=float(raw_close)))

    bars.sort(key=lambda x: x.ts)
    ts_index = [bar.ts for bar in bars]
    return ts_index, bars


def is_blocked_regime(row: TradeRow) -> tuple[bool, str]:
    if row.price_location_bucket == "INSIDE_EMA_STACK":
        return True, "INSIDE_EMA_STACK"
    if (
        row.trend_bucket == "BULL_TREND"
        and row.volatility_bucket == "LOW_VOL"
        and row.price_location_bucket == "ABOVE_EMA_STACK"
    ):
        return True, "BULL_TREND_LOW_VOL_ABOVE_EMA_STACK"
    return False, ""


def classify_side(row: TradeRow) -> str:
    if row.pnl > 0:
        return "LONG" if row.exit_price > row.entry_price else "SHORT"
    if row.pnl < 0:
        return "LONG" if row.exit_price < row.entry_price else "SHORT"
    return "LONG"


def recompute_pnl(entry_price: float, exit_price: float, side: str) -> float:
    if side == "LONG":
        return exit_price - entry_price
    if side == "SHORT":
        return entry_price - exit_price
    raise ValueError(f"Unknown side: {side}")


def find_first_bar_at_or_after(target_ts: datetime, ts_index: list[datetime], bars: list[OhlcBar]) -> OhlcBar | None:
    pos = bisect_left(ts_index, target_ts)
    if pos >= len(bars):
        return None
    return bars[pos]


def make_base_row(row: TradeRow) -> dict[str, Any]:
    hold_hours_original = round(
        (parse_utc(row.exit_time_utc) - parse_utc(row.entry_time_utc)).total_seconds() / 3600.0, 2
    )
    return {
        "strategy_id": WINNER_STRATEGY_ID,
        "trade_id": row.trade_id,
        "signal_id": row.signal_id,
        "entry_time_utc": row.entry_time_utc,
        "effective_exit_time_utc": row.exit_time_utc,
        "entry_price": row.entry_price,
        "effective_exit_price": row.exit_price,
        "side": classify_side(row),
        "original_exit_reason": row.exit_reason,
        "effective_exit_reason": row.exit_reason,
        "conditional_time_stop_applied": False,
        "conditional_max_hold_hours": 4,
        "threshold_bar_time_utc": None,
        "threshold_bar_close": None,
        "threshold_bar_pnl": None,
        "hold_hours_original": hold_hours_original,
        "trend_bucket": row.trend_bucket,
        "volatility_bucket": row.volatility_bucket,
        "price_location_bucket": row.price_location_bucket,
        "original_pnl": row.pnl,
        "effective_pnl": row.pnl,
        "regime_blocked": False,
        "blocked_reason": "",
        "cooldown_skipped_signal": False,
        "cooldown_rule_name": "",
        "included_in_variant": True,
    }


def apply_conditional_max_hold_4h(
    row: TradeRow,
    ts_index: list[datetime],
    bars: list[OhlcBar],
) -> dict[str, Any]:
    out = make_base_row(row)
    entry_dt = parse_utc(row.entry_time_utc)
    exit_dt = parse_utc(row.exit_time_utc)
    original_hold_hours = (exit_dt - entry_dt).total_seconds() / 3600.0

    if original_hold_hours <= 4.0:
        return out

    threshold_dt = entry_dt + timedelta(hours=4)
    threshold_bar = find_first_bar_at_or_after(threshold_dt, ts_index, bars)
    if threshold_bar is None:
        out["effective_exit_reason"] = "original_exit_no_bar_after_threshold"
        return out

    threshold_exit_price = float(threshold_bar.close)
    threshold_pnl = recompute_pnl(row.entry_price, threshold_exit_price, out["side"])

    out["threshold_bar_time_utc"] = iso_z(threshold_bar.ts)
    out["threshold_bar_close"] = threshold_exit_price
    out["threshold_bar_pnl"] = round(threshold_pnl, 10)

    if threshold_pnl <= 0:
        out["effective_exit_time_utc"] = iso_z(threshold_bar.ts)
        out["effective_exit_price"] = threshold_exit_price
        out["effective_exit_reason"] = "conditional_max_hold_exit"
        out["conditional_time_stop_applied"] = True
        out["effective_pnl"] = round(threshold_pnl, 10)

    return out


def max_consecutive_losses(rows: list[dict[str, Any]]) -> int:
    best = 0
    cur = 0
    for row in rows:
        if not row.get("included_in_variant", True):
            continue
        if float(row["effective_pnl"]) <= 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    active_rows = [r for r in rows if r.get("included_in_variant", True)]
    trades = len(active_rows)
    wins = sum(1 for r in active_rows if float(r["effective_pnl"]) > 0)
    losses = trades - wins
    pnl_sum = round(sum(float(r["effective_pnl"]) for r in active_rows), 4)

    gross_profit = sum(float(r["effective_pnl"]) for r in active_rows if float(r["effective_pnl"]) > 0)
    gross_loss_abs = abs(sum(float(r["effective_pnl"]) for r in active_rows if float(r["effective_pnl"]) <= 0))
    payoff_ratio = round(gross_profit / gross_loss_abs, 4) if gross_loss_abs > 0 else 0.0

    conditional_exit_count = sum(1 for r in active_rows if r["effective_exit_reason"] == "conditional_max_hold_exit")
    conditional_from_tp = sum(
        1 for r in active_rows
        if r["effective_exit_reason"] == "conditional_max_hold_exit" and r["original_exit_reason"] == "take_profit"
    )
    conditional_from_sl = sum(
        1 for r in active_rows
        if r["effective_exit_reason"] == "conditional_max_hold_exit" and r["original_exit_reason"] == "stop_loss"
    )

    return {
        "strategy_id": WINNER_STRATEGY_ID,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round((wins * 100.0 / trades), 2) if trades > 0 else 0.0,
        "pnl_sum": pnl_sum,
        "payoff_ratio": payoff_ratio,
        "max_consecutive_losses": max_consecutive_losses(rows),
        "take_profit_count_effective": sum(1 for r in active_rows if r["effective_exit_reason"] == "take_profit"),
        "stop_loss_count_effective": sum(1 for r in active_rows if r["effective_exit_reason"] == "stop_loss"),
        "conditional_max_hold_exit_count": conditional_exit_count,
        "conditional_max_hold_exit_from_original_tp": conditional_from_tp,
        "conditional_max_hold_exit_from_original_sl": conditional_from_sl,
        "cooldown_skipped_count": sum(1 for r in rows if r.get("cooldown_skipped_signal", False)),
        "regime_blocked_count": sum(1 for r in rows if r.get("regime_blocked", False)),
    }


def main() -> None:
    args = parse_args()
    trades_path = Path(args.trades)
    ohlc_path = Path(args.ohlc)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    trades = load_trades(trades_path)
    ts_index, bars = load_ohlc(ohlc_path, args.time_column, args.close_column)

    audit_rows: list[dict[str, Any]] = []
    active_rows: list[dict[str, Any]] = []

    consecutive_losses = 0
    skip_remaining = 0

    for row in trades:
        blocked, blocked_reason = is_blocked_regime(row)
        if blocked:
            out = make_base_row(row)
            out["regime_blocked"] = True
            out["blocked_reason"] = blocked_reason
            out["included_in_variant"] = False
            audit_rows.append(out)
            continue

        candidate = apply_conditional_max_hold_4h(row, ts_index, bars)

        if skip_remaining > 0:
            candidate["included_in_variant"] = False
            candidate["cooldown_skipped_signal"] = True
            candidate["cooldown_rule_name"] = "after_2_losses_skip_1"
            audit_rows.append(candidate)
            skip_remaining -= 1
            continue

        candidate["included_in_variant"] = True
        candidate["cooldown_skipped_signal"] = False
        candidate["cooldown_rule_name"] = "after_2_losses_skip_1"
        audit_rows.append(candidate)
        active_rows.append(candidate)

        if float(candidate["effective_pnl"]) <= 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0

        if consecutive_losses >= 2:
            skip_remaining = 1
            consecutive_losses = 0

    summary = summarize(audit_rows)

    write_jsonl(outdir / "winner_variant_audit_all_rows.jsonl", audit_rows)
    write_jsonl(outdir / "winner_variant_active_rows.jsonl", active_rows)
    write_json(
        outdir / "winner_variant_summary.json",
        {
            "version": VERSION,
            "generated_at_utc": utc_now_iso(),
            "strategy_id": WINNER_STRATEGY_ID,
            "trades_path": str(trades_path),
            "ohlc_path": str(ohlc_path),
            "summary": summary,
            "notes": [
                "winner variant ถูก freeze จากผลวิจัยจริง",
                "ใช้ regime block + conditional max hold 4h if pnl <= 0 + cooldown after 2 losses skip 1",
                "ไฟล์ audit_all_rows เก็บทั้ง included / blocked / skipped เพื่อใช้ตรวจย้อนหลัง",
                "ไฟล์ active_rows เก็บเฉพาะ trades ที่นับจริงใน winner variant",
            ],
        },
    )

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] strategy_id={WINNER_STRATEGY_ID}")
    print(f"[DONE] rows_loaded={len(trades)}")
    print(f"[DONE] active_rows={len(active_rows)}")
    print(f"[DONE] summary={outdir / 'winner_variant_summary.json'}")
    print("=" * 120)


if __name__ == "__main__":
    main()
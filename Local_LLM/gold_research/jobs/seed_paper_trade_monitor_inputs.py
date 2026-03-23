# ============================================================
# ชื่อโค้ด: seed_paper_trade_monitor_inputs.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\seed_paper_trade_monitor_inputs.py
# คำสั่งรัน:
#   python C:\Data\Bot\Local_LLM\gold_research\jobs\seed_paper_trade_monitor_inputs.py --outdir C:\Data\Bot\central_backtest_results\paper_trade
# เวอร์ชัน: v1.0.0
# ============================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path


VERSION = "v1.0.0"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_sample_signals() -> list[dict]:
    return [
        {
            "signal_id": "sig_20260310_001",
            "ts_utc": "2026-03-10T02:30:00Z",
            "strategy_id": "locked_ms_bos_choch_pullback_atr_adx_ema_entry_v2_pullback_deep",
            "timeframe": "M30",
            "side": "BUY",
            "trend_bucket": "BULL_TREND",
            "volatility_bucket": "HIGH_VOL",
            "price_location_bucket": "ABOVE_EMA_STACK",
            "approved": True,
        },
        {
            "signal_id": "sig_20260312_001",
            "ts_utc": "2026-03-12T04:00:00Z",
            "strategy_id": "locked_ms_bos_choch_pullback_atr_adx_ema_entry_v2_pullback_deep",
            "timeframe": "M30",
            "side": "BUY",
            "trend_bucket": "BULL_TREND",
            "volatility_bucket": "HIGH_VOL",
            "price_location_bucket": "ABOVE_EMA_STACK",
            "approved": True,
        },
        {
            "signal_id": "sig_20260317_001",
            "ts_utc": "2026-03-17T03:30:00Z",
            "strategy_id": "locked_ms_bos_choch_pullback_atr_adx_ema_entry_v2_pullback_deep",
            "timeframe": "M30",
            "side": "BUY",
            "trend_bucket": "BULL_TREND",
            "volatility_bucket": "HIGH_VOL",
            "price_location_bucket": "ABOVE_EMA_STACK",
            "approved": True,
        },
        {
            "signal_id": "sig_20260319_001",
            "ts_utc": "2026-03-19T05:00:00Z",
            "strategy_id": "locked_ms_bos_choch_pullback_atr_adx_ema_entry_v2_pullback_deep",
            "timeframe": "M30",
            "side": "BUY",
            "trend_bucket": "BULL_TREND",
            "volatility_bucket": "HIGH_VOL",
            "price_location_bucket": "ABOVE_EMA_STACK",
            "approved": True,
        },
    ]


def build_sample_trades() -> list[dict]:
    return [
        {
            "trade_id": "trade_20260310_001",
            "signal_id": "sig_20260310_001",
            "entry_time_utc": "2026-03-10T02:35:00Z",
            "exit_time_utc": "2026-03-10T08:10:00Z",
            "entry_price": 2895.10,
            "exit_price": 2904.20,
            "pnl": 9.10,
            "exit_reason": "take_profit",
            "trade_date": "2026-03-10",
            "hour_bucket": "08",
            "trend_bucket": "BULL_TREND",
            "volatility_bucket": "HIGH_VOL",
            "price_location_bucket": "ABOVE_EMA_STACK",
        },
        {
            "trade_id": "trade_20260312_001",
            "signal_id": "sig_20260312_001",
            "entry_time_utc": "2026-03-12T04:05:00Z",
            "exit_time_utc": "2026-03-12T09:00:00Z",
            "entry_price": 2901.50,
            "exit_price": 2898.80,
            "pnl": -2.70,
            "exit_reason": "stop_loss",
            "trade_date": "2026-03-12",
            "hour_bucket": "09",
            "trend_bucket": "BULL_TREND",
            "volatility_bucket": "HIGH_VOL",
            "price_location_bucket": "ABOVE_EMA_STACK",
        },
        {
            "trade_id": "trade_20260317_001",
            "signal_id": "sig_20260317_001",
            "entry_time_utc": "2026-03-17T03:35:00Z",
            "exit_time_utc": "2026-03-17T07:40:00Z",
            "entry_price": 2910.00,
            "exit_price": 2918.60,
            "pnl": 8.60,
            "exit_reason": "take_profit",
            "trade_date": "2026-03-17",
            "hour_bucket": "07",
            "trend_bucket": "BULL_TREND",
            "volatility_bucket": "HIGH_VOL",
            "price_location_bucket": "ABOVE_EMA_STACK",
        },
        {
            "trade_id": "trade_20260319_001",
            "signal_id": "sig_20260319_001",
            "entry_time_utc": "2026-03-19T05:10:00Z",
            "exit_time_utc": "2026-03-19T11:30:00Z",
            "entry_price": 2922.20,
            "exit_price": 2931.40,
            "pnl": 9.20,
            "exit_reason": "take_profit",
            "trade_date": "2026-03-19",
            "hour_bucket": "11",
            "trend_bucket": "BULL_TREND",
            "volatility_bucket": "HIGH_VOL",
            "price_location_bucket": "ABOVE_EMA_STACK",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed sample inputs for paper trade monitor.")
    parser.add_argument("--outdir", required=True, help="Output directory for signals.jsonl and trades.jsonl")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    signals_path = outdir / "signals.jsonl"
    trades_path = outdir / "trades.jsonl"

    signals = build_sample_signals()
    trades = build_sample_trades()

    write_jsonl(signals_path, signals)
    write_jsonl(trades_path, trades)

    print("=" * 120)
    print(f"[DONE] seed_paper_trade_monitor_inputs.py version={VERSION}")
    print(f"[DONE] outdir={outdir}")
    print(f"[DONE] signals={signals_path} rows={len(signals)}")
    print(f"[DONE] trades={trades_path} rows={len(trades)}")
    print("=" * 120)


if __name__ == "__main__":
    main()
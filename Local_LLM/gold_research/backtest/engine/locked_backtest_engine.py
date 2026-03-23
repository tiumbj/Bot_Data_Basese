# ============================================================
# ชื่อโค้ด: Locked Backtest Engine
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\backtest\engine\locked_backtest_engine.py
# คำสั่งรัน:
# python C:\Data\Bot\Local_LLM\gold_research\backtest\engine\locked_backtest_engine.py --csv C:\Data\data_base\canonical_dataset\dataset\tf\XAUUSD_M15.csv --strategy ema_crossover --fast-ema 20 --slow-ema 50 --output C:\Data\data_base\backtest_results\gold_research_results\in_sample\ema20_50_M15
# เวอร์ชัน: v1.0.0
# ============================================================

"""
locked_backtest_engine.py
Version: v1.0.0

Purpose:
- เป็น backtest engine กลางของ project locked model
- อ่าน canonical dataset จาก dataset/tf/
- รองรับ strategy มาตรฐานเริ่มต้น:
  1) ema_crossover
  2) buy_and_hold
- คิดผลแบบ position reversal
- รวม transaction cost:
  - spread
  - slippage
  - commission
- สรุป metric กลาง:
  - net profit
  - profit factor
  - expectancy
  - max drawdown
  - total trades
  - win rate
  - avg win
  - avg loss

Canonical CSV format:
- datetime,open,high,low,close,volume

Notes:
- Engine นี้เป็นแกนกลางก่อน
- ขั้นถัดไปค่อยเสียบ strategy family:
  - indicator
  - price action
  - hybrid
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


VERSION = "v1.0.0"


@dataclass
class TradeRecord:
    entry_time: str
    exit_time: str
    side: str
    entry_price: float
    exit_price: float
    pnl_points: float
    bars_held: int


@dataclass
class BacktestSummary:
    version: str
    strategy: str
    input_csv: str
    total_bars: int
    total_trades: int
    wins: int
    losses: int
    win_rate_pct: float
    net_profit_points: float
    gross_profit_points: float
    gross_loss_points: float
    avg_profit_points: float
    avg_win_points: float
    avg_loss_points: float
    profit_factor: float
    max_drawdown_points: float
    expectancy_points: float
    first_datetime: str
    last_datetime: str
    spread_points: float
    slippage_points: float
    commission_points: float
    parameters: Dict[str, float | int | str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked Backtest Engine")
    parser.add_argument("--csv", required=True, help="Canonical TF csv path")
    parser.add_argument("--strategy", required=True, choices=["ema_crossover", "buy_and_hold"], help="Strategy name")
    parser.add_argument("--output", required=True, help="Output folder for reports")
    parser.add_argument("--spread-points", type=float, default=0.0, help="Round-trip spread cost in price points")
    parser.add_argument("--slippage-points", type=float, default=0.0, help="Round-trip slippage cost in price points")
    parser.add_argument("--commission-points", type=float, default=0.0, help="Round-trip commission cost in price points")

    # EMA strategy params
    parser.add_argument("--fast-ema", type=int, default=20, help="Fast EMA period")
    parser.add_argument("--slow-ema", type=int, default=50, help="Slow EMA period")

    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_canonical_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_cols = ["datetime", "open", "high", "low", "close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing canonical columns: {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" not in df.columns:
        df["volume"] = 0.0
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).copy()
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="first").reset_index(drop=True)

    if len(df) < 50:
        raise ValueError("Not enough rows for backtest. Need at least 50 bars.")

    return df


def build_buy_and_hold_signal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["signal"] = 1
    out["cross"] = 0
    out.loc[out.index[0], "cross"] = 1
    return out


def build_ema_crossover_signal(df: pd.DataFrame, fast_ema: int, slow_ema: int) -> pd.DataFrame:
    if fast_ema < 1 or slow_ema < 2:
        raise ValueError("EMA periods must be positive.")
    if fast_ema >= slow_ema:
        raise ValueError("Locked rule violation: fast_ema must be < slow_ema.")

    out = df.copy()
    out["ema_fast"] = out["close"].ewm(span=fast_ema, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=slow_ema, adjust=False).mean()

    out["signal"] = 0
    out.loc[out["ema_fast"] > out["ema_slow"], "signal"] = 1
    out.loc[out["ema_fast"] < out["ema_slow"], "signal"] = -1

    out["signal_prev"] = out["signal"].shift(1).fillna(0)
    out["cross"] = 0
    out.loc[(out["signal_prev"] <= 0) & (out["signal"] == 1), "cross"] = 1
    out.loc[(out["signal_prev"] >= 0) & (out["signal"] == -1), "cross"] = -1

    return out


def build_strategy_frame(df: pd.DataFrame, strategy: str, args: argparse.Namespace) -> pd.DataFrame:
    if strategy == "ema_crossover":
        return build_ema_crossover_signal(df, args.fast_ema, args.slow_ema)
    if strategy == "buy_and_hold":
        return build_buy_and_hold_signal(df)
    raise ValueError(f"Unsupported strategy: {strategy}")


def run_reversal_backtest(
    df: pd.DataFrame,
    spread_points: float,
    slippage_points: float,
    commission_points: float,
) -> List[TradeRecord]:
    round_trip_cost = spread_points + slippage_points + commission_points

    trades: List[TradeRecord] = []
    position = 0
    entry_price: Optional[float] = None
    entry_time: Optional[pd.Timestamp] = None
    entry_index: Optional[int] = None

    closes = df["close"].to_numpy(dtype=float)
    crosses = df["cross"].to_numpy(dtype=int)
    dts = df["datetime"].tolist()

    for i in range(1, len(df)):
        cross = crosses[i]
        price = float(closes[i])
        now = dts[i]

        if cross == 1:
            if position == -1 and entry_price is not None and entry_time is not None and entry_index is not None:
                pnl = entry_price - price - round_trip_cost
                trades.append(
                    TradeRecord(
                        entry_time=str(entry_time),
                        exit_time=str(now),
                        side="SELL",
                        entry_price=float(entry_price),
                        exit_price=price,
                        pnl_points=float(pnl),
                        bars_held=int(i - entry_index),
                    )
                )
                position = 0
                entry_price = None
                entry_time = None
                entry_index = None

            if position == 0:
                position = 1
                entry_price = price
                entry_time = now
                entry_index = i

        elif cross == -1:
            if position == 1 and entry_price is not None and entry_time is not None and entry_index is not None:
                pnl = price - entry_price - round_trip_cost
                trades.append(
                    TradeRecord(
                        entry_time=str(entry_time),
                        exit_time=str(now),
                        side="BUY",
                        entry_price=float(entry_price),
                        exit_price=price,
                        pnl_points=float(pnl),
                        bars_held=int(i - entry_index),
                    )
                )
                position = 0
                entry_price = None
                entry_time = None
                entry_index = None

            if position == 0:
                position = -1
                entry_price = price
                entry_time = now
                entry_index = i

    if position != 0 and entry_price is not None and entry_time is not None and entry_index is not None:
        final_price = float(closes[-1])
        final_time = dts[-1]
        if position == 1:
            pnl = final_price - entry_price - round_trip_cost
            side = "BUY"
        else:
            pnl = entry_price - final_price - round_trip_cost
            side = "SELL"

        trades.append(
            TradeRecord(
                entry_time=str(entry_time),
                exit_time=str(final_time),
                side=side,
                entry_price=float(entry_price),
                exit_price=float(final_price),
                pnl_points=float(pnl),
                bars_held=int(len(df) - 1 - entry_index),
            )
        )

    return trades


def build_equity_curve(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=["trade_no", "equity", "rolling_peak", "drawdown"])

    out = trades_df.copy().reset_index(drop=True)
    out["trade_no"] = np.arange(1, len(out) + 1)
    out["equity"] = out["pnl_points"].cumsum()
    out["rolling_peak"] = out["equity"].cummax()
    out["drawdown"] = out["rolling_peak"] - out["equity"]
    return out[["trade_no", "equity", "rolling_peak", "drawdown"]]


def summarize_backtest(
    trades: List[TradeRecord],
    input_df: pd.DataFrame,
    args: argparse.Namespace,
) -> BacktestSummary:
    trade_df = pd.DataFrame([asdict(t) for t in trades])

    total_trades = int(len(trade_df))
    wins = int((trade_df["pnl_points"] > 0).sum()) if total_trades > 0 else 0
    losses = int((trade_df["pnl_points"] <= 0).sum()) if total_trades > 0 else 0
    win_rate = (wins / total_trades) * 100.0 if total_trades > 0 else 0.0

    net_profit = float(trade_df["pnl_points"].sum()) if total_trades > 0 else 0.0
    gross_profit = float(trade_df.loc[trade_df["pnl_points"] > 0, "pnl_points"].sum()) if total_trades > 0 else 0.0
    gross_loss_signed = float(trade_df.loc[trade_df["pnl_points"] <= 0, "pnl_points"].sum()) if total_trades > 0 else 0.0
    gross_loss_abs = abs(gross_loss_signed)

    avg_profit = float(trade_df["pnl_points"].mean()) if total_trades > 0 else 0.0
    avg_win = float(trade_df.loc[trade_df["pnl_points"] > 0, "pnl_points"].mean()) if wins > 0 else 0.0
    avg_loss = float(trade_df.loc[trade_df["pnl_points"] <= 0, "pnl_points"].mean()) if losses > 0 else 0.0

    profit_factor = (gross_profit / gross_loss_abs) if gross_loss_abs > 0 else 999.0

    equity_df = build_equity_curve(trade_df.rename(columns={"pnl_points": "pnl_points"}))
    max_dd = float(equity_df["drawdown"].max()) if not equity_df.empty else 0.0
    expectancy = avg_profit

    parameters: Dict[str, float | int | str] = {}
    if args.strategy == "ema_crossover":
        parameters = {
            "fast_ema": int(args.fast_ema),
            "slow_ema": int(args.slow_ema),
        }

    return BacktestSummary(
        version=VERSION,
        strategy=args.strategy,
        input_csv=str(Path(args.csv).resolve()),
        total_bars=int(len(input_df)),
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        win_rate_pct=float(win_rate),
        net_profit_points=float(net_profit),
        gross_profit_points=float(gross_profit),
        gross_loss_points=float(gross_loss_signed),
        avg_profit_points=float(avg_profit),
        avg_win_points=float(avg_win),
        avg_loss_points=float(avg_loss),
        profit_factor=float(profit_factor),
        max_drawdown_points=float(max_dd),
        expectancy_points=float(expectancy),
        first_datetime=str(input_df["datetime"].iloc[0]),
        last_datetime=str(input_df["datetime"].iloc[-1]),
        spread_points=float(args.spread_points),
        slippage_points=float(args.slippage_points),
        commission_points=float(args.commission_points),
        parameters=parameters,
    )


def save_outputs(
    output_dir: Path,
    strategy_df: pd.DataFrame,
    trades: List[TradeRecord],
    summary: BacktestSummary,
) -> None:
    ensure_output_dir(output_dir)

    strategy_csv = output_dir / "strategy_frame.csv"
    trades_csv = output_dir / "trades.csv"
    equity_csv = output_dir / "equity_curve.csv"
    summary_json = output_dir / "summary.json"
    summary_txt = output_dir / "summary.txt"

    strategy_df.to_csv(strategy_csv, index=False, encoding="utf-8-sig")

    trades_df = pd.DataFrame([asdict(t) for t in trades])
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=["entry_time", "exit_time", "side", "entry_price", "exit_price", "pnl_points", "bars_held"])
    trades_df.to_csv(trades_csv, index=False, encoding="utf-8-sig")

    equity_df = build_equity_curve(trades_df.rename(columns={"pnl_points": "pnl_points"}))
    equity_df.to_csv(equity_csv, index=False, encoding="utf-8-sig")

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, ensure_ascii=False, indent=2)

    text_lines = [
        f"Locked Backtest Engine | version={summary.version}",
        f"strategy={summary.strategy}",
        f"input_csv={summary.input_csv}",
        f"total_bars={summary.total_bars}",
        f"total_trades={summary.total_trades}",
        f"wins={summary.wins}",
        f"losses={summary.losses}",
        f"win_rate_pct={summary.win_rate_pct:.4f}",
        f"net_profit_points={summary.net_profit_points:.6f}",
        f"gross_profit_points={summary.gross_profit_points:.6f}",
        f"gross_loss_points={summary.gross_loss_points:.6f}",
        f"avg_profit_points={summary.avg_profit_points:.6f}",
        f"avg_win_points={summary.avg_win_points:.6f}",
        f"avg_loss_points={summary.avg_loss_points:.6f}",
        f"profit_factor={summary.profit_factor:.6f}",
        f"max_drawdown_points={summary.max_drawdown_points:.6f}",
        f"expectancy_points={summary.expectancy_points:.6f}",
        f"first_datetime={summary.first_datetime}",
        f"last_datetime={summary.last_datetime}",
        f"spread_points={summary.spread_points:.6f}",
        f"slippage_points={summary.slippage_points:.6f}",
        f"commission_points={summary.commission_points:.6f}",
        f"parameters={summary.parameters}",
    ]
    summary_txt.write_text("\n".join(text_lines), encoding="utf-8")


def print_summary(summary: BacktestSummary, output_dir: Path) -> None:
    print("=" * 90)
    print(f"Locked Backtest Engine | version={summary.version}")
    print("=" * 90)
    print(f"strategy             : {summary.strategy}")
    print(f"input_csv            : {summary.input_csv}")
    print(f"total_bars           : {summary.total_bars}")
    print(f"total_trades         : {summary.total_trades}")
    print(f"wins                 : {summary.wins}")
    print(f"losses               : {summary.losses}")
    print(f"win_rate_pct         : {summary.win_rate_pct:.4f}")
    print(f"net_profit_points    : {summary.net_profit_points:.6f}")
    print(f"gross_profit_points  : {summary.gross_profit_points:.6f}")
    print(f"gross_loss_points    : {summary.gross_loss_points:.6f}")
    print(f"avg_profit_points    : {summary.avg_profit_points:.6f}")
    print(f"avg_win_points       : {summary.avg_win_points:.6f}")
    print(f"avg_loss_points      : {summary.avg_loss_points:.6f}")
    print(f"profit_factor        : {summary.profit_factor:.6f}")
    print(f"max_drawdown_points  : {summary.max_drawdown_points:.6f}")
    print(f"expectancy_points    : {summary.expectancy_points:.6f}")
    print(f"parameters           : {summary.parameters}")
    print("=" * 90)
    print(f"output_dir           : {output_dir}")
    print("=" * 90)


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output)

    df = load_canonical_csv(csv_path)
    strategy_df = build_strategy_frame(df, args.strategy, args)
    trades = run_reversal_backtest(
        strategy_df,
        spread_points=args.spread_points,
        slippage_points=args.slippage_points,
        commission_points=args.commission_points,
    )
    summary = summarize_backtest(trades, df, args)

    save_outputs(
        output_dir=output_dir,
        strategy_df=strategy_df,
        trades=trades,
        summary=summary,
    )
    print_summary(summary, output_dir)


if __name__ == "__main__":
    main()
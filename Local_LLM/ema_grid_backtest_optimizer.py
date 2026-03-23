# ============================================================
# ชื่อโค้ด: EMA Grid Backtest Optimizer
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\ema_grid_backtest_optimizer.py
# คำสั่งรัน: python C:\Data\Bot\Local_LLM\ema_grid_backtest_optimizer.py --csv C:\Data\Bot\Local_LLM\M1_XAUUSD\XAUUSD_M1_ALL.csv
# เวอร์ชัน: v1.1.0
# ============================================================

"""
EMA Grid Backtest Optimizer
Version: v1.1.0

Purpose:
- อ่านไฟล์ HistData XAUUSD M1 แบบไม่มี header ได้โดยตรง
- วนลูป EMA Fast 1-50 และ EMA Slow 20-100
- ทดสอบเฉพาะคู่ที่ fast < slow
- จัดอันดับผลด้วย score ที่คุมทั้ง expectancy, profit factor, drawdown
- บันทึกผลทั้งหมดและผลจัดอันดับเป็น CSV

รองรับไฟล์ 2 แบบ:
1) HistData format:
   date,time,open,high,low,close,volume
   ตัวอย่าง:
   2009.03.15,17:00,929.600000,929.600000,929.600000,929.600000,0

2) Standard OHLC CSV:
   time,open,high,low,close,(optional volume)

หมายเหตุ:
- โค้ดนี้ใช้ EMA crossover เป็น research/filter optimization
- ไม่ใช่ production final strategy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


VERSION = "v1.1.0"


@dataclass
class BacktestResult:
    fast_ema: int
    slow_ema: int
    total_trades: int
    wins: int
    losses: int
    win_rate_pct: float
    net_profit_points: float
    avg_profit_points: float
    avg_win_points: float
    avg_loss_points: float
    profit_factor: float
    max_drawdown_points: float
    expectancy_points: float
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EMA Grid Backtest Optimizer")
    parser.add_argument("--csv", required=True, help="Path to source CSV")
    parser.add_argument("--fast-min", type=int, default=1, help="Minimum fast EMA")
    parser.add_argument("--fast-max", type=int, default=50, help="Maximum fast EMA")
    parser.add_argument("--slow-min", type=int, default=20, help="Minimum slow EMA")
    parser.add_argument("--slow-max", type=int, default=100, help="Maximum slow EMA")
    parser.add_argument("--spread-points", type=float, default=0.0, help="Spread cost per round-trip trade in price points")
    parser.add_argument("--commission-points", type=float, default=0.0, help="Commission cost per round-trip trade in price points")
    parser.add_argument("--slippage-points", type=float, default=0.0, help="Slippage cost per round-trip trade in price points")
    parser.add_argument("--min-trades", type=int, default=30, help="Minimum trades required for ranking")
    parser.add_argument("--top", type=int, default=20, help="Top results to print")
    parser.add_argument("--output-dir", default="ema_grid_output", help="Directory to save result CSV files")
    return parser.parse_args()


def load_price_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # อ่านบรรทัดแรกเพื่อตรวจ format
    with csv_path.open("r", encoding="utf-8-sig") as f:
        first_line = f.readline().strip()

    if not first_line:
        raise ValueError("CSV file is empty.")

    # ตรวจว่าเป็น HistData no-header format หรือ standard CSV
    first_lower = first_line.lower()

    is_histdata = False
    if "," in first_line:
        parts = first_line.split(",")
        if len(parts) >= 7:
            if "." in parts[0] and ":" in parts[1]:
                is_histdata = True

    if is_histdata:
        df = pd.read_csv(
            csv_path,
            header=None,
            names=["date", "time", "open", "high", "low", "close", "volume"],
            dtype={
                "date": str,
                "time": str,
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float,
            },
        )
        df["datetime"] = pd.to_datetime(
            df["date"] + " " + df["time"],
            format="%Y.%m.%d %H:%M",
            errors="coerce",
        )
    else:
        df = pd.read_csv(csv_path)

        expected_cols = {"open", "high", "low", "close"}
        missing = expected_cols - set(df.columns.str.lower() if hasattr(df.columns, "str") else df.columns)
        if missing:
            # พยายาม normalize ชื่อคอลัมน์
            rename_map = {}
            for col in df.columns:
                col_l = str(col).strip().lower()
                rename_map[col] = col_l
            df = df.rename(columns=rename_map)
            missing = expected_cols - set(df.columns)

        if missing:
            raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

        if "time" in df.columns:
            df["datetime"] = pd.to_datetime(df["time"], errors="coerce")
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        elif "date" in df.columns and "time" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
        else:
            raise ValueError("Standard CSV must contain time or datetime column.")

    required_numeric = ["open", "high", "low", "close"]
    for col in required_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    if len(df) < 300:
        raise ValueError("Not enough rows for optimization. Need at least 300 bars.")

    # ลบ duplicate datetime เผื่อไฟล์บางช่วงซ้ำ
    df = df.drop_duplicates(subset=["datetime"], keep="first").reset_index(drop=True)

    return df


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def build_signal(df: pd.DataFrame, fast_period: int, slow_period: int) -> pd.DataFrame:
    local_df = df[["datetime", "open", "high", "low", "close"]].copy()
    local_df["ema_fast"] = compute_ema(local_df["close"], fast_period)
    local_df["ema_slow"] = compute_ema(local_df["close"], slow_period)

    local_df["signal"] = 0
    local_df.loc[local_df["ema_fast"] > local_df["ema_slow"], "signal"] = 1
    local_df.loc[local_df["ema_fast"] < local_df["ema_slow"], "signal"] = -1

    local_df["signal_prev"] = local_df["signal"].shift(1).fillna(0)
    local_df["cross"] = 0
    local_df.loc[(local_df["signal_prev"] <= 0) & (local_df["signal"] == 1), "cross"] = 1
    local_df.loc[(local_df["signal_prev"] >= 0) & (local_df["signal"] == -1), "cross"] = -1

    return local_df


def run_crossover_backtest(
    df: pd.DataFrame,
    fast_period: int,
    slow_period: int,
    spread_points: float,
    commission_points: float,
    slippage_points: float,
) -> BacktestResult:
    bt = build_signal(df, fast_period, slow_period)

    trades: List[float] = []
    position = 0
    entry_price = None

    round_trip_cost = spread_points + commission_points + slippage_points

    closes = bt["close"].to_numpy(dtype=float)
    crosses = bt["cross"].to_numpy(dtype=int)

    for i in range(1, len(bt)):
        cross = crosses[i]
        price = closes[i]

        if cross == 1:
            if position == -1 and entry_price is not None:
                pnl = entry_price - price - round_trip_cost
                trades.append(float(pnl))
                position = 0
                entry_price = None

            if position == 0:
                position = 1
                entry_price = price

        elif cross == -1:
            if position == 1 and entry_price is not None:
                pnl = price - entry_price - round_trip_cost
                trades.append(float(pnl))
                position = 0
                entry_price = None

            if position == 0:
                position = -1
                entry_price = price

    if position != 0 and entry_price is not None:
        final_price = closes[-1]
        if position == 1:
            pnl = final_price - entry_price - round_trip_cost
        else:
            pnl = entry_price - final_price - round_trip_cost
        trades.append(float(pnl))

    if not trades:
        return BacktestResult(
            fast_ema=fast_period,
            slow_ema=slow_period,
            total_trades=0,
            wins=0,
            losses=0,
            win_rate_pct=0.0,
            net_profit_points=0.0,
            avg_profit_points=0.0,
            avg_win_points=0.0,
            avg_loss_points=0.0,
            profit_factor=0.0,
            max_drawdown_points=0.0,
            expectancy_points=0.0,
            score=-999999.0,
        )

    trade_array = np.array(trades, dtype=float)
    wins = trade_array[trade_array > 0]
    losses = trade_array[trade_array <= 0]

    total_trades = int(len(trade_array))
    win_count = int(len(wins))
    loss_count = int(len(losses))
    win_rate_pct = (win_count / total_trades) * 100.0 if total_trades > 0 else 0.0

    net_profit = float(trade_array.sum())
    avg_profit = float(trade_array.mean())
    avg_win = float(wins.mean()) if win_count > 0 else 0.0
    avg_loss = float(losses.mean()) if loss_count > 0 else 0.0

    gross_profit = float(wins.sum()) if win_count > 0 else 0.0
    gross_loss_abs = abs(float(losses.sum())) if loss_count > 0 else 0.0
    profit_factor = gross_profit / gross_loss_abs if gross_loss_abs > 0 else 999.0

    equity_curve = np.cumsum(trade_array)
    rolling_peak = np.maximum.accumulate(equity_curve)
    drawdowns = rolling_peak - equity_curve
    max_drawdown = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

    expectancy = avg_profit

    # score คุมทั้งผลตอบแทนและความเสี่ยง
    score = (expectancy * min(profit_factor, 10.0)) - (max_drawdown * 0.10)

    return BacktestResult(
        fast_ema=fast_period,
        slow_ema=slow_period,
        total_trades=total_trades,
        wins=win_count,
        losses=loss_count,
        win_rate_pct=win_rate_pct,
        net_profit_points=net_profit,
        avg_profit_points=avg_profit,
        avg_win_points=avg_win,
        avg_loss_points=avg_loss,
        profit_factor=profit_factor,
        max_drawdown_points=max_drawdown,
        expectancy_points=expectancy,
        score=score,
    )


def optimize_ema_grid(
    df: pd.DataFrame,
    fast_min: int,
    fast_max: int,
    slow_min: int,
    slow_max: int,
    spread_points: float,
    commission_points: float,
    slippage_points: float,
) -> pd.DataFrame:
    results: List[BacktestResult] = []

    for fast in range(fast_min, fast_max + 1):
        for slow in range(slow_min, slow_max + 1):
            if fast >= slow:
                continue

            result = run_crossover_backtest(
                df=df,
                fast_period=fast,
                slow_period=slow,
                spread_points=spread_points,
                commission_points=commission_points,
                slippage_points=slippage_points,
            )
            results.append(result)

    if not results:
        raise RuntimeError("No valid EMA combinations were generated.")

    return pd.DataFrame([asdict(r) for r in results])


def rank_results(results_df: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    ranked = results_df.copy()
    ranked = ranked[ranked["total_trades"] >= min_trades].copy()

    if ranked.empty:
        raise RuntimeError(
            f"No strategies passed min_trades={min_trades}. "
            "Reduce min_trades or use more data."
        )

    ranked = ranked.sort_values(
        by=["score", "profit_factor", "net_profit_points", "win_rate_pct"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    return ranked


def save_outputs(results_df: pd.DataFrame, ranked_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results_path = output_dir / "ema_grid_all_results.csv"
    ranked_results_path = output_dir / "ema_grid_ranked_results.csv"

    results_df.to_csv(all_results_path, index=False, encoding="utf-8-sig")
    ranked_df.to_csv(ranked_results_path, index=False, encoding="utf-8-sig")

    return all_results_path, ranked_results_path


def print_summary(df: pd.DataFrame, top_n: int) -> None:
    best = df.iloc[0]

    print("=" * 90)
    print(f"EMA Grid Backtest Optimizer | version={VERSION}")
    print("=" * 90)
    print("BEST RESULT")
    print(f"fast_ema            : {int(best['fast_ema'])}")
    print(f"slow_ema            : {int(best['slow_ema'])}")
    print(f"total_trades        : {int(best['total_trades'])}")
    print(f"win_rate_pct        : {best['win_rate_pct']:.2f}")
    print(f"net_profit_points   : {best['net_profit_points']:.6f}")
    print(f"profit_factor       : {best['profit_factor']:.6f}")
    print(f"max_drawdown_points : {best['max_drawdown_points']:.6f}")
    print(f"expectancy_points   : {best['expectancy_points']:.6f}")
    print(f"score               : {best['score']:.6f}")
    print("=" * 90)
    print(f"TOP {top_n} RESULTS")
    print("=" * 90)

    display_cols = [
        "fast_ema",
        "slow_ema",
        "total_trades",
        "win_rate_pct",
        "net_profit_points",
        "profit_factor",
        "max_drawdown_points",
        "expectancy_points",
        "score",
    ]
    print(df[display_cols].head(top_n).to_string(index=False))


def main() -> None:
    args = parse_args()

    if args.fast_min < 1 or args.slow_min < 2:
        raise ValueError("EMA periods must be positive integers.")
    if args.fast_min > args.fast_max:
        raise ValueError("fast-min must be <= fast-max.")
    if args.slow_min > args.slow_max:
        raise ValueError("slow-min must be <= slow-max.")

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)

    df = load_price_data(csv_path)

    results_df = optimize_ema_grid(
        df=df,
        fast_min=args.fast_min,
        fast_max=args.fast_max,
        slow_min=args.slow_min,
        slow_max=args.slow_max,
        spread_points=args.spread_points,
        commission_points=args.commission_points,
        slippage_points=args.slippage_points,
    )

    ranked_df = rank_results(results_df, args.min_trades)
    all_results_path, ranked_results_path = save_outputs(results_df, ranked_df, output_dir)

    print_summary(ranked_df, args.top)
    print("=" * 90)
    print(f"Saved all results   : {all_results_path.resolve()}")
    print(f"Saved ranked result : {ranked_results_path.resolve()}")
    print("=" * 90)


if __name__ == "__main__":
    main()
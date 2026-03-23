# ============================================================
# ชื่อโค้ด: analyze_m30_signal_exit_damage.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\analyze_m30_signal_exit_damage.py
# คำสั่งรัน:
# python C:\Data\Bot\Local_LLM\gold_research\analyze_m30_signal_exit_damage.py --input "C:\Data\data_base\backtest_results\gold_research_outputs\m30_regime_pullback_deep_window01\trades_enriched.csv" --outdir "C:\Data\data_base\backtest_results\gold_research_outputs\m30_signal_exit_pullback_deep_window01"
# เวอร์ชัน: v1.0.0
# ============================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


VERSION = "v1.0.0"


def log(message: str) -> None:
    print(f"[analyze_m30_signal_exit_damage {VERSION}] {message}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def prepare_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df_raw)

    required = [
        "exit_reason",
        "pnl",
        "side",
        "trade_result",
        "hold_minutes",
        "pnl_r",
        "trend_bucket",
        "volatility_bucket",
        "session_bucket",
        "price_location_bucket",
        "hold_bucket",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["exit_reason"] = out["exit_reason"].astype(str).str.strip()
    out["side"] = out["side"].astype(str).str.strip().str.upper()
    out["trade_result"] = out["trade_result"].astype(str).str.strip().str.upper()

    numeric_cols = ["pnl", "hold_minutes", "pnl_r", "adx14", "atr14", "rr_planned"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = to_num(out[col])

    return out


def aggregate_dimension(signal_df: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = (
        signal_df.groupby(column, dropna=False)
        .agg(
            trades=("pnl", "count"),
            wins=("trade_result", lambda s: int((s == "WIN").sum())),
            losses=("trade_result", lambda s: int((s == "LOSS").sum())),
            pnl_sum=("pnl", "sum"),
            pnl_avg=("pnl", "mean"),
            pnl_median=("pnl", "median"),
            pnl_r_avg=("pnl_r", "mean"),
            hold_min_avg=("hold_minutes", "mean"),
        )
        .reset_index()
    )

    grouped["win_rate"] = np.where(grouped["trades"] > 0, grouped["wins"] / grouped["trades"], np.nan)
    grouped["damage_score"] = grouped["pnl_sum"] * -1.0
    grouped["bad_bucket_flag"] = np.where(
        (grouped["pnl_sum"] < 0) & (grouped["win_rate"] < 0.35),
        "YES",
        "NO",
    )

    grouped.insert(0, "dimension", column)
    grouped = grouped.rename(columns={column: "bucket"})
    grouped = grouped.sort_values(
        by=["damage_score", "trades", "pnl_avg"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return grouped


def build_bucket_summary(signal_df: pd.DataFrame) -> pd.DataFrame:
    dimensions = [
        "trend_bucket",
        "volatility_bucket",
        "session_bucket",
        "price_location_bucket",
        "hold_bucket",
        "side",
    ]

    parts: List[pd.DataFrame] = []
    for dim in dimensions:
        parts.append(aggregate_dimension(signal_df, dim))

    return pd.concat(parts, ignore_index=True)


def build_priority_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    bad = summary_df.loc[summary_df["pnl_sum"] < 0].copy()

    bad["priority_rank_score"] = (
        bad["damage_score"].fillna(0.0) * 1.0
        + bad["trades"].fillna(0.0) * 0.25
        + (1.0 - bad["win_rate"].fillna(0.0)) * 100.0
    )

    bad["suggested_action"] = np.where(
        (bad["win_rate"] < 0.30) & (bad["pnl_sum"] < -50),
        "BLOCK_FIRST",
        np.where(
            (bad["win_rate"] < 0.35) & (bad["pnl_sum"] < -20),
            "FILTER_FIRST",
            "REVIEW",
        ),
    )

    bad = bad.sort_values(
        by=["priority_rank_score", "damage_score", "trades"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return bad[
        [
            "dimension",
            "bucket",
            "trades",
            "wins",
            "losses",
            "win_rate",
            "pnl_sum",
            "pnl_avg",
            "pnl_r_avg",
            "hold_min_avg",
            "damage_score",
            "priority_rank_score",
            "suggested_action",
        ]
    ]


def build_signal_exit_trades(signal_df: pd.DataFrame) -> pd.DataFrame:
    out = signal_df.copy()

    display_cols = [
        "entry_time",
        "exit_time",
        "side",
        "entry_price",
        "exit_price",
        "pnl",
        "pnl_r",
        "hold_minutes",
        "trend_bucket",
        "volatility_bucket",
        "session_bucket",
        "price_location_bucket",
        "hold_bucket",
        "adx14",
        "atr14",
        "rr_planned",
        "trade_result",
        "exit_reason",
    ]

    available = [c for c in display_cols if c in out.columns]
    out = out[available].copy()
    out = out.sort_values(by=["pnl", "hold_minutes"], ascending=[True, True]).reset_index(drop=True)
    return out


def build_report(all_df: pd.DataFrame, signal_df: pd.DataFrame, priority_df: pd.DataFrame) -> Dict:
    signal_share = float(len(signal_df) / len(all_df)) if len(all_df) else 0.0
    total_signal_pnl = float(signal_df["pnl"].sum()) if len(signal_df) else 0.0

    top_priority = []
    for _, row in priority_df.head(10).iterrows():
        top_priority.append(
            {
                "dimension": str(row["dimension"]),
                "bucket": str(row["bucket"]),
                "trades": int(row["trades"]),
                "win_rate": float(row["win_rate"]) if pd.notna(row["win_rate"]) else None,
                "pnl_sum": float(row["pnl_sum"]) if pd.notna(row["pnl_sum"]) else None,
                "suggested_action": str(row["suggested_action"]),
            }
        )

    report = {
        "version": VERSION,
        "rows_total": int(len(all_df)),
        "rows_signal_exit": int(len(signal_df)),
        "signal_exit_share": signal_share,
        "signal_exit_net_pnl": total_signal_pnl,
        "signal_exit_win_rate": float((signal_df["trade_result"] == "WIN").mean()) if len(signal_df) else 0.0,
        "top_priority_buckets": top_priority,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze M30 signal-exit damage buckets from trades_enriched.csv")
    parser.add_argument("--input", required=True, help="Path to trades_enriched.csv")
    parser.add_argument("--outdir", required=True, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)

    ensure_dir(outdir)

    log(f"version={VERSION}")
    log(f"input={input_path}")
    log(f"outdir={outdir}")

    df_raw = load_csv(input_path)
    df = prepare_dataframe(df_raw)

    signal_df = df.loc[df["exit_reason"].str.lower() == "signal_exit"].copy()
    if signal_df.empty:
        raise ValueError("No signal_exit rows found in input.")

    signal_trades = build_signal_exit_trades(signal_df)
    summary_df = build_bucket_summary(signal_df)
    priority_df = build_priority_table(summary_df)
    report = build_report(df, signal_df, priority_df)

    signal_trades_path = outdir / "signal_exit_trades.csv"
    summary_path = outdir / "signal_exit_bucket_summary.csv"
    priority_path = outdir / "signal_exit_priority_table.csv"
    report_path = outdir / "signal_exit_damage_report.json"

    signal_trades.to_csv(signal_trades_path, index=False, encoding="utf-8")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    priority_df.to_csv(priority_path, index=False, encoding="utf-8")

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    log(f"saved={signal_trades_path}")
    log(f"saved={summary_path}")
    log(f"saved={priority_path}")
    log(f"saved={report_path}")
    log(f"rows_signal_exit={report['rows_signal_exit']}")
    log(f"signal_exit_net_pnl={report['signal_exit_net_pnl']:.6f}")
    log(f"signal_exit_win_rate={report['signal_exit_win_rate']:.4f}")


if __name__ == "__main__":
    main()
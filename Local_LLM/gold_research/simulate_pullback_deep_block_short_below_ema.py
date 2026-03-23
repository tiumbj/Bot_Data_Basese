# ============================================================
# ชื่อโค้ด: simulate_pullback_deep_block_short_below_ema.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\simulate_pullback_deep_block_short_below_ema.py
# คำสั่งรัน:
# python C:\Data\Bot\Local_LLM\gold_research\simulate_pullback_deep_block_short_below_ema.py --input "C:\Data\data_base\backtest_results\gold_research_outputs\m30_regime_pullback_deep_window01\trades_enriched.csv" --outdir "C:\Data\data_base\backtest_results\gold_research_outputs\m30_ruletest_pullback_deep_block_short_below_ema_window01"
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
    print(f"[simulate_pullback_deep_block_short_below_ema {VERSION}] {message}")


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
        "entry_time",
        "exit_time",
        "side",
        "pnl",
        "exit_reason",
        "price_location_bucket",
        "trade_result",
        "hold_minutes",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], errors="coerce")
    out["side"] = out["side"].astype(str).str.strip().str.upper()
    out["exit_reason"] = out["exit_reason"].astype(str).str.strip()
    out["price_location_bucket"] = out["price_location_bucket"].astype(str).str.strip().str.upper()
    out["trade_result"] = out["trade_result"].astype(str).str.strip().str.upper()
    out["pnl"] = to_num(out["pnl"])
    out["hold_minutes"] = to_num(out["hold_minutes"])

    numeric_cols = ["pnl_r", "adx14", "atr14", "rr_planned", "entry_price", "exit_price", "sl_price", "tp_price"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = to_num(out[col])

    out = out.dropna(subset=["entry_time", "exit_time", "side", "pnl"]).copy()
    out = out.sort_values("entry_time").reset_index(drop=True)

    return out


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    trades = int(len(df))
    wins = int((df["pnl"] > 0).sum())
    losses = int((df["pnl"] < 0).sum())
    flats = int((df["pnl"] == 0).sum())

    gross_profit = float(df.loc[df["pnl"] > 0, "pnl"].sum())
    gross_loss_abs = float(abs(df.loc[df["pnl"] < 0, "pnl"].sum()))
    net_pnl = float(df["pnl"].sum())
    avg_pnl = float(df["pnl"].mean()) if trades else 0.0
    median_pnl = float(df["pnl"].median()) if trades else 0.0
    win_rate = float(wins / trades) if trades else 0.0
    avg_hold_minutes = float(df["hold_minutes"].mean()) if "hold_minutes" in df.columns and trades else 0.0

    if gross_loss_abs > 0:
        profit_factor = gross_profit / gross_loss_abs
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    signal_exit_count = int((df["exit_reason"].str.lower() == "signal_exit").sum())
    signal_exit_net_pnl = float(df.loc[df["exit_reason"].str.lower() == "signal_exit", "pnl"].sum())
    signal_exit_win_rate = float(
        (df.loc[df["exit_reason"].str.lower() == "signal_exit", "pnl"] > 0).mean()
    ) if signal_exit_count else 0.0

    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "flats": flats,
        "net_pnl": net_pnl,
        "gross_profit": gross_profit,
        "gross_loss_abs": gross_loss_abs,
        "profit_factor": profit_factor,
        "avg_pnl": avg_pnl,
        "median_pnl": median_pnl,
        "win_rate": win_rate,
        "avg_hold_minutes": avg_hold_minutes,
        "signal_exit_count": signal_exit_count,
        "signal_exit_share": float(signal_exit_count / trades) if trades else 0.0,
        "signal_exit_net_pnl": signal_exit_net_pnl,
        "signal_exit_win_rate": signal_exit_win_rate,
    }


def apply_rule_block_short_below_ema(df: pd.DataFrame) -> pd.DataFrame:
    blocked_mask = (
        (df["side"] == "SHORT") &
        (df["price_location_bucket"] == "BELOW_EMA_STACK")
    )
    filtered = df.loc[~blocked_mask].copy().reset_index(drop=True)
    return filtered


def build_blocked_trades_table(df: pd.DataFrame) -> pd.DataFrame:
    blocked_mask = (
        (df["side"] == "SHORT") &
        (df["price_location_bucket"] == "BELOW_EMA_STACK")
    )
    blocked = df.loc[blocked_mask].copy()

    preferred_cols = [
        "entry_time",
        "exit_time",
        "side",
        "entry_price",
        "exit_price",
        "sl_price",
        "tp_price",
        "pnl",
        "pnl_r",
        "exit_reason",
        "trade_result",
        "trend_bucket",
        "volatility_bucket",
        "session_bucket",
        "price_location_bucket",
        "hold_bucket",
        "hold_minutes",
        "adx14",
        "atr14",
        "rr_planned",
    ]
    available = [c for c in preferred_cols if c in blocked.columns]
    blocked = blocked[available].copy()
    blocked = blocked.sort_values(["pnl", "entry_time"], ascending=[True, True]).reset_index(drop=True)
    return blocked


def aggregate_dimension(df: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = (
        df.groupby(column, dropna=False)
        .agg(
            trades=("pnl", "count"),
            wins=("pnl", lambda s: int((s > 0).sum())),
            losses=("pnl", lambda s: int((s < 0).sum())),
            pnl_sum=("pnl", "sum"),
            pnl_avg=("pnl", "mean"),
            pnl_median=("pnl", "median"),
        )
        .reset_index()
    )
    grouped["win_rate"] = np.where(grouped["trades"] > 0, grouped["wins"] / grouped["trades"], np.nan)
    grouped = grouped.rename(columns={column: "bucket"})
    grouped.insert(0, "dimension", column)
    grouped = grouped.sort_values(["pnl_sum", "win_rate", "trades"], ascending=[False, False, False]).reset_index(drop=True)
    return grouped


def build_comparison_tables(baseline_df: pd.DataFrame, filtered_df: pd.DataFrame) -> pd.DataFrame:
    dimensions = [
        "trend_bucket",
        "volatility_bucket",
        "session_bucket",
        "price_location_bucket",
        "hold_bucket",
        "exit_reason",
        "side",
    ]

    rows: List[Dict[str, object]] = []

    for dim in dimensions:
        base_group = (
            baseline_df.groupby(dim, dropna=False)
            .agg(
                baseline_trades=("pnl", "count"),
                baseline_pnl_sum=("pnl", "sum"),
                baseline_win_rate=("pnl", lambda s: float((s > 0).mean()) if len(s) else 0.0),
            )
            .reset_index()
        )
        filt_group = (
            filtered_df.groupby(dim, dropna=False)
            .agg(
                filtered_trades=("pnl", "count"),
                filtered_pnl_sum=("pnl", "sum"),
                filtered_win_rate=("pnl", lambda s: float((s > 0).mean()) if len(s) else 0.0),
            )
            .reset_index()
        )

        merged = base_group.merge(filt_group, on=dim, how="outer")
        merged["dimension"] = dim
        merged["trade_reduction"] = merged["baseline_trades"].fillna(0) - merged["filtered_trades"].fillna(0)
        merged["pnl_change"] = merged["filtered_pnl_sum"].fillna(0.0) - merged["baseline_pnl_sum"].fillna(0.0)
        merged["win_rate_change"] = merged["filtered_win_rate"].fillna(0.0) - merged["baseline_win_rate"].fillna(0.0)
        merged = merged.rename(columns={dim: "bucket"})

        for _, row in merged.iterrows():
            rows.append(
                {
                    "dimension": row["dimension"],
                    "bucket": row["bucket"],
                    "baseline_trades": int(row["baseline_trades"]) if pd.notna(row["baseline_trades"]) else 0,
                    "filtered_trades": int(row["filtered_trades"]) if pd.notna(row["filtered_trades"]) else 0,
                    "trade_reduction": int(row["trade_reduction"]) if pd.notna(row["trade_reduction"]) else 0,
                    "baseline_pnl_sum": float(row["baseline_pnl_sum"]) if pd.notna(row["baseline_pnl_sum"]) else 0.0,
                    "filtered_pnl_sum": float(row["filtered_pnl_sum"]) if pd.notna(row["filtered_pnl_sum"]) else 0.0,
                    "pnl_change": float(row["pnl_change"]) if pd.notna(row["pnl_change"]) else 0.0,
                    "baseline_win_rate": float(row["baseline_win_rate"]) if pd.notna(row["baseline_win_rate"]) else 0.0,
                    "filtered_win_rate": float(row["filtered_win_rate"]) if pd.notna(row["filtered_win_rate"]) else 0.0,
                    "win_rate_change": float(row["win_rate_change"]) if pd.notna(row["win_rate_change"]) else 0.0,
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["dimension", "pnl_change", "trade_reduction"], ascending=[True, False, False]).reset_index(drop=True)
    return out


def build_summary_report(
    baseline_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    blocked_df: pd.DataFrame,
) -> Dict[str, object]:
    baseline = compute_metrics(baseline_df)
    filtered = compute_metrics(filtered_df)

    blocked_net_pnl = float(blocked_df["pnl"].sum()) if len(blocked_df) else 0.0
    blocked_win_rate = float((blocked_df["pnl"] > 0).mean()) if len(blocked_df) else 0.0

    report = {
        "version": VERSION,
        "rule_name": "block_short_below_ema_stack",
        "rule_logic": "Remove trades where side == SHORT and price_location_bucket == BELOW_EMA_STACK",
        "baseline": baseline,
        "filtered": filtered,
        "delta": {
            "trade_change": filtered["trades"] - baseline["trades"],
            "net_pnl_change": filtered["net_pnl"] - baseline["net_pnl"],
            "profit_factor_change": (
                filtered["profit_factor"] - baseline["profit_factor"]
                if np.isfinite(filtered["profit_factor"]) and np.isfinite(baseline["profit_factor"])
                else None
            ),
            "win_rate_change": filtered["win_rate"] - baseline["win_rate"],
            "signal_exit_count_change": filtered["signal_exit_count"] - baseline["signal_exit_count"],
            "signal_exit_net_pnl_change": filtered["signal_exit_net_pnl"] - baseline["signal_exit_net_pnl"],
            "signal_exit_share_change": filtered["signal_exit_share"] - baseline["signal_exit_share"],
        },
        "blocked_trades": {
            "count": int(len(blocked_df)),
            "net_pnl_removed": blocked_net_pnl,
            "win_rate_removed": blocked_win_rate,
            "signal_exit_count_removed": int((blocked_df["exit_reason"].str.lower() == "signal_exit").sum()) if len(blocked_df) else 0,
        },
        "decision_hint": "PROMISING" if filtered["net_pnl"] > baseline["net_pnl"] else "NOT_IMPROVED",
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate rule: block SHORT entries when price_location_bucket is BELOW_EMA_STACK.")
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
    baseline_df = prepare_dataframe(df_raw)
    filtered_df = apply_rule_block_short_below_ema(baseline_df)
    blocked_df = build_blocked_trades_table(baseline_df)

    baseline_metrics_df = pd.DataFrame([compute_metrics(baseline_df)])
    filtered_metrics_df = pd.DataFrame([compute_metrics(filtered_df)])
    blocked_metrics_df = pd.DataFrame([compute_metrics(blocked_df)]) if len(blocked_df) else pd.DataFrame()

    baseline_summary_by_dim = pd.concat(
        [
            aggregate_dimension(baseline_df, "trend_bucket"),
            aggregate_dimension(baseline_df, "volatility_bucket"),
            aggregate_dimension(baseline_df, "session_bucket"),
            aggregate_dimension(baseline_df, "price_location_bucket"),
            aggregate_dimension(baseline_df, "hold_bucket"),
            aggregate_dimension(baseline_df, "exit_reason"),
            aggregate_dimension(baseline_df, "side"),
        ],
        ignore_index=True,
    )

    filtered_summary_by_dim = pd.concat(
        [
            aggregate_dimension(filtered_df, "trend_bucket"),
            aggregate_dimension(filtered_df, "volatility_bucket"),
            aggregate_dimension(filtered_df, "session_bucket"),
            aggregate_dimension(filtered_df, "price_location_bucket"),
            aggregate_dimension(filtered_df, "hold_bucket"),
            aggregate_dimension(filtered_df, "exit_reason"),
            aggregate_dimension(filtered_df, "side"),
        ],
        ignore_index=True,
    )

    comparison_table = build_comparison_tables(baseline_df, filtered_df)
    report = build_summary_report(baseline_df, filtered_df, blocked_df)

    baseline_metrics_path = outdir / "baseline_metrics.csv"
    filtered_metrics_path = outdir / "filtered_metrics.csv"
    blocked_trades_path = outdir / "blocked_trades.csv"
    blocked_metrics_path = outdir / "blocked_metrics.csv"
    baseline_summary_path = outdir / "baseline_summary_by_dimension.csv"
    filtered_summary_path = outdir / "filtered_summary_by_dimension.csv"
    comparison_table_path = outdir / "comparison_by_dimension.csv"
    report_path = outdir / "rule_test_report.json"

    baseline_metrics_df.to_csv(baseline_metrics_path, index=False, encoding="utf-8")
    filtered_metrics_df.to_csv(filtered_metrics_path, index=False, encoding="utf-8")
    blocked_df.to_csv(blocked_trades_path, index=False, encoding="utf-8")
    if not blocked_metrics_df.empty:
        blocked_metrics_df.to_csv(blocked_metrics_path, index=False, encoding="utf-8")
    baseline_summary_by_dim.to_csv(baseline_summary_path, index=False, encoding="utf-8")
    filtered_summary_by_dim.to_csv(filtered_summary_path, index=False, encoding="utf-8")
    comparison_table.to_csv(comparison_table_path, index=False, encoding="utf-8")

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    log(f"saved={baseline_metrics_path}")
    log(f"saved={filtered_metrics_path}")
    log(f"saved={blocked_trades_path}")
    log(f"saved={baseline_summary_path}")
    log(f"saved={filtered_summary_path}")
    log(f"saved={comparison_table_path}")
    log(f"saved={report_path}")
    log(f"baseline_net_pnl={report['baseline']['net_pnl']:.6f}")
    log(f"filtered_net_pnl={report['filtered']['net_pnl']:.6f}")
    log(f"net_pnl_change={report['delta']['net_pnl_change']:.6f}")
    log(f"decision_hint={report['decision_hint']}")


if __name__ == "__main__":
    main()
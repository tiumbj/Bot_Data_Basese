# ============================================================
# ชื่อโค้ด: batch_validate_pullback_deep_block_short_below_ema.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\batch_validate_pullback_deep_block_short_below_ema.py
# คำสั่งรัน:
# python C:\Data\Bot\Local_LLM\gold_research\batch_validate_pullback_deep_block_short_below_ema.py --index "C:\Data\data_base\backtest_results\central_backtest_results\index\backtest_index.csv" --ohlc "C:\Data\Bot\central_market_data\tf\XAUUSD_M30.csv" --outdir "C:\Data\data_base\backtest_results\gold_research_outputs\m30_pullback_deep_block_short_below_ema_batch"
# เวอร์ชัน: v1.0.0
# ============================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


VERSION = "v1.0.0"
TARGET_STRATEGY_ID = "locked_ms_bos_choch_pullback_atr_adx_ema_entry_v2_pullback_deep"
TARGET_TIMEFRAME = "M30"


def log(message: str) -> None:
    print(f"[batch_validate_pullback_deep_block_short_below_ema {VERSION}] {message}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def parse_dt(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().all():
        raise ValueError("Datetime parse failed.")
    return parsed


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def find_time_col(df: pd.DataFrame) -> str:
    cols = {c.lower(): c for c in df.columns}
    for key in ["time", "datetime", "timestamp", "date"]:
        if key in cols:
            return cols[key]
    for c in df.columns:
        low = c.lower()
        if "time" in low or "date" in low:
            return c
    raise ValueError("Cannot detect OHLC time column.")


def find_ohlc_cols(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    cols = {c.lower(): c for c in df.columns}

    def pick(keys: List[str], label: str) -> str:
        for k in keys:
            if k in cols:
                return cols[k]
        raise ValueError(f"Cannot detect {label} column.")

    return (
        pick(["open", "o"], "open"),
        pick(["high", "h"], "high"),
        pick(["low", "l"], "low"),
        pick(["close", "c"], "close"),
    )


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    tr = true_range(high, low, close)
    atr_smoothed = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_smoothed
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_smoothed

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def classify_session(hour: int) -> str:
    if 0 <= hour < 7:
        return "ASIA"
    if 7 <= hour < 13:
        return "LONDON"
    if 13 <= hour < 17:
        return "NY_PREOPEN"
    if 17 <= hour < 22:
        return "NEW_YORK"
    return "LATE_US"


def classify_vol_bucket(value: float, q33: float, q66: float) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    if value <= q33:
        return "LOW_VOL"
    if value <= q66:
        return "MID_VOL"
    return "HIGH_VOL"


def classify_trend_bucket(adx14_value: float, ema_gap_pct: float) -> str:
    if pd.isna(adx14_value) or pd.isna(ema_gap_pct):
        return "UNKNOWN"
    if adx14_value >= 25 and ema_gap_pct > 0:
        return "BULL_TREND"
    if adx14_value >= 25 and ema_gap_pct < 0:
        return "BEAR_TREND"
    if adx14_value < 20:
        return "RANGE"
    if ema_gap_pct > 0:
        return "WEAK_BULL"
    if ema_gap_pct < 0:
        return "WEAK_BEAR"
    return "RANGE"


def classify_price_location(close_to_ema20_pct: float, close_to_ema50_pct: float) -> str:
    if pd.isna(close_to_ema20_pct) or pd.isna(close_to_ema50_pct):
        return "UNKNOWN"
    if close_to_ema20_pct > 0 and close_to_ema50_pct > 0:
        return "ABOVE_EMA_STACK"
    if close_to_ema20_pct < 0 and close_to_ema50_pct < 0:
        return "BELOW_EMA_STACK"
    return "INSIDE_EMA_ZONE"


def classify_hold_bucket(minutes: float) -> str:
    if pd.isna(minutes):
        return "UNKNOWN"
    if minutes <= 30:
        return "FAST_EXIT"
    if minutes <= 120:
        return "NORMAL_EXIT"
    if minutes <= 360:
        return "INTRADAY_SWING"
    return "LONG_HOLD"


def prepare_ohlc(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df_raw)
    time_col = find_time_col(df)
    open_col, high_col, low_col, close_col = find_ohlc_cols(df)

    out = pd.DataFrame(
        {
            "bar_time": parse_dt(df[time_col]),
            "open": to_num(df[open_col]),
            "high": to_num(df[high_col]),
            "low": to_num(df[low_col]),
            "close": to_num(df[close_col]),
        }
    )

    out = out.dropna(subset=["bar_time", "open", "high", "low", "close"]).copy()
    out = out.sort_values("bar_time").drop_duplicates(subset=["bar_time"]).reset_index(drop=True)

    out["atr14"] = atr(out["high"], out["low"], out["close"], 14)
    out["adx14"] = adx(out["high"], out["low"], out["close"], 14)
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)

    out["ema_gap_abs"] = out["ema20"] - out["ema50"]
    out["ema_gap_pct"] = np.where(out["ema50"] != 0, out["ema_gap_abs"] / out["ema50"], np.nan)

    out["close_to_ema20_abs"] = out["close"] - out["ema20"]
    out["close_to_ema50_abs"] = out["close"] - out["ema50"]
    out["close_to_ema20_pct"] = np.where(out["ema20"] != 0, out["close_to_ema20_abs"] / out["ema20"], np.nan)
    out["close_to_ema50_pct"] = np.where(out["ema50"] != 0, out["close_to_ema50_abs"] / out["ema50"], np.nan)

    out["bar_range"] = out["high"] - out["low"]
    out["atr_ratio"] = np.where(out["close"] != 0, out["atr14"] / out["close"], np.nan)

    out["hour"] = out["bar_time"].dt.hour
    out["session_bucket"] = out["hour"].apply(classify_session)

    q33 = out["atr_ratio"].quantile(0.33)
    q66 = out["atr_ratio"].quantile(0.66)

    out["volatility_bucket"] = out["atr_ratio"].apply(lambda x: classify_vol_bucket(x, q33, q66))
    out["trend_bucket"] = [classify_trend_bucket(a, g) for a, g in zip(out["adx14"], out["ema_gap_pct"])]
    out["price_location_bucket"] = [
        classify_price_location(a, b)
        for a, b in zip(out["close_to_ema20_pct"], out["close_to_ema50_pct"])
    ]

    return out


def prepare_trades(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df_raw)

    required = [
        "entry_time",
        "exit_time",
        "side",
        "entry_price",
        "exit_price",
        "sl_price",
        "tp_price",
        "pnl",
        "exit_reason",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required trade columns: {missing}")

    out = df.copy()
    out["entry_time"] = parse_dt(out["entry_time"])
    out["exit_time"] = parse_dt(out["exit_time"])
    out["entry_price"] = to_num(out["entry_price"])
    out["exit_price"] = to_num(out["exit_price"])
    out["sl_price"] = to_num(out["sl_price"])
    out["tp_price"] = to_num(out["tp_price"])
    out["pnl"] = to_num(out["pnl"])
    out["side"] = out["side"].astype(str).str.upper().str.strip()
    out["exit_reason"] = out["exit_reason"].astype(str).str.strip()

    out = out.dropna(subset=["entry_time", "exit_time", "entry_price", "exit_price", "pnl"]).copy()
    out = out.sort_values("entry_time").reset_index(drop=True)

    out["hold_minutes"] = (out["exit_time"] - out["entry_time"]).dt.total_seconds() / 60.0
    out["hold_bucket"] = out["hold_minutes"].apply(classify_hold_bucket)
    out["trade_result"] = np.where(out["pnl"] > 0, "WIN", np.where(out["pnl"] < 0, "LOSS", "FLAT"))

    out["risk_distance"] = np.where(
        out["side"] == "LONG",
        out["entry_price"] - out["sl_price"],
        out["sl_price"] - out["entry_price"],
    )
    out["reward_distance"] = np.where(
        out["side"] == "LONG",
        out["tp_price"] - out["entry_price"],
        out["entry_price"] - out["tp_price"],
    )

    out["rr_planned"] = np.where(out["risk_distance"] != 0, out["reward_distance"] / out["risk_distance"], np.nan)
    out["pnl_r"] = np.where(out["risk_distance"] != 0, out["pnl"] / out["risk_distance"], np.nan)

    return out


def enrich_trades(trades_df: pd.DataFrame, ohlc_df: pd.DataFrame) -> pd.DataFrame:
    left = trades_df.sort_values("entry_time").copy()
    right = ohlc_df.sort_values("bar_time").copy()

    enriched = pd.merge_asof(
        left,
        right,
        left_on="entry_time",
        right_on="bar_time",
        direction="backward",
        tolerance=pd.Timedelta("30min"),
    )

    enriched["matched_bar_delta_min"] = (
        (enriched["entry_time"] - enriched["bar_time"]).dt.total_seconds() / 60.0
    )
    enriched["join_status"] = np.where(enriched["bar_time"].isna(), "UNMATCHED", "MATCHED")
    return enriched


def apply_rule(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    blocked_mask = (
        (df["side"] == "SHORT") &
        (df["price_location_bucket"] == "BELOW_EMA_STACK")
    )
    blocked = df.loc[blocked_mask].copy().reset_index(drop=True)
    filtered = df.loc[~blocked_mask].copy().reset_index(drop=True)
    return filtered, blocked


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    trades = int(len(df))
    wins = int((df["pnl"] > 0).sum())
    losses = int((df["pnl"] < 0).sum())
    flats = int((df["pnl"] == 0).sum())

    gross_profit = float(df.loc[df["pnl"] > 0, "pnl"].sum())
    gross_loss_abs = float(abs(df.loc[df["pnl"] < 0, "pnl"].sum()))
    net_pnl = float(df["pnl"].sum())
    avg_pnl = float(df["pnl"].mean()) if trades else 0.0
    win_rate = float(wins / trades) if trades else 0.0
    signal_exit_count = int((df["exit_reason"].str.lower() == "signal_exit").sum())
    signal_exit_net_pnl = float(df.loc[df["exit_reason"].str.lower() == "signal_exit", "pnl"].sum())
    signal_exit_win_rate = float(
        (df.loc[df["exit_reason"].str.lower() == "signal_exit", "pnl"] > 0).mean()
    ) if signal_exit_count else 0.0
    profit_factor = gross_profit / gross_loss_abs if gross_loss_abs > 0 else (float("inf") if gross_profit > 0 else 0.0)

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
        "win_rate": win_rate,
        "signal_exit_count": signal_exit_count,
        "signal_exit_share": float(signal_exit_count / trades) if trades else 0.0,
        "signal_exit_net_pnl": signal_exit_net_pnl,
        "signal_exit_win_rate": signal_exit_win_rate,
    }


def summarize_one_row(meta: Dict[str, str], baseline_df: pd.DataFrame, filtered_df: pd.DataFrame, blocked_df: pd.DataFrame) -> Dict[str, object]:
    baseline = compute_metrics(baseline_df)
    filtered = compute_metrics(filtered_df)
    blocked = compute_metrics(blocked_df) if len(blocked_df) else {
        "trades": 0,
        "net_pnl": 0.0,
        "win_rate": 0.0,
        "signal_exit_count": 0,
    }

    return {
        "strategy_id": meta["strategy_id"],
        "timeframe": meta["timeframe"],
        "validation_mode": meta["validation_mode"],
        "window_name": meta["window_name"],
        "result_path": meta["result_path"],
        "baseline_trades": baseline["trades"],
        "baseline_net_pnl": baseline["net_pnl"],
        "baseline_profit_factor": baseline["profit_factor"],
        "baseline_win_rate": baseline["win_rate"],
        "baseline_signal_exit_count": baseline["signal_exit_count"],
        "baseline_signal_exit_share": baseline["signal_exit_share"],
        "baseline_signal_exit_net_pnl": baseline["signal_exit_net_pnl"],
        "filtered_trades": filtered["trades"],
        "filtered_net_pnl": filtered["net_pnl"],
        "filtered_profit_factor": filtered["profit_factor"],
        "filtered_win_rate": filtered["win_rate"],
        "filtered_signal_exit_count": filtered["signal_exit_count"],
        "filtered_signal_exit_share": filtered["signal_exit_share"],
        "filtered_signal_exit_net_pnl": filtered["signal_exit_net_pnl"],
        "trade_change": filtered["trades"] - baseline["trades"],
        "net_pnl_change": filtered["net_pnl"] - baseline["net_pnl"],
        "profit_factor_change": (
            filtered["profit_factor"] - baseline["profit_factor"]
            if np.isfinite(filtered["profit_factor"]) and np.isfinite(baseline["profit_factor"])
            else np.nan
        ),
        "win_rate_change": filtered["win_rate"] - baseline["win_rate"],
        "signal_exit_count_change": filtered["signal_exit_count"] - baseline["signal_exit_count"],
        "signal_exit_net_pnl_change": filtered["signal_exit_net_pnl"] - baseline["signal_exit_net_pnl"],
        "blocked_trades": blocked["trades"],
        "blocked_net_pnl_removed": blocked["net_pnl"],
        "blocked_win_rate_removed": blocked["win_rate"],
        "blocked_signal_exit_count_removed": blocked["signal_exit_count"],
        "decision_hint": "PROMISING" if filtered["net_pnl"] > baseline["net_pnl"] else "NOT_IMPROVED",
    }


def build_batch_report(results_df: pd.DataFrame) -> Dict[str, object]:
    total_rows = len(results_df)
    promising_rows = int((results_df["decision_hint"] == "PROMISING").sum()) if total_rows else 0

    return {
        "version": VERSION,
        "strategy_id": TARGET_STRATEGY_ID,
        "timeframe": TARGET_TIMEFRAME,
        "windows_total": int(total_rows),
        "windows_promising": promising_rows,
        "promising_ratio": float(promising_rows / total_rows) if total_rows else 0.0,
        "baseline_net_pnl_sum": float(results_df["baseline_net_pnl"].sum()) if total_rows else 0.0,
        "filtered_net_pnl_sum": float(results_df["filtered_net_pnl"].sum()) if total_rows else 0.0,
        "net_pnl_change_sum": float(results_df["net_pnl_change"].sum()) if total_rows else 0.0,
        "baseline_signal_exit_net_pnl_sum": float(results_df["baseline_signal_exit_net_pnl"].sum()) if total_rows else 0.0,
        "filtered_signal_exit_net_pnl_sum": float(results_df["filtered_signal_exit_net_pnl"].sum()) if total_rows else 0.0,
        "signal_exit_net_pnl_change_sum": float(results_df["signal_exit_net_pnl_change"].sum()) if total_rows else 0.0,
        "top_windows_by_improvement": results_df.sort_values("net_pnl_change", ascending=False)
        .head(10)[
            [
                "validation_mode",
                "window_name",
                "baseline_net_pnl",
                "filtered_net_pnl",
                "net_pnl_change",
                "baseline_profit_factor",
                "filtered_profit_factor",
                "decision_hint",
            ]
        ].to_dict(orient="records") if total_rows else [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch validate rule: block SHORT when price_location_bucket == BELOW_EMA_STACK.")
    parser.add_argument("--index", required=True, help="Path to backtest_index.csv")
    parser.add_argument("--ohlc", required=True, help="Path to source OHLC CSV")
    parser.add_argument("--outdir", required=True, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    index_path = Path(args.index)
    ohlc_path = Path(args.ohlc)
    outdir = Path(args.outdir)

    ensure_dir(outdir)

    log(f"version={VERSION}")
    log(f"index={index_path}")
    log(f"ohlc={ohlc_path}")
    log(f"outdir={outdir}")

    index_df = normalize_columns(load_csv(index_path))
    ohlc_df = prepare_ohlc(load_csv(ohlc_path))

    target_rows = index_df.loc[
        (index_df["strategy_id"] == TARGET_STRATEGY_ID) &
        (index_df["timeframe"] == TARGET_TIMEFRAME)
    ].copy()

    if target_rows.empty:
        raise ValueError("No matching rows found in backtest_index.csv for target strategy/timeframe.")

    results: List[Dict[str, object]] = []

    for _, row in target_rows.iterrows():
        result_path = Path(str(row["result_path"]))
        trades_path = result_path / "trades.csv"

        if not trades_path.exists():
            log(f"skip_missing_trades={trades_path}")
            continue

        trades_df = prepare_trades(load_csv(trades_path))
        enriched_df = enrich_trades(trades_df, ohlc_df)

        if int((enriched_df["join_status"] == "UNMATCHED").sum()) > 0:
            log(f"warning_unmatched_rows={trades_path}")

        filtered_df, blocked_df = apply_rule(enriched_df)

        meta = {
            "strategy_id": str(row["strategy_id"]),
            "timeframe": str(row["timeframe"]),
            "validation_mode": str(row["validation_mode"]),
            "window_name": str(row["window_name"]),
            "result_path": str(row["result_path"]),
        }

        result_row = summarize_one_row(meta, enriched_df, filtered_df, blocked_df)
        results.append(result_row)

        log(
            f"done validation_mode={meta['validation_mode']} window={meta['window_name']} "
            f"baseline_net={result_row['baseline_net_pnl']:.6f} "
            f"filtered_net={result_row['filtered_net_pnl']:.6f} "
            f"delta={result_row['net_pnl_change']:.6f} "
            f"decision={result_row['decision_hint']}"
        )

    if not results:
        raise ValueError("No windows processed successfully.")

    results_df = pd.DataFrame(results).sort_values(
        ["validation_mode", "window_name"]
    ).reset_index(drop=True)

    batch_report = build_batch_report(results_df)

    results_csv_path = outdir / "batch_window_results.csv"
    report_json_path = outdir / "batch_report.json"

    results_df.to_csv(results_csv_path, index=False, encoding="utf-8")
    with report_json_path.open("w", encoding="utf-8") as file:
        json.dump(batch_report, file, indent=2, ensure_ascii=False)

    log(f"saved={results_csv_path}")
    log(f"saved={report_json_path}")
    log(f"windows_total={batch_report['windows_total']}")
    log(f"windows_promising={batch_report['windows_promising']}")
    log(f"net_pnl_change_sum={batch_report['net_pnl_change_sum']:.6f}")


if __name__ == "__main__":
    main()
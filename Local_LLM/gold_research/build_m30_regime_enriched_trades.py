# ============================================================
# ชื่อโค้ด: build_m30_regime_enriched_trades.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\build_m30_regime_enriched_trades.py
# คำสั่งรัน:
# python C:\Data\Bot\Local_LLM\gold_research\build_m30_regime_enriched_trades.py --trades "C:\Data\data_base\backtest_results\central_backtest_results\weekly\2026-W12\M30\insample\locked_ms_bos_choch_pullback_atr_adx_ema_entry_v2_adx25\v1.0.0\window_01\trades.csv" --ohlc "C:\Data\Bot\central_market_data\tf\XAUUSD_M30.csv" --outdir "C:\Data\data_base\backtest_results\gold_research_outputs\m30_regime_adx25_window01"
# เวอร์ชัน: v1.0.1
# ============================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


VERSION = "v1.0.1"


def log(message: str) -> None:
    print(f"[build_m30_regime_enriched_trades {VERSION}] {message}")


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

    open_col = pick(["open", "o"], "open")
    high_col = pick(["high", "h"], "high")
    low_col = pick(["low", "l"], "low")
    close_col = pick(["close", "c"], "close")
    return open_col, high_col, low_col, close_col


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
    out["bar_range_ratio"] = np.where(out["close"] != 0, out["bar_range"] / out["close"], np.nan)

    out["hour"] = out["bar_time"].dt.hour
    out["weekday"] = out["bar_time"].dt.day_name()
    out["session_bucket"] = out["hour"].apply(classify_session)

    q33 = out["atr_ratio"].quantile(0.33)
    q66 = out["atr_ratio"].quantile(0.66)

    out["volatility_bucket"] = out["atr_ratio"].apply(lambda x: classify_vol_bucket(x, q33, q66))
    out["trend_bucket"] = [
        classify_trend_bucket(a, g) for a, g in zip(out["adx14"], out["ema_gap_pct"])
    ]
    out["price_location_bucket"] = [
        classify_price_location(a, b) for a, b in zip(out["close_to_ema20_pct"], out["close_to_ema50_pct"])
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


def aggregate_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            trades=("pnl", "count"),
            wins=("trade_result", lambda s: int((s == "WIN").sum())),
            losses=("trade_result", lambda s: int((s == "LOSS").sum())),
            pnl_sum=("pnl", "sum"),
            pnl_avg=("pnl", "mean"),
            pnl_median=("pnl", "median"),
            pnl_r_avg=("pnl_r", "mean"),
            hold_min_avg=("hold_minutes", "mean"),
            atr14_avg=("atr14", "mean"),
            adx14_avg=("adx14", "mean"),
            rr_planned_avg=("rr_planned", "mean"),
        )
        .reset_index()
    )

    grouped["win_rate"] = np.where(grouped["trades"] > 0, grouped["wins"] / grouped["trades"], np.nan)
    grouped = grouped.sort_values(["pnl_sum", "win_rate", "trades"], ascending=[False, False, False])
    return grouped


def build_regime_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []

    dimensions = [
        "trend_bucket",
        "volatility_bucket",
        "session_bucket",
        "price_location_bucket",
        "exit_reason",
        "hold_bucket",
        "side",
    ]

    for dim in dimensions:
        part = aggregate_group(enriched, dim)
        part.insert(0, "dimension", dim)
        part = part.rename(columns={dim: "bucket"})
        parts.append(part)

    return pd.concat(parts, ignore_index=True)


def build_report(enriched: pd.DataFrame, summary: pd.DataFrame, trades_path: Path, ohlc_path: Path, outdir: Path) -> dict:
    report = {
        "version": VERSION,
        "trades_path": str(trades_path),
        "ohlc_path": str(ohlc_path),
        "outdir": str(outdir),
        "rows_trades_enriched": int(len(enriched)),
        "rows_regime_summary": int(len(summary)),
        "matched_rows": int((enriched["join_status"] == "MATCHED").sum()),
        "unmatched_rows": int((enriched["join_status"] == "UNMATCHED").sum()),
        "overall": {
            "trades": int(len(enriched)),
            "wins": int((enriched["trade_result"] == "WIN").sum()),
            "losses": int((enriched["trade_result"] == "LOSS").sum()),
            "flats": int((enriched["trade_result"] == "FLAT").sum()),
            "net_pnl": float(enriched["pnl"].sum()),
            "avg_pnl": float(enriched["pnl"].mean()) if len(enriched) else 0.0,
            "win_rate": float((enriched["trade_result"] == "WIN").mean()) if len(enriched) else 0.0,
            "avg_hold_minutes": float(enriched["hold_minutes"].mean()) if len(enriched) else 0.0,
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build regime-enriched trade dataset for M30 research.")
    parser.add_argument("--trades", required=True, help="Path to trades.csv")
    parser.add_argument("--ohlc", required=True, help="Path to OHLC CSV")
    parser.add_argument("--outdir", required=True, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    trades_path = Path(args.trades)
    ohlc_path = Path(args.ohlc)
    outdir = Path(args.outdir)

    ensure_dir(outdir)

    log(f"version={VERSION}")
    log(f"trades={trades_path}")
    log(f"ohlc={ohlc_path}")
    log(f"outdir={outdir}")

    trades_raw = load_csv(trades_path)
    ohlc_raw = load_csv(ohlc_path)

    trades_df = prepare_trades(trades_raw)
    ohlc_df = prepare_ohlc(ohlc_raw)

    enriched = enrich_trades(trades_df, ohlc_df)
    summary = build_regime_summary(enriched)
    report = build_report(enriched, summary, trades_path, ohlc_path, outdir)

    trades_enriched_path = outdir / "trades_enriched.csv"
    regime_summary_path = outdir / "regime_summary.csv"
    build_report_path = outdir / "build_report.json"

    enriched.to_csv(trades_enriched_path, index=False, encoding="utf-8")
    summary.to_csv(regime_summary_path, index=False, encoding="utf-8")

    with build_report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    log(f"saved={trades_enriched_path}")
    log(f"saved={regime_summary_path}")
    log(f"saved={build_report_path}")
    log(f"matched_rows={report['matched_rows']} unmatched_rows={report['unmatched_rows']}")
    log(f"net_pnl={report['overall']['net_pnl']:.6f} win_rate={report['overall']['win_rate']:.4f}")


if __name__ == "__main__":
    main()
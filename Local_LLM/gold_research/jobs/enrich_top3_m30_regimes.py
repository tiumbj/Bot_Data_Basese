# ============================================================
# ชื่อโค้ด: enrich_top3_m30_regimes.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\enrich_top3_m30_regimes.py
# คำสั่งรัน: python C:\Data\Bot\Local_LLM\gold_research\jobs\enrich_top3_m30_regimes.py
# เวอร์ชัน: v1.0.0
# ============================================================

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

VERSION = "v1.0.0"

ROOT = Path(r"C:\Data\Bot")
SHORTLIST_CSV = ROOT / "central_backtest_results" / "index" / "final_top3_m30_shortlist.csv"
M30_PRICE_CSV = ROOT / "central_market_data" / "tf" / "XAUUSD_M30.csv"

PRICE_COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "timestamp": ["timestamp", "datetime", "date_time", "time", "Time", "DateTime", "DATE_TIME"],
    "open": ["open", "Open", "o", "O"],
    "high": ["high", "High", "h", "H"],
    "low": ["low", "Low", "l", "L"],
    "close": ["close", "Close", "c", "C"],
    "tick_volume": ["tick_volume", "TickVolume", "tickvol", "volume", "Volume"],
    "spread": ["spread", "Spread"],
    "real_volume": ["real_volume", "RealVolume", "realvol"],
}


@dataclass(frozen=True)
class ShortlistRow:
    strategy_id: str
    timeframe: str
    validation_mode: str
    window_name: str
    result_path: Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def find_first_existing_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    lower_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def normalize_price_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    columns = list(raw_df.columns)
    out = pd.DataFrame()

    ts_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["timestamp"])
    open_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["open"])
    high_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["high"])
    low_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["low"])
    close_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["close"])

    if ts_col is None:
        date_col = find_first_existing_column(columns, ["date", "Date", "DATE"])
        time_col = find_first_existing_column(columns, ["time", "Time", "TIME"])
        if date_col is None or time_col is None:
            raise RuntimeError("ไฟล์ราคา M30 ไม่มี timestamp หรือ date+time")
        out["timestamp"] = pd.to_datetime(
            raw_df[date_col].astype(str).str.strip() + " " + raw_df[time_col].astype(str).str.strip(),
            errors="coerce",
        )
    else:
        out["timestamp"] = pd.to_datetime(raw_df[ts_col], errors="coerce")

    missing = []
    for name, col in [("open", open_col), ("high", high_col), ("low", low_col), ("close", close_col)]:
        if col is None:
            missing.append(name)
    if missing:
        raise RuntimeError(f"ไฟล์ราคา M30 คอลัมน์ไม่ครบ | missing={missing}")

    out["open"] = pd.to_numeric(raw_df[open_col], errors="coerce")
    out["high"] = pd.to_numeric(raw_df[high_col], errors="coerce")
    out["low"] = pd.to_numeric(raw_df[low_col], errors="coerce")
    out["close"] = pd.to_numeric(raw_df[close_col], errors="coerce")

    tick_volume_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["tick_volume"])
    spread_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["spread"])
    real_volume_col = find_first_existing_column(columns, PRICE_COLUMN_CANDIDATES["real_volume"])

    out["tick_volume"] = pd.to_numeric(raw_df[tick_volume_col], errors="coerce").fillna(0.0) if tick_volume_col else 0.0
    out["spread"] = pd.to_numeric(raw_df[spread_col], errors="coerce").fillna(0.0) if spread_col else 0.0
    out["real_volume"] = pd.to_numeric(raw_df[real_volume_col], errors="coerce").fillna(0.0) if real_volume_col else 0.0

    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return out


def load_m30_price_context() -> pd.DataFrame:
    if not M30_PRICE_CSV.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ราคา M30 | file={M30_PRICE_CSV}")

    price_df = normalize_price_dataframe(pd.read_csv(M30_PRICE_CSV))

    price_df["ema20"] = price_df["close"].ewm(span=20, adjust=False).mean()
    price_df["ema50"] = price_df["close"].ewm(span=50, adjust=False).mean()

    prev_close = price_df["close"].shift(1)
    tr = pd.concat(
        [
            (price_df["high"] - price_df["low"]).abs(),
            (price_df["high"] - prev_close).abs(),
            (price_df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    price_df["atr14"] = tr.rolling(14, min_periods=14).mean()

    price_df["ema20_slope"] = price_df["ema20"] - price_df["ema20"].shift(3)
    price_df["ema50_slope"] = price_df["ema50"] - price_df["ema50"].shift(3)

    valid_atr = price_df["atr14"].dropna()
    if valid_atr.empty:
        low_q = 0.0
        high_q = 0.0
    else:
        low_q = float(valid_atr.quantile(0.33))
        high_q = float(valid_atr.quantile(0.66))

    price_df["session_bucket"] = price_df["timestamp"].apply(map_session_bucket)
    price_df["trend_bucket"] = price_df.apply(map_trend_bucket, axis=1)
    price_df["volatility_bucket"] = price_df["atr14"].apply(lambda x: map_volatility_bucket(x, low_q, high_q))
    price_df["price_location_bucket"] = price_df.apply(map_price_location_bucket, axis=1)

    # key สำหรับ merge_asof
    price_df = price_df.sort_values("timestamp").reset_index(drop=True)
    return price_df


def map_session_bucket(ts: pd.Timestamp) -> str:
    # ใช้ hour จาก timestamp ในไฟล์โดยตรง
    hour = int(pd.Timestamp(ts).hour)
    if 0 <= hour <= 6:
        return "ASIA"
    if 7 <= hour <= 12:
        return "LONDON"
    if 13 <= hour <= 14:
        return "NY_PREOPEN"
    if 15 <= hour <= 19:
        return "NEW_YORK"
    return "LATE_US"


def map_trend_bucket(row: pd.Series) -> str:
    close = float(row["close"]) if pd.notna(row["close"]) else np.nan
    ema20 = float(row["ema20"]) if pd.notna(row["ema20"]) else np.nan
    ema50 = float(row["ema50"]) if pd.notna(row["ema50"]) else np.nan
    slope20 = float(row["ema20_slope"]) if pd.notna(row["ema20_slope"]) else 0.0
    slope50 = float(row["ema50_slope"]) if pd.notna(row["ema50_slope"]) else 0.0

    if np.isnan(close) or np.isnan(ema20) or np.isnan(ema50):
        return "UNKNOWN"

    if close > ema20 > ema50 and slope20 > 0 and slope50 > 0:
        return "BULL_TREND"
    if close < ema20 < ema50 and slope20 < 0 and slope50 < 0:
        return "BEAR_TREND"
    if close >= ema50:
        return "WEAK_BULL"
    return "WEAK_BEAR"


def map_volatility_bucket(atr_value: float, low_q: float, high_q: float) -> str:
    if pd.isna(atr_value):
        return "UNKNOWN"
    x = float(atr_value)
    if x <= low_q:
        return "LOW_VOL"
    if x <= high_q:
        return "MID_VOL"
    return "HIGH_VOL"


def map_price_location_bucket(row: pd.Series) -> str:
    close = float(row["close"]) if pd.notna(row["close"]) else np.nan
    ema20 = float(row["ema20"]) if pd.notna(row["ema20"]) else np.nan
    ema50 = float(row["ema50"]) if pd.notna(row["ema50"]) else np.nan

    if np.isnan(close) or np.isnan(ema20) or np.isnan(ema50):
        return "UNKNOWN"
    if close > ema20 and close > ema50:
        return "ABOVE_EMA_STACK"
    if close < ema20 and close < ema50:
        return "BELOW_EMA_STACK"
    return "INSIDE_EMA_STACK"


def load_shortlist() -> List[ShortlistRow]:
    if not SHORTLIST_CSV.exists():
        raise FileNotFoundError(f"ไม่พบ shortlist | file={SHORTLIST_CSV}")

    df = pd.read_csv(SHORTLIST_CSV)
    required = ["strategy_id", "timeframe", "validation_mode", "window_name", "result_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"shortlist คอลัมน์ไม่ครบ | missing={missing}")

    rows: List[ShortlistRow] = []
    for _, row in df.iterrows():
        timeframe = str(row["timeframe"]).strip()
        if timeframe != "M30":
            continue
        rows.append(
            ShortlistRow(
                strategy_id=str(row["strategy_id"]).strip(),
                timeframe=timeframe,
                validation_mode=str(row["validation_mode"]).strip(),
                window_name=str(row["window_name"]).strip(),
                result_path=Path(str(row["result_path"]).strip()),
            )
        )
    return rows


def load_trades_csv(trades_path: Path) -> pd.DataFrame:
    if not trades_path.exists():
        raise FileNotFoundError(f"ไม่พบ trades.csv | file={trades_path}")

    df = pd.read_csv(trades_path)
    required = ["entry_time", "exit_time", "side", "entry_price", "exit_price", "sl_price", "tp_price", "pnl", "exit_reason"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"trades.csv คอลัมน์ไม่ครบ | file={trades_path} | missing={missing}")

    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    for col in ["entry_price", "exit_price", "sl_price", "tp_price", "pnl"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["entry_time", "exit_time", "entry_price", "exit_price", "pnl"]).copy()
    return df.reset_index(drop=True)


def enrich_trades(trades_df: pd.DataFrame, price_context_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    entry_map_cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "ema20",
        "ema50",
        "atr14",
        "session_bucket",
        "trend_bucket",
        "volatility_bucket",
        "price_location_bucket",
    ]
    exit_map_cols = entry_map_cols.copy()

    entry_ctx = price_context_df[entry_map_cols].rename(columns=lambda c: f"entry_{c}" if c != "timestamp" else "entry_bar_time")
    exit_ctx = price_context_df[exit_map_cols].rename(columns=lambda c: f"exit_{c}" if c != "timestamp" else "exit_bar_time")

    left = trades_df.sort_values("entry_time").reset_index(drop=True)
    left = pd.merge_asof(
        left.sort_values("entry_time"),
        entry_ctx.sort_values("entry_bar_time"),
        left_on="entry_time",
        right_on="entry_bar_time",
        direction="backward",
        tolerance=pd.Timedelta("30min"),
    )

    left = pd.merge_asof(
        left.sort_values("exit_time"),
        exit_ctx.sort_values("exit_bar_time"),
        left_on="exit_time",
        right_on="exit_bar_time",
        direction="backward",
        tolerance=pd.Timedelta("30min"),
    )

    left["session_bucket"] = left["entry_session_bucket"].fillna("UNKNOWN")
    left["trend_bucket"] = left["entry_trend_bucket"].fillna("UNKNOWN")
    left["volatility_bucket"] = left["entry_volatility_bucket"].fillna("UNKNOWN")
    left["price_location_bucket"] = left["entry_price_location_bucket"].fillna("UNKNOWN")

    left["holding_minutes"] = (left["exit_time"] - left["entry_time"]).dt.total_seconds() / 60.0
    left["rr_realized"] = np.where(
        (left["side"].astype(str).str.upper() == "LONG"),
        (left["exit_price"] - left["entry_price"]) / (left["entry_price"] - left["sl_price"]).replace(0, np.nan),
        (left["entry_price"] - left["exit_price"]) / (left["sl_price"] - left["entry_price"]).replace(0, np.nan),
    )

    matched_entry = int(left["entry_bar_time"].notna().sum())
    matched_exit = int(left["exit_bar_time"].notna().sum())
    report = {
        "trade_rows": int(len(left)),
        "matched_entry_rows": matched_entry,
        "matched_exit_rows": matched_exit,
        "unmatched_entry_rows": int(len(left) - matched_entry),
        "unmatched_exit_rows": int(len(left) - matched_exit),
        "match_rate_entry": round(matched_entry / max(len(left), 1), 6),
        "match_rate_exit": round(matched_exit / max(len(left), 1), 6),
    }

    ordered_cols = [
        "entry_time",
        "exit_time",
        "side",
        "entry_price",
        "exit_price",
        "sl_price",
        "tp_price",
        "pnl",
        "exit_reason",
        "holding_minutes",
        "rr_realized",
        "session_bucket",
        "trend_bucket",
        "volatility_bucket",
        "price_location_bucket",
        "entry_bar_time",
        "entry_open",
        "entry_high",
        "entry_low",
        "entry_close",
        "entry_ema20",
        "entry_ema50",
        "entry_atr14",
        "entry_session_bucket",
        "entry_trend_bucket",
        "entry_volatility_bucket",
        "entry_price_location_bucket",
        "exit_bar_time",
        "exit_open",
        "exit_high",
        "exit_low",
        "exit_close",
        "exit_ema20",
        "exit_ema50",
        "exit_atr14",
        "exit_session_bucket",
        "exit_trend_bucket",
        "exit_volatility_bucket",
        "exit_price_location_bucket",
    ]
    existing_cols = [c for c in ordered_cols if c in left.columns]
    other_cols = [c for c in left.columns if c not in existing_cols]
    left = left[existing_cols + other_cols]

    return left, report


def build_bucket_summary(
    trades_enriched_df: pd.DataFrame,
    bucket_col: str,
    bucket_type: str,
) -> pd.DataFrame:
    if bucket_col not in trades_enriched_df.columns:
        return pd.DataFrame()

    rows = []
    for bucket_name, g in trades_enriched_df.groupby(bucket_col, dropna=False):
        g = g.copy()
        pnl = pd.to_numeric(g["pnl"], errors="coerce").fillna(0.0)
        wins = int((pnl > 0).sum())
        losses = int((pnl < 0).sum())
        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = float(-pnl[pnl < 0].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

        rows.append(
            {
                "bucket_type": bucket_type,
                "bucket_name": str(bucket_name) if pd.notna(bucket_name) else "UNKNOWN",
                "trades": int(len(g)),
                "wins": wins,
                "losses": losses,
                "win_rate": round((wins / max(len(g), 1)) * 100.0, 6),
                "pnl_sum": round(float(pnl.sum()), 6),
                "avg_pnl": round(float(pnl.mean()) if len(g) > 0 else 0.0, 6),
                "profit_factor": round(999999.0 if np.isinf(pf) else float(pf), 6),
                "avg_holding_minutes": round(float(pd.to_numeric(g["holding_minutes"], errors="coerce").mean()), 6),
                "avg_rr_realized": round(float(pd.to_numeric(g["rr_realized"], errors="coerce").mean()), 6),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["bucket_type", "pnl_sum", "profit_factor"], ascending=[True, False, False]).reset_index(drop=True)
    return out


def build_exit_reason_summary(trades_enriched_df: pd.DataFrame) -> pd.DataFrame:
    return build_bucket_summary(trades_enriched_df, "exit_reason", "exit_reason")


def save_dataframe_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    out = df.copy()
    for col in ["entry_time", "exit_time", "entry_bar_time", "exit_bar_time"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)


def process_one_result(short_row: ShortlistRow, price_context_df: pd.DataFrame) -> dict:
    result_path = short_row.result_path
    trades_path = result_path / "trades.csv"
    trades_enriched_path = result_path / "trades_enriched.csv"
    regime_summary_path = result_path / "regime_summary.csv"
    report_path = result_path / "enrichment_report.json"

    trades_df = load_trades_csv(trades_path)
    enriched_df, match_report = enrich_trades(trades_df, price_context_df)

    summaries = []
    summaries.append(build_bucket_summary(enriched_df, "session_bucket", "session_bucket"))
    summaries.append(build_bucket_summary(enriched_df, "trend_bucket", "trend_bucket"))
    summaries.append(build_bucket_summary(enriched_df, "volatility_bucket", "volatility_bucket"))
    summaries.append(build_bucket_summary(enriched_df, "price_location_bucket", "price_location_bucket"))
    summaries.append(build_exit_reason_summary(enriched_df))

    regime_summary_df = pd.concat([x for x in summaries if not x.empty], axis=0, ignore_index=True)

    save_dataframe_csv(trades_enriched_path, enriched_df)
    save_dataframe_csv(regime_summary_path, regime_summary_df)

    report = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "strategy_id": short_row.strategy_id,
        "timeframe": short_row.timeframe,
        "validation_mode": short_row.validation_mode,
        "window_name": short_row.window_name,
        "result_path": str(result_path),
        "source_trades_csv": str(trades_path),
        "output_trades_enriched_csv": str(trades_enriched_path),
        "output_regime_summary_csv": str(regime_summary_path),
        "price_context_csv": str(M30_PRICE_CSV),
        **match_report,
    }
    write_json(report_path, report)
    return report


def main() -> None:
    print("=" * 120)
    print(f"[INFO] enrich_top3_m30_regimes.py version={VERSION}")
    print(f"[INFO] shortlist={SHORTLIST_CSV}")
    print(f"[INFO] m30_price={M30_PRICE_CSV}")
    print("=" * 120)

    shortlist_rows = load_shortlist()
    if not shortlist_rows:
        print("[INFO] shortlist ว่าง หรือไม่มี M30")
        return

    price_context_df = load_m30_price_context()

    reports = []
    error_count = 0

    for row in shortlist_rows:
        try:
            report = process_one_result(row, price_context_df)
            reports.append(report)
            print(
                f"[DONE] strategy={row.strategy_id} "
                f"mode={row.validation_mode} "
                f"window={row.window_name} "
                f"trade_rows={report['trade_rows']} "
                f"matched_entry={report['matched_entry_rows']} "
                f"matched_exit={report['matched_exit_rows']}"
            )
        except Exception as exc:
            error_count += 1
            print(
                f"[ERROR] strategy={row.strategy_id} "
                f"mode={row.validation_mode} "
                f"window={row.window_name} "
                f"error={exc}"
            )

    summary_payload = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "shortlist_csv": str(SHORTLIST_CSV),
        "m30_price_csv": str(M30_PRICE_CSV),
        "processed_rows": len(reports),
        "error_count": error_count,
        "reports": reports,
    }
    summary_path = ROOT / "central_backtest_results" / "index" / "top3_m30_regime_enrichment_summary.json"
    write_json(summary_path, summary_payload)

    print("=" * 120)
    print(f"[SUMMARY] processed_rows={len(reports)} error_count={error_count}")
    print(f"[SUMMARY] summary_json={summary_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()
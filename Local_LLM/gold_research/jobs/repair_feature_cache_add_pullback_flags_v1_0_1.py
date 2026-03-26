#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ชื่อโค้ด: repair_feature_cache_add_pullback_flags_v1_0_1.py
ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\repair_feature_cache_add_pullback_flags_v1_0_1.py
คำสั่งรัน:
    python C:\Data\Bot\Local_LLM\gold_research\jobs\repair_feature_cache_add_pullback_flags_v1_0_1.py
เวอร์ชัน: v1.0.1

เป้าหมาย:
- ซ่อม feature cache schema ให้มี pullback_to_ema20_long / pullback_to_ema20_short
- รองรับ parquet ที่เวลาอาจอยู่ใน column หรือ index
- รองรับชื่อคอลัมน์เวลาได้หลายแบบ
- fallback จาก exact merge ไปเป็น merge_asof เมื่อเวลาไม่ตรง 100%
- เขียน summary พร้อม error ราย TF ชัดเจน

นิยาม repair:
- pullback_to_ema20_long  = (low <= ema_20)  and (close >= ema_20)
- pullback_to_ema20_short = (high >= ema_20) and (close <= ema_20)
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


VERSION = "v1.0.1"
RESEARCH_TFS = ["M1", "M2", "M3", "M4", "M5", "M6", "M10", "M15", "M30", "H1", "H4", "D1"]
TIME_CANDIDATES = ["time", "datetime", "timestamp", "date", "dt"]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_columns_lower(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {c: str(c).strip().lower() for c in df.columns}
    return df.rename(columns=rename_map).copy()


def bring_index_to_column_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if any(str(c).lower() in TIME_CANDIDATES for c in out.columns):
        return out

    idx_name = str(out.index.name).lower() if out.index.name is not None else ""
    if idx_name in TIME_CANDIDATES:
        out = out.reset_index()
        return out

    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
        first_col = out.columns[0]
        out = out.rename(columns={first_col: "time"})
        return out

    return out


def normalize_time_column(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    out = bring_index_to_column_if_needed(df)
    out = normalize_columns_lower(out)

    time_col = None
    for candidate in TIME_CANDIDATES:
        if candidate in out.columns:
            time_col = candidate
            break

    if time_col is None:
        raise KeyError(f"{source_name}: missing time-like column and datetime index")

    if time_col != "time":
        out = out.rename(columns={time_col: "time"})

    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    return out


def normalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_time_column(df, "price parquet")
    out = normalize_columns_lower(out)

    missing = [c for c in ["high", "low", "close"] if c not in out.columns]
    if missing:
        raise KeyError(f"price parquet missing columns: {missing}")

    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["high", "low", "close"]).sort_values("time").reset_index(drop=True)

    out = out.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    return out


def normalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_time_column(df, "feature parquet")
    out = normalize_columns_lower(out)

    if "ema_20" not in out.columns:
        raise KeyError("feature parquet missing 'ema_20' column")

    out["ema_20"] = pd.to_numeric(out["ema_20"], errors="coerce")
    out = out.dropna(subset=["ema_20"]).sort_values("time").reset_index(drop=True)

    out = out.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    return out


def try_exact_merge(feat_df: pd.DataFrame, price_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    merged = feat_df.merge(
        price_df[["time", "high", "low", "close"]],
        on="time",
        how="left",
    )

    missing_rows = int(merged[["high", "low", "close"]].isna().any(axis=1).sum())
    if missing_rows == 0:
        return merged, "exact"

    return merged, f"exact_missing_rows={missing_rows}"


def try_asof_merge(feat_df: pd.DataFrame, price_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    left = feat_df.sort_values("time").reset_index(drop=True)
    right = price_df[["time", "high", "low", "close"]].sort_values("time").reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        on="time",
        direction="nearest",
        tolerance=None,
    )

    missing_rows = int(merged[["high", "low", "close"]].isna().any(axis=1).sum())
    return merged, f"asof_missing_rows={missing_rows}"


def build_pullback_flags(price_df: pd.DataFrame, feat_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    merged_exact, exact_note = try_exact_merge(feat_df, price_df)
    exact_missing = int(merged_exact[["high", "low", "close"]].isna().any(axis=1).sum())

    if exact_missing == 0:
        merged = merged_exact
        merge_mode = "exact"
    else:
        merged_asof, asof_note = try_asof_merge(feat_df, price_df)
        asof_missing = int(merged_asof[["high", "low", "close"]].isna().any(axis=1).sum())

        if asof_missing >= exact_missing:
            merged = merged_exact
            merge_mode = exact_note
        else:
            merged = merged_asof
            merge_mode = asof_note

    still_missing = int(merged[["high", "low", "close"]].isna().any(axis=1).sum())
    if still_missing > 0:
        raise ValueError(f"price/feature merge failed, unresolved missing rows={still_missing}, merge_mode={merge_mode}")

    ema20 = pd.to_numeric(merged["ema_20"], errors="coerce")
    high = pd.to_numeric(merged["high"], errors="coerce")
    low = pd.to_numeric(merged["low"], errors="coerce")
    close = pd.to_numeric(merged["close"], errors="coerce")

    merged["pullback_to_ema20_long"] = ((low <= ema20) & (close >= ema20)).fillna(False).astype(bool)
    merged["pullback_to_ema20_short"] = ((high >= ema20) & (close <= ema20)).fillna(False).astype(bool)

    out = merged[["time", "pullback_to_ema20_long", "pullback_to_ema20_short"]].copy()
    return out, merge_mode


def process_timeframe(
    feature_root: Path,
    data_root: Path,
    symbol: str,
    timeframe: str,
    backup_root: Path,
) -> Dict:
    feature_path = feature_root / f"{symbol}_{timeframe}_base_features.parquet"
    price_path = data_root / f"{symbol}_{timeframe}.parquet"

    row = {
        "timeframe": timeframe,
        "feature_path": str(feature_path),
        "price_path": str(price_path),
        "status": "PENDING",
    }

    if not feature_path.exists():
        row["status"] = "MISSING_FEATURE"
        return row

    if not price_path.exists():
        row["status"] = "MISSING_PRICE"
        return row

    raw_feat = pd.read_parquet(feature_path)
    raw_price = pd.read_parquet(price_path)

    feat_df = normalize_feature_columns(raw_feat)
    price_df = normalize_price_columns(raw_price)

    row["feature_columns"] = [str(c) for c in raw_feat.columns]
    row["price_columns"] = [str(c) for c in raw_price.columns]
    row["feature_rows"] = int(len(feat_df))
    row["price_rows"] = int(len(price_df))

    already_has_long = "pullback_to_ema20_long" in feat_df.columns
    already_has_short = "pullback_to_ema20_short" in feat_df.columns

    if already_has_long and already_has_short:
        row["status"] = "ALREADY_OK"
        row["rows"] = int(len(feat_df))
        return row

    backup_path = backup_root / feature_path.name
    ensure_dir(backup_root)
    if not backup_path.exists():
        shutil.copy2(feature_path, backup_path)

    flags_df, merge_mode = build_pullback_flags(price_df, feat_df)

    repaired = feat_df.merge(flags_df, on="time", how="left")
    repaired["pullback_to_ema20_long"] = repaired["pullback_to_ema20_long"].fillna(False).astype(bool)
    repaired["pullback_to_ema20_short"] = repaired["pullback_to_ema20_short"].fillna(False).astype(bool)

    repaired.to_parquet(feature_path, index=False)

    row["status"] = "REPAIRED"
    row["rows"] = int(len(repaired))
    row["merge_mode"] = merge_mode
    row["backup_path"] = str(backup_path)
    row["pullback_long_true"] = int(repaired["pullback_to_ema20_long"].sum())
    row["pullback_short_true"] = int(repaired["pullback_to_ema20_short"].sum())
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair feature cache by adding EMA20 pullback flags")
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path(r"C:\Data\Bot\central_feature_cache"),
        help="Feature cache root",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Data\Bot\central_market_data\parquet"),
        help="Parquet OHLC root",
    )
    parser.add_argument(
        "--symbol",
        default="XAUUSD",
        help="Trading symbol prefix used in parquet names",
    )
    parser.add_argument(
        "--timeframes",
        default="M1,M2,M3,M4,M5,M6,M10,M15,M30,H1,H4,D1",
        help="Comma-separated timeframes",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path(r"C:\Data\Bot\central_feature_cache_backups\repair_pullback_flags_v1_0_1"),
        help="Backup directory before overwrite",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path(r"C:\Data\Bot\central_backtest_results\repair_feature_cache_add_pullback_flags_v1_0_1"),
        help="Report/log output root",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    timeframes = [x.strip() for x in args.timeframes.split(",") if x.strip()]

    ensure_dir(args.report_root)
    ensure_dir(args.backup_root)

    rows: List[Dict] = []
    for tf in timeframes:
        try:
            result = process_timeframe(
                feature_root=args.feature_root,
                data_root=args.data_root,
                symbol=args.symbol,
                timeframe=tf,
                backup_root=args.backup_root,
            )
        except Exception as exc:
            result = {
                "timeframe": tf,
                "status": "ERROR",
                "error": str(exc),
            }
        rows.append(result)
        print(f"[{result['status']}] timeframe={tf}")

    repaired_count = sum(1 for r in rows if r["status"] == "REPAIRED")
    already_ok_count = sum(1 for r in rows if r["status"] == "ALREADY_OK")
    error_count = sum(1 for r in rows if r["status"] == "ERROR")
    missing_count = sum(1 for r in rows if str(r["status"]).startswith("MISSING"))

    summary = {
        "version": VERSION,
        "generated_at_utc": now_utc_iso(),
        "feature_root": str(args.feature_root),
        "data_root": str(args.data_root),
        "symbol": args.symbol,
        "timeframes": timeframes,
        "repaired_count": repaired_count,
        "already_ok_count": already_ok_count,
        "missing_count": missing_count,
        "error_count": error_count,
        "rows": rows,
    }

    save_json(args.report_root / "summary.json", summary)

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] feature_root={args.feature_root}")
    print(f"[DONE] backup_root={args.backup_root}")
    print(f"[DONE] report_summary={args.report_root / 'summary.json'}")
    print(f"[DONE] repaired_count={repaired_count}")
    print(f"[DONE] already_ok_count={already_ok_count}")
    print(f"[DONE] missing_count={missing_count}")
    print(f"[DONE] error_count={error_count}")
    print("=" * 120)


if __name__ == "__main__":
    main()
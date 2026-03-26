#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ชื่อโค้ด: repair_feature_cache_add_pullback_flags_v1_0_0.py
ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\repair_feature_cache_add_pullback_flags_v1_0_0.py
คำสั่งรัน:
    python C:\Data\Bot\Local_LLM\gold_research\jobs\repair_feature_cache_add_pullback_flags_v1_0_0.py
เวอร์ชัน: v1.0.0

เป้าหมาย:
- ซ่อม feature cache schema ให้มี pullback_to_ema20_long / pullback_to_ema20_short
- ใช้ข้อมูลจาก parquet ราคา + feature parquet เดิม
- เขียนทับไฟล์เดิมแบบปลอดภัย โดย backup ก่อน
- ใช้ได้กับทุก research TF ที่ล็อกไว้

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
from typing import Dict, List

import pandas as pd


VERSION = "v1.0.0"
RESEARCH_TFS = ["M1", "M2", "M3", "M4", "M5", "M6", "M10", "M15", "M30", "H1", "H4", "D1"]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {str(c).lower(): c for c in df.columns}
    rename_map = {}
    for required in ["time", "open", "high", "low", "close"]:
        if required in cols_lower:
            rename_map[cols_lower[required]] = required
    out = df.rename(columns=rename_map).copy()

    missing = [c for c in ["time", "high", "low", "close"] if c not in out.columns]
    if missing:
        raise KeyError(f"price parquet missing columns: {missing}")

    out["time"] = pd.to_datetime(out["time"], utc=False, errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return out


def normalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        lower_map = {str(c).lower(): c for c in df.columns}
        if "time" in lower_map:
            df = df.rename(columns={lower_map["time"]: "time"})
        else:
            raise KeyError("feature parquet missing 'time' column")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=False, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    if "ema_20" not in df.columns:
        raise KeyError("feature parquet missing 'ema_20' column")

    return df


def build_pullback_flags(price_df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    merged = feat_df.merge(
        price_df[["time", "high", "low", "close"]],
        on="time",
        how="left",
        validate="one_to_one",
    )

    if merged[["high", "low", "close"]].isna().any().any():
        missing_rows = int(merged[["high", "low", "close"]].isna().any(axis=1).sum())
        raise ValueError(f"price/feature time alignment failed, missing merged rows={missing_rows}")

    ema20 = pd.to_numeric(merged["ema_20"], errors="coerce")
    high = pd.to_numeric(merged["high"], errors="coerce")
    low = pd.to_numeric(merged["low"], errors="coerce")
    close = pd.to_numeric(merged["close"], errors="coerce")

    merged["pullback_to_ema20_long"] = ((low <= ema20) & (close >= ema20)).fillna(False).astype(bool)
    merged["pullback_to_ema20_short"] = ((high >= ema20) & (close <= ema20)).fillna(False).astype(bool)

    return merged[["time", "pullback_to_ema20_long", "pullback_to_ema20_short"]]


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

    feat_df = normalize_feature_columns(pd.read_parquet(feature_path))
    price_df = normalize_price_columns(pd.read_parquet(price_path))

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

    flags_df = build_pullback_flags(price_df, feat_df)

    repaired = feat_df.merge(flags_df, on="time", how="left", validate="one_to_one")
    repaired["pullback_to_ema20_long"] = repaired["pullback_to_ema20_long"].fillna(False).astype(bool)
    repaired["pullback_to_ema20_short"] = repaired["pullback_to_ema20_short"].fillna(False).astype(bool)

    repaired.to_parquet(feature_path, index=False)

    row["status"] = "REPAIRED"
    row["rows"] = int(len(repaired))
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
        default=Path(r"C:\Data\Bot\central_feature_cache_backups\repair_pullback_flags_v1_0_0"),
        help="Backup directory before overwrite",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path(r"C:\Data\Bot\central_backtest_results\repair_feature_cache_add_pullback_flags_v1_0_0"),
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
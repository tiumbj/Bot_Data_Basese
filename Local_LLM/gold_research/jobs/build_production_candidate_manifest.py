# ============================================================
# ชื่อโค้ด: build_production_candidate_manifest.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\build_production_candidate_manifest.py
# คำสั่งรัน: python C:\Data\Bot\Local_LLM\gold_research\jobs\build_production_candidate_manifest.py
# เวอร์ชัน: v1.0.0
# ============================================================

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

VERSION = "v1.0.0"

ROOT = Path(r"C:\Data\Bot")
INDEX_DIR = ROOT / "central_backtest_results" / "index"
TEST_C_INDEX_CSV = INDEX_DIR / "test_c_top3_m30_index.csv"

OUTPUT_JSON = INDEX_DIR / "production_candidate_manifest.json"
OUTPUT_CSV = INDEX_DIR / "production_candidate_ranking.csv"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    print("=" * 120)
    print(f"[INFO] build_production_candidate_manifest.py version={VERSION}")
    print(f"[INFO] input={TEST_C_INDEX_CSV}")
    print("=" * 120)

    if not TEST_C_INDEX_CSV.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ input | file={TEST_C_INDEX_CSV}")

    df = pd.read_csv(TEST_C_INDEX_CSV)
    required = [
        "strategy_id",
        "validation_mode",
        "window_name",
        "net_profit",
        "profit_factor",
        "win_rate",
        "total_trades",
        "result_path",
        "timeframe",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"คอลัมน์ไม่ครบ | missing={missing}")

    work = df.copy()
    work["net_profit"] = pd.to_numeric(work["net_profit"], errors="coerce").fillna(0.0)
    work["profit_factor"] = pd.to_numeric(work["profit_factor"], errors="coerce").fillna(0.0)
    work["win_rate"] = pd.to_numeric(work["win_rate"], errors="coerce").fillna(0.0)
    work["total_trades"] = pd.to_numeric(work["total_trades"], errors="coerce").fillna(0).astype(int)

    ranking_rows: List[Dict] = []

    for strategy_id, g in work.groupby("strategy_id"):
        outsample = g.loc[g["validation_mode"] == "outsample"].copy()
        walkforward = g.loc[g["validation_mode"] == "walkforward"].copy()
        insample = g.loc[g["validation_mode"] == "insample"].copy()

        outsample_net = float(outsample["net_profit"].sum()) if not outsample.empty else 0.0
        outsample_pf = float(outsample["profit_factor"].mean()) if not outsample.empty else 0.0

        walkforward_positive = int((walkforward["net_profit"] > 0).sum()) if not walkforward.empty else 0
        walkforward_total = int(len(walkforward))
        walkforward_pass_rate = float(walkforward_positive / walkforward_total) if walkforward_total > 0 else 0.0

        ranking_rows.append(
            {
                "strategy_id": strategy_id,
                "timeframe": str(g["timeframe"].iloc[0]),
                "rows": int(len(g)),
                "net_profit_sum": float(g["net_profit"].sum()),
                "avg_profit_factor": float(g["profit_factor"].mean()),
                "avg_win_rate": float(g["win_rate"].mean()),
                "total_trades_sum": int(g["total_trades"].sum()),
                "outsample_net_profit": outsample_net,
                "outsample_profit_factor": outsample_pf,
                "walkforward_positive_windows": walkforward_positive,
                "walkforward_total_windows": walkforward_total,
                "walkforward_pass_rate": walkforward_pass_rate,
                "score": (
                    float(g["net_profit"].sum())
                    + (float(g["profit_factor"].mean()) * 100.0)
                    + (outsample_net * 0.50)
                    + (walkforward_pass_rate * 250.0)
                ),
            }
        )

    ranking_df = pd.DataFrame(ranking_rows).sort_values(
        by=[
            "score",
            "net_profit_sum",
            "outsample_net_profit",
            "avg_profit_factor",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    ensure_parent(OUTPUT_CSV)
    ranking_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    winner = ranking_df.iloc[0].to_dict()
    winner_id = str(winner["strategy_id"])
    winner_rows = work.loc[work["strategy_id"] == winner_id].sort_values(
        by=["validation_mode", "window_name"]
    ).reset_index(drop=True)

    result_paths = winner_rows["result_path"].astype(str).tolist()
    sample_result_path = result_paths[0] if result_paths else ""

    payload = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "source_test_c_index_csv": str(TEST_C_INDEX_CSV),
        "selection_method": {
            "primary_goal": "เลือก production candidate หลักจาก Test C top3",
            "ranking_fields": [
                "score",
                "net_profit_sum",
                "outsample_net_profit",
                "avg_profit_factor",
            ],
            "score_formula": "net_profit_sum + avg_profit_factor*100 + outsample_net_profit*0.50 + walkforward_pass_rate*250",
        },
        "winner": {
            "strategy_id": winner_id,
            "timeframe": str(winner["timeframe"]),
            "experiment_name": "C_regime_filtered_top3_m30",
            "regime_filters": {
                "allow_trend_buckets": ["BULL_TREND"],
                "allow_volatility_buckets": ["HIGH_VOL"],
                "allow_price_location_buckets": ["ABOVE_EMA_STACK"],
                "block_trend_buckets": ["BEAR_TREND"],
                "block_price_location_buckets": ["BELOW_EMA_STACK"],
                "session_filter": [],
            },
            "aggregate_metrics": {
                "rows": int(winner["rows"]),
                "net_profit_sum": float(winner["net_profit_sum"]),
                "avg_profit_factor": float(winner["avg_profit_factor"]),
                "avg_win_rate": float(winner["avg_win_rate"]),
                "total_trades_sum": int(winner["total_trades_sum"]),
                "outsample_net_profit": float(winner["outsample_net_profit"]),
                "outsample_profit_factor": float(winner["outsample_profit_factor"]),
                "walkforward_positive_windows": int(winner["walkforward_positive_windows"]),
                "walkforward_total_windows": int(winner["walkforward_total_windows"]),
                "walkforward_pass_rate": float(winner["walkforward_pass_rate"]),
                "score": float(winner["score"]),
            },
            "sample_result_path": sample_result_path,
        },
        "backup_candidates": [
            {
                "strategy_id": str(row["strategy_id"]),
                "timeframe": str(row["timeframe"]),
                "net_profit_sum": float(row["net_profit_sum"]),
                "avg_profit_factor": float(row["avg_profit_factor"]),
                "outsample_net_profit": float(row["outsample_net_profit"]),
                "walkforward_pass_rate": float(row["walkforward_pass_rate"]),
                "score": float(row["score"]),
            }
            for _, row in ranking_df.iloc[1:].iterrows()
        ],
        "winner_windows": winner_rows.to_dict(orient="records"),
    }

    save_json(OUTPUT_JSON, payload)

    print(f"[DONE] ranking_csv={OUTPUT_CSV}")
    print(f"[DONE] manifest_json={OUTPUT_JSON}")
    print(f"[WINNER] strategy_id={winner_id}")
    print("=" * 120)


if __name__ == "__main__":
    main()
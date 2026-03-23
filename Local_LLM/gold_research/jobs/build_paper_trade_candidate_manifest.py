# ============================================================
# ชื่อโค้ด: build_paper_trade_candidate_manifest.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\build_paper_trade_candidate_manifest.py
# คำสั่งรัน: python C:\Data\Bot\Local_LLM\gold_research\jobs\build_paper_trade_candidate_manifest.py
# เวอร์ชัน: v1.0.1
# ============================================================

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


VERSION = "v1.0.1"

ROOT = Path(r"C:\Data\Bot")
INDEX_DIR = ROOT / "central_backtest_results" / "index"
PRODUCTION_CANDIDATE_MANIFEST = INDEX_DIR / "production_candidate_manifest.json"
OUTPUT_JSON = INDEX_DIR / "paper_trade_candidate_manifest.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 120)
    print(f"[INFO] build_paper_trade_candidate_manifest.py version={VERSION}")
    print(f"[INFO] input={PRODUCTION_CANDIDATE_MANIFEST}")
    print("=" * 120)

    if not PRODUCTION_CANDIDATE_MANIFEST.exists():
        raise FileNotFoundError(
            f"ไม่พบ production candidate manifest | file={PRODUCTION_CANDIDATE_MANIFEST}"
        )

    source = json.loads(PRODUCTION_CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    winner = source["winner"]

    strategy_id = winner["strategy_id"]
    timeframe = winner["timeframe"]
    regime_filters = winner["regime_filters"]
    aggregate_metrics = winner["aggregate_metrics"]

    payload = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "source_manifest": str(PRODUCTION_CANDIDATE_MANIFEST),
        "candidate_type": "paper_trade",
        "status": "ACTIVE_PAPER_TRADE_CANDIDATE",
        "priority_rank": 1,
        "strategy": {
            "strategy_id": strategy_id,
            "timeframe": timeframe,
            "side_policy": "LONG_ONLY",
            "execution_mode": "PAPER_ONLY",
        },
        "regime_filters": {
            "allow_trend_buckets": regime_filters.get("allow_trend_buckets", ["BULL_TREND"]),
            "allow_volatility_buckets": regime_filters.get("allow_volatility_buckets", ["HIGH_VOL"]),
            "allow_price_location_buckets": regime_filters.get("allow_price_location_buckets", ["ABOVE_EMA_STACK"]),
            "block_trend_buckets": regime_filters.get("block_trend_buckets", ["BEAR_TREND"]),
            "block_price_location_buckets": regime_filters.get("block_price_location_buckets", ["BELOW_EMA_STACK"]),
            "session_filter": regime_filters.get("session_filter", []),
        },
        "validation_snapshot": {
            "net_profit_sum": aggregate_metrics["net_profit_sum"],
            "avg_profit_factor": aggregate_metrics["avg_profit_factor"],
            "avg_win_rate": aggregate_metrics["avg_win_rate"],
            "outsample_net_profit": aggregate_metrics["outsample_net_profit"],
            "outsample_profit_factor": aggregate_metrics["outsample_profit_factor"],
            "walkforward_positive_windows": aggregate_metrics["walkforward_positive_windows"],
            "walkforward_total_windows": aggregate_metrics["walkforward_total_windows"],
            "walkforward_pass_rate": aggregate_metrics["walkforward_pass_rate"],
            "total_trades_sum": aggregate_metrics["total_trades_sum"],
        },
        "paper_trade_requirements": {
            "min_live_observation_days": 10,
            "required_checks": [
                "signal_count_per_week",
                "stop_loss_clustering",
                "win_loss_distribution",
                "pnl_concentration_by_day",
                "regime_match_rate",
            ],
            "promotion_gate": {
                "no_rule_break": True,
                "regime_match_required": True,
                "manual_review_required": True,
            },
        },
        "backup_candidates": source.get("backup_candidates", []),
        "notes": [
            "ตัวนี้ผ่าน Test C และเป็น candidate หลักของรอบวิจัย",
            "ยังไม่ใช่ production auto-trade เต็มรูปแบบ",
            "ให้ใช้ LONG_ONLY เท่านั้น",
            "ห้ามเปิด short กลับเข้ามาในรอบ paper trade นี้",
        ],
    }

    save_json(OUTPUT_JSON, payload)

    print(f"[DONE] output={OUTPUT_JSON}")
    print(f"[DONE] strategy_id={strategy_id}")
    print(f"[DONE] timeframe={timeframe}")
    print("=" * 120)


if __name__ == "__main__":
    main()
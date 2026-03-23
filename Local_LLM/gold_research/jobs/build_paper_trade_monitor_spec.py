# ============================================================
# ชื่อโค้ด: build_paper_trade_monitor_spec.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\build_paper_trade_monitor_spec.py
# คำสั่งรัน: python C:\Data\Bot\Local_LLM\gold_research\jobs\build_paper_trade_monitor_spec.py
# เวอร์ชัน: v1.0.1
# ============================================================

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "v1.0.1"

ROOT = Path(r"C:\Data\Bot")
INDEX_DIR = ROOT / "central_backtest_results" / "index"

INPUT_MANIFEST = INDEX_DIR / "paper_trade_candidate_manifest.json"
OUTPUT_SPEC = INDEX_DIR / "paper_trade_monitor_spec.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ input manifest | file={path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 120)
    print(f"[INFO] build_paper_trade_monitor_spec.py version={VERSION}")
    print(f"[INFO] input={INPUT_MANIFEST}")
    print("=" * 120)

    manifest = load_json(INPUT_MANIFEST)

    strategy = manifest["strategy"]
    regime_filters = manifest["regime_filters"]
    validation_snapshot = manifest["validation_snapshot"]
    paper_trade_requirements = manifest["paper_trade_requirements"]
    backup_candidates = manifest.get("backup_candidates", [])

    required_checks = paper_trade_requirements.get("required_checks", [])
    min_live_observation_days = int(
        paper_trade_requirements.get("min_live_observation_days", 10)
    )
    promotion_gate = paper_trade_requirements.get("promotion_gate", {})

    spec = {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "source_manifest": str(INPUT_MANIFEST),
        "monitor_phase": "PAPER_TRADE_VALIDATION",
        "status": "ACTIVE",
        "candidate_identity": {
            "strategy_id": strategy["strategy_id"],
            "timeframe": strategy["timeframe"],
            "side_policy": strategy["side_policy"],
            "execution_mode": strategy["execution_mode"],
            "priority_rank": manifest.get("priority_rank", 1),
        },
        "allowed_regime_profile": {
            "allow_trend_buckets": regime_filters.get("allow_trend_buckets", []),
            "allow_volatility_buckets": regime_filters.get("allow_volatility_buckets", []),
            "allow_price_location_buckets": regime_filters.get("allow_price_location_buckets", []),
            "block_trend_buckets": regime_filters.get("block_trend_buckets", []),
            "block_price_location_buckets": regime_filters.get("block_price_location_buckets", []),
            "session_filter": regime_filters.get("session_filter", []),
        },
        "baseline_reference": {
            "net_profit_sum": validation_snapshot.get("net_profit_sum"),
            "avg_profit_factor": validation_snapshot.get("avg_profit_factor"),
            "avg_win_rate": validation_snapshot.get("avg_win_rate"),
            "outsample_net_profit": validation_snapshot.get("outsample_net_profit"),
            "outsample_profit_factor": validation_snapshot.get("outsample_profit_factor"),
            "walkforward_positive_windows": validation_snapshot.get("walkforward_positive_windows"),
            "walkforward_total_windows": validation_snapshot.get("walkforward_total_windows"),
            "walkforward_pass_rate": validation_snapshot.get("walkforward_pass_rate"),
            "total_trades_sum": validation_snapshot.get("total_trades_sum"),
        },
        "observation_window": {
            "min_live_observation_days": min_live_observation_days,
            "min_completed_weeks": 2,
            "review_frequency": "DAILY",
            "summary_cutoff_time_utc": "23:59:59Z",
        },
        "required_checks": required_checks,
        "check_specs": {
            "signal_count_per_week": {
                "enabled": "signal_count_per_week" in required_checks,
                "description": "นับจำนวนสัญญาณที่ผ่าน regime filter และผ่านเข้า paper flow ต่อสัปดาห์",
                "calculation": {
                    "group_by": ["iso_year", "iso_week"],
                    "count_field": "signal_id",
                    "event_type": "approved_paper_signal",
                },
                "alert_rules": {
                    "zero_signal_week": "WARN",
                    "sudden_drop_vs_previous_week_pct": 60.0,
                    "sudden_spike_vs_previous_week_pct": 150.0,
                },
            },
            "stop_loss_clustering": {
                "enabled": "stop_loss_clustering" in required_checks,
                "description": "ตรวจว่า stop loss กระจุกตัวผิดปกติในวันเดียว ชั่วโมงเดียว หรือ regime เดียวหรือไม่",
                "calculation": {
                    "event_filter": "exit_reason == 'stop_loss'",
                    "group_by": [
                        "trade_date",
                        "hour_bucket",
                        "trend_bucket",
                        "volatility_bucket",
                        "price_location_bucket",
                    ],
                },
                "alert_rules": {
                    "max_stop_loss_share_same_day_pct": 40.0,
                    "max_stop_loss_share_same_hour_pct": 25.0,
                    "max_stop_loss_share_same_regime_pct": 60.0,
                },
            },
            "win_loss_distribution": {
                "enabled": "win_loss_distribution" in required_checks,
                "description": "ตรวจสัดส่วน win/loss และ payoff ว่าไม่เสื่อมจนผิดจาก profile งานวิจัยมากเกินไป",
                "calculation": {
                    "group_by": ["day", "week"],
                    "win_condition": "pnl > 0",
                    "loss_condition": "pnl < 0",
                    "metrics": ["win_rate", "avg_win", "avg_loss", "payoff_ratio"],
                },
                "alert_rules": {
                    "min_weekly_win_rate_pct": 30.0,
                    "min_payoff_ratio": 1.20,
                    "max_consecutive_losses": 8,
                },
            },
            "pnl_concentration_by_day": {
                "enabled": "pnl_concentration_by_day" in required_checks,
                "description": "ตรวจว่ากำไรรวมไม่ได้พึ่งวันเดียวมากเกินไป",
                "calculation": {
                    "group_by": ["trade_date"],
                    "metrics": ["daily_pnl_sum", "daily_trade_count", "positive_day_share"],
                },
                "alert_rules": {
                    "max_single_day_pnl_share_pct": 50.0,
                    "max_top_3_days_pnl_share_pct": 80.0,
                    "min_positive_day_share_pct": 35.0,
                },
            },
            "regime_match_rate": {
                "enabled": "regime_match_rate" in required_checks,
                "description": "ตรวจว่าสัญญาณ live เกิดใน regime ที่ได้รับอนุญาตตาม candidate manifest จริง",
                "calculation": {
                    "event_type": "approved_paper_signal",
                    "metrics": ["allowed_regime_hits", "blocked_regime_hits", "regime_match_rate"],
                },
                "alert_rules": {
                    "min_regime_match_rate_pct": 95.0,
                    "max_blocked_regime_hits": 0,
                },
            },
        },
        "promotion_gate": {
            "no_rule_break": bool(promotion_gate.get("no_rule_break", True)),
            "regime_match_required": bool(promotion_gate.get("regime_match_required", True)),
            "manual_review_required": bool(promotion_gate.get("manual_review_required", True)),
            "minimum_expectations": {
                "min_live_observation_days": min_live_observation_days,
                "max_blocked_regime_hits": 0,
                "min_regime_match_rate_pct": 95.0,
                "min_positive_weeks": 1,
                "max_zero_signal_weeks": 1,
            },
        },
        "runtime_data_contract": {
            "required_signal_fields": [
                "signal_id",
                "ts_utc",
                "strategy_id",
                "timeframe",
                "side",
                "trend_bucket",
                "volatility_bucket",
                "price_location_bucket",
                "approved",
            ],
            "required_trade_fields": [
                "trade_id",
                "signal_id",
                "entry_time_utc",
                "exit_time_utc",
                "entry_price",
                "exit_price",
                "pnl",
                "exit_reason",
                "trade_date",
                "hour_bucket",
                "trend_bucket",
                "volatility_bucket",
                "price_location_bucket",
            ],
        },
        "backup_candidates": backup_candidates,
        "notes": [
            "ใช้ spec นี้เป็น source of truth ของช่วง paper-trade validation",
            "ห้าม promote เป็น production auto-trade ถ้ายังไม่ผ่าน promotion_gate",
            "candidate นี้ยังต้องถูกประเมินแบบ manual review หลังครบ observation window",
        ],
    }

    save_json(OUTPUT_SPEC, spec)

    print(f"[DONE] output={OUTPUT_SPEC}")
    print(f"[DONE] strategy_id={strategy['strategy_id']}")
    print(f"[DONE] timeframe={strategy['timeframe']}")
    print("=" * 120)


if __name__ == "__main__":
    main()
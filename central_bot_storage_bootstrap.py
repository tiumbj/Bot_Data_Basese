# ============================================================
# ชื่อโค้ด: central_bot_storage_bootstrap.py
# ที่อยู่ไฟล์: C:\Data\Bot\central_bot_storage_bootstrap.py
# คำสั่งรัน: python C:\Data\Bot\central_bot_storage_bootstrap.py
# เวอร์ชัน: v1.0.0
# ============================================================

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

VERSION = "v1.0.0"
ROOT = Path(r"C:\Data\Bot")

WEEKLY_TIMEFRAMES = ["M1", "M5", "M10", "M15", "H1", "H4"]


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_structure() -> None:
    # 1) Central market data
    market_data_root = ROOT / "central_market_data"
    tf_root = market_data_root / "tf"
    manifests_root = market_data_root / "manifests"
    logs_root = market_data_root / "logs"

    tf_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    market_readme = """CENTRAL MARKET DATA

หน้าที่:
- เป็นข้อมูลตลาดกลางของทุก bot
- ใช้ canonical symbol = XAUUSD
- ทุก timeframe ต้องอ่านจากโฟลเดอร์ tf

ตำแหน่งหลัก:
- tf/XAUUSD_M1.csv
- tf/XAUUSD_M5.csv
- tf/XAUUSD_M10.csv
- tf/XAUUSD_M15.csv
- tf/XAUUSD_H1.csv
- tf/XAUUSD_H4.csv
- tf/manifest.csv
- tf/manifest.json
"""
    write_text(market_data_root / "README.txt", market_readme)

    write_json(
        tf_root / "manifest.json",
        {
            "version": VERSION,
            "canonical_symbol": "XAUUSD",
            "updated_at_utc": None,
            "dataset_root": str(tf_root),
            "items": [],
        },
    )
    write_text(
        tf_root / "manifest.csv",
        "symbol,timeframe,rows,first_datetime,last_datetime,output_file\n",
    )

    # 2) Central backtest results
    backtest_root = ROOT / "central_backtest_results"
    weekly_root = backtest_root / "weekly"
    latest_root = backtest_root / "latest"
    archive_root = backtest_root / "archive"
    index_root = backtest_root / "index"

    weekly_root.mkdir(parents=True, exist_ok=True)
    latest_root.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)
    index_root.mkdir(parents=True, exist_ok=True)

    for tf in WEEKLY_TIMEFRAMES:
        (weekly_root / tf / "insample").mkdir(parents=True, exist_ok=True)
        (weekly_root / tf / "outsample").mkdir(parents=True, exist_ok=True)
        (weekly_root / tf / "walkforward").mkdir(parents=True, exist_ok=True)
        (latest_root / tf).mkdir(parents=True, exist_ok=True)

    backtest_readme = """CENTRAL BACKTEST RESULTS

หน้าที่:
- เก็บผล backtest กลางที่ bot อื่นใช้ได้
- weekly = ผลรอบประจำสัปดาห์
- latest = logical latest result ของแต่ละ timeframe/strategy
- archive = เก็บ snapshot เก่า
- index = ดัชนีรวมผลเพื่อให้ bot อื่นอ่านได้ง่าย

ผลลัพธ์มาตรฐานต่อ strategy/timeframe:
- metrics.json
- trades.csv
- equity_curve.csv
- walkforward_summary.json
- validation_windows.json
"""
    write_text(backtest_root / "README.txt", backtest_readme)

    write_json(
        index_root / "backtest_index.json",
        {
            "version": VERSION,
            "updated_at_utc": None,
            "timeframes": WEEKLY_TIMEFRAMES,
            "strategies": [],
            "latest_results": [],
        },
    )
    write_text(
        index_root / "backtest_index.csv",
        "strategy_id,version,timeframe,validation_mode,net_profit,profit_factor,expectancy,max_drawdown,total_trades,win_rate,avg_win,avg_loss,result_path,approved_for_registry\n",
    )

    # 3) Central strategy registry
    strategy_root = ROOT / "central_strategy_registry"
    candidates_root = strategy_root / "candidates"
    approved_root = strategy_root / "approved"
    deprecated_root = strategy_root / "deprecated"
    manifests_strategy_root = strategy_root / "manifests"

    candidates_root.mkdir(parents=True, exist_ok=True)
    approved_root.mkdir(parents=True, exist_ok=True)
    deprecated_root.mkdir(parents=True, exist_ok=True)
    manifests_strategy_root.mkdir(parents=True, exist_ok=True)

    strategy_readme = """CENTRAL STRATEGY REGISTRY

หน้าที่:
- candidates = strategy ที่ยังอยู่ระหว่างประเมิน
- approved = strategy ที่ผ่านเกณฑ์และ bot อื่นโหลดได้
- deprecated = strategy ที่เลิกใช้

ไฟล์มาตรฐานต่อ strategy package:
- strategy_spec.json
- approved_parameters.json
- validation_report.json
- deployment_contract.json
"""
    write_text(strategy_root / "README.txt", strategy_readme)

    write_json(
        manifests_strategy_root / "registry_index.json",
        {
            "version": VERSION,
            "updated_at_utc": None,
            "approved": [],
            "candidates": [],
            "deprecated": [],
        },
    )
    write_text(
        manifests_strategy_root / "registry_index.csv",
        "strategy_id,version,status,timeframes,concept,source_result_path,package_path,last_validated_utc\n",
    )

    # 4) Weekly backtest job config for Local_LLM bot
    local_llm_root = ROOT / "Local_LLM"
    gold_research_root = local_llm_root / "gold_research"
    jobs_root = gold_research_root / "jobs"
    configs_root = gold_research_root / "configs"
    schedules_root = gold_research_root / "schedules"

    jobs_root.mkdir(parents=True, exist_ok=True)
    configs_root.mkdir(parents=True, exist_ok=True)
    schedules_root.mkdir(parents=True, exist_ok=True)

    write_json(
        configs_root / "weekly_backtest_scope.json",
        {
            "version": VERSION,
            "canonical_symbol": "XAUUSD",
            "market_data_root": str(tf_root),
            "backtest_results_root": str(backtest_root),
            "strategy_registry_root": str(strategy_root),
            "schedule": "weekly",
            "locked_timeframes": WEEKLY_TIMEFRAMES,
            "validation_modes": ["insample", "outsample", "walkforward"],
            "evaluation_metrics": [
                "net_profit",
                "profit_factor",
                "expectancy",
                "max_drawdown",
                "total_trades",
                "win_rate",
                "avg_win",
                "avg_loss",
            ],
        },
    )

    weekly_schedule_note = """WEEKLY BACKTEST SCHEDULE

รอบที่ต้องรันทุกสัปดาห์:
- M1
- M5
- M10
- M15
- H1
- H4

validation:
- insample
- outsample
- walkforward

ผลลัพธ์ต้องถูกส่งไป:
C:\\Data\\Bot\\central_backtest_results
"""
    write_text(schedules_root / "weekly_backtest_schedule.txt", weekly_schedule_note)

    # 5) Placeholder templates for strategy packages/results
    template_root = ROOT / "central_templates"
    (template_root / "strategy_package").mkdir(parents=True, exist_ok=True)
    (template_root / "backtest_result").mkdir(parents=True, exist_ok=True)

    write_json(
        template_root / "strategy_package" / "strategy_spec.template.json",
        {
            "strategy_id": "",
            "version": "",
            "concept": "",
            "entry_rules": [],
            "exit_rules": [],
            "filters": [],
            "timeframes": WEEKLY_TIMEFRAMES,
            "status": "candidate",
        },
    )
    write_json(
        template_root / "strategy_package" / "approved_parameters.template.json",
        {
            "strategy_id": "",
            "version": "",
            "parameters": {},
            "timeframes": WEEKLY_TIMEFRAMES,
        },
    )
    write_json(
        template_root / "backtest_result" / "metrics.template.json",
        {
            "strategy_id": "",
            "version": "",
            "timeframe": "",
            "validation_mode": "",
            "net_profit": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        },
    )

    # 6) Bootstrap report
    report = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root": str(ROOT),
        "weekly_timeframes": WEEKLY_TIMEFRAMES,
        "paths": {
            "central_market_data": str(market_data_root),
            "central_market_data_tf": str(tf_root),
            "central_backtest_results": str(backtest_root),
            "central_strategy_registry": str(strategy_root),
            "local_llm_gold_research": str(gold_research_root),
            "weekly_backtest_scope_config": str(configs_root / "weekly_backtest_scope.json"),
        },
    }
    write_json(ROOT / "central_bootstrap_report.json", report)


def main() -> None:
    build_structure()
    print(f"[DONE] central bot storage bootstrap complete | version={VERSION}")
    print(f"[ROOT] {ROOT}")
    print(f"[MARKET_DATA] {ROOT / 'central_market_data'}")
    print(f"[BACKTEST_RESULTS] {ROOT / 'central_backtest_results'}")
    print(f"[STRATEGY_REGISTRY] {ROOT / 'central_strategy_registry'}")
    print(f"[REPORT] {ROOT / 'central_bootstrap_report.json'}")


if __name__ == "__main__":
    main()
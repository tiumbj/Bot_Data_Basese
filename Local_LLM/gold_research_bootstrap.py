# ============================================================
# ชื่อโค้ด: Locked Research Project Bootstrap
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research_bootstrap.py
# คำสั่งรัน: python C:\Data\Bot\Local_LLM\gold_research_bootstrap.py
# เวอร์ชัน: v1.0.0
# ============================================================

"""
gold_research_bootstrap.py
Version: v1.0.0

Purpose:
- สร้างโครงโปรเจกต์วิจัย/backtest กลาง
- สร้าง locked config ตามกติกาที่ตกลงกันไว้
- สร้าง docs สรุปกฎของโปรเจกต์
- คัดลอก canonical dataset เข้า project (ถ้ามี)

Locked Rules Included:
1) Base Core:
   - Market Structure
   - BOS
   - CHOCH
   - Swing High / Swing Low
   - Pullback Zone

2) Filter Layer:
   - ATR
   - ADX
   - EMA20 / EMA50

3) EMA Research Rule:
   - Fast EMA = 1..50
   - Slow EMA = 20..100
   - fast < slow
   - EMA ใช้เป็น filter only

4) Dataset Rule:
   - canonical symbol = XAUUSD
   - dataset ทุก TF มาจาก M1 source เดียวกัน
   - backtest ทุกตัวต้องใช้ข้อมูลจาก dataset/tf/ เท่านั้น

5) Validation Rule:
   - in-sample
   - out-of-sample
   - walk-forward

6) Evaluation Rule:
   - net profit
   - profit factor
   - expectancy
   - max drawdown
   - total trades
   - win rate
   - avg win
   - avg loss
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List


VERSION = "v1.0.0"
PROJECT_ROOT = Path(r"C:\Data\Bot\Local_LLM\gold_research")
SOURCE_DATASET_ROOT = Path(r"C:\Data\data_base\canonical_dataset\dataset")


@dataclass
class FolderSpec:
    relative_path: str
    description: str


def build_folder_specs() -> List[FolderSpec]:
    return [
        FolderSpec("config", "Locked project configuration"),
        FolderSpec("dataset", "Canonical dataset root"),
        FolderSpec("dataset/raw", "Clean raw dataset"),
        FolderSpec("dataset/tf", "Resampled timeframe dataset"),
        FolderSpec("dataset/manifest", "Dataset manifest files"),
        FolderSpec("strategies", "Strategy root"),
        FolderSpec("strategies/price_action", "Price action strategy family"),
        FolderSpec("strategies/indicator", "Indicator strategy family"),
        FolderSpec("strategies/hybrid", "Hybrid strategy family"),
        FolderSpec("backtest", "Backtest root"),
        FolderSpec("backtest/engine", "Shared backtest engine"),
        FolderSpec("backtest/runners", "Backtest runners"),
        FolderSpec("backtest/reports", "Backtest reports"),
        FolderSpec("research", "Research root"),
        FolderSpec("research/ema", "EMA research"),
        FolderSpec("research/atr_adx", "ATR and ADX research"),
        FolderSpec("research/regime", "Regime research"),
        FolderSpec("results", "Backtest result root"),
        FolderSpec("results/in_sample", "In-sample results"),
        FolderSpec("results/out_of_sample", "Out-of-sample results"),
        FolderSpec("results/walk_forward", "Walk-forward results"),
        FolderSpec("docs", "Project documentation"),
        FolderSpec("logs", "Bootstrap and project logs"),
    ]


def ensure_folders(project_root: Path) -> List[Path]:
    created: List[Path] = []
    for spec in build_folder_specs():
        folder_path = project_root / spec.relative_path
        folder_path.mkdir(parents=True, exist_ok=True)
        created.append(folder_path)
    return created


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_locked_model_config() -> Dict[str, Any]:
    return {
        "version": VERSION,
        "project_name": "gold_research",
        "canonical_symbol": "XAUUSD",
        "execution_symbol_mapping_note": "Map broker symbol only at execution layer, not dataset/backtest layer.",
        "locked_base_model": {
            "core_base": [
                "market_structure",
                "BOS",
                "CHOCH",
                "swing_high_low",
                "pullback_zone",
            ],
            "filter_layer": [
                "ATR",
                "ADX",
                "EMA20",
                "EMA50",
            ],
            "rules": {
                "price_action_is_primary": True,
                "indicator_is_filter_only": True,
                "forbid_indicator_crossover_as_main_base": True,
            },
        },
        "ema_research_rule": {
            "enabled": True,
            "purpose": "filter_only",
            "fast_ema_min": 1,
            "fast_ema_max": 50,
            "slow_ema_min": 20,
            "slow_ema_max": 100,
            "constraint": "fast < slow",
            "selection_rules": [
                "must_use_minimum_trades",
                "must_use_parameter_zone_stability",
                "must_pass_out_of_sample",
                "must_pass_walk_forward",
            ],
        },
        "dataset_rule": {
            "canonical_symbol": "XAUUSD",
            "source_of_truth": "dataset/tf",
            "source_generation": "resampled_from_m1_single_source",
            "allowed_timeframes": ["M1", "M2","M3", "M4","M5","M6", "M10", "M15", "M30", "H1", "H4", "D1"],
            "forbid_external_mixed_datasets": True,
        },
        "validation_rule": {
            "required_phases": [
                "in_sample",
                "out_of_sample",
                "walk_forward",
            ],
            "forbid_in_sample_only_decision": True,
        },
        "evaluation_rule": {
            "required_metrics": [
                "net_profit",
                "profit_factor",
                "expectancy",
                "max_drawdown",
                "total_trades",
                "win_rate",
                "avg_win",
                "avg_loss",
            ],
            "forbid_profit_only_selection": True,
        },
    }


def build_timeframe_config() -> Dict[str, Any]:
    return {
        "version": VERSION,
        "canonical_symbol": "XAUUSD",
        "timeframes": {
            "M1": {"pandas_rule": "1min", "family": "execution"},
            "M2": {"pandas_rule": "2min", "family": "execution"},
            "M3": {"pandas_rule": "3min", "family": "execution"},
            "M4": {"pandas_rule": "4min", "family": "execution"},
            "M5": {"pandas_rule": "5min", "family": "execution"},
            "M6": {"pandas_rule": "6min", "family": "execution"},
            "M10": {"pandas_rule": "10min", "family": "execution"},
            "M15": {"pandas_rule": "15min", "family": "setup"},
            "M30": {"pandas_rule": "30min", "family": "setup"},
            "H1": {"pandas_rule": "1h", "family": "context"},
            "H4": {"pandas_rule": "4h", "family": "context"},
            "D1": {"pandas_rule": "1D", "family": "regime"},
        },
        "resample_rule": {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        },
        "timezone_policy": {
            "mode": "keep_source_timestamp",
            "note": "Timezone conversion not applied in dataset bootstrap step.",
        },
    }


def build_metric_config() -> Dict[str, Any]:
    return {
        "version": VERSION,
        "selection_policy": {
            "primary_goal": "robustness_over_single_peak_result",
            "must_check": [
                "net_profit",
                "profit_factor",
                "expectancy",
                "max_drawdown",
                "total_trades",
                "win_rate",
                "avg_win",
                "avg_loss",
            ],
            "must_not_use": [
                "profit_only",
                "win_rate_only",
                "single_parameter_peak_only",
            ],
        },
        "ranking_template": {
            "notes": [
                "Do not select best strategy using only max profit.",
                "Use stability zone and risk-adjusted evaluation.",
            ]
        },
    }


def build_broker_symbol_map() -> Dict[str, Any]:
    return {
        "version": VERSION,
        "canonical_symbol": "XAUUSD",
        "mapping_policy": "apply_only_at_execution_layer",
        "brokers": {
            "XM": "GOLD",
            "Exness": "XAUUSDm",
            "Demo": "XAUUSD",
        },
        "rules": {
            "do_not_rename_dataset_symbol": True,
            "do_not_rename_research_symbol": True,
            "change_symbol_only_when_sending_order": True,
        },
    }


def build_readme(project_root: Path) -> str:
    return f"""# gold_research

Version: {VERSION}

## Purpose
โปรเจกต์วิจัย/backtest กลางสำหรับ XAUUSD/GOLD โดยยึด Locked Model ที่ตกลงกันไว้

## Locked Base
### Core Base
- Market Structure
- BOS
- CHOCH
- Swing High / Swing Low
- Pullback Zone

### Filter Layer
- ATR
- ADX
- EMA20 / EMA50

## Locked EMA Research Rule
- Fast EMA: 1 ถึง 50
- Slow EMA: 20 ถึง 100
- fast < slow
- EMA ใช้เป็น filter only

## Locked Dataset Rule
- canonical symbol = XAUUSD
- dataset ทุก TF มาจาก M1 source เดียวกัน
- backtest ทุกตัวต้องใช้ข้อมูลจาก `dataset/tf/` เท่านั้น

## Locked Validation Rule
- in-sample
- out-of-sample
- walk-forward

## Locked Evaluation Rule
- net profit
- profit factor
- expectancy
- max drawdown
- total trades
- win rate
- avg win
- avg loss

## Folder Root
`{project_root}`
"""


def build_locked_project_rules_md() -> str:
    return """# Locked Project Rules

## 1) Base Model
ใช้ Price Action / Market Structure เป็นแกนหลัก
Indicator ใช้เป็น filter เท่านั้น

### Core Base
- Market Structure
- BOS
- CHOCH
- Swing High / Swing Low
- Pullback Zone

### Filter Layer
- ATR
- ADX
- EMA20 / EMA50

## 2) EMA Research
- Fast EMA = 1..50
- Slow EMA = 20..100
- fast < slow
- ใช้เพื่อหา filter layer ที่ดีที่สุดเท่านั้น
- ห้ามยก EMA เป็น base หลัก

## 3) Dataset
- canonical symbol = XAUUSD
- ทุก timeframe มาจาก M1 source เดียวกัน
- canonical dataset path ต้องอยู่ใน dataset/tf/
- ห้าม strategy ใช้ dataset ภายนอกโดยไม่ผ่าน canonical pipeline

## 4) Validation
ทุก strategy ต้องผ่าน:
1. in-sample
2. out-of-sample
3. walk-forward

## 5) Evaluation
ต้องวัดอย่างน้อย:
- net profit
- profit factor
- expectancy
- max drawdown
- total trades
- win rate
- avg win
- avg loss

ห้ามตัดสินจาก profit อย่างเดียว

## 6) Broker Symbol Mapping
- วิจัย/backtest ใช้ XAUUSD
- execution layer ค่อย map เป็น GOLD / XAUUSDm / XAUUSD ตาม broker
"""

def build_folder_manifest(created_paths: List[Path], project_root: Path) -> List[Dict[str, str]]:
    manifest: List[Dict[str, str]] = []
    spec_map = {spec.relative_path: spec.description for spec in build_folder_specs()}

    for path in created_paths:
        rel = path.relative_to(project_root).as_posix()
        manifest.append(
            {
                "relative_path": rel,
                "absolute_path": str(path),
                "description": spec_map.get(rel, ""),
            }
        )
    return manifest


def copy_tree_if_exists(src: Path, dst: Path) -> Dict[str, Any]:
    result = {
        "source_exists": src.exists(),
        "copied": False,
        "files_copied": 0,
        "source": str(src),
        "destination": str(dst),
    }

    if not src.exists():
        return result

    dst.mkdir(parents=True, exist_ok=True)

    files_copied = 0
    for item in src.rglob("*"):
        if item.is_dir():
            continue
        relative = item.relative_to(src)
        target = dst / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        files_copied += 1

    result["copied"] = True
    result["files_copied"] = files_copied
    return result


def main() -> None:
    print("=" * 100)
    print(f"Locked Research Project Bootstrap | version={VERSION}")
    print("=" * 100)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Source data  : {SOURCE_DATASET_ROOT}")
    print("=" * 100)

    created_paths = ensure_folders(PROJECT_ROOT)

    config_dir = PROJECT_ROOT / "config"
    docs_dir = PROJECT_ROOT / "docs"
    logs_dir = PROJECT_ROOT / "logs"

    locked_model_config_path = config_dir / "locked_model_config.json"
    timeframe_config_path = config_dir / "timeframe_config.json"
    metric_config_path = config_dir / "metric_config.json"
    broker_symbol_map_path = config_dir / "broker_symbol_map.json"

    write_json(locked_model_config_path, build_locked_model_config())
    write_json(timeframe_config_path, build_timeframe_config())
    write_json(metric_config_path, build_metric_config())
    write_json(broker_symbol_map_path, build_broker_symbol_map())

    readme_path = PROJECT_ROOT / "README.md"
    rules_md_path = docs_dir / "locked_project_rules.md"
    write_text(readme_path, build_readme(PROJECT_ROOT))
    write_text(rules_md_path, build_locked_project_rules_md())

    dataset_copy_result = copy_tree_if_exists(SOURCE_DATASET_ROOT, PROJECT_ROOT / "dataset")

    folder_manifest = build_folder_manifest(created_paths, PROJECT_ROOT)
    bootstrap_log = {
        "version": VERSION,
        "project_root": str(PROJECT_ROOT),
        "dataset_copy": dataset_copy_result,
        "created_folders": folder_manifest,
        "config_files": [
            str(locked_model_config_path),
            str(timeframe_config_path),
            str(metric_config_path),
            str(broker_symbol_map_path),
        ],
        "doc_files": [
            str(readme_path),
            str(rules_md_path),
        ],
    }

    bootstrap_log_path = logs_dir / "bootstrap_log.json"
    write_json(bootstrap_log_path, bootstrap_log)

    print("[OK] Folders created")
    for item in folder_manifest:
        print(f" - {item['relative_path']}")

    print("=" * 100)
    print("[OK] Config files")
    print(f" - {locked_model_config_path}")
    print(f" - {timeframe_config_path}")
    print(f" - {metric_config_path}")
    print(f" - {broker_symbol_map_path}")

    print("=" * 100)
    print("[OK] Docs")
    print(f" - {readme_path}")
    print(f" - {rules_md_path}")

    print("=" * 100)
    print("[OK] Dataset copy result")
    print(f" - source_exists : {dataset_copy_result['source_exists']}")
    print(f" - copied        : {dataset_copy_result['copied']}")
    print(f" - files_copied  : {dataset_copy_result['files_copied']}")
    print(f" - destination   : {dataset_copy_result['destination']}")

    print("=" * 100)
    print("[DONE] Bootstrap complete")
    print(f"Bootstrap log: {bootstrap_log_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
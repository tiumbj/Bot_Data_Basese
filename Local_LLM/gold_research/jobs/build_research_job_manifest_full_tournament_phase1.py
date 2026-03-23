# version: v1.0.1
# file: C:\Data\Bot\Local_LLM\gold_research\jobs\build_research_job_manifest_full_tournament_phase1.py
# run:
# python C:\Data\Bot\Local_LLM\gold_research\jobs\build_research_job_manifest_full_tournament_phase1.py --config C:\Data\Bot\Local_LLM\gold_research\jobs\research_requirements_full_tournament_phase1.yaml

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import yaml


VERSION = "v1.0.1"
EXPECTED_TIMEFRAMES = ["M1", "M2", "M3", "M4", "M5", "M6", "M10", "M15", "M30", "H1", "H4", "D1"]

TF_ALIAS = {
    "M1": "m1",
    "M2": "m2",
    "M3": "m3",
    "M4": "m4",
    "M5": "m5",
    "M6": "m6",
    "M10": "m10",
    "M15": "m15",
    "M30": "m30",
    "H1": "h1",
    "H4": "h4",
    "D1": "d1",
}

STAGE_ALIAS = {
    "stage_3_full_tournament_phase1": "s3",
}

STRATEGY_ALIAS = {
    "locked_ms_bos_choch_pullback_atr_adx_ema_entry_v2_pullback_deep": "pbdeep",
}

MICRO_EXIT_ALIAS = {
    "micro_exit_v2_fast_invalidation": "mx_fi",
    "micro_exit_v2_momentum_fade": "mx_mf",
}

REGIME_ALIAS = {
    "regime_filter_off": "rg_off",
    "regime_filter_trend_only": "rg_to",
    "regime_filter_trend_or_neutral": "rg_ton",
}


@dataclass(frozen=True)
class BuildConfig:
    project_id: str
    project_name: str
    project_version: str
    output_root: Path
    manifest_jsonl: Path
    manifest_summary_json: Path
    state_jsonl: Path
    symbol: str
    tf_to_ohlc: dict[str, Path]
    stage_name: str
    family_id: str
    strategy_ids: list[str]
    entry_style: str
    micro_exit_ids: list[str]
    cooldown_bars: list[int]
    regime_filter_ids: list[str]
    expected_jobs: int
    defaults: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full tournament phase 1 manifest.")
    parser.add_argument("--config", required=True, help="Path to research_requirements_full_tournament_phase1.yaml")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a mapping/object.")
    return payload


def require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Required mapping '{key}' is missing or invalid.")
    return value


def require_list_of_str(parent: dict[str, Any], key: str) -> list[str]:
    value = parent.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Required list[str] '{key}' is missing or invalid.")
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if not text:
            raise ValueError(f"Invalid empty item in '{key}'")
        out.append(text)
    return out


def require_list_of_int(parent: dict[str, Any], key: str) -> list[int]:
    value = parent.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Required list[int] '{key}' is missing or invalid.")
    out: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"Invalid boolean in '{key}'")
        if isinstance(item, int):
            out.append(item)
            continue
        if isinstance(item, str) and item.strip().isdigit():
            out.append(int(item.strip()))
            continue
        raise ValueError(f"Invalid integer item in '{key}': {item!r}")
    return out


def normalize_tf_to_ohlc(raw: dict[str, Any]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for tf, path_text in raw.items():
        tf_key = str(tf).strip().upper()
        result[tf_key] = Path(str(path_text)).expanduser()
    return result


def load_config(path: Path) -> BuildConfig:
    raw = load_yaml(path)

    project = require_mapping(raw, "project")
    paths = require_mapping(raw, "paths")
    dataset = require_mapping(raw, "dataset")
    selection = require_mapping(raw, "selection")
    defaults = require_mapping(raw, "defaults")

    cfg = BuildConfig(
        project_id=str(project["project_id"]).strip(),
        project_name=str(project["project_name"]).strip(),
        project_version=str(project["version"]).strip(),
        output_root=Path(str(paths["output_root"])).expanduser(),
        manifest_jsonl=Path(str(paths["manifest_jsonl"])).expanduser(),
        manifest_summary_json=Path(str(paths["manifest_summary_json"])).expanduser(),
        state_jsonl=Path(str(paths["state_jsonl"])).expanduser(),
        symbol=str(dataset["symbol"]).strip(),
        tf_to_ohlc=normalize_tf_to_ohlc(require_mapping(dataset, "tf_to_ohlc")),
        stage_name=str(selection["stage_name"]).strip(),
        family_id=str(selection["family_id"]).strip(),
        strategy_ids=require_list_of_str(selection, "strategy_ids"),
        entry_style=str(selection["entry_style"]).strip(),
        micro_exit_ids=require_list_of_str(selection, "micro_exit_ids"),
        cooldown_bars=require_list_of_int(selection, "cooldown_bars"),
        regime_filter_ids=require_list_of_str(selection, "regime_filter_ids"),
        expected_jobs=int(selection["expected_jobs"]),
        defaults=defaults,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: BuildConfig) -> None:
    missing = [tf for tf in EXPECTED_TIMEFRAMES if tf not in cfg.tf_to_ohlc]
    extra = [tf for tf in cfg.tf_to_ohlc if tf not in EXPECTED_TIMEFRAMES]
    if missing:
        raise ValueError(f"Missing timeframes in tf_to_ohlc: {missing}")
    if extra:
        raise ValueError(f"Unexpected timeframes in tf_to_ohlc: {extra}")

    if cfg.family_id != "deep_pullback_continuation":
        raise ValueError(f"Phase 1 is locked to family_id='deep_pullback_continuation'. Got: {cfg.family_id}")
    if cfg.entry_style != "deep":
        raise ValueError(f"Phase 1 is locked to entry_style='deep'. Got: {cfg.entry_style}")

    generated_jobs = (
        len(EXPECTED_TIMEFRAMES)
        * len(cfg.strategy_ids)
        * len(cfg.micro_exit_ids)
        * len(cfg.cooldown_bars)
        * len(cfg.regime_filter_ids)
    )
    if generated_jobs != cfg.expected_jobs:
        raise ValueError(f"expected_jobs mismatch. generated={generated_jobs} expected={cfg.expected_jobs}")


def normalize_slug(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def build_job_id(
    cfg: BuildConfig,
    timeframe: str,
    strategy_id: str,
    micro_exit_id: str,
    cooldown_bars: int,
    regime_filter_id: str,
) -> str:
    parts = [
        cfg.stage_name,
        timeframe,
        strategy_id,
        micro_exit_id,
        f"cooldown_{cooldown_bars}",
        regime_filter_id,
    ]
    return "__".join(normalize_slug(part) for part in parts)


def alias_stage(stage_name: str) -> str:
    return STAGE_ALIAS.get(stage_name, normalize_slug(stage_name)[:12])


def alias_strategy(strategy_id: str) -> str:
    return STRATEGY_ALIAS.get(strategy_id, normalize_slug(strategy_id)[:16])


def alias_micro_exit(micro_exit_id: str) -> str:
    return MICRO_EXIT_ALIAS.get(micro_exit_id, normalize_slug(micro_exit_id)[:12])


def alias_regime(regime_filter_id: str) -> str:
    return REGIME_ALIAS.get(regime_filter_id, normalize_slug(regime_filter_id)[:12])


def build_job_result_dir(
    cfg: BuildConfig,
    timeframe: str,
    strategy_id: str,
    micro_exit_id: str,
    cooldown_bars: int,
    regime_filter_id: str,
) -> Path:
    return (
        cfg.output_root
        / alias_stage(cfg.stage_name)
        / TF_ALIAS[timeframe]
        / "i"
        / alias_strategy(strategy_id)
        / alias_micro_exit(micro_exit_id)
        / f"cd{cooldown_bars}"
        / alias_regime(regime_filter_id)
    )


def build_jobs(cfg: BuildConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    generated_at = utc_now_iso()

    for timeframe, strategy_id, micro_exit_id, cooldown_bars, regime_filter_id in product(
        EXPECTED_TIMEFRAMES,
        cfg.strategy_ids,
        cfg.micro_exit_ids,
        cfg.cooldown_bars,
        cfg.regime_filter_ids,
    ):
        result_dir = build_job_result_dir(
            cfg=cfg,
            timeframe=timeframe,
            strategy_id=strategy_id,
            micro_exit_id=micro_exit_id,
            cooldown_bars=cooldown_bars,
            regime_filter_id=regime_filter_id,
        )

        row = {
            "job_id": build_job_id(
                cfg=cfg,
                timeframe=timeframe,
                strategy_id=strategy_id,
                micro_exit_id=micro_exit_id,
                cooldown_bars=cooldown_bars,
                regime_filter_id=regime_filter_id,
            ),
            "version": VERSION,
            "generated_at_utc": generated_at,
            "project_id": cfg.project_id,
            "project_name": cfg.project_name,
            "stage": cfg.stage_name,
            "stage_name": cfg.stage_name,
            "symbol": cfg.symbol,
            "timeframe": timeframe,
            "family_id": cfg.family_id,
            "strategy_id": strategy_id,
            "entry_style": cfg.entry_style,
            "train_window_id": cfg.defaults["train_window_id"],
            "validation_mode": cfg.defaults["validation_mode"],
            "dataset": {
                "symbol": cfg.symbol,
                "timeframe": timeframe,
                "ohlc_csv": str(cfg.tf_to_ohlc[timeframe]),
            },
            "variant": {
                "micro_exit_id": micro_exit_id,
                "cooldown_bars": cooldown_bars,
                "regime_filter_id": regime_filter_id,
                "risk_model_id": cfg.defaults["risk_model_id"],
                "execution_model_id": cfg.defaults["execution_model_id"],
                "spread_model_id": cfg.defaults["spread_model_id"],
                "slippage_model_id": cfg.defaults["slippage_model_id"],
            },
            "artifact_paths": {
                "job_result_dir": str(result_dir),
                "summary_json": str(result_dir / "summary.json"),
            },
            "tags": list(cfg.defaults.get("tags", [])),
        }
        rows.append(row)

    rows.sort(key=lambda item: item["job_id"])
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_summary(cfg: BuildConfig, rows: list[dict[str, Any]], config_path: Path) -> dict[str, Any]:
    return {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "config_path": str(config_path),
        "project_id": cfg.project_id,
        "project_name": cfg.project_name,
        "stage": cfg.stage_name,
        "stage_name": cfg.stage_name,
        "family_id": cfg.family_id,
        "strategy_ids": cfg.strategy_ids,
        "entry_style": cfg.entry_style,
        "micro_exit_ids": cfg.micro_exit_ids,
        "cooldown_bars": cfg.cooldown_bars,
        "regime_filter_ids": cfg.regime_filter_ids,
        "total_jobs": len(rows),
        "expected_jobs": cfg.expected_jobs,
        "state_jsonl": str(cfg.state_jsonl),
        "manifest_jsonl": str(cfg.manifest_jsonl),
        "output_root": str(cfg.output_root),
        "timeframes": EXPECTED_TIMEFRAMES,
        "path_aliases": {
            "stage": STAGE_ALIAS,
            "strategy": STRATEGY_ALIAS,
            "micro_exit": MICRO_EXIT_ALIAS,
            "regime": REGIME_ALIAS,
            "timeframe": TF_ALIAS,
        },
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)

    rows = build_jobs(cfg)
    if len(rows) != cfg.expected_jobs:
        raise RuntimeError(f"Job count mismatch. actual={len(rows)} expected={cfg.expected_jobs}")

    write_jsonl(cfg.manifest_jsonl, rows)
    write_json(cfg.manifest_summary_json, build_summary(cfg, rows, config_path))

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] config={config_path}")
    print(f"[DONE] total_jobs={len(rows)}")
    print(f"[DONE] manifest={cfg.manifest_jsonl}")
    print(f"[DONE] summary={cfg.manifest_summary_json}")
    print("=" * 120)


if __name__ == "__main__":
    main()
# version: v1.0.0
# file: C:\Data\Bot\Local_LLM\gold_research\jobs\build_research_job_manifest_phase2_finalists.py
# run:
# python C:\Data\Bot\Local_LLM\gold_research\jobs\build_research_job_manifest_phase2_finalists.py --config C:\Data\Bot\Local_LLM\gold_research\jobs\research_requirements_phase2_finalists.yaml

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


VERSION = "v1.0.0"


@dataclass(frozen=True)
class FinalistRow:
    timeframe: str
    strategy_id: str
    micro_exit_id: str
    cooldown_bars: int
    regime_filter_id: str


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
    entry_style: str
    finalists: list[FinalistRow]
    expected_jobs: int
    defaults: dict[str, Any]


TF_ALIAS = {"M2": "m2", "M3": "m3", "M4": "m4", "H1": "h1", "D1": "d1"}
STAGE_ALIAS = {"stage_4_phase2_finalists": "s4"}
STRATEGY_ALIAS = {"locked_ms_bos_choch_pullback_atr_adx_ema_entry_v2_pullback_deep": "pbdeep"}
MICRO_EXIT_ALIAS = {
    "micro_exit_v2_fast_invalidation": "mx_fi",
    "micro_exit_v2_momentum_fade": "mx_mf",
}
REGIME_ALIAS = {
    "regime_filter_trend_only": "rg_to",
    "regime_filter_trend_or_neutral": "rg_ton",
    "regime_filter_off": "rg_off",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build phase 2 finalists manifest.")
    parser.add_argument("--config", required=True, help="Path to research_requirements_phase2_finalists.yaml")
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


def normalize_tf_to_ohlc(raw: dict[str, Any]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for tf, p in raw.items():
        result[str(tf).strip().upper()] = Path(str(p)).expanduser()
    return result


def parse_finalists(raw_items: list[dict[str, Any]]) -> list[FinalistRow]:
    out: list[FinalistRow] = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError("Each finalist must be a mapping")
        out.append(
            FinalistRow(
                timeframe=str(item["timeframe"]).strip().upper(),
                strategy_id=str(item["strategy_id"]).strip(),
                micro_exit_id=str(item["micro_exit_id"]).strip(),
                cooldown_bars=int(item["cooldown_bars"]),
                regime_filter_id=str(item["regime_filter_id"]).strip(),
            )
        )
    return out


def load_config(path: Path) -> BuildConfig:
    raw = load_yaml(path)

    project = require_mapping(raw, "project")
    paths = require_mapping(raw, "paths")
    dataset = require_mapping(raw, "dataset")
    selection = require_mapping(raw, "selection")
    defaults = require_mapping(raw, "defaults")

    finalists_raw = selection.get("finalists")
    if not isinstance(finalists_raw, list) or not finalists_raw:
        raise ValueError("selection.finalists is required and must be a non-empty list")

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
        entry_style=str(selection["entry_style"]).strip(),
        finalists=parse_finalists(finalists_raw),
        expected_jobs=int(selection["expected_jobs"]),
        defaults=defaults,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: BuildConfig) -> None:
    if cfg.family_id != "deep_pullback_continuation":
        raise ValueError("Phase 2 finalists locked to family_id='deep_pullback_continuation'")
    if cfg.entry_style != "deep":
        raise ValueError("Phase 2 finalists locked to entry_style='deep'")
    if len(cfg.finalists) != cfg.expected_jobs:
        raise ValueError(f"expected_jobs mismatch. finalists={len(cfg.finalists)} expected={cfg.expected_jobs}")

    seen: set[tuple[str, str, str, int, str]] = set()
    for row in cfg.finalists:
        key = (row.timeframe, row.strategy_id, row.micro_exit_id, row.cooldown_bars, row.regime_filter_id)
        if key in seen:
            raise ValueError(f"Duplicate finalist row detected: {key}")
        seen.add(key)

        if row.timeframe not in cfg.tf_to_ohlc:
            raise ValueError(f"Missing tf_to_ohlc for finalist timeframe: {row.timeframe}")

        if row.strategy_id != "locked_ms_bos_choch_pullback_atr_adx_ema_entry_v2_pullback_deep":
            raise ValueError(f"Unsupported strategy_id in finalists: {row.strategy_id}")

        if row.micro_exit_id not in {"micro_exit_v2_fast_invalidation", "micro_exit_v2_momentum_fade"}:
            raise ValueError(f"Unsupported micro_exit_id in finalists: {row.micro_exit_id}")

        if row.cooldown_bars not in {0, 3}:
            raise ValueError(f"Unsupported cooldown_bars in finalists: {row.cooldown_bars}")

        if row.regime_filter_id not in {
            "regime_filter_off",
            "regime_filter_trend_only",
            "regime_filter_trend_or_neutral",
        }:
            raise ValueError(f"Unsupported regime_filter_id in finalists: {row.regime_filter_id}")


def normalize_slug(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def build_job_id(stage_name: str, row: FinalistRow) -> str:
    parts = [
        stage_name,
        row.timeframe,
        row.strategy_id,
        row.micro_exit_id,
        f"cooldown_{row.cooldown_bars}",
        row.regime_filter_id,
    ]
    return "__".join(normalize_slug(p) for p in parts)


def build_job_result_dir(cfg: BuildConfig, row: FinalistRow) -> Path:
    return (
        cfg.output_root
        / STAGE_ALIAS.get(cfg.stage_name, "s4")
        / TF_ALIAS[row.timeframe]
        / "i"
        / STRATEGY_ALIAS[row.strategy_id]
        / MICRO_EXIT_ALIAS[row.micro_exit_id]
        / f"cd{row.cooldown_bars}"
        / REGIME_ALIAS[row.regime_filter_id]
    )


def build_jobs(cfg: BuildConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    generated_at = utc_now_iso()

    for finalist in cfg.finalists:
        result_dir = build_job_result_dir(cfg, finalist)
        rows.append(
            {
                "job_id": build_job_id(cfg.stage_name, finalist),
                "version": VERSION,
                "generated_at_utc": generated_at,
                "project_id": cfg.project_id,
                "project_name": cfg.project_name,
                "stage": cfg.stage_name,
                "stage_name": cfg.stage_name,
                "symbol": cfg.symbol,
                "timeframe": finalist.timeframe,
                "family_id": cfg.family_id,
                "strategy_id": finalist.strategy_id,
                "entry_style": cfg.entry_style,
                "train_window_id": cfg.defaults["train_window_id"],
                "validation_mode": cfg.defaults["validation_mode"],
                "dataset": {
                    "symbol": cfg.symbol,
                    "timeframe": finalist.timeframe,
                    "ohlc_csv": str(cfg.tf_to_ohlc[finalist.timeframe]),
                },
                "variant": {
                    "micro_exit_id": finalist.micro_exit_id,
                    "cooldown_bars": finalist.cooldown_bars,
                    "regime_filter_id": finalist.regime_filter_id,
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
        )

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
        "entry_style": cfg.entry_style,
        "total_jobs": len(rows),
        "expected_jobs": cfg.expected_jobs,
        "state_jsonl": str(cfg.state_jsonl),
        "manifest_jsonl": str(cfg.manifest_jsonl),
        "output_root": str(cfg.output_root),
        "finalists": [
            {
                "timeframe": f.timeframe,
                "strategy_id": f.strategy_id,
                "micro_exit_id": f.micro_exit_id,
                "cooldown_bars": f.cooldown_bars,
                "regime_filter_id": f.regime_filter_id,
            }
            for f in cfg.finalists
        ],
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)

    rows = build_jobs(cfg)
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
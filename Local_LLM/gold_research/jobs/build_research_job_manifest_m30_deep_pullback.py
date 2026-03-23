# version: v1.0.1
# file: C:\Data\Bot\Local_LLM\gold_research\jobs\build_research_job_manifest_m30_deep_pullback.py
# run:
# python C:\Data\Bot\Local_LLM\gold_research\jobs\build_research_job_manifest_m30_deep_pullback.py --config C:\Data\Bot\Local_LLM\gold_research\jobs\research_requirements_m30_deep_pullback.yaml

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
    timeframe: str
    ohlc_csv: Path
    stage_name: str
    family_id: str
    strategy_id: str
    entry_style: str
    expected_jobs: int
    micro_exit_ids: list[str]
    cooldown_bars: list[int]
    regime_filter_ids: list[str]
    defaults: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build focused research manifest for M30 deep pullback only."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to research_requirements_m30_deep_pullback.yaml",
    )
    return parser.parse_args()


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
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x.strip() for x in value):
        raise ValueError(f"Required list[str] '{key}' is missing or invalid.")
    return [x.strip() for x in value]


def require_list_of_int(parent: dict[str, Any], key: str) -> list[int]:
    value = parent.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Required list[int] '{key}' is missing or invalid.")
    normalized: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"Invalid boolean found in '{key}'.")
        if isinstance(item, int):
            normalized.append(item)
            continue
        if isinstance(item, str) and item.strip().isdigit():
            normalized.append(int(item.strip()))
            continue
        raise ValueError(f"Invalid item in '{key}': {item!r}")
    return normalized


def load_config(path: Path) -> BuildConfig:
    raw = load_yaml(path)

    project = require_mapping(raw, "project")
    paths = require_mapping(raw, "paths")
    dataset = require_mapping(raw, "dataset")
    selection = require_mapping(raw, "selection")
    variants = require_mapping(raw, "variants")
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
        timeframe=str(dataset["timeframe"]).strip().upper(),
        ohlc_csv=Path(str(dataset["ohlc_csv"])).expanduser(),
        stage_name=str(selection["stage_name"]).strip(),
        family_id=str(selection["family_id"]).strip(),
        strategy_id=str(selection["strategy_id"]).strip(),
        entry_style=str(selection["entry_style"]).strip(),
        expected_jobs=int(selection["expected_jobs"]),
        micro_exit_ids=require_list_of_str(variants, "micro_exit_ids"),
        cooldown_bars=require_list_of_int(variants, "cooldown_bars"),
        regime_filter_ids=require_list_of_str(variants, "regime_filter_ids"),
        defaults=defaults,
    )

    validate_config(cfg)
    return cfg


def validate_config(cfg: BuildConfig) -> None:
    if cfg.timeframe != "M30":
        raise ValueError(f"Only timeframe M30 is allowed. Got: {cfg.timeframe}")
    if cfg.family_id != "deep_pullback_continuation":
        raise ValueError(
            "Only family_id='deep_pullback_continuation' is allowed "
            f"for this builder. Got: {cfg.family_id}"
        )
    if cfg.entry_style != "deep":
        raise ValueError(f"Only entry_style='deep' is allowed. Got: {cfg.entry_style}")
    if not cfg.strategy_id.endswith("pullback_deep"):
        raise ValueError(
            "This builder only supports strategy_id ending with 'pullback_deep'. "
            f"Got: {cfg.strategy_id}"
        )

    generated_jobs = (
        len(cfg.micro_exit_ids)
        * len(cfg.cooldown_bars)
        * len(cfg.regime_filter_ids)
    )
    if generated_jobs != cfg.expected_jobs:
        raise ValueError(
            "Variant grid does not match expected_jobs. "
            f"generated={generated_jobs} expected={cfg.expected_jobs}"
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_slug(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_job_id(
    stage_name: str,
    timeframe: str,
    strategy_id: str,
    micro_exit_id: str,
    cooldown_bars: int,
    regime_filter_id: str,
) -> str:
    parts = [
        stage_name,
        timeframe,
        strategy_id,
        micro_exit_id,
        f"cooldown_{cooldown_bars}",
        regime_filter_id,
    ]
    return "__".join(normalize_slug(part) for part in parts)


def build_job_result_dir(
    cfg: BuildConfig,
    micro_exit_id: str,
    cooldown_bars: int,
    regime_filter_id: str,
) -> Path:
    return (
        cfg.output_root
        / cfg.stage_name
        / cfg.timeframe
        / "insample"
        / cfg.strategy_id
        / micro_exit_id
        / f"cooldown_{cooldown_bars}"
        / regime_filter_id
    )


def build_jobs(cfg: BuildConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    generated_at = utc_now_iso()

    for micro_exit_id, cooldown_bars, regime_filter_id in product(
        cfg.micro_exit_ids,
        cfg.cooldown_bars,
        cfg.regime_filter_ids,
    ):
        result_dir = build_job_result_dir(
            cfg=cfg,
            micro_exit_id=micro_exit_id,
            cooldown_bars=cooldown_bars,
            regime_filter_id=regime_filter_id,
        )
        summary_json = result_dir / "summary.json"

        job_id = build_job_id(
            stage_name=cfg.stage_name,
            timeframe=cfg.timeframe,
            strategy_id=cfg.strategy_id,
            micro_exit_id=micro_exit_id,
            cooldown_bars=cooldown_bars,
            regime_filter_id=regime_filter_id,
        )

        row = {
            "job_id": job_id,
            "version": VERSION,
            "generated_at_utc": generated_at,
            "project_id": cfg.project_id,
            "project_name": cfg.project_name,
            "stage": cfg.stage_name,
            "stage_name": cfg.stage_name,
            "symbol": cfg.symbol,
            "timeframe": cfg.timeframe,
            "family_id": cfg.family_id,
            "strategy_id": cfg.strategy_id,
            "entry_style": cfg.entry_style,
            "train_window_id": cfg.defaults["train_window_id"],
            "validation_mode": cfg.defaults["validation_mode"],
            "dataset": {
                "symbol": cfg.symbol,
                "timeframe": cfg.timeframe,
                "ohlc_csv": str(cfg.ohlc_csv),
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
                "summary_json": str(summary_json),
            },
            "tags": list(cfg.defaults.get("tags", [])),
        }
        rows.append(row)

    rows.sort(key=lambda item: item["job_id"])
    return rows


def build_summary(
    cfg: BuildConfig,
    rows: list[dict[str, Any]],
    config_path: Path,
) -> dict[str, Any]:
    micro_exit_counts: dict[str, int] = {}
    cooldown_counts: dict[str, int] = {}
    regime_counts: dict[str, int] = {}

    for row in rows:
        micro_exit_id = row["variant"]["micro_exit_id"]
        cooldown_bars = str(row["variant"]["cooldown_bars"])
        regime_filter_id = row["variant"]["regime_filter_id"]

        micro_exit_counts[micro_exit_id] = micro_exit_counts.get(micro_exit_id, 0) + 1
        cooldown_counts[cooldown_bars] = cooldown_counts.get(cooldown_bars, 0) + 1
        regime_counts[regime_filter_id] = regime_counts.get(regime_filter_id, 0) + 1

    return {
        "version": VERSION,
        "generated_at_utc": utc_now_iso(),
        "config_path": str(config_path),
        "project_id": cfg.project_id,
        "project_name": cfg.project_name,
        "stage": cfg.stage_name,
        "stage_name": cfg.stage_name,
        "timeframe": cfg.timeframe,
        "family_id": cfg.family_id,
        "strategy_id": cfg.strategy_id,
        "entry_style": cfg.entry_style,
        "total_jobs": len(rows),
        "expected_jobs": cfg.expected_jobs,
        "state_jsonl": str(cfg.state_jsonl),
        "manifest_jsonl": str(cfg.manifest_jsonl),
        "output_root": str(cfg.output_root),
        "counts": {
            "micro_exit_id": micro_exit_counts,
            "cooldown_bars": cooldown_counts,
            "regime_filter_id": regime_counts,
        },
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)

    rows = build_jobs(cfg)
    if len(rows) != cfg.expected_jobs:
        raise RuntimeError(
            f"Job count mismatch after build. actual={len(rows)} expected={cfg.expected_jobs}"
        )

    write_jsonl(cfg.manifest_jsonl, rows)
    summary = build_summary(cfg, rows, config_path)
    write_json(cfg.manifest_summary_json, summary)

    print("=" * 120)
    print(f"[DONE] version={VERSION}")
    print(f"[DONE] config={config_path}")
    print(f"[DONE] timeframe={cfg.timeframe}")
    print(f"[DONE] strategy_id={cfg.strategy_id}")
    print(f"[DONE] family_id={cfg.family_id}")
    print(f"[DONE] total_jobs={len(rows)}")
    print(f"[DONE] manifest={cfg.manifest_jsonl}")
    print(f"[DONE] summary={cfg.manifest_summary_json}")
    print("=" * 120)


if __name__ == "__main__":
    main()
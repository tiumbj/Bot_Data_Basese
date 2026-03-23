# version: v1.0.0
# file: C:\Data\Bot\Local_LLM\gold_research\jobs\build_research_job_manifest_multitf_winner.py
# run:
# python C:\Data\Bot\Local_LLM\gold_research\jobs\build_research_job_manifest_multitf_winner.py --config C:\Data\Bot\Local_LLM\gold_research\jobs\research_requirements_multitf_winner.yaml

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


VERSION = "v1.0.0"
EXPECTED_TIMEFRAMES = ["M1", "M2", "M3", "M4", "M5", "M6", "M10", "M15", "M30", "H1", "H4", "D1"]


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
    strategy_id: str
    entry_style: str
    micro_exit_id: str
    cooldown_bars: int
    regime_filter_id: str
    expected_jobs: int
    defaults: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multi-timeframe winner manifest.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to research_requirements_multitf_winner.yaml",
    )
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

    tf_to_ohlc_raw = require_mapping(dataset, "tf_to_ohlc")

    cfg = BuildConfig(
        project_id=str(project["project_id"]).strip(),
        project_name=str(project["project_name"]).strip(),
        project_version=str(project["version"]).strip(),
        output_root=Path(str(paths["output_root"])).expanduser(),
        manifest_jsonl=Path(str(paths["manifest_jsonl"])).expanduser(),
        manifest_summary_json=Path(str(paths["manifest_summary_json"])).expanduser(),
        state_jsonl=Path(str(paths["state_jsonl"])).expanduser(),
        symbol=str(dataset["symbol"]).strip(),
        tf_to_ohlc=normalize_tf_to_ohlc(tf_to_ohlc_raw),
        stage_name=str(selection["stage_name"]).strip(),
        family_id=str(selection["family_id"]).strip(),
        strategy_id=str(selection["strategy_id"]).strip(),
        entry_style=str(selection["entry_style"]).strip(),
        micro_exit_id=str(selection["micro_exit_id"]).strip(),
        cooldown_bars=int(selection["cooldown_bars"]),
        regime_filter_id=str(selection["regime_filter_id"]).strip(),
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
        raise ValueError(f"Only family_id='deep_pullback_continuation' is allowed. Got: {cfg.family_id}")
    if cfg.entry_style != "deep":
        raise ValueError(f"Only entry_style='deep' is allowed. Got: {cfg.entry_style}")
    if cfg.micro_exit_id != "micro_exit_v2_momentum_fade":
        raise ValueError(
            "This builder is locked to the current M30 winner micro_exit_id='micro_exit_v2_momentum_fade'. "
            f"Got: {cfg.micro_exit_id}"
        )
    if cfg.cooldown_bars != 0:
        raise ValueError(
            "This builder is locked to the current M30 winner cooldown_bars=0. "
            f"Got: {cfg.cooldown_bars}"
        )
    if cfg.regime_filter_id != "regime_filter_trend_only":
        raise ValueError(
            "This builder is locked to the current M30 winner regime_filter_id='regime_filter_trend_only'. "
            f"Got: {cfg.regime_filter_id}"
        )
    if cfg.expected_jobs != len(EXPECTED_TIMEFRAMES):
        raise ValueError(
            f"expected_jobs must equal {len(EXPECTED_TIMEFRAMES)} for the locked timeframe set. "
            f"Got: {cfg.expected_jobs}"
        )


def normalize_slug(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def build_job_id(cfg: BuildConfig, timeframe: str) -> str:
    parts = [
        cfg.stage_name,
        timeframe,
        cfg.strategy_id,
        cfg.micro_exit_id,
        f"cooldown_{cfg.cooldown_bars}",
        cfg.regime_filter_id,
    ]
    return "__".join(normalize_slug(part) for part in parts)


def build_job_result_dir(cfg: BuildConfig, timeframe: str) -> Path:
    return (
        cfg.output_root
        / cfg.stage_name
        / timeframe
        / "insample"
        / cfg.strategy_id
        / cfg.micro_exit_id
        / f"cooldown_{cfg.cooldown_bars}"
        / cfg.regime_filter_id
    )


def build_jobs(cfg: BuildConfig) -> list[dict[str, Any]]:
    generated_at = utc_now_iso()
    rows: list[dict[str, Any]] = []

    for timeframe in EXPECTED_TIMEFRAMES:
        result_dir = build_job_result_dir(cfg, timeframe)
        row = {
            "job_id": build_job_id(cfg, timeframe),
            "version": VERSION,
            "generated_at_utc": generated_at,
            "project_id": cfg.project_id,
            "project_name": cfg.project_name,
            "stage": cfg.stage_name,
            "stage_name": cfg.stage_name,
            "symbol": cfg.symbol,
            "timeframe": timeframe,
            "family_id": cfg.family_id,
            "strategy_id": cfg.strategy_id,
            "entry_style": cfg.entry_style,
            "train_window_id": cfg.defaults["train_window_id"],
            "validation_mode": cfg.defaults["validation_mode"],
            "dataset": {
                "symbol": cfg.symbol,
                "timeframe": timeframe,
                "ohlc_csv": str(cfg.tf_to_ohlc[timeframe]),
            },
            "variant": {
                "micro_exit_id": cfg.micro_exit_id,
                "cooldown_bars": cfg.cooldown_bars,
                "regime_filter_id": cfg.regime_filter_id,
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

    if len(rows) != cfg.expected_jobs:
        raise RuntimeError(f"Job count mismatch. actual={len(rows)} expected={cfg.expected_jobs}")

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
        "strategy_id": cfg.strategy_id,
        "entry_style": cfg.entry_style,
        "micro_exit_id": cfg.micro_exit_id,
        "cooldown_bars": cfg.cooldown_bars,
        "regime_filter_id": cfg.regime_filter_id,
        "total_jobs": len(rows),
        "expected_jobs": cfg.expected_jobs,
        "state_jsonl": str(cfg.state_jsonl),
        "manifest_jsonl": str(cfg.manifest_jsonl),
        "output_root": str(cfg.output_root),
        "timeframes": EXPECTED_TIMEFRAMES,
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
    print(f"[DONE] micro_exit_id={cfg.micro_exit_id}")
    print(f"[DONE] cooldown_bars={cfg.cooldown_bars}")
    print(f"[DONE] regime_filter_id={cfg.regime_filter_id}")
    print(f"[DONE] manifest={cfg.manifest_jsonl}")
    print(f"[DONE] summary={cfg.manifest_summary_json}")
    print("=" * 120)


if __name__ == "__main__":
    main()
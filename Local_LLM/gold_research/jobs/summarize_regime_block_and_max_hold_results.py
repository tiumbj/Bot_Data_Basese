# version: v1.0.0
# file: C:\Data\Bot\Local_LLM\gold_research\jobs\summarize_regime_block_and_max_hold_results.py
# run:
# python C:\Data\Bot\Local_LLM\gold_research\jobs\summarize_regime_block_and_max_hold_results.py --input C:\Data\Bot\central_backtest_results\research_regime_max_hold\variant_results.jsonl

from __future__ import annotations

import argparse
import json
from pathlib import Path


VERSION = "v1.0.0"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize regime-block + max-hold proxy results")
    parser.add_argument("--input", required=True, help="Path to variant_results.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    rows = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"[INFO] version={VERSION}")
    print(f"[INFO] rows={len(rows)}")
    print("")

    rows = sorted(rows, key=lambda x: (x["payoff_ratio"], x["pnl_sum"]), reverse=True)

    for row in rows:
        print("=" * 80)
        print(f"variant: {row['variant']}")
        print(f"trades: {row['trades']}")
        print(f"wins: {row['wins']}")
        print(f"losses: {row['losses']}")
        print(f"win_rate_pct: {row['win_rate_pct']}")
        print(f"pnl_sum: {row['pnl_sum']}")
        print(f"payoff_ratio: {row['payoff_ratio']}")
        print(f"max_consecutive_losses: {row['max_consecutive_losses']}")
        print(f"take_profit_count: {row['take_profit_count']}")
        print(f"stop_loss_count: {row['stop_loss_count']}")
        print(f"max_hold_proxy_count: {row['max_hold_proxy_count']}")
        print(f"max_hold_proxy_from_original_tp: {row['max_hold_proxy_from_original_tp']}")
        print(f"max_hold_proxy_from_original_sl: {row['max_hold_proxy_from_original_sl']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
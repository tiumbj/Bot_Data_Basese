# ============================================================
# ชื่อโค้ด: apply_pullback_deep_block_short_below_ema_patch.py
# ที่อยู่ไฟล์: C:\Data\Bot\Local_LLM\gold_research\jobs\apply_pullback_deep_block_short_below_ema_patch.py
# คำสั่งรัน: python C:\Data\Bot\Local_LLM\gold_research\jobs\apply_pullback_deep_block_short_below_ema_patch.py
# เวอร์ชัน: v1.0.1
# ============================================================

from __future__ import annotations

import shutil
from pathlib import Path


VERSION = "v1.0.1"

TARGET_FILE = Path(r"C:\Data\Bot\Local_LLM\gold_research\jobs\build_entry_v2_component_packages.py")
BACKUP_FILE = Path(
    r"C:\Data\Bot\Local_LLM\gold_research\jobs\build_entry_v2_component_packages.py.bak_block_short_below_ema_v1_0_1"
)


def log(message: str) -> None:
    print(f"[apply_pullback_deep_block_short_below_ema_patch {VERSION}] {message}")


def require_contains(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise RuntimeError(f"Anchor not found: {label}")


def patch_entry_variant_block(text: str) -> str:
    old_block = """    "entry_v2_pullback_deep": {
        "bos_fresh_bars": 6,
        "choch_confirm_bars": 1,
        "swing_lookback": 2,
        "pullback_mode": "deep",
        "adx_threshold": 20.0,
        "ema_gap_min": 0.0,
    },"""

    new_block = """    "entry_v2_pullback_deep": {
        "bos_fresh_bars": 6,
        "choch_confirm_bars": 1,
        "swing_lookback": 2,
        "pullback_mode": "deep",
        "adx_threshold": 20.0,
        "ema_gap_min": 0.0,
        "block_short_below_ema_stack": False,
    },
    "entry_v2_pullback_deep_m30_block_short_below_ema": {
        "bos_fresh_bars": 6,
        "choch_confirm_bars": 1,
        "swing_lookback": 2,
        "pullback_mode": "deep",
        "adx_threshold": 20.0,
        "ema_gap_min": 0.0,
        "block_short_below_ema_stack": True,
    },"""

    if '"entry_v2_pullback_deep_m30_block_short_below_ema"' in text:
        log("entry variant already exists, skip variant patch")
        return text

    if old_block not in text:
        raise RuntimeError("Cannot find exact entry_v2_pullback_deep block to patch.")

    return text.replace(old_block, new_block, 1)


def patch_price_location_bucket(text: str) -> str:
    if 'work["price_location_bucket"]' in text:
        log("price_location_bucket already exists, skip bucket patch")
        return text

    old_block = """    ema_gap = (work["ema20"] - work["ema50"]).abs()"""

    new_block = """    work["price_location_bucket"] = "INSIDE_EMA_ZONE"
    work.loc[
        (work["close"] > work["ema20"]) & (work["close"] > work["ema50"]),
        "price_location_bucket"
    ] = "ABOVE_EMA_STACK"
    work.loc[
        (work["close"] < work["ema20"]) & (work["close"] < work["ema50"]),
        "price_location_bucket"
    ] = "BELOW_EMA_STACK"

    ema_gap = (work["ema20"] - work["ema50"]).abs()"""

    if old_block not in text:
        raise RuntimeError("Cannot find ema_gap anchor for price_location_bucket patch.")

    return text.replace(old_block, new_block, 1)


def patch_short_entry_filter(text: str) -> str:
    old_line = """    work["short_entry"] = bearish_pullback & bear_filter_ok & fresh_bos_short & choch_short_ok"""

    new_block = """    work["short_entry"] = bearish_pullback & bear_filter_ok & fresh_bos_short & choch_short_ok

    if bool(params.get("block_short_below_ema_stack", False)):
        work["short_entry"] = work["short_entry"] & (work["price_location_bucket"] != "BELOW_EMA_STACK")"""

    if '(work["price_location_bucket"] != "BELOW_EMA_STACK")' in text:
        log("short entry block rule already exists, skip short_entry patch")
        return text

    if old_line not in text:
        raise RuntimeError("Cannot find short_entry line to patch.")

    return text.replace(old_line, new_block, 1)


def patch_output_columns(text: str) -> str:
    if '"price_location_bucket": work.loc[ready, "price_location_bucket"].fillna("UNKNOWN").astype(str),' in text:
        log("output column already exists, skip output patch")
        return text

    old_block = """        "short_entry": work.loc[ready, "short_entry"].fillna(False).astype(bool),
        "exit_long": work.loc[ready, "exit_long"].fillna(False).astype(bool),
        "exit_short": work.loc[ready, "exit_short"].fillna(False).astype(bool),"""

    new_block = """        "short_entry": work.loc[ready, "short_entry"].fillna(False).astype(bool),
        "price_location_bucket": work.loc[ready, "price_location_bucket"].fillna("UNKNOWN").astype(str),
        "exit_long": work.loc[ready, "exit_long"].fillna(False).astype(bool),
        "exit_short": work.loc[ready, "exit_short"].fillna(False).astype(bool),"""

    if old_block not in text:
        raise RuntimeError("Cannot find output mapping block to patch.")

    return text.replace(old_block, new_block, 1)


def main() -> None:
    if not TARGET_FILE.exists():
        raise FileNotFoundError(f"Target file not found: {TARGET_FILE}")

    text = TARGET_FILE.read_text(encoding="utf-8")

    require_contains(text, '"entry_v2_pullback_deep": {', "entry_v2_pullback_deep")
    require_contains(text, 'work["ema20"] = ema(work["close"], 20)', "ema20 line")
    require_contains(text, 'work["ema50"] = ema(work["close"], 50)', "ema50 line")
    require_contains(
        text,
        'work["short_entry"] = bearish_pullback & bear_filter_ok & fresh_bos_short & choch_short_ok',
        "short_entry line",
    )

    if not BACKUP_FILE.exists():
        shutil.copy2(TARGET_FILE, BACKUP_FILE)
        log(f"backup_created={BACKUP_FILE}")
    else:
        log(f"backup_exists={BACKUP_FILE}")

    patched = text
    patched = patch_entry_variant_block(patched)
    patched = patch_price_location_bucket(patched)
    patched = patch_short_entry_filter(patched)
    patched = patch_output_columns(patched)

    TARGET_FILE.write_text(patched, encoding="utf-8")
    log(f"patched_file={TARGET_FILE}")
    log("status=SUCCESS")


if __name__ == "__main__":
    main()
from pathlib import Path


TARGET_FILE = Path(r"C:\Data\Bot\Local_LLM\gold_research\jobs\build_entry_v2_component_packages.py")


def patch_text(text: str) -> tuple[str, bool]:
    if '(work["close"] >= work["ema20"]) &' in text:
        return text, False

    old_block = '''        bearish_pullback = (
            (work["bias"] == "BEARISH") &
            (work["high"] >= work["ema20"]) &
            (work["close"] <= ((work["ema20"] + work["ema50"]) / 2.0)) &
            (work["ema20"] < work["ema50"])
        )'''

    new_block = '''        bearish_pullback = (
            (work["bias"] == "BEARISH") &
            (work["high"] >= work["ema20"]) &
            (work["close"] >= work["ema20"]) &
            (work["close"] <= ((work["ema20"] + work["ema50"]) / 2.0)) &
            (work["ema20"] < work["ema50"])
        )'''

    if old_block not in text:
        raise RuntimeError("Expected bearish_pullback block not found.")

    return text.replace(old_block, new_block, 1), True


def main() -> None:
    if not TARGET_FILE.exists():
        raise FileNotFoundError(str(TARGET_FILE))

    source = TARGET_FILE.read_text(encoding="utf-8")
    patched, changed = patch_text(source)
    if changed:
        TARGET_FILE.write_text(patched, encoding="utf-8")
        print("PATCH_APPLIED")
    else:
        print("ALREADY_APPLIED")


if __name__ == "__main__":
    main()

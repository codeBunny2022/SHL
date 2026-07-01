#!/usr/bin/env python3
"""Normalize shl_product_catalog.json into data/catalog.json."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import CATALOG_PATH, DATA_DIR, KEY_TO_LETTER, LETTER_ORDER

SOURCE = ROOT / "shl_product_catalog.json"


def clean_text(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", value).strip()


def keys_to_test_type(keys: list[str]) -> str:
    letters = []
    for letter in LETTER_ORDER:
        for key in keys:
            if KEY_TO_LETTER.get(key) == letter:
                letters.append(letter)
                break
    return ",".join(letters)


def build_embed_text(item: dict) -> str:
    parts = [
        item["name"],
        ", ".join(item.get("keys", [])),
        item.get("description", ""),
        ", ".join(item.get("job_levels", [])),
        ", ".join(item.get("languages", [])),
    ]
    return clean_text(" | ".join(p for p in parts if p))


def normalize(raw: list[dict]) -> list[dict]:
    out = []
    for row in raw:
        keys = row.get("keys") or []
        item = {
            "entity_id": row.get("entity_id", ""),
            "name": clean_text(row.get("name", "")),
            "url": clean_text(row.get("link", "")),
            "test_type": keys_to_test_type(keys),
            "keys": keys,
            "description": clean_text(row.get("description", "")),
            "job_levels": row.get("job_levels") or [],
            "languages": row.get("languages") or [],
            "duration": clean_text(row.get("duration", "")),
            "remote": row.get("remote", ""),
            "adaptive": row.get("adaptive", ""),
        }
        item["embed_text"] = build_embed_text(item)
        out.append(item)
    return out


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SOURCE, encoding="utf-8") as f:
        raw = json.load(f, strict=False)
    catalog = normalize(raw)
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(catalog)} items to {CATALOG_PATH}")


if __name__ == "__main__":
    main()

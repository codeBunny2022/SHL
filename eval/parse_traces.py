#!/usr/bin/env python3
"""Parse GenAI_SampleConversations into gold evaluation traces."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONV_DIR = ROOT / "GenAI_SampleConversations"
OUT = ROOT / "eval" / "traces.json"


def extract_table_names(section: str) -> list[str]:
    match = re.search(
        r"\| # \| Name \|.*?\n\|[-| ]+\|\n((?:\|.*\n)+)",
        section,
        re.DOTALL,
    )
    if not match:
        return []
    names: list[str] = []
    for row in match.group(1).strip().splitlines():
        cols = [c.strip() for c in row.split("|") if c.strip()]
        if len(cols) >= 2 and cols[0].isdigit():
            names.append(cols[1])
    # preserve order, dedupe
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            out.append(name)
    return out


def parse_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    trace_id = path.stem

    user_turns: list[str] = []
    for block in re.split(r"### Turn \d+", text)[1:]:
        user_match = re.search(r"\*\*User\*\*\s*\n\n>\s*(.+?)(?=\n\n\*\*Agent\*\*|\Z)", block, re.DOTALL)
        if user_match:
            user_turns.append(user_match.group(1).strip())

    final_section = text
    end_match = list(re.finditer(r"_`end_of_conversation`:\s*\*\*true\*\*_", text))
    if end_match:
        final_section = text[: end_match[-1].start()]

    gold_final = extract_table_names(final_section.split("### Turn")[-1])

    return {
        "id": trace_id,
        "turns": user_turns,
        "gold_final": gold_final,
    }


def main() -> None:
    traces = [parse_file(p) for p in sorted(CONV_DIR.glob("C*.md"), key=lambda p: int(p.stem[1:]))]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2)
    print(f"Wrote {len(traces)} traces to {OUT}")


if __name__ == "__main__":
    main()

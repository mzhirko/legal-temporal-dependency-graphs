#!/usr/bin/env python3
"""Replace decorative Unicode glyphs in source files with ASCII, in place.

Usage, from a repo root:

    python3 ascii_prompts.py src/                 # experiments repo
    python3 ascii_prompts.py packages/            # timebar repo (optional)

Idempotent: running twice changes nothing. Prints a per-file replacement
count and a total. The mapping matches the thesis appendix exactly, so
after running, the appendix verbatim blocks and the repository files carry
identical prompt text.
"""
import sys
from pathlib import Path

MAPPING = {
    "\u2014": "--",   # em dash
    "\u2013": "-",    # en dash
    "\u2501": "=",    # box drawings heavy horizontal
    "\u2500": "-",    # box drawings light horizontal
    "\u2192": "->",   # rightwards arrow
    "\u2717": "x",    # ballot x
    "\u26a0": "(!)",  # warning sign
    "\u201c": '"', "\u201d": '"',   # curly double quotes
    "\u2018": "'", "\u2019": "'",   # curly single quotes
    "\ufe0f": "",     # variation selector (emoji presentation)
}


def main():
    if len(sys.argv) < 2:
        sys.exit("pass one or more directories, e.g. src/")
    total = 0
    for root in sys.argv[1:]:
        for py in sorted(Path(root).rglob("*.py")):
            if "__pycache__" in str(py):
                continue
            text = py.read_text(encoding="utf-8", errors="strict")
            new = text
            for k, v in MAPPING.items():
                new = new.replace(k, v)
            if new != text:
                n = sum(text.count(k) for k in MAPPING)
                py.write_text(new, encoding="utf-8")
                print(f"{py}: {n} replacement(s)")
                total += n
            leftover = sorted({c for c in new if ord(c) > 127})
            if leftover:
                print(f"{py}: NOTE non-ASCII remains (not in mapping): "
                      f"{[hex(ord(c)) for c in leftover]}")
    print(f"total replacements: {total}")


if __name__ == "__main__":
    main()

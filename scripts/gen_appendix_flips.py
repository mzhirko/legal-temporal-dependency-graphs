#!/usr/bin/env python3
"""Dump every prompt literal from the experiments repo, verbatim.

Run from the repo root (the directory containing src/):

    python3 dump_prompts.py > prompts_dump.txt

It walks src/**/*.py and extracts every module-level triple-quoted string
whose variable name contains PROMPT, TEMPLATE, INSTRUCTION, or SYSTEM,
printing each with its file, variable name, and a separator. Nothing is
reformatted; upload prompts_dump.txt and the relevant blocks go into the
thesis appendix unchanged.

Also flags f-strings and .format() templates so placeholders are visible.
"""
import re
import sys
from pathlib import Path

PATTERN = re.compile(
    r'^(?P<name>[A-Z_]*(?:PROMPT|TEMPLATE|INSTRUCTION|SYSTEM)[A-Z_0-9]*)\s*'
    r'(?::\s*[^=]+)?=\s*(?P<f>f?)(?P<q>"""|\'\'\')(?P<body>.*?)(?P=q)',
    re.MULTILINE | re.DOTALL,
)

def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "src")
    if not root.exists():
        sys.exit(f"{root} not found; run from the repo root or pass the path")
    found = 0
    for py in sorted(root.rglob("*.py")):
        if "__pycache__" in str(py):
            continue
        text = py.read_text(errors="replace")
        for m in PATTERN.finditer(text):
            found += 1
            kind = "f-string" if m.group("f") else (
                "format-template" if "{" in m.group("body") else "plain")
            print("=" * 72)
            print(f"FILE: {py}")
            print(f"NAME: {m.group('name')}   ({kind})")
            print("=" * 72)
            print(m.group("body").strip("\n"))
            print()
    print(f"# {found} prompt literal(s) found", file=sys.stderr)
    if found == 0:
        print("# none matched; prompts may be built inline -- grep for "
              "'messages=' or 'content':", file=sys.stderr)

if __name__ == "__main__":
    main()
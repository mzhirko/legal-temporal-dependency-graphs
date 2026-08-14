#!/usr/bin/env python3
"""verify_facts.py -- audit the gold timeliness annotation chain.

Three checks, all must pass or nothing is written:

1. QUOTES  every quote in FACTS.md attributed to a source txt file is a
   byte-exact substring of that file (curly apostrophes and all). Covers
   both formats:
     quote (2026_EAT_64.txt L69): "..."
     supporting (party-identified, L101): "..."   <- resolves file from
                                                     the enclosing block
2. FIELDS  every machine field in ground_truth_gold.json (anchor_date,
   deadline_stated, acas_day_a/b, presented_date, verdict) matches the
   value recited in the corresponding FACTS.md case block. Patterns match
   mid-line (2025_EAT_155's "acas_day_a: ... / presented: ..." style).
3. OUTPUT  regenerates data/ground_truth/verified_quotes.json (audit store).

Exit 0 on full pass, 1 on any failure (with a listing). Run from repo root:
    python src/verify_facts.py
"""
import glob
import json
import os
import re
import sys

GT = "data/ground_truth/ground_truth_gold.json"
FACTS = "data/ground_truth/FACTS.md"
OUT = "data/ground_truth/verified_quotes.json"

FIELD_PATTERNS = {
    "anchor_date": r"anchor[^:\n]*:\s*(\d{4}-\d{2}-\d{2})",
    "deadline_stated": r"deadline[^:\n]*:\s*(\d{4}-\d{2}-\d{2})",
    "acas_day_a": r"acas_day_a[^:\n]*:\s*(\d{4}-\d{2}-\d{2}|null)",
    "acas_day_b": r"acas_day_b[^:\n]*:\s*(\d{4}-\d{2}-\d{2}|null)",
    "presented_date": r"presented:\s*(\d{4}-\d{2}-\d{2})",
    "verdict": r"verdict:?\s*(in_time|out_of_time)",
}

# quote with explicit filename:  quote (2026_EAT_64.txt L69): "..."
RE_QUOTE_NAMED = re.compile(r"\((\S+\.txt)[^)]*\):\s*\u0022([^\u0022]+)\u0022")
# supporting quote without filename:  supporting (..., L101): "..."
RE_QUOTE_SUPPORTING = re.compile(r"supporting\s*\(([^)]*)\):\s*\u0022([^\u0022]+)\u0022")
RE_BLOCK = re.compile(
    r"### case: (\S+).*?\n(.*?)(?=\n### case:|\n## ====|\Z)", re.S
)
RE_BLOCK_FILE = re.compile(r"^file:\s*(\S+)", re.M)


def load_sources():
    srcs = {}
    for p in glob.glob("data/caselaw/*/txt/*.txt"):
        srcs.setdefault(os.path.basename(p), []).append(p)
    return srcs


def check_quotes(facts, srcs):
    entries, misses = [], []
    seen = set()
    blocks = RE_BLOCK.findall(facts)

    def add(fname, quote, kind):
        key = (fname, quote)
        if key in seen:
            return
        seen.add(key)
        hits = [
            p
            for p in srcs.get(fname, [])
            if quote in open(p, encoding="utf-8").read()
        ]
        entries.append(
            {"file": fname, "kind": kind, "quote": quote,
             "verified": bool(hits), "paths": sorted(hits)}
        )
        if not hits:
            misses.append(f"{fname}: \u0022{quote[:60]}...\u0022")

    for _case, block in blocks:
        fm = RE_BLOCK_FILE.search(block)
        block_file = os.path.basename(fm.group(1)) if fm else None
        for fname, quote in RE_QUOTE_NAMED.findall(block):
            add(fname, quote, "primary")
        for _label, quote in RE_QUOTE_SUPPORTING.findall(block):
            if block_file:
                add(block_file, quote, "supporting")
            else:
                misses.append(f"supporting quote with no file: line in block")
    return entries, misses


def check_fields(gold, facts):
    blocks = dict(RE_BLOCK.findall(facts))
    bad = []
    for case, row in gold["cases"].items():
        block = blocks.get(case)
        if block is None:
            bad.append(f"{case}: not found in FACTS.md")
            continue
        for field, pat in FIELD_PATTERNS.items():
            m = re.search(pat, block)
            facts_val = (
                (None if m.group(1) == "null" else m.group(1)) if m else None
            )
            gold_val = row.get(field)
            if m and facts_val != gold_val:
                bad.append(
                    f"{case}.{field}: gold={gold_val!r} vs FACTS={facts_val!r}"
                )
            elif not m and gold_val is not None:
                bad.append(
                    f"{case}.{field}: gold={gold_val!r} but absent in FACTS block"
                )
    return bad


def main():
    gold = json.load(open(GT))
    facts = open(FACTS, encoding="utf-8").read()
    srcs = load_sources()

    entries, quote_misses = check_quotes(facts, srcs)
    field_bad = check_fields(gold, facts)

    if quote_misses or field_bad:
        print("FAILED \u2014 verified_quotes.json NOT written:")
        for x in field_bad:
            print("  [field]", x)
        for x in quote_misses:
            print("  [quote]", x)
        return 1

    n_primary = sum(1 for e in entries if e["kind"] == "primary")
    n_supporting = sum(1 for e in entries if e["kind"] == "supporting")
    json.dump(
        {"verified": len(entries), "primary": n_primary,
         "supporting": n_supporting, "entries": entries},
        open(OUT, "w"), indent=1, ensure_ascii=False,
    )
    print(
        f"PASS: {len(gold['cases'])} gold cases cross-checked; "
        f"{len(entries)}/{len(entries)} quotes byte-verified "
        f"({n_primary} primary + {n_supporting} supporting). "
        f"{OUT} regenerated."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
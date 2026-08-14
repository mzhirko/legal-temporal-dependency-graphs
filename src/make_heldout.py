#!/usr/bin/env python3
"""Generate a held-out contradiction set mechanically, so that NOBODY --
neither the person who wrote the linker fixes nor the experimenter --
chooses or sees the planted edits before scoring.

Given source .txt documents, this script:
  1. finds candidate value expressions (digit periods in EN/NL units,
     day-month-year dates, ISO dates) with a declared regex set;
  2. picks n of them per document with a seeded RNG and applies a
     mechanical substitution (periods doubled or halved; dates shifted by
     a seeded 8-45 day offset, same surface format);
  3. writes <doc>_mod.txt next to the original and a ground-truth JSON in
     the schema evaluate_contradictions.py already reads.

Blind protocol: run this, do NOT open the _mod files or the JSON, extract
TDGs, score once with evaluate_contradictions.py, and only then read
anything. The seed printed at the end is the full provenance of the set.

Declared limitations (report them, don't hide them): spelled-out numbers
("drie maanden", "six weeks") are not candidates -- only digit periods and
recognisable dates; a document with fewer candidates than requested gets
as many as it has.

Usage:
  python3 make_heldout.py --seed 41 --plants 2 \
      --out-gt heldout_ground_truth.json  doc1.txt doc2.txt ...
"""
from __future__ import annotations

import argparse
import calendar
import json
import random
import re
import sys
from datetime import date, timedelta
from pathlib import Path

# Declared candidate grammar. Units cover the two working languages of the
# source corpus; extending to another language means adding its unit words
# here -- data, not logic.
_UNITS = r"(?:dagen|weken|maanden|jaar|jaren|days?|weeks?|months?|years?)"
_PERIOD = re.compile(r"\b(\d{1,3})\s*(" + _UNITS + r")\b", re.IGNORECASE)
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], 1)}
_MONTHS.update({m: i for i, m in enumerate(
    ["januari", "februari", "maart", "april", "mei", "juni", "juli",
     "augustus", "september", "oktober", "november", "december"], 1)})
_DMY = re.compile(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b")
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def _period_days(n: int, unit: str) -> int:
    u = unit.lower()
    if u.startswith(("dag", "day")):
        return n
    if u.startswith(("week", "wek")):
        return n * 7
    if u.startswith(("maand", "month")):
        return n * 30
    return n * 365


def _iso_period(n: int, unit: str) -> str:
    u = unit.lower()
    if u.startswith(("dag", "day")):
        return f"P{n}D"
    if u.startswith(("week", "wek")):
        return f"P{n}W"
    if u.startswith(("maand", "month")):
        return f"P{n}M"
    return f"P{n}Y"


def _candidates(text: str):
    out = []
    for m in _PERIOD.finditer(text):
        out.append(("period", m))
    for m in _DMY.finditer(text):
        if m.group(2).lower() in _MONTHS:
            out.append(("dmy", m))
    for m in _ISO.finditer(text):
        out.append(("iso", m))
    # one candidate per distinct surface string, position-ordered, so a
    # substitution rewrites every occurrence consistently (a real
    # conflicting version of a document is consistent with itself)
    seen, uniq = set(), []
    for kind, m in sorted(out, key=lambda km: km[1].start()):
        if m.group(0) not in seen:
            seen.add(m.group(0))
            uniq.append((kind, m))
    return uniq


def _substitute(kind: str, m: re.Match, rng: random.Random):
    """Return (new_surface, value_a, value_b, concept_hint)."""
    if kind == "period":
        n, unit = int(m.group(1)), m.group(2)
        new_n = max(1, n * 2 if rng.random() < 0.5 else max(1, n // 2))
        if new_n == n:
            new_n = n + max(1, n)
        return (m.group(0).replace(m.group(1), str(new_n), 1),
                _iso_period(n, unit), _iso_period(new_n, unit),
                f"stated period ({n} -> {new_n} {unit})")
    shift = rng.choice([-1, 1]) * rng.randint(8, 45)
    if kind == "dmy":
        d0 = date(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))
        d1 = d0 + timedelta(days=shift)
        # keep the source language's month word by mapping through index
        same_lang = [w for w, i in _MONTHS.items() if i == d1.month
                     and (w in ("mei", "maart", "januari", "februari", "juni",
                                "juli", "augustus", "oktober"))
                     == (m.group(2).lower() in ("mei", "maart", "januari",
                                                "februari", "juni", "juli",
                                                "augustus", "oktober"))]
        word = same_lang[0] if same_lang else calendar.month_name[d1.month].lower()
        surface = f"{d1.day} {word.capitalize() if m.group(2)[0].isupper() else word} {d1.year}"
        return surface, d0.isoformat(), d1.isoformat(), f"date ({d0} -> {d1})"
    d0 = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    d1 = d0 + timedelta(days=shift)
    return d1.isoformat(), d0.isoformat(), d1.isoformat(), f"date ({d0} -> {d1})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("docs", nargs="+", help="source .txt files")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--plants", type=int, default=2, help="edits per document")
    ap.add_argument("--out-gt", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gt = {
        "description": (
            f"MECHANICALLY GENERATED held-out set (make_heldout.py, seed "
            f"{args.seed}, {args.plants} plants/doc). No human chose or saw "
            "the edits before scoring. Candidate grammar and limitations are "
            "declared in the generator script, which is committed."),
        "expected_contradictions": [],
        "negative_control_pairs": [],
    }
    ids = []
    for path in args.docs:
        p = Path(path)
        doc_id = p.stem
        ids.append(doc_id)
        text = p.read_text()
        cands = _candidates(text)
        if not cands:
            print(f"{doc_id}: no candidates found -- skipped", file=sys.stderr)
            continue
        picks = rng.sample(cands, min(args.plants, len(cands)))
        mod = text
        for kind, m in picks:
            old_surface = m.group(0)
            new_surface, va, vb, concept = _substitute(kind, m, rng)
            mod = mod.replace(old_surface, new_surface)
            start = max(0, m.start() - 40)
            gt["expected_contradictions"].append({
                "doc_a": doc_id, "doc_b": f"{doc_id}_mod",
                "concept": concept, "value_a": va, "value_b": vb,
                "clause_a": text[start:m.end() + 20].strip(),
                "clause_b": (text[start:m.start()] + new_surface
                             + text[m.end():m.end() + 20]).strip(),
                "plant_id": f"{doc_id}_{kind}_{m.start()}",
            })
        (p.parent / f"{doc_id}_mod.txt").write_text(mod)
        print(f"{doc_id}: {len(picks)} plant(s) applied -> {doc_id}_mod.txt")
    for i in range(len(ids) - 1):
        gt["negative_control_pairs"].append(
            {"doc_a": ids[i], "doc_b": ids[i + 1],
             "expectation": "zero disputes between different documents"})
    Path(args.out_gt).write_text(json.dumps(gt, indent=1, ensure_ascii=False))
    print(f"ground truth -> {args.out_gt}  (seed {args.seed}; "
          "do not open before scoring)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Honest, overlap-aware reporting for the TDG-vs-Catala comparison.

Why this exists
---------------
The per-document `match_rate` is `matches / (matches + mismatches)`. It ignores
the fields only one side produced (`tdg_only`, `catala_only`), so a document
where the two systems agree on a single field and never even line up on the
rest still scores 1.0. That number flatters a comparison that is mostly
non-overlapping. This module reports the two things that actually matter,
separately and never collapsed into one figure:

  * COVERAGE       -- of all the temporal fields either system produced, what
                     fraction did BOTH systems produce (i.e. were comparable)?
  * AGREEMENT      -- within that overlap, what fraction agree (exact or
                     semantic), as opposed to disagree?

It also counts the documents where Catala never produced a result
(`interpret_error`, `repair_failed`, typecheck failures, ...) as FAILURES of
the Catala arm, rather than silently dropping them -- excluding them would
flatter the method the same way `match_rate` does.

Usage
-----
  python comparator/honest_report.py ../data/experiments/comparison_50
"""
from __future__ import annotations

import json
import os
import sys
import glob
from collections import Counter

# Status groupings (kept in one place so they cannot drift between scripts).
AGREE = {"match", "off_by_one", "semantic_match", "value_match", "duration_match"}
DISAGREE = {"mismatch", "semantic_mismatch", "duration_mismatch"}
TDG_ONLY = {"tdg_only"}
CATALA_ONLY = {"catala_only"}
OTHER = {"type_mismatch", "placeholder"}
SUCCESS_STATUS = "success"


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.0%})" if d else f"{n}/0 (n/a)"


def load_reports(path: str) -> list[dict]:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*_comparison.json")))
    else:
        files = [path]
    return [json.load(open(f)) for f in files]


def summarise(reports: list[dict]) -> dict:
    runnable, failed = [], []
    fail_reasons = Counter()
    for r in reports:
        if r.get("catala_status") == SUCCESS_STATUS:
            runnable.append(r)
        else:
            failed.append(r)
            fail_reasons[r.get("catala_status", "unknown")] += 1

    buckets = Counter()
    exact = semantic = 0
    for r in runnable:
        for f in r.get("fields", []):
            s = f.get("status")
            buckets[s] += 1
            if s in ("match", "off_by_one", "value_match"):
                exact += 1
            elif s == "semantic_match":
                semantic += 1

    agree = sum(buckets[s] for s in AGREE)
    disagree = sum(buckets[s] for s in DISAGREE)
    tdg_only = sum(buckets[s] for s in TDG_ONLY)
    catala_only = sum(buckets[s] for s in CATALA_ONLY)
    other = sum(buckets[s] for s in OTHER)
    overlap = agree + disagree
    total = overlap + tdg_only + catala_only + other

    return {
        "docs_total": len(reports),
        "docs_runnable": len(runnable),
        "docs_failed": len(failed),
        "fail_reasons": dict(fail_reasons),
        "fields_total": total,
        "overlap": overlap,
        "agree": agree,
        "agree_exact": exact,
        "agree_semantic": semantic,
        "disagree": disagree,
        "tdg_only": tdg_only,
        "catala_only": catala_only,
        "other": other,
        "buckets": dict(buckets),
    }


def print_report(s: dict) -> None:
    print("\n=== TDG vs Catala -- honest comparison report ===\n")
    print("Documents")
    print(f"  total            : {s['docs_total']}")
    print(f"  Catala ran       : {s['docs_runnable']}")
    print(f"  Catala FAILED    : {s['docs_failed']}  {s['fail_reasons'] or ''}")
    print("  (failures are counted as failures of the Catala arm, not excluded.)\n")

    print("Fields (over documents where Catala ran)")
    print(f"  total fields either system produced : {s['fields_total']}")
    print(f"  COVERAGE  (modelled by BOTH)        : "
          f"{_pct(s['overlap'], s['fields_total'])}")
    print(f"  only TDG produced                   : {s['tdg_only']}")
    print(f"  only Catala produced                : {s['catala_only']}")
    if s["other"]:
        print(f"  uncomparable (type/placeholder)     : {s['other']}")
    print()

    print("Agreement (WITHIN the overlap only -- not over all fields)")
    print(f"  agree    : {_pct(s['agree'], s['overlap'])}   "
          f"(exact={s['agree_exact']}, semantic={s['agree_semantic']})")
    print(f"  disagree : {_pct(s['disagree'], s['overlap'])}")
    print()

    cov = s["overlap"] / s["fields_total"] if s["fields_total"] else 0
    agr = s["agree"] / s["overlap"] if s["overlap"] else 0
    print("Headline (report BOTH, never a single number):")
    print(f"  coverage {cov:.0%}  ·  within-overlap agreement {agr:.0%}  "
          f"·  {s['docs_failed']} Catala failures")
    print("\nReading: the two methods are directly comparable on only the "
          "covered fields; high agreement there means they concur where they "
          "overlap, while low coverage means they largely model different "
          "temporal facts (complementary, not redundant).")
    print(f"\n(status buckets: {s['buckets']})")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    reports = load_reports(sys.argv[1])
    if not reports:
        print("No comparison files found.")
        return 1
    print_report(summarise(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
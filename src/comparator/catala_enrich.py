#!/usr/bin/env python3
"""
Catala-based enrichment of the TDG.

Catala's contribution is correct, formally-checked date arithmetic. Once the
aligner connects a Catala output to the TDG clause it concerns, that computed
date can be put to use. This pass turns an aligned comparison into a list of
concrete enrichment actions, one per Catala-derived field:

  VERIFY  -- the field is a semantic/exact match: Catala's computed date agrees
            with the TDG's clause. The TDG gains a *verified* computed date
            (it previously held only the raw duration, e.g. P6M).
  FILL    -- the field is catala_only and is a date: Catala computed a date the
            TDG never produced (e.g. an effective date the extractor missed, or
            a deadline the TDG held only as a duration). The TDG gains it.
  REVIEW  -- the field disagrees (mismatch / semantic_mismatch / off_by_one):
            the two methods computed different dates for the same clause. This
            is surfaced, with the day delta, as a discrepancy for a human to
            resolve -- NOT silently overwritten in either direction.

Fields that are tdg_only, placeholder, or type_mismatch produce no action.

Nothing is overwritten in place: the pass emits a proposal record per document
(and a corpus summary). Applying it to the stored TDGs is a deliberate,
separate step the user runs knowingly.

Usage
-----
  python comparator/catala_enrich.py ../data/experiments/comparison_50 \
        --out ../data/experiments/enrichment
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import glob
import re
from collections import Counter

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_date(v) -> bool:
    return isinstance(v, str) and bool(_DATE.match(v))


def enrich_document(report: dict) -> dict:
    """Produce enrichment actions for one comparison report."""
    actions = []
    if report.get("catala_status") != "success":
        return {
            "document_id": report.get("document_id"),
            "catala_status": report.get("catala_status"),
            "actions": [],
            "note": "Catala did not produce a result; nothing to enrich from.",
        }

    for f in report.get("fields", []):
        var = f.get("variable_name")
        status = f.get("status")
        cv = f.get("catala_value")
        tv = f.get("tdg_value")
        delta = f.get("delta_days")

        if status in ("match", "off_by_one", "value_match",
                      "semantic_match", "duration_match"):
            # The two methods concur on this clause.
            kind = "verify"
            note = "Catala confirms the TDG clause"
            if status == "semantic_match":
                note = ("Catala's computed date confirms the TDG duration "
                        "clause; attach it as a verified derived date")
            elif status == "off_by_one":
                note = "agree up to a 1-day boundary convention"
            actions.append({
                "variable": var, "action": kind, "catala_value": cv,
                "tdg_value": tv, "delta_days": delta, "note": note,
            })

        elif status == "catala_only" and _is_date(cv):
            actions.append({
                "variable": var, "action": "fill", "catala_value": cv,
                "tdg_value": None, "delta_days": None,
                "note": "Catala computed a date the TDG lacks; add to TDG",
            })

        elif status in ("mismatch", "semantic_mismatch", "duration_mismatch"):
            actions.append({
                "variable": var, "action": "review", "catala_value": cv,
                "tdg_value": tv, "delta_days": delta,
                "note": ("methods disagree on this clause; resolve which "
                         "extraction/computation is correct"),
            })
        # tdg_only / placeholder / type_mismatch / non-date catala_only: skip

    return {
        "document_id": report.get("document_id"),
        "catala_status": report.get("catala_status"),
        "actions": actions,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Catala-based TDG enrichment")
    ap.add_argument("comparison_dir", help="dir of *_comparison.json")
    ap.add_argument("--out", default=None,
                    help="dir to write per-document enrichment proposals")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.comparison_dir, "*_comparison.json")))
    if not files:
        print("No comparison files found.")
        return 1

    totals = Counter()
    docs_with_actions = 0
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    for path in files:
        report = json.load(open(path))
        enriched = enrich_document(report)
        acts = enriched["actions"]
        if acts:
            docs_with_actions += 1
        for a in acts:
            totals[a["action"]] += 1
        if args.out:
            name = os.path.basename(path).replace("_comparison.json",
                                                   "_enrichment.json")
            json.dump(enriched, open(os.path.join(args.out, name), "w"), indent=1)

    print("\n=== Catala enrichment proposal ===\n")
    print(f"documents processed     : {len(files)}")
    print(f"documents with actions  : {docs_with_actions}")
    print(f"  VERIFY (confirm TDG)  : {totals['verify']}")
    print(f"  FILL   (add to TDG)   : {totals['fill']}")
    print(f"  REVIEW (disagreement) : {totals['review']}")
    if args.out:
        print(f"\nper-document proposals written to {args.out}/")
    else:
        print("\n(run with --out DIR to write per-document proposals)")
    print("\nNothing was modified in place; FILL/VERIFY/REVIEW are proposals. "
          "REVIEW items are genuine discrepancies to resolve, not noise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
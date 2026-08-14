#!/usr/bin/env python3
"""
Apply Catala enrichment proposals to the TDG JSON files.

Takes the per-document proposals from catala_enrich.py and writes Catala's
verified computed dates into the corresponding TDG, so the TDG gains dates it
either lacked (FILL) or held only as a raw duration, now resolved (VERIFY).

Safety properties (this script edits your data, so they matter):
  * REVIEW actions are NEVER applied. Those are disagreements between the two
    methods and must be resolved by a human, not auto-written.
  * Nothing is destructive or hidden. Every change is marked so it can be
    found and undone:
      - an APPENDED fact carries  "source": "catala_enrichment"  plus the
        originating Catala variable and whether it was a verify or a fill;
      - an EXISTING fact that Catala confirms is annotated in place with
        "catala_verified": true (its extracted value is left untouched).
    To revert, drop facts whose source is "catala_enrichment" and remove the
    "catala_verified"/"catala_variable" annotations.
  * Originals are not overwritten unless you pass --in-place (which first
    writes a .bak). By default, enriched copies go to a separate directory.
  * Idempotent: re-running does not duplicate facts or re-annotate.

Usage
-----
  # preview only, no files written:
  python comparator/apply_enrichment.py \
      --enrichment-dir ../data/experiments/enrichment \
      --tdg-dir        ../data/results_contracts_50 \
      --dry-run

  # write enriched copies to a new directory (recommended):
  python comparator/apply_enrichment.py \
      --enrichment-dir ../data/experiments/enrichment \
      --tdg-dir        ../data/results_contracts_50 \
      --out-dir        ../data/results_contracts_50_enriched

  # edit the TDGs in place (writes .bak first):
  python comparator/apply_enrichment.py ... --in-place
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import glob
import shutil
from collections import Counter

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SOURCE_TAG = "catala_enrichment"


def _is_date(v) -> bool:
    return isinstance(v, str) and bool(_DATE.match(v))


def _has_date(facts: list, iso: str) -> dict | None:
    """Return an existing fact carrying this exact date value, if any."""
    for f in facts:
        if f.get("value") == iso or f.get("date_parsed") == iso:
            return f
    return None


def _make_fact(var: str, iso: str, verified: bool) -> dict:
    """A new, clearly-Catala-derived date fact.

    Includes the keys the TDG loader reads (id, entity, role, value, raw_text,
    timex_type, sentence, confidence, date_parsed) plus provenance keys the
    loader ignores but which keep the fact auditable and reversible.
    """
    kind = "verified by" if verified else "filled by"
    return {
        "id": f"catala_{var}",
        "entity": var.replace("_", " "),
        "role": "UNKNOWN",
        "value": iso,
        "raw_text": iso,
        "timex_type": "DATE",
        "sentence": f"[Catala-derived: {kind} the Catala scope] {var} = {iso}",
        "confidence": 1.0,
        "date_parsed": iso,
        # provenance (ignored by the loader, preserved in the JSON):
        "source": SOURCE_TAG,
        "catala_variable": var,
        "catala_verified": bool(verified),
    }


def apply_to_tdg(tdg: dict, actions: list[dict], counts: Counter) -> bool:
    """Apply FILL/VERIFY actions to one TDG dict. Returns True if it changed."""
    facts = tdg.setdefault("facts", [])
    existing_ids = {f.get("id") for f in facts}
    changed = False

    for a in actions:
        act = a.get("action")
        var = a.get("variable")
        cv = a.get("catala_value")

        if act == "review":
            counts["review_skipped"] += 1
            continue
        if act not in ("fill", "verify") or not _is_date(cv):
            counts["other_skipped"] += 1
            continue

        match = _has_date(facts, cv)
        if match is not None:
            # Catala agrees with a date the TDG already has: annotate in place.
            if not match.get("catala_verified"):
                match["catala_verified"] = True
                match["catala_variable"] = var
                counts["verified_annotated"] += 1
                changed = True
            else:
                counts["already_done"] += 1
            continue

        # No existing date for this clause: append a Catala-derived fact.
        fid = f"catala_{var}"
        if fid in existing_ids:
            counts["already_done"] += 1
            continue
        facts.append(_make_fact(var, cv, verified=(act == "verify")))
        existing_ids.add(fid)
        counts["fact_added"] += 1
        changed = True

    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply Catala enrichment to TDGs")
    ap.add_argument("--enrichment-dir", required=True)
    ap.add_argument("--tdg-dir", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="write enriched copies here (default: <tdg-dir>_enriched)")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite the TDGs in place (writes a .bak first)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change; write nothing")
    args = ap.parse_args()

    enrich_files = sorted(glob.glob(os.path.join(args.enrichment_dir,
                                                 "*_enrichment.json")))
    if not enrich_files:
        print("No enrichment files found.")
        return 1

    if not args.in_place and not args.dry_run:
        out_dir = args.out_dir or (args.tdg_dir.rstrip("/") + "_enriched")
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = None

    counts = Counter()
    docs_changed = 0
    for ef in enrich_files:
        proposal = json.load(open(ef))
        doc_id = proposal.get("document_id")
        actions = proposal.get("actions", [])
        if not actions:
            continue
        tdg_path = os.path.join(args.tdg_dir, f"{doc_id}.json")
        if not os.path.exists(tdg_path):
            print(f"  ! no TDG for {doc_id}, skipping")
            counts["tdg_missing"] += 1
            continue

        tdg = json.load(open(tdg_path))
        changed = apply_to_tdg(tdg, actions, counts)
        if changed:
            docs_changed += 1
            if args.dry_run:
                continue
            if args.in_place:
                if not os.path.exists(tdg_path + ".bak"):
                    shutil.copy(tdg_path, tdg_path + ".bak")
                json.dump(tdg, open(tdg_path, "w"), indent=1)
            else:
                json.dump(tdg, open(os.path.join(out_dir, f"{doc_id}.json"), "w"),
                          indent=1)

    print("\n=== Apply Catala enrichment ===\n")
    print(f"documents changed         : {docs_changed}")
    print(f"  facts added (fill/sem.) : {counts['fact_added']}")
    print(f"  existing facts verified : {counts['verified_annotated']}")
    print(f"  REVIEW left untouched   : {counts['review_skipped']}")
    if counts["already_done"]:
        print(f"  already applied (idemp.): {counts['already_done']}")
    if counts["other_skipped"]:
        print(f"  non-date/other skipped  : {counts['other_skipped']}")
    if counts["tdg_missing"]:
        print(f"  TDG file missing        : {counts['tdg_missing']}")
    if args.dry_run:
        print("\n[dry run] no files written.")
    elif args.in_place:
        print(f"\nedited TDGs in place under {args.tdg_dir} (.bak backups written).")
    else:
        print(f"\nenriched copies written to {out_dir}/ (originals untouched).")
    print("\nAdded facts are tagged source=\"catala_enrichment\"; verified facts "
          "carry catala_verified=true. REVIEW disagreements were never applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
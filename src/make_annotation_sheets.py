#!/usr/bin/env python3
"""
make_annotation_sheets.py -- build human-verification CSV sheets from TDG data.

Annotators JUDGE atomic, quote-anchored claims (Y/N/Unsure). They never
construct the graph. This script decomposes the pipeline's own output into
those rows, so the gold set verifies exactly what the system produced.

Produces, per run:
  <out>/<doc>__A_facts.csv          one row per fact      (Sheet A)
  <out>/<doc>__B_dependencies.csv   one row per edge      (Sheet B)
  <out>/crossdoc__C_links.csv       one row per link      (Sheet C, optional)

Facts policy: EVERYTHING is emitted; rows whose fact has no concrete value
are kept but FLAGGED (flag column) so annotators/you can see them. Non-value
facts are relative references the system cannot resolve to a date -- keeping
them visible is itself informative.

Usage:
  # facts + dependencies for every TDG in a directory
  python make_annotation_sheets.py --tdg-dir ../data/results_uk --out ../data/annotation

  # single file
  python make_annotation_sheets.py --tdg ../data/results_uk/et_37a93baf.json \
      --out ../data/annotation

  # add the cross-doc sheet from a saved linker output (list of CrossDocLink dicts)
  python make_annotation_sheets.py --tdg-dir ../data/results_uk \
      --crossdoc-links ../data/experiments/crossdoc_links.json \
      --out ../data/annotation

Stdlib only -- no pandas, no Ollama, runs anywhere.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Column headers (match the annotation guideline sheets)
# ---------------------------------------------------------------------------

FACTS_HEADER = [
    "row_id", "document_id", "fact_id",
    "quoted_sentence", "extracted_concept", "type", "our_value", "flag",
    "correct (Y/N/Unsure)", "corrected_value", "comment",
]

DEPS_HEADER = [
    "row_id", "document_id", "dependency",
    "claim_to_check", "based_on_sentences", "our_relation",
    "correct (Y/N/Unsure)", "comment",
]

CROSSDOC_HEADER = [
    "row_id", "link_type", "claim_to_check",
    "doc_A", "doc_A_says", "doc_B", "doc_B_says",
    "same_real_event (Y/N/Unsure)", "values_agree (Y/N/N-A)", "comment",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(s) -> str:
    """One-line, trimmed text safe for a CSV cell."""
    if s is None:
        return ""
    return " ".join(str(s).split()).strip()


def _looks_like_date(value, date_parsed) -> bool:
    if date_parsed:
        return True
    v = str(value or "")
    return len(v) >= 8 and v[:4].isdigit() and v.count("-") == 2


def _looks_like_duration(value, role) -> bool:
    v = str(value or "")
    return (role or "").upper() == "DURATION" or (v[:1] == "P" and any(c.isdigit() for c in v))


def _fact_type(fact) -> str:
    v = fact.get("value")
    if _looks_like_date(v, fact.get("date_parsed")):
        return "Date"
    if _looks_like_duration(v, fact.get("role")):
        return "Time period"
    return "--"


def _has_value(fact) -> bool:
    v = fact.get("value")
    return bool(v) or bool(fact.get("date_parsed"))


def _index_facts(tdg: dict) -> dict:
    return {f.get("id"): f for f in tdg.get("facts", [])}


def _fact_label(fact: dict | None, fact_id: str) -> str:
    """Readable name for a fact, falling back to its id."""
    if not fact:
        return fact_id
    ent = _clean(fact.get("entity")) or fact_id
    return ent


def _render_relation(dep: dict, facts_by_id: dict) -> str:
    """Turn an edge into a plain-English claim a layperson can judge."""
    frm = facts_by_id.get(dep.get("from_id"))
    to = facts_by_id.get(dep.get("to_id"))
    a = _fact_label(frm, dep.get("from_id"))
    b = _fact_label(to, dep.get("to_id"))
    delta = dep.get("delta_days")
    ctype = (dep.get("constraint_type") or "").lower()

    if delta not in (None, 0):
        sign = "after" if delta > 0 else "before"
        return f'"{b}" is {abs(delta)} days {sign} "{a}".'
    if ctype in ("ordering", "before", "after"):
        return f'"{b}" happens after "{a}".'
    if ctype == "additive":
        return f'"{b}" is computed from "{a}".'
    return f'"{b}" depends on "{a}" ({ctype or "related"}).'


# ---------------------------------------------------------------------------
# Sheet builders
# ---------------------------------------------------------------------------

def build_facts_rows(tdg: dict) -> list[list]:
    doc = tdg.get("document_id", "?")
    rows = []
    for i, f in enumerate(tdg.get("facts", []), start=1):
        has_val = _has_value(f)
        flag = "" if has_val else "NO_VALUE (relative ref -- not machine-resolvable)"
        rows.append([
            f"{doc}-A{i:02d}",
            doc,
            _clean(f.get("id")),
            _clean(f.get("sentence")),
            _clean(f.get("entity")),
            _fact_type(f),
            _clean(f.get("value")) or _clean(f.get("date_parsed")),
            flag,
            "",   # correct
            "",   # corrected_value
            "",   # comment
        ])
    return rows


def build_deps_rows(tdg: dict) -> list[list]:
    doc = tdg.get("document_id", "?")
    facts_by_id = _index_facts(tdg)
    rows = []
    for i, d in enumerate(tdg.get("dependencies", []), start=1):
        frm = facts_by_id.get(d.get("from_id"))
        to = facts_by_id.get(d.get("to_id"))
        based_on = " + ".join(filter(None, [
            f'"{_clean((frm or {}).get("sentence"))}"' if frm else "",
            f'"{_clean((to or {}).get("sentence"))}"' if to else "",
        ]))
        our_rel = _clean(d.get("constraint_expr")) or _clean(d.get("constraint_type"))
        rows.append([
            f"{doc}-B{i:02d}",
            doc,
            f'{d.get("from_id")} -> {d.get("to_id")}',
            _render_relation(d, facts_by_id),
            based_on,
            our_rel,
            "",   # correct
            "",   # comment
        ])
    return rows


def build_crossdoc_rows(links: list[dict], tdgs_by_doc: dict) -> list[list]:
    rows = []
    for i, lk in enumerate(links, start=1):
        fa_doc, fb_doc = lk.get("from_doc"), lk.get("to_doc")
        fa = _index_facts(tdgs_by_doc.get(fa_doc, {})).get(lk.get("from_fact"))
        fb = _index_facts(tdgs_by_doc.get(fb_doc, {})).get(lk.get("to_fact"))

        def says(fact, fallback_val):
            if fact:
                sent = _clean(fact.get("sentence"))
                val = _clean(fact.get("value")) or _clean(fact.get("date_parsed"))
                return f'"{sent}"' + (f"  [value: {val}]" if val else "")
            return _clean(fallback_val)

        ltype = _clean(lk.get("link_type"))
        a_label = _fact_label(fa, lk.get("from_fact"))
        b_label = _fact_label(fb, lk.get("to_fact"))
        if ltype == "coreference":
            claim = f'"{a_label}" in {fa_doc} is the SAME real-world event as "{b_label}" in {fb_doc}.'
        elif ltype == "contradiction":
            claim = f'"{a_label}" in {fa_doc} and "{b_label}" in {fb_doc} describe the same thing but give different values.'
        else:
            claim = f'"{a_label}" in {fa_doc} relates to "{b_label}" in {fb_doc} ({ltype}).'

        rows.append([
            f"X{i:02d}",
            ltype,
            claim,
            fa_doc, says(fa, lk.get("value_a")),
            fb_doc, says(fb, lk.get("value_b")),
            "",   # same real event
            "",   # values agree
            "",   # comment
        ])
    return rows


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _load_tdg(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate annotation CSV sheets from TDG data.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tdg-dir", help="directory of TDG .json files")
    src.add_argument("--tdg", help="single TDG .json file")
    ap.add_argument("--crossdoc-links", help="JSON list of CrossDocLink dicts (optional)")
    ap.add_argument("--out", required=True, help="output directory for the CSVs")
    args = ap.parse_args()

    out = Path(args.out)

    # collect TDG files
    if args.tdg:
        files = [args.tdg]
    else:
        files = sorted(glob.glob(os.path.join(args.tdg_dir, "*.json")))
    if not files:
        raise SystemExit(f"No TDG .json files found.")

    tdgs_by_doc = {}
    total_facts = total_deps = total_null = 0
    for fp in files:
        tdg = _load_tdg(fp)
        doc = tdg.get("document_id") or Path(fp).stem
        tdg["document_id"] = doc
        tdgs_by_doc[doc] = tdg

        facts_rows = build_facts_rows(tdg)
        deps_rows = build_deps_rows(tdg)
        total_facts += len(facts_rows)
        total_deps += len(deps_rows)
        total_null += sum(1 for r in facts_rows if r[7])  # flag column non-empty

        _write_csv(out / f"{doc}__A_facts.csv", FACTS_HEADER, facts_rows)
        _write_csv(out / f"{doc}__B_dependencies.csv", DEPS_HEADER, deps_rows)
        print(f"  {doc}: {len(facts_rows)} facts ({sum(1 for r in facts_rows if r[7])} flagged null), {len(deps_rows)} deps")

    # cross-doc sheet
    if args.crossdoc_links:
        with open(args.crossdoc_links, encoding="utf-8") as fh:
            links = json.load(fh)
        if isinstance(links, dict):  # tolerate {"links": [...]}
            links = links.get("links", [])
        rows = build_crossdoc_rows(links, tdgs_by_doc)
        _write_csv(out / "crossdoc__C_links.csv", CROSSDOC_HEADER, rows)
        print(f"  cross-doc: {len(rows)} links")
    else:
        # emit an empty template so the folder is complete
        _write_csv(out / "crossdoc__C_links.csv", CROSSDOC_HEADER, [])
        print("  cross-doc: no --crossdoc-links given -> wrote empty template")

    print(f"\nDone. {len(files)} document(s) -> {out}")
    print(f"  total facts rows: {total_facts}  (flagged null/relative: {total_null})")
    print(f"  total dependency rows: {total_deps}")


if __name__ == "__main__":
    main()

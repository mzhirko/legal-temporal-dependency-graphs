#!/usr/bin/env python3
"""
Positive contradiction test (Use Case C).

The existing contradiction "tests" all expect ZERO contradictions, so they only
verify the detector stays quiet -- never that it FIRES. This harness uses the
synthetic ground truth (data/synthetic/ground_truth_synthetic.json), which
documents 4 contradictions with known doc pairs and values, and asserts that
CrossDocLinker.find_contradictions() actually finds them.

It is model-free and deterministic: it consumes already-extracted TDG JSONs.
Extract the 4 synthetic documents to TDG JSON first (Gemma pass), then run this.

Docs involved in the 4 contradictions:
  - en_contracts_seed0          (real, from results_contracts_50)
  - synth_steel_ukraine         (synthetic)
  - synth_fisheries_senegal     (synthetic)
  - synth_fisheries_extension   (synthetic)
(seed10 / synth_nuclear_amendment are coreference cases, not contradictions,
 so they are not required here.)

Usage:
  python evaluate_contradictions.py \
      --synth-tdg-dir ../data/results_synthetic \
      --contracts-tdg-dir ../data/results_contracts_50 \
      --ground-truth ../data/synthetic/ground_truth_synthetic.json

Exit codes: 0 = all expected contradictions found; 1 = some missed;
2 = skipped (required TDGs not present yet).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Optional

from tdg_pipeline.cross_doc import CrossDocLinker
from tdg_pipeline.io import load_tdg_dir, DEFAULT_SIMILARITY_THRESHOLD


# Documents that participate in the expected contradictions.
REQUIRED_DOCS = {
    "en_contracts_seed0",
    "synth_steel_ukraine",
    "synth_fisheries_senegal",
    "synth_fisheries_extension",
}


def _to_days(value: str) -> Optional[int]:
    """
    Normalise a ground-truth or fact value to a day count so values from
    different surface forms can be compared.

    Handles: ISO durations ("P6M", "P90D", "P3M"), and natural-language
    "<n> days" / "<n> months" / "<n> years".
    """
    if value is None:
        return None
    v = str(value).strip()

    # ISO 8601 duration (calendar-approx: 30d/month, 365d/year -- only used to
    # compare *distinctness*, not to compute dates, so approximation is fine).
    m = re.fullmatch(r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)W)?(?:(\d+)D)?", v)
    if m and any(m.groups()):
        y, mo, w, d = (int(g) if g else 0 for g in m.groups())
        return y * 365 + mo * 30 + w * 7 + d

    # "<n> days|months|years"
    m = re.fullmatch(r"(\d+)\s*(day|days|month|months|year|years)", v, re.I)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if unit.startswith("day"):
            return n
        if unit.startswith("month"):
            return n * 30
        if unit.startswith("year"):
            return n * 365
    return None


def _norm_value(value) -> Optional[object]:
    """Normalise a value for GT matching: day count for periods, the ISO
    string itself for dates. _to_days covers durations only, so date-valued
    contradictions could never match a ground-truth entry before this."""
    d = _to_days(value)
    if d is not None:
        return d
    v = str(value or "").strip()
    m = re.fullmatch(r"\d{4}-\d{2}-\d{2}", v)
    return v if m else None


def _link_day_pair(link) -> Optional[frozenset]:
    """The {value_a, value_b} of a detected link, normalised."""
    a = _norm_value(link.value_a)
    b = _norm_value(link.value_b)
    if a is None or b is None:
        return None
    return frozenset({a, b})


def _gt_day_pair(gt: dict) -> Optional[frozenset]:
    a = _norm_value(gt.get("value_a"))
    b = _norm_value(gt.get("value_b"))
    if a is None or b is None:
        return None
    return frozenset({a, b})


def main() -> int:
    ap = argparse.ArgumentParser(description="Positive contradiction test (Use Case C)")
    ap.add_argument("--synth-tdg-dir", default="../data/results_synthetic",
                    help="Directory with TDG JSONs for the synthetic documents")
    ap.add_argument("--contracts-tdg-dir", default="../data/results_contracts_50",
                    help="Directory with TDG JSONs for the contract seeds")
    ap.add_argument("--ground-truth",
                    default="../data/synthetic/ground_truth_synthetic.json")
    ap.add_argument("--similarity-threshold", type=float,
                    default=DEFAULT_SIMILARITY_THRESHOLD)
    ap.add_argument("--sentence-threshold", type=float, default=0.2)
    ap.add_argument("--linking", choices=["composed", "gated"], default="gated",
                    help="gated = published v2 behaviour (default); "
                         "composed = v3 evidence composition")
    ap.add_argument("--embed-url", default=None)
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--tdg-dir", action="append", default=[],
                    help="additional TDG directory (repeatable) for held-out sets")
    args = ap.parse_args()

    # -- Load ground truth ------------------------------------------------
    if not os.path.exists(args.ground_truth):
        print(f"SKIP: ground truth not found at {args.ground_truth}")
        return 2
    gt = json.load(open(args.ground_truth))
    expected = gt.get("expected_contradictions", [])
    # The ground-truth file is the single source of which documents take
    # part: exactly the documents named by the expected contradictions
    # (documents merely described in the file but not part of any expected
    # pair are not loaded -- same semantics as the old hardcoded set). A NEW
    # held-out ground truth (fresh planted pairs, any language, any court)
    # therefore runs with no code change. REQUIRED_DOCS remains only as a
    # fallback for a file whose entries name no documents.
    required = set()
    for e in expected:
        required.update(x for x in (e.get("doc_a"), e.get("doc_b")) if x)
    required = required or set(REQUIRED_DOCS)
    if not expected:
        print("SKIP: ground truth has no expected_contradictions")
        return 2

    # -- Load TDGs --------------------------------------------------------
    tdgs = {}
    for d in [args.contracts_tdg_dir, args.synth_tdg_dir] + list(args.tdg_dir or []):
        if os.path.isdir(d):
            tdgs.update(load_tdg_dir(d))

    missing = sorted(required - set(tdgs))
    if missing:
        print("SKIP: required TDGs not present yet. Missing document_id(s):")
        for m in missing:
            print(f"  - {m}")
        print("\nExtract the synthetic documents to TDG JSON first, e.g.:")
        print("  python demo_llm.py --file ../data/synthetic/synth_steel_ukraine.txt \\")
        print("    --document-id synth_steel_ukraine --output-dir ../data/results_synthetic")
        print("(repeat for synth_fisheries_senegal and synth_fisheries_extension; "
              "seed0 already exists in results_contracts_50)")
        print("\nThe document_id of each TDG must match the names above "
              "(load_tdg_dir falls back to the filename when the JSON id is generic).")
        return 2

    # -- Run the detector -------------------------------------------------
    embedder = None
    if args.embed_url:
        from tdg_pipeline.embeddings import EmbeddingSimilarity
        embedder = EmbeddingSimilarity(base_url=args.embed_url, model=args.embed_model)
        print(f"embeddings: {args.embed_model} @ {args.embed_url}")
    linker = CrossDocLinker(
        similarity_threshold=args.similarity_threshold,
        sentence_threshold=args.sentence_threshold,
        embedder=embedder,
        composed=(args.linking == "composed"),
    )
    print(f"linking mode: {args.linking}")
    # Only add the documents involved, to keep the test focused and the
    # false-positive count interpretable.
    for doc_id in sorted(required):
        linker.add_tdg(tdgs[doc_id])

    if args.linking == "composed":
        # In composed mode disputes are emitted by the single matching pass
        # (find_contradictions() is intentionally empty there): a matched
        # pair whose values disagree IS the contradiction link.
        links = [l for l in linker.find_coreferences()
                 if l.link_type == "contradiction"]
    else:
        links = linker.find_contradictions()
    contradictions = [l for l in links if l.link_type == "contradiction"]
    parallels = [l for l in links if l.link_type == "parallel_application"]

    # -- Match detected contradictions to expected ones -------------------
    # A detected contradiction satisfies an expected one when the document
    # pair matches (unordered) AND the {value_a, value_b} day-pair matches.
    matched_expected = []
    used_links = set()
    for i, e in enumerate(expected):
        want_docs = frozenset({e.get("doc_a"), e.get("doc_b")})
        want_vals = _gt_day_pair(e)
        hit = None
        for j, l in enumerate(contradictions):
            if j in used_links:
                continue
            got_docs = frozenset({l.from_doc, l.to_doc})
            got_vals = _link_day_pair(l)
            if got_docs == want_docs and want_vals is not None and got_vals == want_vals:
                hit = j
                break
        if hit is not None:
            used_links.add(hit)
            matched_expected.append((i, hit))

    found = len(matched_expected)
    total = len(expected)
    extra = [l for j, l in enumerate(contradictions) if j not in used_links]

    # -- Report -----------------------------------------------------------
    print(f"Expected contradictions: {total}")
    print(f"Detected contradictions (these docs): {len(contradictions)} "
          f"(+{len(parallels)} parallel_application)")
    print(f"Matched: {found}/{total}\n")

    for i, e in enumerate(expected):
        status = "FOUND" if any(mi == i for mi, _ in matched_expected) else "MISSED"
        print(f"  [{status}] {e.get('doc_a')} ↔ {e.get('doc_b')}: "
              f"{e.get('concept')} ({e.get('value_a')} vs {e.get('value_b')})")

    if extra:
        print(f"\n  {len(extra)} detected contradiction(s) not in ground truth "
              f"(possible false positives):")
        for l in extra:
            print(f"    - {l.from_doc} ↔ {l.to_doc}: {l.explanation}")

    print()
    if found == total:
        print(f"PASS: detector fired on all {total} known contradictions.")
        if extra:
            print(f"NOTE: {len(extra)} extra contradiction(s) reported -- "
                  f"inspect for precision.")
        return 0
    else:
        print(f"FAIL: {total - found} expected contradiction(s) not detected. "
              f"Re-check extraction of the involved docs and the "
              f"sentence/similarity thresholds.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
    
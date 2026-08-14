#!/usr/bin/env python3
"""
Use Case A: Formal verification of LLM temporal extraction.

Compares LLM-extracted TDGs against hand-annotated ground truth and
produces measurable metrics:

  1. Fact extraction recall (value-based matching)
  2. Entity naming consistency (similarity between matched entity names)
  3. Role accuracy (does the LLM assign the correct temporal role?)
  4. Dependency recall (are ground-truth edges recovered?)
  5. Cross-doc link recovery (does CrossDocLinker find the expected links?)

Inputs:
  --ground-truth   path to ground_truth_uk.json
  --llm-dir        directory with LLM-extracted TDGs (era_s111_tdg.json, etc.)
  --output         path for evaluation results JSON

Usage:
    python evaluate_use_case_a.py \
        --ground-truth ../data/ground_truth/ground_truth_uk.json \
        --llm-dir ../data/llm_extracted/ \
        --output ../data/evaluation_results/use_case_a.json

File layout:
    code/
    ├-- data/
    │   ├-- ground_truth/
    │   │   └-- ground_truth_uk.json
    │   ├-- llm_extracted/
    │   │   ├-- era_s111_tdg.json
    │   │   ├-- ahmed_tdg.json
    │   │   └-- zegay_tdg.json
    │   └-- evaluation_results/
    │       └-- use_case_a.json       ← output
    └-- src/
        └-- evaluate_use_case_a.py    ← this script
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import date
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tdg_pipeline.tdg import (
    TemporalDependencyGraph, TemporalFact, TemporalDependency, TimexSpan,
)
from tdg_pipeline.cross_doc import CrossDocLinker, _entity_similarity, _normalise_entity
from tdg_pipeline.io import load_json, build_tdg, _GENERIC_DOC_IDS


# --- Fact Matching --------------------------------------------------------

def match_fact(gt_fact: dict, llm_facts: list[dict], tolerance_days: int = 5) -> Optional[dict]:
    """
    Find the LLM fact that best matches a ground truth fact.

    Matching strategy (in order):
    1. Exact date match (date_parsed)
    2. Exact duration match (duration_days within tolerance)
    3. Value string match
    Returns the best matching LLM fact or None.
    """
    gt_date = gt_fact.get("date_parsed")
    gt_dur = gt_fact.get("duration_days")
    gt_val = gt_fact.get("value")

    candidates = []

    for lf in llm_facts:
        score = 0
        lf_date = lf.get("date_parsed")
        lf_dur = lf.get("duration_days")
        lf_val = lf.get("value")

        # Date match
        if gt_date and lf_date and gt_date == lf_date:
            score = 3

        # Duration match
        elif gt_dur is not None and lf_dur is not None:
            if abs(gt_dur - lf_dur) <= tolerance_days:
                score = 2

        # Value string match
        elif gt_val and lf_val and gt_val == lf_val:
            score = 1

        if score > 0:
            candidates.append((score, lf))

    if not candidates:
        return None

    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


# --- Evaluation Functions -------------------------------------------------

def evaluate_document(doc_id: str, gt_data: dict, llm_data: dict) -> dict:
    """Evaluate a single document's LLM extraction against ground truth."""
    gt_facts = gt_data["facts"]
    llm_facts = llm_data.get("facts", [])
    gt_deps = gt_data.get("dependencies", [])
    llm_deps = llm_data.get("dependencies", [])

    results = {
        "document_id": doc_id,
        "gt_fact_count": len(gt_facts),
        "llm_fact_count": len(llm_facts),
        "gt_dep_count": len(gt_deps),
        "llm_dep_count": len(llm_deps),
        "fact_matches": [],
        "fact_misses": [],
        "dep_matches": [],
        "dep_misses": [],
    }

    # --- Fact matching ---
    used_llm = set()
    for gf in gt_facts:
        # Only evaluate facts with computable values
        if not gf.get("date_parsed") and gf.get("duration_days") is None and not gf.get("value"):
            continue

        match = match_fact(gf, [lf for lf in llm_facts if lf["id"] not in used_llm])
        if match:
            used_llm.add(match["id"])

            gt_entity = _normalise_entity(gf["entity"])
            llm_entity = _normalise_entity(match["entity"])
            name_sim = _entity_similarity(gf["entity"], match["entity"])
            role_correct = gf["role"] == match["role"]

            results["fact_matches"].append({
                "gt_id": gf["id"],
                "llm_id": match["id"],
                "gt_entity": gf["entity"],
                "llm_entity": match["entity"],
                "entity_similarity": round(name_sim, 3),
                "role_correct": role_correct,
                "gt_role": gf["role"],
                "llm_role": match["role"],
                "value_matched": True,
            })
        else:
            results["fact_misses"].append({
                "gt_id": gf["id"],
                "gt_entity": gf["entity"],
                "gt_role": gf["role"],
                "gt_value": gf.get("date_parsed") or gf.get("value"),
            })

    # --- Dependency matching ---
    for gd in gt_deps:
        # Find matching LLM dep: same constraint type, endpoints match
        # by value (not ID, since IDs differ between GT and LLM)
        gt_from = next((f for f in gt_facts if f["id"] == gd["from_id"]), None)
        gt_to = next((f for f in gt_facts if f["id"] == gd["to_id"]), None)
        if not gt_from or not gt_to:
            continue

        # Find LLM facts that matched these GT facts
        from_match = next((m for m in results["fact_matches"] if m["gt_id"] == gd["from_id"]), None)
        to_match = next((m for m in results["fact_matches"] if m["gt_id"] == gd["to_id"]), None)

        if from_match and to_match:
            llm_from_id = from_match["llm_id"]
            llm_to_id = to_match["llm_id"]

            # Check if this edge exists in LLM deps
            found = any(
                ld["from_id"] == llm_from_id and ld["to_id"] == llm_to_id
                for ld in llm_deps
            )
            # Also check reverse direction
            if not found:
                found = any(
                    ld["from_id"] == llm_to_id and ld["to_id"] == llm_from_id
                    for ld in llm_deps
                )

            if found:
                results["dep_matches"].append({
                    "gt_edge": f"{gd['from_id']}->{gd['to_id']}",
                    "llm_edge": f"{llm_from_id}->{llm_to_id}",
                    "constraint_type": gd["constraint_type"],
                })
            else:
                results["dep_misses"].append({
                    "gt_edge": f"{gd['from_id']}->{gd['to_id']}",
                    "constraint_type": gd["constraint_type"],
                    "reason": "endpoints matched but no edge between them",
                })
        else:
            results["dep_misses"].append({
                "gt_edge": f"{gd['from_id']}->{gd['to_id']}",
                "constraint_type": gd["constraint_type"],
                "reason": f"endpoint not found in LLM (from={'found' if from_match else 'MISSING'}, to={'found' if to_match else 'MISSING'})",
            })

    # --- Compute metrics ---
    computable_gt = sum(1 for f in gt_facts
                        if f.get("date_parsed") or f.get("duration_days") is not None or f.get("value"))
    results["metrics"] = {
        "fact_recall": len(results["fact_matches"]) / computable_gt if computable_gt else 0,
        "avg_entity_similarity": (
            sum(m["entity_similarity"] for m in results["fact_matches"]) / len(results["fact_matches"])
            if results["fact_matches"] else 0
        ),
        "role_accuracy": (
            sum(1 for m in results["fact_matches"] if m["role_correct"]) / len(results["fact_matches"])
            if results["fact_matches"] else 0
        ),
        "dep_recall": (
            len(results["dep_matches"]) / len(gt_deps) if gt_deps else 1.0
        ),
    }

    return results


def evaluate_cross_doc(gt_data: dict, llm_tdgs: dict[str, dict]) -> dict:
    """Evaluate cross-doc link recovery using CrossDocLinker on LLM TDGs."""
    gt_links = gt_data.get("cross_doc_links", [])

    # Build TDG objects from LLM data and run linker
    linker = CrossDocLinker()
    for doc_id, data in llm_tdgs.items():
        linker.add_tdg(build_tdg(data))

    found_links = linker.find_all_links()

    # Match GT links against found links
    recovered = []
    missed = []

    for gt_link in gt_links:
        gt_from_doc = gt_link["from_doc"]
        gt_to_doc = gt_link["to_doc"]
        gt_type = gt_link["type"]

        # Look for any found link between the same two documents with compatible type
        matched = False
        for fl in found_links:
            docs_match = (
                (fl.from_doc == gt_from_doc and fl.to_doc == gt_to_doc) or
                (fl.from_doc == gt_to_doc and fl.to_doc == gt_from_doc)
            )
            # Accept coreference or parallel_application as recovery of a GT coreference
            type_compatible = (
                fl.link_type == gt_type or
                (gt_type == "coreference" and fl.link_type in ("coreference", "parallel_application"))
            )
            if docs_match and type_compatible:
                matched = True
                recovered.append({
                    "gt_type": gt_type,
                    "gt_concept": gt_link.get("concept", ""),
                    "found_type": fl.link_type,
                    "found_conf": round(fl.confidence, 3),
                    "found_explanation": fl.explanation[:100],
                })
                break

        if not matched:
            missed.append({
                "gt_type": gt_type,
                "gt_concept": gt_link.get("concept", ""),
                "gt_docs": f"{gt_from_doc} ↔ {gt_to_doc}",
            })

    return {
        "gt_link_count": len(gt_links),
        "found_link_count": len(found_links),
        "recovered": recovered,
        "missed": missed,
        "link_recall": len(recovered) / len(gt_links) if gt_links else 1.0,
    }


# --- Main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Use Case A: Evaluate LLM temporal extraction")
    parser.add_argument("--ground-truth", required=True, help="Path to ground_truth_uk.json")
    parser.add_argument("--llm-dir", required=True, help="Directory with LLM-extracted TDG JSONs")
    parser.add_argument("--output", default=None, help="Path to save evaluation results JSON")
    args = parser.parse_args()

    # Load ground truth
    gt = load_json(args.ground_truth)

    # Load all LLM-extracted TDGs from directory
    _GENERIC = _GENERIC_DOC_IDS
    llm_tdgs = {}
    for fpath in sorted(glob.glob(os.path.join(args.llm_dir, "*.json"))):
        try:
            data = load_json(fpath)
            doc_id = data.get("document_id", "")
            if doc_id in _GENERIC:
                doc_id = os.path.splitext(os.path.basename(fpath))[0]
            data["document_id"] = doc_id
            llm_tdgs[doc_id] = data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARNING: skipping {fpath}: {e}")

    # --- Per-document evaluation ---
    all_results = {"documents": {}, "cross_doc": {}, "summary": {}}

    print(f"\n{'='*70}")
    print("USE CASE A: FORMAL VERIFICATION OF LLM TEMPORAL EXTRACTION")
    print(f"{'='*70}")

    for doc_id in sorted(set(gt.keys()) & set(llm_tdgs.keys()) - {"cross_doc_links"}):
        if doc_id not in gt or doc_id not in llm_tdgs:
            print(f"\n  SKIP: {doc_id} (missing GT or LLM data)")
            continue

        result = evaluate_document(doc_id, gt[doc_id], llm_tdgs[doc_id])
        all_results["documents"][doc_id] = result

        m = result["metrics"]
        print(f"\n  {doc_id}")
        print(f"    Facts:  GT={result['gt_fact_count']}  LLM={result['llm_fact_count']}  "
              f"matched={len(result['fact_matches'])}  missed={len(result['fact_misses'])}")
        print(f"    Deps:   GT={result['gt_dep_count']}  LLM={result['llm_dep_count']}  "
              f"matched={len(result['dep_matches'])}  missed={len(result['dep_misses'])}")
        print(f"    Fact recall:           {m['fact_recall']:.0%}")
        print(f"    Avg entity similarity: {m['avg_entity_similarity']:.2f}")
        print(f"    Role accuracy:         {m['role_accuracy']:.0%}")
        print(f"    Dependency recall:     {m['dep_recall']:.0%}")

        if result["fact_matches"]:
            print(f"    Entity naming (matched facts):")
            for fm in result["fact_matches"]:
                sim = fm["entity_similarity"]
                role_ok = "✓" if fm["role_correct"] else f"x (GT={fm['gt_role']}, LLM={fm['llm_role']})"
                print(f"      GT \"{fm['gt_entity']}\"  ->  LLM \"{fm['llm_entity']}\"  "
                      f"sim={sim:.2f}  role={role_ok}")

        if result["dep_misses"]:
            print(f"    Missing dependencies:")
            for dm in result["dep_misses"]:
                print(f"      {dm['gt_edge']} [{dm['constraint_type']}]: {dm['reason']}")

    # --- Cross-doc evaluation ---
    print(f"\n{'='*70}")
    print("CROSS-DOCUMENT LINK RECOVERY")
    print(f"{'='*70}")

    cross_result = evaluate_cross_doc(gt, llm_tdgs)
    all_results["cross_doc"] = cross_result

    print(f"  GT links: {cross_result['gt_link_count']}  "
          f"Found links: {cross_result['found_link_count']}  "
          f"Recovered: {len(cross_result['recovered'])}  "
          f"Missed: {len(cross_result['missed'])}")
    print(f"  Link recall: {cross_result['link_recall']:.0%}")

    if cross_result["recovered"]:
        print(f"\n  Recovered:")
        for r in cross_result["recovered"]:
            print(f"    [{r['gt_type']}] {r['gt_concept']}")
            print(f"      -> found as [{r['found_type']}] conf={r['found_conf']}")

    if cross_result["missed"]:
        print(f"\n  Missed:")
        for m in cross_result["missed"]:
            print(f"    [{m['gt_type']}] {m['gt_concept']}")
            print(f"      docs: {m['gt_docs']}")

    # --- Summary ---
    doc_results = list(all_results["documents"].values())
    if doc_results:
        all_results["summary"] = {
            "avg_fact_recall": sum(d["metrics"]["fact_recall"] for d in doc_results) / len(doc_results),
            "avg_entity_similarity": sum(d["metrics"]["avg_entity_similarity"] for d in doc_results) / len(doc_results),
            "avg_role_accuracy": sum(d["metrics"]["role_accuracy"] for d in doc_results) / len(doc_results),
            "avg_dep_recall": sum(d["metrics"]["dep_recall"] for d in doc_results) / len(doc_results),
            "cross_doc_link_recall": cross_result["link_recall"],
        }

        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        s = all_results["summary"]
        print(f"  Avg fact recall:           {s['avg_fact_recall']:.0%}")
        print(f"  Avg entity similarity:     {s['avg_entity_similarity']:.2f}")
        print(f"  Avg role accuracy:         {s['avg_role_accuracy']:.0%}")
        print(f"  Avg dependency recall:     {s['avg_dep_recall']:.0%}")
        print(f"  Cross-doc link recall:     {s['cross_doc_link_recall']:.0%}")

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
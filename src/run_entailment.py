#!/usr/bin/env python3
"""
Use Case B runner: temporal entailment over real statute + case TDGs.

Loads a STATUTE TDG (the rule), a directory of CASE TDGs (instances with
concrete dates), and a ground-truth file (the tribunal's actual verdict +
optional ACAS Day A/Day B per case). Runs the general entailment engine and
reports per-case verdict vs ground truth and overall accuracy.

No hardcoded statute rule: the rule is discovered from the statute TDG's
dependency structure. Swap the statute TDG to evaluate a different rule.

Ground-truth JSON format:
{
  "statute": "era_1996_s111",
  "minus_one_day": true,
  "cases": {
    "et_f4779f17": {
      "verdict": "out_of_time",
      "acas_day_a": null, "acas_day_b": null,
      "split": "real"
    },
    ...
  }
}
verdict is "in_time" | "out_of_time" (mapped to TIMELY | LATE).

Usage:
  python run_entailment.py \
      --statute-tdg ../data/results_uk/era_1996_s111.json \
      --cases-dir   ../data/results_uk \
      --ground-truth ../data/ground_truth/ground_truth_uk.json \
      --output ../data/evaluation_results/use_case_b.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from typing import Optional

from tdg_pipeline.io import load_tdg, load_tdg_dir, load_json
from tdg_pipeline.entailment import check_entailment, find_rules

_VERDICT_MAP = {"in_time": "TIMELY", "out_of_time": "LATE",
                "timely": "TIMELY", "late": "LATE"}


def _parse_date(s: Optional[str]) -> Optional[date]:
    return date.fromisoformat(s) if s else None


def _embedder():
    """Use embeddings for entity matching if Ollama is reachable; else None
    (engine falls back to token-Jaccard). Never fatal."""
    try:
        from tdg_pipeline.embeddings import EmbeddingSimilarity
        emb = EmbeddingSimilarity()
        return emb if emb.is_available() else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Use Case B: temporal entailment")
    ap.add_argument("--statute-tdg", required=True,
                    help="Path to the statute (rule) TDG JSON")
    ap.add_argument("--cases-dir", required=True,
                    help="Directory of case (instance) TDG JSONs")
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-minus-one-day", action="store_true",
                    help="Disable the UK 'beginning with' -1 day convention")
    args = ap.parse_args()

    if not os.path.exists(args.statute_tdg):
        print(f"SKIP: statute TDG not found: {args.statute_tdg}")
        return 2
    if not os.path.isdir(args.cases_dir):
        print(f"SKIP: cases dir not found: {args.cases_dir}")
        return 2
    if not os.path.exists(args.ground_truth):
        print(f"SKIP: ground truth not found: {args.ground_truth}")
        return 2

    gt = load_json(args.ground_truth)
    cases_gt = gt.get("cases", {})
    # None => find_rules reads the -1 day off the statute's own connector
    # ("three months BEGINNING WITH ..."). --no-minus-one-day still forces it
    # off, and a ground-truth file may still pin it explicitly, but neither is
    # needed for the statutes in scope: the text says which convention applies.
    minus_one = None
    if args.no_minus_one_day:
        minus_one = False
    elif "minus_one_day" in gt:
        minus_one = gt["minus_one_day"]      # explicit pin: honour it, but
        minus_one = None if minus_one else False   # True == "discover it"

    statute = load_tdg(args.statute_tdg)
    embedder = _embedder()
    rules = find_rules(statute, minus_one_day=minus_one, embedder=embedder)
    if not rules:
        print(f"SKIP: no temporal rule discovered in statute TDG "
              f"'{statute.document_id}'. The statute extraction must contain an "
              f"additive dependency (anchor + period -> deadline).")
        return 2
    print(f"Statute: {statute.document_id}  |  discovered rule: {rules[0].description}")

    case_tdgs = load_tdg_dir(args.cases_dir)

    results = []
    correct = 0
    scored = 0
    indeterminate = 0
    for case_id, meta in cases_gt.items():
        if case_id not in case_tdgs:
            results.append({"case": case_id, "status": "missing_tdg"})
            continue
        exp = _VERDICT_MAP.get(str(meta.get("verdict", "")).lower())
        da = _parse_date(meta.get("acas_day_a"))
        db = _parse_date(meta.get("acas_day_b"))

        res = check_entailment(statute, case_tdgs[case_id], embedder=embedder,
                               minus_one_day=minus_one,
                               acas_day_a=da, acas_day_b=db)
        if not res:
            results.append({"case": case_id, "status": "no_rule"})
            continue
        r = res[0]  # one rule (the limitation period)
        got = r.verdict
        match = (exp is not None and got == exp)
        if got == "INDETERMINATE":
            indeterminate += 1
        elif exp is not None:
            scored += 1
            correct += match

        results.append({
            "case": case_id, "split": meta.get("split", "real"),
            "expected": exp, "verdict": got, "match": match,
            "days_over": r.days_over, "deadline": r.deadline_computed,
            "anchor_date": r.anchor_date, "action_date": r.action_date,
            "acas_applied": r.acas_applied,
            "confidence": round(r.match_confidence, 2),
            "explanation": r.explanation,
        })

    print(f"\n{'case':28} {'exp':5} {'got':12} {'Δdays':>6}  conf  note")
    print("-" * 78)
    for r in results:
        if r.get("status"):
            print(f"  {r['case']:26} -- {r['status']}")
            continue
        mark = "OK " if r["match"] else ("?? " if r["verdict"] == "INDETERMINATE" else "XX ")
        print(f"{mark}{r['case']:25} {str(r['expected']):5} {r['verdict']:12} "
              f"{str(r['days_over']):>6}  {r['confidence']:.2f}")

    print(f"\nScored {scored} cases (excludes {indeterminate} INDETERMINATE): "
          f"{correct}/{scored} correct"
          + (f"  ({correct/scored:.0%})" if scored else ""))

    out = {
        "statute": statute.document_id,
        "rule": rules[0].description,
        "minus_one_day": minus_one,
        "scored": scored, "correct": correct, "indeterminate": indeterminate,
        "accuracy": (correct / scored) if scored else None,
        "cases": results,
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    
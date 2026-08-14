#!/usr/bin/env python3
"""
Proof-of-concept experiment runner.

Question the experiment answers
-------------------------------
When a legal temporal question requires reasoning over a *connected* chain of
dated events, does the "traditional best approach" (hand the document to a
strong LLM and ask) keep up with a method that makes the dependency structure
explicit and then computes the answer deterministically?

It runs two solvers on the SAME items:
  * structured   -- the project's own engine (no LLM in the reasoning step).
                   Always runs; needs no API.
  * baseline_llm -- a flat-text LLM, given the whole document and the question.
                   Runs only when you pass --model (and --base-url for Ollama).

and, with --real, validates the structured engine on the real UK tribunal TDGs
already in the repo (statute + 2 cases), reproducing the tribunals' verdicts.

Usage
-----
  # method side only -- runs anywhere, no API, proves the engine is exact:
  python proof/run_proof.py --real

  # full head-to-head with a local model (Ollama):
  python proof/run_proof.py --real \
      --model gemma4:e4b --base-url http://localhost:11434/v1

  # or with OpenAI (set OPENAI_API_KEY):
  python proof/run_proof.py --model gpt-4o

Results (per-item answers, per-bucket accuracy, and the benchmark itself for
reproducibility) are written to --output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proof.benchmark import build_benchmark, build_benchmark_scaled, Item
from proof.solvers import (StructuredSolver, CalendarNaiveSolver,
                           BaselineLLMSolver, Answer, score)


# --- bucketing for the degradation curve ------------------------------------

def bucket(item: Item) -> str:
    d = item.difficulty
    if item.task == "deadline":
        return f"deadline / offset={d['offset_form']}"
    return f"cascade / hops={d['hops']} / {d['locality']}"


def aggregate(items, answers_by_solver):
    """Returns {solver: {bucket: (fully_correct, n_items, fields_correct, fields_n)}}."""
    out = {}
    for solver, answers in answers_by_solver.items():
        b = defaultdict(lambda: [0, 0, 0, 0])
        for it, ans in zip(items, answers):
            s = score(it, ans)
            k = bucket(it)
            b[k][0] += int(s["fully_correct"])
            b[k][1] += 1
            b[k][2] += s["n_correct"]
            b[k][3] += s["n_fields"]
        out[solver] = {k: tuple(v) for k, v in b.items()}
    return out


# --- real tribunal validation (uses the repo's actual data + engine) ---------

def run_real(repo_data: Path):
    from tdg_pipeline.io import load_tdg, load_tdg_dir, load_json
    from tdg_pipeline.entailment import check_entailment, find_rules

    statute_p = repo_data / "results_uk" / "era_1996_s111.json"
    cases_dir = repo_data / "results_uk"
    gt_p = repo_data / "ground_truth" / "ground_truth_uk.json"
    if not (statute_p.exists() and gt_p.exists()):
        return None

    gt = load_json(str(gt_p))
    minus_one = gt.get("minus_one_day", True)
    statute = load_tdg(str(statute_p))
    rules = find_rules(statute, minus_one_day=minus_one)
    case_tdgs = load_tdg_dir(str(cases_dir))
    vmap = {"in_time": "TIMELY", "out_of_time": "LATE"}

    rows = []
    correct = 0
    scored = 0
    for cid, meta in gt.get("cases", {}).items():
        if cid not in case_tdgs:
            continue
        exp = vmap.get(str(meta.get("verdict", "")).lower())
        res = check_entailment(statute, case_tdgs[cid], minus_one_day=minus_one)
        if not res:
            continue
        r = res[0]
        ok = (exp is not None and r.verdict == exp)
        scored += 1
        correct += int(ok)
        rows.append({"case": cid, "expected": exp, "verdict": r.verdict,
                     "match": ok, "deadline": r.deadline_computed,
                     "anchor": r.anchor_date, "action": r.action_date,
                     "days_over": r.days_over})
    return {"rule": rules[0].description if rules else None,
            "scored": scored, "correct": correct, "rows": rows}


# --- pretty printing ---------------------------------------------------------

def pct(c, n):
    return f"{c}/{n}" + (f" ({c/n:.0%})" if n else "")


def print_report(items, agg, baseline_keys, real):
    print("\n" + "=" * 74)
    print("  PROOF-OF-CONCEPT  --  flat-text LLM reasoning  vs  structure+compute")
    print("=" * 74)
    print("Ground truth is the calendar (relativedelta), independent of both "
          "solvers.\nDocuments are explicit, so extraction is trivial; only the "
          "reasoning load varies.\n")

    solvers = ["structured", "calendar_naive"] + list(baseline_keys)
    buckets = sorted({bucket(it) for it in items})

    head = f"{'bucket':34}" + "".join(f"{s:>22}" for s in solvers)
    print(head)
    print("-" * len(head))
    for k in buckets:
        row = f"{k:34}"
        for s in solvers:
            c, n, fc, fn = agg[s][k]
            row += f"{pct(c, n):>22}"
        print(row)
    print("-" * len(head))
    # overall
    row = f"{'OVERALL (items fully correct)':34}"
    for s in solvers:
        c = sum(v[0] for v in agg[s].values())
        n = sum(v[1] for v in agg[s].values())
        row += f"{pct(c, n):>22}"
    print(row)
    row = f"{'OVERALL (fields correct)':34}"
    for s in solvers:
        fc = sum(v[2] for v in agg[s].values())
        fn = sum(v[3] for v in agg[s].values())
        row += f"{pct(fc, fn):>22}"
    print(row)

    if not baseline_keys:
        print("\n[no baseline_llm run] Pass --model (repeatable) and --base-url "
              "for Ollama to fill\nbaseline columns. The structured column is the "
              "method's reasoning, computed\ndeterministically -- it does not depend "
              "on any model.")

    if real:
        print("\n" + "-" * 74)
        print("  REAL-DATA VALIDATION  --  structured engine on actual UK tribunal TDGs")
        print("-" * 74)
        print(f"rule discovered from statute text: {real['rule']}")
        for r in real["rows"]:
            mark = "OK" if r["match"] else "XX"
            print(f"  {mark}  {r['case']:26} expected={r['expected']:5} "
                  f"got={r['verdict']:5} deadline={r['deadline']} "
                  f"action={r['action']} ({r['days_over']:+}d)")
        print(f"  verdicts reproduced: {pct(real['correct'], real['scored'])}")
    print()


# --- main --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Flat-LLM vs structured temporal reasoning")
    ap.add_argument("--model", default=None, action="append", dest="models",
                    help="LLM for the baseline (e.g. gemma4:e4b, llama3.1:8b). "
                         "Repeatable: pass --model twice to compare two models "
                         "on the SAME items. If omitted, only the structured "
                         "method is run.")
    ap.add_argument("--base-url", default=None, dest="base_url",
                    help="OpenAI-compatible base URL (e.g. http://localhost:11434/v1)")
    ap.add_argument("--scale", type=int, default=None,
                    help="Use the scaled benchmark with this many items PER "
                         "reported bucket (7 buckets). Omit for the canonical "
                         "26-item set.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for --scale (reproducibility).")
    ap.add_argument("--real", action="store_true",
                    help="Also validate the engine on the repo's real tribunal TDGs")
    ap.add_argument("--minus-one-day", action="store_true",
                    help="Apply the UK 'beginning with' -1 day convention to the "
                         "synthetic deadline items (default off; the synthetic "
                         "items are phrased 'within N of X')")
    ap.add_argument("--output", default="../data/evaluation_results/proof_of_concept.json")
    args = ap.parse_args()

    items = build_benchmark_scaled(n_per_bucket=args.scale, seed=args.seed) \
        if args.scale else build_benchmark()
    if args.scale:
        print(f"Using scaled benchmark: {len(items)} items "
              f"({args.scale}/bucket, seed={args.seed}).", file=sys.stderr)

    structured = StructuredSolver(minus_one_day=args.minus_one_day)
    naive = CalendarNaiveSolver()
    answers_by_solver = {
        "structured": [structured.answer(it) for it in items],
        "calendar_naive": [naive.answer(it) for it in items],
    }

    models = args.models or []
    baseline_keys = []
    for model in models:
        key = f"llm:{model}"
        baseline_keys.append(key)
        base = BaselineLLMSolver(model=model, base_url=args.base_url)
        print(f"Running baseline LLM: {model} "
              f"({args.base_url or 'OpenAI'}) over {len(items)} items ...",
              file=sys.stderr)
        print(f"  warming up {model} (load into memory before scoring) ...",
              file=sys.stderr)
        base.warmup()
        ba = []
        for i, it in enumerate(items, 1):
            try:
                ba.append(base.answer(it))
            except Exception as e:
                ba.append(Answer(it.item_id, it.task, base.name, error=str(e)))
            print(f"  [{i}/{len(items)}] {it.item_id}", file=sys.stderr)
        answers_by_solver[key] = ba

    agg = aggregate(items, answers_by_solver)
    real = run_real(Path(__file__).resolve().parent.parent.parent / "data") \
        if args.real else None

    print_report(items, agg, baseline_keys, real)

    # write everything (reproducible): benchmark + per-item answers + scores
    out = {
        "config": {"models": models, "base_url": args.base_url,
                   "minus_one_day": args.minus_one_day,
                   "scale": args.scale, "seed": args.seed,
                   "n_items": len(items)},
        "benchmark": [
            {"item_id": it.item_id, "task": it.task, "difficulty": it.difficulty,
             "documents": it.documents, "question": it.question, "gold": it.gold,
             "notes": it.notes}
            for it in items
        ],
        "answers": {
            solver: [
                {"item_id": a.item_id, "task": a.task,
                 "deadline": a.deadline, "verdict": a.verdict,
                 "updates": a.updates, "score": score(it, a),
                 "trace": a.trace, "raw": a.raw, "error": a.error}
                for it, a in zip(items, answers)
            ]
            for solver, answers in answers_by_solver.items()
        },
        "buckets": {s: {k: list(v) for k, v in b.items()} for s, b in agg.items()},
        "real": real,
    }
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"Wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
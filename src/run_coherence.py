#!/usr/bin/env python3
"""run_coherence.py -- run and score the label-free coherence battery.

Six independent calls per case (no shared context), frozen prompts, temp 0,
every raw archived. Follows the same instrumentation discipline as
run_baseline.py: per-call infrastructure errors are recorded separately from
model abstentions, and an absent error field is treated as a nonzero one.

MODES
  --emit-prompts DIR        one prompt file per (case, question); no calls.
                            Use for API models run elsewhere.
  --ollama URL --model M    run now against an Ollama endpoint.
  --score-dir DIR           score archived raws (offline, no model calls).
  --engine RESULTS.json     project engine results through the same scorer.

LAYOUT under --out-dir
  raw/<case>.<question>.json     {prompt_sha, raw, call_error, latency_s}
  batteries.json                 parsed six-field battery per case
  reports.json                   per-case constraint evaluation
  summary.json                   corpus aggregate  <- the number you publish

USAGE
  python run_coherence.py --out-dir ../data/experiments/coherence/gemma \\
      --ollama http://localhost:11434 --model gemma4:e4b

  python run_coherence.py --out-dir ../data/experiments/coherence/gemma \\
      --score-dir ../data/experiments/coherence/gemma/raw
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import coherence as C

def _find_root() -> Path:
    """Walk up until data/ground_truth is found.

    run_baseline.py lives at code/src/baseline/ and uses parents[2]; this file
    lives at code/src/ and would need parents[1]. Searching removes the
    dependency on where it is dropped.
    """
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / "data" / "ground_truth").is_dir():
            return cand
    return here.parents[1]


ROOT = _find_root()
GOLD = ROOT / "data" / "ground_truth" / "ground_truth_gold.json"
INPUTS = ROOT / "data" / "experiments" / "baseline" / "inputs"
STATUTES = ROOT / "data" / "statutes" / "txt"

# Reuse the baseline's mapping so the two experiments read the same documents.
CASE_FILE = {
    "2026_EAT_64_s111": "2026_EAT_64.txt",
    "2026_EAT_64_s123": "2026_EAT_64.txt",
    "2025_EAT_155": "2025_EAT_155.txt",
    "2026_EAT_14": "2026_EAT_14.txt",
    "2026_EAT_76": "2026_EAT_76.txt",
    "2026_EAT_59": "2026_EAT_59.txt",
    "kj_2026_EAT_46": "2026_EAT_46.txt",
}

CLAIM_HINT = {
    "2026_EAT_64_s111": " (the unfair dismissal complaint)",
    "2026_EAT_64_s123": " (the discrimination complaint)",
}


def load_cases(gold_path: Path) -> dict:
    return json.loads(gold_path.read_text())["cases"]


def statute_text(statute_id: str) -> str:
    p = STATUTES / f"{statute_id}.txt"
    return p.read_text() if p.exists() else ""


def case_text(case_id: str) -> str:
    fn = CASE_FILE.get(case_id)
    if not fn:
        raise KeyError(f"no input file mapped for {case_id}")
    return (INPUTS / fn).read_text()


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------

def call_ollama(url: str, model: str, prompt: str,
                timeout: int = 900) -> tuple[str, str | None, float]:
    """-> (raw, call_error, latency). call_error is None on success."""
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "seed": 0, "num_ctx": 32768},
    }).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/generate", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
        return payload.get("response", ""), None, time.time() - t0
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError,
            OSError) as e:
        return "", f"{type(e).__name__}: {e}", time.time() - t0


# ---------------------------------------------------------------------------

def cmd_emit(args, cases):
    """Write prompt files for an API model.

    MUST honour --single: the single condition is ONE combined prompt per
    case (<case>.__all__.txt), not six. Emitting six here and scoring with
    --single looks for <case>.__all__.json, finds nothing, and reports
    coverage 0.000 -- an empty run that costs a full set of API calls.
    """
    out = Path(args.emit_prompts)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for cid, meta in cases.items():
        st = statute_text(meta["statute"])
        ct = case_text(cid)
        hint = CLAIM_HINT.get(cid, "")
        if args.single:
            p = C.build_single_prompt(st, ct, hint, C.question_order(cid))
            (out / f"{cid}.{C.SINGLE_ID}.txt").write_text(p)
            n += 1
        else:
            for qid in C.question_order(cid):
                p = C.build_prompt(qid, st, ct, hint)
                (out / f"{cid}.{qid}.txt").write_text(p)
                n += 1
    mode = "single-prompt" if args.single else "independent"
    print(f"wrote {n} {mode} prompts to {out}")


def cmd_run(args, cases):
    raw_dir = Path(args.out_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    infra = 0
    if args.single:
        for cid, meta in cases.items():
            dest = raw_dir / f"{cid}.{C.SINGLE_ID}.json"
            if dest.exists() and not args.redo:
                continue
            prompt = C.build_single_prompt(
                statute_text(meta["statute"]), case_text(cid),
                CLAIM_HINT.get(cid, ""), C.question_order(cid))
            raw, err, dt = call_ollama(args.ollama, args.model, prompt)
            if err:
                infra += 1
            dest.write_text(json.dumps({
                "case_id": cid, "question": C.SINGLE_ID, "model": args.model,
                "mode": "single", "schema_version": C.SCHEMA_VERSION,
                "prompt_sha": sha(prompt), "raw": raw,
                "call_error": err, "latency_s": round(dt, 2),
            }, indent=1))
            print(f"  {cid:28s} {'ALL':11s} {dt:6.1f}s"
                  f"{'  ERROR ' + err if err else ''}")
        print(f"\ninfrastructure errors: {infra}")
        if infra:
            print("NOTE: nonzero infra errors -- rerun before reporting.")
        return
    for cid, meta in cases.items():
        st = statute_text(meta["statute"])
        ct = case_text(cid)
        for qid in C.question_order(cid):
            dest = raw_dir / f"{cid}.{qid}.json"
            if dest.exists() and not args.redo:
                continue
            prompt = C.build_prompt(qid, st, ct, CLAIM_HINT.get(cid, ""))
            raw, err, dt = call_ollama(args.ollama, args.model, prompt)
            if err:
                infra += 1
            dest.write_text(json.dumps({
                "case_id": cid, "question": qid, "model": args.model,
                "schema_version": C.SCHEMA_VERSION,
                "prompt_sha": sha(prompt), "raw": raw,
                "call_error": err, "latency_s": round(dt, 2),
            }, indent=1))
            print(f"  {cid:28s} {qid:11s} {dt:6.1f}s"
                  f"{'  ERROR ' + err if err else ''}")
    print(f"\ninfrastructure errors: {infra}")
    if infra:
        print("NOTE: nonzero infra errors -- rerun the failed calls before "
              "reporting any rate from this run.")


def cmd_score(args, cases):
    raw_dir = Path(args.score_dir)
    batteries, reports, infra, missing_files = [], [], 0, 0

    for cid in cases:
        b = C.Battery(case_id=cid)
        if args.single:
            f = raw_dir / f"{cid}.{C.SINGLE_ID}.json"
            if not f.exists():
                missing_files += 1
                b.statuses = {q: "no_file" for q in C.QUESTION_IDS}
                batteries.append(b)
                reports.append(C.check(b))
                continue
            rec = json.loads(f.read_text())
            if rec.get("call_error"):
                infra += 1
                b.statuses = {q: "infra_error" for q in C.QUESTION_IDS}
            else:
                # one JSON object carries all six fields; pull each in turn
                for qid in C.QUESTION_IDS:
                    val, status = C.parse_answer(qid, rec.get("raw", ""))
                    b.statuses[qid] = status
                    if status == "ok":
                        setattr(b, C._QUESTIONS[qid]["field"], val)
            batteries.append(b)
            reports.append(C.check(b))
            continue
        for qid in C.QUESTION_IDS:
            f = raw_dir / f"{cid}.{qid}.json"
            if not f.exists():
                b.statuses[qid] = "no_file"
                missing_files += 1
                continue
            rec = json.loads(f.read_text())
            if rec.get("call_error"):
                b.statuses[qid] = "infra_error"
                infra += 1
                continue
            val, status = C.parse_answer(qid, rec.get("raw", ""))
            b.statuses[qid] = status
            if status == "ok":
                setattr(b, C._QUESTIONS[qid]["field"], val)
        batteries.append(b)
        reports.append(C.check(b))

    summary = C.aggregate(reports)
    summary["mode"] = "single_prompt" if args.single else "independent_calls"
    summary["call_errors_infra"] = infra
    summary["missing_raw_files"] = missing_files

    if infra or missing_files:
        summary["_WARNING"] = (
            "infrastructure errors or missing raws present -- coverage is "
            "contaminated; rerun before publishing any rate")

    # secondary, reported apart from coherence
    summary["accuracy"] = [C.accuracy(b, cases[b.case_id]) for b in batteries]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "batteries.json").write_text(
        json.dumps([C.to_jsonable(b) for b in batteries], indent=1))
    (out / "reports.json").write_text(json.dumps(reports, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"\ncases            {summary['cases_total']}")
    print(f"complete         {summary['cases_complete']} "
          f"(coverage {summary['coverage']:.3f})" if summary['coverage']
          is not None else "")
    print(f"incoherent       {summary['incoherent']} / {summary['cases_scored']}"
          + (f"  = {summary['incoherence_rate']:.3f}"
             if summary['incoherence_rate'] is not None else ""))
    print(f"  format-only    {summary['format_only']}")
    print(f"infra errors     {infra}")
    print("\nper constraint:")
    for k, v in summary["per_constraint"].items():
        tag = "" if v["independent"] else "  [derived]"
        rate = f"{v['rate']:.3f}" if v["rate"] is not None else "  n/a"
        print(f"  {k}  {v['violations']:3d}/{v['evaluable']:3d}  {rate}{tag}"
              f"   {v['description']}")



# --- engine result shapes -------------------------------------------------
# Three shapes exist in the repo and all three must work, because picking the
# wrong one silently yields an all-abstained run that looks like a clean zero.
#
#   A  {"rows": [{case, effective_deadline, presented, days_over,
#                 engine_verdict}, ...]}          <- results_gold_facts.json,
#                                                    use_case_b*.json
#   B  {"cases": {case_id: <EntailmentResult-ish>}}
#   C  {case_id: <EntailmentResult-ish>}
#
# Shape A uses different key names from EntailmentResult, so it is renamed
# onto the canonical ones here rather than in engine_battery().

_ROW_ALIASES = {
    "deadline_computed": ("deadline_computed", "effective_deadline",
                          "deadline"),
    "action_date": ("action_date", "presented"),
    "days_over": ("days_over",),
    "verdict": ("verdict", "engine_verdict"),
    "anchor_date": ("anchor_date", "anchor"),
    "rule_description": ("rule_description", "rule_desc"),
    "match_confidence": ("match_confidence", "confidence"),
    "acas_applied": ("acas_applied",),
    "explanation": ("explanation", "note"),
}


def _normalise_engine_results(doc) -> dict:
    if isinstance(doc, dict) and "rows" in doc:
        out = {}
        for row in doc["rows"]:
            cid = row.get("case") or row.get("case_id")
            if not cid:
                continue
            out[cid] = {canon: next((row[k] for k in keys if k in row), None)
                        for canon, keys in _ROW_ALIASES.items()}
        return out
    if isinstance(doc, dict) and "cases" in doc:
        cases = doc["cases"]
        if isinstance(cases, dict):
            return cases
        # use_case_b_full_*.json: "cases" is a LIST of rows keyed by "case",
        # with "deadline" rather than "deadline_computed". Rows may also be
        # bare {"case", "status": "missing_tdg"} stubs -> abstention.
        out = {}
        for row in cases:
            cid = row.get("case") or row.get("case_id")
            if not cid:
                continue
            out[cid] = {canon: next((row[k] for k in keys if k in row), None)
                        for canon, keys in _ROW_ALIASES.items()}
        return out
    return doc if isinstance(doc, dict) else {}


def cmd_engine(args, cases):
    """Project engine results through the identical scorer."""
    results = _normalise_engine_results(json.loads(Path(args.engine).read_text()))
    batteries, reports, prov = [], [], {}
    for cid in cases:
        r = results.get(cid)
        if r is None:
            b = C.Battery(case_id=cid, statuses={q: "no_result"
                                                 for q in C.QUESTION_IDS})
            prov[cid] = {"raw_verdict": None}
        else:
            b = C.engine_battery(cid, r)
            prov[cid] = C.engine_provenance(r)
        batteries.append(b)
        reports.append(C.check(b))
    summary = C.aggregate(reports)
    summary["_analytic"] = (
        "Engine fields are projections of a single computation. A zero "
        "violation rate here is analytic, not an empirical win -- report it "
        "as a property of the architecture.")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "provenance.json").write_text(json.dumps(prov, indent=1))
    (out / "batteries.json").write_text(
        json.dumps([C.to_jsonable(b) for b in batteries], indent=1))
    abst = sum(1 for b in batteries if not b.complete())
    print(f"abstained/incomplete: {abst}/{len(batteries)}")
    print(json.dumps(summary["per_constraint"], indent=1))
    print(f"\nincoherent: {summary['incoherent']}/{summary['cases_scored']}")


def cmd_compare(args):
    a, b = (json.loads((Path(d) / "summary.json").read_text())
            for d in args.compare)
    print(f"{'':22s} {'independent':>13s} {'single prompt':>14s}")
    for label, key in (("coverage", "coverage"),
                       ("cases scored", "cases_scored"),
                       ("incoherent", "incoherent"),
                       ("incoherence rate", "incoherence_rate"),
                       ("format-only", "format_only")):
        fa, fb = a.get(key), b.get(key)
        fmt = lambda v: (f"{v:.3f}" if isinstance(v, float)
                         else ("n/a" if v is None else str(v)))
        print(f"{label:22s} {fmt(fa):>13s} {fmt(fb):>14s}")
    print("\nper constraint (violations / evaluable):")
    for k in a["per_constraint"]:
        pa, pb = a["per_constraint"][k], b["per_constraint"][k]
        tag = "" if pa["independent"] else "  [derived]"
        print(f"  {k}  {pa['violations']:3d}/{pa['evaluable']:<3d}"
              f"   {pb['violations']:3d}/{pb['evaluable']:<3d}{tag}")
    ra, rb = a.get("incoherence_rate"), b.get("incoherence_rate")
    if ra is not None and rb is not None:
        print(f"\nGAP  independent - single = {ra - rb:+.3f}")
        print("A large positive gap means consistency within one response is\n"
              "partly the model reading its own earlier answers, not\n"
              "recomputing. Report both numbers; neither alone is the claim.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(GOLD))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--emit-prompts")
    ap.add_argument("--ollama")
    ap.add_argument("--model")
    ap.add_argument("--score-dir")
    ap.add_argument("--engine")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--single", action="store_true",
                    help="all six questions in ONE call (contrast condition)")
    ap.add_argument("--compare", nargs=2, metavar=("INDEP_DIR", "SINGLE_DIR"),
                    help="print independent vs single-prompt side by side")
    args = ap.parse_args()

    cases = load_cases(Path(args.gold))

    if args.compare:
        return cmd_compare(args)
    if args.emit_prompts:
        return cmd_emit(args, cases)
    if args.ollama:
        if not args.model:
            sys.exit("--ollama requires --model")
        return cmd_run(args, cases)
    if args.score_dir:
        return cmd_score(args, cases)
    if args.engine:
        return cmd_engine(args, cases)
    sys.exit("pick a mode: --emit-prompts | --ollama | --score-dir | --engine")


if __name__ == "__main__":
    main()
    
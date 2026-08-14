#!/usr/bin/env python3
"""The PIPELINE condition for the counterfactual test.

The LLM-verdict condition (run_counterfactual.py) reads perturbed judgment text
and asks the model for a verdict directly. This runs YOUR ACTUAL PIPELINE on the
SAME texts: the LLM extracts temporal facts into a TDG, then the DETERMINISTIC
engine computes the deadline and verdict, scored against the oracle.

This is the pipeline the rest of the thesis uses -- LLMPipeline (the same
extractor as the 2x2 / the 12/14 result), NOT the HeidelTime+spaCy TDGPipeline,
which exists in the repo but produced none of the reported results. Using the
real pipeline is the whole point: the comparison is only fair if "your method"
here is the same "your method" everywhere else.

WHAT THIS ISOLATES
------------------
Three conditions now read the SAME 427 perturbed texts:
  - run_counterfactual.py     LLM reads text -> verdict directly (no structure)
  - THIS FILE                 LLM extracts facts -> engine computes (your method)
  - (oracle)                  engine on gold facts = 100% by construction, not run

The contrast between the first two is the thesis claim: does routing the LLM's
reading THROUGH a structured extract-then-compute step beat asking it for the
answer directly, on identical input? The extraction step is still the LLM, so
this is NOT circular -- extraction can drop or misread a date exactly as in the
2x2, and the engine only supplies the arithmetic once facts exist.

NOTE: extraction here calls the LLM, so this NEEDS the endpoint/GPU and runs at
LLM speed (~427 calls), same cost as run_counterfactual.py.

Usage
-----
    # 1. emit the raw perturbed texts (once)
    python src/counterfactual.py --gold ... --statute-tdg ... \
        --surface anchor --range 30 --auto-center \
        --texts data/experiments/baseline/inputs \
        --emit-texts data/experiments/counterfactual/texts_anchor \
        --out data/experiments/counterfactual/anchor_sweep.json

    # 2. run the pipeline on them
    python src/run_counterfactual_pipeline.py \
        --texts  data/experiments/counterfactual/texts_anchor \
        --oracle data/experiments/counterfactual/anchor_sweep.json \
        --statute-tdg era_1996_s111=data/results_uk/era_1996_s111.json \
        --statute-tdg eqa_2010_s123=data/results_uk/eqa_2010_s123.json \
        --endpoint http://localhost:11434/v1 --model gemma4:e4b \
        --out out/cf_anchor_pipeline_gemma
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tdg_pipeline.llm_pipeline import LLMPipeline            # noqa: E402
from tdg_pipeline.entailment import check_entailment, find_rules  # noqa: E402
from tdg_pipeline.io import load_tdg                        # noqa: E402

_NAME_RE = re.compile(r"^(?P<case>.+)__k(?P<k>[+-]\d+)$")


def _chunks(text: str, size: int, overlap: int = 600) -> list[str]:
    """Split on paragraph boundaries into ~size-char pieces with overlap."""
    if len(text) <= size:
        return [text]
    paras = text.split("\n\n")
    out, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > size:
            out.append(cur)
            cur = cur[-overlap:] + "\n\n" + p
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur:
        out.append(cur)
    return out


def extract_tdg(pipe, text: str, pid: str, chunk_chars: int):
    """Extract a TDG; optionally per-chunk with fact union.

    Only FACTS are merged (deduped on (date_parsed, normalised entity)):
    check_entailment reads instance facts, not instance dependencies, so
    cross-chunk relations are not needed for this condition.
    """
    if chunk_chars <= 0 or len(text) <= chunk_chars:
        return pipe.process(text, document_id=pid, document_type="legal",
                            generate_scenarios=False)
    from tdg_pipeline.tdg import TemporalDependencyGraph
    from tdg_pipeline.entailment import normalise_entity
    merged = TemporalDependencyGraph(document_id=pid, document_type="legal",
                                     source_text=text)
    seen = set()
    for ci, ch in enumerate(_chunks(text, chunk_chars)):
        sub = pipe.process(ch, document_id=f"{pid}__c{ci}",
                           document_type="legal", generate_scenarios=False)
        for f in sub.facts:
            key = (f.timex.date_parsed,
                   normalise_entity(f.entity).lower())
            if key in seen:
                continue
            seen.add(key)
            f.id = f"c{ci}_{f.id}"
            merged.facts.append(f)
    return merged

# which statute governs each case (mirrors the gold set's per-case statute)
CASE_STATUTE = {
    "2026_EAT_64_s111": "era_1996_s111",
    "2026_EAT_64_s123": "eqa_2010_s123",
    "2025_EAT_155": "era_1996_s111",
    "2026_EAT_14": "era_1996_s111",
    "2026_EAT_76": "eqa_2010_s123",
    "2026_EAT_59": "era_1996_s111",
    "kj_2026_EAT_46": "eqa_2010_s123",
}


def load_oracle(path: str) -> dict:
    j = json.loads(Path(path).read_text())
    out = {}
    for cid, c in j["cases"].items():
        for it in c["items"]:
            out[(cid, it["k"])] = it
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True, help="dir of <id>__k<k>.txt from "
                                                   "counterfactual.py --emit-texts")
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--statute-tdg", action="append", default=[],
                    metavar="STATUTE=PATH")
    ap.add_argument("--endpoint", default="http://localhost:11434/v1",
                    help="LLM endpoint for extraction (same as the 2x2)")
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--chunk-chars", type=int, default=0,
                    help="if >0, split documents longer than this into "
                         "~chunk-chars pieces on paragraph boundaries, "
                         "extract each separately, and union the facts "
                         "(dedup on (date, normalised entity)). Single-call "
                         "extraction saturates on long judgments and misses "
                         "buried anchors (2026_EAT_59: the only EDT mention sits "
                         "65% into a 23.6KB document; 61/61 abstentions). "
                         "Costs ~doclen/chunk-chars LLM calls per document. "
                         "12000 is a reasonable value. 0 = off (legacy).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", action="append", default=None,
                    help="run only items whose id contains this substring "
                         "(repeatable). For repairing individual items: "
                         "delete their stale raw/*.tdg.json first, rerun "
                         "with --only, then rebuild results.json for the "
                         "whole directory with src/rescore_cf.py.")
    args = ap.parse_args()

    oracle = load_oracle(args.oracle)
    tdg_paths = dict(kv.split("=", 1) for kv in args.statute_tdg)

    # discover each statute's rule once (same discovery path as run_gold_facts)
    rules = {}
    for statute, p in tdg_paths.items():
        rr = find_rules(load_tdg(p), minus_one_day=None)
        if rr:
            rules[statute] = rr[0]

    pipe = LLMPipeline(model=args.model, base_url=args.endpoint)

    # same entity-matching embedder the 2x2 uses; without it check_entailment
    # falls back to token-Jaccard and abstains more, which would understate the
    # pipeline. Never fatal.
    def _embedder():
        try:
            from tdg_pipeline.embeddings import EmbeddingSimilarity
            emb = EmbeddingSimilarity()
            return emb if emb.is_available() else None
        except Exception:
            return None
    embedder = _embedder()
    statute_tdgs = {s: load_tdg(p) for s, p in tdg_paths.items()}  # load once
    texts = sorted(Path(args.texts).glob("*.txt"))
    if args.only:
        texts = [t for t in texts
                 if any(sub in t.stem for sub in args.only)]
        print(f"--only: {len(texts)} item(s) selected")
    if not texts:
        sys.exit(f"no texts in {args.texts} (run counterfactual.py --emit-texts)")
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    rawdir = outdir / "raw"; rawdir.mkdir(exist_ok=True)
    if args.only is None and any(rawdir.glob("*.tdg.json")):
        print(f"! raw/ already contains extractions from a previous run; "
              f"crashed items would keep their STALE files and confuse "
              f"any rescore. Moving them aside.")
        import time
        aside = outdir / f"raw_prev_{int(time.time())}"
        rawdir.rename(aside); rawdir.mkdir()
        print(f"  previous extractions -> {aside}")

    rows = []
    for i, tf in enumerate(texts, 1):
        pid = tf.stem
        m = _NAME_RE.match(pid)
        if not m:
            continue
        cid, k = m.group("case"), int(m.group("k"))
        truth = oracle.get((cid, k))
        if truth is None:
            continue
        statute = CASE_STATUTE.get(cid)
        rule = rules.get(statute)

        pred_verdict = None
        note = "ok"
        why = None            # entailment explanation -- the diagnosis, keep it
        pred_deadline = None
        pred_anchor = None
        try:
            tdg = extract_tdg(pipe, tf.read_text(errors="ignore"), pid,
                              args.chunk_chars)
            # persist the extraction: without it an abstention run is
            # undiagnosable (learned the hard way on cf_anchor_pipeline_gemma)
            (rawdir / f"{pid}.tdg.json").write_text(
                json.dumps(tdg.to_dict(), indent=1), encoding="utf-8")
            if not tdg.facts:
                note = "no-facts-extracted"
            elif rule is None:
                note = "no-rule"
            else:
                da = date.fromisoformat(truth["acas_day_a"]) if truth.get("acas_day_a") else None
                db = date.fromisoformat(truth["acas_day_b"]) if truth.get("acas_day_b") else None
                res = check_entailment(
                    statute_tdgs[statute], tdg, embedder=embedder,
                    acas_day_a=da, acas_day_b=db)
                if not res:
                    note = "no-rule"
                else:
                    got = res[0].verdict          # same as run_entailment.py
                    why = res[0].explanation
                    pred_deadline = res[0].deadline_computed
                    pred_anchor = res[0].anchor_date
                    if got == "INDETERMINATE":
                        note = "indeterminate"
                    elif got in ("TIMELY", "LATE"):
                        pred_verdict = got
                        note = "answered"
                    else:
                        note = f"verdict={got}"
        except Exception as e:
            note = f"error: {e}"

        gold = truth["verdict"]
        rows.append(dict(id=pid, case=cid, k=k, gold=gold, pred=pred_verdict,
                         match=(pred_verdict == gold), note=note,
                         explanation=why,
                         pred_anchor=pred_anchor,
                         pred_deadline=pred_deadline,
                         gold_deadline=truth["effective_deadline"]))
        if i % 50 == 0:
            print(f"[{i}/{len(texts)}] {pid:34s} gold={gold} pred={pred_verdict} {note}")

    # per-case boundary tracking, same shape as the LLM runner
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(r)
    case_report = {}
    for cid, rs in by_case.items():
        rs.sort(key=lambda r: r["k"])
        true_k = next((rs[i]["k"] for i in range(1, len(rs))
                       if rs[i]["gold"] != rs[i - 1]["gold"]), None)
        answered = [r for r in rs if r["pred"] is not None]
        flips = [rs[i]["k"] for i in range(1, len(rs))
                 if rs[i]["pred"] and rs[i - 1]["pred"]
                 and rs[i]["pred"] != rs[i - 1]["pred"]]
        gap = (min(abs(f - true_k) for f in flips)
               if (flips and true_k is not None) else None)
        case_report[cid] = dict(
            n=len(rs), answered=len(answered),
            accuracy=round(sum(r["match"] for r in answered) / len(answered), 3)
            if answered else None,
            true_boundary_k=true_k, model_flip_ks=flips, flip_gap_days=gap)

    answered = [r for r in rows if r["pred"] is not None]
    correct = sum(r["match"] for r in answered)
    summary = dict(
        n=len(rows), answered=len(answered),
        coverage=round(len(answered) / len(rows), 4) if rows else 0,
        accuracy_answered=round(correct / len(answered), 4) if answered else None,
        accuracy_all=round(correct / len(rows), 4) if rows else None,
        cases=len(by_case))

    # abstention reasons, grouped by first clause of the explanation -- this is
    # the table that tells you WHERE the pipeline lost coverage
    from collections import Counter
    reason = Counter()
    for r in rows:
        if r["pred"] is None:
            e = (r.get("explanation") or r["note"])
            reason[e.split(";")[0].split("--")[0].strip()[:60]] += 1
    summary["abstention_reasons"] = dict(reason.most_common())

    (outdir / "results.json").write_text(json.dumps(
        {"system": f"pipeline-counterfactual (LLMPipeline[{args.model}] "
                   f"extract -> engine)",
         "texts": str(args.texts), "oracle": str(args.oracle),
         "summary": summary, "by_case": case_report, "rows": rows},
        indent=1), encoding="utf-8")

    print("\n== summary (PIPELINE on perturbed texts) ==")
    for k, v in summary.items():
        print(f"  {k:20s} {v}")
    print("\n== per-case: does the PIPELINE flip where the statute says? ==")
    print(f"  {'case':26s} {'acc':>5s} {'cov':>5s} {'true@k':>7s} {'flips':>16s} {'gap':>4s}")
    for cid, c in case_report.items():
        print(f"  {cid[:24]:24s} {str(c['accuracy']):>5s} "
              f"{c['answered']}/{c['n']:<3d} {str(c['true_boundary_k']):>7s} "
              f"{str(c['model_flip_ks'])[:16]:>16s} {str(c['flip_gap_days']):>4s}")
    # abstention causes
    from collections import Counter
    print("\n  notes:", dict(Counter(r["note"].split(":")[0] for r in rows)))
    print(f"\nresults -> {outdir}/results.json")
    print("NOT circular: the LLM re-extracts facts from the perturbed text "
          "(same extractor as the 2x2); the engine only supplies the arithmetic "
          "once facts exist. Contrast with run_counterfactual.py (LLM answers "
          "directly) to see whether the structured step helps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    
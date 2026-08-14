#!/usr/bin/env python3
"""Run and score the counterfactual LLM condition.

counterfactual.py emits one prompt per perturbed item (a real judgment with one
date shifted) and an oracle file with the ENGINE-COMPUTED verdict for each. This
runs those prompts through the model and scores the model's verdict against the
oracle. The engine is the ground-truth generator here; the MODEL is under test.

The experiment (C1/C7 as rates, not anecdotes):
  - does the model's verdict flip within a day or two of the true boundary,
    or does it drift / never flip?
  - measured over hundreds of items whose answers are computed, not annotated,
    on documents published after every model's training cutoff.

Two phases, same as run_external:
  --emit is already done by counterfactual.py. This script:
    (default)     call the model on every prompt, score, write results.json
    --score-dir D score responses already sitting in D as <id>.txt

Usage
-----
    python src/external_bench/run_counterfactual.py \
        --prompts data/experiments/counterfactual/prompts_anchor \
        --oracle  data/experiments/counterfactual/anchor_sweep.json \
        --endpoint http://localhost:11434/v1 --model gemma4:e4b \
        --out out/cf_anchor_gemma
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_external import call_chat  # noqa: E402  (reuse the exact model path)

# prompt filename is "<case_id>__k<+NNNN>.txt"; recover (case_id, k) from it
_NAME_RE = re.compile(r"^(?P<case>.+)__k(?P<k>[+-]\d+)$")
_VERDICT_RE = re.compile(r"VERDICT\s*[:=]\s*(in[_\s-]?time|out[_\s-]?of[_\s-]?time|"
                         r"timely|late)", re.I)
_DEADLINE_RE = re.compile(r"DEADLINE\s*[:=]\s*(\d{4}-\d{2}-\d{2})", re.I)


def parse_verdict(text: str):
    """Extract in/out verdict from a model response. Prefer the explicit VERDICT
    line; fall back to the last timely/late token so a model that reasons well
    but formats loosely is not scored as an abstention."""
    m = _VERDICT_RE.search(text or "")
    tok = None
    if m:
        tok = m.group(1).lower()
    else:
        hits = re.findall(r"\b(in[_\s-]?time|out[_\s-]?of[_\s-]?time|timely|late)\b",
                          text or "", re.I)
        if hits:
            tok = hits[-1].lower()
    if tok is None:
        return None
    tok = tok.replace(" ", "").replace("-", "").replace("_", "")
    if tok in ("intime", "timely"):
        return "TIMELY"
    if tok in ("outoftime", "late"):
        return "LATE"
    return None


def parse_deadline(text: str):
    m = _DEADLINE_RE.search(text or "")
    return m.group(1) if m else None


def load_oracle(path: str) -> dict:
    """(case_id, k) -> {verdict, effective_deadline, anchor, ...} from the sweep."""
    j = json.loads(Path(path).read_text())
    out = {}
    for cid, c in j["cases"].items():
        for it in c["items"]:
            out[(cid, it["k"])] = it
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="dir of <id>__k<k>.txt")
    ap.add_argument("--oracle", required=True, help="anchor_sweep.json from "
                                                    "counterfactual.py")
    ap.add_argument("--endpoint", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--num-batch", type=int, default=64)
    ap.add_argument("--score-dir", help="score existing <id>.txt responses here "
                                        "instead of calling the model")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    oracle = load_oracle(args.oracle)
    prompts = sorted(Path(args.prompts).glob("*.txt"))
    if not prompts:
        sys.exit(f"no prompts in {args.prompts}")
    outdir = Path(args.out); (outdir / "raw").mkdir(parents=True, exist_ok=True)

    rows, infra = [], 0
    for i, pf in enumerate(prompts, 1):
        pid = pf.stem
        m = _NAME_RE.match(pid)
        if not m:
            continue
        cid, k = m.group("case"), int(m.group("k"))
        truth = oracle.get((cid, k))
        if truth is None:
            continue

        # obtain the model response (fresh call, or a saved one)
        if args.score_dir:
            rf = Path(args.score_dir) / f"{pid}.txt"
            resp = rf.read_text(errors="ignore") if rf.exists() else ""
        else:
            try:
                resp = call_chat(args.endpoint, args.model, pf.read_text(),
                                 max_tokens=args.max_tokens,
                                 num_ctx=args.num_ctx, num_batch=args.num_batch)
            except Exception as e:
                resp = ""
                infra += 1
                print(f"[{i}/{len(prompts)}] {pid:38s} CALL ERROR: {e}")
        (outdir / "raw" / f"{pid}.txt").write_text(resp or "", encoding="utf-8")

        pred = parse_verdict(resp)
        gold = truth["verdict"]
        match = pred is not None and pred == gold
        rows.append(dict(id=pid, case=cid, k=k, gold=gold, pred=pred,
                         match=match,
                         gold_deadline=truth["effective_deadline"],
                         pred_deadline=parse_deadline(resp),
                         anchor=truth["anchor"]))
        if not args.score_dir and i % 20 == 0:
            print(f"[{i}/{len(prompts)}] {pid:38s} gold={gold} pred={pred} "
                  f"{'OK' if match else 'X'}")

    # -- per-case boundary analysis: where does the MODEL flip vs the oracle? --
    by_case: dict = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(r)
    case_report = {}
    for cid, rs in by_case.items():
        rs.sort(key=lambda r: r["k"])
        true_k = next((rs[i]["k"] for i in range(1, len(rs))
                       if rs[i]["gold"] != rs[i - 1]["gold"]), None)
        answered = [r for r in rs if r["pred"] is not None]
        model_ks = [rs[i]["k"] for i in range(1, len(rs))
                    if rs[i]["pred"] and rs[i - 1]["pred"]
                    and rs[i]["pred"] != rs[i - 1]["pred"]]
        acc = sum(r["match"] for r in answered) / len(answered) if answered else None
        # distance from the model's nearest flip to the true boundary
        gap = (min(abs(mk - true_k) for mk in model_ks)
               if (model_ks and true_k is not None) else None)
        case_report[cid] = dict(
            n=len(rs), answered=len(answered),
            accuracy=round(acc, 3) if acc is not None else None,
            true_boundary_k=true_k, model_flip_ks=model_ks,
            flip_gap_days=gap)

    answered = [r for r in rows if r["pred"] is not None]
    correct = sum(r["match"] for r in answered)
    summary = dict(
        n=len(rows), answered=len(answered),
        coverage=round(len(answered) / len(rows), 4) if rows else 0,
        accuracy_answered=round(correct / len(answered), 4) if answered else None,
        accuracy_all=round(correct / len(rows), 4) if rows else None,
        call_errors_infra=infra,
        cases=len(by_case))

    (outdir / "results.json").write_text(json.dumps(
        {"model": args.model, "system": "llm-counterfactual",
         "prompts": str(args.prompts), "oracle": str(args.oracle),
         "summary": summary, "by_case": case_report, "rows": rows},
        indent=1), encoding="utf-8")

    print("\n== summary ==")
    for k, v in summary.items():
        print(f"  {k:22s} {v}")
    print("\n== per-case: does the model flip where the statute says? ==")
    print(f"  {'case':26s} {'acc':>5s} {'true@k':>7s} {'model flips@k':>16s} {'gap(d)':>6s}")
    for cid, c in case_report.items():
        print(f"  {cid[:24]:24s} {str(c['accuracy']):>5s} "
              f"{str(c['true_boundary_k']):>7s} {str(c['model_flip_ks'])[:16]:>16s} "
              f"{str(c['flip_gap_days']):>6s}")
    print(f"\nresults -> {outdir}/results.json   raw -> {outdir}/raw/")
    print("NOTE: the engine generated these labels; it is the oracle, not a "
          "competitor. This scores the MODEL against the statute-computed truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

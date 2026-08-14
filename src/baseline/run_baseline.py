#!/usr/bin/env python3
"""run_baseline.py -- LLM baseline on the gold timeliness cases.

Task per case: redacted judgment + governing statute section -> the model
must return the anchor date, primary deadline, effective deadline (after
any ACAS early-conciliation extension), verdict, and its arithmetic.

Scoring (the thesis metric is DELTA_DAYS, not verdicts):
  verdict_match   vs gold verdict
  deadline_delta  model effective_deadline vs reference deadline, in days.
                  reference = judge-stated deadline where the judgment
                  recites one (quote-backed), else the engine's verified
                  effective deadline from results_gold_facts.json (the
                  extraction-independent upper bound). Reference source is
                  reported per row -- never conflated.
  anchor_delta    model anchor vs gold anchor (anchor-selection errors are
                  a predicted failure class: internal-appeal anchors,
                  continuing-act endpoints).
Rows whose reference deadline is unavailable (2026_EAT_59: EC dates absent
from the source document) are scored on verdict + anchor only.

Modes:
  --emit-prompts DIR            write one prompt file per case (for API
                                models run elsewhere); no scoring.
  --ollama URL --model NAME     run against an Ollama endpoint now.
  --score-dir DIR               score saved responses (<case>.<model>.json
                                or <case>.txt files containing the model's
                                raw output).
Prompts are IDENTICAL across models. The prompt template is frozen in
PROMPT below -- do not edit between models.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "ground_truth" / "ground_truth_gold.json"
UPPER = ROOT / "data" / "experiments" / "gold_facts" / "results_gold_facts.json"
INPUTS = ROOT / "data" / "experiments" / "baseline" / "inputs"
STATUTES = ROOT / "data" / "statutes" / "txt"

CASE_FILE = {   # gold row -> redacted input file
    "2026_EAT_64_s111": "2026_EAT_64.txt",
    "2026_EAT_64_s123": "2026_EAT_64.txt",
    "2025_EAT_155": "2025_EAT_155.txt",
    "2026_EAT_14": "2026_EAT_14.txt",
    "2026_EAT_76": "2026_EAT_76.txt",
    "2026_EAT_59": "2026_EAT_59.txt",
    "kj_2026_EAT_46": "2026_EAT_46.txt",
}

PROMPT = """\
You are given (A) the text of a UK statute section governing time limits and
(B) a UK Employment Appeal Tribunal judgment from which the tribunal's
timeliness conclusions have been redacted (marked [REDACTED: ...]). All
primary facts and dates remain in the document.

Your task, for the complaint under the statute in (A){claim_hint}:
1. Identify the anchor date from which the limitation period runs.
2. Compute the primary limitation deadline under the statute.
3. If ACAS early conciliation dates (Day A / Day B) appear in the facts,
   compute the effective deadline after the early-conciliation extension
   (the period beginning with the day after Day A and ending with Day B is
   not counted; a claim is also in time if presented within one month after
   Day B where Day A fell within the primary period).
4. State whether the claim was presented in time.
Show your date arithmetic explicitly, then output ONLY a JSON object, no
markdown fences, exactly:
{{"anchor_date":"YYYY-MM-DD","primary_deadline":"YYYY-MM-DD",\
"effective_deadline":"YYYY-MM-DD","verdict":"in_time" or "out_of_time",\
"arithmetic":"<your working as one string>"}}

(A) STATUTE:
{statute}

(B) JUDGMENT (redacted):
{judgment}
"""

CLAIM_HINT = {  # only where one statute covers multiple candidate complaints
    # (2026_EAT_64 pair needs none: the statute text itself identifies the complaint)
    "2026_EAT_76": " (the race-related harassment complaint concerning "
                         "the incident at the dinner)",
    "kj_2026_EAT_46": " (the claims concerning Mr Reilly's harassment)",
}


def _d(s):
    try:
        return date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def load_cases():
    gold = json.loads(GOLD.read_text())["cases"]
    upper = {r["case"]: r for r in json.loads(UPPER.read_text())["rows"]} \
        if UPPER.exists() else {}
    cases = {}
    for cid, row in gold.items():
        stat_txt = (STATUTES / f"{row['statute']}.txt").read_text(encoding="utf-8")
        judg = (INPUTS / CASE_FILE[cid]).read_text(encoding="utf-8")
        stated = row.get("deadline_stated")
        eng = upper.get(cid, {}).get("effective_deadline")
        if stated:
            ref, ref_src = stated, "judge-stated"
        elif row.get("status", "").startswith("conditional"):
            ref, ref_src = None, "none (conditional row: EC dates absent from source)"
        else:
            ref, ref_src = eng, "engine-upper-bound"
        cases[cid] = {
            "prompt": PROMPT.format(statute=stat_txt, judgment=judg,
                                    claim_hint=CLAIM_HINT.get(cid, "")),
            "gold_verdict": row["verdict"],
            "gold_anchor": row["anchor_date"],
            "ref_deadline": ref, "ref_source": ref_src,
        }
    return cases


def parse_answer(raw):
    m = re.search(r"\{.*\}", raw, re.S)   # last-ditch: the JSON object itself
    if not m:
        return None, "no JSON object found"
    try:
        return json.loads(m.group(0)), None
    except Exception as e:
        return None, f"JSON parse error: {e}"


def score(cid, case, ans, err, model):
    if ans is None:
        return {"case": cid, "model": model, "error": err}
    v = ans.get("verdict")
    eff, anc = _d(ans.get("effective_deadline")), _d(ans.get("anchor_date"))
    ref, ganc = _d(case["ref_deadline"]), _d(case["gold_anchor"])
    return {
        "case": cid, "model": model,
        "model_verdict": v, "gold_verdict": case["gold_verdict"],
        "verdict_match": v == case["gold_verdict"],
        "model_anchor": ans.get("anchor_date"), "gold_anchor": case["gold_anchor"],
        "anchor_delta_days": (anc - ganc).days if anc and ganc else None,
        "model_effective_deadline": ans.get("effective_deadline"),
        "ref_deadline": case["ref_deadline"], "ref_source": case["ref_source"],
        "deadline_delta_days": (eff - ref).days if eff and ref else None,
        "arithmetic": ans.get("arithmetic", ""),
    }


def call_openai(base_url, model, prompt, api_key):
    body = json.dumps({"model": model, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    opener = urllib.request.build_opener()  # external API: use env proxy if any
    with opener.open(req, timeout=1200) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def call_ollama(url, model, prompt, num_ctx=32768):
    body = json.dumps({"model": model, "stream": False,
                       "messages": [{"role": "user", "content": prompt}],
                       "options": {"temperature": 0, "num_ctx": num_ctx}}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    # bypass any environment proxy: cluster proxies can't reach localhost
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=1200) as r:
        return json.loads(r.read())["message"]["content"]


def report(rows):
    w = max(len(r["case"]) for r in rows) + 1
    print(f"\n{'case':<{w}} {'model':<18} {'verdict':<22} "
          f"{'anchor d':>9} {'deadline d':>11}  ref")
    for r in rows:
        if "error" in r:
            print(f"{r['case']:<{w}} {r['model']:<18} PARSE-FAIL: {r['error']}")
            continue
        vm = f"{r['model_verdict']}/{r['gold_verdict']}" \
             + (" OK" if r["verdict_match"] else " XX")
        ad = "-" if r["anchor_delta_days"] is None else f"{r['anchor_delta_days']:+}"
        dd = "-" if r["deadline_delta_days"] is None else f"{r['deadline_delta_days']:+}"
        print(f"{r['case']:<{w}} {r['model']:<18} {vm:<22} {ad:>9} {dd:>11}  "
              f"{r['ref_source']}")
    ok = [r for r in rows if r.get("verdict_match") is not None]
    dd = [abs(r["deadline_delta_days"]) for r in rows
          if r.get("deadline_delta_days") is not None]
    exact = sum(1 for x in dd if x == 0)
    print(f"\nverdicts: {sum(r['verdict_match'] for r in ok)}/{len(ok)} | "
          f"deadlines exact: {exact}/{len(dd)} | "
          f"mean |delta_days|: {sum(dd)/len(dd):.1f}" if dd else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-prompts", metavar="DIR")
    ap.add_argument("--ollama", metavar="URL")
    ap.add_argument("--openai", action="store_true",
                    help="use an OpenAI-compatible API; key from OPENAI_API_KEY")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--save-raw", metavar="DIR",
                    help="save each raw model response as <case>.<model>.txt")
    ap.add_argument("--score-dir", metavar="DIR")
    ap.add_argument("--out", metavar="JSON")
    args = ap.parse_args()
    cases = load_cases()

    if args.emit_prompts:
        d = Path(args.emit_prompts); d.mkdir(parents=True, exist_ok=True)
        for cid, c in cases.items():
            (d / f"{cid}.prompt.txt").write_text(c["prompt"], encoding="utf-8")
        print(f"wrote {len(cases)} prompts -> {d}")
        return 0

    rows = []
    if args.score_dir:
        for f in sorted(Path(args.score_dir).glob("*")):
            cid = f.name.split(".")[0]
            if cid not in cases:
                continue
            model = f.name.split(".")[1] if f.name.count(".") >= 2 else "saved"
            ans, err = parse_answer(f.read_text(encoding="utf-8"))
            rows.append(score(cid, cases[cid], ans, err, model))
    elif args.ollama or args.openai:
        import os
        key = os.environ.get("OPENAI_API_KEY")
        if args.openai and not key:
            print("set OPENAI_API_KEY in the environment first"); return 2
        rawdir = Path(args.save_raw) if args.save_raw else None
        if rawdir:
            rawdir.mkdir(parents=True, exist_ok=True)
        for cid, c in cases.items():
            if args.openai:
                raw = call_openai(args.base_url, args.model, c["prompt"], key)
            else:
                raw = call_ollama(args.ollama, args.model, c["prompt"])
            if rawdir:
                safe = args.model.replace("/", "_").replace(":", "_")
                (rawdir / f"{cid}.{safe}.txt").write_text(raw, encoding="utf-8")
            ans, err = parse_answer(raw)
            rows.append(score(cid, c, ans, err, args.model))
    else:
        print("choose --emit-prompts, --ollama, or --score-dir"); return 2

    report(rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
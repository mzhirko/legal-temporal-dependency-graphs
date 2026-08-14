#!/usr/bin/env python3
"""
Head-to-head comparison of the rule-based and LLM extractors, on the
same documents.

Matches documents by document_id, so it only compares like with like and
tells you loudly if the two sets diverge.

Emits a console table, an aggregate summary, and LaTeX for tab:deps.

Usage:
    python compare_ht_llm.py \
        --ht  data/results_heideltime_real \
        --llm data/results_contracts_50 \
        --latex

    # restrict to the four seeds in the current table
    python compare_ht_llm.py --ht ... --llm ... \
        --only en_contracts_seed0 en_contracts_seed2 \
               en_contracts_seed3 en_contracts_seed4 --latex
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            j = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"  skipped unparseable {f.name}")
            continue
        doc_id = j.get("document_id", f.stem)
        deps = j.get("dependencies", []) or []
        out[doc_id] = {
            "facts": len(j.get("facts", []) or []),
            "deps": len(deps),
            "dep_types": Counter(
                (d.get("constraint_type") or d.get("relation_type") or d.get("type") or "?") for d in deps
            ),
            "chars": len(j.get("source_text", "") or ""),
            "stage": (j.get("_provenance") or {}).get("stage_counts", {}),
            "backend": (j.get("_provenance") or {}).get("extractor_backend"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ht", required=True, type=Path)
    ap.add_argument("--llm", required=True, type=Path)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--label", default="Gemma", help="column name for the LLM run")
    args = ap.parse_args()

    ht, llm = load(args.ht), load(args.llm)

    backends = {v["backend"] for v in ht.values()}
    if backends != {"heideltime"}:
        print(f"WARNING: rule-based backends present: {backends}. "
              f"Anything other than 'heideltime' must not be reported as "
              f"HeidelTime.\n")

    ids = sorted(set(ht) & set(llm))
    if args.only:
        ids = [i for i in args.only if i in ht and i in llm]
        missing = [i for i in args.only if i not in ht or i not in llm]
        if missing:
            print(f"WARNING: requested but unavailable: {missing}\n")

    only_ht, only_llm = sorted(set(ht) - set(llm)), sorted(set(llm) - set(ht))
    if only_ht:
        print(f"in HT only ({len(only_ht)}): {only_ht[:6]}{' ...' if len(only_ht) > 6 else ''}")
    if only_llm:
        print(f"in LLM only ({len(only_llm)}): {only_llm[:6]}{' ...' if len(only_llm) > 6 else ''}")
    print(f"comparing {len(ids)} documents present in both\n")

    hdr = (f"{'document':<26}{'timex':>7}{'HTfact':>8}{'HTdep':>7}"
           f"{'LLMfact':>9}{'LLMdep':>8}   HT types / LLM types")
    print(hdr)
    print("-" * (len(hdr) + 12))

    tot = Counter()
    ht_types_all, llm_types_all = Counter(), Counter()
    for i in ids:
        h, l = ht[i], llm[i]
        timex = h["stage"].get("raw_timex", "-")
        tot["timex"] += timex if isinstance(timex, int) else 0
        tot["htf"] += h["facts"]; tot["htd"] += h["deps"]
        tot["lmf"] += l["facts"]; tot["lmd"] += l["deps"]
        ht_types_all.update(h["dep_types"]); llm_types_all.update(l["dep_types"])
        ht_t = ",".join(f"{k}:{v}" for k, v in sorted(h["dep_types"].items())) or "-"
        lm_t = ",".join(f"{k}:{v}" for k, v in sorted(l["dep_types"].items())) or "-"
        print(f"{i:<26}{timex:>7}{h['facts']:>8}{h['deps']:>7}"
              f"{l['facts']:>9}{l['deps']:>8}   {ht_t} / {lm_t}")

    print("-" * (len(hdr) + 12))
    print(f"{'TOTAL':<26}{tot['timex']:>7}{tot['htf']:>8}{tot['htd']:>7}"
          f"{tot['lmf']:>9}{tot['lmd']:>8}")

    n = len(ids)
    print(f"\ndocuments with >=1 dependency:  "
          f"rule-based {sum(1 for i in ids if ht[i]['deps'] > 0)}/{n}   "
          f"{args.label} {sum(1 for i in ids if llm[i]['deps'] > 0)}/{n}")
    print(f"dependency types  rule-based: {dict(ht_types_all) or 'none'}")
    print(f"dependency types  {args.label}: {dict(llm_types_all) or 'none'}")
    print(f"facts  rule-based {tot['htf']}  vs  {args.label} {tot['lmf']}")
    if tot["timex"]:
        print(f"timex -> facts retention (rule-based): "
              f"{tot['htf']}/{tot['timex']} ({100 * tot['htf'] / tot['timex']:.1f}%)")

    if tot["htd"] == 0:
        print("\n>>> zero-dependency finding HOLDS on this set.")
    else:
        print(f"\n>>> zero-dependency finding DOES NOT HOLD: {tot['htd']} edges "
              f"across {sum(1 for i in ids if ht[i]['deps'] > 0)} documents.")
        print("    Inspect them before rewording anything:")
        for i in ids:
            if ht[i]["deps"]:
                print(f"      {i}: {dict(ht[i]['dep_types'])}")

    if args.latex:
        print("\n" + "=" * 62 + "\nLaTeX for tab:deps\n" + "=" * 62)
        print(r"\begin{table}[t]")
        print(r"\centering")
        print(r"\caption{Facts and dependencies extracted per document.")
        print(r"HT = HeidelTime + spaCy (rule-based). " + args.label +
              r" = the LLM pipeline.}")
        print(r"\label{tab:deps}")
        print(r"\begin{tabular}{lcccc}")
        print(r"\toprule")
        print(r"Document & \multicolumn{2}{c}{Facts} &"
              r" \multicolumn{2}{c}{Dependencies} \\")
        print(r"         & HT & " + args.label + r" & HT & " + args.label + r" \\")
        print(r"\midrule")
        for i in ids:
            h, l = ht[i], llm[i]
            hd = (f"{h['deps']}~(\\texttt{{{max(h['dep_types'], key=h['dep_types'].get)}}})"
                  if h["deps"] else "0")
            ld = (f"{l['deps']}~(\\texttt{{{max(l['dep_types'], key=l['dep_types'].get)}}})"
                  if l["deps"] else "0")
            print(f"{i.replace('_', chr(92) + '_')} & {h['facts']} & {l['facts']}"
                  f" & {hd} & {ld} \\\\")
        print(r"\midrule")
        print(f"Total & {tot['htf']} & {tot['lmf']} & {tot['htd']} & {tot['lmd']} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

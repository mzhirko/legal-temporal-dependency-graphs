#!/usr/bin/env python3
"""
Audit the rule-based dependency edges and detect duplicate documents.

Two questions this answers:

  1. Are the 11 edges real dependencies or artefacts? For each edge it
     prints both endpoint facts with entity, role, value, signal verb and
     the source sentence, so you can judge. It also flags the mechanical
     tells: entity names that are clause fragments rather than entities,
     endpoints in different sentences, low confidence.

  2. Is the sample smaller than it looks? Identical source_text across
     seeds means Multi_Legal_Pile returned the same document more than
     once, which inflates every aggregate.

Usage:
    python audit_edges.py --ht ../data/results_heideltime_real \
                          --llm ../data/results_contracts_50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Entity strings that are clause fragments, pronouns, or boilerplate
# rather than a nameable party or instrument.
FRAGMENT_TELLS = re.compile(
    r"^(either|each|any|both|such|the|this|that|these|those|it|its|"
    r"he|she|they|them|his|her|their|a|an|no)\b.{0,30}$",
    re.IGNORECASE,
)
GENERIC = {
    "party", "parties", "either party", "each party", "both parties",
    "it", "they", "them", "the parties", "the party", "unknown", "",
}


def load_dir(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            out[f.stem] = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"  unparseable: {f.name}")
    return out


def facts_by_id(doc: dict) -> dict:
    return {f["id"]: f for f in doc.get("facts", []) or []}


def sent_of(fact: dict) -> str:
    s = (fact.get("sentence") or "").strip().replace("\n", " ")
    return re.sub(r"\s+", " ", s)


def suspicious(entity: str) -> list[str]:
    flags = []
    e = (entity or "").strip()
    if e.lower() in GENERIC:
        flags.append("generic-entity")
    elif FRAGMENT_TELLS.match(e):
        flags.append("clause-fragment")
    if len(e) < 3:
        flags.append("too-short")
    if e and e[0].islower():
        flags.append("lowercase-start")
    if len(e.split()) > 8:
        flags.append("over-long")
    return flags


def audit_edges(docs: dict, label: str) -> None:
    print("=" * 78)
    print(f"EDGE AUDIT -- {label}")
    print("=" * 78)

    total = 0
    flagged = 0
    conf_hist: Counter = Counter()
    cross_sentence = 0

    for doc_id, doc in sorted(docs.items()):
        deps = doc.get("dependencies", []) or []
        if not deps:
            continue
        fmap = facts_by_id(doc)
        print(f"\n--- {doc_id}  ({len(deps)} edge(s), {len(fmap)} facts)")

        for i, dep in enumerate(deps, 1):
            total += 1
            src = fmap.get(dep.get("from_id"), {})
            tgt = fmap.get(dep.get("to_id"), {})
            conf = dep.get("confidence")
            conf_hist[conf] += 1

            s_ent, t_ent = src.get("entity", "?"), tgt.get("entity", "?")
            flags = []
            for who, ent in (("from", s_ent), ("to", t_ent)):
                for fl in suspicious(ent):
                    flags.append(f"{who}:{fl}")
            if s_ent != t_ent:
                flags.append("endpoints-differ")
            s_sent, t_sent = sent_of(src), sent_of(tgt)
            if s_sent and t_sent and s_sent != t_sent:
                flags.append("cross-sentence")
                cross_sentence += 1
            if isinstance(conf, (int, float)) and conf < 0.7:
                flags.append(f"low-conf({conf})")
            if flags:
                flagged += 1

            print(f"  [{i}] {dep.get('constraint_type')}  "
                  f"{dep.get('constraint_expr')}  "
                  f"delta={dep.get('delta_days')}  conf={conf}")
            print(f"      from {dep.get('from_id')}: entity={s_ent!r} "
                  f"role={src.get('role')} value={src.get('value')!r} "
                  f"verb={src.get('signal_verb')!r}")
            print(f"        \"{s_sent[:150]}\"")
            print(f"      to   {dep.get('to_id')}: entity={t_ent!r} "
                  f"role={tgt.get('role')} value={tgt.get('value')!r} "
                  f"verb={tgt.get('signal_verb')!r}")
            print(f"        \"{t_sent[:150]}\"")
            if flags:
                print(f"      FLAGS: {', '.join(flags)}")

    print("\n" + "-" * 78)
    print(f"{label}: {total} edges, {flagged} carrying at least one flag, "
          f"{cross_sentence} spanning different sentences")
    print(f"confidence distribution: {dict(sorted(conf_hist.items(), key=lambda x: (x[0] is None, x[0])))}")
    print("\nFlags are heuristics, not verdicts. Read the sentences and\n"
          "decide yourself whether each edge is a dependency a lawyer\n"
          "would recognise.")


def duplicates(docs: dict, label: str) -> dict:
    print("\n" + "=" * 78)
    print(f"DUPLICATE DETECTION -- {label}")
    print("=" * 78)

    by_hash: dict[str, list[str]] = defaultdict(list)
    for doc_id, doc in sorted(docs.items()):
        txt = doc.get("source_text") or ""
        if not txt:
            continue
        by_hash[hashlib.sha256(txt.encode()).hexdigest()].append(doc_id)

    dupes = {h: ids for h, ids in by_hash.items() if len(ids) > 1}
    if not dupes:
        print("no exact duplicates found")
        return {}

    n_extra = sum(len(ids) - 1 for ids in dupes.values())
    print(f"{len(dupes)} document(s) appear more than once; "
          f"{n_extra} redundant file(s)\n")
    for h, ids in sorted(dupes.items(), key=lambda kv: -len(kv[1])):
        first = docs[ids[0]]
        print(f"  {len(ids)}x  {ids}")
        print(f"       chars={len(first.get('source_text',''))}  "
              f"facts={len(first.get('facts',[]) or [])}  "
              f"deps={len(first.get('dependencies',[]) or [])}")
        print(f"       sha256={h[:16]}")

    keep = sorted(ids[0] for ids in by_hash.values())
    print(f"\nunique documents: {len(keep)} of {len(docs)}")
    print("Deduplicated seed list (keeps the lowest seed of each group):")
    print("  " + " ".join(keep))
    return {"unique": keep, "groups": dupes}


def dedup_totals(docs: dict, keep: list[str], label: str) -> None:
    f = sum(len(docs[k].get("facts", []) or []) for k in keep)
    d = sum(len(docs[k].get("dependencies", []) or []) for k in keep)
    withdep = sum(1 for k in keep if docs[k].get("dependencies"))
    types: Counter = Counter()
    for k in keep:
        for dep in docs[k].get("dependencies", []) or []:
            types[dep.get("constraint_type") or "?"] += 1
    print(f"\n{label} deduplicated: {len(keep)} docs, {f} facts, {d} edges, "
          f"{withdep} docs with >=1 edge, types={dict(types) or 'none'}")
    if f:
        print(f"  edges per fact: {d}/{f} = {100 * d / f:.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ht", required=True, type=Path)
    ap.add_argument("--llm", type=Path, default=None)
    args = ap.parse_args()

    ht = load_dir(args.ht)
    audit_edges(ht, "rule-based")
    info = duplicates(ht, "rule-based")

    if info:
        dedup_totals(ht, info["unique"], "rule-based")
        if args.llm:
            llm = load_dir(args.llm)
            keep = [k for k in info["unique"] if k in llm]
            dedup_totals(llm, keep, "LLM")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

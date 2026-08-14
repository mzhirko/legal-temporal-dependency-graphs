#!/usr/bin/env python3
"""
Audit both extractors on the same documents.

Answers three questions:

  1. Is a "fact" the same unit on both sides? It is not, and comparing
     raw counts is misleading. This breaks facts down by timex type,
     whether a calendar date actually resolved, entity quality, and
     whether the value is a bare year (citation years look like dates).
     It then reports a like-for-like subset: a resolved calendar date
     attached to a nameable entity.

  2. Are the rule-based edges real? Prints both endpoints of every edge
     with entity, role, value, signal verb and source sentence.

  3. Is the sample smaller than it looks? Exact-duplicate source_text
     across seeds inflates every aggregate.

Usage:
    python audit_all.py --ht  ../data/results_heideltime_real \
                        --llm ../data/results_contracts_50

    python audit_all.py --ht ... --llm ... --edges-only
    python audit_all.py --ht ... --llm ... > audit.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

BARE_YEAR = re.compile(r"^\s*\(?\s*(1[89]\d{2}|20\d{2})\s*\)?\s*$")
FRAGMENT = re.compile(
    r"^(either|each|any|both|such|the|this|that|these|those|it|its|he|she|"
    r"they|them|his|her|their|a|an|no|and|or|of|in|on|at|by|for)\b.{0,40}$",
    re.IGNORECASE,
)
GENERIC = {
    "", "unknown", "none", "null", "party", "parties", "either party",
    "each party", "both parties", "the parties", "the party", "it",
    "they", "them", "document", "the document", "agreement",
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


def bad_entity(e: str | None) -> str | None:
    s = (e or "").strip()
    if s.lower() in GENERIC:
        return "generic"
    if not s:
        return "empty"
    if len(s) < 3:
        return "too-short"
    if FRAGMENT.match(s):
        return "clause-fragment"
    if len(s.split()) > 8:
        return "over-long"
    return None


def profile_facts(docs: dict, keep: list[str], label: str) -> dict:
    tt, roles, ent_bad = Counter(), Counter(), Counter()
    n = parsed = bare_year = dup_val = usable = 0
    ent_examples: Counter = Counter()

    for k in keep:
        seen_vals: Counter = Counter()
        for f in docs[k].get("facts", []) or []:
            n += 1
            tt[f.get("timex_type") or "?"] += 1
            roles[f.get("role") or "?"] += 1
            has_date = bool(f.get("date_parsed"))
            parsed += has_date
            raw = str(f.get("raw_text") or f.get("value") or "")
            is_year = bool(BARE_YEAR.match(raw))
            bare_year += is_year
            ent = f.get("entity")
            flag = bad_entity(ent)
            if flag:
                ent_bad[flag] += 1
            else:
                ent_examples[str(ent)[:40]] += 1
            key = (f.get("entity"), f.get("value"))
            seen_vals[key] += 1
            if seen_vals[key] > 1:
                dup_val += 1
            if has_date and not is_year and not flag:
                usable += 1

    pct = lambda x: f"{100 * x / n:.1f}%" if n else "-"
    print(f"\n{label}: {n} facts across {len(keep)} unique documents")
    print(f"  timex types            {dict(tt)}")
    print(f"  roles                  {dict(roles)}")
    print(f"  resolved a date        {parsed} ({pct(parsed)})")
    print(f"  bare year only         {bare_year} ({pct(bare_year)})  "
          f"<- citation years look like dates")
    print(f"  unusable entity        {sum(ent_bad.values())} "
          f"({pct(sum(ent_bad.values()))})  {dict(ent_bad)}")
    print(f"  repeat (entity,value)  {dup_val} ({pct(dup_val)})")
    print(f"  USABLE NODES           {usable} ({pct(usable)})  "
          f"= resolved date + not a bare year + nameable entity")
    if ent_examples:
        print(f"  top entities           {ent_examples.most_common(6)}")
    return {"n": n, "usable": usable, "parsed": parsed}


def audit_edges(docs: dict, keep: list[str], label: str, limit: int) -> None:
    print("\n" + "=" * 78)
    print(f"EDGES -- {label}")
    print("=" * 78)
    total = flagged = 0
    conf: Counter = Counter()
    shown = 0

    for k in keep:
        deps = docs[k].get("dependencies", []) or []
        if not deps:
            continue
        fmap = {f["id"]: f for f in docs[k].get("facts", []) or []}
        header_done = False
        for i, dep in enumerate(deps, 1):
            total += 1
            conf[dep.get("confidence")] += 1
            src, tgt = fmap.get(dep.get("from_id"), {}), fmap.get(dep.get("to_id"), {})
            flags = []
            for who, fx in (("from", src), ("to", tgt)):
                fl = bad_entity(fx.get("entity"))
                if fl:
                    flags.append(f"{who}:{fl}")
            s_sent = re.sub(r"\s+", " ", (src.get("sentence") or "")).strip()
            t_sent = re.sub(r"\s+", " ", (tgt.get("sentence") or "")).strip()
            if s_sent and t_sent and s_sent != t_sent:
                flags.append("cross-sentence")
            c = dep.get("confidence")
            if isinstance(c, (int, float)) and c < 0.7:
                flags.append(f"low-conf({c})")
            if flags:
                flagged += 1

            if limit and shown >= limit:
                continue
            shown += 1
            if not header_done:
                print(f"\n--- {k}  ({len(deps)} edge(s), {len(fmap)} facts)")
                header_done = True
            print(f"  [{i}] {dep.get('constraint_type')}  "
                  f"{dep.get('constraint_expr')}  "
                  f"delta={dep.get('delta_days')}  conf={c}")
            print(f"      from {dep.get('from_id')}: entity={src.get('entity')!r} "
                  f"role={src.get('role')} value={src.get('value')!r} "
                  f"verb={src.get('signal_verb')!r}")
            print(f"        \"{s_sent[:160]}\"")
            print(f"      to   {dep.get('to_id')}: entity={tgt.get('entity')!r} "
                  f"role={tgt.get('role')} value={tgt.get('value')!r} "
                  f"verb={tgt.get('signal_verb')!r}")
            print(f"        \"{t_sent[:160]}\"")
            if flags:
                print(f"      FLAGS: {', '.join(flags)}")

    print("\n" + "-" * 78)
    print(f"{label}: {total} edges, {flagged} with at least one flag")
    print(f"confidence: {dict(sorted(conf.items(), key=lambda x: (x[0] is None, x[0])))}")
    if limit and total > shown:
        print(f"(showed {shown}; rerun with --limit 0 for all)")


def duplicates(docs: dict) -> list[str]:
    print("=" * 78)
    print("DUPLICATE DOCUMENTS")
    print("=" * 78)
    by_hash: dict[str, list[str]] = defaultdict(list)
    for k, v in sorted(docs.items()):
        t = v.get("source_text") or ""
        if t:
            by_hash[hashlib.sha256(t.encode()).hexdigest()].append(k)

    dupes = {h: ids for h, ids in by_hash.items() if len(ids) > 1}
    if not dupes:
        print("none")
    else:
        extra = sum(len(i) - 1 for i in dupes.values())
        print(f"{len(dupes)} document(s) repeated; {extra} redundant file(s)")
        for h, ids in sorted(dupes.items(), key=lambda kv: -len(kv[1])):
            d = docs[ids[0]]
            print(f"  {len(ids)}x {ids}  chars={len(d.get('source_text',''))} "
                  f"facts={len(d.get('facts',[]) or [])}")
    keep = sorted(i[0] for i in by_hash.values())
    print(f"\nunique documents: {len(keep)} of {len(docs)}")
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ht", required=True, type=Path)
    ap.add_argument("--llm", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0,
                    help="max edges to print per side (0 = all)")
    ap.add_argument("--edges-only", action="store_true")
    args = ap.parse_args()

    ht, llm = load_dir(args.ht), load_dir(args.llm)
    keep = duplicates(ht)
    keep = [k for k in keep if k in llm]
    print(f"documents in both, deduplicated: {len(keep)}")

    if not args.edges_only:
        print("\n" + "=" * 78)
        print("FACT QUALITY  (a 'fact' is not the same unit on both sides)")
        print("=" * 78)
        a = profile_facts(ht, keep, "rule-based")
        b = profile_facts(llm, keep, "LLM")
        print("\n" + "-" * 78)
        print(f"raw fact counts     rule-based {a['n']}   LLM {b['n']}")
        print(f"resolved a date     rule-based {a['parsed']}   LLM {b['parsed']}")
        print(f"USABLE NODES        rule-based {a['usable']}   LLM {b['usable']}")
        print("\nReport the usable-node line, not the raw counts. Raw counts\n"
              "compare a node-per-temporal-expression against a\n"
              "node-per-salient-event and are not a like-for-like measure.")

    audit_edges(ht, keep, "rule-based", args.limit)
    audit_edges(llm, keep, "LLM", args.limit if args.limit else 6)

    ht_e = sum(len(ht[k].get("dependencies", []) or []) for k in keep)
    llm_e = sum(len(llm[k].get("dependencies", []) or []) for k in keep)
    ht_d = sum(1 for k in keep if ht[k].get("dependencies"))
    llm_d = sum(1 for k in keep if llm[k].get("dependencies"))
    print("\n" + "=" * 78)
    print(f"DEDUPLICATED TOTALS ({len(keep)} documents)")
    print("=" * 78)
    print(f"  edges              rule-based {ht_e}    LLM {llm_e}")
    print(f"  docs with >=1 edge rule-based {ht_d}/{len(keep)}  LLM {llm_d}/{len(keep)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

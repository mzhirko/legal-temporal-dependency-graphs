#!/usr/bin/env python3
"""Pre-replay gates, field-aware.

The prereg's "byte-identical gold upper bound" cannot be checked as a naive
byte-diff against the ARCHIVED files, because two drifts pre-date v3 (both
verified on the pristine pre-v3 repo, 2026-07-27):

1. results_gold_facts.json was archived before run_gold_facts.py gained the
   `[-1 day from: "..."]` provenance suffix in rule_desc (display field).
2. anchor_sweep.json's `generated_from` records the invocation path, and the
   three text-audit bookkeeping fields (text_hits, bare_year_flags,
   residual_refs) depend on the --texts path used at generation time.

Every SCORED field is identical in both. This gate therefore compares all
fields EXCEPT the documented whitelist, and prints any whitelisted drift it
sees so nothing is silently ignored. To restore a true byte-gate, re-archive
both references once from the pre-v3 commit and delete the whitelist here.

Usage (from code/src):
  python3 scripts/replay_gates.py gold NEW.json ../data/experiments/results_gold_facts.json
  python3 scripts/replay_gates.py sweep NEW.json ../data/experiments/counterfactual/anchor_sweep.json
"""
from __future__ import annotations

import json
import sys

GOLD_DISPLAY = {"rule_desc"}
SWEEP_META = {"generated_from"}
SWEEP_TEXT_AUDIT = {"text_hits", "bare_year_flags", "residual_refs"}


def _fail(msgs: list[str]) -> int:
    for m in msgs:
        print("DIFF", m)
    print("GATE: FAIL")
    return 1


def gate_gold(new_p: str, ref_p: str) -> int:
    a = json.load(open(new_p)); b = json.load(open(ref_p))
    hard, soft = [], []
    for k in set(a) | set(b):
        if k == "rows":
            continue
        if a.get(k) != b.get(k):
            hard.append(f"top-level {k}: {b.get(k)!r} -> {a.get(k)!r}")
    ra = {r["case"]: r for r in a["rows"]}
    rb = {r["case"]: r for r in b["rows"]}
    if ra.keys() != rb.keys():
        hard.append(f"row set: {sorted(set(ra) ^ set(rb))}")
    for c in ra.keys() & rb.keys():
        for k in set(ra[c]) | set(rb[c]):
            if ra[c].get(k) == rb[c].get(k):
                continue
            (soft if k in GOLD_DISPLAY else hard).append(
                f"{c}.{k}: {rb[c].get(k)!r} -> {ra[c].get(k)!r}")
    for m in soft:
        print("known-drift (display only):", m)
    if hard:
        return _fail(hard)
    print(f"GATE: PASS -- all scored fields identical "
          f"({len(soft)} known display drifts)")
    return 0


def gate_sweep(new_p: str, ref_p: str) -> int:
    a = json.load(open(new_p)); b = json.load(open(ref_p))
    hard, soft = [], []
    for k in set(a) | set(b):
        if k == "cases":
            continue
        if a.get(k) != b.get(k):
            (soft if k in SWEEP_META else hard).append(
                f"meta {k}: {b.get(k)!r} -> {a.get(k)!r}")
    for cid in set(a["cases"]) | set(b["cases"]):
        ca, cb = a["cases"].get(cid, {}), b["cases"].get(cid, {})
        for k in set(ca) | set(cb):
            if k == "items" or ca.get(k) == cb.get(k):
                continue
            hard.append(f"{cid}.{k}: {cb.get(k)!r} -> {ca.get(k)!r}")
        ia = {i["k"]: i for i in ca.get("items", [])}
        ib = {i["k"]: i for i in cb.get("items", [])}
        if ia.keys() != ib.keys():
            hard.append(f"{cid} item set: {sorted(set(ia) ^ set(ib))}")
        for kk in ia.keys() & ib.keys():
            for f in set(ia[kk]) | set(ib[kk]):
                if ia[kk].get(f) == ib[kk].get(f):
                    continue
                (soft if f in SWEEP_TEXT_AUDIT else hard).append(
                    f"{cid} k={kk} {f}")
    n_soft = len(soft)
    if soft[:3]:
        print("known-drift (text-audit bookkeeping), e.g.:", "; ".join(soft[:3]))
    if hard:
        return _fail(hard)
    print(f"GATE: PASS -- all label fields identical across all items "
          f"({n_soft} known text-audit drifts)")
    return 0


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] not in ("gold", "sweep"):
        print(__doc__)
        return 2
    return (gate_gold if sys.argv[1] == "gold" else gate_sweep)(
        sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    sys.exit(main())

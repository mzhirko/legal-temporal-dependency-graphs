#!/usr/bin/env python3
"""
Re-score an existing proof_head2head.json WITHOUT re-calling any model.

Why this exists
---------------
The original run scored deadline/cascade dates by exact STRING match against
the ISO gold. A model that computed the right date but wrote it as
"13-02-2023" (instead of "2023-02-13") was marked wrong. The benchmark is
meant to measure temporal *reasoning*, not format compliance, so this tool
re-applies the value-based `proof.solvers.score` to the answers already stored
in the JSON and reprints the bucket table. Items that errored (e.g. Ollama
500 cold-load, or a 404 for an un-pulled model) stay wrong -- they have no
answer to credit -- so the recovered numbers only ever reflect real outputs.

It also reports, per solver, a format-compliance line (how often the model
obeyed the requested ISO layout) and how many items never produced an answer,
so a low score from missing data is never mistaken for a reasoning failure.

Usage
-----
  python proof/rescore.py ../data/evaluation_results/proof_head2head.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

# Make this runnable as `python proof/rescore.py ...` from src/ without needing
# PYTHONPATH=. -- put the directory that contains the `proof` package on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proof.benchmark import Item
from proof.solvers import Answer, score


def _item_from_record(rec: dict) -> Item:
    return Item(
        item_id=rec["item_id"],
        task=rec["task"],
        difficulty=rec.get("difficulty", {}),
        documents=[tuple(d) for d in rec.get("documents", [])],
        question=rec.get("question", ""),
        gold=rec["gold"],
    )


def _answer_from_record(rec: dict) -> Answer:
    return Answer(
        item_id=rec["item_id"],
        task=rec["task"],
        solver=rec.get("solver", "?"),
        deadline=rec.get("deadline"),
        verdict=rec.get("verdict"),
        updates=rec.get("updates") or {},
        error=rec.get("error"),
    )


def _bucket_key(item: Item) -> str:
    d = item.difficulty
    if item.task == "deadline":
        return f"deadline / offset={d.get('offset_form')}"
    return f"cascade / hops={d.get('hops')} / {d.get('locality')}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Value-based re-score of a proof run")
    ap.add_argument("path", help="proof_head2head.json from a previous run")
    args = ap.parse_args()

    data = json.load(open(args.path))
    items = {r["item_id"]: _item_from_record(r) for r in data["benchmark"]}
    order = [r["item_id"] for r in data["benchmark"]]

    solvers = list(data["answers"].keys())
    # bucket -> solver -> [items_ok, n_items, fields_ok, n_fields, iso_ok, errored]
    table: dict = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0, 0, 0]))

    for solver in solvers:
        by_id = {a["item_id"]: a for a in data["answers"][solver]}
        for iid in order:
            item = items[iid]
            rec = by_id.get(iid)
            bk = _bucket_key(item)
            cell = table[bk][solver]
            cell[1] += 1                      # n_items
            nf = 1 if item.task == "deadline" else len(item.gold["updates"])
            cell[3] += nf                     # n_fields
            if rec is None or rec.get("error"):
                cell[5] += 1                  # errored / missing -> uncredited
                continue
            s = score(item, _answer_from_record(rec))
            cell[0] += int(s["fully_correct"])
            cell[2] += s["n_correct"]
            cell[4] += s.get("deadline_format_iso", 0) + s.get("n_format_iso", 0)

    bucket_order = [
        "deadline / offset=digit", "deadline / offset=natural",
        "deadline / offset=vague",
        "cascade / hops=2 / single", "cascade / hops=2 / cross",
        "cascade / hops=3 / single", "cascade / hops=3 / cross",
    ]
    bucket_order += [b for b in table if b not in bucket_order]

    print(f"\nRe-scored (value-based) from {args.path}")
    print("cell = items_fully_correct / n_items   (fields_correct / n_fields)\n")
    head = f"{'bucket':34}" + "".join(f"{s:>26}" for s in solvers)
    print(head)
    print("-" * len(head))
    for bk in bucket_order:
        if bk not in table:
            continue
        row = f"{bk:34}"
        for s in solvers:
            c = table[bk][s]
            row += f"{f'{c[0]}/{c[1]} ({c[2]}/{c[3]})':>26}"
        print(row)

    print("\nper-solver totals:")
    for s in solvers:
        items_ok = sum(table[b][s][0] for b in table)
        n_items = sum(table[b][s][1] for b in table)
        fields_ok = sum(table[b][s][2] for b in table)
        n_fields = sum(table[b][s][3] for b in table)
        iso_ok = sum(table[b][s][4] for b in table)
        errored = sum(table[b][s][5] for b in table)
        answered_fields = n_fields  # iso compliance is over fields that exist
        print(f"  {s:22} items {items_ok:3}/{n_items:<3}  "
              f"fields {fields_ok:3}/{n_fields:<3}  "
              f"ISO-format {iso_ok:3}/{answered_fields:<3}  "
              f"errored/blank items: {errored}")
    print("\nNote: errored/blank items are scored wrong (no answer to credit). "
          "A low score driven by a high errored count is a harness/runtime "
          "problem, not a reasoning result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Pipeline vs direct LLM on the COMMON answered subset of the CF items.

Coverage differs between the two conditions (the pipeline abstains, the
direct baseline answers everything), so acc_all and acc_answered compare
different item sets. The like-for-like number is accuracy of each approach
on exactly the items BOTH answered, plus the McNemar disagreement cells
(pipeline-right/direct-wrong vs pipeline-wrong/direct-right), which say
who wins where they disagree.

Usage (from code/src):
  python3 cf_common_subset.py \
      out/v3_replay/cf_gemma_final/rescored_v2.json \
      external_bench/out/cf_anchor_gemma/results.json
"""
from __future__ import annotations

import json
import sys


def rows_of(path: str) -> dict:
    d = json.load(open(path))
    rows = d["rows"] if isinstance(d, dict) and "rows" in d else d
    return {r["id"]: r for r in rows}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    pipe = rows_of(sys.argv[1])
    direct = rows_of(sys.argv[2])
    ids = sorted(set(pipe) & set(direct))
    if len(ids) != len(pipe) or len(ids) != len(direct):
        print(f"note: item sets differ (pipe {len(pipe)}, direct {len(direct)},"
              f" common {len(ids)}) -- comparing the intersection")

    common = [i for i in ids if pipe[i].get("pred") and direct[i].get("pred")]
    pw = sum(1 for i in common if pipe[i]["match"])
    dw = sum(1 for i in common if direct[i]["match"])
    both = sum(1 for i in common if pipe[i]["match"] and direct[i]["match"])
    p_only = sum(1 for i in common if pipe[i]["match"] and not direct[i]["match"])
    d_only = sum(1 for i in common if direct[i]["match"] and not pipe[i]["match"])
    neither = len(common) - both - p_only - d_only

    n = len(common)
    print(f"items answered by BOTH: {n} of {len(ids)}")
    print(f"  pipeline accuracy on common subset: {pw}/{n} = {pw/n:.4f}")
    print(f"  direct   accuracy on common subset: {dw}/{n} = {dw/n:.4f}")
    print(f"  agreement cells: both-right {both} | pipeline-only-right {p_only}"
          f" | direct-only-right {d_only} | both-wrong {neither}")

    # exact binomial McNemar on the disagreement cells
    b, c = p_only, d_only
    if b + c:
        from math import comb
        m = b + c
        k = min(b, c)
        p = sum(comb(m, j) for j in range(0, k + 1)) / 2 ** m * 2
        print(f"  McNemar exact (two-sided) on {m} disagreements: p = {min(p,1):.4g}")

    # direct accuracy on the pipeline's ABSTAINED items -- what the pipeline
    # gave up, and what answering anyway would have been worth
    abst = [i for i in ids if not pipe[i].get("pred") and direct[i].get("pred")]
    if abst:
        aw = sum(1 for i in abst if direct[i]["match"])
        print(f"pipeline-abstained items answered by direct: {len(abst)}"
              f" | direct accuracy there: {aw}/{len(abst)} = {aw/len(abst):.4f}")

    print("\nper-case (common subset): pipeline vs direct")
    cases = sorted({pipe[i]["case"] for i in common})
    for cse in cases:
        cs = [i for i in common if pipe[i]["case"] == cse]
        cp = sum(1 for i in cs if pipe[i]["match"])
        cd = sum(1 for i in cs if direct[i]["match"])
        print(f"  {cse:28s} n={len(cs):3d}  pipeline {cp/len(cs):.3f}  direct {cd/len(cs):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

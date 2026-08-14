#!/usr/bin/env python3
"""audit_findings.py -- machine audits over the baseline raw responses.

No hand-counted numbers in the thesis tables: every matrix cell for C1 and
C8 is produced by this script over data/experiments/baseline/raw/.

C1 (minus-one-day convention, field-level): given the MODEL'S OWN
    anchor_date, is its emitted primary_deadline == anchor + 3m - 1d?
    Scored per response; NAIVE(+0) = convention not applied.

C8 (reasoning->answer divergence, hard cases only): the LAST timeliness
    conclusion stated inside the model's own `arithmetic` string vs its
    emitted `verdict` field. HEURISTIC LIMITS (state in the chapter): only
    detects hard contradictions where the working's final stated conclusion
    differs from the field; deference cases whose working talks itself into
    the final answer (kj mini/nano) are NOT counted here -- those are
    classified separately from the quoted text.

Usage (repo root):  python src/baseline/audit_findings.py
"""
import glob
import json
import re
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from dateutil.relativedelta import relativedelta as rd

RAW = Path(__file__).resolve().parents[2] / "data" / "experiments" / "baseline" / "raw"


def _d(s):
    try:
        return date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def load(f):
    m = re.search(r"\{.*\}", Path(f).read_text(encoding="utf-8"), re.S)
    return json.loads(m.group(0)) if m else None


def main():
    c1 = defaultdict(lambda: [0, 0])   # model -> [convention_applied, applicable]
    c8 = defaultdict(lambda: [0, 0])   # model -> [hard_divergences, responses]
    rows = []
    for f in sorted(glob.glob(str(RAW / "*.txt"))):
        name = Path(f).name
        case, model = name.split(".")[0], ".".join(name.split(".")[1:-1])
        j = load(f)
        if not j:
            rows.append(f"{name}: NO JSON"); continue

        a, p = _d(j.get("anchor_date")), _d(j.get("primary_deadline"))
        if a and p:
            c1[model][1] += 1
            ok = p == a + rd(months=3) - timedelta(days=1)
            if ok:
                c1[model][0] += 1
            rows.append(f"C1 {model:14s} {case:24s} anchor={a} primary={p} "
                        f"{'OK(-1d)' if ok else 'NAIVE' if p == a + rd(months=3) else 'OTHER'}")

        ar, v = j.get("arithmetic", "").lower(), j.get("verdict", "")
        concl = None
        for m in re.finditer(r"(out of time|in time|out_of_time|in_time)", ar):
            concl = m.group(1).replace(" ", "_")
        c8[model][1] += 1
        if concl and concl != v:
            c8[model][0] += 1
            rows.append(f"C8 {model:14s} {case:24s} field={v} working-concludes={concl}")

    print("\n".join(rows))
    print("\n== C1: minus-one-day convention applied (field-level) ==")
    for m in sorted(c1):
        print(f"  {m:14s} {c1[m][0]}/{c1[m][1]}")
    print("== C8: hard working/field divergences ==")
    for m in sorted(c8):
        print(f"  {m:14s} {c8[m][0]}/{c8[m][1]}")


if __name__ == "__main__":
    main()
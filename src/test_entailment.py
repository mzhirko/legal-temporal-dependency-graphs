#!/usr/bin/env python3
"""
Unit test for the general entailment engine (Use Case B).

Model-free and deterministic -- no Gemma, no Catala. Builds rule + instance TDGs
in code and asserts:
  1. The engine reproduces the deadline real tribunals computed (calendar
     arithmetic, "3 months beginning with" = +3 months - 1 day).
  2. It discriminates TIMELY vs LATE at the decision boundary (this is the
     in-time proof -- it does NOT require a curated in-time judgment).
  3. The ACAS s.207B early-conciliation pause changes the verdict when applied.
  4. The rule is DISCOVERED from the statute TDG, not hardcoded.

Run:  python test_entailment.py
"""
import sys
from datetime import date

from tdg_pipeline.tdg import (
    TemporalDependencyGraph, TemporalFact, TemporalDependency, TimexSpan,
)
from tdg_pipeline.entailment import check_entailment, find_rules


def _fact(fid, entity, role, d=None, sent=""):
    return TemporalFact(
        id=fid, entity=entity, role=role,
        timex=TimexSpan(text=sent, timex_type="DATE" if d else "DURATION",
                        value=(d.isoformat() if d else None),
                        start_char=0, end_char=0, date_parsed=d, duration_days=None),
        sentence=sent)


def _statute_tdg():
    """ERA s.111: complaint within 3 months of EDT. The RULE is an additive
    dependency in the statute text, not a Python constant."""
    edt = _fact("s1", "effective date of termination", "START",
                sent="within three months beginning with the effective date of termination")
    dl = _fact("s2", "time limit for presenting complaint", "END",
               sent="complaint presented within three months of termination")
    dep = TemporalDependency(from_id="s1", to_id="s2", constraint_type="additive",
                             constraint_expr="+3 months", delta_days=90)
    return TemporalDependencyGraph(document_id="era_1996_s111", document_type="legal",
                                   source_text="", facts=[edt, dl], dependencies=[dep])


def _case_tdg(doc_id, edt, et1):
    return TemporalDependencyGraph(
        document_id=doc_id, document_type="legal", source_text="",
        facts=[
            _fact("c1", "effective date of termination", "START", d=edt,
                  sent="the effective date of termination was"),
            _fact("c2", "claim presented to the tribunal", "CONTAINS", d=et1,
                  sent="the claimant presented his claim to the tribunal on"),
        ], dependencies=[])


def main() -> int:
    stat = _statute_tdg()
    failures = []

    # 0. rule discovery: must find exactly the 3-month rule, no hardcode
    rules = find_rules(stat)
    if not rules or rules[0].offset.months != 3:
        failures.append("rule discovery did not yield a 3-month rule from the statute TDG")

    # 1. real tribunal-computed deadlines (verdict + exact deadline date)
    real = [
        ("bano",     date(2021, 8, 28),  date(2022, 1, 17), "LATE", date(2021, 11, 27)),
        ("kpatel",   date(2021, 5, 28),  date(2021, 9, 11), "LATE", date(2021, 8, 27)),
        ("hruskova", date(2023, 12, 11), date(2024, 6, 18), "LATE", date(2024, 3, 10)),
        ("haouari",  date(2020, 8, 2),   date(2021, 9, 15), "LATE", date(2020, 11, 1)),
    ]
    for cid, edt, et1, exp, stated_dl in real:
        r = check_entailment(stat, _case_tdg(cid, edt, et1))[0]
        if r.verdict != exp:
            failures.append(f"{cid}: verdict {r.verdict} != {exp}")
        if r.deadline_computed != stated_dl.isoformat():
            failures.append(f"{cid}: deadline {r.deadline_computed} != tribunal {stated_dl}")

    # 2. boundary: TIMELY vs LATE either side of the deadline (Bano: 2021-11-27)
    for label, et1, exp in [("early", date(2021, 11, 26), "TIMELY"),
                            ("on",    date(2021, 11, 27), "TIMELY"),
                            ("late",  date(2021, 11, 28), "LATE")]:
        r = check_entailment(stat, _case_tdg("boundary", date(2021, 8, 28), et1))[0]
        if r.verdict != exp:
            failures.append(f"boundary {label}: {r.verdict} != {exp}")

    # 3. ACAS pause flips a late filing to timely
    nop = check_entailment(stat, _case_tdg("nopause", date(2022, 1, 1), date(2022, 4, 20)))[0]
    wp = check_entailment(stat, _case_tdg("withpause", date(2022, 1, 1), date(2022, 4, 20)),
                          acas_day_a=date(2022, 2, 1), acas_day_b=date(2022, 3, 15))[0]
    if nop.verdict != "LATE":
        failures.append(f"acas baseline should be LATE, got {nop.verdict}")
    if wp.verdict != "TIMELY" or not wp.acas_applied:
        failures.append(f"acas-extended should be TIMELY+applied, got {wp.verdict}/{wp.acas_applied}")

    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print("OK: entailment engine reproduces real deadlines, discriminates "
          "TIMELY/LATE at the boundary, and applies the ACAS pause. "
          "Rule discovered from statute TDG (not hardcoded).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
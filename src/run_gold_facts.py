#!/usr/bin/env python3
"""
run_gold_facts.py -- Use Case B on VERIFIED GOLD FACTS (extraction bypassed).

Purpose: the extraction-independent upper bound. Takes human-verified case
facts (anchor date, presentation date, ACAS Day A/B) from a gold JSON and
runs ONLY the rule-discovery + calendar/conciliation layers of the engine:

    statute TDG --find_rules()--> TemporalRule (offset, minus-one-day)
    gold anchor --offset.apply()--> primary deadline
                --_apply_conciliation(Day A, Day B)--> effective deadline
    verdict = TIMELY iff presented <= effective deadline

Each case is scored against (a) the tribunal's verdict and (b) the
tribunal-stated deadline where the judgment recites one.

Usage:
  python run_gold_facts.py \
      --gold ../data/ground_truth/ground_truth_gold.json \
      --statute-tdg era_1996_s111=../data/results22-05-26/era_s111_tdg.json \
      [--statute-tdg eqa_2010_s123=../data/results_uk/eqa_2010_s123.json] \
      [--out results_gold_facts.json]

Statutes without a TDG fall back to the registry spec embedded in the gold
JSON ("minus_one_day", months=3), reported as rule_source="registry_fallback"
so discovered-rule and fallback rows are never conflated in the thesis.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tdg_pipeline.entailment import (
    CalendarOffset,
    find_rules,
    _apply_conciliation,
)
from tdg_pipeline.io import load_tdg


def _d(s: Optional[str]) -> Optional[date]:
    return date.fromisoformat(s) if s else None


@dataclass
class Row:
    case: str
    statute: str
    rule_source: str          # "discovered" | "registry_fallback" | "no_rule"
    rule_desc: str
    anchor: Optional[str]
    primary_deadline: Optional[str]
    effective_deadline: Optional[str]
    acas_applied: bool
    presented: Optional[str]
    days_over: Optional[int]
    engine_verdict: str       # TIMELY | LATE | SKIPPED
    gold_verdict: str
    verdict_match: Optional[bool]
    deadline_stated: Optional[str]
    deadline_match: Optional[bool]   # engine deadline vs judge-stated deadline
    boundary_stated: Optional[str] = None   # judge-stated earliest in-time act date
    boundary_match: Optional[bool] = None   # forward check: boundary is exactly first in-time day
    note: str = ""


def _rule_for(statute: str, tdg_paths: dict, minus_one_day: bool):
    """(offset, source, description) for a statute -- discovered or fallback.

    `minus_one_day` is now only the fallback for a clause whose connector is
    unreadable. find_rules reads the -1 day off the statute's own wording
    ("three months BEGINNING WITH ..."), so the whole rule -- N, unit, and the
    inclusivity that makes it 3m-1d rather than 3m -- comes from the text. The
    returned source says "discovered" only when that actually happened.
    """
    path = tdg_paths.get(statute)
    if path:
        tdg = load_tdg(path)
        rules = find_rules(tdg, minus_one_day=None, embedder=None)
        if rules:
            r = rules[0]
            src = ("discovered" if r.offset.inclusivity_source == "discovered"
                   else "discovered_offset_assumed_inclusivity")
            desc = r.description
            if r.offset.inclusivity_evidence:
                desc += f"  [-1 day from: \"{r.offset.inclusivity_evidence}\"]"
            return r.offset, src, desc
        return None, "no_rule", f"no rule discovered in {Path(path).name}"
    # registry fallback: 3 months, statute-level minus_one_day flag
    off = CalendarOffset(months=3, minus_one_day=minus_one_day)
    return off, "registry_fallback", f"presentation = anchor + 3m{' - 1 day' if minus_one_day else ''}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="gold facts JSON")
    ap.add_argument("--statute-tdg", action="append", default=[],
                    metavar="STATUTE=PATH",
                    help="map a statute key to its TDG JSON (repeatable)")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text())
    default_minus_one = bool(gold.get("minus_one_day", True))
    tdg_paths = dict(kv.split("=", 1) for kv in args.statute_tdg)

    rule_cache: dict[str, tuple] = {}
    rows: list[Row] = []

    for case_id, c in gold["cases"].items():
        statute = c.get("statute", gold.get("statute", "unknown"))
        anchor, presented = _d(c.get("anchor_date")), _d(c.get("presented_date"))
        gv = c.get("verdict", "?")
        stated = c.get("deadline_stated")

        if anchor is None or presented is None:
            rows.append(Row(case_id, statute, "-", "-", c.get("anchor_date"),
                            None, None, False, c.get("presented_date"), None,
                            "SKIPPED", gv, None, stated, None,
                            note="missing anchor/presented date (verify-only row)"))
            continue

        if statute not in rule_cache:
            rule_cache[statute] = _rule_for(statute, tdg_paths, default_minus_one)
        offset, source, desc = rule_cache[statute]
        if offset is None:
            rows.append(Row(case_id, statute, source, desc, anchor.isoformat(),
                            None, None, False, presented.isoformat(), None,
                            "SKIPPED", gv, None, stated, None,
                            note="statute rule unavailable"))
            continue

        primary = offset.apply(anchor)
        effective, acas_applied = _apply_conciliation(
            primary, _d(c.get("acas_day_a")), _d(c.get("acas_day_b")))
        days_over = (presented - effective).days
        engine_verdict = "TIMELY" if days_over <= 0 else "LATE"

        verdict_match = (
            (engine_verdict == "LATE" and gv == "out_of_time")
            or (engine_verdict == "TIMELY" and gv == "in_time")
        )
        # deadline check: judge-stated deadlines may include EC extension the
        # gold row lacks; compare against the EFFECTIVE deadline we computed.
        deadline_match = (effective.isoformat() == stated) if stated else None

        # boundary check (forward direction only -- calendar month arithmetic
        # does not commute with day offsets, so never compute this backwards):
        # the judge-stated boundary must be exactly the FIRST act date whose
        # effective deadline (rule + EC pause) reaches the presentation date.
        boundary_stated = c.get("boundary_stated")
        boundary_match = None
        if boundary_stated:
            from datetime import timedelta
            b = _d(boundary_stated)
            def _eff(x):
                e, _ = _apply_conciliation(offset.apply(x),
                                           _d(c.get("acas_day_a")),
                                           _d(c.get("acas_day_b")))
                return e
            boundary_match = (_eff(b) >= presented) and (
                _eff(b - timedelta(days=1)) < presented)

        rows.append(Row(case_id, statute, source, desc, anchor.isoformat(),
                        primary.isoformat(), effective.isoformat(), acas_applied,
                        presented.isoformat(), days_over, engine_verdict, gv,
                        verdict_match, stated, deadline_match,
                        boundary_stated=boundary_stated,
                        boundary_match=boundary_match,
                        note=c.get("note", "")))

    # -- report ------------------------------------------------------------
    scored = [r for r in rows if r.verdict_match is not None]
    v_ok = sum(r.verdict_match for r in scored)
    d_rows = [r for r in rows if r.deadline_match is not None]
    d_ok = sum(r.deadline_match for r in d_rows)
    b_rows = [r for r in rows if r.boundary_match is not None]
    b_ok = sum(r.boundary_match for r in b_rows)

    w = max(len(r.case) for r in rows) + 1
    print(f"{'case':<{w}} {'statute':<14} {'rule':<10} {'anchor':<11} "
          f"{'deadline':<11} {'ACAS':<5} {'ET1':<11} {'over':>5}  "
          f"{'engine':<7} {'gold':<12} ok")
    for r in rows:
        print(f"{r.case:<{w}} {r.statute:<14} {r.rule_source[:9]:<10} "
              f"{r.anchor or '-':<11} {r.effective_deadline or '-':<11} "
              f"{'yes' if r.acas_applied else 'no':<5} {r.presented or '-':<11} "
              f"{('' if r.days_over is None else f'{r.days_over:+}'):>5}  "
              f"{r.engine_verdict:<7} {r.gold_verdict:<12} "
              f"{'-' if r.verdict_match is None else ('OK' if r.verdict_match else 'XX')}"
              + (f"   [stated {r.deadline_stated}: "
                 f"{'MATCH' if r.deadline_match else 'DIFF'}]" if r.deadline_match is not None else "")
              + (f"   [boundary {r.boundary_stated}: "
                 f"{'MATCH' if r.boundary_match else 'DIFF'}]" if r.boundary_match is not None else ""))
    print(f"\nverdicts: {v_ok}/{len(scored)} match tribunal | "
          f"stated-deadline reproduction: {d_ok}/{len(d_rows)} | "
          f"stated-boundary reproduction: {b_ok}/{len(b_rows)} | "
          f"skipped (verify-only): {len(rows) - len(scored)}")
    for stat, (off, src, desc) in rule_cache.items():
        print(f"rule[{stat}] ({src}): {desc}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"rows": [r.__dict__ for r in rows],
             "verdict_accuracy": [v_ok, len(scored)],
             "deadline_reproduction": [d_ok, len(d_rows)],
             "boundary_reproduction": [b_ok, len(b_rows)]}, indent=1))
        print(f"written: {args.out}")
    return 0 if v_ok == len(scored) else 1


if __name__ == "__main__":
    raise SystemExit(main())

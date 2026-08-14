#!/usr/bin/env python3
"""
Use Cases B & C: Temporal Entailment and Contradiction Detection.

USE CASE B -- TEMPORAL ENTAILMENT
Given a statute defining temporal rules and a judgment with case-specific
facts, check whether the case satisfies the statute's constraints.

Example: ERA 1996 s.111 says "file within 3 months of EDT".
         Ahmed's EDT is 16 March 2020, ET1 filed 24 August 2020.
         -> 3 months from 16 March = 15 June 2020. ET1 is LATE.

The checker:
  1. Extracts the rule from the statute TDG (duration + anchor role)
  2. Extracts the case facts from the judgment TDG (dates)
  3. Computes the deadline from rule + anchor date
  4. Compares the action date against the deadline
  5. Returns: TIMELY, LATE, or INDETERMINATE

USE CASE C -- CONTRADICTION DETECTION
Given N documents, find pairs where the same temporal rule has
incompatible values. Uses cross_doc.py's contradiction detection
with evaluation against expected contradictions.

Usage:
    python evaluate_use_cases_bc.py \
        --ground-truth ../data/ground_truth/ground_truth_uk.json \
        --llm-dir ../data/llm_extracted/ \
        --output ../data/evaluation_results/use_cases_bc.json

File layout:
    code/
    ├-- data/
    │   ├-- ground_truth/ground_truth_uk.json
    │   ├-- llm_extracted/  (Gemma TDGs)
    │   └-- evaluation_results/use_cases_bc.json
    └-- src/
        ├-- tdg_pipeline/cross_doc.py
        └-- evaluate_use_cases_bc.py   ← this script
"""

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import Optional, Literal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tdg_pipeline.tdg import (
    TemporalDependencyGraph, TemporalFact, TemporalDependency, TimexSpan,
)
from tdg_pipeline.cross_doc import (
    CrossDocLinker, _normalise_entity, _entity_similarity,
)
from tdg_pipeline.io import load_json, build_tdg, DEFAULT_SIMILARITY_THRESHOLD, _GENERIC_DOC_IDS


# ═══════════════════════════════════════════════════════════════════════════
# USE CASE B: TEMPORAL ENTAILMENT
# ═══════════════════════════════════════════════════════════════════════════

Verdict = Literal["TIMELY", "LATE", "INDETERMINATE"]


@dataclass
class EntailmentResult:
    """Result of checking whether a case satisfies a statutory time limit."""
    statute_doc: str
    case_doc: str
    rule_description: str       # e.g. "file within 3 months of EDT"
    anchor_date: Optional[str]  # e.g. "2020-03-16" (EDT)
    deadline_computed: Optional[str]  # e.g. "2020-06-15"
    action_date: Optional[str]  # e.g. "2020-08-24" (ET1 filed)
    days_over: Optional[int]    # positive = late, negative = early
    verdict: Verdict
    tribunal_finding: Optional[str]  # ground truth from judgment
    matches_tribunal: Optional[bool]
    explanation: str

    def to_dict(self) -> dict:
        return asdict(self)


# -- Statutory rule definitions --------------------------------------------
# These encode the temporal rules as Python. Each rule can also be a
# Catala scope -- see the .catala_en comment for the formal equivalent.

"""
Catala equivalent (for formal verification):

  declaration scope UnfairDismissalTimeLimit:
    input effective_date_of_termination content date
    output primary_deadline content date
    output extended_deadline content date
    input conciliation_pause_days content integer

  scope UnfairDismissalTimeLimit:
    definition primary_deadline equals
      effective_date_of_termination + 3 month - 1 day
    definition extended_deadline equals
      primary_deadline + conciliation_pause_days day
"""


@dataclass
class StatutoryRule:
    """A temporal rule extracted from statute."""
    rule_id: str
    description: str
    duration_days: int          # the time limit (e.g. 90 for 3 months)
    anchor_role: str            # what the deadline is measured from (e.g. "END" = EDT)
    anchor_entity_pattern: str  # entity name pattern to match (e.g. "termination")
    action_role: str            # role of the action being checked (e.g. "CONTAINS")
    action_entity_pattern: str  # entity name pattern (e.g. "et1")
    deadline_entity_pattern: str  # entity name pattern for the computed deadline
    minus_one_day: bool = True  # "3 months beginning with" = add 3 months, subtract 1 day


# Known statutory rules
ERA_S111_RULE = StatutoryRule(
    rule_id="era_s111_3month",
    description="Unfair dismissal complaint must be filed within 3 months of effective date of termination (s.111(2)(a) ERA 1996)",
    duration_days=90,  # approximation; actual rule is "3 calendar months minus 1 day"
    anchor_role="END",
    anchor_entity_pattern="termination|employment",
    action_role="START|CONTAINS",
    action_entity_pattern="et1",
    deadline_entity_pattern="deadline|time limit|claim form",
    minus_one_day=True,
)


def _find_fact_by_pattern(
    facts: list[dict],
    role_pattern: str,
    entity_pattern: str,
) -> Optional[dict]:
    """Find the first fact matching a role and entity name pattern."""
    import re
    roles = set(role_pattern.split("|"))
    entity_re = re.compile(entity_pattern, re.IGNORECASE)
    for f in facts:
        if f["role"] in roles:
            norm = _normalise_entity(f["entity"])
            if entity_re.search(norm) or entity_re.search(f["entity"].lower()):
                return f
    return None


def check_entailment_era(
    rule: StatutoryRule,
    case_facts: list[dict],
    statute_doc_id: str,
    case_doc_id: str,
    tribunal_finding: Optional[str] = None,
) -> EntailmentResult:
    """
    Check whether a case's temporal facts satisfy a statutory rule.

    Pure Python computation -- no Catala needed. The Catala scope can
    verify this independently.
    """
    # Find the anchor date (e.g. EDT)
    anchor = _find_fact_by_pattern(case_facts, rule.anchor_role, rule.anchor_entity_pattern)
    anchor_date_str = anchor.get("date_parsed") if anchor else None

    # Find the action date (e.g. ET1 presented)
    action = _find_fact_by_pattern(case_facts, rule.action_role, rule.action_entity_pattern)
    action_date_str = action.get("date_parsed") if action else None

    # Find the tribunal's computed deadline (if present in the TDG)
    deadline_fact = _find_fact_by_pattern(case_facts, "END", rule.deadline_entity_pattern)
    tribunal_deadline_str = deadline_fact.get("date_parsed") if deadline_fact else None

    if not anchor_date_str or not action_date_str:
        missing = []
        if not anchor_date_str:
            missing.append(f"anchor ({rule.anchor_entity_pattern})")
        if not action_date_str:
            missing.append(f"action ({rule.action_entity_pattern})")
        return EntailmentResult(
            statute_doc=statute_doc_id, case_doc=case_doc_id,
            rule_description=rule.description,
            anchor_date=anchor_date_str, deadline_computed=None,
            action_date=action_date_str, days_over=None,
            verdict="INDETERMINATE",
            tribunal_finding=tribunal_finding, matches_tribunal=None,
            explanation=f"Missing facts: {', '.join(missing)}",
        )

    anchor_date = date.fromisoformat(anchor_date_str)
    action_date = date.fromisoformat(action_date_str)

    # Compute deadline: anchor + 3 calendar months - 1 day
    # (approximation: 3 months ≈ 90 days; the actual UK rule uses calendar months)
    if rule.minus_one_day:
        # Proper calendar month calculation
        month = anchor_date.month + 3
        year = anchor_date.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        try:
            deadline = date(year, month, anchor_date.day) - timedelta(days=1)
        except ValueError:
            # Handle months with fewer days (e.g. 31 Jan + 3 months)
            import calendar
            last_day = calendar.monthrange(year, month)[1]
            deadline = date(year, month, min(anchor_date.day, last_day)) - timedelta(days=1)
    else:
        deadline = anchor_date + timedelta(days=rule.duration_days)

    days_over = (action_date - deadline).days
    verdict: Verdict = "TIMELY" if days_over <= 0 else "LATE"

    matches = None
    if tribunal_finding:
        tribunal_late = "late" in tribunal_finding.lower() or "out of time" in tribunal_finding.lower()
        tribunal_timely = "timely" in tribunal_finding.lower() or "in time" in tribunal_finding.lower()
        if tribunal_late:
            matches = verdict == "LATE"
        elif tribunal_timely:
            matches = verdict == "TIMELY"

    return EntailmentResult(
        statute_doc=statute_doc_id, case_doc=case_doc_id,
        rule_description=rule.description,
        anchor_date=anchor_date_str,
        deadline_computed=deadline.isoformat(),
        action_date=action_date_str,
        days_over=days_over,
        verdict=verdict,
        tribunal_finding=tribunal_finding,
        matches_tribunal=matches,
        explanation=(
            f"EDT={anchor_date_str} + 3 months - 1 day = {deadline.isoformat()}. "
            f"ET1 filed {action_date_str} = {abs(days_over)} days "
            f"{'after' if days_over > 0 else 'before'} deadline. "
            f"Verdict: {verdict}."
            + (f" Tribunal deadline: {tribunal_deadline_str}" if tribunal_deadline_str else "")
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# USE CASE C: CONTRADICTION DETECTION EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ContradictionTestCase:
    """A pair of documents with known contradictions or known agreement."""
    doc_a: str
    doc_b: str
    expected_contradictions: int  # 0 = they agree
    description: str


# Test cases for contradiction detection
CONTRADICTION_TESTS = [
    ContradictionTestCase(
        doc_a="ahmed_v_newcastle", doc_b="zegay_v_boylan",
        expected_contradictions=0,
        description="Two judgments applying the same s.111 rule -- should agree on P3M duration",
    ),
    ContradictionTestCase(
        doc_a="era_1996_s111", doc_b="ahmed_v_newcastle",
        expected_contradictions=0,
        description="Statute and judgment citing it -- should agree on P3M",
    ),
    ContradictionTestCase(
        doc_a="era_1996_s111", doc_b="zegay_v_boylan",
        expected_contradictions=0,
        description="Statute and judgment citing it -- should agree on P3M",
    ),
]


def evaluate_contradictions(llm_tdgs: dict[str, dict]) -> dict:
    """Run contradiction detection and evaluate against expected results."""
    linker = CrossDocLinker(similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD)

    for doc_id, data in llm_tdgs.items():
        data["document_id"] = data.get("document_id", doc_id)
        linker.add_tdg(build_tdg(data))

    all_links = linker.find_all_links()
    contradictions = [l for l in all_links if l.link_type == "contradiction"]
    parallel = [l for l in all_links if l.link_type == "parallel_application"]

    results = {
        "total_contradictions_found": len(contradictions),
        "total_parallel_applications": len(parallel),
        "test_cases": [],
    }

    for tc in CONTRADICTION_TESTS:
        # Count contradictions between this specific pair
        pair_contradictions = [
            c for c in contradictions
            if (c.from_doc == tc.doc_a and c.to_doc == tc.doc_b) or
               (c.from_doc == tc.doc_b and c.to_doc == tc.doc_a)
        ]
        pair_parallel = [
            p for p in parallel
            if (p.from_doc == tc.doc_a and p.to_doc == tc.doc_b) or
               (p.from_doc == tc.doc_b and p.to_doc == tc.doc_a)
        ]

        correct = len(pair_contradictions) == tc.expected_contradictions

        results["test_cases"].append({
            "doc_a": tc.doc_a,
            "doc_b": tc.doc_b,
            "description": tc.description,
            "expected_contradictions": tc.expected_contradictions,
            "found_contradictions": len(pair_contradictions),
            "found_parallel_applications": len(pair_parallel),
            "correct": correct,
            "details": [c.to_dict() for c in pair_contradictions],
        })

    correct_count = sum(1 for tc in results["test_cases"] if tc["correct"])
    results["accuracy"] = correct_count / len(results["test_cases"]) if results["test_cases"] else 0

    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

# Ground truth tribunal findings
TRIBUNAL_FINDINGS = {
    "ahmed_v_newcastle": "out of time -- claim struck out",
    "zegay_v_boylan": "out of time",
}


def main():
    parser = argparse.ArgumentParser(description="Use Cases B & C evaluation")
    parser.add_argument("--ground-truth", required=True, help="Path to ground_truth_uk.json")
    parser.add_argument("--llm-dir", required=True, help="Directory with LLM-extracted TDG JSONs")
    parser.add_argument("--output", default=None, help="Path to save results JSON")
    args = parser.parse_args()

    # Load ground truth and LLM TDGs
    gt = load_json(args.ground_truth)

    _GENERIC = _GENERIC_DOC_IDS
    llm_tdgs = {}
    for fpath in sorted(glob.glob(os.path.join(args.llm_dir, "*.json"))):
        try:
            data = load_json(fpath)
            doc_id = data.get("document_id", "")
            if doc_id in _GENERIC:
                doc_id = os.path.splitext(os.path.basename(fpath))[0]
            data["document_id"] = doc_id
            llm_tdgs[doc_id] = data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARNING: skipping {fpath}: {e}")

    all_results = {"use_case_b": [], "use_case_c": {}}

    # -- USE CASE B: Temporal Entailment -------------------------------
    print(f"\n{'='*70}")
    print("USE CASE B: TEMPORAL ENTAILMENT")
    print(f"{'='*70}")
    print(f"Rule: {ERA_S111_RULE.description}\n")

    for case_id in ["ahmed_v_newcastle", "zegay_v_boylan"]:
        if case_id not in llm_tdgs:
            continue

        # Run on LLM-extracted facts
        llm_result = check_entailment_era(
            rule=ERA_S111_RULE,
            case_facts=llm_tdgs[case_id]["facts"],
            statute_doc_id="era_1996_s111",
            case_doc_id=case_id,
            tribunal_finding=TRIBUNAL_FINDINGS.get(case_id),
        )

        # Run on ground truth facts for comparison
        gt_result = check_entailment_era(
            rule=ERA_S111_RULE,
            case_facts=gt[case_id]["facts"],
            statute_doc_id="era_1996_s111",
            case_doc_id=case_id,
            tribunal_finding=TRIBUNAL_FINDINGS.get(case_id),
        )

        all_results["use_case_b"].append({
            "case": case_id,
            "llm": llm_result.to_dict(),
            "ground_truth": gt_result.to_dict(),
        })

        print(f"  {case_id}")
        print(f"  {'-'*50}")
        print(f"    Ground truth:")
        print(f"      EDT:          {gt_result.anchor_date}")
        print(f"      Deadline:     {gt_result.deadline_computed}")
        print(f"      ET1 filed:    {gt_result.action_date}")
        print(f"      Days over:    {gt_result.days_over}")
        print(f"      Verdict:      {gt_result.verdict}")
        print(f"      Tribunal:     {gt_result.tribunal_finding}")
        print(f"      Matches:      {gt_result.matches_tribunal}")
        print()
        print(f"    LLM-extracted:")
        print(f"      EDT:          {llm_result.anchor_date}")
        print(f"      Deadline:     {llm_result.deadline_computed}")
        print(f"      ET1 filed:    {llm_result.action_date}")
        print(f"      Days over:    {llm_result.days_over}")
        print(f"      Verdict:      {llm_result.verdict}")
        print(f"      Matches:      {llm_result.matches_tribunal}")

        if llm_result.verdict != gt_result.verdict:
            print(f"      (!) DISAGREEMENT: LLM says {llm_result.verdict}, GT says {gt_result.verdict}")
        elif llm_result.verdict == gt_result.verdict:
            print(f"      ✓ LLM and GT agree: {llm_result.verdict}")

        if llm_result.days_over and gt_result.days_over:
            if llm_result.days_over != gt_result.days_over:
                print(f"      Note: days_over differs (LLM={llm_result.days_over}, GT={gt_result.days_over})")
        print()

    # Summary
    b_verdicts = [r["llm"]["verdict"] for r in all_results["use_case_b"]]
    b_matches = [r["llm"]["matches_tribunal"] for r in all_results["use_case_b"]]
    print(f"  SUMMARY:")
    print(f"    Cases checked:     {len(all_results['use_case_b'])}")
    print(f"    Verdicts:          {b_verdicts}")
    print(f"    Match tribunal:    {b_matches}")
    print(f"    Accuracy:          {sum(1 for m in b_matches if m) / len(b_matches):.0%}")

    # -- USE CASE C: Contradiction Detection ---------------------------
    print(f"\n{'='*70}")
    print("USE CASE C: CONTRADICTION DETECTION")
    print(f"{'='*70}\n")

    c_results = evaluate_contradictions(llm_tdgs)
    all_results["use_case_c"] = c_results

    print(f"  Total contradictions found: {c_results['total_contradictions_found']}")
    print(f"  Total parallel applications: {c_results['total_parallel_applications']}")
    print()

    for tc in c_results["test_cases"]:
        status = "✓" if tc["correct"] else "x"
        print(f"  {status} {tc['doc_a']} ↔ {tc['doc_b']}")
        print(f"    {tc['description']}")
        print(f"    Expected contradictions: {tc['expected_contradictions']}  "
              f"Found: {tc['found_contradictions']}  "
              f"Parallel: {tc['found_parallel_applications']}")
        if tc["details"]:
            for d in tc["details"]:
                print(f"    -> {d['explanation'][:100]}")
        print()

    print(f"  SUMMARY:")
    print(f"    Test cases: {len(c_results['test_cases'])}")
    print(f"    Accuracy:   {c_results['accuracy']:.0%}")

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
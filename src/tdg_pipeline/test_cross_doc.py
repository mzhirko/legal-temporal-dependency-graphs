#!/usr/bin/env python3
"""
Cross-doc linking integration test.

Uses the real en_contracts_seed0.json output from Gemma 4 plus two
synthetic TDGs (a statute and a court judgment) to demonstrate:
  - Coreference detection
  - Contradiction detection
  - Structural analogy
  - Staleness propagation

Run:
    python test_cross_doc.py
"""

import json
import sys
import os
from datetime import date

# Allow importing from the work directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tdg_pipeline.tdg import (
    TemporalDependencyGraph,
    TemporalFact,
    TemporalDependency,
    TimexSpan,
)
from tdg_pipeline.cross_doc import CrossDocLinker
from tdg_pipeline.io import load_tdg


# --- Helper to build TDGs quickly ----------------------------------------

def make_fact(id, entity, role, value=None, date_parsed=None,
              duration_days=None, sentence=""):
    timex_type = "DURATION" if duration_days is not None else "DATE"
    return TemporalFact(
        id=id, entity=entity, role=role,
        timex=TimexSpan(
            text=value or "", timex_type=timex_type,
            value=value, start_char=0, end_char=0,
            date_parsed=date_parsed, duration_days=duration_days,
        ),
        sentence=sentence, confidence=0.9,
    )


def make_dep(from_id, to_id, ctype="ordering", expr="", delta=None, verified=False):
    return TemporalDependency(
        from_id=from_id, to_id=to_id,
        constraint_type=ctype, constraint_expr=expr,
        delta_days=delta, verified=verified, confidence=0.85,
    )


# --- Build synthetic statute TDG -----------------------------------------

def build_statute_tdg() -> TemporalDependencyGraph:
    """
    Housing Lease Protection Act §47 -- synthetic statute.

    Facts:
      - Max lease term: P3Y (DURATION)
      - Renewal notice minimum: P60D
      - Renewal notice maximum: P120D
      - Act entry into force: 2019-01-01
      - Denunciation notice: P6M (same as contract -- for contradiction test)

    This statute has a denunciation notice of P6M, same duration as
    the contract, showing coreference. We also add a variant with P90D
    notice to test contradiction detection.
    """
    facts = [
        make_fact("s1", "lease", "DURATION", "P3Y", duration_days=1095,
                  sentence="A residential lease shall not exceed 3 years."),
        make_fact("s2", "renewal notice", "DURATION", "P60D", duration_days=60,
                  sentence="Tenant must provide notice no fewer than 60 days before termination."),
        make_fact("s3", "renewal notice", "DURATION", "P120D", duration_days=120,
                  sentence="No more than 120 days before the termination date."),
        make_fact("s4", "Act", "START", "2019-01-01", date_parsed=date(2019, 1, 1),
                  sentence="This Act entered into force on 1 January 2019."),
        make_fact("s5", "denunciation", "DURATION", "P6M", duration_days=180,
                  sentence="Either party may denounce with six months' notice."),
    ]

    deps = [
        make_dep("s2", "s3", "interval",
                 "notice window is [120d before, 60d before] expiry"),
    ]

    return TemporalDependencyGraph(
        document_id="statute_s47",
        document_type="legislation",
        source_text="Housing Lease Protection Act §47...",
        facts=facts,
        dependencies=deps,
    )


# --- Build synthetic court judgment TDG -----------------------------------

def build_judgment_tdg() -> TemporalDependencyGraph:
    """
    Chen v. Meridian Corp -- synthetic court judgment.

    The Ms. Chen example from the research notes:
      F1 (dismissal) -> +14d -> F2 (notice deadline) -> +30d -> F3 (grievance opens)

    Also has a 60-day grievance window (coreference with statute's P60D renewal notice)
    and a denunciation notice of P3M (contradiction with contract's P6M).
    """
    facts = [
        make_fact("c1", "dismissal", "START", "2025-06-01",
                  date_parsed=date(2025, 6, 1),
                  sentence="Ms. Chen was dismissed on June 1, 2025."),
        make_fact("c2", "notice deadline", "END", "2025-06-15",
                  date_parsed=date(2025, 6, 15),
                  sentence="Employer must issue notice within 14 days, by June 15, 2025."),
        make_fact("c3", "grievance period", "START", "2025-07-15",
                  date_parsed=date(2025, 7, 15),
                  sentence="Grievance period opens 30 days after notice deadline."),
        make_fact("c4", "grievance window", "DURATION", "P60D",
                  duration_days=60,
                  sentence="The 60-day grievance window."),
        make_fact("c5", "grievance filed", "CONTAINS", "2025-07-18",
                  date_parsed=date(2025, 7, 18),
                  sentence="Ms. Chen filed her grievance on July 18, 2025."),
        # Different denunciation notice than contract (P3M vs P6M)
        make_fact("c6", "denunciation", "DURATION", "P3M",
                  duration_days=90,
                  sentence="Employment may be denounced with three months' notice."),
    ]

    deps = [
        make_dep("c1", "c2", "additive", "notice = dismissal + 14d", delta=14, verified=True),
        make_dep("c2", "c3", "additive", "grievance = notice + 30d", delta=30, verified=True),
        make_dep("c3", "c5", "interval", "grievance filed during window"),
    ]

    return TemporalDependencyGraph(
        document_id="court_chen",
        document_type="court_decision",
        source_text="Chen v. Meridian Corp...",
        facts=facts,
        dependencies=deps,
    )


# --- Run tests ------------------------------------------------------------

def main():
    # Load
    seed0_path = os.path.join(os.path.dirname(__file__), "en_contracts_seed0.json")
    if not os.path.exists(seed0_path):
        print(f"ERROR: {seed0_path} not found")
        sys.exit(1)

    contract_tdg = load_tdg(seed0_path)
    statute_tdg = build_statute_tdg()
    judgment_tdg = build_judgment_tdg()

    print(f"Contract: {len(contract_tdg.facts)} facts, {len(contract_tdg.dependencies)} deps")
    print(f"Statute:  {len(statute_tdg.facts)} facts, {len(statute_tdg.dependencies)} deps")
    print(f"Judgment: {len(judgment_tdg.facts)} facts, {len(judgment_tdg.dependencies)} deps")
    print()

    # Link
    linker = CrossDocLinker()
    linker.add_tdg(contract_tdg)
    linker.add_tdg(statute_tdg)
    linker.add_tdg(judgment_tdg)

    links = linker.find_all_links()

    print(f"{'='*70}")
    print(f"CROSS-DOCUMENT LINKS FOUND: {len(links)}")
    print(f"{'='*70}")

    by_type = {}
    for link in links:
        by_type.setdefault(link.link_type, []).append(link)

    for ltype in ["coreference", "contradiction", "structural_analogy"]:
        group = by_type.get(ltype, [])
        print(f"\n-- {ltype.upper()} ({len(group)}) --")
        for link in group:
            print(f"  {link.from_doc}/{link.from_fact} ↔ {link.to_doc}/{link.to_fact}")
            print(f"    conf={link.confidence:.2f}  {link.explanation}")
            if link.value_a or link.value_b:
                print(f"    values: {link.value_a} vs {link.value_b}"
                      f"{'  Δ' + str(link.delta_days) + 'd' if link.delta_days else ''}")
            print()

    # Staleness propagation test
    print(f"\n{'='*70}")
    print("STALENESS PROPAGATION: edit court_chen/c1 (dismissal date)")
    print(f"{'='*70}")

    stale = linker.propagate_staleness("court_chen", "c1", delta_days=30)
    if stale:
        for sf in stale:
            print(f"  STALE: {sf.doc_id}/{sf.fact_id} "
                  f"(was: {sf.old_value}, hops: {sf.hop_distance})")
            print(f"    reason: {sf.reason}")
    else:
        print("  No stale facts found (expected: c2, c3 via additive chain)")

    # Summary
    print(f"\n{'='*70}")
    print("PIPELINE DIAGNOSTIC")
    print(f"{'='*70}")

    print(f"\nContract TDG connectivity:")
    connected = set()
    for d in contract_tdg.dependencies:
        connected.add(d.from_id)
        connected.add(d.to_id)
    print(f"  {len(connected)}/{len(contract_tdg.facts)} facts in edges "
          f"({len(contract_tdg.facts) - len(connected)} isolated)")

    null_facts = sum(1 for f in contract_tdg.facts
                     if not f.timex.date_parsed and f.timex.duration_days is None)
    print(f"  {null_facts}/{len(contract_tdg.facts)} facts have no computable value")

    # Check entity fragmentation
    from collections import Counter
    import re
    prefix = re.compile(r"^Art\.?\s*\d+[\.\d]*\s*", re.IGNORECASE)
    base_entities = Counter()
    for f in contract_tdg.facts:
        base = prefix.sub("", f.entity).strip().lower()
        base_entities[base] += 1
    fragmented = {k: v for k, v in base_entities.items() if v > 1}
    print(f"\n  Entity fragmentation (base names with >1 fact):")
    for name, count in sorted(fragmented.items(), key=lambda x: -x[1]):
        art_variants = set()
        for f in contract_tdg.facts:
            if prefix.sub("", f.entity).strip().lower() == name:
                art_variants.add(f.entity)
        print(f"    '{name}' × {count} -> split into {len(art_variants)} entity groups")

    print(f"\n  -> graph_builder sees {len(set(f.entity for f in contract_tdg.facts))} "
          f"entity groups but only {len(fragmented)} base concepts with >1 fact")
    print(f"  -> article prefix creates {len(set(f.entity for f in contract_tdg.facts)) - len(base_entities)} "
          f"unnecessary groups")


if __name__ == "__main__":
    main()
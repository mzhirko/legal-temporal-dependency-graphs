#!/usr/bin/env python3
"""
Test the TDG pipeline on example texts from all 5 domains.
"""

import os
import sys
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # suppress HuggingFace fork warnings

from tdg_pipeline.pipeline import TDGPipeline

# ---------------------------------------------------------------------------
# Test texts -- one per domain from the research framing doc
# ---------------------------------------------------------------------------

TESTS = [
    {
        "id": "ww2_europe",
        "type": "historical",
        "entity": "World War II in Europe",
        "text": (
            "World War II in Europe began on September 1, 1939, when Germany invaded Poland. "
            "The war lasted approximately 5 years and 8 months, ending on May 8, 1945 with "
            "Germany's unconditional surrender. The Battle of Stalingrad, which took place "
            "between August 23, 1942 and February 2, 1943, marked a crucial turning point "
            "that preceded the Allied advance into Germany."
        ),
    },
    {
        "id": "service_agreement",
        "type": "legal",
        "entity": "Service Agreement",
        "text": (
            "This Service Agreement takes effect on January 15, 2025 (the Effective Date). "
            "All invoices shall be due and payable within 30 days after the Effective Date. "
            "This Agreement shall commence on the Effective Date and continue for a period "
            "of 12 months, terminating on January 15, 2026. Either party may terminate "
            "upon 90 days written notice prior to the termination date."
        ),
    },
    {
        "id": "patient_001",
        "type": "medical",
        "entity": "Patient Treatment",
        "text": (
            "The patient was admitted on March 3, 2024 with acute respiratory symptoms. "
            "Treatment commenced on March 4, 2024 and continued for 14 days. "
            "The patient was discharged on March 17, 2024 with instructions to "
            "return for follow-up within 30 days."
        ),
    },
    {
        "id": "acme_merger",
        "type": "corporate",
        "entity": "Acme Corp Merger",
        "text": (
            "Acme Corp announced the merger on January 10, 2023. "
            "The regulatory review lasted 6 months, concluding on July 8, 2023. "
            "The merger was completed on August 1, 2023, and the integration "
            "period extended for 18 months until February 2025."
        ),
    },
    {
        "id": "marie_curie",
        "type": "biographical",
        "entity": "Marie Curie",
        "text": (
            "Marie Curie was born on November 7, 1867 in Warsaw. She moved to Paris "
            "in 1891 and began her studies at the Sorbonne. She was awarded the Nobel "
            "Prize in Physics on December 10, 1903. Marie Curie died on July 4, 1934, "
            "having lived for 66 years."
        ),
    },
]


def main():
    pipe = TDGPipeline()

    all_results = []
    for test in TESTS:
        print(f"\n{'='*70}")
        print(f"  Domain: {test['type'].upper()} -- {test['id']}")
        print(f"{'='*70}")

        tdg = pipe.process(
            text=test["text"],
            document_id=test["id"],
            document_type=test["type"],
            document_entity=test["entity"],
        )

        print(tdg.summary())

        # Show edit scenarios
        if tdg.edit_scenarios:
            print(f"\n  Sample edit scenario:")
            s = tdg.edit_scenarios[0]
            print(f"    Edit: {s['edit']['role']} {s['edit']['old_value']} -> {s['edit']['new_value']}")
            for c in s["expected_cascades"]:
                note = c.get("note", "")
                print(f"    -> Cascade: {c['role']} {c['old_value']} -> {c['new_value']}  {note}")
            print(f"    Ripple depth={s['ripple_depth']}, breadth={s['ripple_breadth']}")

        all_results.append(tdg.to_dict())
        print()

    # --- Summary statistics ---
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    total_facts = sum(len(r["facts"]) for r in all_results)
    total_deps = sum(len(r["dependencies"]) for r in all_results)
    total_scenarios = sum(len(r["edit_scenarios"]) for r in all_results)
    print(f"  Documents processed: {len(all_results)}")
    print(f"  Total facts extracted: {total_facts}")
    print(f"  Total dependencies found: {total_deps}")
    print(f"  Total edit scenarios generated: {total_scenarios}")

    # Role distribution
    from collections import Counter
    roles = Counter()
    for r in all_results:
        for f in r["facts"]:
            roles[f["role"]] += 1
    print(f"  Role distribution: {dict(roles)}")


if __name__ == "__main__":
    main()

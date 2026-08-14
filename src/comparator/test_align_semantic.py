"""
Regression test for semantic (duration-vs-computed-date) comparison in align.py.

Runs WITHOUT Ollama or Catala -- builds a minimal .catala_en file and an
in-memory TDG duration fact, then drives the public align_and_compare().

Guards two bugs that together made `semantic_match` unreachable (0/45 in the
50-contract run):
  1. the actual Catala input values were never passed into align_and_compare,
     so the anchor date for reconstruction was always missing;
  2. _find_tdg_fact is type-filtered to dates, so a date-valued Catala output
     could never match a TDG *duration* fact -- the semantic branch was dead.

Run:  python -m comparator.test_align_semantic
Exits non-zero on failure.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from comparator.align import align_and_compare
from tdg_pipeline.tdg import TemporalFact, TimexSpan


_CATALA = """# Fisheries Agreement -- duration

This Agreement shall remain in force for a period of four years from its entry into force.

```catala
declaration scope FisheriesAgreement:
  input signing_date content date
  output expiry_date content date

scope FisheriesAgreement:
  definition expiry_date equals signing_date + 4 year
```
"""


def _duration_fact(value: str, days: int, sentence: str) -> TemporalFact:
    return TemporalFact(
        id="f1", entity="duration of the agreement", role="DURATION",
        timex=TimexSpan(text=value, timex_type="DURATION", value=value,
                        start_char=0, end_char=0, duration_days=days),
        sentence=sentence,
    )


def _statuses(report) -> list[str]:
    return [f.status for f in report.fields]


def main() -> int:
    catala_file = Path(tempfile.mkdtemp()) / "fish.catala_en"
    catala_file.write_text(_CATALA)

    catala_outputs = {"expiry_date": "2022-06-30"}  # 2018-07-01 + 1460 days
    inputs = {"signing_date": "2018-07-01"}

    failures = []

    # 1. Without inputs the anchor is unknown -> no semantic match (old behaviour).
    dur = _duration_fact("P4Y", 1460,
                         "This Agreement shall remain in force for a period of four years.")
    rep = align_and_compare(
        document_id="t", tdg_facts=[dur], catala_outputs=catala_outputs,
        catala_status="success", scope_name="FisheriesAgreement",
        repair_attempts=0, catala_file=catala_file, placeholder_fields=set(),
        catala_inputs=None,
    )
    if "semantic_match" in _statuses(rep):
        failures.append("expected NO semantic_match without inputs")

    # 2. With inputs, a matching duration yields semantic_match.
    dur = _duration_fact("P4Y", 1460,
                         "This Agreement shall remain in force for a period of four years.")
    rep = align_and_compare(
        document_id="t", tdg_facts=[dur], catala_outputs=catala_outputs,
        catala_status="success", scope_name="FisheriesAgreement",
        repair_attempts=0, catala_file=catala_file, placeholder_fields=set(),
        catala_inputs=inputs,
    )
    if "semantic_match" not in _statuses(rep):
        failures.append(f"expected semantic_match, got {_statuses(rep)}")

    # 3. A wrong duration yields semantic_mismatch (not a false agreement).
    dur = _duration_fact("P5Y", 1825,
                         "This Agreement shall remain in force for a period of five years.")
    rep = align_and_compare(
        document_id="t", tdg_facts=[dur], catala_outputs=catala_outputs,
        catala_status="success", scope_name="FisheriesAgreement",
        repair_attempts=0, catala_file=catala_file, placeholder_fields=set(),
        catala_inputs=inputs,
    )
    if "semantic_mismatch" not in _statuses(rep):
        failures.append(f"expected semantic_mismatch, got {_statuses(rep)}")

    # 4. Calendar-month case (regression for the false mismatch on seed14):
    #    issue_date 1975-07-19 + 6 month = 1976-01-19 in Catala. A 30-day-month
    #    reconstruction gives 1976-01-15 (delta 4) and wrongly flags mismatch.
    #    Calendar arithmetic must yield semantic_match.
    cal_catala = """# Movement certificate

The certificate must be submitted within six months of the issue date.

```catala
declaration scope MovementCertificateDeadline:
  input issue_date content date
  output submission_deadline content date

scope MovementCertificateDeadline:
  definition submission_deadline equals issue_date + 6 month
```
"""
    cal_file = Path(tempfile.mkdtemp()) / "cert.catala_en"
    cal_file.write_text(cal_catala)
    dur = TemporalFact(
        id="f1", entity="Art.13 Agreement", role="DURATION",
        timex=TimexSpan(text="six months", timex_type="DURATION", value="P6M",
                        start_char=0, end_char=0, duration_days=180),
        sentence="The certificate must be submitted within six months of the issue date.",
    )
    rep = align_and_compare(
        document_id="t", tdg_facts=[dur], catala_outputs={"submission_deadline": "1976-01-19"},
        catala_status="success", scope_name="MovementCertificateDeadline",
        repair_attempts=0, catala_file=cal_file, placeholder_fields=set(),
        catala_inputs={"issue_date": "1975-07-19"},
    )
    if "semantic_match" not in _statuses(rep):
        failures.append(f"seed14 calendar-month case: expected semantic_match, got {_statuses(rep)}")

    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print("OK: semantic_match / semantic_mismatch are reachable and correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
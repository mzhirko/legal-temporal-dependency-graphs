"""
Offline tests for tdg_binder -- no clerk, no LLM, no network.

Uses the REAL seed1 TDG (data/results_contracts_50/en_contracts_seed1.json)
and the input schema implied by the real generated scope
(experiments/catala_50/en_contracts_seed1.catala_en):

    input initial_agreement_date   content date
    input last_amendment_date      content date
    input provisional_start_date   content date
    input termination_notice_period content duration
    output termination_deadline    content date
        = provisional_start_date + termination_notice_period

Acceptance criterion (Step 1 of the plan): with bound inputs, the scope's
arithmetic yields 1995-05-01 (1995-01-01 + 120 days) -- a value-checkable
date instead of a placeholder orphan.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catala_pipeline.tdg_binder import bind_inputs, parse_duration

DATA = Path(__file__).resolve().parents[2] / "data" / "results_contracts_50"

# clerk json-schema property shape: dates/durations arrive as $ref entries.
SEED1_INPUT_PROPERTIES = {
    "initial_agreement_date": {"$ref": "#/definitions/date"},
    "last_amendment_date": {"$ref": "#/definitions/date"},
    "provisional_start_date": {"$ref": "#/definitions/date"},
    "termination_notice_period": {"$ref": "#/definitions/duration"},
}


def _load_seed1_facts() -> list:
    tdg = json.loads((DATA / "en_contracts_seed1.json").read_text())
    return tdg["facts"]


def test_parse_duration():
    assert parse_duration("P120D") == {"years": 0, "months": 0, "days": 120}
    assert parse_duration("P3M") == {"years": 0, "months": 3, "days": 0}
    assert parse_duration("P1Y6M") == {"years": 1, "months": 6, "days": 0}
    assert parse_duration("P2W") == {"years": 0, "months": 0, "days": 14}
    assert parse_duration("120 days notice") == {"years": 0, "months": 0, "days": 120}
    assert parse_duration("6 months") == {"years": 0, "months": 6, "days": 0}
    assert parse_duration("1 year and 6 months") == {"years": 1, "months": 6, "days": 0}
    assert parse_duration("null") is None
    assert parse_duration("2005-09-08") is None  # a date is not a duration
    print("  parse_duration: OK")


def test_seed1_binding():
    facts = _load_seed1_facts()
    result = bind_inputs(SEED1_INPUT_PROPERTIES, facts)
    print(f"  {result.summary_line()}")

    b = result.bindings

    # The three load-bearing bindings.
    assert "termination_notice_period" in b, "notice period must bind"
    assert result.inputs["termination_notice_period"] == {
        "years": 0, "months": 0, "days": 120,
    }, result.inputs["termination_notice_period"]
    assert "notice" in b["termination_notice_period"]["entity"].lower()

    assert "provisional_start_date" in b, "provisional start must bind"
    assert result.inputs["provisional_start_date"] == "1995-01-01"
    assert "provisional" in b["provisional_start_date"]["entity"].lower()

    assert "last_amendment_date" in b, "amendment date must bind"
    assert result.inputs["last_amendment_date"] == "1992-11-27"

    # initial_agreement_date should bind to the 1986 agreement fact --
    # allowed to fall back to LLM if ambiguous, but must not bind wrongly
    # to a 1995 fact.
    if "initial_agreement_date" in b:
        assert result.inputs["initial_agreement_date"] == "1986-07-19", b[
            "initial_agreement_date"
        ]

    # One fact never feeds two variables.
    fact_ids = [x["fact_id"] for x in b.values()]
    assert len(fact_ids) == len(set(fact_ids)), "1:1 binding violated"
    print("  seed1 binding: OK")
    return result


def test_seed1_deadline_arithmetic(result):
    """Simulate the scope body: termination_deadline =
    provisional_start_date + termination_notice_period (pure days here;
    on the real run clerk performs this with Catala date semantics)."""
    anchor = date.fromisoformat(result.inputs["provisional_start_date"])
    dur = result.inputs["termination_notice_period"]
    assert dur["years"] == 0 and dur["months"] == 0  # pure-days case
    deadline = anchor + timedelta(days=dur["days"])
    assert deadline.isoformat() == "1995-05-01", deadline
    print(f"  computed termination_deadline = {deadline} (expected 1995-05-01): OK")


def test_no_binding_below_threshold():
    """Garbage variable names must stay unbound, not bind randomly."""
    facts = _load_seed1_facts()
    result = bind_inputs({"zzz_qqq_xyz": {"$ref": "#/definitions/date"}}, facts)
    assert "zzz_qqq_xyz" in result.unbound
    print("  threshold guard: OK")


if __name__ == "__main__":
    print("tdg_binder offline tests")
    test_parse_duration()
    r = test_seed1_binding()
    test_seed1_deadline_arithmetic(r)
    test_no_binding_below_threshold()
    print("ALL PASS")

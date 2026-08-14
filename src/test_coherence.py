#!/usr/bin/env python3
"""test_coherence.py -- the constraint checker must be right before any number
from it is reportable. Run: python -m pytest test_coherence.py -q
"""
from datetime import date

import coherence as C


def mk(**kw):
    base = dict(case_id="t", presented=date(2023, 1, 10),
                deadline=date(2022, 9, 7), last_date=date(2022, 9, 7),
                verdict=C.OUT_OF_TIME, delta_days=125, complied=False)
    base.update(kw)
    return C.Battery(**base)


# --- the coherent baseline ------------------------------------------------

def test_consistent_battery_is_coherent():
    r = C.check(mk())
    assert r["coherent"] is True
    assert r["violated_independent"] == []
    assert r["constraints"]["K4"] is True  # derived check also holds


def test_in_time_battery_is_coherent():
    r = C.check(mk(presented=date(2022, 8, 1), verdict=C.IN_TIME,
                   delta_days=-37, complied=True))
    assert r["coherent"] is True


def test_exactly_on_deadline_is_in_time():
    """delta 0 must read as in time -- the off-by-one that eats boundary cases."""
    r = C.check(mk(presented=date(2022, 9, 7), verdict=C.IN_TIME,
                   delta_days=0, complied=True))
    assert r["coherent"] is True
    r2 = C.check(mk(presented=date(2022, 9, 7), verdict=C.OUT_OF_TIME,
                    delta_days=0, complied=False))
    assert r2["constraints"]["K3"] is False


# --- each constraint fires independently ----------------------------------

def test_K1_last_date_diverges_from_deadline():
    r = C.check(mk(last_date=date(2022, 9, 8)))
    assert r["constraints"]["K1"] is False
    assert r["coherent"] is False


def test_K2_arithmetic_slip():
    r = C.check(mk(delta_days=124))
    assert r["constraints"]["K2"] is False


def test_K3_verdict_contradicts_own_delta():
    """The C8 shape: arithmetic right, verdict field wrong."""
    r = C.check(mk(verdict=C.IN_TIME, complied=True))
    assert r["constraints"]["K3"] is False
    assert r["constraints"]["K5"] is True   # internally consistent with itself
    assert r["coherent"] is False


def test_K5_paraphrase_only_is_flagged_as_format():
    r = C.check(mk(complied=True))
    assert r["violated_independent"] == ["K5"]
    assert r["format_only"] is True


def test_K4_is_derived_and_excluded_from_the_rate():
    r = C.check(mk(delta_days=124, last_date=date(2022, 9, 8)))
    assert "K4" not in r["evaluable_independent"]
    agg = C.aggregate([r])
    assert agg["per_constraint"]["K4"]["independent"] is False


# --- abstention is not incoherence ----------------------------------------

def test_missing_field_is_not_a_violation():
    r = C.check(mk(deadline=None))
    assert r["complete"] is False
    assert r["constraints"]["K1"] is None
    assert r["constraints"]["K2"] is None
    assert r["coherent"] is not False  # K3/K5 may still be evaluable


def test_all_missing_scores_none_not_zero():
    b = C.Battery(case_id="empty")
    r = C.check(b)
    assert r["coherent"] is None
    agg = C.aggregate([r])
    assert agg["cases_scored"] == 0
    assert agg["incoherence_rate"] is None


def test_aggregate_scores_only_complete_batteries():
    good, partial = C.check(mk()), C.check(mk(presented=None, delta_days=None))
    agg = C.aggregate([good, partial])
    assert agg["cases_total"] == 2
    assert agg["cases_complete"] == 1
    assert agg["cases_scored"] == 1


# --- parsing --------------------------------------------------------------

def test_parses_last_json_object_not_the_restated_schema():
    raw = ('Here is the shape: {"reasoning": "...", "verdict": "in_time"}\n'
           'Working: ...\n{"reasoning": "late by 125", "verdict": "out_of_time"}')
    val, status = C.parse_answer("verdict", raw)
    assert (val, status) == (C.OUT_OF_TIME, "ok")


def test_verdict_synonyms():
    assert C.parse_answer("verdict", '{"verdict": "LATE"}')[0] == C.OUT_OF_TIME
    assert C.parse_answer("verdict", '{"verdict": "Timely"}')[0] == C.IN_TIME


def test_negative_and_string_ints():
    assert C.parse_answer("delta_days", '{"delta_days": -37}')[0] == -37
    assert C.parse_answer("delta_days", '{"delta_days": "125"}')[0] == 125


def test_bool_is_not_an_int():
    """True must not coerce to 1 -- silently turns a bool answer into a delta."""
    assert C.parse_answer("delta_days", '{"delta_days": true}')[1] == "unparseable"


def test_non_iso_date_refused_rather_than_guessed():
    assert C.parse_answer("deadline", '{"deadline": "7 September 2022"}')[1] \
        == "unparseable"
    assert C.parse_answer("deadline", '{"deadline": "2022-13-01"}')[1] \
        == "unparseable"


def test_no_json_and_missing_field_are_distinguished():
    assert C.parse_answer("verdict", "the claim was late")[1] == "no_json"
    assert C.parse_answer("verdict", '{"reasoning": "x"}')[1] == "missing_field"


# --- engine projection ----------------------------------------------------

def test_engine_projection_is_coherent_by_construction():
    """Real EntailmentResult field names: deadline_computed / days_over."""
    b = C.engine_battery("t", {"anchor_date": "2022-06-08",
                               "deadline_computed": "2022-09-07",
                               "action_date": "2023-01-10",
                               "days_over": 125,
                               "verdict": "LATE"})
    r = C.check(b)
    assert b.last_date == b.deadline
    assert b.delta_days == 125
    assert b.complied is False
    assert r["coherent"] is True
    assert r["violated_independent"] == []


def test_engine_timely_projection():
    b = C.engine_battery("t", {"deadline_computed": "2022-09-07",
                               "action_date": "2022-08-01",
                               "days_over": -37, "verdict": "TIMELY"})
    assert b.verdict == C.IN_TIME and b.complied is True
    assert C.check(b)["coherent"] is True


def test_indeterminate_is_abstention_not_a_free_pass():
    """The trap: INDETERMINATE must cost coverage, not score as coherent."""
    b = C.engine_battery("t", {"deadline_computed": "2022-09-07",
                               "action_date": None, "days_over": None,
                               "verdict": "INDETERMINATE"})
    assert b.complete() is False
    assert all(getattr(b, f) is None for f in
               ("presented", "deadline", "last_date", "verdict",
                "delta_days", "complied"))
    assert C.check(b)["coherent"] is None
    agg = C.aggregate([C.check(b)])
    assert agg["cases_scored"] == 0
    assert agg["incoherence_rate"] is None


def test_days_over_is_read_not_recomputed():
    """If the engine ever disagreed with itself, we must SEE it, not paper
    over it by recomputing the delta from the dates."""
    b = C.engine_battery("t", {"deadline_computed": "2022-09-07",
                               "action_date": "2023-01-10",
                               "days_over": 999, "verdict": "LATE"})
    assert b.delta_days == 999
    assert C.check(b)["constraints"]["K2"] is False


def test_engine_accepts_dataclass_not_only_dict():
    class R:
        deadline_computed = "2022-09-07"
        action_date = "2023-01-10"
        days_over = 125
        verdict = "LATE"
    assert C.check(C.engine_battery("t", R()))["coherent"] is True


def test_engine_provenance_is_captured_but_unscored():
    prov = C.engine_provenance({"match_confidence": 0.24,
                                "acas_applied": True,
                                "verdict": "INDETERMINATE",
                                "anchor_date": "2022-06-08"})
    assert prov["match_confidence"] == 0.24
    assert prov["acas_applied"] is True
    assert prov["raw_verdict"] == "INDETERMINATE"


# --- prompts --------------------------------------------------------------

def test_question_order_is_deterministic_and_covers_all():
    a = C.question_order("2025_EAT_155")
    b = C.question_order("2025_EAT_155")
    assert a == b
    assert sorted(a) == sorted(C.QUESTION_IDS)


def test_prompts_differ_only_in_the_question():
    p1 = C.build_prompt("verdict", "STATUTE", "CASE")
    p2 = C.build_prompt("deadline", "STATUTE", "CASE")
    assert p1 != p2
    assert p1.count("=== (B) JUDGMENT ===") == 1


# --- single-prompt condition ----------------------------------------------

def test_single_prompt_contains_every_question_once():
    p = C.build_single_prompt("STATUTE", "JUDGMENT")
    for qid in C.QUESTION_IDS:
        assert C._QUESTIONS[qid]["field"] in p
    assert p.count("=== (B) JUDGMENT ===") == 1


def test_single_prompt_respects_seeded_order():
    order = C.question_order("2025_EAT_155")
    p = C.build_single_prompt("S", "J", "", order)
    positions = [p.index(f"({C._QUESTIONS[q]['field']})") for q in order]
    assert positions == sorted(positions)


def test_one_json_object_supplies_all_six_fields():
    """Single-prompt scoring path: same parser, one object, six pulls."""
    raw = ('{"reasoning": "w", "presented": "2023-01-10", '
           '"deadline": "2022-09-07", "last_date": "2022-09-07", '
           '"verdict": "out_of_time", "delta_days": 125, "complied": false}')
    b = C.Battery(case_id="t")
    for qid in C.QUESTION_IDS:
        val, status = C.parse_answer(qid, raw)
        assert status == "ok"
        setattr(b, C._QUESTIONS[qid]["field"], val)
    assert b.complete()
    assert C.check(b)["coherent"] is True


def test_partial_single_response_is_incomplete_not_incoherent():
    raw = '{"presented": "2023-01-10", "verdict": "out_of_time"}'
    b = C.Battery(case_id="t")
    for qid in C.QUESTION_IDS:
        val, status = C.parse_answer(qid, raw)
        if status == "ok":
            setattr(b, C._QUESTIONS[qid]["field"], val)
    r = C.check(b)
    assert r["complete"] is False
    assert r["coherent"] is not False


def test_single_id_is_dot_free_for_stem_parsing():
    """<case>.<question> is split on the LAST dot, so the question id must
    contain none -- else the API runner mis-parses the filename."""
    assert "." not in C.SINGLE_ID
    
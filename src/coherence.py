#!/usr/bin/env python3
"""coherence.py -- label-free self-consistency battery for timeliness reasoning.

Six questions per case, each asked in its OWN call with NO shared context.
Their answers are mutually determining, so a violation is a self-contradiction
regardless of which answer is correct. No ground truth is consumed anywhere in
this module: scoring is by internal constraint only.

    Q1 presented   (date)  On what date was the claim presented?
    Q2 deadline    (date)  What is the effective deadline?
    Q3 last_date   (date)  Last date on which it could have been presented?
    Q4 verdict     (enum)  Was it presented in time?
    Q5 delta_days  (int)   How many days late? (negative = early)
    Q6 complied    (bool)  Did it comply with the statutory time limit?
                           -- surface variant of Q4, different answer type

CONSTRAINTS

    K1  last_date == deadline
    K2  delta_days == (presented - deadline).days
    K3  (verdict == in_time) == (delta_days <= 0)
    K5  complied == (verdict == in_time)

    K4  (verdict == in_time) == (presented <= last_date)      [DERIVED]

K1, K2, K3, K5 are logically independent. K4 follows from K1+K2+K3 and is
reported as a redundancy check ONLY -- it must not be counted as a fifth
independent test in any rate you publish, or you inflate n.

K5 is the format-sensitivity control: Q4 and Q6 differ in surface form and
answer type but not in content, so a K5-only violation is format sensitivity
rather than arithmetic incoherence. Report that split.

WHY THE ENGINE SCORES ZERO

engine_battery() fills all six fields by projection from ONE EntailmentResult.
Its violation rate is zero analytically, not empirically -- it is a property of
deriving the answers from a single computation rather than answering six times.
Say exactly that in the paper. Presenting it as a won comparison would be
dishonest and a reviewer will catch it.

ABSTENTION IS NOT INCOHERENCE

A case is scored for coherence only when every field it needs is present and
parseable. Unparseable and abstained answers are counted and reported
separately, never folded into the violation rate.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from datetime import date, datetime
from typing import Optional, Any

SCHEMA_VERSION = "coherence-1.0.0"

IN_TIME = "in_time"
OUT_OF_TIME = "out_of_time"

# ---------------------------------------------------------------------------
# Questions. FROZEN. Do not edit between models or between runs -- if you must
# change one, bump SCHEMA_VERSION and rerun every model.
# ---------------------------------------------------------------------------

_PREAMBLE = """\
You are given (A) the text of a statute section governing a time limit and
(B) a tribunal judgment with the tribunal's own timeliness conclusions removed
(marked [REDACTED: ...]). All primary facts and dates remain in the document.

Answer the single question below about the complaint under the statute in (A){claim_hint}.
Think it through, then end your reply with one JSON object and nothing after it.
"""

_QUESTIONS: dict[str, dict[str, Any]] = {
    "presented": {
        "field": "presented",
        "kind": "date",
        "text": "On what date was the claim presented to the tribunal?",
        "schema": '{"reasoning": "<your working>", "presented": "YYYY-MM-DD"}',
    },
    "deadline": {
        "field": "deadline",
        "kind": "date",
        "text": (
            "What is the effective deadline for presenting this complaint -- "
            "that is, the deadline after any early-conciliation extension that "
            "applies on these facts?"
        ),
        "schema": '{"reasoning": "<your working>", "deadline": "YYYY-MM-DD"}',
    },
    "last_date": {
        "field": "last_date",
        "kind": "date",
        "text": (
            "What is the last date on which this complaint could have been "
            "presented and still be in time?"
        ),
        "schema": '{"reasoning": "<your working>", "last_date": "YYYY-MM-DD"}',
    },
    "verdict": {
        "field": "verdict",
        "kind": "verdict",
        "text": "Was this complaint presented in time?",
        "schema": '{"reasoning": "<your working>", "verdict": "in_time" | "out_of_time"}',
    },
    "delta_days": {
        "field": "delta_days",
        "kind": "int",
        "text": (
            "By how many days was this complaint late? Give a single whole "
            "number. Use a NEGATIVE number if it was presented before the "
            "deadline, and 0 if it was presented exactly on the deadline."
        ),
        "schema": '{"reasoning": "<your working>", "delta_days": <integer>}',
    },
    "complied": {
        "field": "complied",
        "kind": "bool",
        "text": (
            "Did the presentation of this complaint comply with the statutory "
            "time limit? Answer true or false."
        ),
        "schema": '{"reasoning": "<your working>", "complied": true | false}',
    },
}

QUESTION_IDS = tuple(_QUESTIONS)


def build_prompt(qid: str, statute_text: str, case_text: str,
                 claim_hint: str = "") -> str:
    """Full prompt for one question. Identical across models by construction."""
    q = _QUESTIONS[qid]
    return (
        _PREAMBLE.format(claim_hint=claim_hint)
        + f"\nQUESTION: {q['text']}\n"
        + f"\nReply with exactly this JSON shape:\n{q['schema']}\n"
        + f"\n=== (A) STATUTE ===\n{statute_text}\n"
        + f"\n=== (B) JUDGMENT ===\n{case_text}\n"
    )


SINGLE_ID = "__all__"

_SINGLE_PREAMBLE = """\
You are given (A) the text of a statute section governing a time limit and
(B) a tribunal judgment with the tribunal's own timeliness conclusions removed
(marked [REDACTED: ...]). All primary facts and dates remain in the document.

Answer ALL of the questions below about the complaint under the statute in (A){claim_hint}.
Think it through, then end your reply with ONE JSON object containing every
field and nothing after it.
"""


def build_single_prompt(statute_text: str, case_text: str,
                        claim_hint: str = "",
                        order: Optional[list[str]] = None) -> str:
    """All six questions in ONE call.

    The contrast condition. Independent calls test whether the model can
    reconstruct the same answer from scratch six times; this tests whether it
    is consistent when it can see its own earlier answers inside a single
    generation. A large gap between the two is evidence that the independent
    incoherence is real recomputation failure rather than a formatting
    artifact -- and that consistency within one response is partly copying.

    Question order is the same seeded order used for the independent run, so
    the two conditions differ in exactly one thing: shared context.
    """
    order = order or list(QUESTION_IDS)
    lines, fields = [], []
    for i, qid in enumerate(order, 1):
        q = _QUESTIONS[qid]
        lines.append(f"{i}. ({q['field']}) {q['text']}")
        fields.append(_SINGLE_FIELD_SHAPE[q["kind"]].format(f=q["field"]))
    return (
        _SINGLE_PREAMBLE.format(claim_hint=claim_hint)
        + "\nQUESTIONS:\n" + "\n".join(lines)
        + '\n\nReply with exactly this JSON shape:\n{"reasoning": "<your working>", '
        + ", ".join(fields) + "}\n"
        + f"\n=== (A) STATUTE ===\n{statute_text}\n"
        + f"\n=== (B) JUDGMENT ===\n{case_text}\n"
    )


_SINGLE_FIELD_SHAPE = {
    "date": '"{f}": "YYYY-MM-DD"',
    "int": '"{f}": <integer>',
    "bool": '"{f}": true | false',
    "verdict": '"{f}": "in_time" | "out_of_time"',
}


def question_order(case_id: str, seed: int = 20260801) -> list[str]:
    """Deterministic per-case question order.

    Calls are independent so order cannot leak information, but recording a
    seeded order makes the run reproducible and forecloses the objection that
    a fixed order interacts with server-side caching.
    """
    import random
    rng = random.Random(f"{seed}:{case_id}")
    ids = list(QUESTION_IDS)
    rng.shuffle(ids)
    return ids


# ---------------------------------------------------------------------------
# Parsing. Strict: a value we cannot read is None (-> abstention), never a guess.
# ---------------------------------------------------------------------------

_JSON_OBJ = re.compile(r"\{[^{}]*\}", re.DOTALL)
_ISO = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$")


def _last_json_object(raw: str) -> Optional[dict]:
    """Last well-formed flat JSON object in the response.

    Last, not first: models often restate the schema before answering it.
    """
    best = None
    for m in _JSON_OBJ.finditer(raw or ""):
        try:
            obj = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            best = obj
    return best


def _as_date(v: Any) -> Optional[date]:
    if isinstance(v, str):
        m = _ISO.match(v)
        if m:
            try:
                return date(int(m[1]), int(m[2]), int(m[3]))
            except ValueError:
                return None
    return None


def _as_int(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and float(v).is_integer():
        return int(v)
    if isinstance(v, str):
        m = re.match(r"^\s*([+-]?\d+)\s*$", v)
        if m:
            return int(m[1])
    return None


def _as_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    return None


def _as_verdict(v: Any) -> Optional[str]:
    if isinstance(v, str):
        s = v.strip().lower().replace(" ", "_").replace("-", "_")
        if s in (IN_TIME, "timely", "in_time_"):
            return IN_TIME
        if s in (OUT_OF_TIME, "late", "outoftime"):
            return OUT_OF_TIME
    return None


_COERCE = {"date": _as_date, "int": _as_int,
           "bool": _as_bool, "verdict": _as_verdict}


def parse_answer(qid: str, raw: str) -> tuple[Any, str]:
    """-> (value, status). status in {ok, no_json, missing_field, unparseable}."""
    obj = _last_json_object(raw)
    if obj is None:
        return None, "no_json"
    field_name = _QUESTIONS[qid]["field"]
    if field_name not in obj:
        return None, "missing_field"
    val = _COERCE[_QUESTIONS[qid]["kind"]](obj[field_name])
    if val is None:
        return None, "unparseable"
    return val, "ok"


# ---------------------------------------------------------------------------
# The battery and its constraints.
# ---------------------------------------------------------------------------

@dataclass
class Battery:
    """One case's six answers. Any field may be None (abstained/unparsed)."""
    case_id: str
    presented: Optional[date] = None
    deadline: Optional[date] = None
    last_date: Optional[date] = None
    verdict: Optional[str] = None
    delta_days: Optional[int] = None
    complied: Optional[bool] = None
    statuses: dict[str, str] = field(default_factory=dict)

    def complete(self) -> bool:
        return all(getattr(self, f) is not None for f in
                   ("presented", "deadline", "last_date",
                    "verdict", "delta_days", "complied"))

    def missing(self) -> list[str]:
        return [f for f in ("presented", "deadline", "last_date",
                            "verdict", "delta_days", "complied")
                if getattr(self, f) is None]


@dataclass
class Constraint:
    key: str
    independent: bool
    description: str


CONSTRAINTS = (
    Constraint("K1", True,
               "last permissible date equals the effective deadline"),
    Constraint("K2", True,
               "delta_days equals presented minus deadline"),
    Constraint("K3", True,
               "verdict agrees with the sign of delta_days"),
    Constraint("K5", True,
               "compliance flag agrees with the verdict (paraphrase control)"),
    Constraint("K4", False,
               "verdict agrees with presented vs last permissible date "
               "(DERIVED from K1+K2+K3; redundancy check only)"),
)

INDEPENDENT = tuple(c.key for c in CONSTRAINTS if c.independent)


def check(b: Battery) -> dict:
    """Evaluate the constraints on one battery.

    Returns per-constraint results, each of which is True (holds), False
    (violated) or None (not evaluable because an input was missing).
    """
    res: dict[str, Optional[bool]] = {}

    res["K1"] = (b.last_date == b.deadline) \
        if (b.last_date and b.deadline) else None

    res["K2"] = (b.delta_days == (b.presented - b.deadline).days) \
        if (b.delta_days is not None and b.presented and b.deadline) else None

    res["K3"] = ((b.verdict == IN_TIME) == (b.delta_days <= 0)) \
        if (b.verdict and b.delta_days is not None) else None

    res["K5"] = (b.complied == (b.verdict == IN_TIME)) \
        if (b.complied is not None and b.verdict) else None

    res["K4"] = ((b.verdict == IN_TIME) == (b.presented <= b.last_date)) \
        if (b.verdict and b.presented and b.last_date) else None

    evaluable = [k for k in INDEPENDENT if res[k] is not None]
    violated = [k for k in evaluable if res[k] is False]

    return {
        "case_id": b.case_id,
        "complete": b.complete(),
        "missing": b.missing(),
        "constraints": res,
        "evaluable_independent": evaluable,
        "violated_independent": violated,
        "coherent": (len(violated) == 0) if evaluable else None,
        # a violation confined to K5 is format sensitivity, not arithmetic
        "format_only": violated == ["K5"],
    }


def aggregate(reports: list[dict]) -> dict:
    """Corpus-level summary. Scored ONLY over fully complete batteries."""
    n = len(reports)
    complete = [r for r in reports if r["complete"]]
    scored = [r for r in complete if r["coherent"] is not None]
    incoherent = [r for r in scored if not r["coherent"]]

    per_constraint = {}
    for c in CONSTRAINTS:
        ev = [r for r in reports if r["constraints"][c.key] is not None]
        vi = [r for r in ev if r["constraints"][c.key] is False]
        per_constraint[c.key] = {
            "independent": c.independent,
            "description": c.description,
            "evaluable": len(ev),
            "violations": len(vi),
            "rate": (len(vi) / len(ev)) if ev else None,
            "cases": sorted(r["case_id"] for r in vi),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "cases_total": n,
        "cases_complete": len(complete),
        "coverage": (len(complete) / n) if n else None,
        "cases_scored": len(scored),
        "incoherent": len(incoherent),
        "incoherence_rate": (len(incoherent) / len(scored)) if scored else None,
        "format_only": sum(1 for r in incoherent if r["format_only"]),
        "per_constraint": per_constraint,
        "incoherent_cases": sorted(r["case_id"] for r in incoherent),
        "_note": ("Rates are over complete batteries only. K4 is derived and "
                  "excluded from incoherence_rate."),
    }


# ---------------------------------------------------------------------------
# Engine side: projection, not answering.
# ---------------------------------------------------------------------------

def engine_battery(case_id: str, result: Any) -> Battery:
    """Project ONE EntailmentResult onto all six fields.

    Field names are wired to tdg_pipeline.entailment.EntailmentResult:

        anchor_date        ISO str | None
        deadline_computed  ISO str | None   -- ALREADY ACAS-extended:
                                              _apply_conciliation() runs before
                                              this is stored, so it is the
                                              EFFECTIVE deadline Q2 asks for
        action_date        ISO str | None
        days_over          int | None       -- >0 late, <=0 timely; same sign
                                              convention as delta_days
        verdict            "TIMELY" | "LATE" | "INDETERMINATE"

    Nothing is re-decided and nothing is recomputed. days_over is READ, not
    derived from (action_date - deadline_computed): deriving it would be a
    second computation, and the absence of a second computation is the whole
    claim. last_date is assigned FROM deadline for the same reason.

    INDETERMINATE is an abstention, not a verdict. It yields an empty battery,
    so an abstaining engine costs coverage instead of scoring a free pass. An
    engine that abstained everywhere would report coverage 0.0 and no
    coherence rate at all -- which is the honest result, not a perfect one.

    Accepts an EntailmentResult, its .to_dict(), or any object exposing the
    same attribute names.
    """
    def g(name):
        if isinstance(result, dict):
            return result.get(name)
        return getattr(result, name, None)

    verdict = _map_engine_verdict(g("verdict"))
    if verdict is None:
        # INDETERMINATE, unrecognised, or absent -> abstention.
        return Battery(case_id=case_id,
                       statuses={q: "engine_abstained" for q in QUESTION_IDS})

    deadline = _coerce_date(g("deadline_computed"))
    presented = _coerce_date(g("action_date"))
    days_over = g("days_over")
    days_over = days_over if isinstance(days_over, int) and not isinstance(
        days_over, bool) else None

    return Battery(
        case_id=case_id,
        presented=presented,
        deadline=deadline,
        last_date=deadline,          # identical by definition, not by agreement
        verdict=verdict,
        delta_days=days_over,        # read off the engine, never recomputed
        complied=(verdict == IN_TIME),
        statuses={q: "engine" for q in QUESTION_IDS},
    )


def engine_provenance(result: Any) -> dict:
    """Non-scored context to archive next to each engine battery.

    match_confidence and acas_applied explain WHY a row abstained or bound as
    it did. Keep them out of the coherence rate and in the run record.
    """
    def g(name):
        if isinstance(result, dict):
            return result.get(name)
        return getattr(result, name, None)
    return {
        "anchor_date": g("anchor_date"),
        "rule_description": g("rule_description"),
        "match_confidence": g("match_confidence"),
        "acas_applied": g("acas_applied"),
        "raw_verdict": g("verdict"),
        "explanation": g("explanation"),
    }


def _coerce_date(v: Any) -> Optional[date]:
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    return _as_date(v)


def _map_engine_verdict(v: Any) -> Optional[str]:
    """TIMELY/LATE -> coherence verdicts. INDETERMINATE -> None (abstention)."""
    if isinstance(v, str):
        s = v.strip().upper()
        if s == "TIMELY":
            return IN_TIME
        if s == "LATE":
            return OUT_OF_TIME
    return None


# ---------------------------------------------------------------------------
# Secondary: accuracy against gold. Kept apart from coherence on purpose.
# ---------------------------------------------------------------------------

def accuracy(b: Battery, gold: dict) -> dict:
    """Compare a battery to gold fields. NEVER feeds the coherence rate."""
    out: dict[str, Any] = {"case_id": b.case_id}
    gp = _as_date(gold.get("presented_date"))
    gd = _as_date(gold.get("deadline_stated"))
    gv = _as_verdict(gold.get("verdict"))

    out["presented_exact"] = (b.presented == gp) if (b.presented and gp) else None
    out["deadline_exact"] = (b.deadline == gd) if (b.deadline and gd) else None
    out["deadline_error_days"] = ((b.deadline - gd).days
                                  if (b.deadline and gd) else None)
    out["verdict_match"] = (b.verdict == gv) if (b.verdict and gv) else None
    return out


def to_jsonable(b: Battery) -> dict:
    d = asdict(b)
    for k, v in d.items():
        if isinstance(v, date):
            d[k] = v.isoformat()
    return d

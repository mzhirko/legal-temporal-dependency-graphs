"""
TDG -> Catala input binder.

The missing wire between the two pipeline halves: instead of asking an LLM
to re-extract input values from raw text (a second, independent extraction
that routinely disagrees with the TDG or returns nulls), bind the Catala
scope's inputs directly to the facts the TDG already extracted.

Binding is deterministic and auditable:
  1. TYPE GATE   -- date inputs only consider dated facts; duration inputs
                   only consider facts whose value/raw_text parses as a
                   duration (ISO-8601 "P120D" or English "120 days").
  2. NAME SCORE  -- token overlap between the snake_case input variable name
                   and the fact's entity name + raw_text (stopwords removed).
  3. ROLE BONUS  -- variable-name cues (start/entry/effective vs
                   end/expiry/termination/deadline) boost facts whose TDG
                   role agrees.
  4. GREEDY 1:1  -- highest-scoring (var, fact) pairs are bound first; each
                   fact binds at most one variable; pairs below MIN_SCORE
                   stay unbound and fall back to the LLM extractor.

Returns full provenance ({var: fact_id, entity, score, value}) so every
number Catala computes can be traced to the TDG node that supplied it.

Pure stdlib. No LLM, no clerk -- safe to unit-test offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------

_ISO_DURATION = re.compile(
    r"^P(?:(?P<years>\d+)Y)?(?:(?P<months>\d+)M)?(?:(?P<weeks>\d+)W)?(?:(?P<days>\d+)D)?$",
    re.IGNORECASE,
)

_TEXT_DURATION = re.compile(
    r"(?P<num>\d+)\s*(?P<unit>year|month|week|day)s?\b", re.IGNORECASE
)


def parse_duration(value) -> dict | None:
    """
    Parse a duration into {"years": int, "months": int, "days": int}.

    Accepts ISO-8601 ("P120D", "P3M", "P1Y6M"), plain English
    ("120 days", "6 months", "1 year and 6 months"), or an
    already-structured dict. Weeks are folded into days.
    Returns None if nothing parseable is found or all components are zero.
    """
    if isinstance(value, dict):
        out = {
            "years": int(value.get("years") or 0),
            "months": int(value.get("months") or 0),
            "days": int(value.get("days") or 0),
        }
        return out if any(out.values()) else None

    if not isinstance(value, str) or not value.strip() or value.strip().lower() == "null":
        return None

    text = value.strip()

    m = _ISO_DURATION.match(text)
    if m and any(m.groupdict().values()):
        g = {k: int(v) if v else 0 for k, v in m.groupdict().items()}
        return {
            "years": g["years"],
            "months": g["months"],
            "days": g["days"] + 7 * g["weeks"],
        }

    totals = {"years": 0, "months": 0, "days": 0}
    for m in _TEXT_DURATION.finditer(text):
        n, unit = int(m.group("num")), m.group("unit").lower()
        if unit == "year":
            totals["years"] += n
        elif unit == "month":
            totals["months"] += n
        elif unit == "week":
            totals["days"] += 7 * n
        else:
            totals["days"] += n
    return totals if any(totals.values()) else None


# ---------------------------------------------------------------------------
# Tokenization / scoring
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "date", "of", "the", "a", "an", "on", "in", "for", "by", "to", "and",
    "or", "as", "at", "this", "that", "with", "its", "be", "is",
}

# variable-name cue -> TDG roles it agrees with
_ROLE_CUES = {
    "start": {"START"},
    "entry": {"START"},
    "effective": {"START"},
    "commencement": {"START"},
    "signature": {"START"},
    "signing": {"START"},
    "signed": {"START"},
    "force": {"START"},
    "initial": {"START"},
    "end": {"END", "DEADLINE"},
    "expiry": {"END", "DEADLINE"},
    "expiration": {"END", "DEADLINE"},
    "termination": {"END", "DEADLINE"},
    "deadline": {"END", "DEADLINE"},
    "notice": {"DURATION"},
    "period": {"DURATION"},
    "duration": {"DURATION"},
    "term": {"DURATION"},
}

MIN_SCORE = 1.0  # below this, leave unbound (LLM fallback handles it)


def _tokens(name: str) -> set[str]:
    parts = re.split(r"[_\W]+", name.lower())
    return {p for p in parts if p and p not in _STOPWORDS}


def _fact_tokens(fact: dict) -> set[str]:
    return _tokens(str(fact.get("entity", ""))) | _tokens(str(fact.get("raw_text", "")))


def _score(var_name: str, fact: dict) -> float:
    """Name-overlap score + role-agreement bonus. Higher is better."""
    var_toks = _tokens(var_name)
    if not var_toks:
        return 0.0
    overlap = var_toks & _fact_tokens(fact)
    score = float(len(overlap))

    role = str(fact.get("role", "")).upper()
    for cue, roles in _ROLE_CUES.items():
        if cue in var_toks:
            if role in roles:
                score += 0.5
            elif role and score > 0:
                score -= 0.25  # cue present but role disagrees
    return score


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------

@dataclass
class BindingResult:
    inputs: dict = field(default_factory=dict)        # {var: clerk-ready value}
    bindings: dict = field(default_factory=dict)      # {var: provenance dict}
    unbound: set = field(default_factory=set)         # vars needing LLM fallback

    def summary_line(self) -> str:
        bound = ", ".join(
            f"{v}<-{b['fact_id']}({b['score']:.1f})" for v, b in self.bindings.items()
        )
        return f"bound: [{bound or '-'}] | unbound: {sorted(self.unbound) or '-'}"


def _input_kind(schema_spec: dict) -> str:
    """'date' | 'duration' | 'other' from a clerk json-schema property spec."""
    ref = str(schema_spec.get("$ref", ""))
    if "date" in ref:
        return "date"
    if "duration" in ref:
        return "duration"
    return "other"


def bind_inputs(input_properties: dict, tdg_facts: list) -> BindingResult:
    """
    Bind Catala scope inputs to TDG facts.

    Args:
        input_properties: {var_name: schema_spec} -- the ``properties`` dict of
            the scope's input schema, exactly as returned by
            ``catala_runner.json_schema(...)[0]["definitions"]
            [f"{scope}_in"]["properties"]``.
        tdg_facts: list of TDG fact dicts (``tdg_data["facts"]``). TemporalFact
            objects also work if they expose the same attributes via __dict__.

    Returns:
        BindingResult with clerk-ready ``inputs`` for every bound variable,
        per-variable ``bindings`` provenance, and the ``unbound`` remainder.
    """
    facts = [f if isinstance(f, dict) else vars(f) for f in tdg_facts]

    # Pre-compute what each fact can supply.
    dated, durations = [], []
    for f in facts:
        d = f.get("date_parsed") or (
            f.get("value") if re.match(r"^\d{4}-\d{2}-\d{2}$", str(f.get("value", ""))) else None
        )
        if d:
            dated.append((f, str(d)))
        dur = parse_duration(f.get("value")) or parse_duration(f.get("raw_text"))
        if dur:
            durations.append((f, dur))

    # Score every compatible (var, fact) pair.
    candidates = []  # (score, var, fact, clerk_value)
    for var, spec in input_properties.items():
        kind = _input_kind(spec)
        if kind == "date":
            pool = [(f, v) for f, v in dated]
        elif kind == "duration":
            pool = [(f, v) for f, v in durations]
        else:
            continue  # bools/ints: no reliable TDG source -- leave to fallback
        for f, value in pool:
            s = _score(var, f)
            if s >= MIN_SCORE:
                candidates.append((s, var, f, value))

    # Greedy one-to-one assignment, best score first.
    candidates.sort(key=lambda c: -c[0])
    result = BindingResult(unbound=set(input_properties.keys()))
    used_facts: set[str] = set()
    for s, var, f, value in candidates:
        fid = str(f.get("id", id(f)))
        if var not in result.unbound or fid in used_facts:
            continue
        result.inputs[var] = value
        result.bindings[var] = {
            "fact_id": fid,
            "entity": f.get("entity"),
            "role": f.get("role"),
            "score": round(s, 2),
            "value": value,
            "raw_text": f.get("raw_text"),
        }
        result.unbound.discard(var)
        used_facts.add(fid)

    # ------------------------------------------------------------------
    # Second pass: role-unique fallback for GENERIC variable names.
    # "start_date" / "signature_date" carry a role cue but no entity
    # tokens, so name overlap is ~0. If the cued role identifies exactly
    # ONE unused type-compatible fact, that binding is safe; if several
    # facts share the role, stay unbound (ambiguity goes to the LLM
    # fallback -- a wrong deterministic bind is worse than no bind).
    # ------------------------------------------------------------------
    for var in sorted(result.unbound):
        kind = _input_kind(input_properties[var])
        if kind == "date":
            pool = dated
        elif kind == "duration":
            pool = durations
        else:
            continue
        cued_roles: set[str] = set()
        for cue in _tokens(var) & set(_ROLE_CUES):
            cued_roles |= _ROLE_CUES[cue]
        if not cued_roles:
            continue
        matches = [
            (f, v) for f, v in pool
            if str(f.get("role", "")).upper() in cued_roles
            and str(f.get("id", id(f))) not in used_facts
        ]
        if len(matches) != 1:
            continue  # zero or ambiguous -- leave to LLM
        f, value = matches[0]
        fid = str(f.get("id", id(f)))
        result.inputs[var] = value
        result.bindings[var] = {
            "fact_id": fid,
            "entity": f.get("entity"),
            "role": f.get("role"),
            "score": 1.0,
            "method": "role_unique",
            "value": value,
            "raw_text": f.get("raw_text"),
        }
        result.unbound.discard(var)
        used_facts.add(fid)

    return result

"""
Two solvers with one shared entry point.

Both receive the SAME item (the document text + the question). The difference
is only in *how they reason*:

BaselineLLMSolver  -- the "traditional best approach": hand the whole document
                     to a strong LLM and ask the question directly. The LLM
                     must read the text AND do the calendar reasoning in one
                     shot. Pluggable backend via the OpenAI-compatible client
                     already used in tdg_pipeline.llm_pipeline (works against
                     OpenAI or a local Ollama server).

StructuredSolver   -- "the method": the dependency structure is made explicit
                     (here, the gold structure read off the deliberately
                     explicit prose), then the answer is COMPUTED with the
                     project's own deterministic, calendar-correct engine:
                       * deadline -> tdg_pipeline.entailment.check_entailment
                       * cascade  -> calendar-correct re-computation along the
                                     additive edges, with cross-document
                                     coreference resolved by shared date.
                     No LLM is involved in the reasoning step, so it runs with
                     no API and cannot make an arithmetic mistake.

Why this is a fair test of the thesis, not a rigged demo
--------------------------------------------------------
* Ground truth is the calendar (computed in benchmark.py with relativedelta),
  independent of both solvers -- so a baseline error is wrong against the
  calendar, not against another model (this is the "shared-blindspot" concern
  from the notes, handled directly).
* The documents are written so extraction is trivial; the only thing that
  varies across items is how much connected multi-step reasoning the answer
  needs. The structured solver is therefore being credited only for the
  reasoning step the thesis claims is its contribution.
* The structured engine is the project's real engine, not a bespoke oracle.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from dateutil.relativedelta import relativedelta

from proof.benchmark import Item
from tdg_pipeline.tdg import TemporalDependencyGraph, TemporalFact
from tdg_pipeline.entailment import check_entailment, _offset_from_text


# --- answer container -------------------------------------------------------

@dataclass
class Answer:
    item_id: str
    task: str
    solver: str
    # deadline
    deadline: Optional[str] = None
    verdict: Optional[str] = None
    # cascade: event-name -> ISO date
    updates: dict = field(default_factory=dict)
    # bookkeeping
    raw: str = ""
    error: Optional[str] = None
    trace: str = ""


# --- structured solver (the method) -----------------------------------------

class StructuredSolver:
    name = "structured"

    def __init__(self, minus_one_day: bool = False):
        # synthetic benchmark uses plain "within N of X" (no UK -1 day rule)
        self.minus_one_day = minus_one_day

    def answer(self, item: Item) -> Answer:
        if item.task == "deadline":
            return self._deadline(item)
        if item.task == "cascade":
            return self._cascade(item)
        return Answer(item.item_id, item.task, self.name,
                      error=f"unknown task {item.task}")

    def _deadline(self, item: Item) -> Answer:
        tdg = item.gold_tdgs[0]
        results = check_entailment(tdg, tdg, minus_one_day=self.minus_one_day)
        if not results:
            return Answer(item.item_id, item.task, self.name,
                          error="no rule discovered")
        r = results[0]
        return Answer(
            item.item_id, item.task, self.name,
            deadline=r.deadline_computed, verdict=r.verdict,
            trace=r.explanation,
        )

    def _cascade(self, item: Item) -> Answer:
        # merge facts/deps across all documents
        facts: dict[str, TemporalFact] = {}
        deps = []
        for tdg in item.gold_tdgs:
            for f in tdg.facts:
                facts[f.id] = f
            deps += list(tdg.dependencies)

        edit = item.gold["edit"]
        root_id = edit["fact"]
        new_root = date.fromisoformat(edit["new"])

        # recompute absolute dates along additive edges (calendar arithmetic)
        new_dates: dict[str, date] = {root_id: new_root}
        # iterate to a fixpoint over the (linear/tree) additive structure
        changed = True
        while changed:
            changed = False
            for d in deps:
                if d.constraint_type != "additive":
                    continue
                if d.from_id in new_dates and d.to_id not in new_dates:
                    off = _offset_from_text(d.constraint_expr or "")
                    if off is None:
                        continue
                    new_dates[d.to_id] = off.apply(new_dates[d.from_id])
                    changed = True

        # cross-document coreference: a fact with no incoming additive edge
        # whose ORIGINAL date equals a re-computed fact's original date is the
        # same real-world event -> it inherits the corrected date. This is the
        # cross-doc link the flat reader misses.
        incoming = {d.to_id for d in deps if d.constraint_type == "additive"}
        for fid, f in facts.items():
            if fid in new_dates or fid in incoming:
                continue
            if f.timex.date_parsed is None:
                continue
            for other_id, of in facts.items():
                if other_id == fid or other_id not in new_dates:
                    continue
                if of.timex.date_parsed == f.timex.date_parsed:
                    new_dates[fid] = new_dates[other_id]
                    break

        # output keyed by event name, excluding the edited root
        updates = {}
        trace_lines = []
        for fid, nd in new_dates.items():
            if fid == root_id:
                continue
            ent = facts[fid].entity
            updates[ent] = nd.isoformat()
            old = facts[fid].timex.date_parsed
            trace_lines.append(
                f"{ent}: {old.isoformat() if old else '?'} -> {nd.isoformat()}")
        return Answer(item.item_id, item.task, self.name, updates=updates,
                      trace="; ".join(trace_lines))


# --- calendar-naive ablation (no LLM) ----------------------------------------
# Same explicit structure as the structured solver, but arithmetic is done the
# "before the calendar fix" way: a month = 30 days, a year = 365 days, and a
# cascade shifts dependents by a fixed day-delta. This is exactly the
# approximation the project removed from the comparator (see notes: "replaced
# 30-day-month / 365-day-year reconstruction with calendar arithmetic ...
# DELETED a hand-tuned tolerance knob"). It is therefore a faithful ablation of
# the calendar component, not a strawman: structured and calendar_naive differ
# ONLY in how the same offsets are applied to the same dates. It shows the cost
# of calendar-naive reasoning on realistic legal dates (month-ends, leap years).

class CalendarNaiveSolver:
    name = "calendar_naive"

    @staticmethod
    def _approx_days(expr: str) -> int:
        off = _offset_from_text(expr or "")
        if off is None:
            return 0
        return off.years * 365 + off.months * 30 + off.days

    def answer(self, item: Item) -> Answer:
        from datetime import timedelta
        if item.task == "deadline":
            tdg = item.gold_tdgs[0]
            dep = next((d for d in tdg.dependencies
                        if d.constraint_type == "additive"), None)
            fmap = {f.id: f for f in tdg.facts}
            anchor = fmap[dep.from_id].timex.date_parsed if dep else None
            if anchor is None:
                return Answer(item.item_id, item.task, self.name,
                              error="no anchor")
            deadline = anchor + timedelta(days=self._approx_days(dep.constraint_expr))
            # action = the latest dated fact after the anchor
            post = [f.timex.date_parsed for f in tdg.facts
                    if f.timex.date_parsed and f.timex.date_parsed > anchor]
            action = max(post) if post else None
            verdict = None
            if action is not None:
                verdict = "LATE" if (action - deadline).days > 0 else "TIMELY"
            return Answer(item.item_id, item.task, self.name,
                          deadline=deadline.isoformat(), verdict=verdict,
                          trace=f"naive: {anchor} + {self._approx_days(dep.constraint_expr)}d")
        if item.task == "cascade":
            facts, deps = {}, []
            for tdg in item.gold_tdgs:
                for f in tdg.facts:
                    facts[f.id] = f
                deps += list(tdg.dependencies)
            edit = item.gold["edit"]
            root_id = edit["fact"]
            new_dates = {root_id: date.fromisoformat(edit["new"])}
            changed = True
            while changed:
                changed = False
                for d in deps:
                    if d.constraint_type != "additive":
                        continue
                    if d.from_id in new_dates and d.to_id not in new_dates:
                        new_dates[d.to_id] = new_dates[d.from_id] + timedelta(
                            days=self._approx_days(d.constraint_expr))
                        changed = True
            incoming = {d.to_id for d in deps if d.constraint_type == "additive"}
            for fid, f in facts.items():
                if fid in new_dates or fid in incoming or f.timex.date_parsed is None:
                    continue
                for oid, of in facts.items():
                    if oid in new_dates and of.timex.date_parsed == f.timex.date_parsed:
                        new_dates[fid] = new_dates[oid]
                        break
            updates = {facts[fid].entity: nd.isoformat()
                       for fid, nd in new_dates.items() if fid != root_id}
            return Answer(item.item_id, item.task, self.name, updates=updates)
        return Answer(item.item_id, item.task, self.name, error="unknown task")


# --- baseline solver (flat-text LLM) -----------------------------------------

_DEADLINE_INSTR = (
    "You are given a legal document. Read it and answer the question. "
    "Show no working. Return ONLY a JSON object on a single line of the form "
    '{"deadline": "YYYY-MM-DD", "verdict": "TIMELY" or "LATE"}. '
    "Use real calendar arithmetic (months have different numbers of days; "
    "watch month-ends and leap years)."
)

_CASCADE_INSTR = (
    "You are given one or more legal documents that describe dated events, "
    "some defined relative to others. A root date is then corrected. Work out "
    "the corrected date of each requested event, following the chain of "
    "dependencies, using real calendar arithmetic. If an event is mentioned in "
    "more than one document it is the same event and must get one consistent "
    "date. Return ONLY a JSON object on a single line mapping each requested "
    'event name to its corrected date, e.g. {"notice deadline": "YYYY-MM-DD", '
    '"grievance window": "YYYY-MM-DD"}.'
)


class BaselineLLMSolver:
    name = "baseline_llm"

    def __init__(self, model: str, base_url: Optional[str] = None,
                 temperature: float = 0.0):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            from openai import OpenAI  # lazy: only needed when actually running
            key = os.environ.get("OPENAI_API_KEY", "ollama")
            self._client = OpenAI(base_url=self.base_url, api_key=key)
        return self._client

    def warmup(self) -> None:
        """Force the backend to load the model before timed scoring begins.

        Ollama loads a model on first request and can 500 ("model failed to
        load … resource limitations") while it is still spinning up under
        memory pressure. Retrying a trivial call until one succeeds means the
        first real items aren't lost to a cold start."""
        import time
        for attempt in range(8):
            try:
                self._chat("Reply with OK.", "ping", _retry=False)
                return
            except Exception:
                time.sleep(min(2 ** attempt, 30))

    def _chat(self, instruction: str, user: str, _retry: bool = True) -> str:
        client = self._client_lazy()
        import time
        attempts = 5 if _retry else 1
        last = None
        for attempt in range(attempts):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[{"role": "system", "content": instruction},
                              {"role": "user", "content": user}],
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # transient 5xx / load errors: back off and retry
                last = e
                msg = str(e)
                transient = ("500" in msg or "failed to load" in msg
                             or "resource" in msg or "timeout" in msg.lower())
                if not _retry or attempt == attempts - 1 or not transient:
                    raise
                time.sleep(min(2 ** attempt, 30))
        raise last  # unreachable, but keeps type checkers happy

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        if not text:
            return None
        t = re.sub(r"```(?:json)?", "", text).strip()
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def answer(self, item: Item) -> Answer:
        docs = "\n\n".join(f"[{did}]\n{txt}" for did, txt in item.documents)
        if item.task == "deadline":
            user = f"{docs}\n\nQuestion: {item.question}"
            raw = self._chat(_DEADLINE_INSTR, user)
            j = self._parse_json(raw) or {}
            return Answer(item.item_id, item.task, self.name,
                          deadline=j.get("deadline"),
                          verdict=(j.get("verdict") or "").upper() or None,
                          raw=raw)
        if item.task == "cascade":
            wanted = sorted(item.gold["updates"].keys())
            user = (f"{docs}\n\nQuestion: {item.question}\n"
                    f"Requested events (use these exact names as keys): "
                    f"{', '.join(wanted)}")
            raw = self._chat(_CASCADE_INSTR, user)
            j = self._parse_json(raw) or {}
            updates = {k: str(v) for k, v in j.items() if isinstance(v, str)}
            return Answer(item.item_id, item.task, self.name,
                          updates=updates, raw=raw)
        return Answer(item.item_id, item.task, self.name,
                      error=f"unknown task {item.task}")


# --- scoring -----------------------------------------------------------------

# Accepted date spellings, tried in order. ISO is the requested format; the
# others are common ways a model restates the same calendar date. We do NOT
# try month-first numeric (%m-%d-%Y): the models observed here write day-first
# (e.g. "13-02-2023"), so allowing month-first would risk crediting a *wrong*
# date that happens to swap to the gold under the other reading.
_DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y",
                 "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y")


def _parse_date_flexible(s: Optional[str]) -> Optional[date]:
    """Parse a date the model emitted, tolerating non-ISO spellings.

    The benchmark measures temporal *reasoning*, not format compliance, so a
    correctly computed date counts even when the model ignored the requested
    ISO layout. Returns None if no known format matches (then it simply can't
    be credited, which is the safe direction)."""
    if not s:
        return None
    s = s.strip()
    for f in _DATE_FORMATS:
        try:
            return datetime.strptime(s, f).date()
        except ValueError:
            continue
    return None


def score(item: Item, ans: Answer) -> dict:
    """Value-based scoring against the calendar ground truth.

    Dates are compared by calendar value, not string, so "13-02-2023" and
    "2023-02-13" count as the same answer. A separate `*_format_iso` flag
    records whether the model also obeyed the requested ISO layout, so format
    compliance stays reportable without contaminating the reasoning score."""
    if item.task == "deadline":
        gold_d = _parse_date_flexible(item.gold["deadline"])
        got_d = _parse_date_flexible(ans.deadline)
        d_ok = (got_d is not None and got_d == gold_d)
        v_ok = (ans.verdict == item.gold["verdict"])
        return {
            "verdict_correct": bool(v_ok),
            "deadline_correct": bool(d_ok),
            "fully_correct": bool(v_ok and d_ok),
            "n_fields": 1,
            "n_correct": int(d_ok),
            # compliance (not part of n_correct): did it answer in ISO as asked?
            "deadline_format_iso": bool(ans.deadline == item.gold["deadline"]),
        }
    if item.task == "cascade":
        gold = item.gold["updates"]
        n = len(gold)
        correct = 0
        iso = 0
        for k, v in gold.items():
            got = ans.updates.get(k)
            if _parse_date_flexible(got) == _parse_date_flexible(v):
                correct += 1
                if got == v:
                    iso += 1
        return {
            "fully_correct": bool(correct == n),
            "n_fields": n,
            "n_correct": correct,
            "n_format_iso": iso,
        }
    return {"fully_correct": False, "n_fields": 0, "n_correct": 0}
"""
Controlled temporal-reasoning benchmark.

Purpose
-------
Isolate the *reasoning* claim of the thesis from the *extraction* problem.
Every document here states its dates and offsets explicitly in plain legal
prose, so extraction is trivial for any competent reader. What varies is the
amount of *connected, multi-step calendar reasoning* the question requires.

The thesis prediction is:

    A current LLM reading the document as flat text and answering directly
    degrades as the dependency chain deepens, the offset is phrased less
    literally, or the chain crosses documents. A method that makes the
    dependency structure explicit once and then computes the answer with
    deterministic calendar arithmetic does not degrade.

Ground truth is computed here with `dateutil.relativedelta`, i.e. real
calendar arithmetic, NOT a 30-day-month / 365-day-year approximation. It is
therefore independent of any LLM and independent of the engine under test:
a wrong answer is wrong against the calendar, not against another model.

Two task types
--------------
deadline : the document states an anchor date, a limitation period, and an
           action date. Question: what is the deadline, and was the action in
           time? (This mirrors the real Use Case B / s.111 task. Single
           additive hop, which the existing entailment engine handles exactly.)

cascade  : the document states a root date and a chain of facts defined
           relative to it (F2 = F1 + d1, F3 = F2 + d2, ...). The root date is
           then corrected. Question: list the new value of every dependent
           fact. This is the Ms. Chen "ripple" problem and is where flat-text
           reading misses the connected structure -- especially the cross-doc
           variant where a second document restates the final event with a
           vague offset and never mentions the intermediate step.

Difficulty axes
---------------
hops        : number of additive steps in the chain (cascade: 2 or 3)
offset_form : how the period is written
                'digit'   -> "within 3 months", "14 days"
                'natural' -> "within three months", "fourteen days"
                'vague'   -> idiom a human resolves to the same offset
                             ("a fortnight" = 14d, "a quarter of a year" = 3m,
                              "half a year" = 6m, "a year and a day" = 1y1d)
locality    : 'single' (one document) or 'cross' (two documents, Ms. Chen)

The point of the 'vague' form is NOT to be ambiguous: each idiom has one
intended offset, stated in the gold record. It tests whether the solver
resolves ordinary legal idiom to the correct arithmetic.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

# make `tdg_pipeline` importable when run from code/src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dateutil.relativedelta import relativedelta

from tdg_pipeline.tdg import (
    TemporalDependencyGraph,
    TemporalFact,
    TemporalDependency,
    TimexSpan,
)


# --- calendar offsets ------------------------------------------------------

@dataclass(frozen=True)
class Offset:
    """A calendar offset with a digit / spelled / idiom surface form."""
    years: int = 0
    months: int = 0
    days: int = 0

    def apply(self, anchor: date) -> date:
        return anchor + relativedelta(years=self.years, months=self.months,
                                      days=self.days)

    def phrase(self, form: str) -> str:
        """Surface text for this offset in the requested form."""
        if form == "vague":
            v = _VAGUE.get((self.years, self.months, self.days))
            if v:
                return v
            # fall through to natural if no idiom defined
            form = "natural"
        parts = []
        units = [("year", self.years), ("month", self.months), ("day", self.days)]
        for name, n in units:
            if n:
                num = str(n) if form == "digit" else _spell(n)
                parts.append(f"{num} {name}{'s' if n != 1 else ''}")
        return " and ".join(parts) if parts else "0 days"


_VAGUE = {
    (0, 0, 14): "a fortnight",
    (0, 3, 0): "a quarter of a year",
    (0, 6, 0): "half a year",
    (1, 0, 1): "a year and a day",
    (0, 1, 15): "a month and a half",
}

_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
          7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
          12: "twelve", 14: "fourteen", 15: "fifteen", 30: "thirty",
          21: "twenty-one"}


def _spell(n: int) -> str:
    return _WORDS.get(n, str(n))


def _fmt(d: date) -> str:
    """Human date as it would appear in a UK legal document: '30 November 2024'."""
    return d.strftime("%-d %B %Y") if hasattr(d, "strftime") else d.isoformat()


# --- benchmark item ---------------------------------------------------------

@dataclass
class Item:
    item_id: str
    task: str                                   # "deadline" | "cascade"
    difficulty: dict
    documents: list[tuple[str, str]]            # (doc_id, prose)
    question: str
    gold: dict                                  # task-specific exact answer
    gold_tdgs: list[TemporalDependencyGraph] = field(default_factory=list)
    notes: str = ""


# --- gold-TDG helpers -------------------------------------------------------
# These encode exactly what a competent extractor reads off the (deliberately
# explicit) prose. The structured solver runs the REAL engine over them; it
# does not get the gold answer, only the gold structure.

def _date_fact(fid: str, entity: str, d: date, sentence: str,
               role: str = "CONTAINS") -> TemporalFact:
    return TemporalFact(
        id=fid, entity=entity, role=role,
        timex=TimexSpan(text=_fmt(d), timex_type="DATE", value=d.isoformat(),
                        start_char=0, end_char=0, date_parsed=d),
        sentence=sentence,
    )


def _dur_fact(fid: str, entity: str, off: Offset, sentence: str) -> TemporalFact:
    iso = "P" + (f"{off.years}Y" if off.years else "") + \
          (f"{off.months}M" if off.months else "") + \
          (f"{off.days}D" if off.days else "")
    dd = off.years * 365 + off.months * 30 + off.days
    return TemporalFact(
        id=fid, entity=entity, role="DURATION",
        timex=TimexSpan(text=sentence, timex_type="DURATION", value=iso,
                        start_char=0, end_char=0, duration_days=dd),
        sentence=sentence,
    )


def _edge(a: str, b: str, off: Offset) -> TemporalDependency:
    # NB: the engine's offset parser (_offset_from_text) needs a word boundary
    # between units, so emit "1y 1d" (spaced), not the compressed "1y1d".
    toks = []
    if off.years:
        toks.append(f"{off.years}y")
    if off.months:
        toks.append(f"{off.months}m")
    if off.days:
        toks.append(f"{off.days}d")
    expr = "+" + " ".join(toks)
    return TemporalDependency(from_id=a, to_id=b, constraint_type="additive",
                              constraint_expr=expr,
                              delta_days=off.apply(date(2000, 1, 1)).toordinal()
                              - date(2000, 1, 1).toordinal())


# --- deadline items ---------------------------------------------------------

def _deadline_item(idx: int, anchor: date, off: Offset, form: str,
                   late_by_days: int) -> Item:
    """One single-hop limitation item. `late_by_days` > 0 => action is late."""
    deadline = off.apply(anchor)
    action = deadline + timedelta(days=late_by_days)
    verdict = "LATE" if late_by_days > 0 else "TIMELY"

    period = off.phrase(form)
    prose = (
        f"The Claimant's employment ended on {_fmt(anchor)}. "
        f"Under the governing rule, a complaint must be presented to the "
        f"tribunal within {period} of the date employment ended. "
        f"The Claimant presented the complaint on {_fmt(action)}."
    )
    doc_id = f"deadline_{idx:02d}"

    facts = [
        _date_fact("a", "date employment ended", anchor,
                   f"The Claimant's employment ended on {_fmt(anchor)}.",
                   role="END"),
        _dur_fact("b", "complaint time limit", off,
                  f"a complaint must be presented to the tribunal within "
                  f"{period} of the date employment ended"),
        _date_fact("c", "complaint presented to the tribunal", action,
                   f"The Claimant presented the complaint on {_fmt(action)}."),
    ]
    deps = [_edge("a", "b", off)]
    tdg = TemporalDependencyGraph(document_id=doc_id, document_type="legal",
                                  source_text=prose, facts=facts, dependencies=deps)

    return Item(
        item_id=doc_id, task="deadline",
        difficulty={"hops": 1, "offset_form": form, "locality": "single"},
        documents=[(doc_id, prose)],
        question=("What is the deadline for presenting the complaint, and was "
                  "the complaint presented in time? Answer with the deadline "
                  "date and the verdict TIMELY or LATE."),
        gold={"deadline": deadline.isoformat(), "action_date": action.isoformat(),
              "verdict": verdict, "days_over": late_by_days},
        gold_tdgs=[tdg],
        notes=f"anchor {anchor} + {off} -> deadline {deadline}; action {action}",
    )


# --- cascade items ----------------------------------------------------------

def _cascade_item(idx: int, root: date, chain: list[Offset], form: str,
                  locality: str, new_root: date) -> Item:
    """
    Build a chain F1 -> F2 -> ... and an edit of F1 to new_root.
    Each Fk+1 = Fk + chain[k] (calendar arithmetic). The gold lists the NEW
    value of every dependent fact after the root edit.
    """
    # original absolute dates along the chain
    dates = [root]
    for off in chain:
        dates.append(off.apply(dates[-1]))
    # new dates after editing the root, recomputed along the SAME offsets
    new_dates = [new_root]
    for off in chain:
        new_dates.append(off.apply(new_dates[-1]))

    n = len(chain)  # number of dependent facts
    doc_id = f"cascade_{idx:02d}"

    # readable names for the chain events
    names = ["dismissal", "notice deadline", "grievance window",
             "appeal deadline", "final review"][: n + 1]

    # -- prose, document A --
    sents = [f"The {names[0]} took effect on {_fmt(dates[0])}."]
    facts = [_date_fact("f1", names[0], dates[0], sents[0], role="END")]
    deps = []
    for k, off in enumerate(chain):
        period = off.phrase(form)
        s = (f"The {names[k+1]} falls {period} after the "
             f"{names[k]}, on {_fmt(dates[k+1])}.")
        sents.append(s)
        facts.append(_date_fact(f"f{k+2}", names[k+1], dates[k+1], s))
        deps.append(_edge(f"f{k+1}", f"f{k+2}", off))
    doc_a = " ".join(sents)
    tdg_a = TemporalDependencyGraph(document_id=doc_id + "_A",
                                    document_type="legal", source_text=doc_a,
                                    facts=facts, dependencies=deps)
    documents = [(doc_id + "_A", doc_a)]
    tdgs = [tdg_a]

    # gold: every dependent event (named) gets its new date
    gold_updates = {names[k + 1]: new_dates[k + 1].isoformat() for k in range(n)}

    # -- cross-document variant: a 2nd doc restates the FINAL event only,
    #    with a vague offset measured from the root, and never mentions the
    #    intermediate steps. This is the Ms. Chen paraphrase. --
    if locality == "cross":
        total = relativedelta(dates[-1], dates[0])  # true root->final offset
        # describe vaguely; the true offset is what matters, not the words
        approx = _approx_phrase(dates[0], dates[-1])
        doc_b_id = doc_id + "_B"
        s_b = (f"Per the internal record, the {names[n]} is not reached until "
               f"about {approx} after the {names[0]} of {_fmt(dates[0])}, "
               f"placing it in {dates[-1].strftime('%B %Y')}.")
        doc_b = s_b
        # In doc B the final event is the SAME real-world event as f{n+1}.
        # Coreference is established by identical original date.
        fb = _date_fact("g1", names[n], dates[-1], s_b)
        tdg_b = TemporalDependencyGraph(document_id=doc_b_id,
                                        document_type="legal", source_text=doc_b,
                                        facts=[fb], dependencies=[])
        documents.append((doc_b_id, doc_b))
        tdgs.append(tdg_b)
        # gold for doc B's restatement: it co-refers to the final event, so
        # the SAME named event must receive the SAME corrected date.
        gold_updates[names[n]] = new_dates[-1].isoformat()

    q = (f"The {names[0]} date is corrected from {_fmt(root)} to "
         f"{_fmt(new_root)}. Give the corrected date of every other dated "
         f"event that depends on it"
         + (", including the one named in the internal record."
            if locality == "cross" else "."))

    return Item(
        item_id=doc_id, task="cascade",
        difficulty={"hops": n, "offset_form": form, "locality": locality},
        documents=documents,
        question=q,
        gold={"edit": {"fact": "f1", "old": root.isoformat(),
                       "new": new_root.isoformat()},
              "updates": gold_updates},
        gold_tdgs=tdgs,
        notes=f"chain {[ (o.years,o.months,o.days) for o in chain]} form={form} {locality}",
    )


def _approx_phrase(a: date, b: date) -> str:
    """A deliberately rounded human description of the a->b gap (for cross-doc)."""
    days = (b - a).days
    months = round(days / 30.4)
    if months <= 1:
        return "a month and a half"
    if abs(months - round(days / 30.4)) == 0 and months % 1 == 0:
        return f"{_spell(months) if months in _WORDS else months} months or so"
    return f"about {months} months"


# --- benchmark assembly -----------------------------------------------------

def build_benchmark() -> list[Item]:
    items: list[Item] = []

    # deadline: stress calendar boundaries (month-end, leap year) x surface form
    # x polarity. Offsets chosen so a 30-day/365-day approximation gives a
    # different (wrong) answer than true calendar arithmetic.
    deadline_specs = [
        # (anchor, offset, forms, late_by for [timely, late])
        (date(2024, 11, 30), Offset(months=3), ["digit", "natural", "vague"]),
        (date(2023, 12, 1),  Offset(years=1, days=1), ["digit", "natural", "vague"]),
        (date(2024, 1, 31),  Offset(months=6), ["digit", "natural", "vague"]),
    ]
    i = 0
    for anchor, off, forms in deadline_specs:
        for form in forms:
            for late_by in (-3, 5):     # one clearly timely, one clearly late
                items.append(_deadline_item(i, anchor, off, form, late_by))
                i += 1

    # cascade: chain length 2 and 3, single + cross, across surface forms.
    # Ms. Chen base chain: dismissal -> +14d notice -> +30d grievance.
    chen = [Offset(days=14), Offset(days=30)]
    longer = [Offset(days=14), Offset(months=1), Offset(months=3)]
    j = 0
    cascade_specs = [
        (date(2025, 6, 1), chen,   "digit",   "single", date(2025, 7, 1)),
        (date(2025, 6, 1), chen,   "natural", "single", date(2025, 7, 1)),
        (date(2025, 1, 30), chen,  "digit",   "single", date(2025, 2, 27)),
        (date(2025, 6, 1), chen,   "digit",   "cross",  date(2025, 7, 1)),
        (date(2025, 6, 1), chen,   "natural", "cross",  date(2025, 7, 1)),
        (date(2024, 1, 31), longer, "digit",  "single", date(2024, 3, 31)),
        (date(2024, 1, 31), longer, "natural","single", date(2024, 3, 31)),
        (date(2024, 1, 31), longer, "digit",  "cross",  date(2024, 3, 31)),
    ]
    for root, chain, form, loc, new_root in cascade_specs:
        items.append(_cascade_item(j, root, chain, form, loc, new_root))
        j += 1

    return items


if __name__ == "__main__":
    bench = build_benchmark()
    print(f"{len(bench)} items\n")
    for it in bench:
        print(f"[{it.item_id}] {it.task:8} {it.difficulty}")
        for did, txt in it.documents:
            print(f"   ({did}) {txt[:120]}{'...' if len(txt) > 120 else ''}")
        print(f"   Q: {it.question[:110]}")
        print(f"   gold: {it.gold}")
        print()


# ════════════════════════════════════════════════════════════════════════════
# Scaled benchmark generation (added for the head-to-head at larger N).
#
# The canonical build_benchmark() above is left untouched so previously stored
# results stay reproducible. build_benchmark_scaled() generates many more
# controlled items per reported bucket, with the SAME guarantees:
#   * ground truth is calendar arithmetic (relativedelta), independent of any
#     solver -- see Offset.apply / _cascade_item;
#   * anchors are biased toward calendar-stressing dates (month-ends, leap-year
#     boundaries) and offsets toward month/year units, so the calendar_naive
#     ablation is genuinely discriminated rather than accidentally agreeing;
#   * 'natural' offsets are always spelled (never collapse to digits) and
#     'vague' offsets are drawn only from the unambiguous idiom set, so each
#     surface-form condition actually tests what it claims to.
# Everything is seeded, so a run is reproducible from (n_per_bucket, seed).
# ════════════════════════════════════════════════════════════════════════════

import random as _random

# Full English speller for 0..999 so 'natural' phrasing never falls back to a
# digit. Agrees with the small _WORDS table on every value it already covered.
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def _spell_n(n: int) -> str:
    if n < 0:
        return str(n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + (f"-{_ONES[o]}" if o else "")
    if n < 1000:
        h, rem = divmod(n, 100)
        return _ONES[h] + " hundred" + (f" and {_spell_n(rem)}" if rem else "")
    return str(n)


# Redirect the original _spell through the full speller. (Verified to produce
# identical text for every number used by the canonical benchmark.)
def _spell(n: int) -> str:  # noqa: F811  (intentional override)
    return _spell_n(n)


# Calendar-stressing building blocks ----------------------------------------

# Month-end / leap-boundary anchors: these are where a 30-day-month / 365-day
# approximation diverges from real calendar arithmetic.
_STRESS_ANCHORS = [
    date(2024, 1, 31), date(2024, 1, 30), date(2024, 1, 29),  # -> short Feb
    date(2023, 1, 31), date(2023, 1, 30), date(2023, 1, 29),  # non-leap Feb
    date(2024, 2, 29), date(2024, 8, 31), date(2024, 10, 31),
    date(2024, 11, 30), date(2023, 12, 1), date(2024, 12, 31),
    date(2024, 3, 31), date(2024, 5, 31), date(2024, 7, 31),
    date(2023, 11, 30), date(2024, 4, 30), date(2024, 6, 30),
]

# Offsets where calendar vs naive most often disagree (month/year units), plus
# a couple of day-denominated ones for coverage.
_MONTHYEAR_OFFSETS = [
    Offset(months=1), Offset(months=3), Offset(months=6), Offset(months=9),
    Offset(months=18), Offset(years=1), Offset(years=2), Offset(years=1, days=1),
]
_DAY_OFFSETS = [Offset(days=14), Offset(days=30), Offset(days=90)]
# Offsets that have an unambiguous idiom in _VAGUE.
_VAGUE_OFFSETS = [Offset(years=y, months=m, days=d) for (y, m, d) in _VAGUE]


def build_benchmark_scaled(n_per_bucket: int = 20, seed: int = 0) -> list[Item]:
    """Generate ~n_per_bucket items for each reported bucket.

    Reported buckets (see run_proof.bucket): deadline/offset={digit,natural,
    vague} and cascade/hops={2,3}/{single,cross}. Returns
    n_per_bucket * 7 items (3 deadline + 4 cascade buckets).
    """
    rng = _random.Random(seed)
    items: list[Item] = []
    idx = 0
    jdx = 0

    # -- deadline buckets: one per surface form --
    for form in ("digit", "natural", "vague"):
        pool = _VAGUE_OFFSETS if form == "vague" else (_MONTHYEAR_OFFSETS + _DAY_OFFSETS)
        for k in range(n_per_bucket):
            anchor = rng.choice(_STRESS_ANCHORS)
            off = rng.choice(pool)
            # alternate clearly-timely / clearly-late with small margins so the
            # exact-date answer (and sometimes the verdict) is calendar-sensitive
            late_by = rng.choice([-3, -1, 4, 7]) if k % 2 else rng.choice([-7, -4, 1, 3])
            items.append(_deadline_item(idx, anchor, off, form, late_by))
            idx += 1

    # -- cascade buckets: hops x locality, surface form alternated --
    cascade_buckets = [
        (2, "single"), (2, "cross"), (3, "single"), (3, "cross"),
    ]
    for hops, loc in cascade_buckets:
        for k in range(n_per_bucket):
            form = "digit" if k % 2 else "natural"
            root = rng.choice(_STRESS_ANCHORS)
            # build a chain mixing day and month offsets to stress the calendar
            chain_pool = [Offset(days=14), Offset(days=30), Offset(months=1),
                          Offset(months=2), Offset(months=3), Offset(years=1)]
            chain = [rng.choice(chain_pool) for _ in range(hops)]
            # edit the root to another stressing date (distinct from root)
            new_root = rng.choice([a for a in _STRESS_ANCHORS if a != root])
            items.append(_cascade_item(jdx, root, chain, form, loc, new_root))
            jdx += 1

    return items
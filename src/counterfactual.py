#!/usr/bin/env python3
"""Counterfactual perturbation harness (Use Case B, Step 4).

WHY THIS EXISTS
---------------
The gold set is n=7. That supports a mechanism taxonomy, not rates, and the v2
matcher was developed on those same 7 cases. Both objections have one answer:
manufacture more cases whose ground truth is COMPUTED rather than annotated.

Take a real judgment, shift one date in the text, and the correct verdict moves
with it in a way the statute fully determines. So we can generate unlimited
items with known answers, at zero annotation cost, on documents published after
every model's training cutoff.

    real case -> shift the anchor by k days -> rewrite the text
              -> recompute the deadline from the statute rule
              -> the verdict is now known WITHOUT a human

This is the GSM-Symbolic move (Mirzadeh et al.) for legal deadlines: perturb the
inputs symbolically, keep an executable oracle, and watch whether the system
under test tracks the boundary or pattern-matches around it.

WHAT IS AND IS NOT UNDER TEST -- READ THIS BEFORE QUOTING ANY NUMBER
------------------------------------------------------------------
The engine computes the ground truth here, so THE ENGINE CANNOT BE SCORED
AGAINST IT. That would be circular and the number would be 100% by construction.
The engine's correctness is established elsewhere and independently: 6/7 real
tribunal verdicts, 2/2 judge-stated deadlines and 1/1 judge-stated boundary
reproduced to the day (run_gold_facts.py), on facts a human verified.

What this harness legitimately tests:

  1. THE LLM BASELINE (`--emit-prompts`), on perturbed text. Does its verdict
     flip on the right day? This is the real experiment: C1/C7 stop being
     anecdotes from 7 cases and become rates over hundreds of items.
  2. THE FULL PIPELINE, later, on the same perturbed text. The perturbed
     documents are UNSEEN by the v2 matcher, which is the direct answer to the
     "tuned on 7" objection.
  3. THE ENGINE'S INTERNAL CONSISTENCY (`--self-check`), which is a property
     test, not an accuracy claim. Verdict must be monotone in the anchor: moving
     the anchor later can never turn TIMELY into LATE. Exactly one flip per
     sweep. A violation is an engine bug, and the s.207B pause is where one
     would hide, because the pause engages only if Day A < deadline -- a genuine
     discontinuity, not a smooth curve.

Perturbation surfaces (the ones the notes identified as real):
  anchor     EDT / date of the act. 2025_EAT_155 is the wild-type instance: a 2-day
             shift flips the real verdict, because Day A lands on the deadline
             and s.207B engages.
  presented  ET1 presentation date. The cheapest flip; tests the boundary only.
  day_a      ACAS Day A. Sweeps across the s.207B precondition, where the engine
             has a step change rather than a slope.

Usage
-----
    # generate items + the engine's oracle answers, and self-check the engine
    python src/counterfactual.py --gold data/ground_truth/ground_truth_gold.json \
        --statute-tdg era_1996_s111=data/results_uk/era_1996_s111.json \
        --statute-tdg eqa_2010_s123=data/results_uk/eqa_2010_s123.json \
        --surface anchor --range 45 --self-check \
        --out data/experiments/counterfactual/anchor_sweep.json

    # emit perturbed DOCUMENTS + frozen prompts for the LLM condition
    python src/counterfactual.py ... --texts data/experiments/baseline/inputs \
        --emit-prompts data/experiments/counterfactual/prompts
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_gold_facts import _apply_conciliation, _d, _rule_for  # noqa: E402

# case_id -> the judgment text file stem, mirroring run_baseline's CASE_FILE
CASE_FILE = {
    "2026_EAT_64_s111": "2026_EAT_64",
    "2026_EAT_64_s123": "2026_EAT_64",
    "2025_EAT_155": "2025_EAT_155",
    "2026_EAT_14": "2026_EAT_14",
    "2026_EAT_76": "2026_EAT_76",
    "2026_EAT_59": "2026_EAT_59",
    "kj_2026_EAT_46": "2026_EAT_46",
}

SURFACES = ("anchor", "presented", "day_a")


# -- date rendering --------------------------------------------------------
def _long(d: date) -> str:
    """'8 June 2022' -- the form UK judgments actually use."""
    return f"{d.day} {d:%B} {d.year}"


def _forms(d: date) -> list[str]:
    """Every surface form of a date we are willing to rewrite, longest first so
    that '08 June 2022' is replaced before '8 June 2022' can match inside it."""
    out = [
        f"{d.day:02d} {d:%B} {d.year}",
        f"{d.day} {d:%B} {d.year}",
        f"{d.day:02d} {d:%b} {d.year}",
        f"{d.day} {d:%b} {d.year}",
        f"{d:%B} {d.day}, {d.year}",
        d.isoformat(),
    ]
    for suf in ("st", "nd", "rd", "th"):
        out.append(f"{d.day}{suf} {d:%B} {d.year}")
    return sorted(set(out), key=len, reverse=True)


def perturb_text(text: str, old: date, new: date) -> tuple[str, int]:
    """Rewrite every mention of `old` as `new`. Returns (text, n_replacements).

    Deliberately literal: we only touch exact date strings. If a judgment refers
    to the date obliquely ('the following Tuesday'), we do NOT chase it -- the
    item is dropped instead (see `--require-hits`), because a half-perturbed
    document has an ambiguous ground truth and would silently poison the set.
    """
    n = 0
    for form in _forms(old):
        if form in text:
            # match the same surface form of the new date
            if re.match(r"^\d{2} ", form):
                rep = f"{new.day:02d} {new:%B} {new.year}"
            elif re.match(r"^\d{1,2}(st|nd|rd|th) ", form):
                suf = {1: "st", 2: "nd", 3: "rd"}.get(
                    new.day if new.day not in (11, 12, 13) else 0,
                    "th") if new.day % 10 in (1, 2, 3) or True else "th"
                suf = ("th" if 11 <= new.day <= 13
                       else {1: "st", 2: "nd", 3: "rd"}.get(new.day % 10, "th"))
                rep = f"{new.day}{suf} {new:%B} {new.year}"
            elif form == old.isoformat():
                rep = new.isoformat()
            elif re.match(r"^[A-Z][a-z]+ \d{1,2},", form):
                rep = f"{new:%B} {new.day}, {new.year}"
            elif re.match(r"^\d{1,2} [A-Z][a-z]{2} \d{4}$", form):
                rep = f"{new.day} {new:%b} {new.year}"
            else:
                rep = _long(new)
            n += text.count(form)
            text = text.replace(form, rep)
    return text, n


def residual_refs(text: str, old: date, new: date) -> int:
    """Count surviving references that still imply the OLD date AFTER
    perturbation. `perturb_text` rewrites full-date surface forms only;
    judgments also say "in July 2020" or "on 11 July". Once the shift
    changes the month (or the day), those become contradictory temporal
    evidence about the anchor: an extractor that picks one up produces a
    second, unshifted anchor candidate, and a conflict-aware matcher
    rightly abstains (observed: cf_anchor_pipeline_gemma, 225/427
    INDETERMINATE). A component is only counted when it actually changed
    -- "July 2020" is still true if the new date is also in July 2020.
    Call on the ALREADY-perturbed text, so full forms are gone and the
    year-lookahead below cannot double-count them.
    """
    n = 0
    if (old.year, old.month) != (new.year, new.month):
        # bare month-year only: a leading day number (incl. ordinal) means
        # this is part of a FULL date, which perturbation already handled
        # (rewritten if it was the old date; deliberately frozen if it is a
        # genuinely different, post-anchor date)
        guard = r"(?<!\d )(?<!\d)(?<!st )(?<!nd )(?<!rd )(?<!th )"
        n += len(re.findall(guard + rf"\b{old:%B} {old.year}\b", text))
        n += len(re.findall(guard + rf"\b{old:%b} {old.year}\b", text))
    if (old.month, old.day) != (new.month, new.day):
        # day-month with NO trailing year (a trailing year means a full
        # date of a different year, i.e. a genuinely different date)
        for day in (str(old.day), f"{old.day:02d}",
                    *(f"{old.day}{s}" for s in ("st", "nd", "rd", "th"))):
            n += len(re.findall(
                rf"\b{day} {old:%B}\b(?!\s+\d{{4}})", text))
        n += len(re.findall(
            rf"\b{old:%B} {old.day}\b(?!,?\s+\d{{4}})", text))
    return n


# -- translate-past perturbation -------------------------------------------
# Shifting ONLY the anchor's full-date strings leaves its satellites behind:
# bare "July 2020" narrative references, the dismissal letter "dated 10 July
# 2020", the whole pre-dismissal history. The document then contradicts
# itself about when the anchor happened; a conflict-gated matcher rightly
# abstains and an LLM silently anchors on the stale mention (measured:
# cf_anchor_pipeline_gemma). Translation fixes this by construction: every
# date expression that parses to ON OR BEFORE the original anchor moves by
# the same k, and every bare month-year in an anchor-or-earlier month moves
# with it. Dates after the anchor (ACAS, presentation, hearings) stay put,
# so the anchor->presented interval still sweeps the statutory boundary.
# Bare years are NOT rewritten (citation years "[2018] EAT", statute years
# "Act 1996" make that unsafe); they are counted and reported instead.

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}
_MON_ABBR = {m[:3]: i for m, i in _MONTHS.items()}
_MON_RE = "|".join(_MONTHS)
_ABBR_RE = "|".join(_MON_ABBR)

_DATE_PATTERNS = [
    # 8 June 2022 / 08 June 2022 / 8th June 2022
    (re.compile(rf"\b(\d{{1,2}})(st|nd|rd|th)? ({_MON_RE}) (\d{{4}})\b"),
     lambda m: (int(m.group(4)), _MONTHS[m.group(3)], int(m.group(1)))),
    # 8 Jun 2022
    (re.compile(rf"\b(\d{{1,2}}) ({_ABBR_RE}) (\d{{4}})\b"),
     lambda m: (int(m.group(3)), _MON_ABBR[m.group(2)], int(m.group(1)))),
    # June 8, 2022
    (re.compile(rf"\b({_MON_RE}) (\d{{1,2}}),? (\d{{4}})\b"),
     lambda m: (int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2)))),
    # 2022-06-08
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
     lambda m: (int(m.group(1)), int(m.group(2)), int(m.group(3)))),
]
_MONTHYEAR_RE = re.compile(rf"\b({_MON_RE}) (\d{{4}})\b")


def _fmt_like(m: re.Match, d: date, pat_idx: int) -> str:
    """Render `d` in the same surface style as the matched original."""
    if pat_idx == 0:
        if m.group(2):  # ordinal
            suf = ("th" if 11 <= d.day <= 13
                   else {1: "st", 2: "nd", 3: "rd"}.get(d.day % 10, "th"))
            return f"{d.day}{suf} {d:%B} {d.year}"
        pad = m.group(1).startswith("0")
        return f"{d.day:02d} {d:%B} {d.year}" if pad else f"{d.day} {d:%B} {d.year}"
    if pat_idx == 1:
        return f"{d.day} {d:%b} {d.year}"
    if pat_idx == 2:
        comma = "," in m.group(0)
        return f"{d:%B} {d.day}{',' if comma else ''} {d.year}"
    return d.isoformat()


def translate_past(text: str, anchor: date, k: int,
                   cutoff: Optional[date] = None) -> tuple[str, int, int]:
    """Shift every date expression parsing to BEFORE `cutoff` by k days,
    plus bare month-year mentions of anchor-or-earlier months and bare
    day-month mentions resolving before the cutoff. Returns (text,
    n_shifted, n_bare_year_flags).

    cutoff defaults to the anchor+1d, but should be the first ORACLE-frozen
    event (ACAS Day A, else presentation): dates between the anchor and
    that event -- service of the dismissal decision, pre-ACAS
    correspondence -- are satellites of the moved anchor, and freezing them
    recreates the contradiction in mirror image (observed on 2025_EAT_155:
    "summary dismissal in September 2020" vs "service was 12 July 2020",
    where the EDT dispute IS about 10-13 July).
    """
    if cutoff is None:
        cutoff = anchor + timedelta(days=1)
    delta = timedelta(days=k)
    spans: list[tuple[int, int, str]] = []
    claimed: list[tuple[int, int]] = []

    for idx, (pat, parse) in enumerate(_DATE_PATTERNS):
        for m in pat.finditer(text):
            if any(s < m.end() and m.start() < e for s, e, _ in spans):
                continue
            try:
                d = date(*parse(m))
            except ValueError:
                continue
            if d < cutoff:
                spans.append((m.start(), m.end(), _fmt_like(m, d + delta, idx)))
            else:
                claimed.append((m.start(), m.end()))  # frozen: on/after cutoff

    # bare day-month ("10 July at 12.18pm"): resolve in the anchor's year;
    # shift when that lands before the cutoff, render again without a year
    bare_dm = re.compile(rf"\b(\d{{1,2}})(st|nd|rd|th)? ({_MON_RE})\b(?! \d{{4}})")
    for m in bare_dm.finditer(text):
        if any(s < m.end() and m.start() < e for s, e, _ in spans):
            continue
        if any(s < m.end() and m.start() < e for s, e in claimed):
            continue
        try:
            d = date(anchor.year, _MONTHS[m.group(3)], int(m.group(1)))
        except ValueError:
            continue
        if d < cutoff:
            nd = d + delta
            if m.group(2):
                suf = ("th" if 11 <= nd.day <= 13
                       else {1: "st", 2: "nd", 3: "rd"}.get(nd.day % 10, "th"))
                spans.append((m.start(), m.end(), f"{nd.day}{suf} {nd:%B}"))
            else:
                spans.append((m.start(), m.end(), f"{nd.day} {nd:%B}"))

    for m in _MONTHYEAR_RE.finditer(text):
        if any(s < m.end() and m.start() < e for s, e, _ in spans):
            continue
        if any(s < m.end() and m.start() < e for s, e in claimed):
            continue                       # part of a frozen full date
        y, mo = int(m.group(2)), _MONTHS[m.group(1)]
        if (y, mo) <= (anchor.year, anchor.month):
            nd = date(y, mo, 15) + delta
            spans.append((m.start(), m.end(), f"{nd:%B} {nd.year}"))

    for s, e, rep in sorted(spans, reverse=True):
        text = text[:s] + rep + text[e:]

    # bare-year flags: anchor year changed and stale bare years may remain
    flags = 0
    new_anchor = anchor + delta
    if new_anchor.year != anchor.year:
        flags = len(re.findall(
            rf"(?<!\[)\b{anchor.year}\b(?!\])", text))
    return text, len(spans), flags


# -- the oracle ------------------------------------------------------------
def decide(offset, anchor: date, presented: date,
           day_a: Optional[date], day_b: Optional[date]) -> dict:
    """The statute, executed. This is the ground truth for a perturbed item."""
    primary = offset.apply(anchor)
    effective, acas = _apply_conciliation(primary, day_a, day_b)
    days_over = (presented - effective).days
    return {
        "primary_deadline": primary.isoformat(),
        "effective_deadline": effective.isoformat(),
        "acas_applied": acas,
        "days_over": days_over,
        "verdict": "TIMELY" if days_over <= 0 else "LATE",
    }


def _shift(case: dict, surface: str, k: int):
    """Return (anchor, presented, day_a, day_b) with `surface` moved by k days."""
    a, p = _d(case.get("anchor_date")), _d(case.get("presented_date"))
    da, db = _d(case.get("acas_day_a")), _d(case.get("acas_day_b"))
    if surface == "anchor":
        a = a + timedelta(days=k)
    elif surface == "presented":
        p = p + timedelta(days=k)
    elif surface == "day_a":
        da = da + timedelta(days=k)
    return a, p, da, db


def find_boundary(offset, case: dict, surface: str, span: int = 1200
                  ) -> Optional[int]:
    """The k at which the verdict flips, by bisection.

    Legitimate only because the verdict is monotone in each surface (later
    anchor -> later deadline -> TIMELY can never revert to LATE), which
    `--self-check` verifies rather than assumes. Returns the FIRST k whose
    verdict differs from k-1, or None if no flip exists within +/-span.
    """
    def verdict(k: int) -> str:
        a, p, da, db = _shift(case, surface, k)
        return decide(offset, a, p, da, db)["verdict"]

    lo, hi = -span, span
    v_lo, v_hi = verdict(lo), verdict(hi)
    if v_lo == v_hi:
        return None                       # no boundary in range
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if verdict(mid) == v_lo:
            lo = mid
        else:
            hi = mid
    return hi


def sweep(offset, case: dict, surface: str, rng: int,
          center: Optional[int] = None) -> list[dict]:
    """Shift one date across center +/- rng days; recompute the truth at each.

    `center` defaults to 0 (the real case). With --auto-center it becomes the
    boundary k, so the items straddle the flip instead of sitting 600 days from
    it -- which is the difference between a set that tests a decision boundary
    and a set of 91 identical LATE answers.
    """
    a0, p0 = _d(case.get("anchor_date")), _d(case.get("presented_date"))
    da0, db0 = _d(case.get("acas_day_a")), _d(case.get("acas_day_b"))
    if a0 is None or p0 is None:
        return []
    if surface == "day_a" and da0 is None:
        return []                       # no EC dates: nothing to sweep
    c = center or 0

    items = []
    for k in range(c - rng, c + rng + 1):
        a, p, da, db = _shift(case, surface, k)
        if surface == "day_a" and db is not None and da > db:
            continue                    # Day A after Day B is not a real state
        if p < a:
            continue                    # presented before the anchor event
        if surface == "anchor" and da is not None and da < a:
            continue                    # ACAS contact before the (shifted)
                                        # anchor is not a real state either
        truth = decide(offset, a, p, da, db)
        items.append(dict(k=k, anchor=a.isoformat(), presented=p.isoformat(),
                          acas_day_a=da.isoformat() if da else None,
                          acas_day_b=db.isoformat() if db else None, **truth))
    return items


def flip_analysis(items: list[dict], surface: str) -> dict:
    """Where does the verdict change, and does the engine behave lawfully?

    Monotonicity: later anchor -> later deadline -> TIMELY is more likely, never
    less. Later presentation -> LATE is more likely, never less. More than one
    flip means the engine is not monotone in that variable, which is a bug.
    """
    if not items:
        return {}
    flips = [(items[i - 1], items[i]) for i in range(1, len(items))
             if items[i]["verdict"] != items[i - 1]["verdict"]]
    order_ok = True
    if surface in ("anchor", "day_a"):
        # LATE ... then TIMELY (shifting the clock start later helps the claimant)
        seq = [it["verdict"] for it in items]
        order_ok = seq == sorted(seq, key=lambda v: 0 if v == "LATE" else 1)
    elif surface == "presented":
        seq = [it["verdict"] for it in items]
        order_ok = seq == sorted(seq, key=lambda v: 0 if v == "TIMELY" else 1)
    return {
        "n_items": len(items),
        "n_flips": len(flips),
        "monotone": order_ok and len(flips) <= 1,
        "flip_at_k": flips[0][1]["k"] if flips else None,
        "boundary_date": (flips[0][1][{"anchor": "anchor",
                                       "presented": "presented",
                                       "day_a": "acas_day_a"}[surface]]
                          if flips else None),
        "verdicts": {"TIMELY": sum(1 for i in items if i["verdict"] == "TIMELY"),
                     "LATE": sum(1 for i in items if i["verdict"] == "LATE")},
        "acas_engaged": {"yes": sum(1 for i in items if i["acas_applied"]),
                         "no": sum(1 for i in items if not i["acas_applied"])},
    }


PROMPT = """You are given the text of an employment tribunal judgment and the \
governing statutory provision.

STATUTE
{statute}

JUDGMENT
{judgment}

TASK
Determine whether the claim was presented in time.
Show your date arithmetic, then end with exactly these three lines:
DEADLINE: <YYYY-MM-DD>
VERDICT: <in_time|out_of_time>
DAYS: <integer, negative if in time>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--statute-tdg", action="append", default=[],
                    metavar="STATUTE=PATH")
    ap.add_argument("--surface", choices=SURFACES, default="anchor")
    ap.add_argument("--range", type=int, default=30,
                    help="sweep +/- this many days (default 30)")
    ap.add_argument("--auto-center", action="store_true",
                    help="centre each sweep on that case's verdict boundary "
                         "instead of on the real dates. Without it a case 597 "
                         "days from its boundary yields 91 identical answers "
                         "and tests nothing; with it every case straddles its "
                         "own flip and the set is near-balanced by "
                         "construction.")
    ap.add_argument("--texts", help="dir of judgment .txt files; enables text "
                                    "perturbation and prompt emission")
    ap.add_argument("--statute-text-dir", help="dir of statute .txt files")
    ap.add_argument("--emit-prompts", help="write one LLM prompt per item here")
    ap.add_argument("--emit-texts", help="write the raw perturbed judgment text "
                                         "per item here (no prompt wrapper) -- for "
                                         "the deterministic pipeline condition")
    ap.add_argument("--require-hits", type=int, default=1,
                    help="drop an item if the date could not be rewritten at "
                         "least this many times in the text (default 1)")
    ap.add_argument("--mode", choices=["single", "translate-past"],
                    default="single",
                    help="single: rewrite only the swept date's own surface "
                         "forms (legacy; leaves satellite references behind "
                         "-- see residual report). translate-past: shift the "
                         "anchor AND every date expression on or before it "
                         "by k, so the pre-anchor history moves as a block "
                         "and the document stays internally consistent by "
                         "construction. anchor surface only.")
    ap.add_argument("--max-residual", type=int, default=None,
                    help="drop an item if, after rewriting, more than this "
                         "many partial references to the ORIGINAL date "
                         "(month-year / day-month) survive in the text. "
                         "Such half-perturbed documents carry contradictory "
                         "anchor evidence -- the same 'ambiguous ground "
                         "truth' --require-hits exists to prevent. Default: "
                         "count and report only (backwards compatible); use "
                         "0 for the strict drop the module docstring "
                         "promises.")
    ap.add_argument("--self-check", action="store_true",
                    help="assert the engine is monotone; exit 1 if not")
    ap.add_argument("--out")
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text())
    default_m1 = bool(gold.get("minus_one_day", True))
    tdg_paths = dict(kv.split("=", 1) for kv in args.statute_tdg)
    texts = {}
    if args.texts:
        texts = {p.stem: p.read_text(errors="ignore")
                 for p in Path(args.texts).glob("*.txt")}

    rule_cache: dict = {}
    out_cases, all_items, bad, bad_residual = {}, 0, [], []

    for cid, c in gold["cases"].items():
        statute = c.get("statute", gold.get("statute", "unknown"))
        if statute not in rule_cache:
            rule_cache[statute] = _rule_for(statute, tdg_paths, default_m1)
        offset, source, desc = rule_cache[statute]
        if offset is None:
            continue

        center = None
        if args.auto_center:
            center = find_boundary(offset, c, args.surface)
            if center is None:
                print(f"  ! {cid}: no verdict boundary within +/-1200d on "
                      f"{args.surface} -- skipped (nothing to straddle)")
                continue
        items = sweep(offset, c, args.surface, args.range, center=center)
        if not items:
            continue
        analysis = flip_analysis(items, args.surface)

        # perturb the document, if we have it
        stem = CASE_FILE.get(cid)
        kept = items
        if texts and stem in texts:
            base = texts[stem]
            orig = _d(c.get({"anchor": "anchor_date",
                             "presented": "presented_date",
                             "day_a": "acas_day_a"}[args.surface]))
            kept = []
            for it in items:
                new = _d(it[{"anchor": "anchor",
                             "presented": "presented",
                             "day_a": "acas_day_a"}[args.surface]])
                if args.mode == "translate-past":
                    if args.surface != "anchor":
                        sys.exit("--mode translate-past supports "
                                 "--surface anchor only")
                    frozen = [d for d in (_d(c.get("acas_day_a")),
                                          _d(c.get("presented_date")))
                              if d is not None]
                    txt, hits, yflags = translate_past(
                        base, orig, it["k"],
                        cutoff=min(frozen) if frozen else None)
                    it["bare_year_flags"] = yflags
                else:
                    txt, hits = perturb_text(base, orig, new)
                if hits < args.require_hits:
                    bad.append((cid, it["k"], hits))
                    continue
                resid = residual_refs(txt, orig, new)
                if args.max_residual is not None and resid > args.max_residual:
                    bad_residual.append((cid, it["k"], resid))
                    continue
                it["text_hits"] = hits
                it["residual_refs"] = resid
                it["_text"] = txt
                kept.append(it)

        out_cases[cid] = {
            "statute": statute, "rule_source": source, "rule_desc": desc,
            "surface": args.surface, "center_k": center,
            "true_anchor": c.get("anchor_date"),
            "true_presented": c.get("presented_date"),
            "real_verdict": c.get("verdict"),
            "analysis": analysis,
            "items": [{k: v for k, v in it.items() if k != "_text"}
                      for it in kept],
        }
        all_items += len(kept)

        if args.emit_texts and texts and stem in texts:
            d = Path(args.emit_texts); d.mkdir(parents=True, exist_ok=True)
            for it in kept:
                (d / f"{cid}__k{it['k']:+04d}.txt").write_text(
                    it["_text"], encoding="utf-8")

        if args.emit_prompts and texts and stem in texts:
            d = Path(args.emit_prompts); d.mkdir(parents=True, exist_ok=True)
            st = ""
            if args.statute_text_dir:
                sp = Path(args.statute_text_dir) / f"{statute}.txt"
                if sp.exists():
                    st = sp.read_text(errors="ignore")
            for it in kept:
                (d / f"{cid}__k{it['k']:+04d}.txt").write_text(
                    PROMPT.format(statute=st or f"[{desc}]",
                                  judgment=it["_text"]), encoding="utf-8")

    # -- report ------------------------------------------------------------
    print(f"surface={args.surface}  range=+/-{args.range}d  "
          f"cases={len(out_cases)}  items={all_items}")
    print(f"\n{'case':26s} {'items':>5s} {'flips':>5s} {'mono':>5s} "
          f"{'flip@k':>7s} {'boundary':>11s}  TIMELY/LATE")
    non_mono = []
    for cid, o in out_cases.items():
        a = o["analysis"]
        if not a.get("monotone"):
            non_mono.append(cid)
        print(f"  {cid[:24]:24s} {a['n_items']:5d} {a['n_flips']:5d} "
              f"{str(a['monotone']):>5s} {str(a['flip_at_k']):>7s} "
              f"{str(a['boundary_date']):>11s}  "
              f"{a['verdicts']['TIMELY']}/{a['verdicts']['LATE']}")
    if bad:
        print(f"\ndropped {len(bad)} items: date not rewritable in the text "
              f"(oblique reference) -- item would have had an ambiguous truth")
    if bad_residual:
        print(f"dropped {len(bad_residual)} items: partial references to the "
              f"ORIGINAL date survived the rewrite (--max-residual)")
    # per-case residual contamination, always shown when texts were perturbed
    poisoned = {cid: [it.get("residual_refs", 0) for it in o["items"]]
                for cid, o in out_cases.items()
                if any(it.get("residual_refs") for it in o["items"])}
    if poisoned:
        print("\n!! RESIDUAL ORIGINAL-DATE REFERENCES survive in the "
              "perturbed texts of these cases (month-year / day-month "
              "mentions that perturb_text does not rewrite). These items "
              "contain contradictory anchor evidence; an extraction "
              "pipeline with a conflict gate will abstain on them, and an "
              "LLM may silently anchor on the unshifted mention:")
        for cid, rs in poisoned.items():
            print(f"   {cid}: {sum(1 for r in rs if r)}/{len(rs)} items, "
                  f"max {max(rs)} refs/item")
        if args.max_residual is None:
            print("   (rerun with --max-residual 0 to drop them)")

    if args.self_check and non_mono:
        print(f"\nSELF-CHECK FAILED: not monotone in {args.surface}: {non_mono}")
        print("Shifting the clock start later must never make a claim MORE late.")
        return 1
    if args.self_check:
        print(f"\nSELF-CHECK PASS: every sweep is monotone with exactly "
              f"<=1 verdict flip. The engine has one boundary per case, "
              f"where the statute says it should.")

    if args.out:
        p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {"surface": args.surface, "range": args.range,
             "n_items": all_items, "generated_from": args.gold,
             "oracle": "engine (run_gold_facts rule discovery + entailment); "
                       "engine is the GENERATOR here and must not be scored "
                       "against these labels -- see module docstring",
             "cases": out_cases}, indent=1), encoding="utf-8")
        print(f"\nwrote {p}  ({all_items} items with computed ground truth)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    
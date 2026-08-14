#!/usr/bin/env python3
"""make_baseline_inputs.py -- build redacted case texts for the LLM baseline.

Design (frozen BEFORE any model call -- see notes 06/07/2026):
  REMOVE  the tribunal's/EAT's timeliness CONCLUSIONS: computed deadlines,
          boundaries, verdicts, anchor-selection findings, deadline
          arithmetic, and appeal framing that reveals the outcome.
  KEEP    every primary fact: event dates, EC Day A/B, presentation dates,
          contentions of the parties (the dispute is part of the task).

All operations are anchor-resolved against the committed txt files:
an op names a unique substring; the builder locates it, cuts whole LINES
(these files are one paragraph per line) or rewrites a line keeping only
byte-exact fact clauses. Every op must resolve exactly once or the build
FAILS. No regex extraction of values anywhere.

After redaction, a leak check runs per case:
  forbidden: the gold stated/computed deadline & boundary values in ISO
             and natural form must be ABSENT,
  required:  the fact dates the model needs must be PRESENT.
Build fails on any violation. A human-readable redaction report is written
next to the outputs -- READ IT before running any model (same protocol as
the gold: script drafts, human verifies).

DECIDED 2026-07-06 (frozen): 2025_EAT_155 L140 contention re s.207B kept
(party argument; the tribunal saw it too); 2026_EAT_76 incident-date fragment
kept (date appears nowhere else in the judgment; the anchor ruling around
it is cut). Both documented here and in the redaction report.

Usage (repo root):
    python src/baseline/make_baseline_inputs.py
Outputs:
    data/experiments/baseline/inputs/<case>.txt
    data/experiments/baseline/redaction_report.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASELAW = ROOT / "data" / "caselaw"
OUTDIR = ROOT / "data" / "experiments" / "baseline" / "inputs"
REPORT = ROOT / "data" / "experiments" / "baseline" / "redaction_report.txt"

RED = "[REDACTED: tribunal timeliness analysis]"

# -- op helpers ----------------------------------------------------------
def CUT(anchor, occ=1):            # cut the whole line containing anchor
    return ("cut", anchor, occ)

def SPAN(start, end, occ=(1, 1)):  # cut lines from start-line to end-line incl.
    return ("span", start, end, occ)

def SPAN_TO_EOF(start, occ=1):
    return ("span_eof", start, occ)

def KEEP(anchor, keeps, occ=1):    # rewrite line: only byte-exact fact clauses
    return ("keep", anchor, keeps, occ)

def KEEP_BEFORE(anchor, split, occ=1):  # keep line text before split marker
    return ("keep_before", anchor, split, occ)


# -- THE SPEC (frozen 2026-07-06) ----------------------------------------
SPEC = {
  # --------- 2026_EAT_64 (serves both gold rows) ---------
  "2026_EAT_64": {
    "file": "gold_eqa_s123/txt/2026_EAT_64.txt",
    "gold_rows": ["2026_EAT_64_s111", "2026_EAT_64_s123"],
    "ops": [
      CUT("Following a preliminary hearing, the Tribunal concluded that (a)"),
      SPAN("Held:", "The appeal was, therefore, dismissed."),
      KEEP("primary limitation period for presenting complaints",
           ["The appellant entered early conciliation on 29 November 2022",
            "An ACAS certificate was issued on 10 January 2023, and an ET1 "
            "claim form was presented on that date."]),
      CUT("Having given correct self-directions on the relevant principles "
          "of law applicable to time limits"),
      CUT("the Tribunal dismissed the complaints of unfair dismissal on the "
          "basis that they were brought out of time"),
      KEEP("the complaints of disability discrimination were also over 4 "
           "months out of time",
           ["Since the final act of alleged discrimination complained of "
            "was the dismissal on 8 June 2022"]),
      KEEP_BEFORE("identified three reasons for her complaints being brought "
                  "out of time",
                  "The appellant identified three reasons"),
      SPAN("The Tribunal noted that the appellant had not put forward any "
           "cogent reason",
           "dismissed the discrimination complaints."),
      SPAN_TO_EOF("Analysis and decision"),
    ],
    "forbidden": ["7 September 2022", "2022-09-07"],
    "required": ["8 June 2022", "29 November 2022", "10 January 2023",
                 "10 and 12 January 2023"],
  },
  # --------- 2025_EAT_155 ---------
  "2025_EAT_155": {
    "file": "gold_era_s111/txt/2025_EAT_155.txt",
    "gold_rows": ["2025_EAT_155"],
    "ops": [
      KEEP("The Tribunal found that the dismissal le",
           ["A preliminary hearing took place before the Tribunal at which "
            "the issues were (a) when the dismissal letter was first seen "
            "by the appellant; and (b) time-bar."]),
      SPAN("Held:", "Introduction", occ=(1, 1)),  # end line NOT cut (header kept)
      CUT("On the basis of the deemed date of service"),
      KEEP("Therefore, he presented his claims out of time",
           ["The claimant notified ACAS of his intention to conciliate on "
            "12 October 2020."]),
      SPAN("On the balance of probabilities, we find that the Claimant saw "
           "the email",
           "We find that they read it together"),
      CUT("the respondent had conceded that the claims were presented in "
          "time, that had been on the basis of incomplete"),
      CUT("Accordingly, we find that the claimant knew that he had been "
          "summarily dismissed on 11 July 2020."),
      CUT("If we are wrong about that, we are satisfied that he had a "
          "reasonable opportunity"),
      SPAN_TO_EOF("Analysis and decision"),
    ],
    "forbidden": ["10 October 2020", "2020-10-10", "11 October 2020"],
    "required": ["12 October 2020", "10 December 2020", "10 July 2020"],
  },
  # --------- 2026_EAT_14 ---------
  "2026_EAT_14": {
    "file": "gold_era_s111/txt/2026_EAT_14.txt",
    "gold_rows": ["2026_EAT_14"],
    "ops": [
      KEEP("Employment Judge Fowell dismissed the claim submitted",
           ["This is an appeal against the judgment of Employment Judge E "
            "Fowell after a hearing on 27 July 2023.",
            "the Employment Tribunal on 6 February 2023"]),
      CUT("The Employment Tribunal concluded that the dismissal had taken "
          "place on 29 September 2022."),
      CUT("The Tribunal erred by considering in isolation an extension of "
          "time for the unfair dismissal claim"),
      SPAN_TO_EOF("Analysis"),
    ],
    "forbidden": ["28 December 2022", "2022-12-28"],
    "required": ["29 September 2022", "6 February 2023"],
  },
  # --------- 2026_EAT_76 ---------
  "2026_EAT_76": {
    "file": "gold_eqa_s123/txt/2026_EAT_76.txt",
    "gold_rows": ["2026_EAT_76"],
    "ops": [
      KEEP("was found to have been submitted out",
           ["The judgment was sent to the parties on 3 September 2024."]),
      CUT("The Employment Tribunal held that the race related harassment "
          "complaint in respect of Ms Foulston was out of time:"),
      # incident date appears NOWHERE else in the judgment (facts section says
      # only "In 2018") -- keep the bare date fragment, cut the anchor ruling
      # ("Time began to run from...") and the continuing-act rejection.
      KEEP("Time began to run from the date of the incident (12 April 2018)",
           ["the date of the incident (12 April 2018)"]),
      CUT("We do not grant the Claimant an extension of time"),
      CUT("extension of time being the exception rather than the rule"),
      CUT("we would also have held it was out of time"),
      CUT("The Employment Tribunal also stated that had it found that the "
          "fist"),
      CUT("do not assert arguable errors of law"),
    ],
    "forbidden": ["11 July 2018", "2018-07-11"],
    "required": ["12 April 2018", "28 February 2020"],
  },
  # --------- 2026_EAT_59 ---------
  "2026_EAT_59": {
    "file": "gold_era_s111/txt/2026_EAT_59.txt",
    "gold_rows": ["2026_EAT_59"],
    "ops": [
      KEEP("presented within the statutory time limit as extended",
           ["This decision may have other consequences.",
            "Based on the pleaded effective date of termination "
            "(18 February 2024)"]),
    ],
    "forbidden": ["17 May 2024", "2024-05-17", "20 July 2024", "2024-07-20"],
    "required": ["18 February 2024", "9 July 2024"],
  },
  # --------- KJ ---------
  "2026_EAT_46": {
    "file": "gold_eqa_s123/txt/2026_EAT_46.txt",
    "gold_rows": ["kj_2026_EAT_46"],
    "ops": [
      CUT("EAT dismissed the respondent"),  # same line also holds the
      # "alternative finding ... was flawed" sentence -- one paragraph/line
      KEEP_BEFORE("was brought in time and that, in any event, it would have "
                  "been just and equitable",
                  "The Respondent contends that the ET erred"),
      KEEP_BEFORE("was the claim of discrimination brought in time",
                  ": was the claim of discrimination brought in time"),
      KEEP("any act complained of before 15 September 2021",
           ["As the claimant started early conciliation with ACAS on "
            "6 November 2021 (Day A), obtained an early conciliation "
            "certificate on 10 January 2022 (Day B), and presented her "
            "claim on 17 February 2022"]),
      CUT("In practical terms, that means that any complaint about the SUC "
          "report itself is in time"),
      CUT("253.All the claims of harassment related to the protected "
          "characteristic of sex are in time"),
      SPAN("Looking at our findings of fact in this case",
           "relating to the same facts both succeed.", occ=(1, 1)),
      CUT("The Respondent cross-appeals with respect to the decision of the "
          "ET to assume jurisdiction"),
      CUT("The ET erred in finding that the claim was brought in time; and"),
      CUT("The ET erred in deciding that it would, in any event, extend "
          "time."),
      CUT("Ms Stone KC submitted that the ET had erred in finding that they "
          "had jurisdiction"),
      CUT("Ms Stone KC also submitted that the ET erred in finding that "
          "they would have extended time"),
      CUT("Mr Milsom resisted the cross-appeal on behalf of the Claimant."),
      CUT("Mr Milsom contended that this was not a mis"),
      CUT("As for the alternative finding that time would be extended"),
      SPAN("The ET decided that they had jurisdiction to consider the",
           "the cross-appeal that the ET had no jurisdiction to entertain "
           "the complaint about Mr Reilly"),
    ],
    "forbidden": ["15 September 2021", "2021-09-15",
                  "21 February 2022", "27 April 2022"],
    "required": ["6 November 2021", "10 January 2022", "17 February 2022",
                 "22 November 2021"],
  },
}


# -- engine --------------------------------------------------------------
def _find_line(lines, anchor, occ):
    hits = [i for i, ln in enumerate(lines) if anchor in ln]
    if len(hits) < occ:
        raise SystemExit(f"ANCHOR NOT FOUND (need occurrence {occ}, "
                         f"got {len(hits)}): {anchor[:80]!r}")
    return hits[occ - 1]


def apply_spec(name, spec, report):
    src_path = CASELAW / spec["file"]
    lines = src_path.read_text(encoding="utf-8").splitlines()
    cuts = []  # (start, end, replacement, description)

    for op in spec["ops"]:
        kind = op[0]
        if kind == "cut":
            i = _find_line(lines, op[1], op[2])
            cuts.append((i, i, RED, f"CUT   L{i+1}: {lines[i][:90]}"))
        elif kind == "span":
            s = _find_line(lines, op[1], op[3][0])
            e = _find_line(lines, op[2], op[3][1])
            if op[2] == "Introduction":            # exclusive-end header case
                e -= 1
            if e < s:
                raise SystemExit(f"SPAN reversed: {op[1][:50]!r}")
            cuts.append((s, e, RED,
                         f"SPAN  L{s+1}-L{e+1}: {lines[s][:60]} ... {lines[e][:60]}"))
        elif kind == "span_eof":
            s = _find_line(lines, op[1], op[2])
            cuts.append((s, len(lines) - 1, RED,
                         f"SPAN  L{s+1}-EOF: {lines[s][:80]}"))
        elif kind == "keep_before":
            i = _find_line(lines, op[1], op[3])
            if op[2] not in lines[i]:
                raise SystemExit(f"split marker not on line L{i+1}: {op[2][:60]!r}")
            new = lines[i].split(op[2])[0].rstrip() + " " + RED
            cuts.append((i, i, new,
                         f"KEEPB L{i+1}: kept prefix before {op[2][:50]!r}"))
        elif kind == "keep":
            i = _find_line(lines, op[1], op[3])
            for k in op[2]:
                if k not in lines[i]:
                    raise SystemExit(f"KEEP clause not on line L{i+1}: {k[:70]!r}")
            new = " ".join(op[2]) + " " + RED
            cuts.append((i, i, " " + new,
                         f"KEEP  L{i+1}: kept {len(op[2])} fact clause(s), rest redacted"))

    # apply from bottom up so indices stay valid; forbid overlaps
    cuts.sort(key=lambda c: c[0])
    for a, b in zip(cuts, cuts[1:]):
        if b[0] <= a[1]:
            raise SystemExit(f"OVERLAPPING ops in {name}: {a[3]} / {b[3]}")
    for s, e, repl, _ in reversed(cuts):
        lines[s:e + 1] = [repl]

    text = "\n".join(lines) + "\n"

    # leak check
    problems = []
    for bad in spec["forbidden"]:
        if bad in text:
            problems.append(f"FORBIDDEN present after redaction: {bad!r}")
    for good in spec["required"]:
        if good not in text:
            problems.append(f"REQUIRED fact missing after redaction: {good!r}")
    if RED not in text:
        problems.append("no redactions applied?")

    report.append(f"\n======== {name}  ({spec['file']}) ========")
    report.extend("  " + c[3] for c in cuts)
    report.append(f"  leak-check: {'PASS' if not problems else 'FAIL'}")
    report.extend("    " + p for p in problems)
    return text, problems


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    report, failed = ["REDACTION REPORT -- human-verify every line below "
                      "against the source judgment BEFORE running models."], []
    for name, spec in SPEC.items():
        text, problems = apply_spec(name, spec, report)
        (OUTDIR / f"{name}.txt").write_text(text, encoding="utf-8")
        if problems:
            failed.append(name)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    if failed:
        print(f"\nBUILD FAILED leak-check: {', '.join(failed)}")
        return 1
    print(f"\nOK: {len(SPEC)} redacted inputs -> {OUTDIR}")
    print(f"Report -> {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
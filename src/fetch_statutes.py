#!/usr/bin/env python3
"""
fetch_statutes.py -- download statute SECTIONS from legislation.gov.uk as
clean CLML XML + plain text, for the rule-discovery side of the thesis (RQ4).

WHY A REGISTRY, NOT A CRAWL
---------------------------
A statute is a *rule the engine discovers once and applies to every case*, so
you don't want volume -- you want a few DISTINCT rule shapes / anchoring
conventions. Ten statutes that all say "X months from Y" prove nothing more
than one. The registry below is curated for variety:

  ERA s.111   "beginning with" the EDT    -> +3 months, minus_one_day = True
  EqA s.123   "starting with" the act     -> +3 months, minus_one_day = True,
                                              + "just and equitable" discretion
  Limitation s.5  "from" accrual          -> +6 years,  minus_one_day = False
  ERA s.207B  ACAS early-conciliation     -> the deadline-extension pause
  ERA s.86    notice by length of service -> a DIFFERENT rule form (not deadline)

That's 3 conventions across 2 areas of law (+2 optional shapes) -- the RQ4
generalisation claim, with no duplicated findings.

POINT-IN-TIME
-------------
A case must be judged against the statute AS IT STOOD at the relevant date.
Pass --as-at YYYY-MM-DD to fetch every section in its version as at that date,
or set a per-entry "version" in the registry. Default = latest revised version.

API: legislation.gov.uk -- append /data.xml to any section URI; version date is
a path segment before /data.xml. Fair-use: set a User-Agent; 3,000 req/5min.
Output is Crown copyright under the Open Government Licence -- keep attribution.

Examples
--------
  python fetch_statutes.py --out ../data/statutes                 # core 3, latest
  python fetch_statutes.py --include all --out ../data/statutes   # all 5 shapes
  python fetch_statutes.py --as-at 2020-08-02 --out ../data/statutes_2020  # point-in-time
  python fetch_statutes.py --akn --out ../data/statutes           # also save Akoma Ntoso

Stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "https://www.legislation.gov.uk"
USER_AGENT = "thesis-tdg-research/1.0 (legislation API fair-use; contact: you@example.com)"

# ---------------------------------------------------------------------------
# Curated statute registry -- distinct rule shapes, not a corpus
# ---------------------------------------------------------------------------
# key: short slug | type/year/number/section | label | convention note | core?
REGISTRY = [
    {
        "slug": "era_1996_s111", "type": "ukpga", "year": 1996, "number": 18,
        "section": "111", "core": True,
        "label": "Employment Rights Act 1996 s.111 (unfair dismissal time limit)",
        "convention": "'beginning with' the effective date of termination -> +3 months, minus_one_day=True",
    },
    {
        "slug": "eqa_2010_s123", "type": "ukpga", "year": 2010, "number": 15,
        "section": "123", "core": True,
        "label": "Equality Act 2010 s.123 (discrimination time limit)",
        "convention": "'starting with' the date of the act -> +3 months, minus_one_day=True, + 'just and equitable' discretion",
    },
    {
        "slug": "limitation_1980_s5", "type": "ukpga", "year": 1980, "number": 58,
        "section": "5", "core": True,
        "label": "Limitation Act 1980 s.5 (simple contract, 6 years)",
        "convention": "'from' the date the cause of action accrued -> +6 years, minus_one_day=False",
    },
    {
        "slug": "era_1996_s207b", "type": "ukpga", "year": 1996, "number": 18,
        "section": "207B", "core": False,
        "label": "Employment Rights Act 1996 s.207B (ACAS early conciliation extension)",
        "convention": "deadline-EXTENSION rule: clock pause between Day A and Day B + one-month floor",
    },
    {
        "slug": "era_1996_s86", "type": "ukpga", "year": 1996, "number": 18,
        "section": "86", "core": False,
        "label": "Employment Rights Act 1996 s.86 (minimum notice by length of service)",
        "convention": "DIFFERENT rule form: period scales with service length (not a fixed deadline)",
    },
]


# ---------------------------------------------------------------------------
# HTTP (UA required; 403 = rate limit; 202 = generating, retry)
# ---------------------------------------------------------------------------

def http_get(url: str, pace: float = 0.5, max_retries: int = 5) -> bytes:
    delay = 5.0
    for _ in range(max_retries):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status == 202:  # dynamically generating; wait and retry
                    sys.stderr.write("    202 generating; waiting 10s\n")
                    time.sleep(10)
                    continue
                time.sleep(pace)
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 403:  # rate limited or blocked
                sys.stderr.write(f"    403 (rate limit); backing off {delay:.0f}s\n")
                time.sleep(delay); delay *= 2; continue
            if e.code == 300:  # multiple choices -- ambiguous identifier
                raise RuntimeError(f"300 Multiple Choices for {url} (ambiguous id)")
            if e.code == 404:
                raise RuntimeError(f"404 Not Found: {url}")
            sys.stderr.write(f"    HTTP {e.code}; retry in {delay:.0f}s\n")
            time.sleep(delay); delay *= 2
        except (urllib.error.URLError, TimeoutError) as e:
            sys.stderr.write(f"    network error ({e}); retry\n")
            time.sleep(delay); delay *= 2
    raise RuntimeError(f"giving up on {url}")


# ---------------------------------------------------------------------------
# URL building + text extraction
# ---------------------------------------------------------------------------

def section_url(entry: dict, ext: str, as_at: str | None) -> str:
    p = [BASE, entry["type"], str(entry["year"]), str(entry["number"]),
         "section", entry["section"]]
    version = as_at or entry.get("version")
    if version:
        p.append(version)
    return "/".join(p) + f"/{ext}"


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_local(root, names: set[str]):
    for el in root.iter():
        if _localname(el.tag) in names:
            return el
    return None


def clml_to_text(xml_bytes: bytes) -> str:
    """
    Extract the section body text from a CLML section fragment.

    CLML section responses are a full <Legislation> wrapper in document order:
    ukm:Metadata (incl. amendment effect refs like 's. 40(1A)'), optional
    <Contents>, then <Primary>/<Body>/<P1group> holding the actual section,
    then <Commentaries>. Two past bugs to avoid: (1) picking the FIRST
    matching element in document order can land on Contents/metadata instead
    of the body; (2) `continue` inside .iter() skips only the element itself,
    not its subtree, so metadata text leaks. Fix: explicit DFS that prunes
    excluded subtrees entirely, rooted at the deepest body-ish container.
    """
    EXCLUDE = {"Metadata", "Commentaries", "Contents", "Versions",
               "Resources", "CommentaryRef", "MarginNote"}

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ""

    def collect(el, parts):
        if _localname(el.tag) in EXCLUDE:
            return  # prune the whole subtree
        if el.text and el.text.strip():
            parts.append(el.text.strip())
        for child in el:
            collect(child, parts)
            if child.tail and child.tail.strip():
                parts.append(child.tail.strip())

    # Prefer the most specific container that exists; each candidate is
    # searched via pruned DFS so a "P1group" inside metadata is never chosen.
    target = None
    for name in ("P1group", "Body", "Primary", "Secondary"):
        stack = [root]
        while stack:
            el = stack.pop(0)
            ln = _localname(el.tag)
            if ln in EXCLUDE:
                continue
            if ln == name:
                target = el
                break
            stack.extend(list(el))
        if target is not None:
            break
    if target is None:
        target = root

    parts: list[str] = []
    collect(target, parts)
    text = " ".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def date_count(text: str) -> int:
    return len(re.findall(r"\b\d{4}\b|\b\d{1,2}\s+(?:January|February|March|April|May|"
                          r"June|July|August|September|October|November|December)", text, re.I))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch statute sections from legislation.gov.uk.")
    ap.add_argument("--include", choices=["core", "all"], default="core",
                    help="'core' = the 3 essential rule shapes; 'all' = include optional 4th/5th")
    ap.add_argument("--only", action="append", default=[],
                    help="fetch only these registry slugs (repeatable), e.g. --only eqa_2010_s123")
    ap.add_argument("--as-at", default=None, metavar="YYYY-MM-DD",
                    help="point-in-time: fetch each section as it stood on this date")
    ap.add_argument("--akn", action="store_true", help="also save Akoma Ntoso (data.akn)")
    ap.add_argument("--pace", type=float, default=0.5, help="seconds between requests")
    ap.add_argument("--out", help="output base directory (required unless --list)")
    ap.add_argument("--list", action="store_true", help="list the registry and exit")
    args = ap.parse_args()

    if args.list:
        for e in REGISTRY:
            tag = "core" if e["core"] else "opt "
            print(f"  [{tag}] {e['slug']:20s} {e['label']}")
            print(f"           {e['convention']}")
        return

    if not args.out:
        sys.exit("--out is required (or use --list)")

    # select entries
    if args.only:
        entries = [e for e in REGISTRY if e["slug"] in set(args.only)]
        missing = set(args.only) - {e["slug"] for e in entries}
        if missing:
            sys.exit(f"Unknown slug(s): {', '.join(missing)} (use --list)")
    elif args.include == "core":
        entries = [e for e in REGISTRY if e["core"]]
    else:
        entries = list(REGISTRY)

    out = Path(args.out)
    (out / "xml").mkdir(parents=True, exist_ok=True)
    (out / "txt").mkdir(parents=True, exist_ok=True)
    if args.akn:
        (out / "akn").mkdir(parents=True, exist_ok=True)

    man = open(out / "manifest.csv", "w", newline="", encoding="utf-8")
    mw = csv.writer(man)
    mw.writerow(["slug", "label", "convention", "version", "n_chars", "url", "status"])

    print(f"Fetching {len(entries)} statute section(s)"
          + (f" as at {args.as_at}" if args.as_at else " (latest)") + f" -> {out}\n")

    ok = 0
    for e in entries:
        url = section_url(e, "data.xml", args.as_at)
        print(f"  {e['slug']}: {url}")
        try:
            xml_bytes = http_get(url, pace=args.pace)
        except Exception as ex:
            print(f"    FAILED: {ex}")
            mw.writerow([e["slug"], e["label"], e["convention"], args.as_at or "latest", "", url, f"FAILED: {ex}"])
            continue

        text = clml_to_text(xml_bytes)
        (out / "xml" / f"{e['slug']}.xml").write_bytes(xml_bytes)
        (out / "txt" / f"{e['slug']}.txt").write_text(text, encoding="utf-8")
        if args.akn:
            try:
                (out / "akn" / f"{e['slug']}.akn.xml").write_bytes(
                    http_get(section_url(e, "data.akn", args.as_at), pace=args.pace))
            except Exception as ex:
                print(f"    (akn failed: {ex})")

        ok += 1
        mw.writerow([e["slug"], e["label"], e["convention"], args.as_at or "latest", len(text), url, "OK"])
        print(f"    OK  ({len(text)} chars)  -- {e['convention']}")

    man.close()
    print(f"\nDone. {ok}/{len(entries)} sections -> {out}")
    print(f"  text: {out}/txt/   xml: {out}/xml/   manifest: {out}/manifest.csv")
    if ok < len(entries):
        print("  (some failed -- check the section numbers/version date; legislation may not exist at that date)")


if __name__ == "__main__":
    main()

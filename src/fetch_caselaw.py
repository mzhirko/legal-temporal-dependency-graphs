#!/usr/bin/env python3
"""
fetch_caselaw.py -- download judgments from The National Archives "Find Case Law".

Targets the public Find Case Law API (Atom feed + per-document LegalDocML XML):
  https://nationalarchives.github.io/ds-find-caselaw-docs/public

It searches the Atom feed, downloads each matching document as XML (clean
LegalDocML -- preferred) and optionally PDF, extracts plain body text, and
DROPS stubs that lack full written reasons (length + date-density filters), so
what lands in --out is pipeline-ready and free of judgment-only one-liners.

LICENCE: programmatic bulk extraction across Find Case Law requires the
computational-analysis licence (free; you said you've applied). This script
respects the documented rate limit (1,000 requests / 5 min / IP) with polite
pacing and 429 back-off. Output is under the Open Justice Licence -- keep the
attribution that the per-document XML carries.

COVERAGE: Find Case Law holds the Employment Appeal Tribunal (court code
`eat`) and higher courts/tribunals as XML. FIRST-INSTANCE Employment Tribunal
decisions (the gov.uk/employment-tribunal-decisions register) are a SEPARATE
system and are NOT served by this API -- see the note at the bottom of the file.

Examples
--------
  # EAT time-limit cases mentioning the s.111 anchor, up to 40, with PDFs too
  python fetch_caselaw.py --query "effective date of termination" \
      --court eat --max 40 --pdf --out ../data/caselaw_eat

  # discrimination time-limit cases (for RQ4 / EqA s.123)
  python fetch_caselaw.py --query "just and equitable time limit" \
      --court eat --max 40 --out ../data/caselaw_eqa

  # broad pull, then filter hard for full reasons
  python fetch_caselaw.py --query "section 111 out of time" \
      --court eat --max 100 --min-chars 6000 --min-dates 4 --out ../data/caselaw_big

Stdlib only (urllib, xml.etree) -- no pip deps, runs on a cluster as-is.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "https://caselaw.nationalarchives.gov.uk"
ATOM = BASE + "/atom.xml"
USER_AGENT = "thesis-tdg-research/1.0 (computational-analysis licence; contact: you@example.com)"

ATOM_NS = "{http://www.w3.org/2005/Atom}"
TNA_NS = "{https://caselaw.nationalarchives.gov.uk}"

# crude but effective date-density signal: ISO dates + "12 March 2024" style
_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# HTTP with polite pacing + 429 back-off
# ---------------------------------------------------------------------------

class Http:
    def __init__(self, pace: float = 0.4, max_retries: int = 5):
        self.pace = pace
        self.max_retries = max_retries

    def get(self, url: str) -> bytes:
        delay = 2.0
        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    time.sleep(self.pace)  # stay well under 1000/5min
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    sys.stderr.write(f"    429 rate-limited; backing off {delay:.0f}s\n")
                    time.sleep(delay)
                    delay *= 2
                    continue
                if e.code == 404:
                    raise
                sys.stderr.write(f"    HTTP {e.code} on {url}; retry\n")
                time.sleep(delay)
                delay *= 2
            except (urllib.error.URLError, TimeoutError) as e:
                sys.stderr.write(f"    network error ({e}); retry\n")
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(f"giving up on {url} after {self.max_retries} tries")


# ---------------------------------------------------------------------------
# Atom feed parsing
# ---------------------------------------------------------------------------

def build_query_url(query, courts, order, page, per_page) -> str:
    params = [("order", order), ("page", str(page)), ("per_page", str(per_page))]
    if query:
        params.append(("query", query))
    for c in courts or []:
        params.append(("court", c))
    return ATOM + "?" + urllib.parse.urlencode(params)


def parse_entries(atom_bytes: bytes) -> tuple[list[dict], bool]:
    """Return (entries, has_next_page)."""
    root = ET.fromstring(atom_bytes)
    has_next = any(
        l.get("rel") == "next" for l in root.findall(f"{ATOM_NS}link")
    )
    entries = []
    for e in root.findall(f"{ATOM_NS}entry"):
        title = (e.findtext(f"{ATOM_NS}title") or "").strip()
        published = (e.findtext(f"{ATOM_NS}published") or "").strip()
        uri = (e.findtext(f"{TNA_NS}uri") or "").strip()

        xml_url = pdf_url = html_url = ""
        for l in e.findall(f"{ATOM_NS}link"):
            typ, rel, href = l.get("type", ""), l.get("rel", ""), l.get("href", "")
            if typ == "application/akn+xml":
                xml_url = href
            elif typ == "application/pdf":
                pdf_url = href
            elif rel == "alternate" and not typ:
                html_url = href

        ncn = ""
        for ident in e.findall(f"{TNA_NS}identifier"):
            if ident.get("type") == "ukncn":
                ncn = (ident.text or "").strip()
                break

        entries.append({
            "title": title, "published": published, "uri": uri,
            "ncn": ncn, "xml_url": xml_url, "pdf_url": pdf_url, "html_url": html_url,
        })
    return entries, has_next


# ---------------------------------------------------------------------------
# Document text extraction (namespace-agnostic over Akoma Ntoso)
# ---------------------------------------------------------------------------

def xml_to_text(xml_bytes: bytes) -> str:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ""
    # itertext() walks all descendants regardless of namespace
    text = " ".join(t for t in root.itertext())
    # collapse whitespace but keep sentence breaks readable
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def date_count(text: str) -> int:
    return len(_DATE_RE.findall(text))


def slug_from(entry: dict) -> str:
    """Filesystem-safe id: prefer neutral citation, else the document URI."""
    base = entry["ncn"] or entry["uri"] or entry["title"]
    base = base.replace("[", "").replace("]", "")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_") or "doc"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Predefined thesis data splits (run all with --split-plan)
# ---------------------------------------------------------------------------
# Each split: subfolder, query, court, target count, and filter thresholds.
# Tuned so the gold splits skew toward substantial, date-dense judgments.
SPLIT_PLAN = [
    {
        "name": "gold_era_s111",
        "query": "effective date of termination time limit",
        "court": ["eat"], "max": 7, "min_chars": 6000, "min_dates": 4,
        "note": "ERA s.111 unfair-dismissal time-limit cases (READ + ANNOTATE)",
    },
    {
        "name": "gold_eqa_s123",
        "query": "just and equitable time limit discrimination",
        "court": ["eat"], "max": 4, "min_chars": 6000, "min_dates": 4,
        "note": "EqA s.123 discrimination time-limit cases (READ + ANNOTATE; RQ4 2nd area)",
    },
    {
        "name": "pool_eat_timelimit",
        "query": "out of time presented complaint",
        "court": ["eat"], "max": 33, "min_chars": 5000, "min_dates": 3,
        "note": "Extraction eval pool (auto-metrics only, not hand-annotated)",
    },
]


def run_split_plan(args, http) -> None:
    """Fetch every predefined split into its own subfolder under --out."""
    base = Path(args.out)
    print("Running thesis split plan -> ", base, "\n")
    grand = 0
    for sp in SPLIT_PLAN:
        print(f"\n=== SPLIT: {sp['name']} (target {sp['max']}) -- {sp['note']} ===")
        sub = argparse.Namespace(**vars(args))
        sub.out = str(base / sp["name"])
        sub.query = sp["query"]
        sub.court = sp["court"]
        sub.max = sp["max"]
        sub.min_chars = sp["min_chars"]
        sub.min_dates = sp["min_dates"]
        kept = fetch_one(sub, http)
        grand += kept
    print(f"\n\nSplit plan complete. {grand} documents across {len(SPLIT_PLAN)} splits in {base}")
    print("  gold_era_s111 / gold_eqa_s123  -> read these, then run make_annotation_sheets.py")
    print("  pool_eat_timelimit             -> extraction metrics only, no annotation")
    print("  (statutes: extract the 3 you already hold; contracts: 45 already done)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch judgments from Find Case Law.")
    ap.add_argument("--split-plan", action="store_true",
                    help="ignore --query/--court/--max and fetch the predefined thesis "
                         "data splits (gold_era_s111, gold_eqa_s123, pool_eat_timelimit) "
                         "into subfolders of --out")
    ap.add_argument("--query", default="", help="full-text query (all words must appear)")
    ap.add_argument("--court", action="append", default=[],
                    help="court/tribunal code, repeatable (e.g. --court eat)")
    ap.add_argument("--max", type=int, default=40, help="max documents to KEEP")
    ap.add_argument("--order", default="-date",
                    choices=["date", "-date", "updated", "-updated",
                             "transformation", "-transformation"])
    ap.add_argument("--per-page", type=int, default=50)
    ap.add_argument("--min-chars", type=int, default=5000,
                    help="drop documents whose body text is shorter (stub filter)")
    ap.add_argument("--min-dates", type=int, default=3,
                    help="drop documents with fewer date mentions (date-density filter)")
    ap.add_argument("--from-year", type=int, default=None, help="keep only published >= this year")
    ap.add_argument("--to-year", type=int, default=None, help="keep only published <= this year")
    ap.add_argument("--pdf", action="store_true", help="also download the PDF")
    ap.add_argument("--keep-xml", action="store_true", default=True,
                    help="save the raw XML (default on)")
    ap.add_argument("--pace", type=float, default=0.4, help="seconds between requests")
    ap.add_argument("--out", required=True, help="output base directory")
    args = ap.parse_args()
    http = Http(pace=args.pace)

    if args.split_plan:
        run_split_plan(args, http)
        return
    fetch_one(args, http)


def fetch_one(args, http) -> int:
    """Run one query/filter configuration; returns number of documents kept."""
    out = Path(args.out)
    (out / "xml").mkdir(parents=True, exist_ok=True)
    (out / "txt").mkdir(parents=True, exist_ok=True)
    if args.pdf:
        (out / "pdf").mkdir(parents=True, exist_ok=True)

    manifest_path = out / "manifest.csv"
    manifest = open(manifest_path, "w", newline="", encoding="utf-8")
    mw = csv.writer(manifest)
    mw.writerow(["slug", "neutral_citation", "published", "n_chars", "n_dates",
                 "kept", "reason", "uri", "html_url"])

    kept = 0
    seen = set()
    page = 1
    print(f"Query: {args.query!r}  courts={args.court or 'all'}  target={args.max}\n")

    while kept < args.max:
        url = build_query_url(args.query, args.court, args.order, page, args.per_page)
        print(f"[page {page}] {url}")
        try:
            entries, has_next = parse_entries(http.get(url))
        except Exception as e:
            print(f"  feed error: {e}")
            break
        if not entries:
            print("  no more entries.")
            break

        for entry in entries:
            if kept >= args.max:
                break
            slug = slug_from(entry)
            if slug in seen:
                continue
            seen.add(slug)

            # year filter (client-side; Atom params don't expose a date range)
            yr = None
            if entry["published"][:4].isdigit():
                yr = int(entry["published"][:4])
            if args.from_year and (yr is None or yr < args.from_year):
                mw.writerow([slug, entry["ncn"], entry["published"], "", "", "no", "before from-year", entry["uri"], entry["html_url"]])
                continue
            if args.to_year and (yr is None or yr > args.to_year):
                mw.writerow([slug, entry["ncn"], entry["published"], "", "", "no", "after to-year", entry["uri"], entry["html_url"]])
                continue

            if not entry["xml_url"]:
                mw.writerow([slug, entry["ncn"], entry["published"], "", "", "no", "no XML (PDF-only doc)", entry["uri"], entry["html_url"]])
                print(f"  - {slug}: no XML, skipped")
                continue

            # download XML, extract text, apply quality filters
            try:
                xml_bytes = http.get(entry["xml_url"])
            except Exception as e:
                mw.writerow([slug, entry["ncn"], entry["published"], "", "", "no", f"xml fetch failed: {e}", entry["uri"], entry["html_url"]])
                continue

            text = xml_to_text(xml_bytes)
            n_chars, n_dates = len(text), date_count(text)

            if n_chars < args.min_chars:
                mw.writerow([slug, entry["ncn"], entry["published"], n_chars, n_dates, "no", "too short (stub)", entry["uri"], entry["html_url"]])
                print(f"  - {slug}: {n_chars} chars -- stub, dropped")
                continue
            if n_dates < args.min_dates:
                mw.writerow([slug, entry["ncn"], entry["published"], n_chars, n_dates, "no", "too few dates", entry["uri"], entry["html_url"]])
                print(f"  - {slug}: {n_dates} dates -- low temporal content, dropped")
                continue

            # keep it
            if args.keep_xml:
                (out / "xml" / f"{slug}.xml").write_bytes(xml_bytes)
            (out / "txt" / f"{slug}.txt").write_text(text, encoding="utf-8")
            if args.pdf and entry["pdf_url"]:
                try:
                    (out / "pdf" / f"{slug}.pdf").write_bytes(http.get(entry["pdf_url"]))
                except Exception as e:
                    print(f"    (pdf failed: {e})")

            kept += 1
            mw.writerow([slug, entry["ncn"], entry["published"], n_chars, n_dates, "yes", "", entry["uri"], entry["html_url"]])
            print(f"  + {slug}  ({n_chars} chars, {n_dates} dates)  [{kept}/{args.max}]")

        if not has_next:
            print("  reached last page.")
            break
        page += 1

    manifest.close()
    print(f"\nDone. kept {kept} document(s) -> {out}")
    print(f"  text:     {out}/txt/")
    print(f"  raw XML:  {out}/xml/")
    if args.pdf:
        print(f"  PDFs:     {out}/pdf/")
    print(f"  manifest: {manifest_path}  (lists every candidate + why it was kept/dropped)")
    return kept


if __name__ == "__main__":
    main()

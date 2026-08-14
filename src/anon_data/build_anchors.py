#!/usr/bin/env python3
"""
build_anchors.py — project FACTS.md into a distributable anchor file.

Reads (private, never modified):
    data/ground_truth/FACTS.md
    data/caselaw/<split>/txt/<slug>.txt
    data/caselaw/<split>/manifest.csv

Writes (distributable — contains no judgment text):
    anchors.json

Each anchor locates a span by content hash, with character offsets as a fast
path. A verifier that has fetched the judgment can confirm the span is
byte-identical to what was annotated, without ever having received it.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import re
import sys
from pathlib import Path

CONTEXT = 64  # chars of context hashed either side, for drift diagnosis


# --- the pinned normalizer -------------------------------------------------
# Byte-for-byte identical to fetch_caselaw.xml_to_text. Duplicated here so the
# published artefact can pin it by source hash; import it instead if you'd
# rather have a single definition.

def xml_to_text(xml_bytes: bytes) -> str:
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ""
    text = " ".join(t for t in root.itertext())
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def normalizer_id() -> str:
    src = inspect.getsource(xml_to_text).encode("utf-8")
    return "xml_to_text@sha256:" + hashlib.sha256(src).hexdigest()[:16]


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# --- FACTS.md parsing ------------------------------------------------------

RE_CASE = re.compile(r"^### case:\s*(\S+)")
RE_FILE = re.compile(r"^file:\s*(\S+)")
RE_QUOTE = re.compile(r'^\s+(quote|supporting)\s*\(([^)]*?)L(\d+)\):\s*"(.*)"\s*$')
SKIP_FIELDS = {"file", "statute", "note", "caveats"}


def field_label(raw: str) -> str | None:
    """Full label up to the first ':' or '(' — keeps 'row A' / 'row B' distinct."""
    head = re.split(r"[:(]", raw, maxsplit=1)[0].strip()
    if not head or not head[0].islower():
        return None
    return re.sub(r"\s+", "_", head).lower()


def parse_facts(path: Path):
    """Yield (case_id, field, kind, line_no, quote_text)."""
    case_id = field = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = RE_CASE.match(raw)
        if m:
            case_id, field = m.group(1), None
            continue
        if RE_FILE.match(raw):
            continue
        if raw[:1] not in (" ", "", "[", "#"):
            lbl = field_label(raw)
            if lbl and lbl not in SKIP_FIELDS:
                field = lbl
            continue
        m = RE_QUOTE.match(raw)
        if m and case_id:
            kind, _label, line_no, text = m.groups()
            yield case_id, field or "unlabelled", kind, int(line_no), text


RE_SURNAME_KEY = re.compile(r"^[a-z][a-z\-']{2,}_(?=\d{4}_EAT_\d+)")


def public_case_id(case_id: str) -> str:
    """Drop the surname prefix: 2026_EAT_64_s111 -> 2026_EAT_64_s111."""
    return RE_SURNAME_KEY.sub("", case_id)


def case_file_map(path: Path) -> dict[str, str]:
    """case_id -> txt relpath (tolerates trailing '<-- NOTE' comments)."""
    out, case_id = {}, None
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = RE_CASE.match(raw)
        if m:
            case_id = m.group(1)
        m = RE_FILE.match(raw)
        if m and case_id:
            out[case_id] = m.group(1)
    return out


def load_manifests(root: Path) -> dict[str, dict]:
    """slug -> {uri, html_url, neutral_citation, published}."""
    out = {}
    for mf in sorted(root.glob("*/manifest.csv")):
        for row in csv.DictReader(mf.open(encoding="utf-8-sig")):
            out[row["slug"]] = {
                "uri": row["uri"],
                "html_url": row["html_url"],
                "neutral_citation": row["neutral_citation"],
                "published": row["published"],
            }
    return out


# --- anchoring -------------------------------------------------------------

def line_start_offsets(text: str) -> list[int]:
    offs, pos = [0], 0
    for ln in text.split("\n")[:-1]:
        pos += len(ln) + 1
        offs.append(pos)
    return offs


def anchor_one(text: str, quote: str, hinted_line: int):
    """Locate quote; prefer the occurrence on the hinted line. Returns dict or None."""
    hits = []
    i = text.find(quote)
    while i != -1:
        hits.append(i)
        i = text.find(quote, i + 1)
    if not hits:
        return None

    starts = line_start_offsets(text)
    hint_lo = starts[hinted_line - 1] if 0 < hinted_line <= len(starts) else None
    hint_hi = (starts[hinted_line] if hint_lo is not None and hinted_line < len(starts)
               else len(text))

    chosen = next((h for h in hits if hint_lo is not None and hint_lo <= h < hint_hi),
                  hits[0])
    on_hinted_line = hint_lo is not None and hint_lo <= chosen < hint_hi
    end = chosen + len(quote)

    return {
        "start": chosen,
        "end": end,
        "length": len(quote),
        "occurrence": hits.index(chosen),
        "n_occurrences": len(hits),
        "line": text.count("\n", 0, chosen) + 1,
        "line_hint_matched": on_hinted_line,
        "span_sha256": sha(quote),
        "prefix_sha256": sha(text[max(0, chosen - CONTEXT):chosen]),
        "suffix_sha256": sha(text[end:end + CONTEXT]),
    }


def main(code_root: Path, out_path: Path) -> int:
    facts = code_root / "data/ground_truth/FACTS.md"
    caselaw = code_root / "data/caselaw"

    files = case_file_map(facts)
    manifests = load_manifests(caselaw)
    cache: dict[str, str] = {}

    anchors, failures = [], []
    for case_id, field, kind, line_no, quote in parse_facts(facts):
        rel = files.get(case_id)
        if not rel:
            failures.append((case_id, field, "no file: line for case"))
            continue
        if rel not in cache:
            cache[rel] = (caselaw / rel).read_text(encoding="utf-8")
        text = cache[rel]
        slug = Path(rel).stem

        a = anchor_one(text, quote, line_no)
        if a is None:
            failures.append((case_id, field, f"quote not found in {rel}"))
            continue

        meta = manifests.get(slug, {})
        anchors.append({
            "case_id": public_case_id(case_id),
            "field": field,
            "kind": kind,
            "doc": slug,
            "neutral_citation": meta.get("neutral_citation"),
            "uri": meta.get("uri"),
            "html_url": meta.get("html_url"),
            "published": meta.get("published"),
            "doc_sha256": sha(text),
            **a,
        })

    artefact = {
        "schema": "fcl-span-anchors/1.0",
        "normalizer": normalizer_id(),
        "context_chars": CONTEXT,
        "contains_licensed_text": False,
        "contains_personal_data": False,
        "note": ("Character offsets index the normalized text produced by the "
                 "pinned normalizer from the current Find Case Law XML. No "
                 "judgment text is reproduced here; spans are identified by "
                 "SHA-256 and are relocatable by hash if offsets drift."),
        "n_anchors": len(anchors),
        "anchors": anchors,
    }
    out_path.write_text(json.dumps(artefact, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"anchored {len(anchors)}  failed {len(failures)}  -> {out_path}")
    known = {"anchor", "deadline", "acas_day_a", "acas_day_b", "presented",
             "verdict", "eat_outcome", "edt", "boundary", "row_a", "row_b"}
    odd = sorted({a["field"] for a in anchors} - known)
    if odd:
        print("  unrecognised field label(s) — normalise in FACTS.md or add to ALIASES:")
        for f in odd:
            print(f"    {f!r}")

    multi = [a for a in anchors if a["n_occurrences"] > 1]
    offline = [a for a in anchors if not a["line_hint_matched"]]
    if multi:
        print(f"  {len(multi)} span(s) non-unique in document (occurrence index recorded)")
    if offline:
        print(f"  {len(offline)} span(s) not on the hinted FACTS.md line:")
        for a in offline:
            print(f"    {a['case_id']}/{a['field']}: found L{a['line']}")
    for f in failures:
        print(f"  FAIL {f[0]} / {f[1]}: {f[2]}")
    return 1 if failures else 0


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "anchors.json")
    sys.exit(main(root, out))

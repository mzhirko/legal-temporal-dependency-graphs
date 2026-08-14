#!/usr/bin/env python3
"""
verify_anchors.py — confirm every anchored span against Find Case Law.

Usage:
    python verify_anchors.py anchors.json --fetch          # live, needs a licence
    python verify_anchors.py anchors.json --txt-dir DIR    # against local copies

Three-tier resolution per span:
  1. FAST      offsets hit and span hash matches            -> OK
  2. RELOCATED offsets missed, hash found elsewhere          -> OK (drift noted)
  3. LOST      hash absent from the document                 -> FAIL (revised text)

Tier 2 is why offsets are not load-bearing: the span is found by hashing every
window of the recorded length. A judgment is ~50k chars, so this is instant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def relocate(text: str, length: int, target: str) -> list[int]:
    """Every offset whose window of `length` chars hashes to `target`."""
    return [i for i in range(0, len(text) - length + 1)
            if sha(text[i:i + length]) == target]


def check(text: str, a: dict) -> tuple[str, str]:
    if a["end"] <= len(text) and sha(text[a["start"]:a["end"]]) == a["span_sha256"]:
        return "FAST", ""
    hits = relocate(text, a["length"], a["span_sha256"])
    if not hits:
        return "LOST", "span absent — judgment text has been revised"
    if a["occurrence"] < len(hits):
        new = hits[a["occurrence"]]
    else:
        new = hits[0]
    return "RELOCATED", f"{a['start']} -> {new} (delta {new - a['start']:+d})"


def load_texts(spec: dict, args) -> dict[str, str]:
    """doc slug -> normalized text."""
    texts = {}
    docs = {(a["doc"], a["uri"]) for a in spec["anchors"]}
    if args.txt_dir:
        for slug, _ in docs:
            hits = list(Path(args.txt_dir).rglob(f"{slug}.txt"))
            if hits:
                texts[slug] = hits[0].read_text(encoding="utf-8")
    else:
        import urllib.request
        sys.path.insert(0, str(Path(__file__).parent))
        from build_anchors import xml_to_text, normalizer_id
        if normalizer_id() != spec["normalizer"]:
            print(f"WARNING: normalizer mismatch\n  artefact: {spec['normalizer']}"
                  f"\n  local:    {normalizer_id()}", file=sys.stderr)
        base = "https://caselaw.nationalarchives.gov.uk"
        for slug, uri in sorted(docs):
            url = f"{base}/{uri}/data.xml" if uri.startswith("d-") else f"{base}/{uri}/data.xml"
            req = urllib.request.Request(url, headers={"User-Agent": args.user_agent})
            with urllib.request.urlopen(req, timeout=30) as r:
                texts[slug] = xml_to_text(r.read())
            print(f"  fetched {slug}", file=sys.stderr)
    return texts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("anchors")
    p.add_argument("--txt-dir", help="verify against local normalized .txt files")
    p.add_argument("--fetch", action="store_true", help="fetch current XML from FCL")
    p.add_argument("--user-agent", default="anchor-verifier/1.0 (computational-analysis licence)")
    args = p.parse_args()

    spec = json.loads(Path(args.anchors).read_text(encoding="utf-8"))
    texts = load_texts(spec, args)

    tally = {"FAST": 0, "RELOCATED": 0, "LOST": 0, "MISSING_DOC": 0}
    revised = set()

    for a in spec["anchors"]:
        text = texts.get(a["doc"])
        if text is None:
            tally["MISSING_DOC"] += 1
            print(f"  MISSING_DOC {a['case_id']}/{a['field']} ({a['doc']})")
            continue
        if sha(text) != a["doc_sha256"]:
            revised.add(a["doc"])
        status, detail = check(text, a)
        tally[status] += 1
        if status != "FAST":
            print(f"  {status:10s} {a['case_id']}/{a['field']}  {detail}")

    print(f"\n{spec['n_anchors']} anchors: " +
          "  ".join(f"{k}={v}" for k, v in tally.items() if v))
    if revised:
        print(f"documents whose text has changed since annotation: {sorted(revised)}")
        print("-> check these are still published, and re-run the builder")
    return 1 if tally["LOST"] or tally["MISSING_DOC"] else 0


if __name__ == "__main__":
    sys.exit(main())

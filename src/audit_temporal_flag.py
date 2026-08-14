#!/usr/bin/env python3
"""
Temporal-content flag audit (no Gemma -- deterministic).

Applies the _has_temporal_signal logic to ALREADY-EXTRACTED TDG JSONs and
reports which facts WOULD be flagged temporal_content=False, so you can inspect
the flagged set before trusting it. Nothing is dropped or modified on disk.

This deliberately runs on existing extractions rather than re-running the LLM:
Gemma is nondeterministic, so a fresh extraction would conflate "flag effect"
with "the model rolled differently". Same input -> clean before/after diff.

Usage:
  python audit_temporal_flag.py --tdg-dir ../data/results_contracts_50
  python audit_temporal_flag.py --tdg-dir ../data/results_contracts_50 --show all
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from tdg_pipeline.llm_pipeline import _has_temporal_signal


def fact_has_value(fd: dict) -> bool:
    v = fd.get("value")
    return (v not in (None, "null", "None", "")
            or fd.get("duration_days") is not None
            or fd.get("date_parsed"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdg-dir", default="../data/results_contracts_50")
    ap.add_argument("--show", choices=["flagged", "kept-relative", "all", "none"],
                    default="flagged",
                    help="flagged = facts that would be marked non-temporal; "
                         "kept-relative = null-value facts KEPT via a trigger")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.tdg_dir, "*.json")))
    if not files:
        print(f"No TDG JSONs in {args.tdg_dir}")
        return 2

    n_facts = n_flagged = n_kept_relative = 0
    flagged, kept_relative = [], []
    per_role_flagged = {}

    for f in files:
        d = json.load(open(f))
        doc = d.get("document_id", os.path.basename(f))
        for fd in d.get("facts", []):
            n_facts += 1
            has_val = fact_has_value(fd)
            text = f"{fd.get('raw_text','')} {fd.get('sentence','')}"
            signal = _has_temporal_signal(text)
            temporal = bool(has_val) or signal
            if not temporal:
                n_flagged += 1
                per_role_flagged[fd.get("role")] = per_role_flagged.get(fd.get("role"), 0) + 1
                flagged.append((doc, fd.get("role"), fd.get("entity"),
                                (fd.get("raw_text") or "")[:60]))
            elif not has_val and signal:
                n_kept_relative += 1
                kept_relative.append((doc, fd.get("role"), fd.get("entity"),
                                      (fd.get("raw_text") or "")[:60]))

    print(f"facts={n_facts}  would-flag={n_flagged} ({n_flagged/n_facts:.1%})  "
          f"kept-relative(null+signal)={n_kept_relative}")
    print(f"flagged by role: {per_role_flagged}\n")

    if args.show in ("flagged", "all"):
        print(f"=== WOULD FLAG temporal_content=False ({len(flagged)}) -- "
              f"inspect for false drops ===")
        for r in (flagged if args.show == "all" else flagged):
            print(f"  [{r[1]:8}] {r[0]:24} {r[2]!r}  <- {r[3]!r}")
    if args.show in ("kept-relative", "all"):
        print(f"\n=== KEPT via temporal trigger though value is null "
              f"({len(kept_relative)}) -- these must NOT be lost ===")
        for r in kept_relative:
            print(f"  [{r[1]:8}] {r[0]:24} {r[2]!r}  <- {r[3]!r}")

    print("\nReview the flagged list. If any are genuinely temporal (periodicity "
          "like 'annually', ordinal periods, etc.), tighten _has_temporal_signal "
          "BEFORE enabling any downstream drop. Flagging changes nothing on disk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

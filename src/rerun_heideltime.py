#!/usr/bin/env python3
"""
Re-run the rule-based TDG pipeline with real HeidelTime.

Reads source_text out of an existing results directory, so the input is
byte-identical to whatever produced those results. No HuggingFace, no
re-sampling, no seed drift.

Records per document:
    raw_timex        spans HeidelTime returned
    timex_by_type    DATE / DURATION / SET / TIME breakdown
    facts            surviving role classification + entity linking + dedup
    deps             dependency edges, and the count by type
    seconds          wall clock

The stage counts matter: the gap between raw_timex and facts is where
the rule-based pipeline loses material, and that is a separate question
from whether the graph builder can type an edge.

Usage:
    # the 45 documents the LLM run covered (recommended)
    python rerun_heideltime.py \
        --input  data/results_contracts_50 \
        --output data/results_heideltime_real

    # the original 5 seeds only
    python rerun_heideltime.py \
        --input  data/results_heidel_raw \
        --output data/results_heideltime_seeds5

    # sanity check before committing to a long run
    python rerun_heideltime.py --input data/results_contracts_50 \
        --output /tmp/ht_probe --limit 3
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import date
from pathlib import Path

_MONTHS = ("january february march april may june july august september "
           "october november december").split()

def earliest_date(text: str):
    """Earliest explicit calendar date in the text, or None.

    Used as a per-document creation time. HeidelTime resolves
    underspecified expressions ("by 31 March") against a document
    creation time; a contract corpus supplies none, and a fixed global
    value pushes such dates decades outside the document's own period.
    A document's earliest explicit date is a defensible stand-in.
    Bare years are ignored: citation years are not document dates.
    """
    import re
    out = []
    pat_dmy = re.compile(r"\b(\d{1,2})\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})\b", re.I)
    pat_mdy = re.compile(r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2}),?\s+(\d{4})\b", re.I)
    pat_iso = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
    for m in pat_dmy.finditer(text):
        try: out.append(date(int(m.group(3)), _MONTHS.index(m.group(2).lower()) + 1, int(m.group(1))))
        except ValueError: pass
    for m in pat_mdy.finditer(text):
        try: out.append(date(int(m.group(3)), _MONTHS.index(m.group(1).lower()) + 1, int(m.group(2))))
        except ValueError: pass
    for m in pat_iso.finditer(text):
        try: out.append(date(*map(int, m.groups())))
        except ValueError: pass
    out = [d for d in out if 1900 <= d.year <= 2100]
    if out:
        return min(out)
    # Fallback: earliest bare year in the document. Weaker evidence than a
    # full date, but far better than the library default of today, which
    # would resolve an undated expression to the current year in a
    # decades-old instrument.
    yrs = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text)]
    yrs = [y for y in yrs if 1900 <= y <= 2100]
    return date(min(yrs), 1, 1) if yrs else None


def preflight(strict_model: str) -> dict:
    """Fail before a long run rather than during it."""
    info: dict = {}

    try:
        out = subprocess.run(["java", "-version"], capture_output=True, text=True)
        info["java"] = (out.stderr or out.stdout).splitlines()[0].strip()
    except FileNotFoundError:
        sys.exit("FATAL: no `java` on PATH. HeidelTime needs a JRE 11+.")

    try:
        import py_heideltime  # noqa: F401
        info["py_heideltime"] = getattr(py_heideltime, "__version__", "unknown")
    except ImportError:
        sys.exit("FATAL: py_heideltime not installed.  pip install py-heideltime")

    try:
        import spacy
        info["spacy"] = spacy.__version__
    except ImportError:
        sys.exit("FATAL: spacy not installed.")

    try:
        import spacy
        spacy.load(strict_model)
        info["spacy_model"] = strict_model
    except OSError:
        sys.exit(
            f"FATAL: spaCy model {strict_model!r} not available.\n"
            f"  python -m spacy download {strict_model}\n"
            f"Do not silently substitute a different model: it changes "
            f"role classification and entity linking, and the results "
            f"would not be comparable to the published run."
        )

    # Prove HeidelTime actually executes, not just imports.
    from py_heideltime import heideltime
    probe = heideltime(
        "Payment is due within 30 days of 15 January 2025.",
        language="English", document_type="news", dct="2025-01-01",
    )
    kinds = {p["type"] for p in probe}
    if not {"DATE", "DURATION"} <= kinds:
        sys.exit(f"FATAL: HeidelTime probe returned {probe!r}; expected a "
                 f"DATE and a DURATION. The backend is not working.")
    info["heideltime_probe"] = f"{len(probe)} spans, types={sorted(kinds)}"
    info["python"] = platform.python_version()
    return info


def git_rev(repo: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        )
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True,
        ).stdout.strip()
        return r.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="directory of JSONs carrying source_text")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--doctype", default="legal",
                    help="TDGPipeline document_type (default: legal)")
    ap.add_argument("--spacy-model", default="en_core_web_trf")
    ap.add_argument("--reference-date", default=None,
                    help="YYYY-MM-DD, or 'auto' to use each document's own "
                         "earliest explicit date as its creation time. "
                         "'auto' is recommended for an undated corpus.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-scenarios", action="store_true")
    args = ap.parse_args()

    env = preflight(args.spacy_model)
    print("environment")
    for k, v in env.items():
        print(f"  {k:<18} {v}")

    files = sorted(args.input.glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit(f"FATAL: no JSON files in {args.input}")
    print(f"\n{len(files)} documents from {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)

    # Instrument the extractor so raw span counts are visible per document.
    from tdg_pipeline import timex_extractor as tx_mod
    from tdg_pipeline.pipeline import TDGPipeline

    probe: dict = {}
    _orig_extract = tx_mod.HeidelTimeExtractor.extract

    def _instrumented(self, text, reference_date=None):
        spans = _orig_extract(self, text, reference_date)
        probe["raw_timex"] = len(spans)
        probe["by_type"] = dict(Counter(s.timex_type for s in spans))
        probe["backend"] = getattr(self, "backend_used", None)
        probe["unique_offsets"] = len({(s.start_char, s.end_char) for s in spans})
        return spans

    tx_mod.HeidelTimeExtractor.extract = _instrumented

    fixed_ref = None
    if args.reference_date and args.reference_date != "auto":
        fixed_ref = date.fromisoformat(args.reference_date)
    pipe = TDGPipeline()
    auto_refs = {}

    rows, failures = [], []
    hdr = (f"{'document':<28}{'chars':>7}{'timex':>7}{'facts':>7}"
           f"{'deps':>6}{'add':>5}{'ord':>5}{'sec':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))

    for f in files:
        src = json.loads(f.read_text())
        text = src.get("source_text")
        doc_id = src.get("document_id", f.stem)
        if not text:
            failures.append((doc_id, "no source_text"))
            continue

        if args.reference_date == "auto":
            ref = earliest_date(text)
            auto_refs[doc_id] = ref.isoformat() if ref else None
            if ref is not None and ref.month == 1 and ref.day == 1:
                auto_refs[doc_id] += " (year-only)"
        else:
            ref = fixed_ref

        probe.clear()
        t0 = time.time()
        try:
            tdg = pipe.process(
                text=text,
                document_id=doc_id,
                document_type=args.doctype,
                reference_date=ref,
                generate_scenarios=not args.no_scenarios,
            )
        except Exception as e:
            failures.append((doc_id, f"{type(e).__name__}: {e}"))
            print(f"{doc_id:<28}  FAILED  {type(e).__name__}: {str(e)[:60]}")
            traceback.print_exc(file=sys.stderr)
            continue
        secs = time.time() - t0

        backend = probe.get("backend")
        if backend != "heideltime":
            sys.exit(
                f"\nFATAL: extractor backend was {backend!r} on {doc_id}.\n"
                f"The regex fallback ran. Apply apply_patches.py (strict "
                f"mode) and confirm py_heideltime imports. Refusing to "
                f"write output that would be mislabelled."
            )

        deps = getattr(tdg, "dependencies", []) or []
        dep_types = Counter(
            getattr(d, "constraint_type", None) or getattr(d, "relation_type", None) or "?"
            for d in deps
        )
        facts = getattr(tdg, "facts", []) or []

        row = {
            "document_id": doc_id,
            "source_file": str(f),
            "chars": len(text),
            "raw_timex": probe.get("raw_timex", 0),
            "timex_by_type": probe.get("by_type", {}),
            "unique_offsets": probe.get("unique_offsets", 0),
            "facts": len(facts),
            "deps": len(deps),
            "dep_types": dict(dep_types),
            "seconds": round(secs, 2),
            "reference_date": (ref.isoformat() if ref else None),
        }
        rows.append(row)

        out = tdg.to_dict() if hasattr(tdg, "to_dict") else {}
        out["_provenance"] = {
            "extractor_backend": backend,
            "environment": env,
            "git": git_rev(Path.cwd()),
            "reference_date": args.reference_date,
        "auto_reference_dates": auto_refs,
            "spacy_model": args.spacy_model,
            "document_type": args.doctype,
            "stage_counts": row,
            "input_file": str(f),
        }
        (args.output / f"{doc_id}.json").write_text(json.dumps(out, indent=2, default=str))

        print(f"{doc_id:<28}{row['chars']:>7}{row['raw_timex']:>7}"
              f"{row['facts']:>7}{row['deps']:>6}"
              f"{dep_types.get('additive', 0):>5}{dep_types.get('ordering', 0):>5}"
              f"{row['seconds']:>7.1f}")

    # ---------------------------------------------------------------- totals
    print("-" * len(hdr))
    t_timex = sum(r["raw_timex"] for r in rows)
    t_facts = sum(r["facts"] for r in rows)
    t_deps = sum(r["deps"] for r in rows)
    all_types: Counter = Counter()
    for r in rows:
        all_types.update(r["dep_types"])
    print(f"{'TOTAL (' + str(len(rows)) + ' docs)':<28}"
          f"{sum(r['chars'] for r in rows):>7}{t_timex:>7}{t_facts:>7}{t_deps:>6}"
          f"{all_types.get('additive', 0):>5}{all_types.get('ordering', 0):>5}"
          f"{sum(r['seconds'] for r in rows):>7.1f}")

    print(f"\ntimex -> facts retention: {t_facts}/{t_timex} "
          f"({100 * t_facts / t_timex:.1f}%)" if t_timex else "")
    print(f"documents with >=1 dependency: "
          f"{sum(1 for r in rows if r['deps'] > 0)}/{len(rows)}")
    if args.reference_date == "auto":
        miss = [k for k, v in auto_refs.items() if v is None]
        weak = [k for k, v in auto_refs.items() if v and "year-only" in v]
        print(f"auto reference dates resolved: "
              f"{len(auto_refs) - len(miss)}/{len(auto_refs)}"
              f"  (of which year-only: {len(weak)})")
        if weak:
            print(f"  year-only DCT: {weak}")
        if miss:
            print(f"  NO date at all, library default (today) used: {miss}")
    print(f"dependency types: {dict(all_types) or 'none'}")

    if failures:
        print(f"\n{len(failures)} failures:")
        for d, why in failures:
            print(f"  {d}: {why}")

    summary = {
        "environment": env,
        "git": git_rev(Path.cwd()),
        "input_dir": str(args.input),
        "output_dir": str(args.output),
        "n_documents": len(rows),
        "n_failures": len(failures),
        "failures": failures,
        "totals": {
            "raw_timex": t_timex,
            "facts": t_facts,
            "deps": t_deps,
            "dep_types": dict(all_types),
            "docs_with_deps": sum(1 for r in rows if r["deps"] > 0),
        },
        "per_document": rows,
    }
    (args.output / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(rows)} graphs + _summary.json to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Re-run the LLM extractor over archived source_text.

Companion to rerun_heideltime.py. Reads the same input files, so both
columns of the comparison come from identical text and the current
codebase, rather than one column being a run of unknown vintage.

Records status per document, including the three ways LLMPipeline can
return an empty graph, so an empty result is never confused with a
document that genuinely has no temporal content.

Usage:
    python rerun_llm.py \
        --input  ../data/results_contracts_50 \
        --output ../data/results_llm_current \
        --model gemma4:e4b --base-url http://localhost:11434/v1

    # resume after an interruption: skips documents already written
    python rerun_llm.py ... --resume
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
from pathlib import Path


def git_rev(repo: Path) -> str:
    try:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
        return r.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1", dest="base_url")
    ap.add_argument("--doctype", default="legal")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=8192, dest="max_tokens")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-scenarios", action="store_true")
    args = ap.parse_args()

    from tdg_pipeline.llm_pipeline import LLMPipeline

    files = sorted(args.input.glob("*.json"))
    files = [f for f in files if not f.name.startswith("_")]
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit(f"FATAL: no JSON files in {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)

    env = {
        "model": args.model,
        "base_url": args.base_url or "openai",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "python": platform.python_version(),
        "git": git_rev(Path.cwd()),
    }
    print("environment")
    for k, v in env.items():
        print(f"  {k:<14} {v}")
    print(f"\n{len(files)} documents from {args.input}")

    pipe = LLMPipeline(model=args.model, temperature=args.temperature,
                       base_url=args.base_url, max_tokens=args.max_tokens)

    hdr = (f"{'document':<28}{'chars':>7}{'facts':>7}{'deps':>6}"
           f"{'add':>5}{'ord':>5}{'int':>5}{'sec':>7}  status")
    print("\n" + hdr)
    print("-" * (len(hdr) + 10))

    rows, statuses = [], Counter()

    for f in files:
        src = json.loads(f.read_text())
        text = src.get("source_text")
        doc_id = src.get("document_id", f.stem)
        target = args.output / f"{doc_id}.json"

        if args.resume and target.exists():
            print(f"{doc_id:<28}  skipped (exists)")
            continue
        if not text:
            statuses["no-source-text"] += 1
            print(f"{doc_id:<28}  SKIP no source_text")
            continue

        t0 = time.time()
        try:
            tdg = pipe.process(text=text, document_id=doc_id,
                               document_type=args.doctype,
                               generate_scenarios=not args.no_scenarios)
        except Exception as e:
            statuses[f"exception:{type(e).__name__}"] += 1
            print(f"{doc_id:<28}  FAILED {type(e).__name__}: {str(e)[:50]}")
            traceback.print_exc(file=sys.stderr)
            continue
        secs = time.time() - t0

        status = getattr(pipe, "last_status", "ok")
        statuses[status.split(":")[0]] += 1

        deps = getattr(tdg, "dependencies", []) or []
        facts = getattr(tdg, "facts", []) or []
        dt = Counter(getattr(d, "constraint_type", None) or "?" for d in deps)

        row = {
            "document_id": doc_id, "chars": len(text),
            "facts": len(facts), "deps": len(deps),
            "dep_types": dict(dt), "seconds": round(secs, 2),
            "status": status, "input_file": str(f),
        }
        rows.append(row)

        out = tdg.to_dict() if hasattr(tdg, "to_dict") else {}
        out["_provenance"] = {
            "extractor_backend": f"llm:{args.model}",
            "environment": env, "stage_counts": row,
            "llm_status": status, "input_file": str(f),
        }
        target.write_text(json.dumps(out, indent=2, default=str))

        print(f"{doc_id:<28}{row['chars']:>7}{row['facts']:>7}{row['deps']:>6}"
              f"{dt.get('additive', 0):>5}{dt.get('ordering', 0):>5}"
              f"{dt.get('interval', 0):>5}{secs:>7.1f}  {status[:28]}")

    print("-" * (len(hdr) + 10))
    tf = sum(r["facts"] for r in rows)
    td = sum(r["deps"] for r in rows)
    allt: Counter = Counter()
    for r in rows:
        allt.update(r["dep_types"])
    print(f"{'TOTAL (' + str(len(rows)) + ' docs)':<28}"
          f"{sum(r['chars'] for r in rows):>7}{tf:>7}{td:>6}"
          f"{allt.get('additive', 0):>5}{allt.get('ordering', 0):>5}"
          f"{allt.get('interval', 0):>5}{sum(r['seconds'] for r in rows):>7.1f}")
    print(f"\nstatuses: {dict(statuses)}")
    print(f"documents with >=1 dependency: "
          f"{sum(1 for r in rows if r['deps'] > 0)}/{len(rows)}")
    print(f"dependency types: {dict(allt) or 'none'}")

    (args.output / "_summary.json").write_text(json.dumps({
        "environment": env, "input_dir": str(args.input),
        "n_documents": len(rows), "statuses": dict(statuses),
        "totals": {"facts": tf, "deps": td, "dep_types": dict(allt)},
        "per_document": rows,
    }, indent=2))
    print(f"\nwrote {len(rows)} graphs + _summary.json to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""
Cross-doc linking runner.

Loads all TDG JSONs from a directory, runs CrossDocLinker across them,
and outputs coreferences, contradictions, and staleness propagation.

Usage:
    python run_cross_doc.py --dir ../data/results_contracts_50 \
        --output ../data/experiments/cross_doc_50.json

    # With embeddings
    python run_cross_doc.py --dir ../data/results_contracts_50 \
        --output ../data/experiments/cross_doc_50_embed.json \
        --embed-url http://localhost:11434/v1
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tdg_pipeline.cross_doc import CrossDocLinker
from tdg_pipeline.io import load_tdg_dir
from tdg_pipeline.embeddings import EmbeddingSimilarity


def run(tdgs, embedder=None, output_path=None):
    """Run CrossDocLinker on a list of TDGs, print results, optionally save JSON."""
    total_facts = 0
    for t in tdgs:
        connected = set()
        for d in t.dependencies:
            connected.add(d.from_id)
            connected.add(d.to_id)
        print(f"  {t.document_id:30s} {t.document_type:18s} "
              f"{len(t.facts):2d} facts  {len(t.dependencies):2d} deps  "
              f"{len(connected)}/{len(t.facts)} connected")
        total_facts += len(t.facts)

    linker = CrossDocLinker(embedder=embedder)
    for t in tdgs:
        linker.add_tdg(t)

    links = linker.find_all_links()
    print(f"\n  LINKS FOUND: {len(links)}")

    by_type = {}
    for link in links:
        by_type.setdefault(link.link_type, []).append(link)

    for ltype in ["coreference", "contradiction", "structural_analogy"]:
        group = by_type.get(ltype, [])
        if not group:
            continue
        print(f"\n  -- {ltype.upper()} ({len(group)}) --")
        for link in group:
            print(f"    {link.from_doc}/{link.from_fact} ↔ {link.to_doc}/{link.to_fact}")
            print(f"      conf={link.confidence:.2f}  {link.explanation}")
            if link.value_a or link.value_b:
                delta_str = f"  Δ{link.delta_days}d" if link.delta_days else ""
                print(f"      values: {link.value_a} vs {link.value_b}{delta_str}")

    # Staleness: edit the first dated fact in the first document
    if tdgs:
        t = tdgs[0]
        edit_fact = None
        for f in t.facts:
            if f.timex.date_parsed:
                edit_fact = f
                break
        if edit_fact:
            print(f"\n  -- STALENESS PROPAGATION --")
            print(f"  Edit: {t.document_id}/{edit_fact.id} ({edit_fact.entity})")
            stale = linker.propagate_staleness(t.document_id, edit_fact.id, delta_days=30)
            if stale:
                for sf in stale:
                    print(f"    STALE: {sf.doc_id}/{sf.fact_id} "
                          f"(was: {sf.old_value}, hops: {sf.hop_distance})")
                    print(f"      reason: {sf.reason}")
            else:
                print(f"    No stale facts found via additive chains from {edit_fact.id}")

    # Save JSON output
    if output_path:
        results = {
            "documents": len(tdgs),
            "total_facts": total_facts,
            "total_links": len(links),
            "links_by_type": {lt: len(group) for lt, group in by_type.items()},
            "links": [
                {
                    "from_doc": link.from_doc,
                    "from_fact": link.from_fact,
                    "to_doc": link.to_doc,
                    "to_fact": link.to_fact,
                    "link_type": link.link_type,
                    "confidence": link.confidence,
                    "explanation": link.explanation,
                    "value_a": link.value_a,
                    "value_b": link.value_b,
                    "delta_days": link.delta_days,
                }
                for link in links
            ],
        }
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  ✓ Results saved to {output_path}")

    return links


def main():
    parser = argparse.ArgumentParser(description="Run cross-doc linking on a directory of TDGs")
    parser.add_argument("--dir", required=True,
                        help="Directory of TDG JSON files")
    parser.add_argument("--output", default=None,
                        help="Path to save JSON results")
    parser.add_argument("--embed-url", default=None,
                        help="Ollama URL for embeddings (e.g. http://localhost:11434/v1)")
    parser.add_argument("--embed-model", default="nomic-embed-text",
                        help="Embedding model name (default: nomic-embed-text)")
    args = parser.parse_args()

    embedder = None
    if args.embed_url:
        embedder = EmbeddingSimilarity(base_url=args.embed_url, model=args.embed_model)
        print(f"Using embeddings: {args.embed_model} @ {args.embed_url}")

    tdgs_dict = load_tdg_dir(args.dir, max_source_text=300)
    if not tdgs_dict:
        print(f"No TDG JSONs found in {args.dir}")
        return

    print(f"\n{'='*70}")
    print(f"  CROSS-DOC: {len(tdgs_dict)} documents from {args.dir}")
    print(f"{'='*70}")

    run(list(tdgs_dict.values()), embedder=embedder, output_path=args.output)


if __name__ == "__main__":
    main()
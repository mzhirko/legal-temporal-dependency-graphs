"""
run_comparison.py -- entry point for the dual-pipeline comparison.

Runs both pipelines independently on each document in a results directory:
  Pipeline A (TDG):    reads existing TDG JSON from results_contracts/
  Pipeline B (Catala): generates .catala_en, repairs if needed, runs scope

Then compares outputs and writes a ComparisonReport JSON per document.

Usage:
    python src/run_comparison.py \
      --results-dir data/results_contracts \
      --output-dir experiments/comparison_results \
      --catala-dir experiments/catala_generated \
      --examples-dir experiments/catala_examples \
      --ollama-url http://localhost:11434 \
      --embed-url http://localhost:11437 \
      --model gemma4:e4b \
      --embed-model nomic-embed-text

    # Rebuild RAG index (do this once after adding new examples):
    python src/run_comparison.py --rebuild-index ...
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Simple progress printer
# ---------------------------------------------------------------------------

def _step(n: int, total: int, label: str) -> None:
    print(f"  [{n}/{total}] {label}", flush=True)

def _ok(msg: str) -> None:
    print(f"  ✓ {msg}", flush=True)

def _warn(msg: str) -> None:
    print(f"  x {msg}", flush=True)

def _section(title: str) -> None:
    print(f"\n{'-' * 60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'-' * 60}", flush=True)


# ---------------------------------------------------------------------------
# Imports (deferred to after progress prints start)
# ---------------------------------------------------------------------------

from catala_pipeline.catala_runner import run_scope, setup_project_dir, json_schema
from catala_pipeline.example_retriever import ExampleRetriever
from catala_pipeline.input_extractor import InputExtractor
from catala_pipeline.tdg_binder import bind_inputs
from catala_pipeline.llm_encoder import LLMEncoder
from catala_pipeline.repair_loop import run_repair_loop
from comparator.align import align_and_compare
from comparator.report import ComparisonReport
from tdg_pipeline.io import load_json, build_tdg
from tdg_pipeline.embeddings import EmbeddingSimilarity


def _extract_scope_name(catala_content: str) -> str | None:
    """Extract the first scope name declared in a .catala_en file."""
    m = re.search(r"declaration scope (\w+)", catala_content)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Per-document processing
# ---------------------------------------------------------------------------

def process_document(
    tdg_path: Path,
    catala_dir: Path,
    encoder: LLMEncoder,
    extractor: InputExtractor,
    retriever: ExampleRetriever,
    embedder: Optional[EmbeddingSimilarity] = None,
    max_repair_attempts: int = 3,
    force_regen: bool = False,
) -> ComparisonReport:
    """Run both pipelines on one document and return a ComparisonReport."""
    tdg_data = load_json(str(tdg_path))

    document_id = tdg_data["document_id"]
    source_text = tdg_data["source_text"]
    n_facts = len(tdg_data.get("facts", []))

    _section(f"Document: {document_id}")
    print(f"  TDG facts loaded: {n_facts}", flush=True)

    catala_file = catala_dir / f"{document_id}.catala_en"
    STEPS = 6

    # Step 1: RAG retrieval
    _step(1, STEPS, "Retrieving RAG examples...")
    t0 = time.time()
    examples = retriever.retrieve(source_text[:2000], top_k=2)
    _ok(f"Retrieved {len(examples)} examples ({time.time()-t0:.1f}s)")

    # Step 2: LLM Call 1 -- generate .catala_en (skip if already exists)
    _step(2, STEPS, f"Generating Catala scope with {encoder.model}...")
    t0 = time.time()
    if catala_file.exists() and not force_regen:
        n_lines = len(catala_file.read_text().splitlines())
        _ok(f"Reusing existing {catala_file.name} ({n_lines} lines)")
    else:
        encoder.encode(
            legal_text=source_text,
            retrieved_examples=examples,
            output_path=catala_file,
        )
        n_lines = len(catala_file.read_text().splitlines())
        _ok(f"Generated {catala_file.name} ({n_lines} lines, {time.time()-t0:.1f}s)")

    # Step 3: typecheck + repair loop
    _step(3, STEPS, "Typechecking (repair loop)...")
    t0 = time.time()
    catala_result = run_repair_loop(
        catala_file=catala_file,
        legal_text=source_text,
        retrieved_examples=examples,
        encoder=encoder,
        scope_name="",
        document_id=document_id,
        max_attempts=max_repair_attempts,
    )
    elapsed = time.time() - t0
    if catala_result.status == "success":
        _ok(f"Typecheck passed (repairs: {catala_result.repair_attempts}, {elapsed:.1f}s)")
    else:
        _warn(f"Typecheck failed after {catala_result.repair_attempts} attempts ({elapsed:.1f}s)")
        _warn(f"Status: {catala_result.status} -- saved error log alongside .catala_en")
        return align_and_compare(
            document_id=document_id,
            tdg_facts=[],
            catala_outputs={},
            catala_status=catala_result.status,
            scope_name=None,
            repair_attempts=catala_result.repair_attempts,
        )

    # Step 4: detect scope name
    _step(4, STEPS, "Detecting scope name...")
    scope_name = _extract_scope_name(catala_file.read_text())
    if not scope_name:
        _warn("Could not detect scope name in generated file.")
        return align_and_compare(
            document_id=document_id,
            tdg_facts=[],
            catala_outputs={},
            catala_status="interpret_error",
            scope_name=None,
            repair_attempts=catala_result.repair_attempts,
        )
    _ok(f"Scope: {scope_name}")

    # Step 5: bind inputs from the TDG (deterministic), LLM fallback for the rest
    _step(5, STEPS, f"Binding TDG facts to scope inputs for {scope_name}...")
    t0 = time.time()

    binding = None
    schemas = json_schema(catala_file, scope_name)
    if schemas is not None:
        input_properties = (
            schemas[0]
            .get("definitions", {})
            .get(f"{scope_name}_in", {})
            .get("properties", {})
        )
        binding = bind_inputs(input_properties, tdg_data.get("facts", []))
        _ok(binding.summary_line())

    if binding is not None and not binding.unbound:
        # Every input supplied by the TDG -- no second LLM call needed.
        inputs, placeholder_fields = dict(binding.inputs), set()
    else:
        # LLM fallback for unbound (or schema-less) inputs only.
        inputs, placeholder_fields, error = extractor.extract(
            catala_file, scope_name, source_text
        )
        if inputs is None:
            if binding is None or not binding.inputs:
                _warn(f"Input extraction failed: {error}")
                return align_and_compare(
                    document_id=document_id,
                    tdg_facts=[],
                    catala_outputs={},
                    catala_status="interpret_error",
                    scope_name=scope_name,
                    repair_attempts=catala_result.repair_attempts,
                    catala_file=catala_file,
                    placeholder_fields=set(),
                )
            inputs, placeholder_fields = {}, set()
        if binding is not None:
            # TDG-bound values override the LLM's re-extraction; a bound
            # field is by definition not a placeholder.
            inputs.update(binding.inputs)
            placeholder_fields -= set(binding.inputs.keys())

    # Persist binding provenance next to the .catala_en for auditability.
    if binding is not None:
        bindings_path = catala_file.with_name(f"{document_id}_bindings.json")
        bindings_path.write_text(json.dumps(
            {
                "document_id": document_id,
                "scope_name": scope_name,
                "bindings": binding.bindings,
                "unbound": sorted(binding.unbound),
                "llm_fallback_used": bool(binding.unbound),
            },
            indent=2,
        ))
    _ok(
        f"Inputs ready: {list(inputs.keys())} | "
        f"tdg_bound: {sorted(binding.inputs.keys()) if binding else []} | "
        f"placeholders: {placeholder_fields} ({time.time()-t0:.1f}s)"
    )

    # Step 6: run the scope
    _step(6, STEPS, "Running Catala scope...")
    t0 = time.time()
    success, outputs, run_error = run_scope(catala_file, scope_name, inputs)
    if not success:
        _warn(f"Scope execution failed: {run_error}")
        return align_and_compare(
            document_id=document_id,
            tdg_facts=[],
            catala_outputs={},
            catala_status="interpret_error",
            scope_name=scope_name,
            repair_attempts=catala_result.repair_attempts,
        )
    _ok(f"Outputs: {outputs} ({time.time()-t0:.1f}s)")

    # Comparison
    tdg_facts = build_tdg(tdg_data).facts
    report = align_and_compare(
        document_id=document_id,
        tdg_facts=tdg_facts,
        catala_outputs=outputs,
        catala_status="success",
        scope_name=scope_name,
        repair_attempts=catala_result.repair_attempts,
        catala_file=catala_file,
        placeholder_fields=placeholder_fields,
        embedder=embedder,
        catala_inputs=inputs,
    )

    # Print comparison summary
    print(f"\n  Comparison:", flush=True)
    for field in report.fields:
        icon = "✓" if field.status == "match" else "~" if field.status == "off_by_one" else "x"
        tdg_val = field.tdg_value or "(missing)"
        cat_val = field.catala_value or "(missing)"
        print(f"    {icon} {field.variable_name}: TDG={tdg_val} | Catala={cat_val} [{field.status}]", flush=True)
    if report.match_rate is not None:
        print(f"  Match rate: {report.match_rate:.0%} ({report.match_count}/{report.match_count + report.mismatch_count} aligned fields)", flush=True)

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run TDG vs Catala comparison pipeline.")
    parser.add_argument("--results-dir", required=True, help="Directory with TDG JSON files")
    parser.add_argument("--output-dir", required=True, help="Directory for comparison reports")
    parser.add_argument("--catala-dir", required=True, help="Directory for generated .catala_en files")
    parser.add_argument("--examples-dir", required=True, help="Directory with verified .catala_en examples")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL for LLM calls")
    parser.add_argument("--embed-url", default=None, help="Ollama base URL for embedding (defaults to --ollama-url)")
    parser.add_argument("--model", default="gemma4:e4b", help="Ollama model name")
    parser.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild RAG index before running")
    parser.add_argument("--max-repairs", type=int, default=3, help="Max LLM repair attempts")
    parser.add_argument("--force-regen", action="store_true", help="Regenerate .catala_en even if it already exists")
    parser.add_argument("--doc", help="Process only this document ID (for testing)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    catala_dir = Path(args.catala_dir)
    examples_dir = Path(args.examples_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    catala_dir.mkdir(parents=True, exist_ok=True)

    llm_url = args.ollama_url
    embed_url = args.embed_url if args.embed_url else args.ollama_url

    print(f"\n{'=' * 60}", flush=True)
    print(f"  TDG vs Catala Comparison Pipeline", flush=True)
    print(f"  Model:  {args.model} @ {llm_url}", flush=True)
    print(f"  Embed:  {args.embed_model} @ {embed_url}", flush=True)
    print(f"{'=' * 60}", flush=True)

    # Set up Catala project dir
    print("\nSetting up Catala project directory...", flush=True)
    setup_project_dir(catala_dir)
    _ok("Catala project ready")

    # Set up RAG index
    # Index lives alongside the catala_pipeline module, not in cwd
    index_dir = Path(__file__).parent / "catala_pipeline" / "index"
    retriever = ExampleRetriever(
        examples_dir=examples_dir,
        index_dir=index_dir,
        embed_model=args.embed_model,
        ollama_base_url=embed_url,
    )
    if args.rebuild_index or not index_dir.exists():
        print("\nBuilding RAG index...", flush=True)
        t0 = time.time()
        n = retriever.build_index(force_rebuild=args.rebuild_index)
        _ok(f"Indexed {n} examples ({time.time()-t0:.1f}s)")
    else:
        print(f"\nUsing existing RAG index at {index_dir}", flush=True)

    encoder = LLMEncoder(model=args.model, base_url=f"{llm_url}/v1")
    extractor = InputExtractor(model=args.model, base_url=f"{llm_url}/v1")
    embedder = EmbeddingSimilarity(
        base_url=f"{embed_url}/v1",
        model=args.embed_model,
    )

    # Find documents to process
    tdg_files = sorted(results_dir.glob("*.json"))
    if args.doc:
        tdg_files = [f for f in tdg_files if args.doc in f.stem]
    if not tdg_files:
        print(f"\nNo JSON files found in {results_dir}")
        return

    print(f"\nFound {len(tdg_files)} document(s) to process.", flush=True)
    pipeline_start = time.time()

    all_reports = []
    for i, tdg_path in enumerate(tdg_files, 1):
        print(f"\n[Document {i}/{len(tdg_files)}]", flush=True)
        report = process_document(
            tdg_path=tdg_path,
            catala_dir=catala_dir,
            encoder=encoder,
            extractor=extractor,
            retriever=retriever,
            embedder=embedder,
            max_repair_attempts=args.max_repairs,
            force_regen=args.force_regen,
        )
        output_path = output_dir / f"{report.document_id}_comparison.json"
        output_path.write_text(report.to_json(), encoding="utf-8")
        _ok(f"Report saved: {output_path.name}")
        all_reports.append(report.to_dict())

    # Aggregate summary
    total_elapsed = time.time() - pipeline_start
    def _sum_field(reports, field):
        return sum(r["summary"].get(field, 0) for r in reports)

    docs_with_rate = [r for r in all_reports if r["summary"]["match_rate"] is not None]

    summary = {
        "total_documents": len(all_reports),
        "catala_success": sum(1 for r in all_reports if r["catala_status"] == "success"),
        "catala_repair_failed": sum(1 for r in all_reports if r["catala_status"] == "repair_failed"),
        "catala_typecheck_error": sum(1 for r in all_reports if r["catala_status"] == "typecheck_error"),
        "catala_interpret_error": sum(1 for r in all_reports if r["catala_status"] == "interpret_error"),
        "field_totals": {
            "match":              _sum_field(all_reports, "match"),
            "mismatch":           _sum_field(all_reports, "mismatch"),
            "tdg_only":           _sum_field(all_reports, "tdg_only"),
            "catala_only":        _sum_field(all_reports, "catala_only"),
            "placeholder":        sum(
                sum(1 for f in r["fields"] if f["status"] == "placeholder")
                for r in all_reports
            ),
            "semantic_match":     sum(
                sum(1 for f in r["fields"] if f["status"] == "semantic_match")
                for r in all_reports
            ),
            "value_match":        sum(
                sum(1 for f in r["fields"] if f["status"] == "value_match")
                for r in all_reports
            ),
            "duration_match":     sum(
                sum(1 for f in r["fields"] if f["status"] == "duration_match")
                for r in all_reports
            ),
            "duration_mismatch":  sum(
                sum(1 for f in r["fields"] if f["status"] == "duration_mismatch")
                for r in all_reports
            ),
            "type_mismatch":      sum(
                sum(1 for f in r["fields"] if f["status"] == "type_mismatch")
                for r in all_reports
            ),
        },
        "avg_match_rate": _safe_avg([
            r["summary"]["match_rate"] for r in docs_with_rate
        ]),
        "docs_with_comparable_fields": len(docs_with_rate),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "documents": all_reports,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Final summary print
    print(f"\n{'=' * 60}", flush=True)
    print(f"  DONE -- {len(all_reports)} documents in {total_elapsed:.0f}s", flush=True)
    print(f"  Catala success:       {summary['catala_success']}/{len(all_reports)}", flush=True)
    print(f"  Repair failed:        {summary['catala_repair_failed']}", flush=True)
    print(f"  Interpret error:      {summary['catala_interpret_error']}", flush=True)
    ft = summary["field_totals"]
    print(f"  Field totals across all documents:", flush=True)
    print(f"    match/off_by_one:    {ft['match']}", flush=True)
    print(f"    semantic_match:      {ft['semantic_match']}", flush=True)
    print(f"    value_match:         {ft['value_match']}", flush=True)
    print(f"    mismatch:            {ft['mismatch']}", flush=True)
    print(f"    placeholder:         {ft['placeholder']}", flush=True)
    print(f"    tdg_only:            {ft['tdg_only']}", flush=True)
    print(f"    catala_only:         {ft['catala_only']}", flush=True)
    print(f"    duration_match:      {ft['duration_match']}", flush=True)
    print(f"    duration_mismatch:   {ft['duration_mismatch']}", flush=True)
    print(f"    type_mismatch:       {ft['type_mismatch']}", flush=True)
    if summary["avg_match_rate"] is not None:
        print(f"  Avg match rate (aligned fields only): {summary['avg_match_rate']:.0%}", flush=True)
        print(f"  Documents with comparable fields: {summary['docs_with_comparable_fields']}/{len(all_reports)}", flush=True)
    print(f"  Summary saved:        {summary_path}", flush=True)
    print(f"{'=' * 60}\n", flush=True)


def _safe_avg(values: list) -> float | None:
    return sum(values) / len(values) if values else None


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Run the LLM-based TDG pipeline on legal text.

Input modes:
  1. Multiple random samples from Multi_Legal_Pile (default, use --n)
  2. Single random sample (--n 1 or omit --n)
  3. Raw text passed directly via --text
  4. Text from a file via --file

Text selection modes (dataset mode only):
  default        : best single paragraph scored for temporal content
  --full-cleaned : all cleaned paragraphs concatenated, up to --max-chars
  --raw          : raw document text, whitespace normalized, up to --max-chars
                   (no cleaning -- lets the LLM handle noise directly)

Usage:
    # Raw full document -- no cleaning
    python demo_llm.py --n 5 --model gemma4:e4b \\
        --base-url http://localhost:11434/v1 \\
        --subset en_contracts --no-scenarios \\
        --raw --output-dir results_raw/

    # Full cleaned document
    python demo_llm.py --n 5 --model gemma4:e4b \\
        --base-url http://localhost:11434/v1 \\
        --subset en_contracts --no-scenarios \\
        --full-cleaned --output-dir results_full/

    # Single best paragraph (original behaviour)
    python demo_llm.py --n 5 --model gemma4:e4b \\
        --base-url http://localhost:11434/v1 \\
        --subset en_contracts --no-scenarios \\
        --output-dir results_single/

    # Pass a text file
    python demo_llm.py --file my_contract.txt --model gemma4:e4b \\
        --base-url http://localhost:11434/v1

Requirements:
    pip install openai datasets
    For OpenAI: pip install python-dotenv + .env with OPENAI_API_KEY=sk-...
    For Ollama: ollama pull gemma4:e4b  (no extra packages needed)
"""

import os
import sys
import json
import argparse
import random
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tdg_pipeline.llm_pipeline import LLMPipeline
from tdg_pipeline.text_cleaner import best_paragraph, clean_to_paragraphs

_DEFAULT_MAX_CHARS = 15000


def load_sample(subset: str, seed: int) -> dict:
    from datasets import load_dataset

    ds = load_dataset(
        "joelniklaus/Multi_Legal_Pile",
        subset,
        split="train",
        streaming=True,
    )

    rng = random.Random(seed)
    skip = rng.randint(0, 500)

    for i, record in enumerate(ds):
        if i >= skip:
            return record

    raise RuntimeError("Could not load sample from dataset.")


def select_text(raw_text: str, mode: str, max_chars: int) -> str:
    """
    Select text from a raw document based on mode.

    mode:
      'best'         -- single best paragraph (original behaviour)
      'full_cleaned' -- all cleaned paragraphs concatenated, truncated to max_chars
      'raw'          -- raw text, whitespace normalized, truncated to max_chars
    """
    if mode == "best":
        return best_paragraph(raw_text)

    if mode == "full_cleaned":
        paragraphs = clean_to_paragraphs(raw_text)
        if not paragraphs:
            # fallback to raw if cleaning destroys everything
            return re.sub(r"\s+", " ", raw_text).strip()[:max_chars]
        return " ".join(paragraphs)[:max_chars]

    if mode == "raw":
        return re.sub(r"\s+", " ", raw_text).strip()[:max_chars]

    raise ValueError(f"Unknown mode: {mode}")


def run_single(pipe: LLMPipeline, text: str, doc_id: str,
               doc_type: str, generate_scenarios: bool) -> dict:
    """Run pipeline on one text, print summary, return TDG dict."""
    tdg = pipe.process(
        text=text,
        document_id=doc_id,
        document_type=doc_type,
        generate_scenarios=generate_scenarios,
    )

    print(f"\n{'='*60}")
    print(tdg.summary())
    print(f"{'='*60}")

    if tdg.edit_scenarios:
        s = tdg.edit_scenarios[0]
        print(f"\nSample edit scenario:")
        print(f"  Edit: {s['edit']['role']} {s['edit']['target_id']}: "
              f"{s['edit']['old_value']} -> {s['edit']['new_value']}")
        for c in s["expected_cascades"]:
            note = f"  [{c.get('note', '')}]" if c.get("note") else ""
            print(f"  -> Cascade: {c['role']} {c['fact_id']}: "
                  f"{c['old_value']} -> {c['new_value']}{note}")
        print(f"  Ripple depth={s['ripple_depth']}, breadth={s['ripple_breadth']}")

    return tdg.to_dict()


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-based TDG pipeline on legal text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--text", type=str, default=None,
        help="Raw text to process directly"
    )
    input_group.add_argument(
        "--file", type=str, default=None,
        help="Path to a text file to process"
    )

    # Dataset options
    parser.add_argument("--subset", default="en_caselaw",
                        help="Multi_Legal_Pile config (default: en_caselaw). "
                             "Options: en_caselaw, en_contracts, en_legislation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for single sample (default: 42). "
                             "Ignored when --n > 1 (uses seeds 0..n-1).")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of random samples to run (default: 1). "
                             "Uses seeds 0, 1, 2, ... n-1.")

    # Text selection mode (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--full-cleaned", action="store_true",
                            help="Send all cleaned paragraphs concatenated "
                                 "(up to --max-chars).")
    mode_group.add_argument("--raw", action="store_true",
                            help="Send raw document text with normalized whitespace "
                                 "(up to --max-chars). No cleaning -- LLM handles noise.")

    parser.add_argument("--max-chars", type=int, default=_DEFAULT_MAX_CHARS,
                        dest="max_chars",
                        help=f"Character limit for full-doc modes "
                             f"(default: {_DEFAULT_MAX_CHARS})")

    # Pipeline options
    parser.add_argument("--type", default="legal",
                        help="Document type (default: legal)")
    parser.add_argument("--document-id", default=None, dest="document_id",
                        help="Document ID for the output TDG. Defaults to filename "
                             "or 'cli_input' for --text mode.")
    parser.add_argument("--no-scenarios", action="store_true",
                        help="Skip edit scenario generation")
    parser.add_argument("--show-text", action="store_true",
                        help="Print cleaned paragraphs and exit (single sample only)")

    # Model options
    parser.add_argument("--model", default="gpt-4o",
                        help="Model name (default: gpt-4o). "
                             "For Ollama use e.g. gemma4:e4b")
    parser.add_argument("--base-url", default=None, dest="base_url",
                        help="Ollama base URL, e.g. http://localhost:11434/v1")

    # Output options
    parser.add_argument("--output", default=None,
                        help="Save TDG JSON to this file (single sample mode)")
    parser.add_argument("--output-dir", default=None, dest="output_dir",
                        help="Directory to save one JSON per sample. "
                             "Created if it does not exist.")

    args = parser.parse_args()

    # Determine text selection mode
    if args.full_cleaned:
        text_mode = "full_cleaned"
    elif args.raw:
        text_mode = "raw"
    else:
        text_mode = "best"

    # Init pipeline
    pipe = LLMPipeline(model=args.model, base_url=args.base_url)
    backend = f"Ollama ({args.base_url})" if args.base_url else "OpenAI"
    print(f"Backend: {backend}  |  Model: {args.model}  |  Mode: {text_mode}\n")

    # --- Direct text / file mode (always single) ---
    if args.text or args.file:
        if args.text:
            text = args.text
            doc_id = args.document_id or "cli_input"
            print(f"Using --text input ({len(text)} chars)")
        else:
            if not os.path.exists(args.file):
                print(f"Error: file not found: {args.file}")
                sys.exit(1)
            with open(args.file) as f:
                text = f.read()
            doc_id = args.document_id or os.path.splitext(os.path.basename(args.file))[0]
            print(f"Loaded file: {args.file} ({len(text)} chars)")

        result = run_single(pipe, text, doc_id, args.type, not args.no_scenarios)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved to: {args.output}")
        return

    # --- Dataset mode ---
    seeds = list(range(args.n)) if args.n > 1 else [args.seed]

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"  Sample seed={seed}  subset={args.subset}")
        print(f"{'#'*60}")

        print(f"Loading from Multi_Legal_Pile ({args.subset})...")
        sample = load_sample(args.subset, seed)
        raw_text = sample.get("text", sample.get("content", ""))
        if not raw_text:
            print(f"  Skipping seed={seed}: no text in sample.")
            continue

        jurisdiction = sample.get("jurisdiction", "unknown")
        print(f"Jurisdiction: {jurisdiction}  |  Raw length: {len(raw_text)} chars")

        if args.show_text and args.n == 1:
            paragraphs = clean_to_paragraphs(raw_text)
            print(f"\nCleaned into {len(paragraphs)} paragraphs:\n")
            for i, p in enumerate(paragraphs):
                print(f"[{i}] {p[:300]}\n")
            return

        text = select_text(raw_text, text_mode, args.max_chars)
        doc_id = f"{args.subset}_seed{seed}"
        print(f"Selected text ({len(text)} chars):\n  {text[:200]}...\n")

        result = run_single(pipe, text, doc_id, args.type, not args.no_scenarios)

        if args.output_dir:
            out_path = os.path.join(args.output_dir, f"{doc_id}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved: {out_path}")
        elif args.output and args.n == 1:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved to: {args.output}")

    if args.output_dir:
        print(f"\nDone. {len(seeds)} samples saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
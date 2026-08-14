# Recent updates — Catala comparison fix + enrichment

Brief notes on the comparator changes. (Proof-of-concept notes are documented
separately.)

## What changed

- **Aligner fix** (`comparator/align.py`). The semantic bridge that matches a
  Catala-computed *date* to a TDG *duration* clause now handles subtractions and
  computed/literal anchors (e.g. `deadline = expiry - notice_period`), which were
  previously dropped as `catala_only`. Duration-clause matching also scores on
  the best of provenance overlap and entity-name overlap, recovering paraphrases.
- **Honest reporting** (`comparator/honest_report.py`). Replaces the misleading
  `match_rate` (which was computed only over the overlap and ignored fields one
  side never produced). Reports **coverage** (fields modelled by both) and
  **within-overlap agreement** separately, and counts Catala failures
  (`interpret_error`, `repair_failed`, …) instead of dropping them.
- **Enrichment** (`comparator/catala_enrich.py`, `comparator/apply_enrichment.py`).
  Uses Catala's verified dates to **VERIFY** (confirm a TDG clause), **FILL** (add
  a date the TDG lacks), or **REVIEW** (flag a disagreement). REVIEW items are
  never auto-applied.

## New files

| File | Purpose |
|---|---|
| `comparator/align.py` | Patched aligner (semantic date↔duration matching) |
| `comparator/honest_report.py` | Coverage + within-overlap agreement report |
| `comparator/catala_enrich.py` | Emit VERIFY / FILL / REVIEW proposals |
| `comparator/apply_enrichment.py` | Write FILL/VERIFY dates into TDGs (reversible) |

## Running them

All from `code/src/`. Adjust paths to your comparison output dir.

### 1. Re-align with the patched aligner

Re-runs execution + alignment over the **existing** `.catala_en` files (no LLM
regeneration). Omit `--force-regen` so the Catala scripts are reused, not rewritten.

```bash
python run_comparison.py \
  --results-dir ../data/results_contracts \
  --output-dir ../data/experiments/comparison_results \
  --catala-dir ../data/experiments/catala_generated \
  --examples-dir ../data/experiments/catala_examples \
  --ollama-url http://localhost:11434 --embed-url http://localhost:11434 \
  --model gemma4:e4b --embed-model nomic-embed-text
```

Sanity check the fix took effect: coverage should rise above the pre-fix value
and `semantic_match` should appear in the status buckets.

### 2. Honest comparison report

```bash
python comparator/honest_report.py ../data/experiments/comparison_results
```

### 3. Enrichment proposals

```bash
python comparator/catala_enrich.py ../data/experiments/comparison_results \
  --out ../data/experiments/enrichment
```

### 4. Apply enrichment to the TDGs

Preview first (writes nothing), then write enriched **copies** (originals
untouched; use `--in-place` to edit originals, which writes a `.bak` first).

```bash
python comparator/apply_enrichment.py \
  --enrichment-dir ../data/experiments/enrichment \
  --tdg-dir ../data/results_contracts --dry-run

python comparator/apply_enrichment.py \
  --enrichment-dir ../data/experiments/enrichment \
  --tdg-dir ../data/results_contracts \
  --out-dir ../data/results_contracts_enriched
```

Added facts are tagged `source="catala_enrichment"`; confirmed facts carry
`catala_verified=true`. To revert: drop `source=catala_enrichment` facts and
remove the `catala_verified` / `catala_variable` annotations.

## Notes

- Run order matters: re-align (1) **before** enrichment (3,4), or the proposals
  reflect the old alignment (e.g. a computed deadline shows as FILL instead of a
  semantic VERIFY).
- Status vocabulary: `match`/`off_by_one`/`value_match` (exact agreement),
  `semantic_match` (Catala date consistent with a TDG duration clause),
  `*_mismatch` (disagreement), `tdg_only` / `catala_only` (one side only).
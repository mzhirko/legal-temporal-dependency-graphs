# LexTime — running a *valid* TDG-vs-LLM comparison

## What was wrong before

The old runs are not comparable, for two independent reasons:

1. **Single-class baseline.** `run_external.py` selects items with
   `items[:limit]`. The official `lextime.csv` is sorted with all 259
   `entailment` rows first. So `--limit 200` off that file gave a subset whose
   gold is **`{YES: 200}`** — every item positive. On a single-class set,
   "accuracy" is just the model's YES-rate; a constant "always YES" classifier
   scores 1.0. The baseline's 0.79 is therefore *not* a skill measurement, and
   it was never tested on a single negative item.
2. **Misaligned conditions.** The TDG run used `lextime_shuffled.csv` (balanced,
   `{NO: 95, YES: 105}`); the LLM runs used the sorted file. Same positional
   IDs, different underlying items — 95 shared IDs have conflicting gold.
   `compare_lextime.py` now refuses to compare runs in this state.

Separately, the pipeline's `pred: null` abstentions (`unresolved-mention`,
`no-date-no-edge`) were scored as *wrong* (`accuracy_all = correct / n`), which
is what produced 0.21. That conflates **coverage** with **correctness**.

## The fixes (in `run_external.py`)

- `--seed` (default `20240517`): deterministic shuffle of lextime/tracie
  **before** `--limit`, so `--system llm` and `--system tdg` draw the identical
  subset. `--seed 0` restores the old (broken) positional slice.
- Every `results.json` now carries a `meta` block: resolved `data_path`,
  `gold_distribution`, and a `split_sha256` over the `(id, gold, pair_type)`
  triples. Two runs meant to be compared must share the same hash.
- A single-class evaluation set now prints a loud warning and sets
  `meta.INVALID_SINGLE_CLASS`.
- `--tdg-fallback llm`: on a genuine abstention (not an infra error), back off to
  the raw LLM on the same prompt → a TDG-where-grounded / LLM-elsewhere hybrid,
  so `accuracy_all` is head-to-head comparable (forced-answer framing).
- `coverage` is now in every summary.

## Commands (run from `src/external_bench/`)

Run one model at a time — the pipeline warns that concurrent models trigger the
GGML/GPU-contention crash. Use `--limit 0` to score all 514 items (fully
balanced, no sampling noise); drop to `--limit 200` only if you need speed.

```bash
REPO=/path/to/thesis-personal-progress/code          # the --repo root
EP=http://localhost:11434/v1
SEED=20240517

# ---- llama3.1:8b -----------------------------------------------------------
# 1) LLM baseline, balanced full set
python run_external.py --dataset lextime --data lextime.csv --limit 0 --seed $SEED \
    --endpoint $EP --model llama3.1:8b \
    --out out/lextime_llm_llama_fair

# 2) TDG pipeline, SAME set (abstentions kept -> selective-prediction view)
python run_external.py --dataset lextime --data lextime.csv --limit 0 --seed $SEED \
    --system tdg --repo $REPO \
    --endpoint $EP --model llama3.1:8b \
    --out out/lextime_tdg_llama_fair

# 3) honest side-by-side (this is the number you report)
python compare_lextime.py \
    --llm out/lextime_llm_llama_fair \
    --tdg out/lextime_tdg_llama_fair
```

`compare_lextime.py` synthesizes the forced-answer hybrid from runs (1) and (2),
so you do **not** need a separate fallback run to see it. If you want the "real"
hybrid actually executed by the harness (identical result, but materialized in
`raw/`), add step 2b:

```bash
python run_external.py --dataset lextime --data lextime.csv --limit 0 --seed $SEED \
    --system tdg --tdg-fallback llm --repo $REPO \
    --endpoint $EP --model llama3.1:8b \
    --out out/lextime_hybrid_llama_fair
```

Repeat 1–3 with `--model gemma4:e4b` and `--out out/..._gemma_fair`.

## Reading the output

- **View 1** is raw coverage / accuracy_answered / accuracy_all per system.
- **View 2 (selective prediction)** is the honest framing for an abstaining
  system: "answers X% of items at Y% accuracy" vs the full-coverage LLM. In a
  legal setting, an abstention is not the same failure as a confident wrong
  answer — say so explicitly and cite the coverage.
- **View 3 (hybrid)** answers the question a reviewer will ask: *does the
  pipeline add anything over just prompting the LLM?* If
  `hybrid − LLM > 0`, the graph is contributing signal on the items it grounds.
- **Significance** block: McNemar exact p on the items both systems answered.
  Report it as a conditional comparison (that subset is where the TDG chose to
  commit, not a random sample).

## What is and isn't claimed

This does not guarantee the pipeline beats the LLM. It guarantees the comparison
is *valid* — same items, same gold, abstention accounted for honestly — so
whatever number comes out is defensible. The pipeline's main lever is coverage:
the 139 abstentions were mostly mention-resolution / missing-edge failures in
graph construction, which are fixable, and each abstention converted to a
committed answer (currently ~0.69 accurate) moves `accuracy_all` up steeply.

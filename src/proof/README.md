# Proof of concept: flat-text LLM reasoning vs. structure-then-compute

This package is a small, self-contained experiment that argues the thesis's
central claim on its own terms:

> An LLM reads a legal document as flat text. When the answer to a temporal
> question lives in a *connected chain* of dated events, flat-text reasoning
> degrades — while making the dependency structure explicit once and then
> computing the answer with calendar-correct arithmetic stays exact.

It is deliberately **not** about extraction quality. The benchmark documents
are written so plainly that reading the structure off them is trivial. That is
the point: it isolates the *reasoning* step so the comparison is about how each
approach handles the arithmetic and the cascade, not about who parsed the prose
better.

## What it compares

Three solvers answer the **same** items and are scored against the **same**
gold answers. Gold is pure calendar arithmetic (`dateutil.relativedelta`),
computed independently of every solver — so neither the engine nor the LLM can
define itself as correct.

| solver | what it is | reasoning uses an LLM? |
|---|---|---|
| `structured` | **the method.** Build the dependency graph once, then compute deterministically: the real `entailment.check_entailment` engine for deadlines, calendar-correct recomputation along additive edges for cascades. | no |
| `calendar_naive` | **honest ablation.** Identical explicit structure, but the 30-day-month / 365-day-year / day-delta approximation the project deliberately removed. Isolates the calendar-arithmetic component — not a strawman. | no |
| `baseline_llm` | **the "traditional best approach."** Hand the whole document(s) and the question to a strong LLM and ask for the answer as JSON. | yes |

## Two task types

* **deadline** — one additive hop: "the deadline is X; given the action date,
  was it timely?" Surface form of the offset is varied: `digit` ("within 90
  days"), `natural` ("within three months"), `vague` ("within a quarter",
  "a fortnight", "half a year", "a year and a day"). Anchors are chosen to
  stress the calendar (month-length differences, a leap-year boundary).
* **cascade** — the Ms. Chen ripple. Edit a root date; list the corrected
  downstream dates. Varies by chain length (`hops=2` vs `hops=3`) and locality
  (`single` document vs `cross`-document, where a second document restates the
  final event with a vague offset and must be reconciled by coreference).

## What it proves, and what it doesn't

* The `structured` column is **exact by construction**. Reporting it at 100% is
  not a result *about a model* — it is a demonstration that, once the structure
  is explicit, the answer is a calendar computation with no room for the failure
  modes the thesis is about. Do not present it as "our system beats GPT-x"; it
  is "the reasoning step, done right, is deterministic."
* The `calendar_naive` column shows the cost of the one component the project
  removed. It has the same structure as the method, so wherever it fails, the
  failure is *purely* the day-count approximation: month/year offsets and the
  leap-year boundary, plus any cascade hop carrying a month- or year-based edge.
  This is the honest answer to "did you just pick an easy baseline?"
* The `baseline_llm` column is the real measurement of the limitation. **It is
  not filled in here** — you run it on your machine with your model. The
  prediction (the thing the thesis claims) is that baseline accuracy is roughly
  fine on single-hop `digit` deadlines and **degrades** as you move to vague
  offsets, calendar-boundary anchors, longer chains, and cross-document cascades
  — exactly the cells where the calendar and the chain depth bite.

Do not fabricate the baseline numbers. The deliverable is the harness plus the
stated prediction; the numbers come from your run.

## How to run

From `code/src/`:

```bash
# method side only — runs anywhere, no API. Proves the engine is exact
# and validates it on the real UK tribunal TDGs already in the repo.
python proof/run_proof.py --real
```

```bash
# full three-way head-to-head with a local model via Ollama:
python proof/run_proof.py --real \
    --model gemma4:e4b --base-url http://localhost:11434/v1

# or with OpenAI (set OPENAI_API_KEY):
python proof/run_proof.py --real --model gpt-4o

# or any OpenAI-compatible endpoint (Claude via a gateway, etc.):
python proof/run_proof.py --real --model <name> --base-url <url>
```

`structured` and `calendar_naive` always run (no API). The `baseline_llm`
column appears only when you pass `--model`. Everything — the benchmark itself,
every per-item answer, per-bucket scores, traces, and the real-data validation
— is written to `../data/evaluation_results/proof_of_concept.json` for
inspection and reproducibility.

Flags: `--model`, `--base-url`, `--real`, `--minus-one-day` (apply the UK
"beginning with" −1-day convention to the synthetic deadline items; off by
default because they are phrased "within N of X"), `--output`.

## Reading the output

The report prints one row per difficulty bucket and an overall line, with a
column per solver. The expected shape, once you fill the baseline column, is:
`structured` flat at 100%; `calendar_naive` failing the calendar-sensitive
cells; `baseline_llm` strong on the easy bucket and falling off down the
difficulty axis. The `--real` block separately shows the engine reproducing two
real tribunals' verdicts from their actual Gemma-extracted TDGs, including the
deadline the tribunal itself computed — evidence the engine isn't only correct
on synthetic inputs.

## A real engine caveat (worth a sentence in the write-up)

The entailment engine's offset parser (`entailment._offset_from_text`) reads
single-unit offsets (`+3m`, `14d`, `1y`) and **space-separated** multi-unit
offsets (`+1y 1d`), but not compressed multi-unit forms (`+1y1d` / `P1Y1D`):
its word-boundary regex breaks on the digit-letter-digit run. The benchmark
emits the spaced form for the "year and a day" cases so they parse. This is a
genuine limitation of the current engine, not of the method — flag it as
future work rather than hiding it.

## Files

* `benchmark.py` — deterministic benchmark generator (26 items) + gold via
  `relativedelta` + gold-TDG construction helpers.
* `solvers.py` — the three solvers and the shared scorer.
* `run_proof.py` — runner: builds the benchmark, runs the solvers, buckets and
  prints the comparison, validates on real data, writes the JSON.

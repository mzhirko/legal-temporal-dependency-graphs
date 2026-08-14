# Temporal dependency graphs for legal deadline computation -- research artefacts

Code, rule packs and results for the MSc thesis *Temporal dependency
extraction and verified deadline computation for legal documents*
(Leiden University, 2026).

The thesis asks how a language model's answer can be made checkable in a
setting where mistakes cannot be undone. The system splits the task in two: a
model reads a document into a graph of dated facts, and a deterministic engine
computes over that graph, reports the reasoning behind every date, and stops
where a choice is a legal judgment rather than a reading.

**This repository holds the experiments.** The tool itself is released
separately as Timebar.

---

## What is here, and what is not

| | |
|---|---|
| **Here** | extraction and evaluation code, rule packs, generated Catala programs, scored results, prompts, annotation files |
| **Not here** | the tribunal judgments the experiments read, and any text extracted from them |

The judgments are Crown copyright, published by The National Archives through
the Find Case Law service. They are licensed material containing personal
data, and free text cannot be reliably de-identified, so no judgment text is
published here in any form. Every result file is keyed by neutral citation
(for example `2026_EAT_64_s111`), which retrieves the judgment on Find Case
Law, so each computed date remains checkable against the identified public
source. See `LICENSE-DATA.md`.

Two third-party corpora used in the evaluation are **not vendored**. Fetch
them from their own repositories:

- DeonticBench -- used in Section 5.5, its own licence applies
- TRACIE -- used as the non-legal control, Apache-2.0

---

## Layout

    src/                    experiment code
      tdg_pipeline/         extraction: rule-based and LLM
      catala_pipeline/      Catala generation, repair loop, input binding
      comparator/           field-by-field comparison of the two representations
      baseline/             the four-model direct baseline
      external_bench/       public benchmark runs
      proof/                the generated benchmark of Section 5.1
      anon_data/            pre-publication identifier gate (see below)
      counterfactual.py     the 427-item perturbation harness

    data/
      annotation/           the gold annotation, as CSV (mirror, see below)
      evaluation_results/   scored results per experiment
      experiments/          per-run outputs, including generated Catala
      statutes/             statute sections used as rule sources
      benchmarks/           LexTime input files

    experiments/            Catala worked examples and comparison outputs
    scripts/                figure and metric generation for the thesis
    docs/                   prompts, collected metrics, development notes

---

## Reproducing the results

Every number in the thesis is produced by script, not counted by hand.

    python scripts/collect_thesis_metrics.py      # regenerates docs/thesis_metrics.json

The runs that call a model need either a local Ollama endpoint or an
OpenAI-compatible one; the runs that do not call a model work offline from the
saved extractions. The counterfactual replays in Section 5.6 are entirely
offline: they rescore archived extractions and call nothing.

Results that depend on judgment text cannot be reproduced from this repository
alone. A user licensed by The National Archives can re-derive them by placing
the retrieved judgments where each script expects them; the paths are
documented in `src/baseline/README` and in the counterfactual harness.

---

## Prompts

Every prompt is a frozen artefact. `docs/prompts.txt` reproduces them as
templates, with the document body left as a `{judgment}` placeholder. Prompts
were not edited between runs, and temperature was 0 throughout.

---

## Before publishing anything derived from this

`src/anon_data/` is the gate that separates the working repository from
anything published. It treats two categories differently:

- **text** -- judgment text, prompt files with a document body, quoted
  sentences. Never published. Excluded rather than scrubbed, because free
  text cannot be reliably de-identified and is licensed material regardless.
- **metadata** -- results and graphs keyed by case. Published after the key is
  rewritten to a citation-only form.

The mapping file that reverses the pseudonyms is private and must stay
gitignored.

    python src/anon_data/scrub.py scan <tree>     # what identifiers are where
    python src/anon_data/scrub.py plan <tree>     # exclusion list and rekey map
    python src/anon_data/scrub.py rekey <tree> <out>

Run `scan` over the whole tree, including code and logs, before any release.

---

## Licences

- **Code** -- Apache-2.0, see `LICENSE`.
- **Data and documents** -- see `LICENSE-DATA.md`. Several different terms
  apply depending on the source, and one of them is a transactional licence
  that does not transfer to you.
- **Attribution** -- see `NOTICE`.

## Citing

See `CITATION.cff`.

## The gold set

The annotation in `data/ground_truth/` and `data/annotation/` is also released
on its own, with its own README, schema and licence terms, at
[legal-temporal-dependency-graphs-gold](https://github.com/legal-temporal-dependency-graphs-gold).
That repository is the canonical version and the one to cite if you use the
annotation rather than the code. The copy here exists so that this repository
runs after a clone without fetching anything, and the two are kept identical.

## Supervision

Supervised by Lifeng Han and Suzan Verberne, Leiden Institute of Advanced
Computer Science, Leiden University.

## A note on what is not here

There are no run logs in this repository. Every number the thesis reports
comes from a JSON result file. One family of runs that printed to stdout without writing JSON is the
contradiction detection of Use Case C, and those runs are reproducible from
`src/evaluate_contradictions.py` and the bundles in `data/synthetic/`.

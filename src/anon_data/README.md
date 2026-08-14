# `anon_data` — pre-publication tooling

Three scripts that sit between the working repository and anything published.
They exist because the corpus is licensed material containing personal data,
and the published artefacts must contain neither.

| script | role |
|---|---|
| `scrub.py` | finds identifiers across the tree; rewrites case keys |
| `build_anchors.py` | projects `FACTS.md` into a distributable anchor file |
| `verify_anchors.py` | reconstructs and checks anchors against Find Case Law |

Nothing here modifies the working data. `FACTS.md`, the corpus and the
annotation CSVs stay exactly as they are. Every script either reports, or
writes to a new location.

---

## Terminology

This tooling performs **de-identification**, not anonymisation.

Anchors are keyed on neutral citation (`2026_EAT_64`), which resolves to the
named case in one lookup. The data is therefore not anonymous and must not be
described as such in the thesis or in any artefact.

What the tooling does achieve: **no personal data is published by us**. What
goes out is pointers to already-public documents plus factual annotations
(dates, durations, in-time / out-of-time verdicts). Party names, first-instance
case numbers and EAT appeal numbers do not appear in any published file.

---

## Two categories

Identifier-bearing files are treated differently depending on what they are.

**TEXT** — judgment text. Corpus files, prompt files, raw model outputs that
quote judgments, and CSV columns such as `quoted_sentence`. These are
**excluded, never scrubbed**. Free text cannot be reliably de-identified, and
the text is licensed material regardless of whose name is in it. Detected by
path component: any directory named `caselaw`, `cases`, `prompts`, `raw`,
`inputs`, `txt`, `xml`, `pdf`.

**OPAQUE** — archives and binary stores (`.zip`, `.tar.*`, `.sqlite3`, `.db`,
`.pkl`, `.pyc`, `.npz`). Excluded outright. Their contents bypass every
path-based rule: `coherence.zip` in this repo contains the very prompt files
the TEXT rule excludes, surnames in the filenames and all. An archive of an
excluded directory would otherwise sail straight through the gate.

**METADATA** — results, manifests and graphs keyed by case id. These are
published, after the key is rewritten:

- Find Case Law appeals → drop the surname prefix.
  `2026_EAT_64_s111` → `2026_EAT_64_s111`
- First-instance tribunal cases → salted pseudonym, because the case number is
  itself an identifier tied to an individual.
  `et_b7d67901` → `et_8c30495b`

CSV columns named `quoted_sentence`, `quote`, `sentence`, `text` or `excerpt`
are replaced by a SHA-256 of their value and the column renamed `<col>_sha256`.
The annotation survives — concept, type, value, flags, comments — and each row
stays verifiable against the source, but the text itself does not travel. Note
that `data/annotation/` sits in a directory the path rule reads as METADATA, so
without this the CSVs would be published verbatim.

The salt and the reverse mapping are written to `rekey_map.PRIVATE.json`.
**That file is gitignored and must stay that way.** It is the only thing that
reverses the pseudonyms.

---

## Workflow

Run from the repository root.

```bash
# 1. what is where
python code/src/anon_data/scrub.py scan code

# 2. propose an exclusion list and a rekey map
python code/src/anon_data/scrub.py plan code

# 3. apply, into a fresh tree (refuses to overwrite)
python code/src/anon_data/scrub.py rekey code ../publish

# 4. the step that matters — the gate must pass on the OUTPUT
python code/src/anon_data/scrub.py scan ../publish
```

Step 4 is not optional. Passing on the input tells you nothing.

`scan` exits non-zero when it finds anything, so it can be wired into CI or a
pre-commit hook.

### Building the anchor file

```bash
python code/src/anon_data/build_anchors.py code anchors.json
```

Reads `data/ground_truth/FACTS.md`, the corpus, and the split manifests.
Emits citation-keyed anchors. Reports non-unique spans, spans that did not
land on their hinted line, and unrecognised field labels.

### Verifying

```bash
# against local copies
python code/src/anon_data/verify_anchors.py anchors.json --txt-dir code/data/caselaw

# against the live service (requires your own licence)
python code/src/anon_data/verify_anchors.py anchors.json --fetch
```

---

## What `anchors.json` contains

Per annotated span:

- neutral citation, Find Case Law URI and URL, publication date
- the field annotated (`anchor`, `deadline`, `acas_day_a`, `presented`,
  `verdict`, …) and whether the span is primary or supporting
- character offsets into the normalized text, plus the line for debugging
- `span_sha256`, and `prefix_sha256` / `suffix_sha256` over 64 chars of context
- `occurrence` / `n_occurrences`, for spans appearing more than once
- `doc_sha256` over the whole normalized document
- `normalizer`, pinning the extraction function by hash of its own source

It contains no judgment text. Hashes are one-way, so no span can be
reconstructed from the artefact.

## How verification works

Three tiers, per span:

1. **FAST** — the recorded offsets slice out a span whose hash matches.
2. **RELOCATED** — offsets missed, but the hash is found elsewhere in the
   document. Offsets are a fast path, not the source of truth: the verifier
   hashes every window of the recorded length until it finds the span. A
   judgment is ~50k characters, so this is instant.
3. **LOST** — the hash is absent. The judgment has been revised. Reported as a
   failure, never a silent pass.

Tier 2 is why upstream reformatting does not break the artefact. Tier 3 is how
the obligation to use the current version is satisfied structurally rather than
by undertaking: a withdrawn or revised judgment produces a visible failure.

Verified: 30/30 anchors resolve FAST against the annotated corpus. Under a
simulated 501-character upstream insertion, every affected span self-relocated
by exactly +501 and the changed document was flagged.

---

## `.gitignore`

```
code/data/caselaw/*/txt/
code/data/caselaw/*/xml/
code/data/caselaw/*/pdf/
code/data/cases/txt/
code/data/experiments/*/inputs/
code/data/experiments/*/prompts/
code/data/experiments/*/raw/
code/data/experiments/coherence/*/prompts/
code/data/experiments/coherence/*/raw/
rekey_map.PRIVATE.json
```

Run `scrub.py plan` to regenerate this list — it is derived from what is
actually in the tree, so it will change as the tree does.

---

## Known limitations

- **`kj_2026_EAT_46` survives rekeying.** The surname pattern requires three
  characters. In this instance that is arguably correct, since `KJ` is the
  court's own anonymisation rather than ours, but it is a gap in the rule and
  not a deliberate exemption.
- **The scanner over-reports.** `titled_name` and `v_caption` counts are
  inflated by DeonticBench, which is external and separately excluded. Triage
  the file list; do not read the totals as a risk score.
- **Offsets are normalizer-dependent.** They index the output of the pinned
  extraction function, not the raw XML. Changing that function invalidates
  every offset — hence the `normalizer` field, and the mismatch warning in the
  verifier. The span hashes survive; the offsets do not.
- **Prose is not rewritten, only keys.** Model outputs that quote a judgment
  back — `results_*.json`, `redaction_report.txt`, `sweep_triage.txt` — still
  carry appeal numbers and titled names after rekeying. `scan` reports them and
  exits non-zero; deciding whether to exclude, truncate or hash those files is
  a human call the tooling deliberately does not make.
- **Unscanned extensions are copied.** `rekey` warns which ones. `.xhtml`
  (legislation), `.pdf` and `.xlsx` in this tree are non-FCL material, but the
  warning exists because that will not always be true.
- **`data/cases/` is not Find Case Law material.** Those first-instance
  decisions come from the gov.uk Employment Tribunal decisions database, a
  different service under different terms. They are excluded here, but their
  licensing position is a separate question and is not resolved by this
  tooling.

---

## Attribution

Any published artefact must carry, prominently:

> Crown copyright material reproduced by permission of The National Archives.
> The contents of the judgment can be used under the Open Justice Licence.

and a statement that the material only partially represents the activities of
the courts and tribunals. Exact wording to be confirmed with the Licensing &
Publishing Department; do not publish before that confirmation.

Anyone running `verify_anchors.py --fetch` is performing computational analysis
on Find Case Law and requires their own licence. State this wherever the
artefact is distributed.

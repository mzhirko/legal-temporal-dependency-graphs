# Baseline v2.1 (audited) — cross-model failure matrix (4 models, frozen prompts, temp 0)

Models (pinned before any call): gemma4:e4b (local 4B-class), gpt-5.4-nano,
gpt-5.4-mini, gpt-5.4. Identical prompts; raw responses archived in
data/experiments/baseline/raw/.

## Headline table
| model        | verdicts | deadlines exact | mean |dd| |
|--------------|----------|-----------------|-----------|
| gemma4:e4b   | 1/7      | 0/6             | 87.2      |
| gpt-5.4-nano | 3/7      | 0/6             | 77.3      |
| gpt-5.4-mini | 1/7      | 0/6             | 99.0      |
| gpt-5.4      | 3/7      | 3/6             | 50.5      |

Non-monotonic verdicts across scale; the meaningful monotone signal is
deadline exactness (0,0,0 -> 3) and, more importantly, WHICH classes
persist (below).

## Finding 1 — reasoning->answer divergence at the frontier (VERIFY IN RAW)
gpt-5.4, both 2026_EAT_64 rows: the model's own `arithmetic` field is fully
correct — anchor right, "-1 day" right ("deadline = 2022-09-07"), s.207B
precondition right ("Day A was after the primary deadline, so the 'one
month after Day B' rule does not apply"), and it CONCLUDES "Verdict: out
of time" — while the JSON `verdict` field says in_time with an
effective_deadline (2022-12-18) that appears nowhere in the working.
= scope laundering (vocabulary: arXiv 2606.16118), observed on real
post-cutoff cases. Verdict accuracy scores this wrong-for-the-wrong-reason;
value accuracy misses that the computation was right. Only working-vs-
output comparison detects it. PENDING: confirm in
raw/2026_EAT_64_s111.gpt-5.4.txt that the contradiction is inside the
model's single emitted JSON object (parser exclusion).

## Finding 2 — verdict leakage via appeal direction (methods disclosure)
The kept (by frozen protocol) summary contention in kj — "the British
Council cross appealed ... arguing that the sexual harassment claim was
out of time" — lets a model infer the ET verdict (one appeals what one
lost). mini and nano exploited it: both computed out_of_time in their own
arithmetic, then wrote "the judgment indicates the claim was treated as in
time. Therefore ... in time." Consequences, stated honestly:
- verdict metric: contaminated on kj (weakly 2026_EAT_59) for models that
  defer; appeal DIRECTION is structurally unremovable from EAT judgments.
- delta_days metric: UNAFFECTED — no deadline value survived redaction
  (leak-check held); mini/nano kj deadline deltas are -240/-76.
- new taxonomy row: DEFERENCE OVERRIDE — model abandons its own
  computation to agree with the document's implied authority.

## Finding 3 — class persistence across scale
| class (from shown working)              | gemma | nano | mini | gpt-5.4 |
|-----------------------------------------|-------|------|------|---------|
| C1 -1d convention applied (AUDITED)     | 0/7   | 0/7  | 1/7  | 7/7     |
| C2 s.207B pause/precondition mishandled | yes   | yes  | yes  | no      |
| C3 fabricated Day B                     | yes   | no   | no   | no      |
| C4 assuming the conclusion              | yes   | yes  | yes  | no      |
| C5 rejected-contention anchor (2026_EAT_14)| yes   | yes  | yes  | NO — explicitly rejected: "appeal outcome ... does not change the EDT" |
| C6 Gisda Cyf anchor (2025_EAT_155)          | miss  | +2d  | miss | EXACT (2020-07-11, 0-delta row) |
| C7 continuing-act scoping (kj)          | -206  | -206 | -206 | -206    |
| C8 reasoning->answer divergence         | no    | yes (2026_EAT_59, kj) | yes (2026_EAT_64 s111, kj) | yes (2026_EAT_64 x2) |
C1 and C8 cells are machine-audited (src/baseline/audit_findings.py), not
hand-counted; mini's single C1 pass is 2025_EAT_155. nano/mini sometimes STATE
the convention in the working without applying it in the emitted value
(mini 2026_EAT_76: working reaches "2018-07-11", emits 2018-08-12).
C8 hard-divergence rates (field contradicts working's final conclusion):
gemma 0/7, nano 2/7, mini 2/7, gpt-5.4 2/7 — 6/21 across the GPT family,
0/7 local, ALL flipping the same direction (working: out_of_time ->
field: in_time). Heuristic counts hard contradictions only; kj deference
cases are classified separately (Finding 2).

Reading: arithmetic conventions (C1-C6) are largely SOLVED at the frontier;
what persists at every scale is (C7) a legal-judgment divergence and (C8)
unfaithful reporting. C7 framing must be fair: the EAT called the
tribunal's scoping "clearly open to it" — i.e., the models' contrary view
was arguable; classify as judgment divergence, not arithmetic error. All
four models independently chose the same (respondent-side) scoping.

## Exhibits to carry into the chapter verbatim
- gpt-5.4 2025_EAT_155: perfect row on the hardest anchor problem — knowledge
  date 2020-07-11 via the tribunal's reading-rule facts, 0-delta deadline,
  correct 207B non-engagement. The frontier CAN do this; the failures
  elsewhere are therefore not capability absence.
- gpt-5.4 2026_EAT_64: correct computation, contradicted verdict field (Finding 1).
- mini 2026_EAT_76: self-correcting working that lands on the right date and
  emits a different one — value/working divergence at small scale.
- gemma 2025_EAT_155: compensating errors -> correct primary deadline (v1).

## Consequence for the thesis narrative
The baseline result is NOT "LLMs cannot compute deadlines." It is sharper:
(i) small/local models fail on conventions and rule mechanics; (ii) the
frontier model computes correctly but reports unfaithfully often enough
that its outputs cannot be trusted without verifying the working; (iii) a
deterministic verified layer eliminates class C8 by construction — the
engine cannot report a verdict its computation did not produce — which is
precisely the provenance/verification contribution branch anticipated in
the notes. C7 (anchor/scoping judgment) remains open for ALL systems,
including the pipeline, whose gold anchor is judge-found: state this as
the shared boundary of automation.

# Evaluation results

Three claims, in ascending order of external validity. Each has its own testbed; none is asked to carry the others.

### Claim 1 — same task, same 210 items, same gold

| method | fully correct | by task: deadline | cascade |
|---|:-:|:-:|:-:|
| STRUCTURED (yours) | 100.0% | 100.0% | 100.0% |
| gemma4 (LLM) | 78.1% | 56.7% | 94.2% |
| calendar-naive | 41.4% | 62.2% | 25.8% |
| llama3.1 (LLM) | 20.0% | 42.2% | 3.3% |

*Synthetic, templated. Clean gold and controlled difficulty; external validity comes from Claim 2 (real judgments).*

### Claim 2 — real judgments, engine on verified facts

- verdicts reproduced: **6/7**
- judge-stated deadlines to the day: **2/2**
- judge-stated boundary: **1/1**
- rule [era_1996_s111]: discovered

End-to-end (extraction -> engine), 8 conditions: **12/14** correct, 14 abstentions.
*Matching is the bottleneck, not the engine; tuned on the 7, so this is a mitigation not a held-out result.*

### Claim 3 — 427 perturbed real judgments, boundary tracking

The engine sits on every boundary exactly (0-day error, by construction — the sweep self-check verifies monotonicity with one flip per case). The LLM baseline, same documents:

| baseline | verdict acc | mean boundary error | never-flips |
|---|:-:|:-:|:-:|
| anchor gemma | 59.0% | 4.7d | 1/7 |
| anchor llama | 58.8% | 1.1d | 0/7 |

*The engine's 0-day error is by construction, so this is not an accuracy contest — it is evidence that the LLM cannot reliably locate a statutory boundary while the structured method cannot miss it.*

### Supporting — LLMs alone are unreliable (external benchmarks)

| benchmark | model | metric | vs majority |
|---|---|:-:|:-:|
| airline | gemma4 | 1/65 exact | - |
| airline | llama3.1 | 0/75 exact | - |
| housing | gemma4 | 0.256 | below |
| housing | llama3.1 | 0.282 | below |
| uscis | gemma4 | 0.536 | above |
| uscis | llama3.1 | 0.464 | below |
| sara_numeric | gemma4 | 2/35 exact | - |
| sara_numeric | llama3.1 | 0/33 exact | - |
| sara_binary | gemma4 | 0.600 | above |
| sara_binary | llama3.1 | 0.433 | below |

*LLM-only baselines. exact-match is the honest metric for numeric tasks. These establish the problem your method addresses; they are not tests of your method.*

---
The honest one-paragraph summary: the structured method computes statutory deadlines correctly where LLMs do not — 100% vs 20-78% on 210 controlled items, 6/7 real tribunal verdicts, and exact boundary tracking across 427 perturbed judgments where the LLM drifts or fails to flip. It does NOT beat LLMs on pairwise ordering (LexTime) or one-shot statute QA (DeonticBench), because those are not the task; on its own task — deadline computation — it wins. The open problem is the extraction/matching layer, which bounds end-to-end coverage.

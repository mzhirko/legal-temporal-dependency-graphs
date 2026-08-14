Model list pinned 2026-07-06, before any API call; all results will be reported:
gemma4:e4b (local), gpt-5.4, gpt-5.4-mini, gpt-5.4-nano. Temperature 0 throughout.
Extraction 2x2 pinned 2026-07-06 before runs: extractor {gemma4:e4b, gpt-5.4-mini}
x input {redacted, fulltext}. Same engine, same thresholds, same fallback across
all cells; all results reported. Rationale: redacted condition provably blocks
2025_EAT_155 (gold anchor was a redacted finding); extractor swap isolates whether
the bottleneck is model capability or pipeline architecture.
Extraction 2x2 pinned 2026-07-06 pre-run: extractor {gemma4:e4b, gpt-5.4-mini}
x input {redacted, fulltext}. Same engine/thresholds/fallback in all cells;
all results reported. gemma served with OLLAMA_CONTEXT_LENGTH=32768 (largest
input ~18K tokens, no truncation). v1 gemma/redacted run (era/, eqa/) kept
as committed history; superseded by the gemma_redacted cell.

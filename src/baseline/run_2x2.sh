#!/usr/bin/env bash
# run_2x2.sh — extraction 2x2: extractor {gemma4:e4b, gpt-5.4-mini}
#              x input {redacted, fulltext}; then entailment per statute.
# Run from repo root inside tmux:  bash src/baseline/run_2x2.sh
set -u

# ---------- 0. pre-flight ----------
if ! curl -s http://localhost:11500/api/tags | grep -q gemma4; then
  echo "Ollama on :11500 not serving gemma4 — start it first:"
  echo '  pkill -f "ollama serve"'
  echo '  OLLAMA_HOST=127.0.0.1:11500 CUDA_VISIBLE_DEVICES=0 \'
  echo '    OLLAMA_MODELS=<path> OLLAMA_CONTEXT_LENGTH=32768 \'
  echo '    ollama serve > /tmp/ollama_11500.log 2>&1 &'
  exit 1
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "export OPENAI_API_KEY=sk-...  first"; exit 1
fi

# ---------- 1. case maps ----------
RIN=data/experiments/baseline/inputs
declare -A SRC=( \
  [2026_EAT_64_s111]=data/caselaw/gold_eqa_s123/txt/2026_EAT_64.txt \
  [2026_EAT_64_s123]=data/caselaw/gold_eqa_s123/txt/2026_EAT_64.txt \
  [2025_EAT_155]=data/caselaw/gold_era_s111/txt/2025_EAT_155.txt \
  [2026_EAT_14]=data/caselaw/gold_era_s111/txt/2026_EAT_14.txt \
  [2026_EAT_59]=data/caselaw/gold_era_s111/txt/2026_EAT_59.txt \
  [2026_EAT_76]=data/caselaw/gold_eqa_s123/txt/2026_EAT_76.txt \
  [kj_2026_EAT_46]=data/caselaw/gold_eqa_s123/txt/2026_EAT_46.txt )
declare -A RED=( [2026_EAT_64_s111]=2026_EAT_64 [2026_EAT_64_s123]=2026_EAT_64 \
  [2025_EAT_155]=2025_EAT_155 [2026_EAT_14]=2026_EAT_14 \
  [2026_EAT_59]=2026_EAT_59 [2026_EAT_76]=2026_EAT_76 \
  [kj_2026_EAT_46]=2026_EAT_46 )
declare -A SUB=( [2026_EAT_64_s111]=era [2025_EAT_155]=era \
  [2026_EAT_14]=era [2026_EAT_59]=era \
  [2026_EAT_64_s123]=eqa [2026_EAT_76]=eqa [kj_2026_EAT_46]=eqa )

# ---------- 2. extraction: 4 conditions x 7 cases (skips cells already done) ----------
for cond in gemma_redacted gemma_fulltext mini_redacted mini_fulltext; do
  for cid in "${!SRC[@]}"; do
    sub=${SUB[$cid]}
    out="data/results_pipeline/$cond/$sub/$cid.json"
    [ -s "$out" ] && { echo "skip (exists): $out"; continue; }
    mkdir -p "data/results_pipeline/$cond/$sub"
    case $cond in
      *redacted) file="$RIN/${RED[$cid]}.txt" ;;
      *)         file="${SRC[$cid]}" ;;
    esac
    case $cond in
      gemma*) MODELARGS="--base-url http://localhost:11500/v1 --model gemma4:e4b" ;;
      *)      MODELARGS="--model gpt-5.4-mini" ;;
    esac
    echo "=== $cond / $cid <- $file"
    python src/demo_llm.py --file "$file" --document-id "$cid" --raw \
      --max-chars 80000 $MODELARGS --no-scenarios --output "$out" \
      || echo "FAILED: $cond/$cid (rerun this cell individually)"
  done
done

# ---------- 3. verify all 28 cells before entailment ----------
python3 - <<'EOF'
import json, glob, os, sys
expected = ['2026_EAT_64_s111','2025_EAT_155','2026_EAT_14',
            '2026_EAT_59','2026_EAT_64_s123','2026_EAT_76','kj_2026_EAT_46']
bad = 0
for cond in ['gemma_redacted','gemma_fulltext','mini_redacted','mini_fulltext']:
    found = {}
    for f in glob.glob(f'data/results_pipeline/{cond}/*/*.json'):
        try:
            j = json.load(open(f)); found[j.get('document_id')] = len(j.get('facts', []))
        except Exception as e:
            found[os.path.basename(f)] = f'INVALID: {e}'
    for cid in expected:
        v = found.get(cid, 'NOT RUN / FAILED')
        ok = isinstance(v, int) and v > 0
        bad += 0 if ok else 1
        print(f"{cond:16s} {cid:26s} facts={v}{'' if ok else '   <-- rerun this cell'}")
print(f"\n{'ALL 28 CELLS OK' if bad == 0 else str(bad) + ' cells need rerunning — fix before entailment'}")
sys.exit(0 if bad == 0 else 1)
EOF
[ $? -ne 0 ] && { echo "Stopping before entailment."; exit 1; }

# ---------- 4. entailment: 4 conditions x 2 statutes ----------
mkdir -p data/evaluation_results
for cond in gemma_redacted gemma_fulltext mini_redacted mini_fulltext; do
  python src/run_entailment.py --statute-tdg data/results_uk/era_1996_s111.json \
    --cases-dir "data/results_pipeline/$cond/era" \
    --ground-truth data/ground_truth/ground_truth_gold.json \
    --output "data/evaluation_results/full_${cond}_era.json"
  python src/run_entailment.py --statute-tdg data/results_uk/eqa_2010_s123.json \
    --cases-dir "data/results_pipeline/$cond/eqa" \
    --ground-truth data/ground_truth/ground_truth_gold.json \
    --output "data/evaluation_results/full_${cond}_eqa.json"
done

echo "Done. Review the 8 tables above, then:"
echo '  git add data/results_pipeline data/evaluation_results data/experiments/baseline/README.md'
echo '  git commit -m "pipeline 2x2: extractor {gemma,gpt-5.4-mini} x input {redacted,fulltext}, ctx=32768"'

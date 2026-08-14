#!/usr/bin/env bash
# Clean up unneeded runs and (re)run the LexTime experiments on a valid,
# balanced, aligned split. Run this from src/external_bench/.
#
#   bash run_experiments.sh
#
# Old/unneeded runs are ARCHIVED (moved), not deleted, so nothing is lost.
# TRACIE is dropped entirely (it doesn't test date-ordering, so it's not
# meaningful for a temporal-reasoning report).
set -euo pipefail
cd "$(dirname "$0")"

# ---- config (override by exporting before you run) -------------------------
REPO="${REPO:-$(cd ../.. && pwd)}"                 # thesis .../code root
EP="${EP:-http://localhost:11434/v1}"              # Ollama endpoint
DATA="${DATA:-../../lextime.csv}"                  # official Entailment_Dataset.csv
SEED="${SEED:-20240517}"
LIMIT="${LIMIT:-0}"                                # 0 = all 514 items (balanced)

test -f "$DATA" || { echo "ERROR: LexTime CSV not found at $DATA
Set it: DATA=/path/to/lextime.csv bash run_experiments.sh"; exit 1; }
echo "REPO=$REPO"; echo "EP=$EP"; echo "DATA=$DATA"; echo "SEED=$SEED LIMIT=$LIMIT"

# ---- 1. archive folders you no longer need (reversible) --------------------
ARCH="out/_archive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCH"
for d in \
  tracie_llm_llama tracie_llm_gemma smoke_tracie_llm smoke_tracie_tdg \
  smoke_sara smoke_sara_llama \
  lextime_llm_llama lextime_llm_gemma lextime_llm_llama_v3 \
  lextime_tdg_gemma lextime_tdg_llama_v2 lextime_tdg_llama_v4 ; do
  if [ -e "out/$d" ]; then mv "out/$d" "$ARCH/"; echo "  archived out/$d"; fi
done
echo "Old/unneeded runs moved to $ARCH"
echo "(when you're happy with the new results, delete them with:  rm -rf $ARCH )"
echo

# ---- 2. LexTime runs — ONE MODEL AT A TIME (concurrent models crash GGML) --
run () {  # run <model> <extra-flags...>
  local model="$1"; shift
  local tag="${model%%:*}"                         # llama3.1 / gemma4
  echo ">>> [$model] LLM baseline"
  python run_external.py --dataset lextime --data "$DATA" --limit "$LIMIT" --seed "$SEED" \
      --endpoint "$EP" --model "$model" "$@" \
      --out "out/lextime_llm_${tag}_fair"
  echo ">>> [$model] TDG pipeline"
  python run_external.py --dataset lextime --data "$DATA" --limit "$LIMIT" --seed "$SEED" \
      --system tdg --repo "$REPO" \
      --endpoint "$EP" --model "$model" "$@" \
      --out "out/lextime_tdg_${tag}_fair"
}

run llama3.1:8b
run gemma4:e4b --native --num-batch 128            # crash-avoidance for Gemma

# ---- 3. honest side-by-side (the numbers you report) -----------------------
echo; echo "############################## LLAMA ##############################"
python compare_lextime.py \
    --llm out/lextime_llm_llama3.1_fair \
    --tdg out/lextime_tdg_llama3.1_fair
echo; echo "############################## GEMMA ##############################"
python compare_lextime.py \
    --llm out/lextime_llm_gemma4_fair \
    --tdg out/lextime_tdg_gemma4_fair

echo; echo "Done. Results in out/lextime_*_fair/results.json"
echo "If a Gemma TDG run shows many 'empty-graph' / 'call-error' notes, the"
echo "extractor is still crashing — that's infra, not model quality; rerun it"
echo "alone with Ollama freshly restarted before trusting the number."

#!/usr/bin/env bash
# Run identical GRPO per cohort, 4 cohorts in parallel (one per GB200), in waves.
# Cohort = the only variable. Each GPU trains one cohort to completion, then the
# next wave starts. Designed to be launched DETACHED in tmux under the GPU lease:
#
#   ssh devbox 'cd ~/hackathon/lad && bash ~/hackathon/gpu_lease.sh run LAD -- \
#       tmux new -d -s lad "bash scripts/run_waves.sh data/run results/lifts 150 2>&1 | tee results/waves.log"'
#
# Then poll results/waves.log and results/lifts/*.json.
set -uo pipefail

DATADIR="${1:-data/run}"
OUTDIR="${2:-results/lifts}"
STEPS="${3:-150}"
SEEDS="${4:-0 1}"        # seeds per cohort (averaged)
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MAXLEN="${MAXLEN:-320}"          # GSM8K answers terminate ~100 tok; 320 is ample, much faster than 512
EVAL_TOK="${EVAL_TOK:-320}"

mkdir -p "$OUTDIR"
source .venv/bin/activate 2>/dev/null || true
EVAL="$DATADIR/eval_set.json"

# --- Compute acc_before ONCE (identical across all cohorts: same base model,
#     same fixed eval set) and reuse it for every run, saving ~26 base evals. ---
ACCB_FILE="$OUTDIR/acc_before.json"
if [ ! -f "$ACCB_FILE" ]; then
  echo "[waves] computing base acc_before ONCE on GPU0..."
  CUDA_VISIBLE_DEVICES=0 python -m lad.eval_base \
    --model "$MODEL" --eval_tasks "$EVAL" --out "$ACCB_FILE" \
    --eval_max_tokens "$EVAL_TOK" > "$OUTDIR/log_acc_before.txt" 2>&1 \
    || { echo "[waves] base eval failed; runs will compute acc_before themselves"; }
fi
ACCB=""
if [ -f "$ACCB_FILE" ]; then
  ACCB=$(python -c "import json;print(json.load(open('$ACCB_FILE'))['acc_before'])" 2>/dev/null)
  echo "[waves] acc_before=$ACCB (reused by all cohorts)"
fi

# Build the (cohort, seed) job list.
JOBS=()
for cohort in "$DATADIR"/cohorts/*.json; do
  name=$(basename "$cohort" .json)
  for seed in $SEEDS; do
    out="$OUTDIR/lift_${name}_seed${seed}.json"
    [ -f "$out" ] && { echo "[skip] $out exists"; continue; }
    JOBS+=("$name|$seed|$cohort|$out")
  done
done
echo "[waves] ${#JOBS[@]} jobs, 4 GPUs in parallel, steps=$STEPS maxlen=$MAXLEN"

run_one() {
  local gpu="$1" name="$2" seed="$3" cohort="$4" out="$5"
  echo "[gpu$gpu] START $name seed$seed -> $out"
  local accb_arg=()
  [ -n "$ACCB" ] && accb_arg=(--acc_before "$ACCB")
  CUDA_VISIBLE_DEVICES="$gpu" python -m lad.grpo_train \
    --cohort "$cohort" --eval_tasks "$EVAL" --out "$out" \
    --model "$MODEL" --steps "$STEPS" --seed "$seed" \
    --max_completion_len "$MAXLEN" --eval_max_tokens "$EVAL_TOK" \
    "${accb_arg[@]}" \
    > "$OUTDIR/log_${name}_seed${seed}.txt" 2>&1
  echo "[gpu$gpu] DONE  $name seed$seed (exit $?)"
}

i=0
N=${#JOBS[@]}
while [ $i -lt $N ]; do
  pids=()
  for gpu in 0 1 2 3; do
    [ $i -ge $N ] && break
    IFS='|' read -r name seed cohort out <<< "${JOBS[$i]}"
    run_one "$gpu" "$name" "$seed" "$cohort" "$out" &
    pids+=($!)
    i=$((i+1))
  done
  # wait for this wave to finish before starting the next
  for p in "${pids[@]}"; do wait "$p"; done
  echo "[waves] wave complete; $i/$N jobs done"
done
echo "[waves] ALL DONE"

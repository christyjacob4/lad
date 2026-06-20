#!/usr/bin/env bash
# Run identical GRPO per cohort, 4 cohorts in parallel (one per GB200), in waves.
# Cohort = the only variable. Each GPU trains one cohort to completion, then the
# next wave starts. Designed to be launched DETACHED in tmux under the GPU lease:
#
#   ssh devbox 'cd ~/hackathon/lad && bash ~/hackathon/gpu_lease.sh run LAD -- \
#       tmux new -d -s lad "bash scripts/run_waves.sh data/run results/lifts 200 2>&1 | tee results/waves.log"'
#
# Then poll results/waves.log and results/lifts/*.json.
set -uo pipefail

DATADIR="${1:-data/run}"
OUTDIR="${2:-results/lifts}"
STEPS="${3:-200}"
SEEDS="${4:-0 1}"        # seeds per cohort (averaged)
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

mkdir -p "$OUTDIR"
source .venv/bin/activate 2>/dev/null || true
EVAL="$DATADIR/eval_set.json"

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
echo "[waves] ${#JOBS[@]} jobs, 4 GPUs in parallel"

run_one() {
  local gpu="$1" name="$2" seed="$3" cohort="$4" out="$5"
  echo "[gpu$gpu] START $name seed$seed -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" python -m lad.grpo_train \
    --cohort "$cohort" --eval_tasks "$EVAL" --out "$out" \
    --model "$MODEL" --steps "$STEPS" --seed "$seed" \
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

#!/usr/bin/env bash
# Probe 8-way (2 jobs/GPU) memory safety at the CHOSEN GRPO config on GPU0, then
# launch the ordered orchestrator with PACK=8 (safe) or PACK=4 (fallback) DETACHED
# under the lease in tmux, plus the durable finisher tmux.
set -uo pipefail
cd ~/hackathon/lad
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/scripts:${PYTHONPATH:-}"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false

STEPS="${STEPS:-70}"
MAXLEN="${MAXLEN:-320}"
COHORT_DIR=data/run/cohorts
PROBE_COH=$(ls "$COHORT_DIR"/*.json | head -1)
mkdir -p results/probe

echo "[probe] launching 2 identical GRPO jobs on GPU0 (STEPS=$STEPS) to test 2/GPU packing..."
ACCB=$(python -c "import json;print(json.load(open('results/lifts/acc_before.json'))['acc_before'])" 2>/dev/null || echo "")
ACCB_ARG=(); [ -n "$ACCB" ] && ACCB_ARG=(--acc_before "$ACCB")
for j in 1 2; do
  CUDA_VISIBLE_DEVICES=0 python -m lad.grpo_train \
    --cohort "$PROBE_COH" --eval_tasks data/run/eval_set.json \
    --out "results/probe/probe_${j}.json" --steps 12 --seed "$j" \
    --max_completion_len "$MAXLEN" --eval_max_tokens "$MAXLEN" "${ACCB_ARG[@]}" \
    > "results/probe/probe_${j}.log" 2>&1 &
done
# watch GPU0 peak memory for ~3 min while both train
PEAK=0; OOM=0
for t in $(seq 1 36); do
  sleep 5
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)
  [ -n "$m" ] && [ "$m" -gt "$PEAK" ] 2>/dev/null && PEAK=$m
  if grep -qi "out of memory\|CUDA error\|OutOfMemory" results/probe/probe_*.log 2>/dev/null; then OOM=1; break; fi
  # both finished early?
  if [ -f results/probe/probe_1.json ] && [ -f results/probe/probe_2.json ]; then break; fi
done
wait 2>/dev/null
echo "[probe] peak GPU0 mem during 2-job test: ${PEAK} MiB ; OOM=$OOM"

PACK=8
if [ "$OOM" = "1" ] || [ "$PEAK" -gt 175000 ] 2>/dev/null; then
  PACK=4
  echo "[probe] UNSAFE for 2/GPU (peak ${PEAK}MiB or OOM) -> fallback PACK=4"
else
  echo "[probe] SAFE for 2/GPU (peak ${PEAK}MiB < 175000) -> PACK=8"
fi
echo "$PACK" > results/PACK.txt
echo "$PACK" > results/PROBE_RESULT.txt
echo "[probe] PACK=$PACK chosen"

# Launch orchestrator DETACHED under the lease in tmux. The lease wraps the
# FOREGROUND tmux-attached process so the lease is held for the whole run.
tmux kill-session -t lad-paper 2>/dev/null || true
tmux new -d -s lad-paper "cd ~/hackathon/lad && bash ~/hackathon/gpu_lease.sh run LAD -- env PACK=$PACK STEPS=$STEPS bash scripts/run_ordered.sh 2>&1 | tee results/ordered.log"
echo "[probe] launched orchestrator tmux 'lad-paper' (PACK=$PACK STEPS=$STEPS)"

# Launch the durable finisher in its OWN tmux (safety net: partial PAPER + final marker).
tmux kill-session -t lad-finish 2>/dev/null || true
tmux new -d -s lad-finish "cd ~/hackathon/lad && MIN_LIFTS=8 bash scripts/finish_paper.sh 2>&1 | tee results/finish.log"
echo "[probe] launched durable finisher tmux 'lad-finish'"
tmux ls

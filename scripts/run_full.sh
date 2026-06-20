#!/usr/bin/env bash
# Self-contained FULL sweep body, run in the FOREGROUND under the GPU lease so the
# lease is held for the entire duration (precompute + all GRPO waves). Launch it
# detached on the devbox like:
#
#   tmux new -d -s lad "cd ~/hackathon/lad && bash ~/hackathon/gpu_lease.sh run LAD \
#       -- bash scripts/run_full.sh 2>&1 | tee results/full_run.log"
#
# Then poll results/full_run.log, results/waves.log, results/lifts/*.json.
set -uo pipefail
cd ~/hackathon/lad
source .venv/bin/activate

POOL="${POOL:-2500}"
COHORT="${COHORT:-256}"
EVAL="${EVAL:-500}"
K="${K:-8}"
STEPS="${STEPS:-150}"
SEEDS="${SEEDS:-0 1}"

echo "[full_run] $(date -Is) starting; pool=$POOL cohort=$COHORT eval=$EVAL k=$K steps=$STEPS"

# 1. Precompute (cheap, training-free) — once.
if [ ! -f data/run/cohort_meta.json ]; then
  echo "[full_run] precompute (base rollouts + embeddings + cohorts)..."
  python scripts/precompute.py --outdir data/run --pool_size "$POOL" \
    --cohort_size "$COHORT" --eval_size "$EVAL" --k "$K" 2>&1 | tail -40
else
  echo "[full_run] precompute artifacts already exist; skipping."
fi

# 2. GRPO waves — 4 cohorts/GPU in parallel, all 4 GB200 saturated.
echo "[full_run] launching GRPO waves..."
bash scripts/run_waves.sh data/run results/lifts "$STEPS" "$SEEDS" 2>&1 | tee results/waves.log

echo "[full_run] $(date -Is) ALL DONE"

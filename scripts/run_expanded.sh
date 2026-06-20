#!/usr/bin/env bash
# EXPANDED paper-grade sweep, FOREGROUND under the GPU lease (so the lease is held
# for the whole run). Launch detached in tmux UNDER THE LEASE:
#
#   bash ~/hackathon/gpu_lease.sh run LAD -- \
#     tmux new -d -s lad-paper "cd ~/hackathon/lad && bash scripts/run_expanded.sh 2>&1 | tee results/expanded.log"
#
# Stages (all 4 GB200 saturated, 4 jobs/wave):
#   1. precompute  : k=32 base rollouts (enables the free reliability sweep) +
#                    embeddings + ALL cohort families  (once)
#   2. main waves  : identical GRPO per cohort, 2 seeds, WITH mechanistic logging
#   3. causal sel  : select top/bottom/random/easy/hard/var/div/passrate cohorts
#                    + dose-response, then identical GRPO on each (2 seeds)
# The finisher (finish_paper.sh) runs the no-GPU analysis + figures + PAPER fill and
# writes results/PAPER_FINAL_MARKER.txt; launch it in its OWN durable tmux.
set -uo pipefail
cd ~/hackathon/lad
source .venv/bin/activate 2>/dev/null || true

POOL="${POOL:-3000}"
COHORT="${COHORT:-256}"
EVAL="${EVAL:-500}"
K="${K:-32}"                 # k=32 -> reliability sweep is free via subsampling
STEPS="${STEPS:-150}"
SEEDS="${SEEDS:-0 1}"
CAUSAL_SEEDS="${CAUSAL_SEEDS:-0 1}"
DO_CAUSAL="${DO_CAUSAL:-1}"

echo "[expanded] $(date -Is) pool=$POOL cohort=$COHORT k=$K steps=$STEPS seeds='$SEEDS'"

# 1. precompute (cheap, training-free) -- once.
if [ ! -f data/run/cohort_meta.json ]; then
  echo "[expanded] precompute (base rollouts k=$K + embeddings + cohorts)..."
  python scripts/precompute.py --outdir data/run --pool_size "$POOL" \
    --cohort_size "$COHORT" --eval_size "$EVAL" --k "$K" 2>&1 | tail -50
else
  echo "[expanded] precompute artifacts present; skipping."
fi

# 2. main GRPO waves WITH mechanistic logging (Claim 1 + 2 + 3 come from here).
echo "[expanded] main waves (mechanistic logging on)..."
MECH_DIR="$PWD/results/mech" \
  bash scripts/run_waves.sh data/run results/lifts "$STEPS" "$SEEDS" 2>&1 | tee results/waves.log

# 3. causal selection experiment (Claim 4 -- the GPU headline).
if [ "$DO_CAUSAL" = "1" ]; then
  echo "[expanded] building causal selection cohorts..."
  python scripts/causal_select.py --datadir data/run --outdir data/causal \
    --size "$COHORT" --with_dose 2>&1 | tail -30
  echo "[expanded] causal GRPO waves..."
  # reuse the identical-GRPO machinery; no mech logging needed for causal lifts
  bash scripts/run_waves.sh data/causal results/causal "$STEPS" "$CAUSAL_SEEDS" 2>&1 | tee results/causal_waves.log
fi

echo "[expanded] $(date -Is) ALL GPU STAGES DONE"
touch results/EXPANDED_GPU_DONE.txt

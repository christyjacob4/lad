#!/usr/bin/env bash
# One-shot resume script for the GB200 box. Run after gcloud auth is restored:
#
#   rsync -az --exclude '.git' --exclude data --exclude .venv \
#       /Users/christy/christyjacob4/lad/ gb200-devbox-webdocs-euw4-v2:~/hackathon/lad/
#   ssh gb200-devbox-webdocs-euw4-v2 'cd ~/hackathon/lad && bash scripts/devbox_resume.sh smoke'
#   ssh gb200-devbox-webdocs-euw4-v2 'cd ~/hackathon/lad && bash scripts/devbox_resume.sh full'
#
# Stages: env -> smoke (1-2 tiny cohorts, few steps) -> full (precompute + waves).
set -uo pipefail
cd ~/hackathon/lad
export PATH=$HOME/.local/bin:$PATH
STAGE="${1:-env}"

ensure_env() {
  [ -d .venv ] || uv venv --python 3.12 .venv
  source .venv/bin/activate
  python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
    || uv pip install torch --index-url https://download.pytorch.org/whl/cu128
  python -c "import vllm" 2>/dev/null || uv pip install vllm
  python -c "import trl, datasets, scipy, sklearn, matplotlib, sentence_transformers" 2>/dev/null \
    || uv pip install trl peft accelerate datasets scipy scikit-learn matplotlib pandas sentence-transformers
  python -c "import torch,vllm,trl; print('env OK torch',torch.__version__,'vllm',vllm.__version__,'gpus',torch.cuda.device_count())"
}

case "$STAGE" in
  env) ensure_env ;;

  smoke)
    # End-to-end smoke: precompute a tiny pool + 2 cohorts, GRPO few steps, confirm lift measured.
    ensure_env
    bash ~/hackathon/gpu_lease.sh run LAD -- bash -c '
      source .venv/bin/activate
      python scripts/precompute.py --outdir data/smoke --pool_size 300 \
        --cohort_size 64 --eval_size 120 --k 8 --max_tokens 400 2>&1 | tail -25
      # train two contrasting cohorts, few steps, one per GPU
      mkdir -p results/smoke
      gpu=0
      for c in diff_p50 diff_p95; do
        CUDA_VISIBLE_DEVICES=$gpu python -m lad.grpo_train \
          --cohort data/smoke/cohorts/$c.json --eval_tasks data/smoke/eval_set.json \
          --out results/smoke/lift_${c}_seed0.json --steps 20 --seed 0 \
          --batch_prompts 16 > results/smoke/log_$c.txt 2>&1 &
        gpu=$((gpu+1))
      done
      wait
      echo "=== SMOKE LIFTS ==="; cat results/smoke/lift_*.json 2>/dev/null
    '
    ;;

  full)
    ensure_env
    # Precompute (once) then launch waves detached in tmux under the lease.
    bash ~/hackathon/gpu_lease.sh run LAD -- bash -c '
      source .venv/bin/activate
      [ -f data/run/cohort_meta.json ] || \
        python scripts/precompute.py --outdir data/run --pool_size 2500 \
          --cohort_size 256 --eval_size 500 --k 8 2>&1 | tail -30
      tmux kill-session -t lad 2>/dev/null || true
      tmux new -d -s lad "source .venv/bin/activate; \
        bash scripts/run_waves.sh data/run results/lifts 200 \"0 1\" 2>&1 | tee results/waves.log"
      echo "[full] waves launched in tmux session lad; poll results/waves.log"
    '
    ;;

  analyze)
    ensure_env
    python scripts/analyze.py --datadir data/run --results results/lifts --outdir results
    ;;

  *) echo "usage: devbox_resume.sh {env|smoke|full|analyze}"; exit 1 ;;
esac

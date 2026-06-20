#!/usr/bin/env bash
# Durable finisher: waits for the GPU stages to finish, then runs the ENTIRE
# no-GPU analysis -> figures -> tables -> PAPER fill, and writes a final marker so
# it survives session/SSH restarts. Launch in its OWN persistent tmux:
#
#   tmux new -d -s lad-finish "cd ~/hackathon/lad && bash scripts/finish_paper.sh 2>&1 | tee results/finish.log"
#
# It polls for results/EXPANDED_GPU_DONE.txt (written by run_expanded.sh) OR for
# enough lift files, whichever comes first, so a partial sweep still produces a
# paper.
set -uo pipefail
cd ~/hackathon/lad
source .venv/bin/activate 2>/dev/null || true
export PYTHONPATH="$PWD:$PWD/scripts:${PYTHONPATH:-}"

MIN_LIFTS="${MIN_LIFTS:-8}"        # produce a paper once this many cohorts have lifts
MAX_WAIT_MIN="${MAX_WAIT_MIN:-720}"
echo "[finish] $(date -Is) waiting for GPU stages (min_lifts=$MIN_LIFTS)..."

deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))
while true; do
  ndone=$(ls results/lifts/lift_*.json 2>/dev/null | wc -l | tr -d ' ')
  ncoh=$(ls results/lifts/lift_*.json 2>/dev/null | sed -E 's/.*lift_(.*)_seed[0-9]+\.json/\1/' | sort -u | wc -l | tr -d ' ')
  if [ -f results/EXPANDED_GPU_DONE.txt ]; then
    echo "[finish] EXPANDED_GPU_DONE present; proceeding."
    break
  fi
  if [ "${ncoh:-0}" -ge "$MIN_LIFTS" ]; then
    echo "[finish] $ncoh cohorts have lifts (>= $MIN_LIFTS); could finalize, but waiting for GPU-done or timeout to capture all."
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "[finish] max wait reached; finalizing with whatever is present."
    break
  fi
  sleep 60
done

echo "[finish] computing all metrics from cached rollouts..."
python scripts/compute_all_metrics.py --datadir data/run --outdir results 2>&1 | tail -5

echo "[finish] master analysis (predictive + causal + mech + reliability + cost + acceptance)..."
python scripts/analyze.py --datadir data/run --results results/lifts \
  --mech results/mech --causal_results results/causal --outdir results 2>&1 | tail -60

echo "[finish] filling PAPER.md..."
python scripts/make_paper.py --outdir results 2>&1 | tail -5

echo "[finish] $(date -Is) DONE -> results/PAPER.md, results/figs/, results/tables/, results/summary.json"
date -Is > results/PAPER_FINAL_MARKER.txt
{
  echo "PAPER_FINAL_MARKER $(date -Is)"
  python - <<'PY' 2>/dev/null
import json
s=json.load(open("results/summary.json"))
pr=s.get("predictive",{})
print("n_cohorts", s.get("n_cohorts"))
print("LAD_loco_rho", pr.get("LAD",{}).get("rho_loco"))
print("naive_variance_loco_rho", pr.get("naive_variance",{}).get("rho_loco"))
print("diversity_loco_rho", pr.get("embedding_diversity",{}).get("rho_loco"))
print("passrate_loco_rho", pr.get("mean_pass_rate",{}).get("rho_loco"))
c=s.get("causal",{})
print("causal_top_minus_random", c.get("top_minus_random"))
print("causal_top_minus_bottom", c.get("top_minus_bottom"))
print("mech_lad_vs_advvar", s.get("mechanistic",{}).get("mean_group_reward_var",{}).get("spearman"))
PY
} >> results/PAPER_FINAL_MARKER.txt
cat results/PAPER_FINAL_MARKER.txt

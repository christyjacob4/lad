#!/usr/bin/env bash
# LAD run status dashboard. Standalone-runnable over ssh:
#   ssh gb200-devbox-webdocs-euw4-v2 'bash ~/hackathon/lad/status.sh'
# Prints: current phase, GRPO jobs done/total (main + causal separately),
# per-job avg minutes, live nvidia-smi util, computed ETA + elapsed-vs-4h budget.
set -uo pipefail
LAD=~/hackathon/lad
cd "$LAD" 2>/dev/null || { echo "no $LAD"; exit 1; }

STARTF="$LAD/results/RUN_START.txt"
PHASEF="$LAD/results/PHASE.txt"
N_MAIN_TOTAL_F="$LAD/results/N_MAIN_TOTAL.txt"
N_CAUSAL_TOTAL_F="$LAD/results/N_CAUSAL_TOTAL.txt"

now=$(date +%s)
echo "================= LAD STATUS  $(date -Is) ================="

# --- phase + elapsed/budget ---
phase="(unknown)"; [ -f "$PHASEF" ] && phase=$(cat "$PHASEF")
echo "PHASE: $phase"
if [ -f "$STARTF" ]; then
  start=$(cat "$STARTF")
  elapsed=$(( now - start ))
  em=$(( elapsed / 60 ))
  echo "ELAPSED: ${em} min / 240 min budget   (hard launch-stop at 220 min)"
  remain=$(( 240 - em )); [ $remain -lt 0 ] && remain=0
  echo "REMAINING to 4h: ${remain} min"
else
  echo "ELAPSED: (run not started; no RUN_START.txt)"
  em=0
fi

# --- main GRPO jobs ---
main_done=$(ls "$LAD"/results/lifts/lift_*.json 2>/dev/null | wc -l | tr -d ' ')
main_total="?"; [ -f "$N_MAIN_TOTAL_F" ] && main_total=$(cat "$N_MAIN_TOTAL_F")
echo "----------------------------------------------------------"
echo "MAIN GRPO:   $main_done / $main_total jobs done"
main_coh=$(ls "$LAD"/results/lifts/lift_*.json 2>/dev/null | sed -E 's/.*lift_(.*)_seed[0-9]+\.json/\1/' | sort -u | wc -l | tr -d ' ')
echo "MAIN cohorts with >=1 lift: $main_coh"

# --- causal GRPO jobs ---
causal_done=$(ls "$LAD"/results/causal/lift_*.json 2>/dev/null | wc -l | tr -d ' ')
causal_total="?"; [ -f "$N_CAUSAL_TOTAL_F" ] && causal_total=$(cat "$N_CAUSAL_TOTAL_F")
echo "CAUSAL GRPO: $causal_done / $causal_total jobs done"

# --- per-job avg minutes (from completed lift mtimes spread) ---
total_done=$(( main_done + causal_done ))
avg_min="n/a"
if [ "$total_done" -ge 1 ] && [ -f "$STARTF" ]; then
  # crude: elapsed training time / jobs done, scaled by parallelism (assume packing file)
  pack=4; [ -f "$LAD/results/PACK.txt" ] && pack=$(cat "$LAD/results/PACK.txt")
  # wall minutes elapsed since first lift appeared
  first_lift=$(ls -t "$LAD"/results/lifts/lift_*.json "$LAD"/results/causal/lift_*.json 2>/dev/null | tail -1)
  if [ -n "${first_lift:-}" ]; then
    fl_mtime=$(stat -c %Y "$first_lift" 2>/dev/null || echo "$now")
    last_lift=$(ls -t "$LAD"/results/lifts/lift_*.json "$LAD"/results/causal/lift_*.json 2>/dev/null | head -1)
    ll_mtime=$(stat -c %Y "$last_lift" 2>/dev/null || echo "$now")
    span=$(( ll_mtime - fl_mtime ))
    [ $span -lt 1 ] && span=1
    # per-job wall = span / (jobs since first / pack)
    if [ "$total_done" -gt 1 ]; then
      njobs_span=$(( total_done - 1 ))
      [ $njobs_span -lt 1 ] && njobs_span=1
      avg_min=$(awk "BEGIN{printf \"%.1f\", ($span/60.0)/($njobs_span/$pack)}")
    fi
  fi
fi
pack_show=4; [ -f "$LAD/results/PACK.txt" ] && pack_show=$(cat "$LAD/results/PACK.txt")
echo "PARALLELISM: ${pack_show}-way (jobs concurrent)"
echo "AVG min/job (wall, est): $avg_min"

# --- ETA for remaining A-C jobs ---
if [ "$main_total" != "?" ] && [ "$causal_total" != "?" ] && [ "$avg_min" != "n/a" ]; then
  remaining_jobs=$(( (main_total - main_done) + (causal_total - causal_done) ))
  [ $remaining_jobs -lt 0 ] && remaining_jobs=0
  eta=$(awk "BEGIN{printf \"%.0f\", ($remaining_jobs/$pack_show)*$avg_min}")
  echo "REMAINING jobs (A-C): $remaining_jobs  -> est ETA: ${eta} min"
  if [ -f "$STARTF" ]; then
    finish_at=$(( em + eta ))
    echo "EST FINISH at ~${finish_at} min elapsed  (budget 240 min)"
  fi
fi

# --- final marker ---
if [ -f "$LAD/results/PAPER_FINAL_MARKER.txt" ]; then
  echo "----------------------------------------------------------"
  echo "PAPER_FINAL_MARKER PRESENT:"
  cat "$LAD/results/PAPER_FINAL_MARKER.txt"
fi
if [ -f "$LAD/results/summary.json" ]; then
  echo "----------------------------------------------------------"
  echo "LATEST PAPER HEADLINE (from summary.json):"
  "$LAD/.venv/bin/python" - <<'PY' 2>/dev/null
import json
try:
    s=json.load(open("results/summary.json"))
    pr=s.get("predictive",{})
    print("  n_cohorts:", s.get("n_cohorts"))
    def g(m): return (pr.get(m,{}) or {}).get("rho_loco")
    print("  LAD LOCO rho      :", g("LAD"))
    print("  naive_var LOCO rho:", g("naive_variance"))
    print("  diversity LOCO rho:", g("embedding_diversity"))
    print("  passrate LOCO rho :", g("mean_pass_rate"))
    c=s.get("causal",{}) or {}
    print("  causal top-random :", c.get("top_minus_random"))
    print("  causal top-bottom :", c.get("top_minus_bottom"))
except Exception as e:
    print("  (summary not ready:", e, ")")
PY
fi

# --- live nvidia-smi ---
echo "----------------------------------------------------------"
echo "LIVE GPUs:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null \
  | awk -F, '{printf "  gpu%s util%6s  mem%9s /%s\n",$1,$2,$3,$4}'
echo "----------------------------------------------------------"
echo "tmux sessions:"; tmux ls 2>/dev/null | sed 's/^/  /' || echo "  (none)"
echo "lease: $(bash ~/hackathon/gpu_lease.sh status 2>/dev/null)"
echo "=========================================================="

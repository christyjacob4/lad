#!/usr/bin/env bash
# ANYTIME-ORDERED LAD paper run. Stopping at ANY wave boundary still yields a
# coherent PAPER.md (analyze+make_paper re-run after every wave).
#
# Order (priority): A precompute -> B1 seed0 main (difficulty/LAD-spectrum
# interleave) -> B2 seed1 main -> C1 top/random/bottom-LAD -> C2/C3 rest causal +
# dose-response -> (D robustness only if time).
#
# Launch DETACHED under the lease in tmux:
#   bash ~/hackathon/gpu_lease.sh run LAD -- \
#     tmux new -d -s lad-paper "cd ~/hackathon/lad && bash scripts/run_ordered.sh 2>&1 | tee results/ordered.log"
#
# Config via env: STEPS, PACK (jobs concurrent), POOL, COHORT, EVAL, K, MAXLEN.
set -uo pipefail
cd ~/hackathon/lad
source .venv/bin/activate 2>/dev/null || true
export PYTHONPATH="$PWD:$PWD/scripts:${PYTHONPATH:-}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

POOL="${POOL:-3000}"
COHORT="${COHORT:-256}"
EVAL="${EVAL:-500}"
K="${K:-32}"
STEPS="${STEPS:-70}"   # uniform across ALL cohorts/conditions/seeds (comparability preserved)
MAXLEN="${MAXLEN:-320}"
EVAL_TOK="${EVAL_TOK:-320}"
PACK="${PACK:-8}"                 # total concurrent GRPO jobs across 4 GPUs (8 = 2/GPU)
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
WALL_LAUNCH_STOP_MIN="${WALL_LAUNCH_STOP_MIN:-220}"  # stop LAUNCHING new jobs after this many minutes

DATADIR="data/run"
EVALSET="$DATADIR/eval_set.json"
LIFTS="results/lifts"
CAUSAL_DATA="data/causal"
CAUSAL_LIFTS="results/causal"
MECH_DIR="$PWD/results/mech"
mkdir -p "$LIFTS" "$CAUSAL_LIFTS" "$MECH_DIR" results

# Preserve an existing RUN_START (set when precompute/GPU work began) so the 4h
# wall covers the WHOLE experiment, not just this orchestrator's launch time.
if [ -f results/RUN_START.txt ]; then
  START=$(cat results/RUN_START.txt)
else
  START=$(date +%s); echo "$START" > results/RUN_START.txt
fi
echo "$PACK" > results/PACK.txt
echo "[ordered] $(date -Is) pool=$POOL cohort=$COHORT k=$K steps=$STEPS pack=$PACK maxlen=$MAXLEN"

set_phase(){ echo "$1" > results/PHASE.txt; echo "[ordered] PHASE -> $1"; }
elapsed_min(){ echo $(( ( $(date +%s) - START ) / 60 )); }
launch_stop(){ [ "$(elapsed_min)" -ge "$WALL_LAUNCH_STOP_MIN" ]; }

# ---------- analyze + make_paper (no GPU) after a wave ----------
reanalyze(){
  echo "[ordered] re-analyze @ $(elapsed_min)min elapsed"
  python scripts/compute_all_metrics.py --datadir "$DATADIR" --outdir results >/dev/null 2>>results/analyze.log || true
  python scripts/analyze.py --datadir "$DATADIR" --results "$LIFTS" \
    --mech "$MECH_DIR" --causaldir "$CAUSAL_DATA" --causal_results "$CAUSAL_LIFTS" \
    --outdir results >>results/analyze.log 2>&1 || true
  python scripts/make_paper.py --outdir results >>results/analyze.log 2>&1 || true
  echo "[ordered] PAPER.md refreshed ($(ls results/PAPER.md 2>/dev/null && wc -l <results/PAPER.md 2>/dev/null) lines)"
}

# ---------- run ONE GRPO job on a specific gpu ----------
run_one(){
  local gpu="$1" name="$2" seed="$3" cohort="$4" out="$5" mech="$6"
  local accb_arg=() mech_arg=()
  [ -n "${ACCB:-}" ] && accb_arg=(--acc_before "$ACCB")
  [ "$mech" = "1" ] && mech_arg=(--mech_dir "$MECH_DIR")
  echo "[gpu$gpu] START $name seed$seed @ $(elapsed_min)min"
  CUDA_VISIBLE_DEVICES="$gpu" python -m lad.grpo_train \
    --cohort "$cohort" --eval_tasks "$EVALSET" --out "$out" \
    --model "$MODEL" --steps "$STEPS" --seed "$seed" \
    --max_completion_len "$MAXLEN" --eval_max_tokens "$EVAL_TOK" \
    "${accb_arg[@]}" "${mech_arg[@]}" \
    > "$(dirname "$out")/log_${name}_seed${seed}.txt" 2>&1
  echo "[gpu$gpu] DONE  $name seed$seed (exit $?) @ $(elapsed_min)min"
}

# ---------- run a LIST of "name|seed|cohort|out|mech" jobs, PACK-concurrent,
#            reanalyze after each wave, respect the launch wall ----------
run_jobs(){
  local -n JOBS=$1
  local N=${#JOBS[@]} i=0
  # gpu assignment: round-robin gpu = (slot % 4)
  while [ $i -lt $N ]; do
    if launch_stop; then
      echo "[ordered] LAUNCH WALL reached ($(elapsed_min)min >= ${WALL_LAUNCH_STOP_MIN}); not launching remaining $((N-i)) jobs."
      break
    fi
    pids=()
    for slot in $(seq 0 $((PACK-1))); do
      [ $i -ge $N ] && break
      IFS='|' read -r name seed cohort out mech <<< "${JOBS[$i]}"
      if [ -f "$out" ]; then echo "[skip] $out exists"; i=$((i+1)); continue; fi
      gpu=$(( slot % 4 ))
      run_one "$gpu" "$name" "$seed" "$cohort" "$out" "$mech" &
      pids+=($!)
      i=$((i+1))
    done
    for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p"; done
    echo "[ordered] wave done; $i/$N in this list @ $(elapsed_min)min"
    reanalyze
  done
}

# =====================================================================
# PHASE A: precompute (k=32 -> free reliability sweep) + all cohorts
# =====================================================================
set_phase "A_precompute"
if [ ! -f "$DATADIR/cohort_meta.json" ]; then
  echo "[ordered] precompute (k=$K base rollouts + embeddings + cohorts)..."
  python scripts/precompute.py --outdir "$DATADIR" --pool_size "$POOL" \
    --cohort_size "$COHORT" --eval_size "$EVAL" --k "$K" 2>&1 | tail -40
else
  echo "[ordered] precompute present; skipping."
fi

# base acc_before ONCE (identical across all cohorts).
ACCB_FILE="$LIFTS/acc_before.json"
if [ ! -f "$ACCB_FILE" ]; then
  echo "[ordered] computing base acc_before ONCE on GPU0..."
  CUDA_VISIBLE_DEVICES=0 python -m lad.eval_base \
    --model "$MODEL" --eval_tasks "$EVALSET" --out "$ACCB_FILE" \
    --eval_max_tokens "$EVAL_TOK" > "$LIFTS/log_acc_before.txt" 2>&1 || \
    echo "[ordered] base eval failed; jobs will self-compute acc_before"
fi
ACCB=""
[ -f "$ACCB_FILE" ] && ACCB=$(python -c "import json;print(json.load(open('$ACCB_FILE'))['acc_before'])" 2>/dev/null)
echo "[ordered] acc_before=$ACCB"

# Build the LAD-spectrum interleaved cohort ORDER (NOT easy-first). Span the
# difficulty bands + adversarial/diversity families so a partial B1 already gives
# a full-spread LOCO scatter. Order computed from cohort_meta p_hat + tags.
ORDER=$(python scripts/cohort_order.py --datadir "$DATADIR" 2>/dev/null)
[ -z "$ORDER" ] && ORDER=$(ls "$DATADIR"/cohorts/*.json | xargs -n1 basename | sed 's/.json//')
ORDER_ARR=($ORDER)
echo "[ordered] cohort order (${#ORDER_ARR[@]}): $ORDER"

# record totals for status.sh
echo $(( ${#ORDER_ARR[@]} * 2 )) > results/N_MAIN_TOTAL.txt    # 2 seeds

reanalyze   # produces an (insufficient) stub immediately so PAPER.md always exists

# =====================================================================
# PHASE B1: seed0 main waves, mechanistic logging ON (Claims 1+2+3)
# =====================================================================
set_phase "B1_main_seed0"
B1=()
for name in "${ORDER_ARR[@]}"; do
  B1+=("$name|0|$DATADIR/cohorts/$name.json|$LIFTS/lift_${name}_seed0.json|1")
done
run_jobs B1
touch results/B1_DONE.txt
echo "[ordered] B1 (headline) COMPLETE @ $(elapsed_min)min"

# =====================================================================
# PHASE B2: seed1 main waves (error bars)
# =====================================================================
if ! launch_stop; then
  set_phase "B2_main_seed1"
  B2=()
  for name in "${ORDER_ARR[@]}"; do
    B2+=("$name|1|$DATADIR/cohorts/$name.json|$LIFTS/lift_${name}_seed1.json|1")
  done
  run_jobs B2
  touch results/B2_DONE.txt
fi

# =====================================================================
# PHASE C: causal selection. Build selections, then run in priority order.
# =====================================================================
if ! launch_stop && [ ! -f "$CAUSAL_DATA/cohort_meta.json" ]; then
  set_phase "C_build_causal"
  python scripts/causal_select.py --datadir "$DATADIR" --outdir "$CAUSAL_DATA" \
    --size "$COHORT" --with_dose 2>&1 | tail -25
fi
# totals for status.sh (selection conditions present)
if [ -d "$CAUSAL_DATA/cohorts" ]; then
  ncausal=$(ls "$CAUSAL_DATA"/cohorts/*.json 2>/dev/null | wc -l | tr -d ' ')
  echo $(( ncausal * 2 )) > results/N_CAUSAL_TOTAL.txt
fi

# C1: the core causal headline FIRST: top / random / bottom-LAD (seed0).
if ! launch_stop; then
  set_phase "C1_causal_headline"
  C1=()
  for name in top_lad random bottom_lad; do
    f="$CAUSAL_DATA/cohorts/$name.json"
    [ -f "$f" ] && C1+=("$name|0|$f|$CAUSAL_LIFTS/lift_${name}_seed0.json|0")
  done
  run_jobs C1
  touch results/C1_DONE.txt
fi

# C2/C3: the rest (seed0): easy/hard/variance/diversity/passrate, then dose-response.
if ! launch_stop; then
  set_phase "C2_causal_rest_seed0"
  C2=()
  for name in easy hard highest_naive_variance highest_diversity highest_pass_rate lowest_pass_rate \
              dose_top10 dose_top25 dose_top50 dose_bottom25 dose_random25; do
    f="$CAUSAL_DATA/cohorts/$name.json"
    [ -f "$f" ] && [ ! -f "$CAUSAL_LIFTS/lift_${name}_seed0.json" ] && \
      C2+=("$name|0|$f|$CAUSAL_LIFTS/lift_${name}_seed0.json|0")
  done
  run_jobs C2
  touch results/C2_DONE.txt
fi

# C3: causal seed1 (error bars on the causal bars), full priority order again.
if ! launch_stop; then
  set_phase "C3_causal_seed1"
  C3=()
  for name in top_lad random bottom_lad easy hard highest_naive_variance highest_diversity \
              highest_pass_rate lowest_pass_rate dose_top10 dose_top25 dose_top50 dose_bottom25 dose_random25; do
    f="$CAUSAL_DATA/cohorts/$name.json"
    [ -f "$f" ] && [ ! -f "$CAUSAL_LIFTS/lift_${name}_seed1.json" ] && \
      C3+=("$name|1|$f|$CAUSAL_LIFTS/lift_${name}_seed1.json|0")
  done
  run_jobs C3
  touch results/C3_DONE.txt
fi

set_phase "DONE_GPU"
touch results/EXPANDED_GPU_DONE.txt
echo "[ordered] $(date -Is) ALL GPU STAGES DONE (or launch wall hit) @ $(elapsed_min)min"
reanalyze

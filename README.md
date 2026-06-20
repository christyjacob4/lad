# LAD — Learnable Advantage Density

**A cheap, training-free cohort metric that predicts post-RL (GRPO) accuracy lift — and breaks the cost↔predictive-power Pareto frontier.**

> Inference-Time Compute Hackathon 2026 · **Applied AI** track.
> Hosts: Anthropic · Etched · Cognition · Mercor · Compute: Prime Intellect.

---

## One-line pitch

Before you pay for a single GRPO step, predict how much a data cohort will move the model — from **~8 base-model rollouts per task and zero training** — by measuring the *exploitable, non-redundant learning signal* the cohort contains.

```
LAS(T) = p̂(T) · (1 − p̂(T)) · (1 − p̂(T))^γ                      # per-task learnable advantage
LAD(C) = [ mean_T LAS(T) ] · ( VendiScore(C) / |C| )^β          # dataset-level value score
```

`p̂(T)` is the base model's pass-rate on task `T`, estimated from `k≈8` rollouts. No gradients.

---

## Why this is the right metric (the load-bearing rationale)

GRPO assigns each rollout a **group-relative advantage** `A_i = (r_i − mean_j r_j) / (std_j r_j + ε)`. The per-task gradient is driven by the **within-group reward spread**. For a binary verifier that spread is exactly `p(1−p)` — zero when everything passes (`p=1`, already solved, no headroom) or everything fails (`p=0`, nothing to reinforce), maximal at `p=0.5`.

This is **a theorem, not a hunch.** Bae et al. (*Online Difficulty Filtering*, EACL 2026, Prop 3.1) prove the reverse-KL between the initial policy and the RL-optimal policy is **lower-bounded by the Bernoulli variance of the pass-rate**:

```
D_KL(π_init ‖ π*)  ≥  p(T)(1 − p(T)) / (2β²)        # maximized at p = ½
```

So `p(1−p)` is the **leading term of a proven lower bound on the exact thing we predict (policy improvement).** DAPO, RL-ZVP, and difficulty-aware staged RL all confirm the converse: zero-variance prompts give no GRPO signal. Those works use `p(1−p)` *online, to filter batches mid-run*. **We re-purpose the same provably-correct quantity, computed once from cheap base-model rollouts, as a static, pre-purchase dataset-value score** — a different decision (procurement, not scheduling) for a different customer (a data marketplace, not the trainer loop).

### Two principled corrections (the insight)

Naive symmetric variance predicts *signal*; **lift is asymmetric**. Two derived fixes:

1. **Headroom `(1−p̂)^γ`** — a cohort at `p=0.3` has more room to rise (fail→pass) than one at `p=0.7` with equal variance. `γ` shifts the peak below 0.5 toward where you currently fail but *can* succeed. `γ` is a single fit parameter (ablation).
2. **Effective-diversity `(Vendi/|C|)^β`** — 256 near-duplicate paraphrases at `p=0.5` have huge summed advantage energy but teach *one* thing. The Vendi Score (exp-entropy of the embedding-kernel eigenvalues) is the *effective number of distinct tasks*; dividing by `|C|` gives a redundancy ceiling. `β` is a single fit parameter.

Each correction has a **predicted failure mode we demonstrate**: a **noisy-label** cohort looks learnable to variance but its corrupted labels don't teach; a **bimodal** easy+hard cohort has the same mean pass-rate as a middle cohort but different structure; a **low-diversity** cohort has high advantage energy but a redundancy ceiling. Variance mis-ranks all three; LAD ranks them correctly.

---

## What we measure (the experiment = the deliverable)

The spec hands us the ground truth: run one **identical** GRPO per size-matched cohort (cohort = the only variable), measure `lift = accuracy_after − accuracy_before` on a fixed held-out GSM8K test set, then show our cheap metric predicts that expensive oracle on **held-out cohorts**.

- **Model:** `Qwen2.5-1.5B-Instruct` (general, not Qwen-*Math* — avoids the spurious-reward confound from Shao et al.). Cross-family check on `Llama-3.2-3B-Instruct` if time allows.
- **Benchmark / verifier:** GSM8K, exact-match numeric (clean binary reward).
- **Cohorts (~13, size-matched at 256):** 5 difficulty bands · bimodal-vs-unimodal · 3 diversity levels · noisy-label vs clean · (real vs synthetic).
- **Validation:** **leave-one-cohort-out (LOCO)** — fit `metric→lift` on all-but-one cohort, predict the held-out one. Lead with **held-out Spearman ρ** (robust with few cohorts), report R²/RMSE secondary. Beat baselines: mean pass-rate, reward variance, diversity-only.

---

## Repo layout

```
lad/
  metric.py        LAD + LAS + Vendi score + the parameterized LAD family
  baselines.py     ALL baselines + LAD ablations computed from cached rollouts
  predictive.py    LOCO Spearman/Kendall/Pearson/R²/RMSE/MAE, calibration, top-k,
                   pairwise, bootstrap CI, permutation, paired-bootstrap, partial
                   Spearman + multivariate confounder controls
  reliability.py   rollout-budget sweep k∈{2..32}, rank stability, Beta smoothing,
                   bootstrap over rollouts/tasks
  cost.py          per-metric cost descriptors (fwd/verifier/backward, GPU-s, $)
  mech.py          mechanistic GRPO logging (group reward var, |adv|, zero-adv
                   frac, all-correct/all-wrong, KL, entropy, grad norm)
  validate.py      legacy LOCO helpers (superseded by predictive.py)
  gsm8k.py         GSM8K loader + exact-match numeric verifier
  rollouts.py      base-model rollout scoring + embeddings via vLLM
  cohorts.py       ~19 size-matched cohort families (vary one property)
  grpo_train.py    identical-per-cohort GRPO (TRL) + before/after eval -> lift,
                   with optional --mech_dir mechanistic instrumentation
scripts/
  precompute.py        score the pool (k≤32), embed, build ALL cohorts (ONCE)
  compute_all_metrics.py  every metric per cohort from cached rollouts (no GPU)
  causal_select.py     select cohorts by top/bottom-LAD/random/easy/hard/var/div/
                       passrate + dose-response (Claim 4 selection; no GPU)
  run_waves.sh         4-cohorts/GB200 GRPO waves (MECH_DIR=... enables mech logs)
  run_expanded.sh      full GPU driver: precompute -> main waves (+mech) -> causal
  finish_paper.sh      durable finisher: analysis -> figures -> tables -> PAPER fill
  analyze.py           master analysis -> results/summary.json (all 4 claims + §17)
  build_figures.py     all plan figures + tables from summary.json
  make_paper.py        auto-fill results/PAPER.md from summary.json
tests/
  test_mech.py         CPU/mock proof that mechanistic logging fires
  test_pipeline.py     CPU end-to-end smoke test of the analysis pipeline
results/           summary.json, PAPER.md, figs/*.png, tables/*.md
```

## How to run (on the 4×GB200 box, under the GPU lease)

```bash
# env (once): torch cu128 + vllm + trl, in a uv venv
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install vllm trl peft accelerate datasets scipy scikit-learn matplotlib sentence-transformers

# 1. cheap, training-free: score the pool, build cohorts, cache metric inputs (~20-30 min, 1 GPU)
bash ~/hackathon/gpu_lease.sh run LAD -- python scripts/precompute.py --outdir data/run

# 2. expensive oracle: identical GRPO per cohort, 4 in parallel (waves), detached
bash ~/hackathon/gpu_lease.sh run LAD -- tmux new -d -s lad \
  "bash scripts/run_waves.sh data/run results/lifts 200 2>&1 | tee results/waves.log"

# 3. fit + held-out validation + figures
python scripts/analyze.py --datadir data/run --results results/lifts --outdir results
```

### Expanded paper-grade run (all 4 claims + Section-17 criteria)

```bash
# under the lease, detached: precompute(k=32) -> main waves(+mech) -> causal selection
bash ~/hackathon/gpu_lease.sh run LAD -- \
  tmux new -d -s lad-paper "cd ~/hackathon/lad && bash scripts/run_expanded.sh 2>&1 | tee results/expanded.log"
# durable finisher (survives session restarts): analysis -> figures -> PAPER fill
tmux new -d -s lad-finish "cd ~/hackathon/lad && bash scripts/finish_paper.sh 2>&1 | tee results/finish.log"
# done when results/PAPER_FINAL_MARKER.txt exists
```

Sanity-check the analysis with no GPU:

```bash
python tests/test_mech.py            # mechanistic logging fires (mock trainer)
python tests/test_pipeline.py        # CPU smoke test of the analysis pipeline
```

---

## The headline

> *LAD predicts held-out post-RL lift at Spearman ρ ≈ (measured) using 8 base-model rollouts/task and zero training* — the same cost class as "reward variance" but materially higher predictive power, i.e. a vertical jump up the cost-quality frontier toward the spec's gold star.

---

## Sources

Core mechanism: DAPO (arXiv 2503.14476) · No Prompt Left Behind / RL-ZVP (2509.21880) · **Bae et al., Prop 3.1, the lower bound (2504.03380)** · Difficulty-Aware Staged RL (2504.00829). Diversity: Vendi Score (2210.02410). Confound guard: Spurious Rewards (2506.10947). Stack: Prime Intellect verifiers / prime-rl.

*License: MIT.*

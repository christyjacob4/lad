# LAD — Research & Validation Plan (authoritative)

This is the protocol the LAD experiments MUST follow. Source: reviewer-grade validation checklist.
Goal: make LAD **publishably convincing** by validating four claims.

## The thesis (claim exactly this — do NOT overclaim)
> LAD is a cheap, no-training, rollout-only dataset metric that predicts the realized downstream lift
> from GRPO better *per unit cost* than common data-quality proxies.

## The four claims and their tests
1. **Mechanistic** — LAD measures the GRPO learnable signal → LAD correlates with logged reward variance,
   group-advantage magnitude, and update signal *during actual GRPO*.
2. **Predictive** — LAD predicts post-RL lift on unseen cohorts → predicts Δacc = acc_after − acc_before
   on held-out cohorts (LOCO).
3. **Incremental** — LAD beats simpler and more expensive baselines at the same or lower cost.
4. **Causal** — selecting data by LAD causes better RL than random/easy/hard/diversity-only/variance-only.

## Metric definition (LOCK before training)
p̂_i = (s_i + α)/(k + α + β)   # s_i successes of k rollouts (k=8 default); α,β optional Beta smoothing
learnability_i = p̂_i (1 − p̂_i)
headroom_i     = (1 − p̂_i)^γ
LAD(D) = mean_i[ p̂_i (1−p̂_i)(1−p̂_i)^γ ] × effective-diversity(D)   # eff-div = Vendi score
Sanity: p=0 or 1 → ~0 contribution; γ=0 peaks at p=0.5; ↑γ shifts peak to harder-but-learnable;
duplicates → eff-div drops; orthogonal → eff-div rises; noisy labels must NOT dominate clean cohorts;
small rollout perturbation → ranking stable.

## Experimental design
- **Unit = cohort** (not task). All cohorts: same #tasks, same train budget, same base model, same GRPO
  config, same eval set, same verifier, same rollout budget for the metric; vary ONE data property.
- Cohort families (include those that BREAK naive metrics): very-easy, very-hard, intermediate-difficulty,
  high-diversity, low-diversity/duplicate-heavy, noisy-label/broken-verifier, synthetic-clean,
  synthetic-noisy, domain-matched, domain-shifted, long/verbose, adversarial/reward-hacking, mixed.
- Target ≥ ~13 cohorts for hackathon evidence; scale up + multi-seed if GPU budget allows.

## Outcome metrics (don't only report final acc)
raw lift; normalized lift Δ/(1−acc_before); AUC lift; best-ckpt lift; final-ckpt lift; transfer lift
(held-out benchmark); overfitting gap (train−heldout); negative-transfer rate; seed variance.

## Predictive validation (headline = LOCO Spearman; secondary = RMSE/R²)
Per cohort x_c=LAD(c), y_c=Δacc(c); fit ŷ=a+b·LAD; report LOCO Spearman ρ, Kendall τ, Pearson r,
R², RMSE, MAE, calibration slope/intercept, top-k precision, pairwise accuracy, bootstrap CI,
permutation p-value. Show scatter + predicted-vs-actual + residuals.

## Baselines LAD must beat
- Cheap/no-model: token length, char length, #reasoning steps, domain label, embedding-diversity-only,
  dedup score, semantic-cluster-count, synthetic-vs-real label.
- Cheap rollout: mean pass-rate, headroom mean(1−p̂), naive variance mean(p̂(1−p̂)), reward entropy,
  pass@k, majority-correctness, self-consistency, trajectory entropy, verifier max/mean/var score.
- LAD ablations: no-headroom, no-diversity, diversity-as-divisor, γ∈{0,0.5,1,2}, entropy-instead-of-p(1−p),
  hard-band(0.2<p<0.8), smoothed-p̂.
- Expensive (DEFERRED unless time): gradient norm, LESS/influence, LearnAlign, RHO-Loss, datamodels,
  Data Shapley, small-model RL lift, small-budget GRPO lift, full GRPO lift (oracle).

## Causal selection tests (intervention — train identical GRPO on each)
top-LAD, bottom-LAD, random, easy, hard, highest-naive-variance, highest-diversity, highest-pass-rate,
lowest-pass-rate. Headline: top-LAD beats all cheap baselines at equal RL compute.
Dose-response: top 10/25/50% LAD vs bottom 25% vs random 25% → expect monotone in LAD bucket.

## Mechanistic GRPO logging (per cohort, during training)
mean group reward variance (+), mean |advantage| (+), zero-advantage-group fraction (−), policy grad
norm (+ up to stability), KL/update (moderate +), entropy (no collapse), train reward improvement (+),
all-correct groups (more in easy), all-wrong groups (more in hard). Show: p≈0 and p≈1 → ~0 reward
variance; p≈0.5 → high. This connects LAD's derivation to observed optimizer behavior.

## Reliability of p̂ (it's only ~8 rollouts — prove it's not a sampling artifact)
rollout-budget sweep k∈{2,4,8,16,32}; rank stability (Spearman between k's); bootstrap over rollouts &
over tasks (CIs); Beta smoothing vs raw; temperature {0.6,0.8,1.0}; prompt-format sensitivity; verifier
repeatability + false-positive audit; reward-sparsity audit. KEY PLOT: LAD ranking k=8 ≈ k=16/32 →
strengthens the cheapness claim.

## Confounder controls
Partial Spearman / multivariate regression: Δacc_c = β0 + β1·LAD + β2·length + β3·diversity + β4·passrate + ε.
β1 must stay positive & significant. Control for length, completion length, domain, source, difficulty-only,
diversity-only, noise, duplicates, contamination, verifier leakage, seed, cohort size, training tokens.

## Cost → Pareto
Log per metric: forward passes, verifier calls, backward passes, GPU-seconds, wall-clock, $, peak mem,
requires training/gradients/labels/RL. Plot x=log(cost), y=LOCO Spearman/R² with bootstrap error bars.
Claim: LAD lies above cheaper metrics and left of more expensive metrics of comparable power. Use the
spec's graph structure (token-length → reward-variance/entropy → gradient influence → datamodels →
small-model RL lift → full post-train oracle).

## Required figures
LAD derivation p(1−p)(1−p)^γ vs p; cohort map; LAD-vs-lift scatter (with LOCO preds); predicted-vs-actual;
residuals; Pareto frontier; baseline bar chart (Spearman/RMSE); ablation chart; rollout-budget curve;
per-cohort learning curves; advantage-variance diagnostic (LAD vs GRPO adv var); top/random/bottom-LAD
causal bars; Vendi eigenvalue spectrum; noise/duplicate stress test.

## Required tables
cohort table; metric table (formula/cost/needs-grad/needs-train/score); main prediction table
(LOCO Spearman/Kendall/R²/RMSE/MAE); ablation table; baseline table; robustness table; cost table;
failure cases; verifier audit; full hyperparameters.

## Minimum acceptance criteria (Section 17)
LOCO Spearman clearly positive (ideally >0.6); LAD > naive variance (bootstrap CI / consistent across
benchmarks); LAD > diversity-only; LAD > pass-rate-only; top-LAD intervention beats random & bottom-LAD;
cost ≪ gradient/influence/small-RL; same direction across ≥2 models OR ≥2 benchmarks; ablation shows
removing headroom/diversity worsens prediction; mechanistic diagnostic agrees with GRPO logs; honest
failure analysis.

## Statistics
Spearman/Kendall + bootstrap CI; paired bootstrap over cohorts for "LAD beats X"; permutation tests;
multivariate regression / likelihood-ratio for added info; nested CV or pre-registered γ (no overfit);
mixed-effects for seeds; emphasize CIs over p-values at small n.

## Honesty rules
Pre-register or nested-CV the γ choice. Report failure modes (high-LAD-no-lift, low-LAD-high-lift,
diversity penalty too strong, headroom too strong, verifier artifacts, small-k noise, saturation,
forgetting, curriculum). Keep held-out honesty: fit on train cohorts, report on left-out.

## Related work to cite
GRPO/DeepSeekMath (2402.03300); LearnAlign; PODS down-sampling (2504.13818); DOTS difficulty;
RHO-Loss; Vendi (2210.02410); LESS (2402.04333); datamodels; Data Shapley.

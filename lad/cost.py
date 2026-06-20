"""Cost accounting for the Pareto frontier (RESEARCH_PLAN "Cost -> Pareto").

Each metric has a cost descriptor: forward passes, verifier calls, backward
passes, whether it needs training / gradients / labels / RL. We convert those
into a single comparable "cost" axis (log forward-equivalent passes) and into
estimated GPU-seconds / dollars so the Pareto plot can use x = log(cost).

The deferred-but-positioned expensive baselines (gradient norm, LESS/influence,
datamodels, Data Shapley, small-RL lift, full-RL oracle) are listed here with
their *known* cost class even when we don't run them, so the frontier figure can
place them honestly. Values are per-cohort, parameterized by k (rollouts/task),
n (#tasks/cohort), G (GRPO group size), and S (GRPO steps).
"""

import numpy as np


# Reference hardware throughput for GPU-second / dollar estimates (order-of-
# magnitude; the Pareto claim is about *relative* placement, stated as such).
GB200_DOLLARS_PER_HOUR = 6.0          # rough on-demand-class rate
TOKENS_PER_SEC_FWD = 6000.0          # 1.5B fwd-only batched gen, per GPU (approx)
AVG_GEN_TOKENS = 256.0               # mean completion length used for costing


def metric_cost(name, k=8, n=256, G=8, steps=150, fwd_tokens=AVG_GEN_TOKENS):
    """Return a cost descriptor for a metric over ONE cohort of n tasks.

    forward_passes: # model forward generations (token-generating passes).
    verifier_calls: # verifier invocations.
    backward_passes: # gradient/backprop passes.
    needs_{grad,train,labels,rl}: booleans.
    The headline scalar is `forward_equiv` (forward-pass-equivalents), used as the
    Pareto x-axis after a log.
    """
    # The cheap rollout family (LAD + all p-hat baselines) shares the SAME cost:
    # k rollouts per task on the base model, no gradients.
    rollout_fwd = k * n
    no_model = {
        # tier 0: free-ish (string ops / one embedding pass). We charge a tiny
        # nonzero cost (embedding fwd) so log() is finite and ordering is sane.
        "token_length": (0.0, 0, 0, False, False, False, False),
        "char_length": (0.0, 0, 0, False, False, False, False),
        "reasoning_steps": (0.0, 0, 0, False, False, False, False),
        "domain_label": (0.0, 0, 0, False, False, True, False),
        "synth_vs_real": (0.0, 0, 0, False, False, True, False),
        # embedding-based: one fwd per task through a small embedder (~0.1 of gen)
        "embedding_diversity": (0.1 * n, 0, 0, False, False, False, False),
        "dedup_score": (0.1 * n, 0, 0, False, False, False, False),
        "semantic_cluster_count": (0.1 * n, 0, 0, False, False, False, False),
    }
    if name in no_model:
        fe, vc, bp, ng, nt, nl, nr = no_model[name]
        return _cost_dict(name, fe, vc, bp, ng, nt, nl, nr, fwd_tokens)

    # Rollout-cost family: every pass-rate baseline + every LAD variant.
    rollout_family = {
        "mean_pass_rate", "headroom", "naive_variance", "reward_entropy",
        "pass_at_k", "majority_correct", "self_consistency", "trajectory_entropy",
        "verifier_max", "verifier_mean", "verifier_var", "reward_std",
    }
    if name in rollout_family or name.startswith("LAD"):
        # k*n forward gens + k*n verifier calls; LAD adds one embedding pass.
        extra_emb = 0.1 * n if name.startswith("LAD") and name not in (
            "LAD_no_diversity",) else 0.0
        return _cost_dict(name, rollout_fwd + extra_emb, k * n, 0,
                          False, False, True, False, fwd_tokens)

    # Expensive / deferred baselines -- positioned, not necessarily run.
    expensive = {
        # one backward pass per task (gradient norm) -- needs gradients+labels
        "gradient_norm": (rollout_fwd, k * n, n, True, False, True, False),
        # LESS/influence: gradient features for train+val, projection -> ~few backward
        "LESS_influence": (rollout_fwd, k * n, 5 * n, True, True, True, False),
        "LearnAlign": (rollout_fwd, k * n, 3 * n, True, True, True, False),
        "RHO_Loss": (rollout_fwd, k * n, 2 * n, True, True, True, False),
        # datamodels: many retrainings (~100x gradient cost)
        "datamodels": (50 * rollout_fwd, 50 * k * n, 50 * n * steps, True, True, True, True),
        "data_shapley": (50 * rollout_fwd, 50 * k * n, 50 * n * steps, True, True, True, True),
        # small-budget GRPO lift: a short RL run as the metric
        "small_rl_lift": (n * G * (steps // 4), n * G * (steps // 4),
                          n * (steps // 4), True, True, True, True),
        # full GRPO lift: the oracle (this is what we PREDICT)
        "full_rl_oracle": (n * G * steps, n * G * steps, n * steps,
                           True, True, True, True),
    }
    if name in expensive:
        fe, vc, bp, ng, nt, nl, nr = expensive[name]
        return _cost_dict(name, fe, vc, bp, ng, nt, nl, nr, fwd_tokens)

    # Unknown metric: assume rollout-cost.
    return _cost_dict(name, rollout_fwd, k * n, 0, False, False, True, False, fwd_tokens)


def _cost_dict(name, forward_equiv, verifier_calls, backward_passes,
               needs_grad, needs_train, needs_labels, needs_rl, fwd_tokens):
    # backward ~ 2x forward FLOPs; fold into a forward-equivalent for the cost axis
    total_fwd_equiv = forward_equiv + 2.0 * backward_passes
    gen_tokens = total_fwd_equiv * fwd_tokens
    gpu_seconds = gen_tokens / TOKENS_PER_SEC_FWD if gen_tokens > 0 else 0.0
    dollars = gpu_seconds / 3600.0 * GB200_DOLLARS_PER_HOUR
    return {
        "metric": name,
        "forward_passes": float(forward_equiv),
        "verifier_calls": float(verifier_calls),
        "backward_passes": float(backward_passes),
        "forward_equiv": float(total_fwd_equiv),
        "gpu_seconds": float(gpu_seconds),
        "dollars": float(dollars),
        "needs_gradients": bool(needs_grad),
        "needs_training": bool(needs_train),
        "needs_labels": bool(needs_labels),
        "needs_rl": bool(needs_rl),
        # log cost axis: +1 so tier-0 (0 passes) maps to 0, not -inf
        "log_cost": float(np.log10(total_fwd_equiv + 1.0)),
    }


def cost_table(metric_names, **kw):
    return {m: metric_cost(m, **kw) for m in metric_names}

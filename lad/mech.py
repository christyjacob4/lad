"""Mechanistic GRPO instrumentation (Claim 1: LAD measures the GRPO learnable signal).

The thesis is that LAD is the leading term of a proven lower bound on policy
improvement (Bae et al. Prop 3.1: D_KL(pi_init||pi*) >= p(1-p)/2beta^2). To show
LAD *measures the thing GRPO actually exploits*, we log, PER COHORT, the optimizer
quantities that the derivation predicts:

  - mean group reward variance         (+)  == p(1-p) for binary reward
  - mean |advantage|                   (+)  the GRPO gradient driver
  - zero-advantage-group fraction      (-)  groups where all rollouts tie -> no signal
  - policy grad norm                   (+)  up to stability
  - KL to ref / update                 (moderate +)
  - entropy                            (no collapse)
  - train reward / improvement         (+)
  - all-correct group count            (more in easy cohorts)
  - all-wrong group count              (more in hard cohorts)

These are accumulated per step and summarized per cohort, then persisted to
results/mech/<cohort>.json. The headline mechanistic test (Section-17) is that
LAD(cohort) correlates with the logged advantage variance / |advantage| across
cohorts -- i.e. the cheap pre-training score agrees with the realized optimizer
behaviour.

This module is import-light (numpy only) so it can be unit-tested on CPU with a
mock trainer. The TRL callback is attached in grpo_train.py.
"""

import json
import os
import numpy as np


# Per-group diagnostics computed purely from a group's reward vector. These are
# the quantities GRPO's group-relative advantage is built from, so they are the
# honest mechanistic ground truth for "is there a learnable signal here?".

def group_diagnostics(rewards, eps=1e-4):
    """Diagnostics for ONE group of G rollout rewards for a single prompt.

    rewards: 1d array-like of length G (for a binary verifier, 0/1).
    Returns a dict of per-group quantities. For a binary group with pass-rate p,
    var == p(1-p) exactly, which is precisely the quantity LAD estimates.
    """
    r = np.asarray(rewards, dtype=float)
    G = len(r)
    mean = float(r.mean())
    var = float(r.var())                      # == p(1-p) for binary reward
    std = float(r.std())
    # GRPO group-relative advantage A_i = (r_i - mean)/(std + eps).
    adv = (r - mean) / (std + eps)
    mean_abs_adv = float(np.mean(np.abs(adv)))
    zero_adv = bool(std < eps)                # degenerate group -> no gradient
    all_correct = bool(np.all(r >= 1.0 - 1e-9))
    all_wrong = bool(np.all(r <= 1e-9))
    return {
        "G": G,
        "reward_mean": mean,
        "reward_var": var,
        "reward_std": std,
        "mean_abs_adv": mean_abs_adv,
        "zero_adv": zero_adv,
        "all_correct": all_correct,
        "all_wrong": all_wrong,
    }


class MechAccumulator:
    """Accumulates per-group GRPO diagnostics across all steps of one cohort's
    training run, plus per-step trainer log scalars (grad norm, KL, entropy,
    loss, reward), and produces a per-cohort summary.

    Designed to be fed either by a real TRL callback (grpo_train.py) or by a
    mock loop (the CPU unit test) -- it only depends on numpy.
    """

    def __init__(self, cohort, seed=0):
        self.cohort = cohort
        self.seed = seed
        self.n_groups = 0
        self.sum_var = 0.0
        self.sum_abs_adv = 0.0
        self.n_zero_adv = 0
        self.n_all_correct = 0
        self.n_all_wrong = 0
        self.sum_reward = 0.0
        # per-step trainer scalars
        self.steps = []          # list of dicts: {step, grad_norm, kl, entropy, loss, reward}
        # per-step group-variance trace (mean over the step's groups)
        self.step_var_trace = []
        self.step_absadv_trace = []
        self.step_reward_trace = []

    def add_groups(self, group_reward_matrix, step=None):
        """group_reward_matrix: (n_prompts, G) reward array for the step's batch.

        Accumulates the per-group diagnostics. Returns the step-mean variance so
        the caller can also keep a per-step trace.
        """
        M = np.asarray(group_reward_matrix, dtype=float)
        if M.ndim == 1:
            M = M[None, :]
        step_vars, step_absadv, step_rewards = [], [], []
        for row in M:
            d = group_diagnostics(row)
            self.n_groups += 1
            self.sum_var += d["reward_var"]
            self.sum_abs_adv += d["mean_abs_adv"]
            self.n_zero_adv += int(d["zero_adv"])
            self.n_all_correct += int(d["all_correct"])
            self.n_all_wrong += int(d["all_wrong"])
            self.sum_reward += d["reward_mean"]
            step_vars.append(d["reward_var"])
            step_absadv.append(d["mean_abs_adv"])
            step_rewards.append(d["reward_mean"])
        if step_vars:
            self.step_var_trace.append((step, float(np.mean(step_vars))))
            self.step_absadv_trace.append((step, float(np.mean(step_absadv))))
            self.step_reward_trace.append((step, float(np.mean(step_rewards))))
        return float(np.mean(step_vars)) if step_vars else 0.0

    def add_step_scalars(self, step, grad_norm=None, kl=None, entropy=None,
                         loss=None, reward=None):
        """Record per-step trainer log scalars (from TRL's log dict)."""
        self.steps.append({
            "step": step,
            "grad_norm": _f(grad_norm),
            "kl": _f(kl),
            "entropy": _f(entropy),
            "loss": _f(loss),
            "reward": _f(reward),
        })

    def summary(self):
        ng = max(1, self.n_groups)
        # train-reward improvement: last - first from the per-step reward trace
        reward_imp = float("nan")
        if len(self.step_reward_trace) >= 2:
            reward_imp = self.step_reward_trace[-1][1] - self.step_reward_trace[0][1]
        elif self.steps:
            rs = [s["reward"] for s in self.steps if s["reward"] is not None]
            if len(rs) >= 2:
                reward_imp = rs[-1] - rs[0]

        def _last(key):
            vals = [s[key] for s in self.steps if s[key] is not None]
            return float(vals[-1]) if vals else float("nan")

        def _mean(key):
            vals = [s[key] for s in self.steps if s[key] is not None]
            return float(np.mean(vals)) if vals else float("nan")

        return {
            "cohort": self.cohort,
            "seed": self.seed,
            "n_groups": self.n_groups,
            # the headline mechanistic quantities (these should track LAD)
            "mean_group_reward_var": self.sum_var / ng,
            "mean_abs_advantage": self.sum_abs_adv / ng,
            "zero_advantage_group_frac": self.n_zero_adv / ng,
            "all_correct_group_frac": self.n_all_correct / ng,
            "all_wrong_group_frac": self.n_all_wrong / ng,
            "mean_train_reward": self.sum_reward / ng,
            "train_reward_improvement": reward_imp,
            # trainer scalars (mean over run + final)
            "mean_grad_norm": _mean("grad_norm"),
            "final_grad_norm": _last("grad_norm"),
            "mean_kl": _mean("kl"),
            "final_kl": _last("kl"),
            "mean_entropy": _mean("entropy"),
            "final_entropy": _last("entropy"),
            "entropy_collapse": _entropy_collapse(self.steps),
            "n_steps_logged": len(self.steps),
            # traces (for learning-curve / advantage-variance diagnostic figures)
            "step_var_trace": self.step_var_trace,
            "step_absadv_trace": self.step_absadv_trace,
            "step_reward_trace": self.step_reward_trace,
            "step_scalars": self.steps,
        }

    def save(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"{self.cohort}_seed{self.seed}.json")
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
        return path


def _f(x):
    if x is None:
        return None
    try:
        v = float(x)
        return None if (np.isnan(v) or np.isinf(v)) else v
    except (TypeError, ValueError):
        return None


def _entropy_collapse(steps):
    """Return the relative entropy drop (start->min) as a collapse indicator.
    >~0.5 suggests entropy collapse; near 0 means entropy held."""
    ents = [s["entropy"] for s in steps if s["entropy"] is not None]
    if len(ents) < 2:
        return float("nan")
    e0 = ents[0]
    if abs(e0) < 1e-9:
        return float("nan")
    return float((e0 - min(ents)) / abs(e0))


def load_mech_summaries(mech_dir):
    """Load all per-cohort mechanistic summaries, averaging across seeds.

    Returns dict cohort -> averaged-summary (scalar fields only).
    """
    import glob
    by_cohort = {}
    for path in glob.glob(os.path.join(mech_dir, "*.json")):
        with open(path) as f:
            s = json.load(f)
        by_cohort.setdefault(s["cohort"], []).append(s)
    scalar_keys = [
        "mean_group_reward_var", "mean_abs_advantage", "zero_advantage_group_frac",
        "all_correct_group_frac", "all_wrong_group_frac", "mean_train_reward",
        "train_reward_improvement", "mean_grad_norm", "final_grad_norm",
        "mean_kl", "final_kl", "mean_entropy", "final_entropy", "entropy_collapse",
    ]
    out = {}
    for c, runs in by_cohort.items():
        agg = {"cohort": c, "n_seeds": len(runs)}
        for k in scalar_keys:
            vals = [r[k] for r in runs if k in r and r[k] is not None
                    and not _isnan(r[k])]
            agg[k] = float(np.mean(vals)) if vals else float("nan")
        out[c] = agg
    return out


def _isnan(x):
    try:
        return np.isnan(float(x))
    except (TypeError, ValueError):
        return False

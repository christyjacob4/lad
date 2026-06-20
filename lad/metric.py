"""LAD — Learnable Advantage Density.

A cheap, training-free cohort metric that predicts post-RL (GRPO) accuracy lift.

The core quantity is derived from the GRPO group-relative advantage:

    A_i(T) = (r_i - mean_j r_j) / (std_j r_j + eps)

For a *binary* verifier, the within-group reward variance is exactly p(1-p),
maximized at p=0.5. Bae et al. (EACL 2026, Prop 3.1) prove the reverse-KL
between the initial policy and the RL-optimal policy is lower-bounded by the
Bernoulli variance of the pass-rate:

    D_KL(pi_init || pi*) >= p(T)(1-p(T)) / (2 beta^2)

So p(1-p) is the *leading term of a proven lower bound on policy improvement* —
not a grid-searched correlation. We compute it once from ~k base-model rollouts
(no training, no gradients) and aggregate it into a dataset-level value score.

Two principled corrections on top of naive variance:
  (a) Headroom asymmetry: lift on a benchmark is asymmetric. A cohort at p=0.3
      has more room to rise than one at p=0.7 (more mass can convert fail->pass).
      We weight by (1-p)^gamma.
  (b) Redundancy: lift is bounded by the number of *distinct* skills a cohort can
      teach. We divide by an effective-diversity fraction (Vendi/|C|)^beta.

Task-level:   LAS(T)  = p_hat * (1 - p_hat) * (1 - p_hat)^gamma
Dataset-level: LAD(C) = mean_T LAS(T) * (VendiScore(C)/|C|)^beta
"""

import numpy as np


def pass_rate_hat(rollout_correct):
    """Estimate per-task pass-rate p_hat from a list/array of {0,1} rollout outcomes.

    rollout_correct: array shape (n_tasks, k) of 0/1, or list of 1d arrays.
    Returns: array shape (n_tasks,) of p_hat in [0,1].
    """
    if isinstance(rollout_correct, np.ndarray) and rollout_correct.ndim == 2:
        return rollout_correct.mean(axis=1)
    return np.array([np.mean(r) for r in rollout_correct], dtype=float)


def las(p_hat, gamma=1.0):
    """Per-task learnable-advantage signal: p(1-p) * (1-p)^gamma.

    gamma=0 recovers pure Bernoulli variance p(1-p) (peak at p=0.5).
    gamma=1 gives p(1-p)^2 (peak at p=1/3 — biased toward headroom).
    gamma=2 gives p(1-p)^3 (peak at p=0.25).
    """
    p = np.asarray(p_hat, dtype=float)
    return p * (1.0 - p) * np.power(1.0 - p, gamma)


def vendi_score(embeddings, normalize=True):
    """Vendi Score: exp(Shannon entropy of the eigenvalues of the n x n
    cosine-similarity kernel / n). Interpretable as the *effective number of
    distinct elements* in the cohort.

    embeddings: array (n, d). Returns a float in [1, n].
    Cost is O(d^2 n) to build the kernel + O(n^3) eigendecomp — cheap, no training.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    n = X.shape[0]
    if n <= 1:
        return float(n)
    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        X = X / norms
    # Cosine similarity kernel, scaled by 1/n so trace == 1.
    K = (X @ X.T) / n
    # Symmetrize for numerical safety.
    K = 0.5 * (K + K.T)
    eigvals = np.linalg.eigvalsh(K)
    eigvals = eigvals[eigvals > 1e-12]
    eigvals = eigvals / eigvals.sum()  # normalize to a probability distribution
    entropy = -np.sum(eigvals * np.log(eigvals))
    return float(np.exp(entropy))


def lad(p_hat, embeddings=None, gamma=1.0, beta=1.0, vendi=None):
    """Dataset-level LAD score for one cohort.

    p_hat:      array (n,) of per-task base-model pass-rates.
    embeddings: array (n, d) of task embeddings (for the diversity term).
    gamma:      headroom exponent.
    beta:       diversity exponent. beta=0 disables the diversity term.
    vendi:      optionally pass a precomputed Vendi score to avoid recomputation.

    Returns dict with components and the final score.
    """
    p_hat = np.asarray(p_hat, dtype=float)
    n = len(p_hat)
    advantage_energy = float(las(p_hat, gamma=gamma).mean())

    if beta == 0 or embeddings is None:
        div_fraction = 1.0
        vendi_val = float("nan")
    else:
        vendi_val = vendi if vendi is not None else vendi_score(embeddings)
        div_fraction = vendi_val / n

    score = advantage_energy * (div_fraction ** beta)
    return {
        "lad": score,
        "advantage_energy": advantage_energy,
        "vendi": vendi_val,
        "div_fraction": div_fraction,
        "n": n,
        "gamma": gamma,
        "beta": beta,
    }


# ---- Baseline metrics (the cheaper points on the Pareto frontier) ----

def baseline_mean_pass_rate(p_hat):
    """Mean base-model pass-rate. (Tier 0-ish, cheap.)"""
    return float(np.mean(p_hat))


def baseline_variance(p_hat):
    """Mean Bernoulli variance mean_T p(1-p). This is gamma=0, beta=0 LAD —
    the 'naive variance' baseline we must beat. Symmetric about p=0.5."""
    return float(las(p_hat, gamma=0.0).mean())


def baseline_reward_std(rollout_rewards):
    """Mean within-group reward std across tasks. For binary rewards this is
    ~sqrt(p(1-p)); included as the spec's 'reward variance' point."""
    stds = [float(np.std(r)) for r in rollout_rewards]
    return float(np.mean(stds))


def baseline_vendi_only(embeddings):
    """Embedding-only diversity (Tier 0.5). No pass-rate info at all."""
    return vendi_score(embeddings)


def baseline_mean_length(token_lengths):
    """Mean completion/prompt length (Tier 0, ~free). Expected low correlation."""
    return float(np.mean(token_lengths))

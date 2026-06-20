"""All baseline metrics + LAD ablations, computed per cohort from cached rollouts.

One module that turns the cheap artifacts (per-task pass-rate, per-rollout
correctness, task embeddings, completion texts/lengths) into EVERY metric in the
RESEARCH_PLAN baseline + ablation list, so analyze/predictive can compare LAD
against all of them at once.

Metric tiers (RESEARCH_PLAN "Baselines LAD must beat"):
  Tier 0  (cheap, no model): token/char length, #reasoning-steps, domain label,
          embedding-diversity-only, dedup score, semantic-cluster-count, synth-vs-real.
  Tier 1  (cheap rollout): mean pass-rate, headroom mean(1-p), naive variance
          mean(p(1-p)), reward entropy, pass@k, self-consistency, trajectory
          entropy, verifier max/mean/var.
  LAD ablations: no-headroom, no-diversity, diversity-as-divisor, gamma in
          {0,0.5,1,2}, entropy-variant, hard-band(0.2<p<0.8), smoothed-p-hat.

Each metric is a function (cohort_features) -> float. `all_metrics` builds the
full table. Every metric also has a cost descriptor in lad.cost.
"""

import numpy as np

from .metric import vendi_score, las, las_family, lad_family


# --------------------------------------------------------------------------
# Cohort feature container
# --------------------------------------------------------------------------

class CohortFeatures:
    """Everything cheaply derivable for one cohort, used by every metric.

    p_hat:        (n,) per-task base-model pass-rate.
    correct:      (n, k) 0/1 per-rollout outcomes (None if unavailable).
    embeddings:   (n, d) task embeddings (None -> diversity metrics degrade to nan).
    token_lengths:(n,) mean completion token length per task (optional).
    char_lengths: (n,) mean completion char length per task (optional).
    n_steps:      (n,) #reasoning steps (e.g. newline-delimited) per task (optional).
    domain:       str or list (optional) -- domain label.
    is_synth:     bool or (n,) -- synthetic-vs-real label (optional).
    clusters:     (n,) semantic cluster ids (optional).
    """

    def __init__(self, p_hat, correct=None, embeddings=None, token_lengths=None,
                 char_lengths=None, n_steps=None, clusters=None,
                 is_synth=None, domain=None, vendi=None, name=None):
        self.p_hat = np.asarray(p_hat, dtype=float)
        self.n = len(self.p_hat)
        self.correct = None if correct is None else np.asarray(correct, dtype=float)
        self.k = None if self.correct is None else self.correct.shape[1]
        self.embeddings = None if embeddings is None else np.asarray(embeddings, dtype=float)
        self.token_lengths = _arr(token_lengths)
        self.char_lengths = _arr(char_lengths)
        self.n_steps = _arr(n_steps)
        self.clusters = None if clusters is None else np.asarray(clusters)
        self.is_synth = is_synth
        self.domain = domain
        self.name = name
        self._vendi = vendi  # cache

    @property
    def vendi(self):
        if self._vendi is None and self.embeddings is not None:
            self._vendi = vendi_score(self.embeddings)
        return self._vendi


def _arr(x):
    return None if x is None else np.asarray(x, dtype=float)


# --------------------------------------------------------------------------
# Tier 0 -- cheap, no model
# --------------------------------------------------------------------------

def m_token_length(cf):
    return float(np.mean(cf.token_lengths)) if cf.token_lengths is not None else float("nan")

def m_char_length(cf):
    return float(np.mean(cf.char_lengths)) if cf.char_lengths is not None else float("nan")

def m_reasoning_steps(cf):
    return float(np.mean(cf.n_steps)) if cf.n_steps is not None else float("nan")

def m_domain_label(cf):
    # numeric proxy: #distinct domains (a degenerate baseline -- usually constant)
    if cf.domain is None:
        return float("nan")
    if isinstance(cf.domain, str):
        return 1.0
    return float(len(set(cf.domain)))

def m_embedding_diversity(cf):
    """Vendi score over embeddings (effective # distinct), no pass-rate info."""
    return float(cf.vendi) if cf.vendi is not None else float("nan")

def m_dedup_score(cf):
    """Fraction of NON-redundant tasks = vendi/n (1 = all distinct, ->0 = many dups)."""
    if cf.vendi is None:
        return float("nan")
    return float(cf.vendi / cf.n)

def m_semantic_cluster_count(cf):
    """# distinct semantic clusters present (cheap diversity proxy)."""
    if cf.clusters is not None:
        return float(len(np.unique(cf.clusters)))
    if cf.embeddings is None:
        return float("nan")
    # fallback: greedy distinct count via cosine threshold (no sklearn dependency)
    return float(_greedy_cluster_count(cf.embeddings, thresh=0.85))

def m_synth_vs_real(cf):
    if cf.is_synth is None:
        return float("nan")
    if isinstance(cf.is_synth, (bool, np.bool_, int, float)):
        return float(cf.is_synth)
    return float(np.mean(np.asarray(cf.is_synth, dtype=float)))


# --------------------------------------------------------------------------
# Tier 1 -- cheap rollout metrics
# --------------------------------------------------------------------------

def m_mean_pass_rate(cf):
    return float(np.mean(cf.p_hat))

def m_headroom(cf):
    """Mean improvement headroom mean(1-p) -- how much room to rise."""
    return float(np.mean(1.0 - cf.p_hat))

def m_naive_variance(cf):
    """Mean Bernoulli variance mean(p(1-p)) -- the key naive baseline (gamma=0)."""
    return float(las(cf.p_hat, gamma=0.0).mean())

def m_reward_entropy(cf):
    """Mean per-task Bernoulli entropy H(p) = -p log p - (1-p) log(1-p)."""
    p = np.clip(cf.p_hat, 1e-9, 1 - 1e-9)
    h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return float(np.mean(h))

def m_pass_at_k(cf):
    """Mean pass@k = fraction of tasks solved by >=1 of the k rollouts."""
    if cf.correct is None:
        return float(np.mean(cf.p_hat > 0))  # degrade: any pass
    return float(np.mean(cf.correct.max(axis=1) > 0))

def m_majority_correct(cf):
    """Mean majority-vote correctness across the k rollouts (self-consistency-ish)."""
    if cf.correct is None:
        return float(np.mean(cf.p_hat >= 0.5))
    return float(np.mean(cf.correct.mean(axis=1) >= 0.5))

def m_self_consistency(cf):
    """Self-consistency = mean agreement of the k rollouts with their own mode.
    For a binary verifier this is mean(max(p,1-p))."""
    return float(np.mean(np.maximum(cf.p_hat, 1.0 - cf.p_hat)))

def m_trajectory_entropy(cf):
    """Mean entropy of the per-task correct/incorrect distribution across rollouts.
    Proxy for trajectory diversity from the verifier's perspective."""
    if cf.correct is None:
        return m_reward_entropy(cf)
    p = cf.correct.mean(axis=1)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return float(np.mean(h))

def m_verifier_max(cf):
    if cf.correct is None:
        return float(np.mean(cf.p_hat))
    return float(np.mean(cf.correct.max(axis=1)))

def m_verifier_mean(cf):
    return float(np.mean(cf.p_hat))

def m_verifier_var(cf):
    """Mean within-task reward variance across the k rollouts (== p(1-p)*k/(k-1))."""
    if cf.correct is None:
        return m_naive_variance(cf)
    return float(np.mean(cf.correct.var(axis=1)))

def m_reward_std(cf):
    if cf.correct is None:
        return float(np.mean(np.sqrt(cf.p_hat * (1 - cf.p_hat))))
    return float(np.mean(cf.correct.std(axis=1)))


# --------------------------------------------------------------------------
# LAD ablations
# --------------------------------------------------------------------------

def lad_default(cf, gamma=1.0, beta=1.0):
    return lad_family(cf.p_hat, cf.embeddings, gamma=gamma, beta=beta,
                      vendi=cf.vendi, n=cf.n, headroom="power", agg="mean",
                      div="vendi_frac")

def m_lad(cf):
    return lad_default(cf, gamma=1.0, beta=1.0)

def m_lad_no_headroom(cf):
    """gamma=0 -> pure variance x diversity (ablate the headroom correction)."""
    return lad_family(cf.p_hat, cf.embeddings, gamma=0.0, beta=1.0,
                      vendi=cf.vendi, n=cf.n, headroom="power", div="vendi_frac")

def m_lad_no_diversity(cf):
    """beta=0 -> advantage energy only (ablate the diversity correction)."""
    return lad_family(cf.p_hat, None, gamma=1.0, beta=0.0, div="none")

def m_lad_div_as_divisor(cf):
    """Diversity DIVIDES instead of multiplies (wrong sign sanity check):
    energy / (vendi/n)^beta -- redundancy should NOT be rewarded; this should
    predict worse, confirming the multiplicative form is right."""
    energy = float(las_family(cf.p_hat, gamma=1.0, headroom="power").mean())
    if cf.vendi is None:
        return energy
    frac = max(cf.vendi / cf.n, 1e-6)
    return float(energy / frac)

def m_lad_gamma0(cf):
    return lad_family(cf.p_hat, cf.embeddings, gamma=0.0, beta=1.0, vendi=cf.vendi, n=cf.n)

def m_lad_gamma05(cf):
    return lad_family(cf.p_hat, cf.embeddings, gamma=0.5, beta=1.0, vendi=cf.vendi, n=cf.n)

def m_lad_gamma1(cf):
    return lad_family(cf.p_hat, cf.embeddings, gamma=1.0, beta=1.0, vendi=cf.vendi, n=cf.n)

def m_lad_gamma2(cf):
    return lad_family(cf.p_hat, cf.embeddings, gamma=2.0, beta=1.0, vendi=cf.vendi, n=cf.n)

def m_lad_entropy_variant(cf):
    """Replace p(1-p) advantage energy with Bernoulli entropy H(p) (still peaks at
    0.5 but heavier tails). Tests whether the *specific* variance form matters."""
    p = np.clip(cf.p_hat, 1e-9, 1 - 1e-9)
    h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    energy = float(np.mean(h * np.power(1 - p, 1.0)))  # headroom-weighted entropy
    if cf.vendi is None or cf.embeddings is None:
        return energy
    return float(energy * (cf.vendi / cf.n))

def m_lad_hard_band(cf):
    """Hard-band indicator: mean fraction of tasks with 0.2<p<0.8 (the 'learnable
    band' as a simple gate), x diversity. A discretized LAD."""
    band = ((cf.p_hat > 0.2) & (cf.p_hat < 0.8)).astype(float)
    energy = float(np.mean(band))
    if cf.vendi is None or cf.embeddings is None:
        return energy
    return float(energy * (cf.vendi / cf.n))

def m_lad_smoothed(cf, alpha=1.0, beta_s=1.0):
    """LAD on Beta-smoothed p-hat: p~ = (s+alpha)/(k+alpha+beta). Robustifies the
    p estimate at small k. Falls back to raw p_hat if k unknown."""
    if cf.correct is not None and cf.k:
        s = cf.correct.sum(axis=1)
        p_sm = (s + alpha) / (cf.k + alpha + beta_s)
    else:
        p_sm = cf.p_hat
    return lad_family(p_sm, cf.embeddings, gamma=1.0, beta=1.0,
                      vendi=cf.vendi, n=cf.n, headroom="power", div="vendi_frac")


# --------------------------------------------------------------------------
# Registry + table builder
# --------------------------------------------------------------------------

BASELINES = {
    # tier 0
    "token_length": m_token_length,
    "char_length": m_char_length,
    "reasoning_steps": m_reasoning_steps,
    "domain_label": m_domain_label,
    "embedding_diversity": m_embedding_diversity,
    "dedup_score": m_dedup_score,
    "semantic_cluster_count": m_semantic_cluster_count,
    "synth_vs_real": m_synth_vs_real,
    # tier 1
    "mean_pass_rate": m_mean_pass_rate,
    "headroom": m_headroom,
    "naive_variance": m_naive_variance,
    "reward_entropy": m_reward_entropy,
    "pass_at_k": m_pass_at_k,
    "majority_correct": m_majority_correct,
    "self_consistency": m_self_consistency,
    "trajectory_entropy": m_trajectory_entropy,
    "verifier_max": m_verifier_max,
    "verifier_mean": m_verifier_mean,
    "verifier_var": m_verifier_var,
    "reward_std": m_reward_std,
}

ABLATIONS = {
    "LAD": m_lad,
    "LAD_no_headroom": m_lad_no_headroom,
    "LAD_no_diversity": m_lad_no_diversity,
    "LAD_div_as_divisor": m_lad_div_as_divisor,
    "LAD_gamma0": m_lad_gamma0,
    "LAD_gamma0.5": m_lad_gamma05,
    "LAD_gamma1": m_lad_gamma1,
    "LAD_gamma2": m_lad_gamma2,
    "LAD_entropy_variant": m_lad_entropy_variant,
    "LAD_hard_band": m_lad_hard_band,
    "LAD_smoothed": m_lad_smoothed,
}

ALL_METRICS = {**BASELINES, **ABLATIONS}


def compute_all(cohort_features):
    """cohort_features: dict cohort_name -> CohortFeatures.
    Returns dict metric_name -> {cohort_name -> value}."""
    table = {m: {} for m in ALL_METRICS}
    for cname, cf in cohort_features.items():
        for mname, fn in ALL_METRICS.items():
            try:
                table[mname][cname] = float(fn(cf))
            except Exception:
                table[mname][cname] = float("nan")
    return table


def as_aligned_arrays(table, names):
    """Convert {metric -> {cohort -> val}} into {metric -> np.array aligned to names}.
    Drops metrics that are all-nan / constant (no rank information)."""
    out = {}
    for m, d in table.items():
        arr = np.array([d.get(c, float("nan")) for c in names], dtype=float)
        if np.all(np.isnan(arr)):
            continue
        if np.nanstd(arr) < 1e-12:
            continue  # constant metric carries no ranking signal
        out[m] = arr
    return out


def _greedy_cluster_count(embeddings, thresh=0.85):
    """No-sklearn distinct-element count: greedily assign each row to an existing
    centroid if cosine sim > thresh, else start a new cluster."""
    X = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    Xn = X / norms
    centroids = []
    for row in Xn:
        if not centroids:
            centroids.append(row)
            continue
        sims = np.array([row @ c for c in centroids])
        if sims.max() < thresh:
            centroids.append(row)
    return len(centroids)

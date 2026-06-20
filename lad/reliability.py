"""Reliability of p-hat (RESEARCH_PLAN "Reliability of p-hat").

p-hat is estimated from only ~k=8 rollouts. We must prove the LAD ranking is not
a sampling artifact:

  - rollout-budget sweep k in {2,4,8,16,32} (inference only -- subsample the
    cached k-max rollouts; NO new GPU work needed if precompute used k>=32)
  - rank stability: Spearman between the cohort LAD ranking at each k and at k_max
  - bootstrap over rollouts (resample the k columns) and over tasks (resample rows)
  - Beta smoothing vs raw p-hat

KEY PLOT: LAD ranking at k=8 ~= k=16/32 -> strengthens the cheapness claim.

numpy + scipy only.
"""

import numpy as np
from scipy import stats

from .metric import vendi_score, lad_family


def subsample_correct(correct, k, seed=0):
    """Subsample k of the available rollout columns (without replacement)."""
    correct = np.asarray(correct, float)
    k_avail = correct.shape[1]
    if k >= k_avail:
        return correct
    rng = np.random.default_rng(seed)
    cols = rng.choice(k_avail, size=k, replace=False)
    return correct[:, cols]


def p_hat_from_correct(correct, smoothing=None):
    """p-hat from a (n,k) correct matrix. smoothing=(alpha,beta) -> Beta-smoothed."""
    correct = np.asarray(correct, float)
    k = correct.shape[1]
    s = correct.sum(axis=1)
    if smoothing is None:
        return s / k
    a, b = smoothing
    return (s + a) / (k + a + b)


def lad_at_k(cohort_correct, cohort_embeddings, k, gamma=1.0, beta=1.0,
             smoothing=None, seed=0, vendis=None):
    """LAD score per cohort using only k rollouts.

    cohort_correct: dict cohort -> (n,k_max) matrix.
    cohort_embeddings: dict cohort -> (n,d) (or None).
    Returns dict cohort -> LAD value.
    """
    out = {}
    for c, corr in cohort_correct.items():
        sub = subsample_correct(corr, k, seed=seed)
        p = p_hat_from_correct(sub, smoothing=smoothing)
        emb = cohort_embeddings.get(c) if cohort_embeddings else None
        v = (vendis or {}).get(c)
        out[c] = lad_family(p, emb, gamma=gamma, beta=beta, vendi=v,
                            n=len(p), headroom="power", div="vendi_frac")
    return out


def rollout_budget_sweep(cohort_correct, cohort_embeddings, lifts_by_cohort,
                         ks=(2, 4, 8, 16, 32), gamma=1.0, beta=1.0,
                         smoothing=None, n_seeds=5, vendis=None):
    """For each k, average LAD over n_seeds rollout-subsamples, then report:
      - LOCO Spearman of LAD(k) vs lift
      - rank stability vs the largest k (Spearman of cohort LAD rankings)

    Returns dict with per-k rho_lift, rho_rank_vs_kmax, and the LAD vectors.
    """
    from .predictive import loco_predictions
    names = [c for c in cohort_correct if c in lifts_by_cohort]
    names.sort()
    lifts = np.array([lifts_by_cohort[c] for c in names], float)
    k_avail = min(corr.shape[1] for corr in cohort_correct.values())
    ks = [k for k in ks if k <= k_avail] or [k_avail]
    kmax = max(ks)

    lad_by_k = {}
    for k in ks:
        accum = {c: [] for c in names}
        for s in range(n_seeds):
            d = lad_at_k(cohort_correct, cohort_embeddings, k, gamma=gamma,
                         beta=beta, smoothing=smoothing, seed=s, vendis=vendis)
            for c in names:
                accum[c].append(d[c])
        lad_by_k[k] = np.array([np.mean(accum[c]) for c in names])

    ref = lad_by_k[kmax]
    result = {"ks": ks, "cohorts": names, "lifts": lifts.tolist(),
              "lad_by_k": {k: v.tolist() for k, v in lad_by_k.items()},
              "rho_lift": {}, "rho_rank_vs_kmax": {}}
    for k in ks:
        preds = loco_predictions(lad_by_k[k], lifts)
        rho_lift, _ = stats.spearmanr(preds, lifts) if len(names) >= 3 else (float("nan"), 0)
        rho_rank, _ = stats.spearmanr(lad_by_k[k], ref) if len(names) >= 3 else (float("nan"), 0)
        result["rho_lift"][k] = float(rho_lift)
        result["rho_rank_vs_kmax"][k] = float(rho_rank)
    return result


def bootstrap_over_rollouts(correct, n_boot=500, gamma=1.0, seed=0,
                            embeddings=None, vendi=None):
    """Bootstrap a single cohort's LAD by resampling rollout columns with
    replacement. Returns mean + 95% CI of the cohort LAD score."""
    correct = np.asarray(correct, float)
    n, k = correct.shape
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        cols = rng.integers(0, k, k)
        p = correct[:, cols].mean(axis=1)
        vals.append(lad_family(p, embeddings, gamma=gamma, beta=1.0, vendi=vendi, n=n))
    vals = np.array(vals)
    return {"mean": float(vals.mean()),
            "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5))}


def bootstrap_over_tasks(correct, n_boot=500, gamma=1.0, seed=0,
                         embeddings=None):
    """Bootstrap a single cohort's LAD by resampling TASKS with replacement."""
    correct = np.asarray(correct, float)
    n, k = correct.shape
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        rows = rng.integers(0, n, n)
        p = correct[rows].mean(axis=1)
        emb = embeddings[rows] if embeddings is not None else None
        vals.append(lad_family(p, emb, gamma=gamma, beta=1.0, n=n))
    vals = np.array(vals)
    return {"mean": float(vals.mean()),
            "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5))}

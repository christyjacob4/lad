"""Cohort construction.

A cohort is a size-matched set of GSM8K tasks. We vary ONE property at a time so
that cohort identity is the only variable feeding GRPO. The base-model pass-rate
p_hat (precomputed once over a large pool) drives difficulty-band bucketing.

Cohort families (size fixed, e.g. 256):
  - difficulty bands: p_hat concentrated at ~0.05/0.25/0.5/0.75/0.95
  - mixture: bimodal easy+hard (mean ~0.5) vs unimodal middle (same mean)
  - diversity: high vs low (paraphrase/near-dup clusters) vs medium
  - synthetic quality: clean vs noisy-label (answers corrupted) vs templated
  - real vs synthetic
"""

import numpy as np


def _pick_band(pool_p, lo, hi, size, rng):
    """Indices of pool tasks with p_hat in [lo, hi]. Sample `size` (with
    replacement only if necessary)."""
    idx = np.where((pool_p >= lo) & (pool_p <= hi))[0]
    if len(idx) == 0:
        return np.array([], dtype=int)
    replace = len(idx) < size
    return rng.choice(idx, size=size, replace=replace)


def difficulty_band_cohorts(pool_p, size=256, seed=0):
    """5 cohorts, each concentrated in a narrow difficulty band."""
    rng = np.random.default_rng(seed)
    bands = {
        "diff_p05": (0.0, 0.15),
        "diff_p25": (0.15, 0.35),
        "diff_p50": (0.35, 0.65),
        "diff_p75": (0.65, 0.85),
        "diff_p95": (0.85, 1.0),
    }
    out = {}
    for name, (lo, hi) in bands.items():
        out[name] = _pick_band(pool_p, lo, hi, size, rng).tolist()
    return out


def mixture_cohorts(pool_p, size=256, seed=1):
    """Bimodal (half easy, half hard; mean ~0.5) vs unimodal middle (mean ~0.5).
    Same mean pass-rate, different structure -> tests variance vs LAD/headroom."""
    rng = np.random.default_rng(seed)
    half = size // 2
    easy = _pick_band(pool_p, 0.85, 1.0, half, rng)
    hard = _pick_band(pool_p, 0.0, 0.15, half, rng)
    bimodal = np.concatenate([easy, hard])
    middle = _pick_band(pool_p, 0.40, 0.60, size, rng)
    return {"mix_bimodal": bimodal.tolist(), "mix_unimodal": middle.tolist()}


def diversity_cohorts(pool_p, pool_clusters, size=256, seed=2):
    """High vs low diversity at matched difficulty.

    pool_clusters: per-task cluster id (e.g. k-means on embeddings). Low-diversity
    cohort draws from few clusters; high-diversity spreads across many.
    Restrict to the mid-difficulty band so difficulty is matched.
    """
    rng = np.random.default_rng(seed)
    mid = np.where((pool_p >= 0.25) & (pool_p <= 0.75))[0]
    clusters = pool_clusters[mid]
    uniq = np.unique(clusters)

    # Low diversity: pick a few clusters, draw all from them.
    few = rng.choice(uniq, size=min(3, len(uniq)), replace=False)
    low_idx = mid[np.isin(clusters, few)]
    low = rng.choice(low_idx, size=size, replace=len(low_idx) < size)

    # High diversity: spread across as many clusters as possible.
    high = rng.choice(mid, size=size, replace=len(mid) < size)

    # Medium: moderate cluster restriction.
    some = rng.choice(uniq, size=min(8, len(uniq)), replace=False)
    med_idx = mid[np.isin(clusters, some)]
    med = rng.choice(med_idx, size=size, replace=len(med_idx) < size)

    return {
        "div_low": low.tolist(),
        "div_med": med.tolist(),
        "div_high": high.tolist(),
    }


def noisy_label_cohort(pool_p, tasks, size=256, corrupt_frac=0.3, seed=3):
    """An adversarial cohort: mid-difficulty tasks whose gold answers are
    corrupted on `corrupt_frac` of them. Naive variance sees a 'learnable'
    p~0.5 cohort; but a chunk of the signal is now noise. Returns (indices,
    corrupted_answers_map) so the caller can override answers when scoring/training.
    """
    rng = np.random.default_rng(seed)
    mid = np.where((pool_p >= 0.35) & (pool_p <= 0.65))[0]
    idx = rng.choice(mid, size=size, replace=len(mid) < size)
    n_corrupt = int(corrupt_frac * size)
    corrupt_positions = rng.choice(size, size=n_corrupt, replace=False)
    overrides = {}
    for pos in corrupt_positions:
        ti = int(idx[pos])
        gold = tasks[ti]["answer"]
        # Corrupt: perturb the final numeric answer.
        overrides[pos] = _corrupt_answer(gold, rng)
    return idx.tolist(), overrides


def _corrupt_answer(answer_field, rng):
    import re
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_field)
    if not m:
        return answer_field
    num = m.group(1).replace(",", "")
    try:
        val = float(num)
    except ValueError:
        return answer_field
    # Add a random nonzero offset.
    delta = rng.choice([-7, -3, -1, 1, 2, 5, 11, 13])
    new = val + delta
    new_str = str(int(new)) if new == int(new) else str(new)
    return answer_field[: m.start(1)] + new_str

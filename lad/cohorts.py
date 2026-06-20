"""Cohort construction.

A cohort is a size-matched set of GSM8K tasks. We vary ONE property at a time so
that cohort identity is the only variable feeding GRPO. The base-model pass-rate
p_hat (precomputed once over a large pool) drives difficulty-band bucketing.

Cohort families (RESEARCH_PLAN, the ~13 types, scaled toward 20-30 when cheap):
  - difficulty bands: very-easy/very-hard + intermediate (p ~ 0.05..0.95)
  - mixture: bimodal easy+hard (mean ~0.5) vs unimodal middle (same mean)
  - diversity: high vs low (paraphrase/near-dup clusters) vs medium
  - duplicate-heavy: extreme redundancy (few seeds replicated)
  - synthetic quality: clean vs noisy-label (answers corrupted) vs templated
  - broken-verifier: a fraction of golds replaced so the verifier mis-scores
  - real vs synthetic; domain-matched vs domain-shifted (length-band proxy)
  - long/verbose; adversarial/reward-hacking; mixed (a bit of everything)

`build_all_cohorts` is the single entry point used by precompute; it returns a
spec dict the precompute script materializes (it owns embeddings/answers).
"""

import re
import numpy as np


# --------------------------------------------------------------------------
# Difficulty bands
# --------------------------------------------------------------------------

def _pick_band(pool_p, lo, hi, size, rng):
    idx = np.where((pool_p >= lo) & (pool_p <= hi))[0]
    if len(idx) == 0:
        # widen the band if empty (small pools)
        idx = np.argsort(np.abs(pool_p - (lo + hi) / 2))[:max(size, 1)]
    replace = len(idx) < size
    return rng.choice(idx, size=size, replace=replace)


def difficulty_band_cohorts(pool_p, size=256, seed=0):
    """6 cohorts spanning very-easy .. very-hard + an extra intermediate band."""
    rng = np.random.default_rng(seed)
    bands = {
        "diff_p05": (0.0, 0.12),     # very hard
        "diff_p20": (0.12, 0.30),
        "diff_p35": (0.28, 0.45),    # intermediate (learnable band)
        "diff_p50": (0.40, 0.60),
        "diff_p65": (0.55, 0.75),
        "diff_p80": (0.72, 0.88),
        "diff_p95": (0.88, 1.0),     # very easy
    }
    return {name: _pick_band(pool_p, lo, hi, size, rng).tolist()
            for name, (lo, hi) in bands.items()}


# --------------------------------------------------------------------------
# Mixture structure (same mean p, different shape)
# --------------------------------------------------------------------------

def mixture_cohorts(pool_p, size=256, seed=1):
    rng = np.random.default_rng(seed)
    half = size // 2
    easy = _pick_band(pool_p, 0.85, 1.0, half, rng)
    hard = _pick_band(pool_p, 0.0, 0.15, half, rng)
    bimodal = np.concatenate([easy, hard])
    middle = _pick_band(pool_p, 0.40, 0.60, size, rng)
    return {"mix_bimodal": bimodal.tolist(), "mix_unimodal": middle.tolist()}


# --------------------------------------------------------------------------
# Diversity (matched difficulty; vary effective # distinct tasks)
# --------------------------------------------------------------------------

def diversity_cohorts(pool_p, pool_clusters, size=256, seed=2):
    rng = np.random.default_rng(seed)
    mid = np.where((pool_p >= 0.25) & (pool_p <= 0.75))[0]
    if len(mid) < size:
        mid = np.argsort(np.abs(pool_p - 0.5))[:max(size * 4, size)]
    clusters = pool_clusters[mid]
    uniq = np.unique(clusters)

    few = rng.choice(uniq, size=min(3, len(uniq)), replace=False)
    low_idx = mid[np.isin(clusters, few)]
    low = rng.choice(low_idx, size=size, replace=len(low_idx) < size)

    high = rng.choice(mid, size=size, replace=len(mid) < size)

    some = rng.choice(uniq, size=min(8, len(uniq)), replace=False)
    med_idx = mid[np.isin(clusters, some)]
    med = rng.choice(med_idx, size=size, replace=len(med_idx) < size)

    # duplicate-heavy: a handful of distinct seeds, each replicated many times
    seeds_idx = rng.choice(mid, size=max(4, size // 32), replace=False)
    dup = rng.choice(seeds_idx, size=size, replace=True)

    return {
        "div_low": low.tolist(),
        "div_med": med.tolist(),
        "div_high": high.tolist(),
        "dup_heavy": dup.tolist(),
    }


# --------------------------------------------------------------------------
# Length / domain proxies (no extra dataset: bucket by question length)
# --------------------------------------------------------------------------

def length_cohorts(pool_p, pool_lengths, size=256, seed=6):
    """long/verbose vs short, at matched mid difficulty. Domain-shift proxy:
    GSM8K has no domain labels, so we use question-length quantiles as a stand-in
    distributional shift (documented as a proxy in PAPER limitations)."""
    rng = np.random.default_rng(seed)
    mid = np.where((pool_p >= 0.2) & (pool_p <= 0.8))[0]
    if len(mid) < size:
        mid = np.argsort(np.abs(pool_p - 0.5))[:max(size * 4, size)]
    L = pool_lengths[mid]
    order = np.argsort(L)
    short_idx = mid[order[:max(size, len(order) // 3)]]
    long_idx = mid[order[-max(size, len(order) // 3):]]
    long_c = rng.choice(long_idx, size=size, replace=len(long_idx) < size)
    return {"long_verbose": long_c.tolist()}


# --------------------------------------------------------------------------
# Synthetic quality / noisy label / broken verifier (the adversarial cohorts)
# --------------------------------------------------------------------------

def noisy_label_indices(pool_p, size=256, seed=3):
    rng = np.random.default_rng(seed)
    mid = np.where((pool_p >= 0.35) & (pool_p <= 0.65))[0]
    if len(mid) < size:
        mid = np.argsort(np.abs(pool_p - 0.5))[:max(size * 4, size)]
    return rng.choice(mid, size=size, replace=len(mid) < size)


def corrupt_overrides(idx, tasks, corrupt_frac=0.3, seed=3):
    """Map cohort-position -> corrupted gold answer for `corrupt_frac` of tasks.
    The verifier will then mis-score these -> a cohort that looks learnable by
    naive variance but whose corrupted signal does not teach."""
    rng = np.random.default_rng(seed + 100)
    n = len(idx)
    n_corrupt = int(corrupt_frac * n)
    positions = rng.choice(n, size=n_corrupt, replace=False)
    overrides = {}
    for pos in positions:
        ti = int(idx[pos])
        overrides[int(pos)] = _corrupt_answer(tasks[ti]["answer"], rng)
    return overrides


def _corrupt_answer(answer_field, rng):
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_field)
    if not m:
        return answer_field
    num = m.group(1).replace(",", "")
    try:
        val = float(num)
    except ValueError:
        return answer_field
    delta = rng.choice([-7, -3, -1, 1, 2, 5, 11, 13])
    new = val + delta
    new_str = str(int(new)) if new == int(new) else str(new)
    return answer_field[: m.start(1)] + new_str


def adversarial_indices(pool_p, size=256, seed=7):
    """Reward-hackable / trivially-formatted: very-easy tasks padded to look
    learnable. We approximate with the easiest band (p high) -- naive headroom
    would score it low, but a metric that confuses 'lots of rollouts' with signal
    could be fooled. Documented as the reward-hacking stress cohort."""
    rng = np.random.default_rng(seed)
    easy = np.where(pool_p >= 0.85)[0]
    if len(easy) < size:
        easy = np.argsort(-pool_p)[:max(size * 2, size)]
    return rng.choice(easy, size=size, replace=len(easy) < size)


def mixed_indices(pool_p, size=256, seed=8):
    """A realistic 'mixed bag': sample across the whole difficulty range."""
    rng = np.random.default_rng(seed)
    return rng.choice(len(pool_p), size=size, replace=len(pool_p) < size)


# --------------------------------------------------------------------------
# Top-level builder: returns a full spec the precompute script materializes
# --------------------------------------------------------------------------

def build_all_cohorts(pool_p, pool_clusters, pool_lengths, tasks, size=256,
                      noisy_frac=0.3, broken_frac=0.2):
    """Build the full cohort family. Returns dict:
       name -> {"indices": [...], "overrides": {pos: new_answer} or {},
                "is_synth": bool, "tags": [...]}
    The caller (precompute) writes the cohort jsons + meta from this.
    """
    spec = {}

    def add(name, idx, overrides=None, is_synth=False, tags=()):
        spec[name] = {"indices": [int(i) for i in idx],
                      "overrides": {int(k): v for k, v in (overrides or {}).items()},
                      "is_synth": bool(is_synth), "tags": list(tags)}

    for name, idx in difficulty_band_cohorts(pool_p, size=size, seed=0).items():
        add(name, idx, tags=["difficulty"])
    for name, idx in mixture_cohorts(pool_p, size=size, seed=1).items():
        add(name, idx, tags=["mixture"])
    for name, idx in diversity_cohorts(pool_p, pool_clusters, size=size, seed=2).items():
        add(name, idx, tags=["diversity"])
    for name, idx in length_cohorts(pool_p, pool_lengths, size=size, seed=6).items():
        add(name, idx, tags=["length"])

    # adversarial / synthetic-quality cohorts (the money-shot set)
    noisy_idx = noisy_label_indices(pool_p, size=size, seed=3)
    overrides = corrupt_overrides(noisy_idx, tasks, corrupt_frac=noisy_frac, seed=3)
    add("noisy_label", noisy_idx, overrides=overrides, is_synth=True,
        tags=["adversarial", "noisy"])
    # clean version of the SAME tasks (true labels) -- direct contrast
    add("clean_synth", noisy_idx, tags=["synthetic", "clean"])

    # broken-verifier: corrupt a larger fraction (verifier systematically wrong)
    broken_idx = noisy_label_indices(pool_p, size=size, seed=9)
    broken_ov = corrupt_overrides(broken_idx, tasks, corrupt_frac=broken_frac + 0.2,
                                  seed=9)
    add("broken_verifier", broken_idx, overrides=broken_ov, is_synth=True,
        tags=["adversarial", "broken_verifier"])

    add("adversarial", adversarial_indices(pool_p, size=size, seed=7),
        tags=["adversarial", "reward_hack"])
    add("mixed", mixed_indices(pool_p, size=size, seed=8), tags=["mixed"])

    return spec

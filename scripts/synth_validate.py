"""Synthetic end-to-end validation of the LAD pipeline (no GPU).

Purpose: prove the metric + LOCO validation machinery is correct and that LAD
beats naive variance on the *kinds of cohorts we will build*, using a simulator
whose lift is generated from the hypothesised mechanism plus realistic noise and
two adversarial confounds (noisy labels, redundancy). This is a software test of
the analysis, not a scientific result — the real result comes from GRPO runs.

Mechanism for simulated lift (per cohort C of binary-verifier tasks):
    lift(C) = A * [ mean_T p(1-p)(1-p)^G ]               # learnable advantage + headroom
                  * (effective_distinct_fraction)         # redundancy ceiling
                  * (1 - noisy_label_frac)                # corrupted labels don't teach
              + eps                                         # GRPO run noise
with ground-truth exponents G (headroom) and a multiplicative diversity ceiling,
so a metric that ignores headroom/diversity/label-noise should underperform.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lad.metric import (
    lad, las, vendi_score,
    baseline_mean_pass_rate, baseline_variance, baseline_vendi_only,
)
from lad.validate import compare_metrics, grid_fit_lad, loco_report


def make_cohort(kind, n=256, d=64, rng=None):
    """Return (p_hat, embeddings, noisy_frac, distinct_fraction_truth)."""
    rng = rng or np.random.default_rng()

    def emb_from_clusters(n_clusters):
        centers = rng.normal(size=(n_clusters, d)) * 3.0
        assign = rng.integers(0, n_clusters, size=n)
        embs = centers[assign] + rng.normal(size=(n, d)) * 0.3
        return embs

    if kind.startswith("diff_"):
        target = {"diff_p05": 0.06, "diff_p25": 0.25, "diff_p50": 0.5,
                  "diff_p75": 0.75, "diff_p95": 0.94}[kind]
        p = np.clip(rng.normal(target, 0.05, n), 0.01, 0.99)
        embs = emb_from_clusters(40)  # diverse
        return p, embs, 0.0
    if kind == "mix_bimodal":
        p = np.concatenate([rng.normal(0.93, 0.04, n // 2),
                            rng.normal(0.07, 0.04, n - n // 2)])
        p = np.clip(p, 0.01, 0.99)
        embs = emb_from_clusters(40)
        return p, embs, 0.0
    if kind == "mix_unimodal":
        p = np.clip(rng.normal(0.5, 0.05, n), 0.01, 0.99)
        embs = emb_from_clusters(40)
        return p, embs, 0.0
    if kind == "div_low":
        p = np.clip(rng.normal(0.5, 0.12, n), 0.01, 0.99)
        embs = emb_from_clusters(3)   # redundant
        return p, embs, 0.0
    if kind == "div_med":
        p = np.clip(rng.normal(0.5, 0.12, n), 0.01, 0.99)
        embs = emb_from_clusters(10)
        return p, embs, 0.0
    if kind == "div_high":
        p = np.clip(rng.normal(0.5, 0.12, n), 0.01, 0.99)
        embs = emb_from_clusters(60)
        return p, embs, 0.0
    if kind == "noisy_label":
        # Looks learnable (p~0.5) but 30% of labels are corrupt -> less real lift.
        p = np.clip(rng.normal(0.5, 0.1, n), 0.01, 0.99)
        embs = emb_from_clusters(40)
        return p, embs, 0.30
    if kind == "clean_synth":
        p = np.clip(rng.normal(0.45, 0.1, n), 0.01, 0.99)
        embs = emb_from_clusters(30)
        return p, embs, 0.0
    if kind == "templated":
        p = np.clip(rng.normal(0.45, 0.1, n), 0.01, 0.99)
        embs = emb_from_clusters(4)   # templated -> low diversity
        return p, embs, 0.0
    raise ValueError(kind)


def true_lift(p, embs, noisy_frac, G_truth=1.0, noise=0.0008, rng=None):
    rng = rng or np.random.default_rng()
    adv = las(p, gamma=G_truth).mean()
    distinct = vendi_score(embs) / len(p)
    base = 0.35 * adv * distinct * (1.0 - noisy_frac)
    return base + rng.normal(0, noise)


def main():
    rng = np.random.default_rng(0)
    kinds = (["diff_p05", "diff_p25", "diff_p50", "diff_p75", "diff_p95"]
             + ["mix_bimodal", "mix_unimodal"]
             + ["div_low", "div_med", "div_high"]
             + ["noisy_label", "clean_synth", "templated"])
    # Two seeds each (average lift) as in the real plan.
    p_hats, embs_list, noisy_fracs, lifts, names = [], [], [], [], []
    for k in kinds:
        p, embs, nf = make_cohort(k, rng=rng)
        l1 = true_lift(p, embs, nf, rng=rng)
        l2 = true_lift(p, embs, nf, rng=rng)
        p_hats.append(p)
        embs_list.append(embs)
        noisy_fracs.append(nf)
        lifts.append(0.5 * (l1 + l2))
        names.append(k)
    lifts = np.array(lifts)

    # Precompute Vendi per cohort (reused).
    vendis = [vendi_score(e) for e in embs_list]

    # Fit LAD (gamma, beta) on held-out LOCO rho.
    fit = grid_fit_lad(p_hats, embs_list, lifts,
                       gammas=(0.0, 0.5, 1.0, 1.5, 2.0),
                       betas=(0.0, 0.5, 1.0, 1.5, 2.0),
                       vendis=vendis, select_on="rho_loco")
    best = fit["best"]
    lad_vals = best["vals"]

    # Baselines.
    table = {
        "mean_pass_rate": np.array([baseline_mean_pass_rate(p) for p in p_hats]),
        "variance (p(1-p))": np.array([baseline_variance(p) for p in p_hats]),
        "vendi_only": np.array(vendis),
        f"LAD (gamma={best['gamma']},beta={best['beta']})": lad_vals,
    }
    reports = compare_metrics(table, lifts)

    print("=" * 74)
    print(f"Synthetic LOCO validation  ({len(names)} cohorts x 2 seeds)")
    print("=" * 74)
    print(f"{'metric':<34}{'rho_loco':>10}{'rho_in':>10}{'R2_loco':>10}")
    print("-" * 74)
    for name, rep in reports.items():
        print(f"{name:<34}{rep['rho_loco']:>10.3f}{rep['rho_insample']:>10.3f}{rep['r2_loco']:>10.3f}")
    print("-" * 74)
    print(f"best LAD params: gamma={best['gamma']}, beta={best['beta']}")
    print()
    # Adversarial subset: do the metrics rank noisy_label / div_low correctly?
    print("Per-cohort (sorted by actual lift):")
    order = np.argsort(-lifts)
    print(f"{'cohort':<16}{'lift':>10}{'variance':>12}{'LAD':>12}")
    for i in order:
        print(f"{names[i]:<16}{lifts[i]:>10.4f}{table['variance (p(1-p))'][i]:>12.4f}{lad_vals[i]:>12.4f}")

    lad_name = f"LAD (gamma={best['gamma']},beta={best['beta']})"
    assert reports[lad_name]["rho_loco"] > reports["variance (p(1-p))"]["rho_loco"], \
        "LAD should beat naive variance on held-out rho in the simulator"
    print("\nPASS: LAD beats naive variance on held-out LOCO rho (pipeline correct).")


if __name__ == "__main__":
    main()

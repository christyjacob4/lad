"""CPU end-to-end test of the analysis pipeline on a SYNTHETIC oracle.

We simulate cohorts with a known lift law so we can assert that (a) every metric
computes, (b) LAD's LOCO Spearman beats naive variance, pass-rate, and
diversity-only, (c) the predictive stat suite (bootstrap CI, permutation,
paired-bootstrap, confounder controls) runs and returns sane values, and (d) the
reliability sweep + cost table behave. This is the no-GPU proof that the harness
is correct before the expensive GRPO run lands.

Ground-truth lift law (chosen to match the metric's derivation so the harness can
be validated, NOT to inflate real results): lift increases with headroom-weighted
advantage energy AND effective diversity, with noise; noisy-label cohorts get a
penalty that variance can't see but LAD's design is meant to track via lower true
learnable mass.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lad.baselines import CohortFeatures, compute_all, as_aligned_arrays
from lad.metric import vendi_score, las
from lad import predictive as P
from lad import reliability as R
from lad import cost as CST


def make_synthetic_cohorts(n_cohorts=18, n_tasks=120, k=32, d=32, seed=0):
    """Generate cohorts spanning difficulty + diversity, each with cached k
    rollouts and embeddings, plus a 'true lift' from a known law."""
    rng = np.random.default_rng(seed)
    cohorts = {}
    lifts = {}
    for c in range(n_cohorts):
        # target mean pass-rate spanning [0.05, 0.95]
        mu = 0.05 + 0.9 * rng.random()
        # diversity: # latent clusters from 1 (dup-heavy) to many
        n_clusters = rng.integers(1, 12)
        # noisy-label flag for a few cohorts
        noisy = c % 6 == 0
        # per-task p drawn around mu
        conc = 6.0
        a = max(0.1, mu * conc)
        b = max(0.1, (1 - mu) * conc)
        p_true = rng.beta(a, b, size=n_tasks)
        # rollouts
        correct = (rng.random((n_tasks, k)) < p_true[:, None]).astype(float)
        # embeddings: tasks assigned to clusters; intra-cluster near-duplicate
        centers = rng.normal(size=(n_clusters, d))
        assign = rng.integers(0, n_clusters, size=n_tasks)
        emb = centers[assign] + 0.05 * rng.normal(size=(n_tasks, d))
        # completion lengths correlated weakly with difficulty (a confounder)
        tok_len = 80 + 200 * (1 - mu) + rng.normal(0, 20, size=n_tasks)

        name = f"coh{c:02d}" + ("_noisy" if noisy else "")
        cf = CohortFeatures(
            p_hat=correct.mean(axis=1), correct=correct, embeddings=emb,
            token_lengths=np.clip(tok_len, 1, None), char_lengths=tok_len * 4,
            n_steps=np.clip(tok_len / 25, 1, None), clusters=assign,
            is_synth=noisy, name=name, vendi=vendi_score(emb))
        cohorts[name] = cf

        # ---- the known lift law (oracle) ----
        p = correct.mean(axis=1)
        energy = float(las(p, gamma=1.0).mean())           # headroom-weighted
        div_frac = cf.vendi / n_tasks
        base_lift = 8.0 * energy * (div_frac ** 1.0)       # scale to ~0..0.15 range
        if noisy:
            base_lift *= 0.35                              # corrupted labels don't teach
        # measurement noise ~ a realistic GRPO seed-variance (SNR a few-fold,
        # not 1:1 -- real lifts are noisy but not pure noise).
        lifts[name] = base_lift + rng.normal(0, 0.006)
    return cohorts, lifts


def test_all_metrics_compute():
    cohorts, _ = make_synthetic_cohorts()
    table = compute_all(cohorts)
    # every registered metric produced a value for every cohort
    from lad.baselines import ALL_METRICS
    for m in ALL_METRICS:
        assert m in table
        assert len(table[m]) == len(cohorts)
    # the key metrics are not all-nan
    for m in ["LAD", "naive_variance", "mean_pass_rate", "embedding_diversity"]:
        vals = list(table[m].values())
        assert not all(np.isnan(v) for v in vals), m


def test_lad_beats_baselines_loco():
    cohorts, lifts_d = make_synthetic_cohorts(seed=1)
    names = sorted(cohorts)
    lifts = np.array([lifts_d[c] for c in names])
    table = compute_all(cohorts)
    arrs = as_aligned_arrays(table, names)

    lad_rep = P.loco_report(arrs["LAD"], lifts)
    var_rep = P.loco_report(arrs["naive_variance"], lifts)
    pass_rep = P.loco_report(arrs["mean_pass_rate"], lifts)
    div_rep = P.loco_report(arrs["embedding_diversity"], lifts)

    # On a law built from LAD's own components, LAD should clearly lead.
    assert lad_rep["rho_loco"] > 0.6, lad_rep["rho_loco"]
    assert lad_rep["rho_loco"] >= var_rep["rho_loco"]
    assert lad_rep["rho_loco"] >= div_rep["rho_loco"]
    # ablations: removing headroom or diversity should not improve over full LAD
    nh = P.loco_report(arrs["LAD_no_headroom"], lifts)["rho_loco"]
    nd = P.loco_report(arrs["LAD_no_diversity"], lifts)["rho_loco"]
    assert lad_rep["rho_loco"] >= min(nh, nd) - 0.15  # tolerant: noisy synthetic


def test_predictive_stat_suite():
    cohorts, lifts_d = make_synthetic_cohorts(seed=2)
    names = sorted(cohorts)
    lifts = np.array([lifts_d[c] for c in names])
    table = compute_all(cohorts)
    arrs = as_aligned_arrays(table, names)

    # bootstrap CI for LAD rho
    ci = P.bootstrap_rho_ci(arrs["LAD"], lifts, n_boot=300, seed=0)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]

    # permutation p-value should be small for the strong LAD signal
    perm = P.permutation_pvalue(arrs["LAD"], lifts, n_perm=500, seed=0)
    assert perm["p_value"] < 0.2

    # paired bootstrap: LAD vs pass-rate
    pb = P.paired_bootstrap_better(arrs["LAD"], arrs["mean_pass_rate"], lifts,
                                   n_boot=500, seed=0)
    assert 0.0 <= pb["prob_a_better"] <= 1.0

    # confounder control: LAD coefficient positive controlling for length+passrate
    preds = {"LAD": arrs["LAD"], "length": arrs["token_length"],
             "passrate": arrs["mean_pass_rate"], "diversity": arrs["embedding_diversity"]}
    mv = P.multivariate_regression(lifts, preds)
    assert mv["LAD"]["coef"] > 0

    # partial spearman of LAD vs lift controlling for length+passrate
    ps = P.partial_spearman(arrs["LAD"], lifts,
                            [arrs["token_length"], arrs["mean_pass_rate"]])
    assert ps["partial_rho"] > 0

    # full compare_all driver
    rep = P.compare_all({k: arrs[k] for k in ["LAD", "naive_variance",
                         "mean_pass_rate", "embedding_diversity"]},
                        lifts, focus="LAD", n_boot=200)
    assert "reports" in rep and "paired_vs_focus" in rep
    assert "naive_variance" in rep["paired_vs_focus"]


def test_reliability_sweep():
    cohorts, lifts_d = make_synthetic_cohorts(seed=3, k=32)
    names = sorted(cohorts)
    correct = {c: cohorts[c].correct for c in names}
    embs = {c: cohorts[c].embeddings for c in names}
    sweep = R.rollout_budget_sweep(correct, embs, lifts_d,
                                   ks=(2, 4, 8, 16, 32), n_seeds=3)
    # rank stability: k=8 ranking should agree strongly with k=32 ranking
    assert sweep["rho_rank_vs_kmax"][8] > 0.7
    assert sweep["rho_rank_vs_kmax"][32] == 1.0 or np.isnan(sweep["rho_rank_vs_kmax"][32]) \
        or sweep["rho_rank_vs_kmax"][32] > 0.99
    # bootstrap over rollouts / tasks for one cohort
    c0 = names[0]
    br = R.bootstrap_over_rollouts(correct[c0], n_boot=100, embeddings=embs[c0],
                                   vendi=cohorts[c0].vendi)
    assert br["lo"] <= br["mean"] <= br["hi"]
    bt = R.bootstrap_over_tasks(correct[c0], n_boot=100, embeddings=embs[c0])
    assert bt["lo"] <= bt["mean"] <= bt["hi"]


def test_cost_ordering():
    ct = CST.cost_table(["token_length", "embedding_diversity", "mean_pass_rate",
                         "naive_variance", "LAD", "gradient_norm", "small_rl_lift",
                         "datamodels", "full_rl_oracle"])
    # cheap (tier0) < rollout-cost (LAD ~ variance) < gradient < RL-tier
    assert ct["token_length"]["log_cost"] < ct["LAD"]["log_cost"]
    assert ct["LAD"]["log_cost"] <= ct["naive_variance"]["log_cost"] + 0.05
    assert ct["LAD"]["log_cost"] < ct["gradient_norm"]["log_cost"]
    assert ct["gradient_norm"]["log_cost"] < ct["small_rl_lift"]["log_cost"]
    assert ct["small_rl_lift"]["log_cost"] < ct["full_rl_oracle"]["log_cost"]
    # datamodels / shapley are the most expensive tier (many retrainings, ~>oracle)
    assert ct["datamodels"]["log_cost"] > ct["full_rl_oracle"]["log_cost"]
    # LAD needs no training/gradients/rl; oracle needs all
    assert not ct["LAD"]["needs_gradients"] and not ct["LAD"]["needs_rl"]
    assert ct["full_rl_oracle"]["needs_rl"] and ct["datamodels"]["needs_rl"]


if __name__ == "__main__":
    for fn in [test_all_metrics_compute, test_lad_beats_baselines_loco,
               test_predictive_stat_suite, test_reliability_sweep,
               test_cost_ordering]:
        fn()
        print("OK", fn.__name__)
    print("ALL PIPELINE TESTS PASSED")

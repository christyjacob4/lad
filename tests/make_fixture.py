"""Generate a fake precompute+lifts+mech+causal fixture so the FULL scripts
(analyze.py, build_figures.py, make_paper.py) can be exercised on CPU without GPU.

Writes a self-contained tree under --root:
  <root>/data/run/{cohort_meta.json, pool_scores.json, pool_embeddings.npy, eval_set.json}
  <root>/results/lifts/lift_<cohort>_seed{0,1}.json
  <root>/results/mech/<cohort>_seed{0,1}.json
  <root>/results/causal/lift_<condition>_seed0.json
"""

import argparse
import json
import os
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lad.metric import vendi_score, las
from lad.mech import MechAccumulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--pool", type=int, default=1200)
    ap.add_argument("--cohort_size", type=int, default=96)
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    datadir = os.path.join(args.root, "data", "run")
    os.makedirs(os.path.join(datadir, "cohorts"), exist_ok=True)

    # --- pool ---
    n = args.pool
    d = 32
    n_clusters = 30
    centers = rng.normal(size=(n_clusters, d))
    assign = rng.integers(0, n_clusters, size=n)
    embs = (centers[assign] + 0.1 * rng.normal(size=(n, d))).astype(np.float32)
    p_true = rng.beta(1.5, 1.5, size=n)
    correct = (rng.random((n, args.k)) < p_true[:, None]).astype(int)
    p_hat = correct.mean(axis=1)
    comp_tokens = (80 + 200 * (1 - p_hat) + rng.normal(0, 15, size=n)).clip(1)
    q_lengths = (100 + 150 * rng.random(n))

    json.dump({
        "questions": [f"q{i}: 2+{i}?" for i in range(n)],
        "answers": [f"#### {i}" for i in range(n)],
        "p_hat": p_hat.tolist(), "correct": correct.tolist(),
        "clusters": assign.tolist(), "comp_tokens": comp_tokens.tolist(),
        "q_lengths": q_lengths.tolist(), "k": args.k,
    }, open(os.path.join(datadir, "pool_scores.json"), "w"))
    np.save(os.path.join(datadir, "pool_embeddings.npy"), embs)
    json.dump([{"question": "q", "answer": "#### 1"}], open(os.path.join(datadir, "eval_set.json"), "w"))

    # --- cohorts (a spread of difficulty/diversity/adversarial) ---
    from lad import cohorts as C
    pool_tasks = [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(n)]
    spec = C.build_all_cohorts(p_hat, assign, q_lengths, pool_tasks, size=args.cohort_size)
    meta = {}
    lifts = {}
    sz = args.cohort_size
    for name, sp in spec.items():
        idx = np.asarray(sp["indices"], int)
        meta[name] = {"indices": idx.tolist(), "p_hat": p_hat[idx].tolist(),
                      "is_synth": sp["is_synth"], "tags": sp["tags"],
                      "noisy_frac": (len(sp["overrides"]) / len(idx)) if sp["overrides"] else 0.0}
        json.dump({"name": name, "questions": [f"q{i}" for i in idx],
                   "answers": [f"#### {i}" for i in idx]},
                  open(os.path.join(datadir, "cohorts", f"{name}.json"), "w"))
        # oracle lift law (built from LAD's components + adversarial penalties)
        p = p_hat[idx]; emb = embs[idx]
        energy = float(las(p, gamma=1.0).mean())
        v = vendi_score(emb)
        base = 8.0 * energy * (v / len(idx))
        if "noisy" in sp["tags"] or "broken_verifier" in sp["tags"]:
            base *= 0.3
        if "reward_hack" in sp["tags"]:
            base *= 0.4
        lifts[name] = base
    json.dump(meta, open(os.path.join(datadir, "cohort_meta.json"), "w"), indent=2)

    # --- lift files (2 seeds) + mech logs ---
    liftdir = os.path.join(args.root, "results", "lifts")
    mechdir = os.path.join(args.root, "results", "mech")
    os.makedirs(liftdir, exist_ok=True); os.makedirs(mechdir, exist_ok=True)
    acc_before = 0.44
    for name in spec:
        idx = np.asarray(meta[name]["indices"], int)
        p = p_hat[idx]
        for seed in (0, 1):
            lift = lifts[name] + rng.normal(0, 0.005)
            json.dump({"cohort": name, "seed": seed, "acc_before": acc_before,
                       "acc_after": acc_before + lift, "lift": lift,
                       "model": "synthetic", "steps": 150},
                      open(os.path.join(liftdir, f"lift_{name}_seed{seed}.json"), "w"))
            # mech log: simulate per-group rewards drawn at the cohort's p
            acc = MechAccumulator(name, seed=seed)
            for step in range(30):
                groups = (rng.random((8, 8)) < p[rng.integers(0, len(p), 8)][:, None]).astype(float)
                acc.add_groups(groups, step=step)
                acc.add_step_scalars(step, grad_norm=0.3 + 0.5 * groups.var(),
                                     kl=0.001 * step, entropy=2.0 - 0.01 * step,
                                     loss=-0.05, reward=float(groups.mean()) + 0.002 * step)
            acc.save(mechdir)

    # --- causal selection lifts (top_lad should win) ---
    causaldir = os.path.join(args.root, "results", "causal")
    os.makedirs(causaldir, exist_ok=True)
    from scripts.causal_select import build_selections, build_dose_response, per_task_energy
    sel = build_selections(p_hat, assign, embs, sz, gamma=1.0, seed=0)
    sel.update(build_dose_response(p_hat, assign, sz, gamma=1.0, seed=0))
    for cond, idx in sel.items():
        idx = np.asarray(idx, int)
        p = p_hat[idx]; emb = embs[idx]
        energy = float(las(p, gamma=1.0).mean()); v = vendi_score(emb)
        lift = 8.0 * energy * (v / len(idx)) + rng.normal(0, 0.004)
        json.dump({"cohort": cond, "seed": 0, "acc_before": acc_before,
                   "acc_after": acc_before + lift, "lift": lift,
                   "model": "synthetic", "steps": 150},
                  open(os.path.join(causaldir, f"lift_{cond}_seed0.json"), "w"))

    print(f"[fixture] wrote {len(spec)} cohorts + {len(sel)} causal conditions -> {args.root}")


if __name__ == "__main__":
    main()

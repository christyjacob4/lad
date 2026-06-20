"""Causal selection experiment (Claim 4 -- the GPU headline).

From a large pre-scored task pool, SELECT a fixed-size training cohort by each of
several selection rules, then train identical GRPO on each selected cohort and
compare realized lift. If selecting by LAD causes more lift than random / easy /
hard / diversity-only / variance-only / pass-rate selection at equal RL compute,
that is the causal claim.

Selection conditions (RESEARCH_PLAN "Causal selection tests"):
  top_lad, bottom_lad, random, easy, hard, highest_naive_variance,
  highest_diversity, highest_pass_rate, lowest_pass_rate
Dose-response buckets: top 10/25/50% LAD vs bottom 25% vs random 25%.

This script ONLY does the selection (cheap, CPU) and materializes the selected
cohorts as cohort jsons under <outdir>/cohorts/. The GRPO training of each
selected cohort is then driven by the SAME run_waves.sh (cohort = the only
variable), so the causal runs reuse the exact identical-GRPO machinery.

Per-task LAD contribution is the headroom-weighted advantage energy p(1-p)(1-p)^g;
the cohort-level diversity term is applied at SELECTION time by greedily picking
high-energy tasks while spreading across clusters (so 'top-LAD' is genuinely a
high-LAD cohort, not just high-energy duplicates).
"""

import argparse
import json
import os
import numpy as np


def per_task_energy(p_hat, gamma=1.0):
    p = np.asarray(p_hat, float)
    return p * (1 - p) * np.power(1 - p, gamma)


def select_diverse_top(scores, clusters, size, rng, spread=True):
    """Pick `size` indices maximizing `scores` while (optionally) spreading across
    clusters: round-robin over clusters taking each cluster's best remaining task.
    This realizes LAD's 'high energy AND non-redundant' selection."""
    order = np.argsort(-scores)
    if not spread or clusters is None:
        return order[:size].tolist()
    by_cluster = {}
    for i in order:
        by_cluster.setdefault(int(clusters[i]), []).append(int(i))
    picked = []
    cluster_ids = list(by_cluster)
    rng.shuffle(cluster_ids)
    ptr = {c: 0 for c in cluster_ids}
    while len(picked) < size:
        progressed = False
        for c in cluster_ids:
            if ptr[c] < len(by_cluster[c]):
                picked.append(by_cluster[c][ptr[c]])
                ptr[c] += 1
                progressed = True
                if len(picked) >= size:
                    break
        if not progressed:
            break
    if len(picked) < size:  # backfill
        for i in order:
            if int(i) not in set(picked):
                picked.append(int(i))
                if len(picked) >= size:
                    break
    return picked[:size]


def build_selections(p_hat, clusters, embeddings, size, gamma=1.0, seed=0):
    """Return dict condition -> list-of-indices for each selection rule."""
    rng = np.random.default_rng(seed)
    n = len(p_hat)
    energy = per_task_energy(p_hat, gamma=gamma)
    var = p_hat * (1 - p_hat)
    sel = {}

    # LAD-driven: high per-task energy, spread across clusters (diversity-aware)
    sel["top_lad"] = select_diverse_top(energy, clusters, size, rng, spread=True)
    sel["bottom_lad"] = select_diverse_top(-energy, clusters, size, rng, spread=True)
    # random
    sel["random"] = rng.choice(n, size=size, replace=n < size).tolist()
    # easy / hard (by pass-rate extremes)
    sel["easy"] = np.argsort(-p_hat)[:size].tolist()
    sel["hard"] = np.argsort(p_hat)[:size].tolist()
    # highest naive variance (symmetric, ignores headroom/diversity)
    sel["highest_naive_variance"] = select_diverse_top(var, clusters, size, rng, spread=False)
    # highest diversity (spread across as many clusters as possible, ignore p)
    sel["highest_diversity"] = select_diverse_top(np.ones(n), clusters, size, rng, spread=True)
    # pass-rate extremes
    sel["highest_pass_rate"] = np.argsort(-p_hat)[:size].tolist()
    sel["lowest_pass_rate"] = np.argsort(p_hat)[:size].tolist()
    return sel


def build_dose_response(p_hat, clusters, size, gamma=1.0, seed=0):
    """Dose-response: top 10/25/50% LAD-energy vs bottom 25% vs random 25%.
    Each bucket selects `size` tasks from within the named percentile slice."""
    rng = np.random.default_rng(seed + 7)
    n = len(p_hat)
    energy = per_task_energy(p_hat, gamma=gamma)
    order = np.argsort(-energy)

    def slice_pick(lo_frac, hi_frac):
        lo, hi = int(lo_frac * n), int(hi_frac * n)
        pool = order[lo:hi]
        return rng.choice(pool, size=size, replace=len(pool) < size).tolist()

    return {
        "dose_top10": slice_pick(0.0, 0.10),
        "dose_top25": slice_pick(0.0, 0.25),
        "dose_top50": slice_pick(0.0, 0.50),
        "dose_bottom25": slice_pick(0.75, 1.0),
        "dose_random25": rng.choice(n, size=size, replace=n < size).tolist(),
    }


def write_cohort(outdir, name, idx, pool):
    os.makedirs(os.path.join(outdir, "cohorts"), exist_ok=True)
    questions = [pool["questions"][i] for i in idx]
    answers = [pool["answers"][i] for i in idx]
    with open(os.path.join(outdir, "cohorts", f"{name}.json"), "w") as f:
        json.dump({"name": name, "questions": questions, "answers": answers}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data/run", help="precompute artifacts")
    ap.add_argument("--outdir", default="data/causal", help="where to write selected cohorts")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with_dose", action="store_true",
                    help="also build the dose-response buckets")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    pool = json.load(open(os.path.join(args.datadir, "pool_scores.json")))
    p_hat = np.asarray(pool["p_hat"], float)
    clusters = np.asarray(pool.get("clusters", [])) if pool.get("clusters") else None
    embeddings = None
    emb_path = os.path.join(args.datadir, "pool_embeddings.npy")
    if os.path.exists(emb_path):
        embeddings = np.load(emb_path)

    os.makedirs(args.outdir, exist_ok=True)
    # reuse the same fixed eval set so causal lifts are comparable to the main sweep
    src_eval = os.path.join(args.datadir, "eval_set.json")
    if os.path.exists(src_eval):
        import shutil
        shutil.copy(src_eval, os.path.join(args.outdir, "eval_set.json"))

    sel = build_selections(p_hat, clusters, embeddings, args.size,
                           gamma=args.gamma, seed=args.seed)
    if args.with_dose:
        sel.update(build_dose_response(p_hat, clusters, args.size,
                                       gamma=args.gamma, seed=args.seed))

    meta = {}
    for name, idx in sel.items():
        idx = [int(i) for i in idx]
        write_cohort(args.outdir, name, idx, pool)
        meta[name] = {"indices": idx, "p_hat": p_hat[idx].tolist(),
                      "selection": name}
    with open(os.path.join(args.outdir, "cohort_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[causal_select] wrote {len(sel)} selection cohorts -> {args.outdir}/cohorts/")
    for name, m in meta.items():
        e = per_task_energy(np.array(m["p_hat"]), gamma=args.gamma).mean()
        print(f"  {name:<22} p_mean={np.mean(m['p_hat']):.3f} "
              f"mean_energy={e:.4f} n={len(m['indices'])}")


if __name__ == "__main__":
    main()

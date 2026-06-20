"""Compute EVERY metric (baselines + LAD ablations) per cohort from the cached
precompute artifacts. No GPU. Writes results/metrics_table.json.

Reads:
  data/run/cohort_meta.json   (per-cohort indices, tags, is_synth)
  data/run/pool_scores.json   (p_hat, correct[k], comp_tokens, q_lengths, clusters)
  data/run/pool_embeddings.npy

This is the single place that turns cheap artifacts into the full comparison
table consumed by analyze/predictive/figures.
"""

import argparse
import json
import os
import numpy as np


def build_cohort_features(datadir):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from lad.baselines import CohortFeatures
    from lad.metric import vendi_score

    meta = json.load(open(os.path.join(datadir, "cohort_meta.json")))
    pool = json.load(open(os.path.join(datadir, "pool_scores.json")))
    embs = np.load(os.path.join(datadir, "pool_embeddings.npy"))

    correct_pool = np.asarray(pool["correct"], dtype=float)        # (pool, k)
    comp_tokens = np.asarray(pool.get("comp_tokens", []), dtype=float)
    clusters = np.asarray(pool.get("clusters", []))

    features = {}
    for name, m in meta.items():
        idx = np.asarray(m["indices"], dtype=int)
        corr = correct_pool[idx]
        emb = embs[idx]
        tl = comp_tokens[idx] if comp_tokens.size else None
        cl = clusters[idx] if clusters.size else None
        cf = CohortFeatures(
            p_hat=corr.mean(axis=1), correct=corr, embeddings=emb,
            token_lengths=tl, char_lengths=(tl * 4 if tl is not None else None),
            n_steps=(np.clip(tl / 20, 1, None) if tl is not None else None),
            clusters=cl, is_synth=m.get("is_synth", False),
            name=name, vendi=vendi_score(emb))
        features[name] = cf
    return features, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data/run")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from lad.baselines import compute_all

    features, meta = build_cohort_features(args.datadir)
    table = compute_all(features)

    os.makedirs(args.outdir, exist_ok=True)
    out = {"cohorts": sorted(features), "table": table,
           "vendi": {c: float(features[c].vendi) for c in features},
           "tags": {c: meta[c].get("tags", []) for c in meta}}
    with open(os.path.join(args.outdir, "metrics_table.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[compute_all_metrics] {len(features)} cohorts x {len(table)} metrics "
          f"-> {args.outdir}/metrics_table.json")


if __name__ == "__main__":
    main()

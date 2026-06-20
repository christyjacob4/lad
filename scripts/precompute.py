"""Step 1 (the cheap, training-free part): score a GSM8K pool with k base-model
rollouts, embed the tasks, build all cohorts, and persist everything.

This runs ONCE up front on a single GPU. Everything downstream (the metric
family, every cohort) is computed from these cached artifacts.

Outputs (under --outdir):
  pool_scores.json   : p_hat, correct[k], embeddings for the whole pool
  cohorts/<name>.json: per-cohort task list (questions+answers) for GRPO
  eval_set.json      : the FIXED held-out GSM8K test set (same eval before/after)
  cohort_meta.json   : per-cohort p_hat array + embedding indices (for the metric)
"""

import argparse
import json
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--embed_model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--pool_size", type=int, default=2000)
    ap.add_argument("--cohort_size", type=int, default=256)
    ap.add_argument("--eval_size", type=int, default=500)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--outdir", default="data/run")
    ap.add_argument("--gpu_mem_frac", type=float, default=0.45)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from lad.gsm8k import load_gsm8k
    from lad.rollouts import score_tasks_vllm
    from lad import cohorts as C

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "cohorts"), exist_ok=True)

    rng = np.random.default_rng(0)
    train_pool = load_gsm8k("train")
    test_all = load_gsm8k("test")

    # Fixed eval set (same before/after).
    eval_idx = rng.choice(len(test_all), size=min(args.eval_size, len(test_all)), replace=False)
    eval_tasks = [test_all[int(i)] for i in eval_idx]
    with open(os.path.join(args.outdir, "eval_set.json"), "w") as f:
        json.dump(eval_tasks, f)

    # Sample a pool to pre-score for difficulty bucketing.
    pool_idx = rng.choice(len(train_pool), size=min(args.pool_size, len(train_pool)), replace=False)
    pool = [train_pool[int(i)] for i in pool_idx]

    # --- vLLM base-model rollouts (the cost of LAD) ---
    from vllm import LLM
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem_frac,
              max_model_len=1024, dtype="bfloat16", enforce_eager=False)
    print(f"[precompute] scoring {len(pool)} pool tasks x k={args.k} rollouts...")
    scores = score_tasks_vllm(pool, llm, k=args.k, max_tokens=args.max_tokens)
    p_hat = scores["p_hat"]
    print(f"[precompute] pool p_hat: mean={p_hat.mean():.3f} "
          f"frac in (0,1)={(((p_hat>0)&(p_hat<1)).mean()):.2f}")

    # --- embeddings for diversity term ---
    print("[precompute] embedding pool tasks...")
    embs = embed_pool(pool, args.embed_model, args.gpu_mem_frac)

    # cluster ids for diversity cohort construction
    from sklearn.cluster import KMeans
    n_clusters = min(40, max(2, len(pool) // 20))
    clusters = KMeans(n_clusters=n_clusters, n_init=4, random_state=0).fit_predict(embs)

    # persist pool scores
    with open(os.path.join(args.outdir, "pool_scores.json"), "w") as f:
        json.dump({
            "questions": [t["question"] for t in pool],
            "answers": [t["answer"] for t in pool],
            "p_hat": p_hat.tolist(),
            "correct": scores["correct"].tolist(),
            "clusters": clusters.tolist(),
        }, f)
    np.save(os.path.join(args.outdir, "pool_embeddings.npy"), embs)

    # --- build cohorts ---
    cohort_index = {}
    cohort_index.update(C.difficulty_band_cohorts(p_hat, size=args.cohort_size, seed=0))
    cohort_index.update(C.mixture_cohorts(p_hat, size=args.cohort_size, seed=1))
    cohort_index.update(C.diversity_cohorts(p_hat, clusters, size=args.cohort_size, seed=2))

    # adversarial: noisy-label (override answers)
    noisy_idx, noisy_overrides = C.noisy_label_cohort(p_hat, pool, size=args.cohort_size, seed=3)

    meta = {}
    for name, idx in cohort_index.items():
        idx = [int(i) for i in idx]
        write_cohort(args.outdir, name, idx, pool)
        meta[name] = {
            "indices": idx,
            "p_hat": p_hat[idx].tolist(),
        }

    # write noisy-label cohort with corrupted answers
    write_cohort(args.outdir, "noisy_label", [int(i) for i in noisy_idx], pool,
                 answer_overrides=noisy_overrides)
    meta["noisy_label"] = {
        "indices": [int(i) for i in noisy_idx],
        "p_hat": p_hat[noisy_idx].tolist(),
        "noisy_frac": len(noisy_overrides) / len(noisy_idx),
    }
    # clean version of same difficulty band for contrast
    clean_idx = noisy_idx  # same tasks, true labels
    write_cohort(args.outdir, "clean_synth", [int(i) for i in clean_idx], pool)
    meta["clean_synth"] = {"indices": [int(i) for i in clean_idx],
                           "p_hat": p_hat[clean_idx].tolist()}

    with open(os.path.join(args.outdir, "cohort_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[precompute] built {len(meta)} cohorts -> {args.outdir}/cohorts/")
    print("[precompute] cohort p_hat means:")
    for name, m in meta.items():
        print(f"  {name:<14} p_hat_mean={np.mean(m['p_hat']):.3f}  n={len(m['indices'])}")


def embed_pool(pool, embed_model, gpu_mem_frac):
    """Embed via sentence-transformers (robust) or vLLM embedding if available."""
    texts = [t["question"] for t in pool]
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(embed_model, device="cuda")
        return np.asarray(m.encode(texts, batch_size=128, show_progress_bar=False,
                                   normalize_embeddings=True), dtype=np.float32)
    except Exception as e:
        print(f"[precompute] sentence-transformers failed ({e}); using vLLM embed")
        from vllm import LLM
        em = LLM(model=embed_model, task="embed", gpu_memory_utilization=gpu_mem_frac)
        outs = em.embed(texts)
        return np.array([o.outputs.embedding for o in outs], dtype=np.float32)


def write_cohort(outdir, name, idx, pool, answer_overrides=None):
    questions = [pool[i]["question"] for i in idx]
    answers = [pool[i]["answer"] for i in idx]
    if answer_overrides:
        for pos, new_ans in answer_overrides.items():
            answers[int(pos)] = new_ans
    with open(os.path.join(outdir, "cohorts", f"{name}.json"), "w") as f:
        json.dump({"name": name, "questions": questions, "answers": answers}, f)


if __name__ == "__main__":
    main()

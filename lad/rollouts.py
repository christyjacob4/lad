"""Base-model rollout scoring + task embeddings via vLLM.

This is the *entire* cost of LAD: k rollouts per task on the BASE model (no
gradients), plus one embedding pass for the diversity term. Everything is
batched through vLLM for throughput on the GB200s.
"""

import json
import numpy as np


def score_tasks_vllm(tasks, llm, k=8, temperature=1.0, max_tokens=512, seed=0):
    """Run k sampled rollouts per task on the base model, verify each, and
    return per-task outcome arrays.

    tasks: list of dicts with 'question' and 'answer'.
    llm:   a vllm.LLM instance.
    Returns: dict with
      'p_hat'   : (n,) base-model pass-rate
      'correct' : (n, k) 0/1 outcomes
      'rewards' : (n, k) reward (==correct for binary)
      'sample_completions': list of first completion per task (for inspection)
    """
    from vllm import SamplingParams
    from .gsm8k import build_prompt, is_correct

    prompts = [build_prompt(t["question"]) for t in tasks]
    sp = SamplingParams(
        n=k, temperature=temperature, top_p=1.0, max_tokens=max_tokens, seed=seed,
        stop=["\nQuestion:", "\n\nQuestion"],
    )
    outputs = llm.generate(prompts, sp)

    n = len(tasks)
    correct = np.zeros((n, k), dtype=np.int8)
    sample_completions = []
    for i, out in enumerate(outputs):
        gold = tasks[i]["answer"]
        comps = out.outputs
        for j in range(k):
            text = comps[j].text if j < len(comps) else ""
            correct[i, j] = is_correct(text, gold)
        sample_completions.append(comps[0].text if comps else "")
    p_hat = correct.mean(axis=1)
    return {
        "p_hat": p_hat,
        "correct": correct,
        "rewards": correct.astype(float),
        "sample_completions": sample_completions,
    }


def embed_tasks_vllm(tasks, embed_llm):
    """Embed task questions for the Vendi diversity term using a vLLM embedding
    model. Returns (n, d) float array."""
    texts = [t["question"] for t in tasks]
    outputs = embed_llm.embed(texts)
    embs = np.array([o.outputs.embedding for o in outputs], dtype=np.float32)
    return embs


def save_scores(path, tasks, scores, embeddings=None):
    """Persist scores to disk so the metric family can be recomputed without
    re-running rollouts."""
    payload = {
        "questions": [t["question"] for t in tasks],
        "answers": [t["answer"] for t in tasks],
        "p_hat": scores["p_hat"].tolist(),
        "correct": scores["correct"].tolist(),
    }
    if embeddings is not None:
        payload["embeddings"] = embeddings.tolist()
    with open(path, "w") as f:
        json.dump(payload, f)


def load_scores(path):
    with open(path) as f:
        payload = json.load(f)
    out = {
        "questions": payload["questions"],
        "answers": payload["answers"],
        "p_hat": np.array(payload["p_hat"], dtype=float),
        "correct": np.array(payload["correct"], dtype=np.int8),
    }
    if "embeddings" in payload:
        out["embeddings"] = np.array(payload["embeddings"], dtype=np.float32)
    return out

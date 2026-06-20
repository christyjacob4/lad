"""Compute the base model's accuracy on the FIXED eval set, ONCE.

acc_before is identical across every cohort (same base model, same eval set), so
the wave runner computes it a single time and passes it to all GRPO runs via
--acc_before. This avoids ~N redundant base evals across the sweep.
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--eval_tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval_max_tokens", type=int, default=320)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lad.grpo_train import evaluate_accuracy

    with open(args.eval_tasks) as f:
        eval_tasks = json.load(f)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    acc = evaluate_accuracy(model, eval_tasks, tokenizer=tok,
                            max_tokens=args.eval_max_tokens)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"acc_before": acc, "model": args.model,
                   "n_eval": len(eval_tasks)}, f, indent=2)
    print(f"[eval_base] acc_before={acc:.4f} on {len(eval_tasks)} eval tasks")


if __name__ == "__main__":
    main()

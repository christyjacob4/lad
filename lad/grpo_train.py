"""Identical GRPO training per cohort (TRL GRPOTrainer).

This is the expensive oracle. Cohort identity is the ONLY variable: every other
hyperparameter is fixed across cohorts. We measure
    lift = accuracy_after - accuracy_before
on a FIXED held-out GSM8K test set (same eval before and after).

Run as a standalone process so we can place one cohort per GPU and run 4 in
parallel under the lease. The reward is the GSM8K exact-match binary verifier.
"""

import argparse
import json
import os
import numpy as np


def load_cohort_tasks(cohort_path):
    with open(cohort_path) as f:
        c = json.load(f)
    # c: {"name":..., "questions":[...], "answers":[...]}
    return c


def gsm8k_reward(completions, answer, **kwargs):
    """TRL reward fn: list[str] completions, `answer` is the per-sample gold
    field broadcast by the dataset column. Returns list[float] in {0,1}."""
    from lad.gsm8k import is_correct
    return [float(is_correct(c, a)) for c, a in zip(completions, answer)]


def build_dataset(tasks):
    from datasets import Dataset
    from lad.gsm8k import build_prompt
    rows = {
        "prompt": [build_prompt(q) for q in tasks["questions"]],
        "answer": list(tasks["answers"]),
    }
    return Dataset.from_dict(rows)


def evaluate_accuracy(model_path_or_obj, eval_tasks, tokenizer=None, max_tokens=512,
                      batch_size=256, use_vllm_llm=None):
    """Greedy/sampled eval of pass@1 accuracy on a fixed test set.

    If use_vllm_llm is provided (a vllm.LLM), use it for fast batched generation.
    """
    from lad.gsm8k import build_prompt, is_correct
    prompts = [build_prompt(t["question"]) for t in eval_tasks]
    golds = [t["answer"] for t in eval_tasks]

    if use_vllm_llm is not None:
        from vllm import SamplingParams
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=max_tokens,
                            stop=["\nQuestion:", "\n\nQuestion"])
        outs = use_vllm_llm.generate(prompts, sp)
        texts = [o.outputs[0].text for o in outs]
    else:
        # HF generation. `model_path_or_obj` may be EITHER a path/repo string OR
        # an already-loaded nn.Module (we pass the live base/trained model so we
        # don't reload weights). Detect which and act accordingly.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if hasattr(model_path_or_obj, "generate") and not isinstance(model_path_or_obj, str):
            model = model_path_or_obj
            tok = tokenizer or AutoTokenizer.from_pretrained(model.name_or_path)
        else:
            tok = tokenizer or AutoTokenizer.from_pretrained(model_path_or_obj)
            model = AutoModelForCausalLM.from_pretrained(
                model_path_or_obj, dtype=torch.bfloat16, device_map="cuda")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # Left-pad for correct batched decoder-only generation.
        prev_side = tok.padding_side
        tok.padding_side = "left"
        was_training = model.training
        model.eval()
        dev = next(model.parameters()).device
        texts = []
        for i in range(0, len(prompts), 16):
            batch = prompts[i:i + 16]
            enc = tok(batch, return_tensors="pt", padding=True).to(dev)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            for g in gen:
                texts.append(tok.decode(g[enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True))
        tok.padding_side = prev_side
        if was_training:
            model.train()
    correct = [is_correct(t, g) for t, g in zip(texts, golds)]
    return float(np.mean(correct))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, help="path to cohort json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--eval_tasks", required=True, help="path to fixed eval-set json")
    ap.add_argument("--out", required=True, help="path to write result json")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04, help="KL coeff")
    ap.add_argument("--batch_prompts", type=int, default=64)
    ap.add_argument("--max_completion_len", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_max_tokens", type=int, default=512)
    ap.add_argument("--acc_before", type=float, default=None,
                    help="precomputed base accuracy on the eval set (identical "
                         "across cohorts for a fixed base model); skips the base "
                         "eval when provided.")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    with open(args.eval_tasks) as f:
        eval_tasks = json.load(f)
    cohort = load_cohort_tasks(args.cohort)
    train_ds = build_dataset(cohort)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- accuracy_before ----
    # The base model + eval set are identical across all cohorts, so acc_before
    # is the same for every run; the wave runner computes it ONCE and passes it
    # in via --acc_before to avoid 26 redundant base evals.
    if args.acc_before is not None:
        acc_before = float(args.acc_before)
        print(f"[grpo_train] using precomputed acc_before={acc_before:.4f}")
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda")
        acc_before = evaluate_accuracy(base_model, eval_tasks, tokenizer=tok,
                                       max_tokens=args.eval_max_tokens)
        del base_model
        torch.cuda.empty_cache()

    # ---- GRPO ----
    cfg = GRPOConfig(
        output_dir=f"/tmp/grpo_{cohort['name']}_seed{args.seed}",
        per_device_train_batch_size=args.batch_prompts,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_len,
        learning_rate=args.lr,
        beta=args.beta,
        max_steps=args.steps,
        temperature=args.temperature,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        seed=args.seed,
        report_to=[],
        use_vllm=False,  # colocated TRL gen; vLLM-serve mode wired separately if needed
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=gsm8k_reward,
        args=cfg,
        train_dataset=train_ds,
        processing_class=tok,
    )
    trainer.train()

    # ---- accuracy_after ----
    trained = trainer.model
    # GRPO trains with gradient checkpointing / use_cache=False; re-enable the
    # KV cache so generation during eval is fast and correct.
    if hasattr(trained, "gradient_checkpointing_disable"):
        try:
            trained.gradient_checkpointing_disable()
        except Exception:
            pass
    if hasattr(trained, "config"):
        trained.config.use_cache = True
    acc_after = evaluate_accuracy(trained, eval_tasks, tokenizer=tok,
                                  max_tokens=args.eval_max_tokens)

    result = {
        "cohort": cohort["name"],
        "seed": args.seed,
        "acc_before": acc_before,
        "acc_after": acc_after,
        "lift": acc_after - acc_before,
        "model": args.model,
        "steps": args.steps,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

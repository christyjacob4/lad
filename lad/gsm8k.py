"""GSM8K dataset loading + exact-match numeric verifier.

The verifier is a clean binary reward: extract the final numeric answer from the
model's completion and compare to the gold answer. This is V: str -> {str:int}
collapsed to r: ... -> {0,1} for our purposes.
"""

import re


GSM8K_SYSTEM = (
    "You are a careful math assistant. Solve the problem step by step, then give "
    "the final numeric answer on its own line in the form '#### <number>'."
)


def build_prompt(question):
    return (
        f"{GSM8K_SYSTEM}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_gold(answer_field):
    """GSM8K gold answers end with '#### <number>'."""
    if "####" in answer_field:
        tail = answer_field.split("####")[-1]
    else:
        tail = answer_field
    return _normalize_number(tail)


def extract_pred(completion):
    """Extract the model's final numeric answer.

    Prefer a '#### <num>' marker; otherwise take the last number in the text.
    """
    if "####" in completion:
        tail = completion.split("####")[-1]
        n = _normalize_number(tail)
        if n is not None:
            return n
    # Fallback: last number anywhere.
    nums = _NUM_RE.findall(completion)
    if not nums:
        return None
    return _to_float(nums[-1])


def _normalize_number(text):
    nums = _NUM_RE.findall(text)
    if not nums:
        return None
    return _to_float(nums[0])


def _to_float(s):
    s = s.replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def is_correct(completion, gold_answer_field):
    """Binary verifier: 1 if the extracted prediction matches gold, else 0."""
    gold = extract_gold(gold_answer_field)
    pred = extract_pred(completion)
    if gold is None or pred is None:
        return 0
    return int(abs(pred - gold) < 1e-4)


def load_gsm8k(split="train"):
    """Load GSM8K via the `datasets` library. Returns list of dicts with
    'question' and 'answer'."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [{"question": ex["question"], "answer": ex["answer"]} for ex in ds]

"""CPU/mock unit test: the mechanistic GRPO logging fires and produces the
correct diagnostics, WITHOUT any GPU or real trainer.

We simulate the reward fn being called step-by-step (as TRL would, with the G
rollouts of each prompt contiguous), and a fake on_log stream of trainer scalars,
then assert the persisted summary has the expected mechanistic quantities and
that the p=0/p=1 -> ~0 variance, p=0.5 -> high variance relationship holds.
"""

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lad.mech import (MechAccumulator, group_diagnostics, load_mech_summaries)


def test_group_diagnostics_binary_variance():
    # all-correct group -> var 0, zero advantage, all_correct flag
    d = group_diagnostics([1, 1, 1, 1])
    assert d["reward_var"] == 0.0
    assert d["zero_adv"] is True
    assert d["all_correct"] is True and d["all_wrong"] is False
    assert d["mean_abs_adv"] == 0.0

    # all-wrong group -> var 0, zero advantage, all_wrong flag
    d = group_diagnostics([0, 0, 0, 0])
    assert d["reward_var"] == 0.0
    assert d["zero_adv"] is True
    assert d["all_wrong"] is True

    # balanced group p=0.5 -> max variance 0.25, nonzero advantage
    d = group_diagnostics([1, 1, 0, 0])
    assert abs(d["reward_var"] - 0.25) < 1e-9
    assert d["zero_adv"] is False
    assert d["mean_abs_adv"] > 0


def test_p_curve_variance_monotone():
    # The derivation's core claim: reward variance == p(1-p), peaks at 0.5.
    G = 8
    vars_by_p = {}
    for s in range(G + 1):
        rewards = [1] * s + [0] * (G - s)
        vars_by_p[s / G] = group_diagnostics(rewards)["reward_var"]
    assert vars_by_p[0.0] == 0.0
    assert vars_by_p[1.0] == 0.0
    assert vars_by_p[0.5] == max(vars_by_p.values())
    assert vars_by_p[0.5] > vars_by_p[0.25] > vars_by_p[0.0]


def _mock_step_rewards(p, n_prompts, G, rng):
    """One step's flat reward vector: n_prompts groups of G rollouts, each group
    at pass-rate ~ p (TRL passes the flat list)."""
    flat = []
    for _ in range(n_prompts):
        s = rng.binomial(G, p)
        grp = [1.0] * s + [0.0] * (G - s)
        rng.shuffle(grp)
        flat.extend(grp)
    return flat


def test_accumulator_fires_and_summarizes():
    """Simulate a full training run via the reward fn + on_log, mimicking how
    grpo_train wires MechAccumulator -- no GPU, no trainer."""
    from lad.grpo_train import make_reward_fn

    rng = np.random.default_rng(0)
    G, n_prompts, steps = 8, 4, 20

    # learnable cohort (p~0.5) should show higher variance + |adv| than easy
    learnable = MechAccumulator("learnable", seed=0)
    easy = MechAccumulator("easy", seed=0)

    for acc, p in [(learnable, 0.5), (easy, 0.95)]:
        reward_fn = make_reward_fn(acc, num_generations=G)
        for step in range(steps):
            flat = _mock_step_rewards(p, n_prompts, G, rng)
            answers = ["x"] * len(flat)
            # The reward fn returns 0/1 by verifying completions; we bypass the
            # verifier by passing completions whose correctness == our planned
            # reward. Simplest: feed via a direct accumulator path is what we test,
            # so call add_groups directly to mirror the reward-fn reshape branch.
            acc.add_groups(np.asarray(flat).reshape(-1, G), step=step)
            # simulate trainer on_log scalars
            acc.add_step_scalars(step, grad_norm=0.5 + 0.01 * step,
                                 kl=0.001 * step, entropy=2.0 - 0.02 * step,
                                 loss=-0.1, reward=p + 0.001 * step)

    s_learn = learnable.summary()
    s_easy = easy.summary()

    # mechanistic ground truth: p~0.5 cohort has more variance + |adv| signal,
    # and FEWER zero-advantage / all-correct groups than the easy cohort.
    assert s_learn["mean_group_reward_var"] > s_easy["mean_group_reward_var"]
    assert s_learn["mean_abs_advantage"] > s_easy["mean_abs_advantage"]
    assert s_learn["zero_advantage_group_frac"] < s_easy["zero_advantage_group_frac"]
    assert s_easy["all_correct_group_frac"] > s_learn["all_correct_group_frac"]

    # trainer scalars were captured
    assert s_learn["n_steps_logged"] == steps
    assert not np.isnan(s_learn["mean_grad_norm"])
    assert not np.isnan(s_learn["final_kl"])

    # persistence round-trips
    with tempfile.TemporaryDirectory() as d:
        learnable.save(d)
        easy.save(d)
        loaded = load_mech_summaries(d)
        assert "learnable" in loaded and "easy" in loaded
        assert loaded["learnable"]["mean_group_reward_var"] > \
               loaded["easy"]["mean_group_reward_var"]


def test_reward_fn_reshape_branch():
    """Exercise make_reward_fn's actual reshape path (not add_groups directly):
    feed it completions that the GSM8K verifier scores, and assert the
    accumulator received groups."""
    from lad.grpo_train import make_reward_fn

    acc = MechAccumulator("c", seed=0)
    G = 4
    reward_fn = make_reward_fn(acc, num_generations=G)
    # 2 prompts x G=4 rollouts. gold answer "#### 7"; correct completions say 7.
    completions = ["#### 7", "#### 9", "#### 7", "#### 1",   # prompt 1
                   "#### 7", "#### 7", "#### 7", "#### 7"]   # prompt 2 (all correct)
    answers = ["#### 7"] * 8
    rewards = reward_fn(completions, answers)
    assert len(rewards) == 8
    assert acc.n_groups == 2
    s = acc.summary()
    # prompt 2 is all-correct -> at least one zero-advantage / all-correct group
    assert s["all_correct_group_frac"] >= 0.5
    assert s["zero_advantage_group_frac"] >= 0.5


if __name__ == "__main__":
    test_group_diagnostics_binary_variance()
    test_p_curve_variance_monotone()
    test_accumulator_fires_and_summarizes()
    test_reward_fn_reshape_branch()
    print("OK test_mech")

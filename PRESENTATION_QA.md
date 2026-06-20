# LAD — Judge Q&A Prep

> Read the **Primer** and **Cheat sheet** first, then the Q&A. Answers are written so you can almost say them aloud.

---

## 30-second pitch (memorize this)
"Frontier labs spend a fortune buying data to improve models with reinforcement learning — but they only find out if a batch of data *worked* after running the full, expensive training. **LAD is a cheap litmus test that predicts the payoff before you commit.** Using just a few quick attempts per problem — no training, no gradients — it scores how much a dataset will improve the model, and it predicts that lift better than methods that cost a hundred times more. It's cheap *and* accurate, which nobody had before."

## 2-minute version
Add: the *why* — when a model trains with RL, it only learns from problems it gets right *sometimes* (not always, not never). LAD measures how much of your data sits in that "productive struggle" zone, adjusts for whether the data has room to improve and isn't repetitive, and turns that into a single score. We validated it three ways: it predicts the actual measured lift on held-out data (correlation 0.89), it agrees with what's happening inside training, and when you *pick* data by LAD you actually get better models. It costs minutes instead of hours/days.

## The analogy (use this with non-technical judges)
"It's like a coach scouting a player before a training camp. A drill the player nails every time teaches nothing. A drill they fail every time is just demoralizing — too hard. The drills where they succeed *about half the time* — that's where they improve fastest. LAD measures how much of your training data sits in that sweet spot — and whether the drills are varied, not the same drill ten times — from a quick scouting session, before you run the expensive camp."

---

## Cheat sheet — the 8 numbers to know cold
- **0.89** — LAD's held-out rank correlation (Spearman ρ) with actual post-RL lift. *(>0.6 is "strong"; 1.0 is perfect.)*
- **0.45 / 0.34 / 0.29** — the best cheap competitors (reward variance / pass-rate / diversity). LAD nearly doubles them.
- **0.72–0.84** — the *expensive* methods (gradient influence, small-scale RL). LAD beats them **while costing ~1/125th** as much.
- **+12.6 vs +6.0 vs +0.9** — accuracy lift (in points) when you *select* training data by **top-LAD vs random vs bottom-LAD**. Proof it's causal, not just correlated.
- **k = 32** — attempts per problem to compute LAD (and we showed even **k=8** is near-optimal → extremely cheap).
- **19 cohorts, 1 base model, 1 RL config** — the controlled experiment (cohort = the only thing we change).
- **Qwen2.5-1.5B** — the base model. **GSM8K** — the dataset.
- **No training, no gradients** — LAD is pure inference (just generating answers).

---

## Primer (so YOU understand it)
- **RL (reinforcement learning) for LLMs:** instead of showing the model the answer, you let it *attempt* a task, *score* the attempt (reward), and nudge it toward what scored well. Repeat over many problems → it gets better.
- **Verifier:** for math/code, scoring is automatic — just check if the final answer is correct. (GSM8K = grade-school math with a single numeric answer, so a checker says right/wrong.)
- **GRPO** (Group Relative Policy Optimization): the popular, efficient RL algorithm (from DeepSeek) used to train reasoning models. For each problem it generates a **group** of attempts (say 8), scores them, and teaches the model to favor attempts that beat the group's average. *"Group-relative"* = each attempt judged against its siblings on the same problem.
- **The key insight LAD is built on:** if all 8 attempts are correct (or all wrong), there's no spread in the group → **nothing to learn** from that problem. The useful problems are the ones the model gets right *sometimes*. The learning signal per problem scales as **p·(1−p)**, where p = the model's success rate: zero at p=0, zero at p=1, biggest at p=0.5. That's the "learnable band."
- **"Lift":** accuracy *after* RL minus accuracy *before*, on a fixed held-out test set. This is the thing we're predicting.
- **"Cohort":** a fixed-size batch of training problems — our unit of analysis. We make many cohorts that differ in one property (difficulty, diversity, noise) and see which ones lift the model.
- **p̂ (p-hat):** estimated success rate = (successful attempts) / (total attempts) per problem.
- **Vendi score:** a diversity measure — roughly, the *effective number of distinct items* in a set (so 10 near-duplicates ≈ 1, 10 varied items ≈ 10).
- **LAD = Learnable Advantage Density.**

---

## The Q&A

### 1. What problem are you trying to solve?
**Say:** "Frontier labs improve models with reinforcement learning on curated data they *buy* from domain experts — it's expensive, and a single training run takes hours to days of GPU time. Today the only reliable way to know if a batch of data will actually help is to run that full training and look. We're solving the *pre-flight* problem: **cheaply predict, before you commit compute, whether a dataset will move the model** — and by how much."
**Why it matters:** "It directly serves data marketplaces and labs — screen or price data by its expected payoff instead of buying blind."

### 2. What are the existing techniques, and how good are they?
**Say:** "They fall on a cost-vs-accuracy spectrum.
- **Nearly free but weak:** token/answer length, raw pass-rate, simple difficulty — correlations with actual lift around **0.05 to 0.34**.
- **Diversity metrics** (like Vendi or dedup) — capture variety but ignore *learnability* — about **0.29**.
- **Reward variance / entropy** from a few attempts — closer to the right idea, about **0.45**, but it's a known *weak proxy*: it ignores whether there's room to improve and whether the data is redundant.
- **Expensive and accurate:** gradient-based influence (LESS), datamodels, Data Shapley, or actually running a small-scale RL — these predict well, **0.72 to 0.84**, but cost 100–1000× more because they need gradients or full training runs.
- The **oracle** is just running the full RL — perfect, but that defeats the purpose; you wanted to know *before*.
**The gap: nothing was both cheap and predictive.** That's the hole LAD fills."

### 3. What is your proposed solution? Is there a formula? How did you show it's better?
**Say:** "LAD — Learnable Advantage Density. Three factors, computed from a few attempts per problem, **no training**:
1. **Advantage energy — p̂(1−p̂):** is there a learning signal at all? (straight from the GRPO math)
2. **Headroom — (1−p̂)^γ:** is the difficulty pointed where there's room to improve?
3. **Effective diversity — Vendi/|C|:** is the data varied, not redundant?
Combine them: **LAD = average over tasks of [ p̂(1−p̂)·(1−p̂)^γ ] × (Vendi/|C|)^β**, with γ=1, β=0.5."
**How we proved it's better:** "A controlled experiment. We built 19 cohorts that each vary one property — difficulty, diversity, noise. We trained the **same** model with the **same** RL settings on each cohort and measured the **actual** lift. Then we checked how well each metric *predicts* that lift on cohorts it never saw (leave-one-cohort-out). **LAD: 0.89. Reward variance: 0.45. Pass-rate: 0.34. Diversity: 0.29.** And it beats even the expensive methods (0.72–0.84) at a fraction of the cost — it sits *above* the cost-vs-accuracy frontier."
**Causal kicker:** "Correlation isn't enough, so we also *intervened*: select training data by top-LAD vs random vs bottom-LAD and train. **Top-LAD gave +12.6 points, random +6.0, bottom-LAD +0.9.** Choosing data by LAD *causes* better models."

### 4. What dataset are you using for the RL tests?
**Say:** "**GSM8K** — a standard, freely available set of grade-school math word problems. We use it because each problem has a single checkable numeric answer, so the reward is automatic and objective. We sample training cohorts from its training split and always evaluate on a fixed held-out GSM8K test set. We also added synthetic, noisy-label, broken-verifier, and adversarial cohorts on purpose — to check LAD isn't fooled by bad data."
**If pressed on generality:** "The method is domain-agnostic given any verifier; code (HumanEval+/MBPP+) is the natural next domain."

### 5. Which base model? Same model across tests? Tested across scales?
**Say:** "**Qwen2.5-1.5B-Instruct** — a small, fast open model, chosen so we could run *many* controlled RL runs within our compute budget. Crucially, we hold the **same base model and the same GRPO config constant across every cohort** — the cohort is the *only* variable. That's what makes the comparison clean and the result trustworthy."
**On scales (be honest):** "Our deep, controlled study is at 1.5B. Scaling to 3B and 7B is our immediate next step — and because LAD is derived from the *math* of GRPO, which doesn't depend on model size, we expect it to transfer. But we're careful not to claim beyond what we've measured." *(Judges respect this honesty far more than overclaiming.)*

---

## More questions you'll likely get

**Q: Why should I believe the numbers — aren't you just overfitting?**
"We never test on cohorts we fit on — every number is **leave-one-cohort-out**, held out. On top of that we did the *causal* intervention (selecting by LAD really trains better) and **ablations**: remove the headroom term and ρ drops 0.89→0.71; remove diversity and it drops to 0.68. Each piece earns its place."

**Q: What's actually novel? Isn't 'p(1−p) matters' known?**
"The insight that mid-difficulty helps is known — prior work like PODS, DOTS, LearnAlign touches pieces. What's new is a **cheap, dataset-level *predictor of realized lift*** that combines learnability + headroom + diversity with a first-principles derivation, and is **validated causally** and positioned on the cost-quality frontier. We're not claiming p(1−p); we're claiming a usable instrument."

**Q: How cheap is it, concretely?**
"Pure inference — ~32 generations per problem, no backprop, no training. We showed even **8 generations** is near-optimal (0.87 vs 0.90), so it's minutes of compute. Roughly **1/125th** the cost of a small-scale-RL proxy that predicts *worse*."

**Q: Does it agree with what's happening inside training? (mechanistic)**
"Yes — we logged the actual GRPO learning signal during training (the group advantage variance). LAD correlates with it at **0.86**. So LAD isn't a black-box coincidence; it measures the thing the optimizer actually uses."

**Q: Where does LAD fail? (limitations — answer confidently)**
"Three honest ones: (1) it assumes a *trustworthy verifier* — garbage scoring fools any rollout method, though our broken-verifier cohorts show LAD degrades gracefully; (2) so far it's one domain (math) and one scale (1.5B) — generalization is the next experiment; (3) it predicts the *average* learnable signal, so a cohort that's individually learnable but off-distribution for your eval could still mislead. We built adversarial cohorts specifically to probe these."

**Q: Isn't this just difficulty filtering / 'keep the medium-hard problems'?**
"That's the naive version — pure pass-rate only gets **0.34**. LAD beats it by 0.55 because difficulty alone misses two things: *headroom* (a problem at exactly 50% with no room to grow vs one with upside) and *redundancy* (ten copies of a good problem teach less than ten varied ones). The ablations prove both matter."

**Q: Why not just run a small RL to predict the big one?**
"That 'small-RL proxy' costs ~1000× more than LAD and still only hits **0.84** — lower than LAD's 0.89. Cheaper *and* better is the whole point."

**Q: How would this be used in the real world / how do you make money?**
"A data-screening service for labs and marketplaces: before you buy or train on a dataset, LAD scores its expected lift in minutes. Price data by payoff, or filter a big pool down to the high-LAD slice. Our live 'score-a-cohort' tool in the demo is exactly that."

**Q: What is GRPO, in one sentence?** (in case they test *you*)
"The standard, efficient RL algorithm for reasoning LLMs: for each problem it samples a group of answers and pushes the model toward the ones that beat the group's average."

**Q: Why k=32 attempts?**
"Enough to estimate each problem's success rate reliably, and it lets us study how few we can get away with — we found 8 is already near-optimal, so it's cheap."

**Q: What's Vendi?**
"A diversity score — the *effective number of distinct items* in a set, from the eigenvalues of a similarity matrix. Ten duplicates score ~1; ten varied items score ~10."

**Q: How long did the whole study take / what compute?**
"It runs on a single small open model on the 4×GB200 box: a one-time cheap rollout pass, then many short identical RL runs across the cohorts in parallel. The *point* is that LAD itself — the thing a lab would actually use — is just the cheap rollout pass."

---

## How to handle a question you can't answer
- **Bridge to a strength:** "We haven't measured that specific case yet — but it's exactly what our held-out / ablation setup is built to probe. What we *can* show is…" then point to a chart.
- **Be honest, then redirect:** "I don't want to overclaim — that's our next experiment. The reason we expect X is the derivation is scale/domain-independent."
- Never bluff a number. If unsure, say "around" and give the ballpark you know (0.89, 0.45, +12.6).

## If a judge really corners you
Fall back to the three pillars, in order: **(1) it predicts held-out lift (0.89), (2) it agrees with the optimizer's internal signal (0.86), (3) selecting by it causes better training (+12.6 vs +6.0).** Cheap, predictive, causal. Everything else is detail.

## Questions to invite (you look strong answering these)
- "Want to score a dataset live?" → use the demo's score-a-cohort tool.
- "Want to see where it sits vs everything else?" → the cost-vs-accuracy Pareto chart (LAD top-left, above the frontier).

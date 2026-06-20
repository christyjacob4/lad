"""All figures + tables from RESEARCH_PLAN, built from results/summary.json
(+ cached data for the derivation/Vendi figures). No GPU.

Figures (saved to <outdir>/figs/):
  derivation.png            p(1-p)(1-p)^gamma vs p (the metric's shape)
  cohort_map.png            cohort difficulty x diversity scatter, colored by lift
  lad_vs_lift.png           LAD vs lift scatter with LOCO fit line
  predicted_vs_actual.png   per-metric LOCO predicted vs actual (LAD, var, div, passrate)
  residuals.png             LAD LOCO residuals vs predicted
  pareto.png                log(cost) vs LOCO rho, error bars (the frontier)
  baseline_bars.png         per-metric LOCO Spearman bar chart (sorted)
  ablation_bars.png         LAD ablation LOCO Spearman bars
  rollout_budget.png        rho_lift + rank-stability vs k
  learning_curves.png       per-cohort train-reward trace (from mech logs)
  advantage_variance.png    LAD vs logged GRPO advantage variance (mechanistic)
  causal_bars.png           selection-condition lift bars (top/random/bottom...)
  dose_response.png         dose-response lift vs LAD bucket
  vendi_spectrum.png        Vendi eigenvalue spectrum for high vs low diversity cohort
  noise_dup_stress.png      noisy/dup cohorts: variance rank error vs LAD rank error

Tables (saved to <outdir>/tables/ as .md):
  cohort_table, metric_table, prediction_table, ablation_table, baseline_table,
  robustness_table, cost_table, acceptance_table.
"""

import json
import os
import numpy as np


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def build_all(outdir="results", datadir="data/run"):
    s = json.load(open(os.path.join(outdir, "summary.json")))
    figs = os.path.join(outdir, "figs")
    tabs = os.path.join(outdir, "tables")
    os.makedirs(figs, exist_ok=True)
    os.makedirs(tabs, exist_ok=True)

    fig_derivation(figs)
    if s.get("status") == "insufficient_lifts":
        print("[build_figures] insufficient lifts; emitted derivation only")
        return
    names = s["cohorts"]
    lifts = np.array([s["lifts"][c] for c in names])
    mv = {m: np.array(v, float) for m, v in s["metric_values"].items()}

    fig_cohort_map(figs, s, names, lifts, mv)
    fig_lad_vs_lift(figs, s, lifts, mv)
    fig_predicted_vs_actual(figs, s, lifts)
    fig_residuals(figs, s, lifts)
    fig_pareto(figs, s)
    fig_baseline_bars(figs, s)
    fig_ablation_bars(figs, s)
    fig_rollout_budget(figs, s)
    fig_advantage_variance(figs, s, names, mv)
    fig_learning_curves(figs, s)
    fig_causal(figs, s)
    fig_dose_response(figs, s)
    fig_vendi_spectrum(figs, s, names, datadir)
    fig_noise_dup_stress(figs, s, names, lifts, mv)

    write_tables(tabs, s, names, lifts, mv)
    print(f"[build_figures] wrote figures -> {figs}/ and tables -> {tabs}/")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig_derivation(figs):
    plt = _mpl()
    p = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(6, 4))
    for g in [0.0, 0.5, 1.0, 2.0]:
        s = p * (1 - p) * (1 - p) ** g
        peak = p[np.argmax(s)]
        ax.plot(p, s, label=f"gamma={g} (peak p={peak:.2f})")
    ax.set_xlabel("base-model pass-rate p")
    ax.set_ylabel("per-task learnable signal  p(1-p)(1-p)^gamma")
    ax.set_title("LAD derivation: advantage energy x headroom")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "derivation.png"), dpi=140); plt.close(fig)


def fig_cohort_map(figs, s, names, lifts, mv):
    plt = _mpl()
    p = mv.get("mean_pass_rate")
    div = mv.get("embedding_diversity", mv.get("dedup_score"))
    if p is None or div is None:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(p, div, c=lifts, s=90, cmap="viridis", edgecolor="k")
    for i, n in enumerate(names):
        ax.annotate(n, (p[i], div[i]), fontsize=6, xytext=(3, 3),
                    textcoords="offset points")
    fig.colorbar(sc, label="measured GRPO lift")
    ax.set_xlabel("mean pass-rate (difficulty)"); ax.set_ylabel("effective diversity (Vendi)")
    ax.set_title("Cohort map: difficulty x diversity, colored by lift")
    fig.tight_layout(); fig.savefig(os.path.join(figs, "cohort_map.png"), dpi=140); plt.close(fig)


def fig_lad_vs_lift(figs, s, lifts, mv):
    plt = _mpl()
    if "LAD" not in mv:
        return
    x = mv["LAD"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, lifts, s=80, edgecolor="k", color="crimson")
    if np.std(x) > 1e-12:
        a, b = np.polyfit(x, lifts, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, a * xs + b, "--", color="gray")
    rho = s["predictive"]["LAD"]["rho_insample"]
    ax.set_xlabel("LAD score"); ax.set_ylabel("measured GRPO lift")
    ax.set_title(f"LAD vs lift (in-sample Spearman rho={rho:.2f})")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "lad_vs_lift.png"), dpi=140); plt.close(fig)


def fig_predicted_vs_actual(figs, s, lifts):
    plt = _mpl()
    metrics = [m for m in ["LAD", "naive_variance", "embedding_diversity",
                           "mean_pass_rate"] if m in s["loco_preds"]]
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics), 4.2))
    if len(metrics) == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        preds = np.array(s["loco_preds"][m], float)
        ax.scatter(preds, lifts, s=55, edgecolor="k", alpha=0.85)
        lo = min(np.nanmin(preds), lifts.min()); hi = max(np.nanmax(preds), lifts.max())
        ax.plot([lo, hi], [lo, hi], "--", color="gray")
        r = s["predictive"][m]
        ax.set_title(f"{m}\nrho={r['rho_loco']:.2f} R2={r['r2_loco']:.2f}", fontsize=9)
        ax.set_xlabel("LOCO predicted lift"); ax.set_ylabel("actual lift")
    fig.suptitle("Leave-one-cohort-out: predicted vs actual lift")
    fig.tight_layout(); fig.savefig(os.path.join(figs, "predicted_vs_actual.png"), dpi=140); plt.close(fig)


def fig_residuals(figs, s, lifts):
    plt = _mpl()
    if "LAD" not in s["loco_preds"]:
        return
    preds = np.array(s["loco_preds"]["LAD"], float)
    resid = lifts - preds
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color="gray", ls="--")
    ax.scatter(preds, resid, s=60, edgecolor="k", color="steelblue")
    ax.set_xlabel("LOCO predicted lift"); ax.set_ylabel("residual (actual - predicted)")
    ax.set_title("LAD LOCO residuals")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "residuals.png"), dpi=140); plt.close(fig)


def fig_pareto(figs, s):
    plt = _mpl()
    cost = s["cost"]; pr = s["predictive"]
    pts = []
    for m in pr:
        if m in cost:
            ci = pr[m]["bootstrap_rho_ci"]
            pts.append((m, cost[m]["log_cost"], pr[m]["rho_loco"],
                        ci.get("lo"), ci.get("hi")))
    # deferred expensive points (positioned by known cost; rho left as None marker)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m, lc, rho, lo, hi in pts:
        is_lad = m == "LAD"
        yerr = None
        if lo is not None and hi is not None and not (np.isnan(lo) or np.isnan(hi)):
            yerr = [[max(0, rho - lo)], [max(0, hi - rho)]]
        ax.errorbar(lc, rho, yerr=yerr, fmt="*" if is_lad else "o",
                    ms=18 if is_lad else 7,
                    color="crimson" if is_lad else "steelblue",
                    ecolor="gray", capsize=2, zorder=3 if is_lad else 2)
        if is_lad or m in ("naive_variance", "embedding_diversity",
                           "mean_pass_rate", "token_length"):
            ax.annotate(m, (lc, rho), fontsize=7, xytext=(5, 4), textcoords="offset points")
    # expensive deferred markers
    for m in ["gradient_norm", "small_rl_lift", "datamodels", "full_rl_oracle"]:
        if m in cost:
            ax.axvline(cost[m]["log_cost"], color="lightgray", ls=":", zorder=0)
            ax.annotate(m, (cost[m]["log_cost"], ax.get_ylim()[0]), rotation=90,
                        fontsize=6, color="gray", va="bottom")
    ax.set_xlabel("log10 cost  (forward-pass-equivalents; training-free metrics on the left)")
    ax.set_ylabel("predictive power  (LOCO Spearman rho)")
    ax.set_title("Cost-quality frontier: LAD (star) at rollout cost, high rho")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "pareto.png"), dpi=140); plt.close(fig)


def fig_baseline_bars(figs, s):
    plt = _mpl()
    pr = s["predictive"]
    base = [m for m in pr if not m.startswith("LAD") or m == "LAD"]
    base = sorted(base, key=lambda m: -(pr[m]["rho_loco"] if pr[m]["rho_loco"] == pr[m]["rho_loco"] else -9))
    vals = [pr[m]["rho_loco"] for m in base]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["crimson" if m == "LAD" else "steelblue" for m in base]
    ax.bar(range(len(base)), vals, color=colors, edgecolor="k")
    ax.set_xticks(range(len(base))); ax.set_xticklabels(base, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("LOCO Spearman rho"); ax.axhline(0, color="k", lw=0.6)
    ax.set_title("LAD vs all baselines (held-out Spearman)")
    fig.tight_layout(); fig.savefig(os.path.join(figs, "baseline_bars.png"), dpi=140); plt.close(fig)


def fig_ablation_bars(figs, s):
    plt = _mpl()
    pr = s["predictive"]
    abl = [m for m in pr if m.startswith("LAD")]
    abl = sorted(abl, key=lambda m: -(pr[m]["rho_loco"] if pr[m]["rho_loco"] == pr[m]["rho_loco"] else -9))
    if not abl:
        return
    vals = [pr[m]["rho_loco"] for m in abl]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["crimson" if m == "LAD" else "slategray" for m in abl]
    ax.bar(range(len(abl)), vals, color=colors, edgecolor="k")
    ax.set_xticks(range(len(abl))); ax.set_xticklabels(abl, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("LOCO Spearman rho"); ax.axhline(0, color="k", lw=0.6)
    ax.set_title("LAD ablations (removing headroom/diversity should hurt)")
    fig.tight_layout(); fig.savefig(os.path.join(figs, "ablation_bars.png"), dpi=140); plt.close(fig)


def fig_rollout_budget(figs, s):
    plt = _mpl()
    rel = s.get("reliability", {})
    if not rel or "ks" not in rel:
        return
    ks = rel["ks"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(ks, [rel["rho_lift"][str(k)] if str(k) in rel["rho_lift"] else rel["rho_lift"].get(k)
                 for k in ks], "-o", label="LOCO rho vs lift", color="crimson")
    ax.plot(ks, [rel["rho_rank_vs_kmax"][str(k)] if str(k) in rel["rho_rank_vs_kmax"]
                 else rel["rho_rank_vs_kmax"].get(k) for k in ks],
            "--s", label="rank stability vs k_max", color="steelblue")
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel("rollout budget k"); ax.set_ylabel("Spearman rho")
    ax.set_title("Reliability: LAD ranking stable from small k")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "rollout_budget.png"), dpi=140); plt.close(fig)


def fig_advantage_variance(figs, s, names, mv):
    plt = _mpl()
    mech = s.get("mech_summaries", {})
    if not mech or "LAD" not in mv:
        return
    mc = [c for c in names if c in mech]
    if len(mc) < 3:
        return
    x = np.array([mv["LAD"][names.index(c)] for c in mc])
    y = np.array([mech[c].get("mean_group_reward_var", np.nan) for c in mc])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=70, edgecolor="k", color="darkgreen")
    rho = s.get("mechanistic", {}).get("mean_group_reward_var", {}).get("spearman", float("nan"))
    ax.set_xlabel("LAD (pre-training)"); ax.set_ylabel("logged GRPO group reward variance")
    ax.set_title(f"Mechanistic: LAD vs realized advantage variance (rho={rho:.2f})")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "advantage_variance.png"), dpi=140); plt.close(fig)


def fig_learning_curves(figs, s):
    plt = _mpl()
    mech = s.get("mech_summaries", {})
    if not mech:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for c, m in mech.items():
        tr = m.get("step_reward_trace")
        if not tr:
            continue
        steps = [t[0] for t in tr if t[0] is not None]
        rew = [t[1] for t in tr if t[0] is not None]
        if len(steps) >= 2:
            ax.plot(steps, rew, label=c, alpha=0.7)
            plotted += 1
    if plotted == 0:
        plt.close(fig); return
    ax.set_xlabel("GRPO step"); ax.set_ylabel("mean train reward")
    ax.set_title("Per-cohort GRPO learning curves")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "learning_curves.png"), dpi=140); plt.close(fig)


def fig_causal(figs, s):
    plt = _mpl()
    causal = s.get("causal", {})
    sl = causal.get("selection_lifts", {})
    order = ["top_lad", "highest_diversity", "highest_naive_variance",
             "highest_pass_rate", "random", "easy", "hard", "lowest_pass_rate",
             "bottom_lad"]
    conds = [c for c in order if c in sl] + [c for c in sl if c not in order and not c.startswith("dose")]
    if not conds:
        return
    vals = [sl[c] for c in conds]
    colors = ["crimson" if c == "top_lad" else "steelblue" for c in conds]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(conds)), vals, color=colors, edgecolor="k")
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("measured GRPO lift"); ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Causal: selecting by LAD vs other rules (equal RL compute)")
    fig.tight_layout(); fig.savefig(os.path.join(figs, "causal_bars.png"), dpi=140); plt.close(fig)


def fig_dose_response(figs, s):
    plt = _mpl()
    dose = s.get("causal", {}).get("dose_response", {})
    if not dose:
        return
    order = ["dose_bottom25", "dose_random25", "dose_top50", "dose_top25", "dose_top10"]
    keys = [d for d in order if d in dose]
    vals = [dose[d] for d in keys]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(len(keys)), vals, "-o", color="crimson")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("measured GRPO lift")
    ax.set_title("Dose-response: more top-LAD data -> more lift")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "dose_response.png"), dpi=140); plt.close(fig)


def fig_vendi_spectrum(figs, s, names, datadir):
    plt = _mpl()
    try:
        meta = json.load(open(os.path.join(datadir, "cohort_meta.json")))
        embs = np.load(os.path.join(datadir, "pool_embeddings.npy"))
    except Exception:
        return
    cands = [("div_high", "high diversity"), ("div_low", "low diversity"),
             ("dup_heavy", "duplicate-heavy")]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    any_plotted = False
    for cname, lbl in cands:
        if cname not in meta:
            continue
        idx = np.asarray(meta[cname]["indices"], int)
        X = embs[idx]
        X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
        K = (X @ X.T) / len(X)
        ev = np.sort(np.linalg.eigvalsh(0.5 * (K + K.T)))[::-1]
        ev = ev[ev > 1e-12]
        ax.plot(ev / ev.sum(), label=lbl)
        any_plotted = True
    if not any_plotted:
        plt.close(fig); return
    ax.set_yscale("log"); ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("normalized eigenvalue (log)")
    ax.set_title("Vendi spectrum: diverse cohort has flatter eigenvalues")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "vendi_spectrum.png"), dpi=140); plt.close(fig)


def fig_noise_dup_stress(figs, s, names, lifts, mv):
    plt = _mpl()
    if "LAD" not in mv or "naive_variance" not in mv:
        return
    from scipy.stats import rankdata
    stress = [c for c in names if any(t in s["tags"].get(c, [])
              for t in ["noisy", "broken_verifier", "reward_hack"]) or "dup" in c]
    if not stress:
        return
    lad_rank = rankdata(mv["LAD"]); var_rank = rankdata(mv["naive_variance"])
    lift_rank = rankdata(lifts)
    idx = [names.index(c) for c in stress]
    lad_err = np.abs(lad_rank[idx] - lift_rank[idx])
    var_err = np.abs(var_rank[idx] - lift_rank[idx])
    x = np.arange(len(stress))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, var_err, 0.4, label="naive variance rank error", color="darkorange")
    ax.bar(x + 0.2, lad_err, 0.4, label="LAD rank error", color="crimson")
    ax.set_xticks(x); ax.set_xticklabels(stress, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("|metric rank - true lift rank|")
    ax.set_title("Stress: LAD ranks noisy/dup/adversarial cohorts better than variance")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "noise_dup_stress.png"), dpi=140); plt.close(fig)


# --------------------------------------------------------------------------
# Tables (markdown)
# --------------------------------------------------------------------------

def write_tables(tabs, s, names, lifts, mv):
    from lad import cost as CST  # noqa: F401 (kept for parity)

    # cohort table
    rows = ["| cohort | tags | mean p̂ | lift | seed σ | acc_before | acc_after |",
            "|---|---|---|---|---|---|---|"]
    for c in names:
        p = mv.get("mean_pass_rate")
        pv = p[names.index(c)] if p is not None else float("nan")
        rows.append(f"| {c} | {','.join(s['tags'].get(c, []))} | {pv:.3f} | "
                    f"{s['lifts'][c]:+.4f} | {s['seed_variance'].get(c,0):.4f} | "
                    f"{s['acc_before'][c]:.3f} | {s['acc_after'][c]:.3f} |")
    _w(tabs, "cohort_table.md", rows)

    # prediction table
    pr = s["predictive"]
    rows = ["| metric | rho_loco | Kendall(in) | R²_loco | RMSE | MAE | top-k prec | pairwise | boot95 CI | perm p |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    order = sorted(pr, key=lambda m: -(pr[m]["rho_loco"] if pr[m]["rho_loco"] == pr[m]["rho_loco"] else -9))
    for m in order:
        r = pr[m]; ci = r["bootstrap_rho_ci"]; pm = r["permutation"]
        rows.append(f"| {m} | {_n(r['rho_loco'])} | {_n(r['kendall_insample'])} | "
                    f"{_n(r['r2_loco'])} | {_n(r['rmse_loco'])} | {_n(r['mae_loco'])} | "
                    f"{_n(r['top_k_precision'])} | {_n(r['pairwise_accuracy'])} | "
                    f"[{_n(ci.get('lo'))},{_n(ci.get('hi'))}] | {_n(pm.get('p_value'))} |")
    _w(tabs, "prediction_table.md", rows)

    # baseline vs LAD paired table
    paired = s.get("paired_vs_LAD", {})
    rows = ["| baseline | Δrho (LAD−X) | 95% CI | P(LAD better) |", "|---|---|---|---|"]
    for m, pv in sorted(paired.items(), key=lambda kv: -(kv[1].get("delta_mean") or -9)):
        rows.append(f"| {m} | {_n(pv.get('delta_mean'))} | "
                    f"[{_n(pv.get('lo'))},{_n(pv.get('hi'))}] | {_n(pv.get('prob_a_better'))} |")
    _w(tabs, "baseline_table.md", rows)

    # ablation table
    rows = ["| ablation | rho_loco | R²_loco |", "|---|---|---|"]
    for m in [x for x in order if x.startswith("LAD")]:
        rows.append(f"| {m} | {_n(pr[m]['rho_loco'])} | {_n(pr[m]['r2_loco'])} |")
    _w(tabs, "ablation_table.md", rows)

    # cost table
    cost = s["cost"]
    rows = ["| metric | fwd passes | verifier | backward | GPU-s | $ | needs grad | needs RL | log cost |",
            "|---|---|---|---|---|---|---|---|---|"]
    for m, c in sorted(cost.items(), key=lambda kv: kv[1]["log_cost"]):
        rows.append(f"| {m} | {c['forward_passes']:.0f} | {c['verifier_calls']:.0f} | "
                    f"{c['backward_passes']:.0f} | {c['gpu_seconds']:.1f} | {c['dollars']:.4f} | "
                    f"{c['needs_gradients']} | {c['needs_rl']} | {c['log_cost']:.2f} |")
    _w(tabs, "cost_table.md", rows)

    # robustness table (reliability sweep)
    rel = s.get("reliability", {})
    if rel and "ks" in rel:
        rows = ["| k | LOCO rho vs lift | rank stability vs k_max |", "|---|---|---|"]
        for k in rel["ks"]:
            rl = rel["rho_lift"].get(str(k), rel["rho_lift"].get(k))
            rr = rel["rho_rank_vs_kmax"].get(str(k), rel["rho_rank_vs_kmax"].get(k))
            rows.append(f"| {k} | {_n(rl)} | {_n(rr)} |")
        _w(tabs, "robustness_table.md", rows)

    # causal table
    causal = s.get("causal", {})
    if causal.get("selection_lifts"):
        rows = ["| selection rule | lift | seed σ |", "|---|---|---|"]
        for c, lv in sorted(causal["selection_lifts"].items(), key=lambda kv: -kv[1]):
            rows.append(f"| {c} | {lv:+.4f} | {_n(causal.get('seed_variance', {}).get(c))} |")
        _w(tabs, "causal_table.md", rows)

    # acceptance table
    acc = s["acceptance_criteria"]
    rows = ["| Section-17 criterion | met | detail |", "|---|---|---|"]
    for k, v in acc.items():
        met = "✅" if v.get("met") is True else ("⬜ (n/a)" if v.get("met") is None else "❌")
        detail = v.get("detail", "") or "; ".join(f"{kk}={_n(vv)}" for kk, vv in v.items()
                                                  if kk not in ("met", "detail"))
        rows.append(f"| {k} | {met} | {detail} |")
    _w(tabs, "acceptance_table.md", rows)


def _w(tabs, name, rows):
    with open(os.path.join(tabs, name), "w") as f:
        f.write("\n".join(rows) + "\n")


def _n(x):
    if x is None:
        return "n/a"
    try:
        xf = float(x)
        if np.isnan(xf):
            return "n/a"
        return f"{xf:.3f}"
    except (TypeError, ValueError):
        return str(x)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--datadir", default="data/run")
    a = ap.parse_args()
    build_all(a.outdir, a.datadir)

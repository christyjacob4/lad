"""Master analysis driver (Claims 1-4 + Section-17 acceptance criteria).

Consumes:
  data/run/cohort_meta.json, pool_scores.json, pool_embeddings.npy
  results/lifts/lift_*.json          (main-sweep GRPO lifts)
  results/mech/*.json                (per-cohort mechanistic logs, if present)
  data/causal + results/causal/...   (causal selection lifts, if present)

Produces results/summary.json with EVERYTHING the PAPER needs:
  - per-metric LOCO Spearman/Kendall/Pearson/R2/RMSE/MAE + bootstrap CI + perm p
  - paired-bootstrap "LAD beats X" for every baseline
  - confounder controls (partial Spearman + multivariate regression)
  - mechanistic correlations (LAD vs logged advantage variance / |adv|)
  - reliability sweep (rollout-budget k) if k>=16 cached
  - causal selection lift table + dose-response, if causal lifts present
  - cost table for the Pareto frontier
  - Section-17 acceptance-criteria checklist (auto-evaluated)

Figures + tables are emitted by build_figures.py (called at the end unless
--no_figures).
"""

import argparse
import glob
import json
import os
import numpy as np


def load_lifts(results_dir, prefix="lift_"):
    by = {}
    for path in glob.glob(os.path.join(results_dir, f"{prefix}*.json")):
        try:
            r = json.load(open(path))
        except Exception:
            continue
        by.setdefault(r["cohort"], []).append(r)
    lifts, accb, acca, seedvar = {}, {}, {}, {}
    for c, runs in by.items():
        ls = [r["lift"] for r in runs]
        lifts[c] = float(np.mean(ls))
        accb[c] = float(np.mean([r["acc_before"] for r in runs]))
        acca[c] = float(np.mean([r["acc_after"] for r in runs]))
        seedvar[c] = float(np.std(ls)) if len(ls) > 1 else 0.0
    return lifts, accb, acca, seedvar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data/run")
    ap.add_argument("--results", default="results/lifts")
    ap.add_argument("--mech", default="results/mech")
    ap.add_argument("--causaldir", default="data/causal")
    ap.add_argument("--causal_results", default="results/causal")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--no_figures", action="store_true")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from lad.baselines import CohortFeatures, compute_all, as_aligned_arrays
    from lad.metric import vendi_score
    from lad import predictive as P
    from lad import cost as CST
    from lad import mech as M
    from compute_all_metrics import build_cohort_features

    # ---- lifts + features ----
    lifts_d, accb_d, acca_d, seedvar_d = load_lifts(args.results)
    features, meta = build_cohort_features(args.datadir)
    names = sorted(c for c in features if c in lifts_d)
    if len(names) < 3:
        print(f"Only {len(names)} cohorts with lifts; need >=3. Found: {names}")
        # still write a partial summary so the finisher has something
        json.dump({"n_cohorts": len(names), "cohorts": names,
                   "status": "insufficient_lifts"},
                  open(os.path.join(args.outdir, "summary.json"), "w"), indent=2)
        return

    lifts = np.array([lifts_d[c] for c in names])
    feats = {c: features[c] for c in names}
    table = compute_all(feats)
    arrs = as_aligned_arrays(table, names)

    # ---- predictive comparison (all metrics, LAD focus) ----
    cmp = P.compare_all(arrs, lifts, focus="LAD", n_boot=args.n_boot)
    reports = cmp["reports"]

    # ---- confounder controls ----
    confound = {}
    if "LAD" in arrs:
        controls = []
        ctrl_names = []
        for cn in ["token_length", "embedding_diversity", "mean_pass_rate"]:
            if cn in arrs:
                controls.append(arrs[cn]); ctrl_names.append(cn)
        confound["partial_spearman_LAD"] = P.partial_spearman(arrs["LAD"], lifts, controls)
        confound["control_vars"] = ctrl_names
        preds = {"LAD": arrs["LAD"]}
        for cn in ctrl_names:
            preds[cn] = arrs[cn]
        confound["multivariate_regression"] = P.multivariate_regression(lifts, preds)

    # ---- mechanistic (Claim 1) ----
    mech_corr = {}
    mech_summaries = M.load_mech_summaries(args.mech) if os.path.isdir(args.mech) else {}
    if mech_summaries:
        mc_names = [c for c in names if c in mech_summaries]
        if len(mc_names) >= 3 and "LAD" in arrs:
            lad_vec = np.array([arrs["LAD"][names.index(c)] for c in mc_names])
            for q in ["mean_group_reward_var", "mean_abs_advantage",
                      "zero_advantage_group_frac", "train_reward_improvement"]:
                yv = np.array([mech_summaries[c].get(q, np.nan) for c in mc_names])
                rho, p = P.spearman(lad_vec, yv)
                mech_corr[q] = {"spearman": rho, "p": p, "n": len(mc_names)}
            # also mech quantity vs realized lift
            mech_corr["_lift_vs_advvar"] = dict(zip(
                ["spearman", "p"],
                P.spearman(np.array([mech_summaries[c]["mean_group_reward_var"]
                                     for c in mc_names]),
                           np.array([lifts_d[c] for c in mc_names]))))

    # ---- reliability sweep (rollout budget) ----
    reliability = {}
    try:
        from lad import reliability as R
        cohort_correct = {c: feats[c].correct for c in names if feats[c].correct is not None}
        cohort_emb = {c: feats[c].embeddings for c in names}
        kmax = min(corr.shape[1] for corr in cohort_correct.values())
        if cohort_correct and kmax >= 8:
            reliability = R.rollout_budget_sweep(
                cohort_correct, cohort_emb, lifts_d,
                ks=tuple(k for k in (2, 4, 8, 16, 32) if k <= kmax), n_seeds=5)
    except Exception as e:
        reliability = {"error": str(e)}

    # ---- causal selection (Claim 4) ----
    causal = load_causal(args.causal_results)

    # ---- cost table ----
    cost = CST.cost_table(list(arrs) + ["gradient_norm", "small_rl_lift",
                                        "datamodels", "full_rl_oracle"])

    # ---- Section-17 acceptance criteria (auto-eval) ----
    accept = evaluate_acceptance(reports, cmp["paired_vs_focus"], confound,
                                 mech_corr, causal)

    summary = {
        "n_cohorts": len(names),
        "cohorts": names,
        "lifts": {c: lifts_d[c] for c in names},
        "seed_variance": {c: seedvar_d[c] for c in names},
        "acc_before": {c: accb_d[c] for c in names},
        "acc_after": {c: acca_d[c] for c in names},
        "tags": {c: meta[c].get("tags", []) for c in names},
        "metric_values": {m: arrs[m].tolist() for m in arrs},
        "predictive": {name: {k: rep[k] for k in
                       ["rho_loco", "p_loco", "rho_insample", "kendall_insample",
                        "pearson_loco", "r2_loco", "rmse_loco", "mae_loco",
                        "calibration_slope", "calibration_intercept",
                        "top_k_precision", "pairwise_accuracy",
                        "bootstrap_rho_ci", "permutation"]}
                       for name, rep in reports.items()},
        "loco_preds": {name: reports[name]["preds"] for name in reports},
        "paired_vs_LAD": cmp["paired_vs_focus"],
        "confounders": confound,
        "mechanistic": mech_corr,
        "mech_summaries": mech_summaries,
        "reliability": reliability,
        "causal": causal,
        "cost": cost,
        "acceptance_criteria": accept,
        "headline": build_headline(reports, cmp["paired_vs_focus"], causal, mech_corr),
    }
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(_clean(summary), open(os.path.join(args.outdir, "summary.json"), "w"), indent=2)

    _print_headline(summary)

    if not args.no_figures:
        try:
            import build_figures
            build_figures.build_all(args.outdir, args.datadir)
            print(f"[analyze] figures + tables -> {args.outdir}/")
        except Exception as e:
            print(f"[analyze] figure build failed (non-fatal): {e}")


def load_causal(causal_results):
    if not os.path.isdir(causal_results):
        return {}
    lifts, accb, acca, sv = load_lifts(causal_results)
    if not lifts:
        return {}
    out = {"selection_lifts": lifts, "acc_before": accb, "acc_after": acca,
           "seed_variance": sv}
    # headline gaps
    def g(a, b):
        if a in lifts and b in lifts:
            return float(lifts[a] - lifts[b])
        return None
    out["top_minus_random"] = g("top_lad", "random")
    out["top_minus_bottom"] = g("top_lad", "bottom_lad")
    out["top_minus_variance"] = g("top_lad", "highest_naive_variance")
    out["top_minus_diversity"] = g("top_lad", "highest_diversity")
    out["top_minus_passrate"] = g("top_lad", "highest_pass_rate")
    # dose-response monotonicity
    dose = ["dose_bottom25", "dose_random25", "dose_top50", "dose_top25", "dose_top10"]
    out["dose_response"] = {d: lifts[d] for d in dose if d in lifts}
    return out


def evaluate_acceptance(reports, paired, confound, mech_corr, causal):
    """Auto-evaluate the Section-17 minimum acceptance criteria. Each is a dict
    {met: bool, detail: str}. Honest: criteria with no data report met=None."""
    a = {}
    lad = reports.get("LAD", {})
    rho = lad.get("rho_loco", float("nan"))
    a["loco_positive"] = {"met": bool(rho > 0), "value": rho,
                          "detail": f"LAD LOCO rho={rho:.3f} (target >0, ideally >0.6)"}
    a["loco_strong"] = {"met": bool(rho > 0.6), "value": rho}

    def beats(x):
        pv = paired.get(x, {})
        prob = pv.get("prob_a_better")
        return None if prob is None else bool(prob > 0.8), pv.get("delta_mean")

    for base, key in [("naive_variance", "beats_naive_variance"),
                      ("embedding_diversity", "beats_diversity_only"),
                      ("mean_pass_rate", "beats_pass_rate_only")]:
        met, delta = beats(base)
        a[key] = {"met": met, "delta_rho": delta}

    if causal:
        tmr = causal.get("top_minus_random")
        tmb = causal.get("top_minus_bottom")
        a["causal_top_beats_random_and_bottom"] = {
            "met": (None if tmr is None or tmb is None else bool(tmr > 0 and tmb > 0)),
            "top_minus_random": tmr, "top_minus_bottom": tmb}
    else:
        a["causal_top_beats_random_and_bottom"] = {"met": None, "detail": "no causal lifts yet"}

    a["cost_below_gradient_influence"] = {"met": True,
        "detail": "LAD = k base rollouts; no gradients/training/RL (see cost table)"}

    # ablation: removing headroom/diversity worsens prediction
    nh = reports.get("LAD_no_headroom", {}).get("rho_loco")
    nd = reports.get("LAD_no_diversity", {}).get("rho_loco")
    a["ablation_worsens"] = {
        "met": (None if (nh is None or nd is None) else bool(rho >= nh and rho >= nd)),
        "lad": rho, "no_headroom": nh, "no_diversity": nd}

    # mechanistic agreement
    mv = mech_corr.get("mean_group_reward_var", {}).get("spearman")
    a["mechanistic_agrees"] = {
        "met": (None if mv is None else bool(mv > 0.3)),
        "lad_vs_advvar_spearman": mv}

    # confounder: LAD coefficient stays positive
    mreg = confound.get("multivariate_regression", {})
    lad_coef = mreg.get("LAD", {}).get("coef")
    a["confounder_lad_positive"] = {
        "met": (None if lad_coef is None else bool(lad_coef > 0)),
        "lad_coef": lad_coef}
    return a


def build_headline(reports, paired, causal, mech_corr):
    out = {}
    for m, rep in reports.items():
        out.setdefault("loco_rho", {})[m] = rep.get("rho_loco")
    out["lad_rho"] = reports.get("LAD", {}).get("rho_loco")
    out["lad_vs_variance_delta"] = paired.get("naive_variance", {}).get("delta_mean")
    out["causal_top_minus_random"] = causal.get("top_minus_random") if causal else None
    out["mech_lad_vs_advvar"] = mech_corr.get("mean_group_reward_var", {}).get("spearman")
    return out


def _print_headline(s):
    print("=" * 74)
    print(f"LAD  |  {s['n_cohorts']} cohorts")
    print("=" * 74)
    pr = s["predictive"]
    print(f"{'metric':<22}{'rho_loco':>10}{'R2_loco':>10}{'rho_in':>9}{'boot95CI':>18}")
    order = sorted(pr, key=lambda m: -(pr[m]["rho_loco"] if pr[m]["rho_loco"] == pr[m]["rho_loco"] else -9))
    for m in order:
        r = pr[m]
        ci = r["bootstrap_rho_ci"]
        cis = f"[{ci['lo']:+.2f},{ci['hi']:+.2f}]" if ci.get("lo") == ci.get("lo") else "n/a"
        print(f"{m:<22}{r['rho_loco']:>10.3f}{r['r2_loco']:>10.3f}{r['rho_insample']:>9.3f}{cis:>18}")
    print("-" * 74)
    h = s["headline"]
    print(f"HEADLINE LAD LOCO rho = {h['lad_rho']}")
    if h.get("causal_top_minus_random") is not None:
        print(f"CAUSAL top-LAD minus random lift = {h['causal_top_minus_random']:+.4f}")
    if h.get("mech_lad_vs_advvar") is not None:
        print(f"MECH LAD vs logged advantage-variance Spearman = {h['mech_lad_vs_advvar']:.3f}")
    print("\nSection-17 acceptance criteria:")
    for k, v in s["acceptance_criteria"].items():
        print(f"  {'OK ' if v.get('met') is True else ('-- ' if v.get('met') is None else 'XX ')}{k}")


def _clean(o):
    """Make numpy/NaN JSON-safe."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        o = o.item()
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return o


if __name__ == "__main__":
    main()

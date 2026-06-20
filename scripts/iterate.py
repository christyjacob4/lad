"""Iterate phase (auto-research loop, NO GPU): the GRPO lifts are now fixed
ground truth, so we cheaply search the LAD metric FORM to maximize held-out
Spearman rho WITHOUT overfitting.

Two honesty levels are reported:

  1. LOCO rho per form  — for each candidate metric form, fit the 1-D metric->lift
     map leave-one-cohort-out and correlate held-out predictions with actual lift.
     Selecting the *form* on this number can mildly overfit the form choice to the
     cohorts (the form is a few discrete knobs, so the risk is small but real).

  2. NESTED rho (the honest headline) — outer LOCO over cohorts; for each outer
     held-out cohort, pick the best metric FORM using only the inner cohorts
     (inner LOCO rho), fit its map on the inner cohorts, predict the outer cohort.
     The form is never chosen with knowledge of the cohort it is scored on. This
     is the number we report as "held-out rho after iterating the metric".

Every candidate is logged to results/iterate_log.json with its LOCO rho so we
keep only changes that improve held-out rho and can see what was tried.
"""

import argparse
import glob
import itertools
import json
import os
import numpy as np


def load_lifts(results_dir):
    by = {}
    for path in glob.glob(os.path.join(results_dir, "lift_*.json")):
        with open(path) as f:
            r = json.load(f)
        by.setdefault(r["cohort"], []).append(r)
    lifts, accb, acca = {}, {}, {}
    for c, runs in by.items():
        lifts[c] = float(np.mean([r["lift"] for r in runs]))
        accb[c] = float(np.mean([r["acc_before"] for r in runs]))
        acca[c] = float(np.mean([r["acc_after"] for r in runs]))
    return lifts, accb, acca


# ---- candidate metric-form grid ----
GAMMAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
BETAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
HEADROOMS = ("none", "power", "exp")
AGGS = ("mean", "rms", "p75", "softmax")
DIVS = ("none", "vendi_frac", "vendi_log")


def enumerate_forms():
    forms = []
    for g, b, h, a, d in itertools.product(GAMMAS, BETAS, HEADROOMS, AGGS, DIVS):
        # exp headroom ignores gamma=0 (==none); dedup trivially-equal forms
        if h == "none" and g not in (0.0,):
            continue  # gamma irrelevant when headroom is none -> only keep one
        if d == "none" and b not in (0.0,):
            continue  # beta irrelevant when no diversity term
        forms.append({"gamma": g, "beta": b, "headroom": h, "agg": a, "div": d})
    return forms


def form_values(form, p_hats, embs_list, vendis, ns):
    from lad.metric import lad_family
    return np.array([
        lad_family(p_hats[i], embs_list[i], gamma=form["gamma"], beta=form["beta"],
                   vendi=vendis[i], n=ns[i], headroom=form["headroom"],
                   agg=form["agg"], div=form["div"])
        for i in range(len(p_hats))
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data/run")
    ap.add_argument("--results", default="results/lifts")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from lad.metric import (vendi_score, baseline_variance,
                            baseline_mean_pass_rate, lad as lad_fn)
    from lad.validate import loco_report, loco_predictions, spearman

    meta = json.load(open(os.path.join(args.datadir, "cohort_meta.json")))
    pool_embs = np.load(os.path.join(args.datadir, "pool_embeddings.npy"))
    lifts_d, _, _ = load_lifts(args.results)
    names = sorted(c for c in meta if c in lifts_d)
    if len(names) < 4:
        print(f"Need >=4 cohorts; have {len(names)}: {names}")
        return
    p_hats = [np.array(meta[c]["p_hat"]) for c in names]
    embs_list = [pool_embs[meta[c]["indices"]] for c in names]
    vendis = [vendi_score(e) for e in embs_list]
    ns = [len(p) for p in p_hats]
    lifts = np.array([lifts_d[c] for c in names])
    Ncoh = len(names)

    forms = enumerate_forms()
    print(f"[iterate] {Ncoh} cohorts, {len(forms)} candidate metric forms")

    # --- baselines + the original default LAD (gamma=1,beta=1,power,mean,vendi_frac) ---
    baselines = {
        "mean_pass_rate": np.array([baseline_mean_pass_rate(p) for p in p_hats]),
        "variance": np.array([baseline_variance(p) for p in p_hats]),
        "vendi_only": np.array(vendis),
    }
    default_form = {"gamma": 1.0, "beta": 1.0, "headroom": "power",
                    "agg": "mean", "div": "vendi_frac"}
    default_vals = form_values(default_form, p_hats, embs_list, vendis, ns)
    default_rho = loco_report(default_vals, lifts)["rho_loco"]

    # --- 1. LOCO rho for every form; log all, find global best ---
    log = []
    best = None
    for f in forms:
        vals = form_values(f, p_hats, embs_list, vendis, ns)
        rep = loco_report(vals, lifts)
        entry = {**f, "rho_loco": rep["rho_loco"], "rho_in": rep["rho_insample"],
                 "r2_loco": rep["r2_loco"]}
        log.append(entry)
        if best is None or (not np.isnan(entry["rho_loco"]) and
                            entry["rho_loco"] > best["rho_loco"]):
            best = entry
    log.sort(key=lambda e: (-(e["rho_loco"] if not np.isnan(e["rho_loco"]) else -9)))

    # --- 2. NESTED honest rho: choose form on inner cohorts only ---
    nested_preds = np.empty(Ncoh)
    chosen_forms = []
    for i in range(Ncoh):
        inner = [j for j in range(Ncoh) if j != i]
        inner_lifts = lifts[inner]
        # pick form maximizing inner LOCO rho
        bf, brho = None, -9
        for f in forms:
            vals = form_values(f, [p_hats[j] for j in inner],
                               [embs_list[j] for j in inner],
                               [vendis[j] for j in inner], [ns[j] for j in inner])
            r = loco_report(vals, inner_lifts)["rho_loco"]
            if not np.isnan(r) and r > brho:
                brho, bf = r, f
        chosen_forms.append(bf)
        # fit chosen form's map on inner cohorts, predict outer cohort i
        all_vals = form_values(bf, p_hats, embs_list, vendis, ns)
        xm, ym = all_vals[inner], inner_lifts
        if np.std(xm) < 1e-12:
            nested_preds[i] = ym.mean()
        else:
            slope, intercept = np.polyfit(xm, ym, 1)
            nested_preds[i] = slope * all_vals[i] + intercept
    nested_rho, nested_p = spearman(nested_preds, lifts)

    # --- baseline LOCO rhos ---
    base_rhos = {k: loco_report(v, lifts)["rho_loco"] for k, v in baselines.items()}

    summary = {
        "n_cohorts": Ncoh,
        "cohorts": names,
        "default_lad_rho_loco": default_rho,
        "best_form": {k: best[k] for k in ("gamma", "beta", "headroom", "agg", "div")},
        "best_form_rho_loco": best["rho_loco"],
        "nested_honest_rho": nested_rho,
        "nested_honest_p": nested_p,
        "baseline_rho_loco": base_rhos,
        "top10_forms": log[:10],
    }
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "iterate_log.json"), "w") as fjson:
        json.dump({"summary": summary, "all_forms": log}, fjson, indent=2)

    print("=" * 72)
    print("ITERATE: searching LAD metric form to maximize held-out rho")
    print("=" * 72)
    print(f"baselines (LOCO rho):  variance={base_rhos['variance']:.3f}  "
          f"pass_rate={base_rhos['mean_pass_rate']:.3f}  vendi={base_rhos['vendi_only']:.3f}")
    print(f"default LAD (g=1,b=1,power,mean,vendi_frac) LOCO rho = {default_rho:.3f}")
    print(f"BEST form by LOCO rho = {summary['best_form']}  -> rho={best['rho_loco']:.3f}")
    print(f"NESTED honest rho (form chosen on inner cohorts only) = {nested_rho:.3f} (p={nested_p:.3f})")
    print("\nTop forms:")
    for e in log[:8]:
        print(f"  g={e['gamma']:<4} b={e['beta']:<5} {e['headroom']:<6} "
              f"{e['agg']:<8} {e['div']:<11} rho_loco={e['rho_loco']:.3f}")
    keep = "KEEP (improves over default)" if best["rho_loco"] > default_rho + 1e-9 else "default already best"
    print(f"\n{keep}")
    print(f"log -> {args.outdir}/iterate_log.json")


if __name__ == "__main__":
    main()

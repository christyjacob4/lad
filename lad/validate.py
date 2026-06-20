"""Leave-one-cohort-out (LOCO) validation of metric -> lift.

The scientific core: we have a small number of cohorts, each with a measured
GRPO lift (the expensive oracle) and several cheap metrics. We ask whether each
cheap metric *predicts held-out lift*.

We lead with held-out Spearman rho (robust with few cohorts) and report R²/RMSE
as secondary. LOCO: for each cohort, fit a 1-D monotone (linear, on ranks for
Spearman) map on the other cohorts, predict the held-out one, then correlate the
held-out predictions against actual lift across all cohorts.
"""

import numpy as np
from scipy import stats


def spearman(x, y):
    if len(x) < 3:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def pearson(x, y):
    if len(x) < 3:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def loco_predictions(metric_vals, lifts):
    """Leave-one-cohort-out predicted lift from a single metric.

    For each held-out cohort i, fit an OLS line lift ~ metric on the other
    cohorts and predict lift_i. Returns array of held-out predictions aligned
    with `lifts`.
    """
    metric_vals = np.asarray(metric_vals, dtype=float)
    lifts = np.asarray(lifts, dtype=float)
    n = len(lifts)
    preds = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        xm, ym = metric_vals[mask], lifts[mask]
        # OLS slope/intercept
        if np.std(xm) < 1e-12:
            preds[i] = ym.mean()
            continue
        slope, intercept = np.polyfit(xm, ym, 1)
        preds[i] = slope * metric_vals[i] + intercept
    return preds


def loco_report(metric_vals, lifts):
    """Full LOCO report for one metric.

    Returns dict with:
      - rho_insample, p_insample: Spearman of raw metric vs lift (rank, no fit)
      - rho_loco: Spearman of held-out predictions vs actual lift
      - r2_loco: out-of-sample R² (1 - SS_res/SS_tot) of held-out predictions
      - rmse_loco
    """
    metric_vals = np.asarray(metric_vals, dtype=float)
    lifts = np.asarray(lifts, dtype=float)

    rho_in, p_in = spearman(metric_vals, lifts)
    preds = loco_predictions(metric_vals, lifts)
    rho_loco, p_loco = spearman(preds, lifts)

    ss_res = float(np.sum((lifts - preds) ** 2))
    ss_tot = float(np.sum((lifts - lifts.mean()) ** 2))
    r2_loco = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    rmse_loco = float(np.sqrt(ss_res / len(lifts)))

    return {
        "rho_insample": rho_in,
        "p_insample": p_in,
        "rho_loco": rho_loco,
        "p_loco": p_loco,
        "r2_loco": r2_loco,
        "rmse_loco": rmse_loco,
        "preds": preds,
    }


def compare_metrics(metric_table, lifts):
    """metric_table: dict name -> array of per-cohort metric values.
    lifts: array of per-cohort measured lift.
    Returns dict name -> loco_report (plus 'preds' stripped into the dict)."""
    out = {}
    for name, vals in metric_table.items():
        rep = loco_report(vals, lifts)
        out[name] = rep
    return out


def grid_fit_lad(p_hats, embeddings_list, lifts, gammas=(0.0, 0.5, 1.0, 1.5, 2.0),
                 betas=(0.0, 0.5, 1.0, 1.5, 2.0), vendis=None, select_on="rho_loco"):
    """Search (gamma, beta) to maximize held-out LOCO rho for the LAD metric.

    This is itself an ablation: gamma is the headroom exponent, beta the
    diversity exponent. We select on LOCO (held-out) rho, not in-sample, so the
    selection is honest. Returns best params + the full grid.
    """
    from .metric import lad as lad_fn

    n = len(lifts)
    grid = []
    best = None
    for g in gammas:
        for b in betas:
            vals = np.array([
                lad_fn(p_hats[i], embeddings_list[i] if embeddings_list else None,
                       gamma=g, beta=b,
                       vendi=(vendis[i] if vendis is not None else None))["lad"]
                for i in range(n)
            ])
            rep = loco_report(vals, lifts)
            entry = {"gamma": g, "beta": b, "vals": vals, **{k: v for k, v in rep.items() if k != "preds"}}
            grid.append(entry)
            score = rep[select_on]
            if best is None or (not np.isnan(score) and score > best[select_on]):
                best = entry
    return {"best": best, "grid": grid}

"""Predictive validation (Claim 2 & 3): does the cheap metric predict held-out
GRPO lift, and does LAD beat the baselines?

Everything the RESEARCH_PLAN "Predictive validation" + "Statistics" sections ask
for, computed from per-cohort (metric, lift) pairs:

  - LOCO Spearman / Kendall / Pearson / R^2 / RMSE / MAE
  - calibration slope/intercept
  - top-k precision, pairwise rank accuracy
  - bootstrap CIs (over cohorts)
  - permutation p-values
  - paired-bootstrap "LAD beats X" test (delta-rho CI + p)
  - partial Spearman / multivariate regression confounder control
    (controls for length, diversity, pass-rate)

numpy + scipy only (no sklearn) so it validates on CPU.
"""

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------
# Correlations & LOCO
# --------------------------------------------------------------------------

def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def kendall(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan"), float("nan")
    r, p = stats.kendalltau(x, y)
    return float(r), float(p)


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def loco_predictions(metric_vals, lifts):
    """Leave-one-cohort-out predicted lift via OLS line fit on the others."""
    m = np.asarray(metric_vals, float)
    y = np.asarray(lifts, float)
    n = len(y)
    preds = np.empty(n)
    for i in range(n):
        mask = np.ones(n, bool)
        mask[i] = False
        xm, ym = m[mask], y[mask]
        if np.nanstd(xm) < 1e-12:
            preds[i] = np.nanmean(ym)
            continue
        slope, intercept = np.polyfit(xm, ym, 1)
        preds[i] = slope * m[i] + intercept
    return preds


def loco_report(metric_vals, lifts):
    """Full LOCO report for one metric vs lift."""
    m = np.asarray(metric_vals, float)
    y = np.asarray(lifts, float)
    rho_in, p_in = spearman(m, y)
    tau_in, ptau_in = kendall(m, y)
    r_in, pr_in = pearson(m, y)

    preds = loco_predictions(m, y)
    rho_loco, p_loco = spearman(preds, y)
    r_loco, _ = pearson(preds, y)

    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_loco = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    rmse = float(np.sqrt(ss_res / len(y)))
    mae = float(np.mean(np.abs(y - preds)))

    # calibration: regress actual on predicted (slope 1, intercept 0 = perfect)
    if np.nanstd(preds) > 1e-12:
        cal_slope, cal_int = np.polyfit(preds, y, 1)
    else:
        cal_slope, cal_int = float("nan"), float("nan")

    return {
        "rho_insample": rho_in, "p_insample": p_in,
        "kendall_insample": tau_in, "kendall_p_insample": ptau_in,
        "pearson_insample": r_in, "pearson_p_insample": pr_in,
        "rho_loco": rho_loco, "p_loco": p_loco,
        "pearson_loco": r_loco,
        "r2_loco": r2_loco, "rmse_loco": rmse, "mae_loco": mae,
        "calibration_slope": float(cal_slope), "calibration_intercept": float(cal_int),
        "top_k_precision": top_k_precision(m, y),
        "pairwise_accuracy": pairwise_accuracy(m, y),
        "preds": preds.tolist(),
    }


def top_k_precision(metric_vals, lifts, k=None):
    """Fraction of the metric's top-k cohorts that are in the true top-k by lift."""
    m = np.asarray(metric_vals, float)
    y = np.asarray(lifts, float)
    n = len(y)
    if k is None:
        k = max(1, n // 3)
    top_m = set(np.argsort(-m)[:k].tolist())
    top_y = set(np.argsort(-y)[:k].tolist())
    return float(len(top_m & top_y) / k)


def pairwise_accuracy(metric_vals, lifts):
    """Fraction of cohort pairs ordered consistently by metric and by lift."""
    m = np.asarray(metric_vals, float)
    y = np.asarray(lifts, float)
    n = len(y)
    agree = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y[i] == y[j]:
                continue
            total += 1
            if np.sign(m[i] - m[j]) == np.sign(y[i] - y[j]):
                agree += 1
    return float(agree / total) if total else float("nan")


# --------------------------------------------------------------------------
# Bootstrap & permutation
# --------------------------------------------------------------------------

def bootstrap_rho_ci(metric_vals, lifts, n_boot=2000, ci=0.95, seed=0,
                     loco=True):
    """Bootstrap CI for (LOCO) Spearman rho by resampling cohorts with replacement."""
    m = np.asarray(metric_vals, float)
    y = np.asarray(lifts, float)
    n = len(y)
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(idx)) < 3:
            continue
        mm, yy = m[idx], y[idx]
        if loco:
            preds = loco_predictions(mm, yy)
            r, _ = spearman(preds, yy)
        else:
            r, _ = spearman(mm, yy)
        if not np.isnan(r):
            rhos.append(r)
    if not rhos:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rhos = np.array(rhos)
    lo = float(np.percentile(rhos, 100 * (1 - ci) / 2))
    hi = float(np.percentile(rhos, 100 * (1 - (1 - ci) / 2)))
    return {"mean": float(rhos.mean()), "lo": lo, "hi": hi, "n": len(rhos)}


def permutation_pvalue(metric_vals, lifts, n_perm=5000, seed=0, loco=False):
    """Permutation p-value: shuffle lifts, recompute Spearman rho, count how often
    the permuted rho >= observed. Tests H0: no monotone relation."""
    m = np.asarray(metric_vals, float)
    y = np.asarray(lifts, float)
    obs, _ = spearman(loco_predictions(m, y) if loco else m, y)
    rng = np.random.default_rng(seed)
    count = 0
    valid = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        r, _ = spearman(loco_predictions(m, yp) if loco else m, yp)
        if np.isnan(r):
            continue
        valid += 1
        if r >= obs:
            count += 1
    return {"observed_rho": float(obs),
            "p_value": float((count + 1) / (valid + 1)) if valid else float("nan"),
            "n_perm": valid}


def paired_bootstrap_better(metric_a, metric_b, lifts, n_boot=3000, seed=0,
                            loco=True):
    """Paired bootstrap test of "metric_a predicts lift better than metric_b".

    Resample cohorts with replacement; compute delta = rho(a) - rho(b) on each
    resample using the SAME cohort indices for both metrics. Returns the mean
    delta, CI, and the bootstrap probability that delta>0 (a one-sided p that
    a does NOT beat b is 1 - that probability).
    """
    a = np.asarray(metric_a, float)
    b = np.asarray(metric_b, float)
    y = np.asarray(lifts, float)
    n = len(y)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(idx)) < 3:
            continue
        yy = y[idx]
        if loco:
            ra, _ = spearman(loco_predictions(a[idx], yy), yy)
            rb, _ = spearman(loco_predictions(b[idx], yy), yy)
        else:
            ra, _ = spearman(a[idx], yy)
            rb, _ = spearman(b[idx], yy)
        if np.isnan(ra) or np.isnan(rb):
            continue
        deltas.append(ra - rb)
    if not deltas:
        return {"delta_mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "prob_a_better": float("nan"), "p_a_not_better": float("nan")}
    d = np.array(deltas)
    prob_better = float(np.mean(d > 0))
    return {
        "delta_mean": float(d.mean()),
        "lo": float(np.percentile(d, 2.5)),
        "hi": float(np.percentile(d, 97.5)),
        "prob_a_better": prob_better,
        "p_a_not_better": float(1.0 - prob_better),
        "n_boot": len(d),
    }


# --------------------------------------------------------------------------
# Confounder controls: partial Spearman + multivariate regression
# --------------------------------------------------------------------------

def _rank(x):
    return stats.rankdata(np.asarray(x, float))


def partial_spearman(x, y, controls):
    """Partial Spearman correlation of x and y controlling for `controls`
    (a list of 1d arrays). Computed as the Pearson correlation of the residuals
    of rank(x) and rank(y) after regressing out the ranks of the controls.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(y)
    if n < 4:
        return {"partial_rho": float("nan"), "p": float("nan")}
    rx, ry = _rank(x), _rank(y)
    if controls:
        Z = np.column_stack([_rank(c) for c in controls])
        Z = np.column_stack([np.ones(n), Z])
        rx_res = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
        ry_res = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    else:
        rx_res, ry_res = rx - rx.mean(), ry - ry.mean()
    if np.std(rx_res) < 1e-12 or np.std(ry_res) < 1e-12:
        return {"partial_rho": float("nan"), "p": float("nan")}
    r, p = stats.pearsonr(rx_res, ry_res)
    return {"partial_rho": float(r), "p": float(p)}


def multivariate_regression(lift, predictors):
    """OLS: lift = b0 + sum_j b_j * z_j, on standardized predictors.

    predictors: dict name -> 1d array (e.g. {"LAD":..., "length":..., ...}).
    Returns per-predictor standardized coefficient + t-stat + p-value, so we can
    show beta_LAD stays positive & significant controlling for length/diversity/
    pass-rate. (Small-n caveat reported alongside.)
    """
    y = np.asarray(lift, float)
    n = len(y)
    names = list(predictors)
    cols = []
    for nm in names:
        z = np.asarray(predictors[nm], float)
        s = np.std(z)
        cols.append((z - z.mean()) / s if s > 1e-12 else np.zeros(n))
    X = np.column_stack([np.ones(n)] + cols)
    p = X.shape[1]
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(1, n - p)
    sigma2 = float(resid @ resid) / dof
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0.0))
    out = {}
    for j, nm in enumerate(["intercept"] + names):
        b = float(beta[j])
        s = float(se[j])
        t = b / s if s > 1e-12 else float("nan")
        pv = float(2 * (1 - stats.t.cdf(abs(t), dof))) if not np.isnan(t) else float("nan")
        out[nm] = {"coef": b, "se": s, "t": t, "p": pv}
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out["_r2"] = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    out["_n"] = n
    out["_dof"] = dof
    return out


# --------------------------------------------------------------------------
# Full comparison driver
# --------------------------------------------------------------------------

def compare_all(metric_table, lifts, focus="LAD", n_boot=2000, seed=0):
    """metric_table: {metric_name -> array aligned to lifts}.
    Returns a full report dict: per-metric LOCO report + bootstrap CI, plus
    paired-bootstrap "focus beats X" for every other metric."""
    lifts = np.asarray(lifts, float)
    reports = {}
    for name, vals in metric_table.items():
        rep = loco_report(vals, lifts)
        rep["bootstrap_rho_ci"] = bootstrap_rho_ci(vals, lifts, n_boot=n_boot, seed=seed)
        rep["permutation"] = permutation_pvalue(vals, lifts, seed=seed)
        reports[name] = rep

    paired = {}
    if focus in metric_table:
        for name, vals in metric_table.items():
            if name == focus:
                continue
            paired[name] = paired_bootstrap_better(
                metric_table[focus], vals, lifts, n_boot=max(n_boot, 3000), seed=seed)
    return {"reports": reports, "paired_vs_focus": paired, "focus": focus}

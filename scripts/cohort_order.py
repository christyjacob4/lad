"""Emit the cohort training ORDER for the anytime run, one name per line.

Goal (HARD requirement): order so a PARTIAL seed-0 sweep already SPANS the
difficulty / LAD spectrum -> a full-spread LOCO scatter even if we stop early.
NOT easy-first: we interleave difficulty bands (very-hard ... very-easy) with the
adversarial / diversity / mixture families, alternating high/low difficulty so the
first few cohorts already cover both ends of the p_hat range plus the special
families that break naive metrics.
"""

import argparse
import json
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data/run")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.datadir, "cohort_meta.json")))

    names = list(meta)
    pmean = {c: float(np.mean(meta[c]["p_hat"])) for c in names}
    tags = {c: meta[c].get("tags", []) for c in names}

    difficulty = sorted([c for c in names if "difficulty" in tags[c]], key=lambda c: pmean[c])
    # interleave difficulty as extreme-first, spanning the spectrum quickly:
    # hardest, easiest, mid, then fill outward -> first 3 already span p in [~0.05,~0.95]
    diff_spread = _spread(difficulty)

    specials = [c for c in names if "difficulty" not in tags[c]]
    # order specials to front-load the metric-breaking ones (adversarial/noisy/
    # diversity/mixture) and spread their difficulty too
    def special_key(c):
        t = tags[c]
        pri = 0
        if "adversarial" in t or "noisy" in t or "broken_verifier" in t:
            pri = 0
        elif "diversity" in t:
            pri = 1
        elif "mixture" in t:
            pri = 2
        else:
            pri = 3
        return (pri, pmean[c])
    specials = sorted(specials, key=special_key)

    # weave: take from diff_spread and specials alternately so early stop has both
    order = []
    di, si = 0, 0
    while di < len(diff_spread) or si < len(specials):
        if di < len(diff_spread):
            order.append(diff_spread[di]); di += 1
        if si < len(specials):
            order.append(specials[si]); si += 1
    print("\n".join(order))


def _spread(sorted_by_p):
    """Reorder an ascending-by-difficulty list to extreme-first spanning:
    [hardest, easiest, median, then alternate inward]."""
    xs = list(sorted_by_p)
    if not xs:
        return xs
    out = []
    lo, hi = 0, len(xs) - 1
    # start with the two extremes + middle to span fast
    if len(xs) >= 1:
        out.append(xs[lo]); lo += 1
    if hi >= lo:
        out.append(xs[hi]); hi -= 1
    if hi >= lo:
        mid = (lo + hi) // 2
        out.append(xs[mid])
        xs2 = xs[lo:mid] + xs[mid + 1:hi + 1]
        # alternate from both ends of the remainder
        a, b = 0, len(xs2) - 1
        while a <= b:
            if a == b:
                out.append(xs2[a]); break
            out.append(xs2[a]); out.append(xs2[b]); a += 1; b -= 1
    return out


if __name__ == "__main__":
    main()

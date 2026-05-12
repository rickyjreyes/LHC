#!/usr/bin/env python
"""
Unbinned extended likelihood skeleton.

This is not a final physics likelihood. It is a scaffold:
- load event-level q2 from ROOT or CSV
- define normalized q2 PDF
- compare SM/charm/WCT mixture models
- add angular variables later

Use only after you have real ROOT files, not HTML login pages.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy.integrate import quad
    from scipy.optimize import minimize
except Exception as e:
    raise SystemExit("This script requires scipy. Install with: pip install scipy") from e

from wct_models import wct_log_periodic, charm_tail


def load_q2_from_csv(path):
    df = pd.read_csv(path)
    if "q2" not in df.columns:
        raise ValueError("CSV must contain q2 column")
    return df["q2"].to_numpy(float)


def load_q2_from_root(path, tree_name, branch):
    try:
        import uproot
    except Exception as e:
        raise SystemExit("Install uproot first: pip install uproot") from e

    with uproot.open(path) as f:
        tree = f[tree_name]
        q2 = tree[branch].array(library="np")
    return np.asarray(q2, dtype=float)


def positive_pdf_shape(q2, theta, model):
    """
    Positive q2-only toy shape. Replace this with the real angular PDF.

    theta:
        background slope and model deformation parameters.
    """
    q2 = np.asarray(q2, dtype=float)

    if model == "sm":
        c1, = theta
        base = 1.0 + c1 * (q2 - 4.0) / 4.0
        return np.clip(base, 1e-12, np.inf)

    if model == "charm":
        c1, scale = theta
        base = 1.0 + c1 * (q2 - 4.0) / 4.0 + scale * charm_tail(q2, scale=1.0)
        return np.clip(base, 1e-12, np.inf)

    if model == "wct":
        c1, A, k, phi, mu, sigma_w = theta
        base = 1.0 + c1 * (q2 - 4.0) / 4.0
        mod = wct_log_periodic(q2, A, k, phi, mu, sigma_w)
        return np.clip(base * (1.0 + mod), 1e-12, np.inf)

    raise ValueError(model)


def normalize(theta, model, q2_min, q2_max):
    val, _ = quad(lambda x: float(positive_pdf_shape(np.array([x]), theta, model)[0]), q2_min, q2_max, limit=200)
    return max(val, 1e-300)


def neg_log_likelihood(theta, q2, model, q2_min, q2_max):
    z = normalize(theta, model, q2_min, q2_max)
    p = positive_pdf_shape(q2, theta, model) / z
    return -float(np.sum(np.log(np.clip(p, 1e-300, np.inf))))


def fit_model(q2, model, q2_min, q2_max):
    ell = np.log(q2)
    if model == "sm":
        x0 = np.array([0.0])
        bounds = [(-2.0, 2.0)]
    elif model == "charm":
        x0 = np.array([0.0, 0.1])
        bounds = [(-2.0, 2.0), (-1.0, 1.0)]
    elif model == "wct":
        x0 = np.array([0.0, 0.05, 11.7, 0.0, float(np.mean(ell)), 0.7])
        bounds = [(-2, 2), (-0.9, 0.9), (2, 25), (-np.pi, np.pi), (np.min(ell), np.max(ell)), (0.05, 3.0)]
    else:
        raise ValueError(model)

    res = minimize(
        lambda th: neg_log_likelihood(th, q2, model, q2_min, q2_max),
        x0,
        method="Nelder-Mead",
        options={"maxiter": 5000},
    )
    return res


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv")
    src.add_argument("--root")
    p.add_argument("--tree", default="DecayTree")
    p.add_argument("--branch", default="q2")
    p.add_argument("--q2-min", type=float, default=1.1)
    p.add_argument("--q2-max", type=float, default=8.0)
    args = p.parse_args()

    if args.csv:
        q2 = load_q2_from_csv(args.csv)
    else:
        q2 = load_q2_from_root(args.root, args.tree, args.branch)

    q2 = q2[(q2 >= args.q2_min) & (q2 <= args.q2_max)]
    if len(q2) == 0:
        raise SystemExit("No events in selected q2 range.")

    print(f"Loaded {len(q2)} events in {args.q2_min} <= q2 <= {args.q2_max}")

    fits = {}
    for model in ["sm", "charm", "wct"]:
        res = fit_model(q2, model, args.q2_min, args.q2_max)
        fits[model] = res
        print(f"{model:8s} NLL={res.fun:.6f} theta={res.x}")

    print("\nLikelihood-ratio diagnostics:")
    for alt in ["charm", "wct"]:
        d = 2 * (fits["sm"].fun - fits[alt].fun)
        print(f"2 Delta logL {alt} vs sm = {d:.6f}")


if __name__ == "__main__":
    main()

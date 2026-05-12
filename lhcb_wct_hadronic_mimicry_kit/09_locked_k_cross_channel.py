#!/usr/bin/env python
"""
Blind cross-channel prediction test.

Input: multiple CSVs with q2,y,sigma.
Constraint: k is fixed and never floated.

Model:
    y = A exp[-(ln q2 - mu)^2/(2 sigma_w^2)] cos(k_fixed ln q2 + phi)

Default locked mode:
    fixed k, fixed mu/sigma optional.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy.optimize import curve_fit
except Exception as e:
    raise SystemExit("This script requires scipy. Install with: pip install scipy") from e

from wct_models import wct_log_periodic, chi2, aic, bic, constant_shift


def fit_locked_k(q2, y, sigma, k_fixed, fixed_mu=None, fixed_sigma_w=None):
    ell = np.log(q2)
    y_scale = max(np.std(y), 1e-3)

    if fixed_mu is not None and fixed_sigma_w is not None:
        def model(q, A, phi):
            return wct_log_periodic(q, A, k_fixed, phi, fixed_mu, fixed_sigma_w)
        p0 = [y_scale, 0.0]
        bounds = ([-10*y_scale, -np.pi], [10*y_scale, np.pi])
        npar = 2
        names = ["A", "phi"]
    elif fixed_mu is not None:
        def model(q, A, phi, sigma_w):
            return wct_log_periodic(q, A, k_fixed, phi, fixed_mu, sigma_w)
        p0 = [y_scale, 0.0, 0.8]
        bounds = ([-10*y_scale, -np.pi, 0.05], [10*y_scale, np.pi, 3.0])
        npar = 3
        names = ["A", "phi", "sigma_w"]
    else:
        def model(q, A, phi, mu, sigma_w):
            return wct_log_periodic(q, A, k_fixed, phi, mu, sigma_w)
        p0 = [y_scale, 0.0, np.mean(ell), 0.8]
        bounds = ([-10*y_scale, -np.pi, np.min(ell), 0.05], [10*y_scale, np.pi, np.max(ell), 3.0])
        npar = 4
        names = ["A", "phi", "mu", "sigma_w"]

    popt, _ = curve_fit(model, q2, y, sigma=sigma, p0=p0, bounds=bounds, maxfev=20000)
    yhat = model(q2, *popt)
    c2 = chi2(y, yhat, sigma)
    params = dict(zip(names, popt))
    params["k"] = k_fixed
    if fixed_mu is not None:
        params["mu"] = fixed_mu
    if fixed_sigma_w is not None:
        params["sigma_w"] = fixed_sigma_w
    return params, c2, npar


def fit_constant(q2, y, sigma):
    w = 1.0 / np.maximum(sigma, 1e-12) ** 2
    c = np.sum(w * y) / np.sum(w)
    yhat = constant_shift(q2, c)
    return {"c": c}, chi2(y, yhat, sigma), 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=float, required=True)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", default="locked_k_results.csv")
    p.add_argument("--fixed-mu", type=float, default=None)
    p.add_argument("--fixed-sigma-w", type=float, default=None)
    args = p.parse_args()

    rows = []
    for path in args.inputs:
        df = pd.read_csv(path)
        q2 = df["q2"].to_numpy(float)
        y = df["y"].to_numpy(float)
        sigma = df["sigma"].to_numpy(float)
        n = len(y)

        params_wct, chi_wct, p_wct = fit_locked_k(
            q2, y, sigma, args.k,
            fixed_mu=args.fixed_mu,
            fixed_sigma_w=args.fixed_sigma_w,
        )
        params_c, chi_c, p_c = fit_constant(q2, y, sigma)

        row = {
            "input": path,
            "n": n,
            "k_locked": args.k,
            "chi2_const": chi_c,
            "chi2_locked_wct": chi_wct,
            "delta_chi2_const_minus_locked": chi_c - chi_wct,
            "aic_const": aic(chi_c, p_c),
            "aic_locked_wct": aic(chi_wct, p_wct),
            "bic_const": bic(chi_c, p_c, n),
            "bic_locked_wct": bic(chi_wct, p_wct, n),
        }
        row.update({f"wct_{k}": v for k, v in params_wct.items()})
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()

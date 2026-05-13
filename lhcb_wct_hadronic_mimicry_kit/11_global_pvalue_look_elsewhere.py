#!/usr/bin/env python
"""
Look-elsewhere correction by pseudo-experiment.

Given a real observed delta chi2 from a k scan, estimate global p-value:
    p_global = fraction(null pseudo-experiments with max_delta >= observed_delta)

Null model here is constant-shift + Gaussian errors.
For charm-null, use 08_scan_wct_on_injections.py MC mode.
"""

import argparse
from pathlib import Path
import importlib
import numpy as np
import pandas as pd

scan_mod = importlib.import_module("08_scan_wct_on_injections")
scan_one = scan_mod.scan_one
fit_constant = scan_mod.fit_constant


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--observed-delta", type=float, required=True)
    p.add_argument("--n-mc", type=int, default=1000)
    p.add_argument("--k-min", type=float, default=2.0)
    p.add_argument("--k-max", type=float, default=25.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="look_elsewhere_mc.csv")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    q2 = df["q2"].to_numpy(float)
    y = df["y"].to_numpy(float)
    sigma = df["sigma"].to_numpy(float)

    const = fit_constant(q2, y, sigma)
    y0 = const["yhat"]

    rng = np.random.default_rng(args.seed)
    rows = []
    exceed = 0

    for i in range(args.n_mc):
        yfake = y0 + rng.normal(0.0, sigma)
        dfi = pd.DataFrame({"q2": q2, "y": yfake, "sigma": sigma})
        _, stats = scan_one(dfi, args.k_min, args.k_max)
        max_delta = stats["delta_chi2_vs_const"]
        hit = max_delta >= args.observed_delta
        exceed += int(hit)
        rows.append({
            "mc": i,
            "max_delta": max_delta,
            "best_k": stats["best_k"],
            "exceeds_observed": hit,
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    p_global = exceed / max(args.n_mc, 1)
    print(f"p_global = {p_global:.6g}  ({exceed}/{args.n_mc})")
    print(f"Wrote {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()

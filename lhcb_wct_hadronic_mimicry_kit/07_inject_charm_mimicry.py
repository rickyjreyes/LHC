#!/usr/bin/env python
"""
Generate charm-only Standard-Model-like pseudo-data with Breit-Wigner tails.

Output CSV columns:
    q2,y,sigma,true_charm,true_smooth
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from wct_models import charm_tail, smooth_background


def make_q2_grid(q2_min, q2_max, n_bins):
    edges = np.linspace(q2_min, q2_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def generate_dataset(
    n_bins=16,
    q2_min=1.1,
    q2_max=8.0,
    seed=1,
    sigma=0.15,
    charm_scale=0.35,
    smooth_c0=0.0,
    smooth_c1=-0.01,
    smooth_c2=0.0,
):
    rng = np.random.default_rng(seed)
    edges, q2 = make_q2_grid(q2_min, q2_max, n_bins)

    y_charm = charm_tail(
        q2,
        a_jpsi=1.0,
        phi_jpsi=0.0,
        a_psi2s=0.35,
        phi_psi2s=1.1,
        scale=charm_scale,
        offset=0.0,
    )
    y_smooth = smooth_background(q2, c0=smooth_c0, c1=smooth_c1, c2=smooth_c2)
    sig = np.full_like(q2, sigma, dtype=float)
    y = y_smooth + y_charm + rng.normal(0.0, sig)

    return pd.DataFrame({
        "bin_low": edges[:-1],
        "bin_high": edges[1:],
        "q2": q2,
        "y": y,
        "sigma": sig,
        "true_charm": y_charm,
        "true_smooth": y_smooth,
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="fake_charm.csv")
    p.add_argument("--n-bins", type=int, default=16)
    p.add_argument("--q2-min", type=float, default=1.1)
    p.add_argument("--q2-max", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--charm-scale", type=float, default=0.35)
    args = p.parse_args()

    df = generate_dataset(
        n_bins=args.n_bins,
        q2_min=args.q2_min,
        q2_max=args.q2_max,
        seed=args.seed,
        sigma=args.sigma,
        charm_scale=args.charm_scale,
    )
    out = Path(args.out)
    df.to_csv(out, index=False)
    print(f"Wrote {out.resolve()}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()

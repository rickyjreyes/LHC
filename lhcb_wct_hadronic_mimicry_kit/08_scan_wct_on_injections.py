#!/usr/bin/env python
"""
Run WCT log-periodic scan on charm-only pseudo-data.

v2 fixes:
- WCT optimizer is strictly bounded. No Nelder-Mead escape outside k range.
- Adds WCT-vs-charm diagnostics, not only WCT-vs-constant.
- Reports whether WCT is actually preferred by AIC/BIC over charm toy.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy.optimize import differential_evolution, minimize, curve_fit
    from scipy.stats import chi2 as chi2_dist
except Exception as e:
    raise SystemExit("This script requires scipy. Install with: pip install scipy") from e

from wct_models import (
    wct_log_periodic,
    constant_shift,
    charm_tail,
    smooth_background,
    chi2,
    aic,
    bic,
)


def fit_constant(q2, y, sigma):
    w = 1.0 / np.maximum(sigma, 1e-12) ** 2
    c = np.sum(w * y) / np.sum(w)
    yhat = constant_shift(q2, c)
    return {"params": {"c": c}, "chi2": chi2(y, yhat, sigma), "yhat": yhat, "n_params": 1}


def fit_charm_toy(q2, y, sigma):
    """
    Toy charm-tail + linear background:
        y = c0 + c1(q2-6) + scale * charm_tail(q2, phases fixed)
    """
    def model(q, c0, c1, scale):
        return smooth_background(q, c0=c0, c1=c1, c2=0.0) + charm_tail(q, scale=scale)

    p0 = [np.mean(y), 0.0, 0.2]
    bounds = ([-5, -5, -5], [5, 5, 5])
    popt, _ = curve_fit(model, q2, y, sigma=sigma, p0=p0, bounds=bounds, maxfev=20000)
    yhat = model(q2, *popt)
    return {
        "params": {"c0": popt[0], "c1": popt[1], "scale": popt[2]},
        "chi2": chi2(y, yhat, sigma),
        "yhat": yhat,
        "n_params": 3,
    }


def fit_wct(q2, y, sigma, k_min=2.0, k_max=25.0, envelope_max=3.0):
    """
    Strictly bounded 5-parameter WCT fit.

    Critical correction:
    The old script used unconstrained Nelder-Mead after differential evolution.
    That allowed k, mu, sigma_w, and A to leave the allowed domain.
    """
    ell = np.log(q2)
    mu_min, mu_max = float(np.min(ell)), float(np.max(ell))
    y_scale = max(float(np.std(y)), 1e-3)

    bounds = [
        (-5*y_scale, 5*y_scale),       # A
        (k_min, k_max),                # k
        (-np.pi, np.pi),               # phi
        (mu_min, mu_max),              # mu
        (0.05, envelope_max),          # sigma_w
    ]

    def objective(theta):
        A, k, phi, mu, sigma_w = theta
        yhat = wct_log_periodic(q2, A, k, phi, mu, sigma_w)
        val = chi2(y, yhat, sigma)
        if not np.isfinite(val):
            return 1e300
        return val

    result_de = differential_evolution(
        objective,
        bounds,
        seed=123,
        polish=False,
        updating="immediate",
        workers=1,
        maxiter=1000,
        tol=1e-8,
    )

    result = minimize(
        objective,
        result_de.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 20000, "ftol": 1e-12},
    )

    theta = result.x if result.fun <= result_de.fun else result_de.x
    A, k, phi, mu, sigma_w = theta
    yhat = wct_log_periodic(q2, A, k, phi, mu, sigma_w)

    return {
        "params": {"A": A, "k": k, "phi": phi, "mu": mu, "sigma_w": sigma_w},
        "chi2": chi2(y, yhat, sigma),
        "yhat": yhat,
        "n_params": 5,
    }


def model_row(name, fit, n):
    p = fit["n_params"]
    row = {
        "model": name,
        "chi2": fit["chi2"],
        "n_params": p,
        "dof": n - p,
        "aic": aic(fit["chi2"], p),
        "bic": bic(fit["chi2"], p, n),
    }
    row.update({f"par_{k}": v for k, v in fit["params"].items()})
    return row


def scan_one(df, k_min, k_max, envelope_max=3.0):
    q2 = df["q2"].to_numpy(float)
    y = df["y"].to_numpy(float)
    sigma = df["sigma"].to_numpy(float)
    n = len(y)

    const = fit_constant(q2, y, sigma)
    charm = fit_charm_toy(q2, y, sigma)
    wct = fit_wct(q2, y, sigma, k_min=k_min, k_max=k_max, envelope_max=envelope_max)

    rows = [
        model_row("constant", const, n),
        model_row("charm_toy", charm, n),
        model_row("wct_5param", wct, n),
    ]
    out = pd.DataFrame(rows)

    delta_const = const["chi2"] - wct["chi2"]
    delta_charm = charm["chi2"] - wct["chi2"]
    p_local_vs_const = 1.0 - chi2_dist.cdf(max(delta_const, 0.0), df=max(wct["n_params"] - const["n_params"], 1))
    p_local_vs_charm = 1.0 - chi2_dist.cdf(max(delta_charm, 0.0), df=max(wct["n_params"] - charm["n_params"], 1))

    stats = {
        "delta_chi2_vs_const": float(delta_const),
        "delta_chi2_vs_charm": float(delta_charm),
        "p_local_rough_vs_const": float(p_local_vs_const),
        "p_local_rough_vs_charm": float(p_local_vs_charm),
        "best_k": float(wct["params"]["k"]),
        "wct_aic_minus_charm_aic": float(aic(wct["chi2"], wct["n_params"]) - aic(charm["chi2"], charm["n_params"])),
        "wct_bic_minus_charm_bic": float(bic(wct["chi2"], wct["n_params"], n) - bic(charm["chi2"], charm["n_params"], n)),
        "wct_preferred_over_charm_by_aic": bool(aic(wct["chi2"], wct["n_params"]) < aic(charm["chi2"], charm["n_params"])),
        "wct_preferred_over_charm_by_bic": bool(bic(wct["chi2"], wct["n_params"], n) < bic(charm["chi2"], charm["n_params"], n)),
    }
    return out, stats


def mc_false_positive(df, n_mc, k_min, k_max, seed=1234, target_delta=9.0, target_k=None, k_window=1.0, envelope_max=3.0):
    """
    Reinject noise around fitted charm model.

    Main false-positive definition:
        WCT beats charm toy by target_delta chi2 AND, optionally, lands in target k-window.

    Secondary constant-based rate is retained only as a sanity diagnostic.
    """
    rng = np.random.default_rng(seed)
    q2 = df["q2"].to_numpy(float)
    sigma = df["sigma"].to_numpy(float)
    y = df["y"].to_numpy(float)

    charm_fit = fit_charm_toy(q2, y, sigma)
    y0 = charm_fit["yhat"]

    hits_vs_const = 0
    hits_vs_charm = 0
    hits_vs_charm_and_k = 0
    hits_k = 0
    hits_aic_charm = 0
    hits_bic_charm = 0
    rows = []

    for i in range(n_mc):
        yfake = y0 + rng.normal(0.0, sigma)
        dfi = pd.DataFrame({"q2": q2, "y": yfake, "sigma": sigma})
        _, stats = scan_one(dfi, k_min, k_max, envelope_max=envelope_max)

        hit_const = stats["delta_chi2_vs_const"] >= target_delta
        hit_charm = stats["delta_chi2_vs_charm"] >= target_delta
        hit_k = False
        if target_k is not None:
            hit_k = abs(stats["best_k"] - target_k) <= k_window
        hit_charm_and_k = hit_charm and (hit_k if target_k is not None else True)

        hits_vs_const += int(hit_const)
        hits_vs_charm += int(hit_charm)
        hits_vs_charm_and_k += int(hit_charm_and_k)
        hits_k += int(hit_k)
        hits_aic_charm += int(stats["wct_preferred_over_charm_by_aic"])
        hits_bic_charm += int(stats["wct_preferred_over_charm_by_bic"])

        rows.append({
            "mc": i,
            **stats,
            "hit_delta_vs_const": hit_const,
            "hit_delta_vs_charm": hit_charm,
            "hit_k_window": hit_k,
            "hit_delta_vs_charm_and_k": hit_charm_and_k,
        })

    denom = max(n_mc, 1)
    return pd.DataFrame(rows), {
        "n_mc": n_mc,
        "false_positive_rate_vs_const": hits_vs_const / denom,
        "false_positive_rate_vs_charm": hits_vs_charm / denom,
        "false_positive_rate_vs_charm_and_k": hits_vs_charm_and_k / denom,
        "k_window_hit_rate": hits_k / denom if target_k is not None else None,
        "wct_preferred_over_charm_by_aic_rate": hits_aic_charm / denom,
        "wct_preferred_over_charm_by_bic_rate": hits_bic_charm / denom,
        "target_delta": target_delta,
        "target_k": target_k,
        "k_window": k_window,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="CSV with q2,y,sigma columns")
    p.add_argument("--out-dir", default="mimicry_results")
    p.add_argument("--k-min", type=float, default=2.0)
    p.add_argument("--k-max", type=float, default=25.0)
    p.add_argument("--envelope-max", type=float, default=3.0)
    p.add_argument("--n-mc", type=int, default=0)
    p.add_argument("--target-delta", type=float, default=9.0)
    p.add_argument("--target-k", type=float, default=None)
    p.add_argument("--k-window", type=float, default=1.0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    required = {"q2", "y", "sigma"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    fit_table, stats = scan_one(df, args.k_min, args.k_max, envelope_max=args.envelope_max)
    fit_table.to_csv(out_dir / "fit_comparison.csv", index=False)

    with open(out_dir / "scan_stats.json", "w") as f:
        import json
        json.dump(stats, f, indent=2)

    print("=== Fit comparison ===")
    print(fit_table.to_string(index=False))
    print("\n=== WCT scan stats ===")
    print(stats)

    if args.n_mc > 0:
        mc_rows, mc_stats = mc_false_positive(
            df,
            n_mc=args.n_mc,
            k_min=args.k_min,
            k_max=args.k_max,
            target_delta=args.target_delta,
            target_k=args.target_k,
            k_window=args.k_window,
            envelope_max=args.envelope_max,
        )
        mc_rows.to_csv(out_dir / "mc_false_positive_rows.csv", index=False)
        with open(out_dir / "mc_false_positive_summary.json", "w") as f:
            import json
            json.dump(mc_stats, f, indent=2)
        print("\n=== MC false-positive summary ===")
        print(mc_stats)


if __name__ == "__main__":
    main()

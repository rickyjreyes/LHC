"""
13_wct_integer_winding_scan.py

Integer-winding test for the repaired KDE/polar two-mode Poisson analysis.

Purpose
-------
Instead of scanning continuous k2 and asking whether the best k lands near a
reference, this script tests discrete active-domain winding modes

    k_n = 2*pi*n / Delta ell_active

where Delta ell_active is the retained log-q2 support after the widened
charmonium vetoes used in 09d_two_mode_kde_baseline_polar_cupy.py.

It reuses the fitted machinery from 09d:
    - KDE baseline
    - widened vetoes
    - polar L-BFGS-B exact real-data fits
    - CuPy batched Poisson nulls

Outputs
-------
outputs_wct_integer_winding/
    integer_winding_summary.csv
    integer_winding_summary.json
    integer_winding_null.csv
    integer_winding_scores.png
    integer_winding_best_by_bandwidth.png

Run
---
python .\13_wct_integer_winding_scan.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_core_module():
    path = Path("09d_two_mode_kde_baseline_polar_cupy.py")
    if not path.exists():
        raise FileNotFoundError(
            "Missing 09d_two_mode_kde_baseline_polar_cupy.py in the current folder. "
            "Run this script from lhcb_wct_analysis_kit."
        )
    spec = importlib.util.spec_from_file_location("core09d", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


core = load_core_module()

# =============================================================================
# Config for this integer-winding test
# =============================================================================

OUT_DIR = Path("outputs_wct_integer_winding")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Use the same KDE ladder already tested.
KDE_BANDWIDTH_SCALES = [0.50, 0.75, 1.00, 1.25, 1.50]

# Integer winding range to test. Includes the branches seen in the continuous run.
N_MIN = 10
N_MAX = 22

# For a compact table, these are highlighted in the console.
HIGHLIGHT_N = [14, 15, 16, 17, 18]

# Keep the same null count as 09d unless you want a fast dry run.
NULL_N = int(getattr(core, "NULL_N", 5000))
NULL_BATCH = int(getattr(core, "NULL_BATCH", 512))
SEED = int(getattr(core, "SEED", 12345))

# Use the same base/reference settings as 09d.
K1_FIXED = float(core.K1_FIXED)
REFERENCE_K2 = float(core.REFERENCE_K2)
REFERENCE_TOL = float(core.REFERENCE_TOL)


# =============================================================================
# Helpers
# =============================================================================

def active_log_intervals():
    """Return retained q2 intervals after vetoes and their total log length."""
    qmin = float(core.Q2_MIN)
    qmax = float(core.Q2_MAX)
    jlo, jhi = map(float, core.JPSI_VETO)
    plo, phi = map(float, core.PSI2S_VETO)

    intervals = [
        (qmin, min(jlo, qmax)),
        (max(jhi, qmin), min(plo, qmax)),
        (max(phi, qmin), qmax),
    ]
    intervals = [(lo, hi) for lo, hi in intervals if hi > lo and lo > 0]
    delta = float(sum(np.log(hi / lo) for lo, hi in intervals))
    return intervals, delta


def p_value(real: float, null_vals: np.ndarray) -> float:
    return float((1 + np.sum(null_vals >= real)) / (1 + len(null_vals)))


def sigma_from_p(p: float):
    return core.gaussian_sigma_from_p(float(p))


def winding_grid(delta_ell_active: float) -> pd.DataFrame:
    rows = []
    for n in range(N_MIN, N_MAX + 1):
        k = float(2.0 * np.pi * n / delta_ell_active)
        rows.append({
            "n": int(n),
            "k_n": k,
            "delta_k_from_reference": float(k - REFERENCE_K2),
            "abs_delta_k_from_reference": float(abs(k - REFERENCE_K2)),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Main analysis
# =============================================================================

def run_one_bandwidth(q2_values: np.ndarray, scale: float, kgrid_df: pd.DataFrame, rng: np.random.Generator):
    print("\n" + "=" * 80)
    print(f"[bandwidth] KDE_BANDWIDTH_SCALE={scale:.2f}")
    print("=" * 80)

    core.KDE_BANDWIDTH_SCALE = float(scale)
    core.NULL_N = int(NULL_N)
    core.NULL_BATCH = int(NULL_BATCH)

    data = core.make_binned_counts(q2_values, mode=core.RUN_MODE)
    N = data["N"]
    B = data["B"]
    ell = data["ell"]

    base = core.fit_base_cpu_bounded(N, B, ell, K1_FIXED)
    D_base = float(base["D_base"])
    print(f"[base] D_base={D_base:.4f}, A1={base['A1']:.4f}, success={base['success']}")

    # Real exact fits for each integer winding.
    real_rows = []
    for _, r in kgrid_df.iterrows():
        n = int(r["n"])
        k2 = float(r["k_n"])
        fit = core.fit_two_cpu_bounded(N, B, ell, K1_FIXED, k2)
        deltaD = float(D_base - fit["D_two"])
        real_rows.append({
            "KDE_BANDWIDTH_SCALE": float(scale),
            "n": n,
            "k2": k2,
            "D_base": D_base,
            "A1_base": float(base["A1"]),
            "base_success": bool(base["success"]),
            "D_two": float(fit["D_two"]),
            "deltaD": deltaD,
            "C": float(fit["C"]),
            "A1": float(fit["A1"]),
            "phi1": float(fit["phi1"]),
            "A2": float(fit["A2"]),
            "phi2": float(fit["phi2"]),
            "success": bool(fit["success"]),
            "A1_bound_active": bool(fit.get("amplitude1_bound_active", False)),
            "A2_bound_active": bool(fit.get("amplitude2_bound_active", False)),
            "delta_k_from_reference": float(k2 - REFERENCE_K2),
            "abs_delta_k_from_reference": float(abs(k2 - REFERENCE_K2)),
        })

    real_df = pd.DataFrame(real_rows)

    # GPU null over the same discrete integer-winding grid.
    k2_grid = real_df["k2"].to_numpy(float)
    cache = core.make_gpu_cache(ell, B, K1_FIXED, k2_grid)
    lam_base = base["lambda_base"]

    null_best = np.empty(NULL_N, dtype=float)
    null_best_n = np.empty(NULL_N, dtype=int)
    null_by_n = np.empty((NULL_N, len(k2_grid)), dtype=float)

    n_done = 0
    while n_done < NULL_N:
        bsz = min(NULL_BATCH, NULL_N - n_done)
        Y = rng.poisson(lam=lam_base[None, :], size=(bsz, len(N))).astype(np.float64)
        out_null = core.gpu_scan_two_batch(Y, B, ell, K1_FIXED, k2_grid, cache)

        delta_mat = core.cp.asnumpy(out_null["delta_add"]).T  # shape B x K
        best_delta = out_null["best_delta"]
        best_idx = out_null["best_idx"]

        null_by_n[n_done:n_done + bsz, :] = delta_mat
        null_best[n_done:n_done + bsz] = best_delta
        null_best_n[n_done:n_done + bsz] = real_df["n"].to_numpy(int)[best_idx]

        n_done += bsz
        if n_done % 500 == 0 or n_done == NULL_N:
            print(f"  null {n_done}/{NULL_N}")

    # Add p-values to real rows.
    p_scan = []
    p_fixed = []
    z_scan = []
    z_fixed = []
    for j, row in real_df.iterrows():
        p_s = p_value(float(row["deltaD"]), null_best)
        p_f = p_value(float(row["deltaD"]), null_by_n[:, j])
        p_scan.append(p_s)
        p_fixed.append(p_f)
        z_scan.append(sigma_from_p(p_s))
        z_fixed.append(sigma_from_p(p_f))

    real_df["p_vs_integer_scanmax_null"] = p_scan
    real_df["z_vs_integer_scanmax_null"] = z_scan
    real_df["p_vs_fixed_n_null"] = p_fixed
    real_df["z_vs_fixed_n_null"] = z_fixed

    real_df["null_best_p95"] = float(np.percentile(null_best, 95))
    real_df["null_best_p99"] = float(np.percentile(null_best, 99))
    real_df["null_best_mean"] = float(np.mean(null_best))

    # Identify best integer winding for this bandwidth.
    best_idx = int(real_df["deltaD"].idxmax())
    best_row = real_df.loc[best_idx]
    nearest_ref_idx = int(real_df["abs_delta_k_from_reference"].idxmin())
    nearest_ref_row = real_df.loc[nearest_ref_idx]

    print(
        f"[best n] n={int(best_row['n'])}, k={best_row['k2']:.4f}, "
        f"DeltaD={best_row['deltaD']:.4f}, "
        f"p_scan={best_row['p_vs_integer_scanmax_null']:.5f}, "
        f"A2={best_row['A2']:.4f}"
    )
    print(
        f"[nearest ref n] n={int(nearest_ref_row['n'])}, k={nearest_ref_row['k2']:.4f}, "
        f"DeltaD={nearest_ref_row['deltaD']:.4f}, "
        f"p_scan={nearest_ref_row['p_vs_integer_scanmax_null']:.5f}, "
        f"p_fixed={nearest_ref_row['p_vs_fixed_n_null']:.5f}, "
        f"A2={nearest_ref_row['A2']:.4f}"
    )

    null_df = pd.DataFrame({
        "KDE_BANDWIDTH_SCALE": float(scale),
        "null_best_deltaD": null_best,
        "null_best_n": null_best_n,
    })
    for j, n in enumerate(real_df["n"].to_numpy(int)):
        null_df[f"null_deltaD_n{n}"] = null_by_n[:, j]

    return real_df, null_df


def make_plots(summary_df: pd.DataFrame):
    # Plot DeltaD by n for each bandwidth.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for scale, d in summary_df.groupby("KDE_BANDWIDTH_SCALE"):
        ax.plot(d["n"], d["deltaD"], marker="o", label=f"bw={scale:.2f}")
    ax.axvline(15, color="black", ls=":", lw=1.2, label="n=15")
    ax.set_xlabel("active-domain integer winding n")
    ax.set_ylabel(r"$\Delta D_{add}$")
    ax.set_title("Integer-winding two-mode Poisson scan")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "integer_winding_scores.png", dpi=160)
    plt.close(fig)

    # Best n by bandwidth.
    best_rows = summary_df.loc[summary_df.groupby("KDE_BANDWIDTH_SCALE")["deltaD"].idxmax()].copy()
    nearest_rows = summary_df.loc[summary_df.groupby("KDE_BANDWIDTH_SCALE")["abs_delta_k_from_reference"].idxmin()].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(best_rows["KDE_BANDWIDTH_SCALE"], best_rows["n"], marker="o", label="best integer n")
    ax.plot(nearest_rows["KDE_BANDWIDTH_SCALE"], nearest_rows["n"], marker="o", label="nearest reference n")
    ax.axhline(15, color="black", ls=":", lw=1.2, label="n=15")
    ax.set_xlabel("KDE bandwidth scale")
    ax.set_ylabel("integer winding n")
    ax.set_title("Best integer winding vs KDE bandwidth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "integer_winding_best_by_bandwidth.png", dpi=160)
    plt.close(fig)


def main():
    intervals, delta_active = active_log_intervals()
    kgrid_df = winding_grid(delta_active)

    print("=" * 80)
    print("WCT INTEGER-WINDING SCAN")
    print("=" * 80)
    print(f"active intervals: {intervals}")
    print(f"Delta ell active = {delta_active:.8f}")
    print(f"reference k2 = {REFERENCE_K2:.6f}")
    print("integer grid:")
    print(kgrid_df[kgrid_df["n"].isin(HIGHLIGHT_N)].to_string(index=False))

    rng = np.random.default_rng(SEED)

    print("\n[load] ROOT events via 09d loader")
    df = core.load_events()
    q2_values = df["q2"].to_numpy()
    print(f"[info] selected events before charmonium handling: {len(q2_values):,}")

    all_summary = []
    all_null = []
    for scale in KDE_BANDWIDTH_SCALES:
        s_df, n_df = run_one_bandwidth(q2_values, float(scale), kgrid_df, rng)
        all_summary.append(s_df)
        all_null.append(n_df)

    summary_df = pd.concat(all_summary, ignore_index=True)
    null_df = pd.concat(all_null, ignore_index=True)

    summary_path = OUT_DIR / "integer_winding_summary.csv"
    null_path = OUT_DIR / "integer_winding_null.csv"
    summary_df.to_csv(summary_path, index=False)
    null_df.to_csv(null_path, index=False)

    make_plots(summary_df)

    best_rows = summary_df.loc[summary_df.groupby("KDE_BANDWIDTH_SCALE")["deltaD"].idxmax()].copy()
    nearest_rows = summary_df.loc[summary_df.groupby("KDE_BANDWIDTH_SCALE")["abs_delta_k_from_reference"].idxmin()].copy()
    n15_rows = summary_df[summary_df["n"] == 15].copy()

    report = {
        "test": "wct_integer_active_domain_winding_scan",
        "active_intervals_q2": intervals,
        "delta_ell_active": delta_active,
        "n_range": [N_MIN, N_MAX],
        "k1_fixed": K1_FIXED,
        "reference_k2": REFERENCE_K2,
        "reference_nearest_integer_n": int(kgrid_df.loc[kgrid_df["abs_delta_k_from_reference"].idxmin(), "n"]),
        "reference_nearest_integer_k": float(kgrid_df.loc[kgrid_df["abs_delta_k_from_reference"].idxmin(), "k_n"]),
        "kde_bandwidth_scales": KDE_BANDWIDTH_SCALES,
        "null_n": NULL_N,
        "best_n_by_bandwidth": best_rows[[
            "KDE_BANDWIDTH_SCALE", "n", "k2", "deltaD", "p_vs_integer_scanmax_null", "A2", "A2_bound_active"
        ]].to_dict(orient="records"),
        "nearest_reference_n_by_bandwidth": nearest_rows[[
            "KDE_BANDWIDTH_SCALE", "n", "k2", "deltaD", "p_vs_integer_scanmax_null", "p_vs_fixed_n_null", "A2", "A2_bound_active"
        ]].to_dict(orient="records"),
        "n15_by_bandwidth": n15_rows[[
            "KDE_BANDWIDTH_SCALE", "n", "k2", "deltaD", "p_vs_integer_scanmax_null", "p_vs_fixed_n_null", "A2", "A2_bound_active"
        ]].to_dict(orient="records"),
        "files": {
            "summary_csv": str(summary_path),
            "null_csv": str(null_path),
            "score_png": str(OUT_DIR / "integer_winding_scores.png"),
            "best_png": str(OUT_DIR / "integer_winding_best_by_bandwidth.png"),
        },
        "interpretation_note": (
            "This is a discrete winding diagnostic. A pass means an integer active-domain "
            "winding improves the yield model under the same KDE/polar Poisson machinery. "
            "It is not a sideband/control-channel discovery test."
        ),
    }
    json_path = OUT_DIR / "integer_winding_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("INTEGER WINDING SUMMARY")
    print("=" * 80)
    print("Best n by bandwidth:")
    print(best_rows[["KDE_BANDWIDTH_SCALE", "n", "k2", "deltaD", "p_vs_integer_scanmax_null", "A2", "A2_bound_active"]].to_string(index=False))
    print("\nn=15 rows:")
    print(n15_rows[["KDE_BANDWIDTH_SCALE", "n", "k2", "deltaD", "p_vs_integer_scanmax_null", "p_vs_fixed_n_null", "A2", "A2_bound_active"]].to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()

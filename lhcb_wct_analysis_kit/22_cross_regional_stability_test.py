"""
22_cross_region_scaling_stability_test.py

Stability and pair-direction sweep for the cross-region scaling result
reported in 21_cross_region_scaling_phase_test.py.

What this does:
  1. For each top_wells setting in TOP_WELLS_GRID, pick the best-Koide-error
     triplet from each region (same definition as the parent script:
     argmin koide_error over all C(top_wells, 3) ordered triplets).
  2. For each ordered region pair (A, B) in PAIR_DIRECTIONS, compute:
        - pure scale fit  n_B = a * n_A
        - affine fit      n_B = a * n_A + b
        - phase coherence C  (length of mean unit vector of dphi)
        - p-values from the same triplet-pair null as the parent script:
          enumerate all triplet pairs (i_A, j_B) at this top_wells setting,
          rank the observed (best-koide vs best-koide) RMSE/C among them.

  3. Aggregate everything into one stability table:
        top_wells, pair, scale_a, scale_rmse, p_scale, affine_a, affine_b,
        affine_rmse, p_affine, phase_C, p_phase, p_affine_and_phase

  4. Save:
        outputs_wct_cross_region_stability/
          stability_table.csv         <- one row per (top_wells, pair)
          stability_summary.json      <- run config + verdict
          scale_a_vs_top_wells.png    <- the trajectory plot you want

Methodology notes:
  - Triplet-pair null is the SAME structure as the parent script: enumerate
    all ordered triplet pairs at the current top_wells setting, then
    p_scale = fraction with scale_rmse <= observed.
    This makes the p-values directly comparable to the parent run.
  - For the cross-pair B_low <-> B_high, the "best" triplet from each side
    is also chosen by argmin koide_error in that side, same convention.
  - top_wells settings are applied symmetrically: same value for A and B.
    If a region has fewer wells than top_wells, it's clipped.

Inputs:
  outputs_wct_well_first_koide/well_first_wells.csv

Usage:
  python 22_cross_region_scaling_stability_test.py

Or override config:
  python 22_cross_region_scaling_stability_test.py --top-wells 6,8,10,12,15,18

This script reuses the wells CSV produced by 19_koide_well.py / well_first
upstream stage. It does NOT recompute wells. If you change the well-finding
parameters upstream, re-run this script to re-aggregate.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Config defaults
# =============================================================================

INPUT_WELLS = Path("outputs_wct_well_first_koide/well_first_wells.csv")
OUT_DIR = Path("outputs_wct_cross_region_stability")

# Symmetric top_wells settings (same for both sides of every pair).
TOP_WELLS_GRID_DEFAULT = [6, 8, 10, 12, 15, 18]

# Region keys must match those in the wells CSV.
REGION_BLOW   = "B_low_sideband_Kst_signal"
REGION_BHIGH  = "B_high_sideband_Kst_signal"
REGION_SIGNAL = "signal_B_signal_Kst"

# Three pair directions for the diagnostic. Tuple is (A, B): n_B ~ a * n_A.
PAIR_DIRECTIONS_DEFAULT = [
    (REGION_BLOW,  REGION_SIGNAL),  # the original 21_ comparison
    (REGION_BHIGH, REGION_SIGNAL),  # diagnostic: does the high sideband also align?
    (REGION_BLOW,  REGION_BHIGH),   # diagnostic: do the two sidebands align with each other?
]

# Reference Koide value, only used to compute koide_error for ranking.
KOIDE_Q = 2.0 / 3.0


# =============================================================================
# Triplet-level helpers
# =============================================================================

def koide_metrics(n1: float, n2: float, n3: float) -> dict:
    """Match the upstream definition exactly:
       Q_low  = n1 / n2
       Q_high = n3 / (2 * n2)
       koide_error = sqrt((Q_low - 2/3)^2 + (Q_high - 2/3)^2)
    """
    Q_low  = n1 / n2
    Q_high = n3 / (2.0 * n2)
    Q_mean = 0.5 * (Q_low + Q_high)
    err = float(np.sqrt((Q_low - KOIDE_Q) ** 2 + (Q_high - KOIDE_Q) ** 2))
    return dict(Q_low=float(Q_low), Q_high=float(Q_high),
                Q_mean=float(Q_mean), koide_error=err)


def enumerate_triplets(wells_df: pd.DataFrame, top_wells: int) -> pd.DataFrame:
    """Take the top `top_wells` rows of wells_df (already sorted by rank in
    the upstream CSV) and return a DataFrame of all C(N, 3) ordered triplets
    (n1 < n2 < n3 by n_eff, NOT by k or rank). Includes phi from each well.
    """
    # Upstream wells CSV is already ordered by well_rank; trust that.
    sub = wells_df.head(top_wells).copy()
    n = sub["n_eff"].to_numpy(dtype=np.float64)
    phi = sub["phi2"].to_numpy(dtype=np.float64)
    N = len(sub)
    if N < 3:
        return pd.DataFrame()

    rows = []
    # Ordered by n_eff so n1 < n2 < n3 (matches upstream).
    order = np.argsort(n)
    n_sorted = n[order]
    phi_sorted = phi[order]
    for a, b, c in itertools.combinations(range(N), 3):
        n1, n2, n3 = n_sorted[a], n_sorted[b], n_sorted[c]
        p1, p2, p3 = phi_sorted[a], phi_sorted[b], phi_sorted[c]
        m = koide_metrics(n1, n2, n3)
        rows.append(dict(
            i1=a, i2=b, i3=c,
            n1=float(n1), n2=float(n2), n3=float(n3),
            phi1=float(p1), phi2=float(p2), phi3=float(p3),
            **m,
        ))
    return pd.DataFrame(rows)


# =============================================================================
# Pair-level helpers (scale, affine, phase coherence)
# =============================================================================

def fit_scale(nA: np.ndarray, nB: np.ndarray) -> tuple[float, float]:
    """Pure-scale least squares: nB = a * nA. Returns (a, rmse)."""
    a = float(np.dot(nA, nB) / np.dot(nA, nA))
    resid = nB - a * nA
    rmse = float(np.sqrt(np.mean(resid * resid)))
    return a, rmse


def fit_affine(nA: np.ndarray, nB: np.ndarray) -> tuple[float, float, float]:
    """Affine least squares: nB = a * nA + b. Returns (a, b, rmse)."""
    A = np.column_stack([nA, np.ones_like(nA)])
    sol, *_ = np.linalg.lstsq(A, nB, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    resid = nB - (a * nA + b)
    rmse = float(np.sqrt(np.mean(resid * resid)))
    return a, b, rmse


def phase_coherence(phiA: np.ndarray, phiB: np.ndarray) -> tuple[float, float]:
    """Mean-resultant length of dphi = phiB - phiA, modulo 2pi.
    Returns (C, mean_phase_offset).
    Matches upstream: C in [0, 1]; C=1 => all dphi equal.
    """
    dphi = (phiB - phiA)
    z = np.exp(1j * dphi)
    z_mean = np.mean(z)
    C = float(np.abs(z_mean))
    mean_off = float(np.angle(z_mean))
    return C, mean_off


def pair_metrics(triplet_A: dict, triplet_B: dict) -> dict:
    """Return all metrics for a given (A, B) triplet pair."""
    nA = np.array([triplet_A["n1"], triplet_A["n2"], triplet_A["n3"]])
    nB = np.array([triplet_B["n1"], triplet_B["n2"], triplet_B["n3"]])
    pA = np.array([triplet_A["phi1"], triplet_A["phi2"], triplet_A["phi3"]])
    pB = np.array([triplet_B["phi1"], triplet_B["phi2"], triplet_B["phi3"]])

    sa, srmse = fit_scale(nA, nB)
    aa, ab, armse = fit_affine(nA, nB)
    C, mphi = phase_coherence(pA, pB)
    return dict(
        scale_a=sa, scale_rmse=srmse,
        affine_a=aa, affine_b=ab, affine_rmse=armse,
        phase_C=C, mean_phase_offset=mphi,
    )


# =============================================================================
# Per-pair test at one top_wells setting
# =============================================================================

def run_pair(wells_A: pd.DataFrame, wells_B: pd.DataFrame,
             region_A: str, region_B: str,
             top_wells: int) -> dict:
    """Run the full test for one ordered region pair at one top_wells.
    Returns one row of the stability table.
    """
    tw_A = min(top_wells, len(wells_A))
    tw_B = min(top_wells, len(wells_B))
    if tw_A < 3 or tw_B < 3:
        return dict(top_wells=top_wells, pair=f"{region_A}->{region_B}",
                    error=f"insufficient wells (A={tw_A}, B={tw_B})")

    tA = enumerate_triplets(wells_A, tw_A)
    tB = enumerate_triplets(wells_B, tw_B)

    # Best-Koide triplet on each side: argmin koide_error.
    bestA = tA.iloc[tA["koide_error"].idxmin()].to_dict()
    bestB = tB.iloc[tB["koide_error"].idxmin()].to_dict()

    obs = pair_metrics(bestA, bestB)

    # Triplet-pair null: enumerate ALL pairs (tA x tB) and rank observed.
    # Vectorize the four metrics across the full Cartesian product.
    nA_arr = tA[["n1","n2","n3"]].to_numpy()       # (Na, 3)
    nB_arr = tB[["n1","n2","n3"]].to_numpy()       # (Nb, 3)
    pA_arr = tA[["phi1","phi2","phi3"]].to_numpy() # (Na, 3)
    pB_arr = tB[["phi1","phi2","phi3"]].to_numpy() # (Nb, 3)
    Na, Nb = len(tA), len(tB)

    # Pure-scale a and rmse vectorized over (Na, Nb).
    # For each pair (i,j): a_ij = (nA_i . nB_j) / (nA_i . nA_i)
    A_dot_A = np.sum(nA_arr * nA_arr, axis=1)            # (Na,)
    A_dot_B = nA_arr @ nB_arr.T                          # (Na, Nb)
    a_grid  = A_dot_B / A_dot_A[:, None]                 # (Na, Nb)
    pred    = a_grid[..., None] * nA_arr[:, None, :]     # (Na, Nb, 3)
    resid   = nB_arr[None, :, :] - pred                  # (Na, Nb, 3)
    scale_rmse_grid = np.sqrt(np.mean(resid ** 2, axis=2))  # (Na, Nb)

    # Affine a, b, rmse vectorized over (Na, Nb).
    # Closed form: with x = nA_i (3 pts), y = nB_j (3 pts),
    # a = cov(x,y) / var(x), b = mean(y) - a*mean(x).
    mA = nA_arr.mean(axis=1, keepdims=True)              # (Na, 1)
    mB = nB_arr.mean(axis=1, keepdims=True)              # (Nb, 1)
    xc = nA_arr - mA                                     # (Na, 3) centered
    yc = nB_arr - mB                                     # (Nb, 3) centered
    var_x = np.sum(xc * xc, axis=1)                      # (Na,)
    cov_xy = xc @ yc.T                                   # (Na, Nb)
    a_aff = cov_xy / var_x[:, None]                      # (Na, Nb)
    b_aff = mB.T - a_aff * mA                            # (Na, Nb)
    pred_aff = a_aff[..., None] * nA_arr[:, None, :] + b_aff[..., None]
    resid_aff = nB_arr[None, :, :] - pred_aff
    affine_rmse_grid = np.sqrt(np.mean(resid_aff ** 2, axis=2))

    # Phase coherence vectorized.
    dphi = pB_arr[None, :, :] - pA_arr[:, None, :]       # (Na, Nb, 3)
    z = np.exp(1j * dphi)
    z_mean = z.mean(axis=2)
    C_grid = np.abs(z_mean)                              # (Na, Nb)

    total_pairs = Na * Nb

    p_scale  = float((np.sum(scale_rmse_grid <= obs["scale_rmse"])) / total_pairs)
    p_affine = float((np.sum(affine_rmse_grid <= obs["affine_rmse"])) / total_pairs)
    p_phase  = float((np.sum(C_grid >= obs["phase_C"])) / total_pairs)
    p_joint  = float((np.sum((affine_rmse_grid <= obs["affine_rmse"]) &
                             (C_grid >= obs["phase_C"]))) / total_pairs)

    out = dict(
        top_wells=top_wells,
        pair=f"{region_A}->{region_B}",
        region_A=region_A, region_B=region_B,
        n_triplets_A=Na, n_triplets_B=Nb, n_pairs=total_pairs,

        bestA_n=[bestA["n1"], bestA["n2"], bestA["n3"]],
        bestA_Q_mean=bestA["Q_mean"],
        bestA_koide_error=bestA["koide_error"],

        bestB_n=[bestB["n1"], bestB["n2"], bestB["n3"]],
        bestB_Q_mean=bestB["Q_mean"],
        bestB_koide_error=bestB["koide_error"],

        scale_a=obs["scale_a"], scale_rmse=obs["scale_rmse"], p_scale=p_scale,
        affine_a=obs["affine_a"], affine_b=obs["affine_b"],
        affine_rmse=obs["affine_rmse"], p_affine=p_affine,
        phase_C=obs["phase_C"], mean_phase_offset=obs["mean_phase_offset"],
        p_phase=p_phase,
        p_affine_and_phase=p_joint,
    )
    return out


# =============================================================================
# Top-level
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-wells", default=str(INPUT_WELLS))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--top-wells",
                    default=",".join(str(x) for x in TOP_WELLS_GRID_DEFAULT),
                    help="Comma-separated top-wells settings.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    top_wells_grid = [int(x.strip()) for x in args.top_wells.split(",") if x.strip()]
    pair_dirs = PAIR_DIRECTIONS_DEFAULT

    print("=" * 100)
    print("CROSS-REGION SCALING STABILITY TEST")
    print("=" * 100)
    print(f"[input] {args.input_wells}")
    print(f"[top_wells_grid] {top_wells_grid}")
    print(f"[pair_directions]")
    for a, b in pair_dirs:
        print(f"    {a}  ->  {b}")
    print("=" * 100)

    wells_all = pd.read_csv(args.input_wells)
    if "n_eff" not in wells_all.columns:
        raise SystemExit(
            "wells CSV missing 'n_eff' column; expected output of "
            "well-first scan stage."
        )

    # Per-region wells, sorted by well_rank ascending (best first).
    wells_by_region = {}
    for r in wells_all["region"].unique():
        sub = wells_all[wells_all["region"] == r].copy()
        if "well_rank" in sub.columns:
            sub = sub.sort_values("well_rank")
        wells_by_region[str(r)] = sub.reset_index(drop=True)
        print(f"[wells] {r}: {len(sub)} rows")

    rows = []
    for tw in top_wells_grid:
        print("\n" + "-" * 100)
        print(f"[top_wells = {tw}]")
        print("-" * 100)
        for region_A, region_B in pair_dirs:
            if region_A not in wells_by_region or region_B not in wells_by_region:
                print(f"  skip {region_A}->{region_B}: missing region")
                continue
            r = run_pair(wells_by_region[region_A], wells_by_region[region_B],
                         region_A, region_B, tw)
            rows.append(r)
            if "error" in r:
                print(f"  {r['pair']}: ERROR {r['error']}")
                continue
            print(
                f"  {region_A.replace('_Kst_signal','').replace('_signal_Kst',''):>30s}"
                f" -> {region_B.replace('_Kst_signal','').replace('_signal_Kst',''):>15s} | "
                f"a={r['scale_a']:.4f}  rmse={r['scale_rmse']:.4f}  "
                f"p_scale={r['p_scale']:.4f}  | "
                f"p_aff={r['p_affine']:.4f}  C={r['phase_C']:.3f}  "
                f"p_phase={r['p_phase']:.3f}  p_aff∧φ={r['p_affine_and_phase']:.4f}"
            )

    # Build the stability table. JSON-friendly.
    df = pd.DataFrame(rows)
    csv_cols = [c for c in df.columns
                if c not in ("bestA_n", "bestB_n")]
    # Flatten the n-triplet lists into separate columns.
    if "bestA_n" in df.columns:
        for i, col in enumerate(["bestA_n1", "bestA_n2", "bestA_n3"]):
            df[col] = df["bestA_n"].apply(lambda v, i=i: v[i] if isinstance(v, list) else None)
        for i, col in enumerate(["bestB_n1", "bestB_n2", "bestB_n3"]):
            df[col] = df["bestB_n"].apply(lambda v, i=i: v[i] if isinstance(v, list) else None)
        df = df.drop(columns=["bestA_n", "bestB_n"])

    csv_path = out_dir / "stability_table.csv"
    df.to_csv(csv_path, index=False)

    # Plot: scale_a vs top_wells, one line per pair.
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for pair, sub in df.groupby("pair"):
        sub = sub.sort_values("top_wells")
        label = pair.replace("_Kst_signal","").replace("_signal_Kst","")
        ax.plot(sub["top_wells"], sub["scale_a"], marker="o", label=label)
    ax.axhline(1.0, color="grey", lw=0.8, ls=":", alpha=0.6)
    # 21_ baseline: a = 1.2283 for B_low -> signal at top_wells = 12.
    ax.axhline(1.2283, color="purple", lw=0.8, ls="--", alpha=0.6,
               label="21_ baseline a=1.2283")
    ax.set_xlabel("top_wells")
    ax.set_ylabel("scale a  (n_B = a · n_A)")
    ax.set_title("Cross-region scale factor stability")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = out_dir / "scale_a_vs_top_wells.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    # Second plot: p_scale trajectory.
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for pair, sub in df.groupby("pair"):
        sub = sub.sort_values("top_wells")
        label = pair.replace("_Kst_signal","").replace("_signal_Kst","")
        ax.plot(sub["top_wells"], sub["p_scale"], marker="o", label=label)
    ax.axhline(0.05, color="red", lw=0.8, ls=":", alpha=0.7, label="p=0.05")
    ax.axhline(0.01, color="darkred", lw=0.8, ls=":", alpha=0.7, label="p=0.01")
    ax.set_xlabel("top_wells")
    ax.set_ylabel("p_scale  (triplet-pair null)")
    ax.set_yscale("log")
    ax.set_title("Cross-region scale-fit p-value stability")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    p_plot_path = out_dir / "p_scale_vs_top_wells.png"
    fig.savefig(p_plot_path, dpi=160)
    plt.close(fig)

    # Verdict logic.
    # A "stable real" result wants:
    #   B_low -> signal: p_scale <= 0.05 at >= 3 of the top_wells settings,
    #     scale_a in narrow range (range/median <= 0.05)
    #   Other pairs: substantially worse (median p_scale > 0.10) at most settings
    bsig = df[df["pair"] == f"{REGION_BLOW}->{REGION_SIGNAL}"]
    bhsig = df[df["pair"] == f"{REGION_BHIGH}->{REGION_SIGNAL}"]
    blbh = df[df["pair"] == f"{REGION_BLOW}->{REGION_BHIGH}"]

    def _summary(sub):
        if len(sub) == 0:
            return dict(n=0)
        return dict(
            n=int(len(sub)),
            scale_a_median=float(sub["scale_a"].median()),
            scale_a_min=float(sub["scale_a"].min()),
            scale_a_max=float(sub["scale_a"].max()),
            scale_a_relrange=float((sub["scale_a"].max() - sub["scale_a"].min())
                                   / max(abs(sub["scale_a"].median()), 1e-12)),
            p_scale_median=float(sub["p_scale"].median()),
            p_scale_min=float(sub["p_scale"].min()),
            p_scale_max=float(sub["p_scale"].max()),
            n_pass_p05=int((sub["p_scale"] <= 0.05).sum()),
            n_pass_p01=int((sub["p_scale"] <= 0.01).sum()),
        )

    pair_summary = {
        f"{REGION_BLOW}->{REGION_SIGNAL}":  _summary(bsig),
        f"{REGION_BHIGH}->{REGION_SIGNAL}": _summary(bhsig),
        f"{REGION_BLOW}->{REGION_BHIGH}":   _summary(blbh),
    }

    bsig_s = pair_summary[f"{REGION_BLOW}->{REGION_SIGNAL}"]
    bhsig_s = pair_summary[f"{REGION_BHIGH}->{REGION_SIGNAL}"]
    blbh_s  = pair_summary[f"{REGION_BLOW}->{REGION_BHIGH}"]

    blow_sig_clean = (
        bsig_s.get("n", 0) >= 3
        and bsig_s.get("n_pass_p05", 0) >= max(3, bsig_s["n"] - 1)
        and bsig_s.get("scale_a_relrange", 1.0) <= 0.05
    )
    others_quiet = (
        bhsig_s.get("p_scale_median", 1.0) > 0.10
        and blbh_s.get("p_scale_median", 1.0) > 0.10
    )

    if blow_sig_clean and others_quiet:
        verdict = "BLOW_SIGNAL_SCALE_COUPLING_STABLE_AND_SPECIFIC"
        reason = (
            "B_low->signal scale factor stable across top_wells, p_scale<=0.05 "
            "at all/most settings, and other pairs do not produce comparably "
            "good scale alignment."
        )
    elif blow_sig_clean and not others_quiet:
        verdict = "BLOW_SIGNAL_SCALE_COUPLING_STABLE_BUT_NOT_UNIQUE"
        reason = (
            "B_low->signal scale factor stable, but at least one other pair "
            "also produces non-trivial scale alignment. Coupling is real but "
            "not signal-specific."
        )
    elif not blow_sig_clean and bsig_s.get("n_pass_p05", 0) >= 1:
        verdict = "BLOW_SIGNAL_SCALE_COUPLING_UNSTABLE"
        reason = (
            "B_low->signal scale factor varies with top_wells beyond the "
            "5% relative-range threshold; the original p=0.0059 result depends "
            "on triplet selection."
        )
    else:
        verdict = "NO_STABLE_SCALE_COUPLING"
        reason = (
            "No pair shows stable, robust scale coupling under top_wells "
            "variation. The original 21_ result is not reproduced across "
            "settings."
        )

    summary = dict(
        test="cross_region_scaling_stability",
        input_wells_csv=args.input_wells,
        top_wells_grid=top_wells_grid,
        pair_directions=[list(p) for p in pair_dirs],
        koide_Q=KOIDE_Q,
        per_pair_summary=pair_summary,
        verdict=dict(label=verdict, reason=reason),
        files=dict(
            stability_csv=str(csv_path),
            scale_plot_png=str(plot_path),
            p_scale_plot_png=str(p_plot_path),
        ),
        notes=[
            "Triplet-pair null is the same as 21_: enumerate all (tripletA, "
            "tripletB) pairs at the current top_wells, p = fraction at least "
            "as good as observed.",
            "Best triplet on each side: argmin koide_error over the C(N,3) "
            "triplets in the top N wells, same as 21_.",
            "Top wells are taken in well_rank order from the upstream wells "
            "CSV (no re-ranking).",
        ],
    )

    summary_path = out_dir / "stability_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 100)
    print("STABILITY VERDICT")
    print("=" * 100)
    print(json.dumps(summary["verdict"], indent=2))
    print()
    print("Per-pair stability summary:")
    for k, v in pair_summary.items():
        if v.get("n", 0) == 0:
            continue
        nm = k.replace("_Kst_signal","").replace("_signal_Kst","")
        print(f"  {nm}")
        print(f"      scale_a:   median={v['scale_a_median']:.4f}  "
              f"range=[{v['scale_a_min']:.4f}, {v['scale_a_max']:.4f}]  "
              f"rel_range={v['scale_a_relrange']:.4f}")
        print(f"      p_scale:   median={v['p_scale_median']:.4f}  "
              f"range=[{v['p_scale_min']:.4f}, {v['p_scale_max']:.4f}]")
        print(f"      pass p<=.05: {v['n_pass_p05']}/{v['n']}    "
              f"pass p<=.01: {v['n_pass_p01']}/{v['n']}")
    print()
    print(f"Saved:")
    print(f"  {csv_path}")
    print(f"  {summary_path}")
    print(f"  {plot_path}")
    print(f"  {p_plot_path}")


if __name__ == "__main__":
    main()
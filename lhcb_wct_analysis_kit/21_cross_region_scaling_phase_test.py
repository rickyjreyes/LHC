#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cross-Region Scaling + Phase Coherence Test
-------------------------------------------

Purpose:
    Test whether the best raw-well triplets in two regions are related by a
    coherent spectral map rather than being independent Koide-like coincidences.

Input:
    outputs_wct_well_first_koide/well_first_wells.csv

Expected columns:
    region
    k
    n_eff
    deltaD
    A2
    phi2
    bound_active

Core tests:

1. Pure scale map:
        n_B ≈ a n_A

2. Affine map:
        n_B ≈ a n_A + b

3. Null tests:
        Randomly choose triplets from each region's top wells and ask how often
        the null alignments are as good as the observed best-triplet alignment.

4. Phase coherence:
        For a paired triplet, compare fitted phases:
            Δφ_i = wrap(φ_B,i - φ_A,i)
        and compute circular coherence:
            C = |mean(exp(i Δφ_i))|

Interpretation:
    small p_scale / p_affine:
        cross-region spectral scaling is unlikely by random triplet pairing.

    high phase coherence C:
        matched wells are phase-related, not only frequency-related.

Default comparison:
    A = B_low_sideband_Kst_signal
    B = signal_B_signal_Kst
"""

import os
import json
import math
import argparse
from itertools import combinations

import numpy as np
import pandas as pd


# ============================================================
# Defaults
# ============================================================

DEFAULT_WELLS = "outputs_wct_well_first_koide/well_first_wells.csv"
OUTDIR = "outputs_wct_cross_region_scaling"
os.makedirs(OUTDIR, exist_ok=True)

DEFAULT_REGION_A = "B_low_sideband_Kst_signal"
DEFAULT_REGION_B = "signal_B_signal_Kst"

DEFAULT_TOP = 12
DEFAULT_N_NULL = 250_000
DEFAULT_SEED = 20260509

KOIDE_Q = 2.0 / 3.0


# ============================================================
# Utilities
# ============================================================

def wrap_phase(x):
    """Wrap phase to (-pi, pi]."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def circular_coherence(delta_phi):
    delta_phi = np.asarray(delta_phi, dtype=float)
    if len(delta_phi) == 0:
        return np.nan
    return float(abs(np.mean(np.exp(1j * delta_phi))))


def koide_metrics(n):
    """
    n must be ordered length-3:
        n1 < n2 < n3

    Q_low  = n1/n2
    Q_high = n3/(2 n2)
    """
    n1, n2, n3 = map(float, n)
    q_low = n1 / n2
    q_high = n3 / (2.0 * n2)
    q_mean = 0.5 * (q_low + q_high)
    eps_k = math.sqrt((q_low - KOIDE_Q) ** 2 + (q_high - KOIDE_Q) ** 2)
    return q_low, q_high, q_mean, eps_k


def pure_scale_fit(x, y):
    """
    Fit y ≈ a x.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = float(np.dot(x, x))
    if denom <= 0:
        return np.nan, np.inf, np.full_like(y, np.nan)

    a = float(np.dot(x, y) / denom)
    yhat = a * x
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return a, rmse, yhat


def affine_fit(x, y):
    """
    Fit y ≈ a x + b.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    X = np.vstack([x, np.ones_like(x)]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    a = float(beta[0])
    b = float(beta[1])
    yhat = a * x + b
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return a, b, rmse, yhat


def pair_triplets_by_order(trip_a, trip_b):
    """
    The triplets are already ordered by n.
    Pair by rank/order:
        low ↔ low, center ↔ center, high ↔ high
    """
    aa = trip_a.sort_values("n_eff").reset_index(drop=True)
    bb = trip_b.sort_values("n_eff").reset_index(drop=True)
    return aa, bb


def triplet_rows_from_top(region_df, top):
    """
    Use top wells by deltaD, then all ordered triplets by n_eff.
    """
    g = region_df.sort_values("deltaD", ascending=False).head(top).copy()
    g = g.sort_values("n_eff").reset_index(drop=True)

    triplets = []

    for idxs in combinations(range(len(g)), 3):
        sub = g.iloc[list(idxs)].copy()
        sub = sub.sort_values("n_eff").reset_index(drop=True)

        n = sub["n_eff"].values.astype(float)
        k = sub["k"].values.astype(float)
        d = sub["deltaD"].values.astype(float)
        phi = sub["phi2"].values.astype(float) if "phi2" in sub.columns else np.full(3, np.nan)

        q_low, q_high, q_mean, eps_k = koide_metrics(n)

        triplets.append({
            "idxs": idxs,
            "n": n,
            "k": k,
            "deltaD": d,
            "phi": phi,
            "mean_deltaD": float(np.mean(d)),
            "Q_low": q_low,
            "Q_high": q_high,
            "Q_mean": q_mean,
            "koide_error": eps_k,
        })

    return triplets


def alignment_metrics(trip_a, trip_b):
    """
    Compute pure-scale, affine, and phase coherence for two ordered triplets.
    """
    n_a = np.asarray(trip_a["n"], dtype=float)
    n_b = np.asarray(trip_b["n"], dtype=float)

    a_scale, rmse_scale, yhat_scale = pure_scale_fit(n_a, n_b)
    a_aff, b_aff, rmse_aff, yhat_aff = affine_fit(n_a, n_b)

    phi_a = np.asarray(trip_a["phi"], dtype=float)
    phi_b = np.asarray(trip_b["phi"], dtype=float)

    valid_phase = np.isfinite(phi_a) & np.isfinite(phi_b)
    if np.any(valid_phase):
        dphi = wrap_phase(phi_b[valid_phase] - phi_a[valid_phase])
        C = circular_coherence(dphi)
        mean_dphi = float(np.angle(np.mean(np.exp(1j * dphi))))
    else:
        dphi = np.array([])
        C = np.nan
        mean_dphi = np.nan

    return {
        "scale_a": float(a_scale),
        "scale_rmse": float(rmse_scale),
        "affine_a": float(a_aff),
        "affine_b": float(b_aff),
        "affine_rmse": float(rmse_aff),
        "phase_coherence": float(C) if np.isfinite(C) else np.nan,
        "mean_phase_offset": float(mean_dphi) if np.isfinite(mean_dphi) else np.nan,
        "delta_phi": dphi.tolist(),
        "scale_yhat": yhat_scale.tolist(),
        "affine_yhat": yhat_aff.tolist(),
    }


def choose_best_koide_triplet(triplets):
    """
    Best by Koide error, then closeness to (10,15,20), then high mean deltaD.
    """
    def integer_error(t):
        n = np.asarray(t["n"], dtype=float)
        return float(np.sqrt(np.sum((n - np.array([10.0, 15.0, 20.0])) ** 2)))

    return sorted(
        triplets,
        key=lambda t: (t["koide_error"], integer_error(t), -t["mean_deltaD"])
    )[0]


def choose_best_alignment_pair(triplets_a, triplets_b, mode="affine"):
    """
    Search all triplet pairs and return the best aligned pair.

    mode:
        "affine" -> minimize affine RMSE
        "scale"  -> minimize pure-scale RMSE
        "hybrid" -> minimize affine RMSE + mild Koide penalties
    """
    rows = []

    for ta in triplets_a:
        for tb in triplets_b:
            m = alignment_metrics(ta, tb)

            if mode == "scale":
                objective = m["scale_rmse"]
            elif mode == "affine":
                objective = m["affine_rmse"]
            elif mode == "hybrid":
                objective = (
                    m["affine_rmse"]
                    + 0.25 * ta["koide_error"]
                    + 0.25 * tb["koide_error"]
                )
            else:
                raise ValueError(mode)

            rows.append((objective, ta, tb, m))

    rows.sort(key=lambda x: x[0])
    return rows[0]


def null_alignment(triplets_a, triplets_b, obs_scale_rmse, obs_affine_rmse,
                   obs_phase_C, n_null, rng):
    """
    Randomly sample triplet pairs from the two regions.
    """
    n_a = len(triplets_a)
    n_b = len(triplets_b)

    count_scale = 0
    count_affine = 0
    count_phase = 0
    count_affine_and_phase = 0

    scale_vals = []
    affine_vals = []
    phase_vals = []

    for _ in range(n_null):
        ta = triplets_a[rng.integers(0, n_a)]
        tb = triplets_b[rng.integers(0, n_b)]

        m = alignment_metrics(ta, tb)

        sr = m["scale_rmse"]
        ar = m["affine_rmse"]
        pc = m["phase_coherence"]

        scale_vals.append(sr)
        affine_vals.append(ar)
        phase_vals.append(pc)

        if sr <= obs_scale_rmse:
            count_scale += 1

        if ar <= obs_affine_rmse:
            count_affine += 1

        if np.isfinite(pc) and np.isfinite(obs_phase_C) and pc >= obs_phase_C:
            count_phase += 1

        if (
            ar <= obs_affine_rmse
            and np.isfinite(pc)
            and np.isfinite(obs_phase_C)
            and pc >= obs_phase_C
        ):
            count_affine_and_phase += 1

    return {
        "p_scale_rmse": (1.0 + count_scale) / (n_null + 1.0),
        "p_affine_rmse": (1.0 + count_affine) / (n_null + 1.0),
        "p_phase_coherence": (1.0 + count_phase) / (n_null + 1.0),
        "p_affine_and_phase": (1.0 + count_affine_and_phase) / (n_null + 1.0),
        "null_scale_rmse_mean": float(np.mean(scale_vals)),
        "null_scale_rmse_05": float(np.quantile(scale_vals, 0.05)),
        "null_scale_rmse_01": float(np.quantile(scale_vals, 0.01)),
        "null_affine_rmse_mean": float(np.mean(affine_vals)),
        "null_affine_rmse_05": float(np.quantile(affine_vals, 0.05)),
        "null_affine_rmse_01": float(np.quantile(affine_vals, 0.01)),
        "null_phase_C_mean": float(np.nanmean(phase_vals)),
        "null_phase_C_95": float(np.nanquantile(phase_vals, 0.95)),
        "null_phase_C_99": float(np.nanquantile(phase_vals, 0.99)),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wells", default=DEFAULT_WELLS)
    parser.add_argument("--region-a", default=DEFAULT_REGION_A)
    parser.add_argument("--region-b", default=DEFAULT_REGION_B)
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    parser.add_argument("--n-null", type=int, default=DEFAULT_N_NULL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--pair-mode",
        default="koide",
        choices=["koide", "best_affine", "best_scale", "hybrid"],
        help=(
            "koide: compare each region's best Koide triplet. "
            "best_affine/best_scale/hybrid: search all triplet pairs."
        ),
    )

    args = parser.parse_args()

    if not os.path.exists(args.wells):
        raise FileNotFoundError(args.wells)

    rng = np.random.default_rng(args.seed)

    wells = pd.read_csv(args.wells)

    required = {"region", "k", "n_eff", "deltaD"}
    missing = required - set(wells.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    if "phi2" not in wells.columns:
        print("[warn] phi2 missing; phase coherence will be NaN.")
        wells["phi2"] = np.nan

    regions = sorted(wells["region"].unique().tolist())
    print("=" * 100)
    print("CROSS-REGION SCALING + PHASE COHERENCE TEST")
    print("=" * 100)
    print(f"[input] {args.wells}")
    print(f"[regions found] {regions}")
    print(f"[region A] {args.region_a}")
    print(f"[region B] {args.region_b}")
    print(f"[top wells] {args.top}")
    print(f"[null draws] {args.n_null}")
    print(f"[pair mode] {args.pair_mode}")
    print("=" * 100)

    if args.region_a not in regions:
        raise RuntimeError(f"region-a not found: {args.region_a}")

    if args.region_b not in regions:
        raise RuntimeError(f"region-b not found: {args.region_b}")

    A = wells[wells["region"] == args.region_a].copy()
    B = wells[wells["region"] == args.region_b].copy()

    top_a = min(args.top, len(A))
    top_b = min(args.top, len(B))

    if top_a < 3 or top_b < 3:
        raise RuntimeError("Need at least 3 wells in each region.")

    triplets_a = triplet_rows_from_top(A, top_a)
    triplets_b = triplet_rows_from_top(B, top_b)

    print(f"[triplets A] {len(triplets_a)}")
    print(f"[triplets B] {len(triplets_b)}")

    best_koide_a = choose_best_koide_triplet(triplets_a)
    best_koide_b = choose_best_koide_triplet(triplets_b)

    if args.pair_mode == "koide":
        obs_a = best_koide_a
        obs_b = best_koide_b
        pair_objective = None

    elif args.pair_mode == "best_affine":
        pair_objective, obs_a, obs_b, _ = choose_best_alignment_pair(
            triplets_a, triplets_b, mode="affine"
        )

    elif args.pair_mode == "best_scale":
        pair_objective, obs_a, obs_b, _ = choose_best_alignment_pair(
            triplets_a, triplets_b, mode="scale"
        )

    elif args.pair_mode == "hybrid":
        pair_objective, obs_a, obs_b, _ = choose_best_alignment_pair(
            triplets_a, triplets_b, mode="hybrid"
        )

    else:
        raise ValueError(args.pair_mode)

    obs_metrics = alignment_metrics(obs_a, obs_b)

    print("\n" + "-" * 100)
    print("[observed triplet A]")
    print(f"region={args.region_a}")
    print(f"n={obs_a['n']}")
    print(f"k={obs_a['k']}")
    print(f"Q_low={obs_a['Q_low']:.6f}, Q_high={obs_a['Q_high']:.6f}, "
          f"Q_mean={obs_a['Q_mean']:.6f}, epsK={obs_a['koide_error']:.6f}")
    print(f"mean_deltaD={obs_a['mean_deltaD']:.6f}")
    print(f"phi={obs_a['phi']}")

    print("\n[observed triplet B]")
    print(f"region={args.region_b}")
    print(f"n={obs_b['n']}")
    print(f"k={obs_b['k']}")
    print(f"Q_low={obs_b['Q_low']:.6f}, Q_high={obs_b['Q_high']:.6f}, "
          f"Q_mean={obs_b['Q_mean']:.6f}, epsK={obs_b['koide_error']:.6f}")
    print(f"mean_deltaD={obs_b['mean_deltaD']:.6f}")
    print(f"phi={obs_b['phi']}")

    print("\n[observed cross-region map]")
    print(f"pure scale: n_B ≈ {obs_metrics['scale_a']:.6f} n_A")
    print(f"pure scale RMSE = {obs_metrics['scale_rmse']:.6f}")
    print(f"affine: n_B ≈ {obs_metrics['affine_a']:.6f} n_A + {obs_metrics['affine_b']:.6f}")
    print(f"affine RMSE = {obs_metrics['affine_rmse']:.6f}")
    print(f"phase coherence C = {obs_metrics['phase_coherence']}")
    print(f"mean phase offset = {obs_metrics['mean_phase_offset']}")
    print(f"delta_phi = {obs_metrics['delta_phi']}")

    null = null_alignment(
        triplets_a=triplets_a,
        triplets_b=triplets_b,
        obs_scale_rmse=obs_metrics["scale_rmse"],
        obs_affine_rmse=obs_metrics["affine_rmse"],
        obs_phase_C=obs_metrics["phase_coherence"],
        n_null=args.n_null,
        rng=rng,
    )

    print("\n[null results]")
    for k, v in null.items():
        print(f"{k}: {v}")

    # Full pair table for audit.
    pair_rows = []
    for ia, ta in enumerate(triplets_a):
        for ib, tb in enumerate(triplets_b):
            m = alignment_metrics(ta, tb)
            pair_rows.append({
                "i_A": ia,
                "i_B": ib,
                "region_A": args.region_a,
                "region_B": args.region_b,

                "A_n1": ta["n"][0],
                "A_n2": ta["n"][1],
                "A_n3": ta["n"][2],
                "A_Q_low": ta["Q_low"],
                "A_Q_high": ta["Q_high"],
                "A_Q_mean": ta["Q_mean"],
                "A_koide_error": ta["koide_error"],
                "A_mean_deltaD": ta["mean_deltaD"],

                "B_n1": tb["n"][0],
                "B_n2": tb["n"][1],
                "B_n3": tb["n"][2],
                "B_Q_low": tb["Q_low"],
                "B_Q_high": tb["Q_high"],
                "B_Q_mean": tb["Q_mean"],
                "B_koide_error": tb["koide_error"],
                "B_mean_deltaD": tb["mean_deltaD"],

                "scale_a": m["scale_a"],
                "scale_rmse": m["scale_rmse"],
                "affine_a": m["affine_a"],
                "affine_b": m["affine_b"],
                "affine_rmse": m["affine_rmse"],
                "phase_coherence": m["phase_coherence"],
                "mean_phase_offset": m["mean_phase_offset"],
            })

    pair_df = pd.DataFrame(pair_rows)

    pair_df_sorted = pair_df.sort_values(
        ["affine_rmse", "scale_rmse"],
        ascending=[True, True],
    )

    out_pairs = os.path.join(OUTDIR, "cross_region_pair_table.csv")
    out_summary = os.path.join(OUTDIR, "cross_region_scaling_summary.json")
    out_top = os.path.join(OUTDIR, "cross_region_top_pairs.csv")

    pair_df.to_csv(out_pairs, index=False)
    pair_df_sorted.head(100).to_csv(out_top, index=False)

    payload = {
        "test": "cross_region_scaling_phase_coherence",
        "input_wells_csv": args.wells,
        "region_A": args.region_a,
        "region_B": args.region_b,
        "top_wells_A": top_a,
        "top_wells_B": top_b,
        "n_null": args.n_null,
        "pair_mode": args.pair_mode,
        "observed_A": {
            "n": obs_a["n"].tolist(),
            "k": obs_a["k"].tolist(),
            "deltaD": obs_a["deltaD"].tolist(),
            "phi": obs_a["phi"].tolist(),
            "Q_low": obs_a["Q_low"],
            "Q_high": obs_a["Q_high"],
            "Q_mean": obs_a["Q_mean"],
            "koide_error": obs_a["koide_error"],
            "mean_deltaD": obs_a["mean_deltaD"],
        },
        "observed_B": {
            "n": obs_b["n"].tolist(),
            "k": obs_b["k"].tolist(),
            "deltaD": obs_b["deltaD"].tolist(),
            "phi": obs_b["phi"].tolist(),
            "Q_low": obs_b["Q_low"],
            "Q_high": obs_b["Q_high"],
            "Q_mean": obs_b["Q_mean"],
            "koide_error": obs_b["koide_error"],
            "mean_deltaD": obs_b["mean_deltaD"],
        },
        "observed_map": obs_metrics,
        "null_results": null,
        "files": {
            "pair_table_csv": out_pairs,
            "top_pairs_csv": out_top,
            "summary_json": out_summary,
        },
        "interpretation": {
            "small_p_scale_rmse": "Pure dilation between regions is unusually good.",
            "small_p_affine_rmse": "Affine spectral map between regions is unusually good.",
            "high_phase_C": "Paired triplets have coherent phase offsets.",
            "small_p_affine_and_phase": "Frequency scaling and phase coherence jointly support a shared spectral skeleton.",
            "warning": "If pair_mode searches all pairs, p-values must be interpreted as alignment-search p-values, not fixed-hypothesis p-values.",
        },
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 100)
    print("TOP AFFINE-ALIGNED PAIRS")
    print("=" * 100)
    print(pair_df_sorted.head(20)[[
        "A_n1", "A_n2", "A_n3",
        "B_n1", "B_n2", "B_n3",
        "A_Q_mean", "B_Q_mean",
        "A_koide_error", "B_koide_error",
        "scale_a", "scale_rmse",
        "affine_a", "affine_b", "affine_rmse",
        "phase_coherence",
    ]].to_string(index=False))

    print("\nSaved:")
    print(f"  {out_pairs}")
    print(f"  {out_top}")
    print(f"  {out_summary}")


if __name__ == "__main__":
    main()
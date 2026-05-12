#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Well-First Koide Geometry Proof Test
------------------------------------

Purpose:
    Test whether Koide-like geometry appears in the raw wells before any
    Koide-comb model is imposed.

Input:
    well_first_wells.csv from script 19_koide_well.py

Core logic:
    The previous well-first script found raw local maxima in DeltaD(k).
    This script does not refit the data.
    It asks:

        Given the same number of raw wells, how often would random well
        locations produce triplets as close to Koide as the observed wells?

Definitions:
    For ordered raw well windings:

        n1 < n2 < n3

    define:

        Q_low  = n1/n2
        Q_high = n3/(2*n2)

    Koide target:

        Q = 2/3

    Koide error:

        eps_K = sqrt((Q_low - 2/3)^2 + (Q_high - 2/3)^2)

    Exact integer target:

        (10,15,20)

    Integer error:

        eps_int = sqrt((n1-10)^2 + (n2-15)^2 + (n3-20)^2)

Outputs:
    outputs_wct_well_proof/
        koide_geometry_proof_by_region.csv
        koide_geometry_proof_global.csv
        koide_geometry_proof_summary.json

Interpretation:
    Small p_uniform:
        Koide-like geometry is unlikely under uniformly random well positions.

    Small p_empirical:
        Koide-like geometry is unlikely even after using the empirical well
        distribution from all regions.

    Strongest possible result:
        signal region has small p and sidebands do not.

    Current expected result from your earlier output:
        B-low sideband likely has the strongest Koide-like raw geometry.
"""

import os
import json
import math
import argparse
from itertools import combinations

import numpy as np
import pandas as pd


# ============================================================
# Defaults matching your previous run
# ============================================================

DEFAULT_WELLS_CSV_OPTIONS = [
    "outputs_wct_well_first_koide/well_first_wells.csv",
    "well_first_wells.csv",
]

OUTDIR = "outputs_wct_well_proof"
os.makedirs(OUTDIR, exist_ok=True)

Q_KOIDE = 2.0 / 3.0

# From script 19
DELTA_ELL_ACTIVE = 4.780150335923678
K_SCAN_MIN = 6.0
K_SCAN_MAX = 32.0

N_MIN = K_SCAN_MIN * DELTA_ELL_ACTIVE / (2.0 * math.pi)
N_MAX = K_SCAN_MAX * DELTA_ELL_ACTIVE / (2.0 * math.pi)

DEFAULT_TOP_WELLS = 12
DEFAULT_N_NULL = 250_000
DEFAULT_BATCH = 10_000
DEFAULT_SEED = 20260509


# ============================================================
# Geometry functions
# ============================================================

def koide_metrics_for_triplet(n1, n2, n3):
    q_low = n1 / n2
    q_high = n3 / (2.0 * n2)
    q_mean = 0.5 * (q_low + q_high)

    eps_k = math.sqrt(
        (q_low - Q_KOIDE) ** 2 +
        (q_high - Q_KOIDE) ** 2
    )

    eps_int = math.sqrt(
        (n1 - 10.0) ** 2 +
        (n2 - 15.0) ** 2 +
        (n3 - 20.0) ** 2
    )

    return {
        "Q_low": q_low,
        "Q_high": q_high,
        "Q_mean": q_mean,
        "koide_error": eps_k,
        "integer_error_10_15_20": eps_int,
    }


def all_triplets_from_wells(wells_df, top_n):
    """
    Use the top_n wells by DeltaD, then sort them by n_eff.
    This matches the well-first script logic.
    """
    top = wells_df.sort_values("deltaD", ascending=False).head(top_n).copy()
    top = top.sort_values("n_eff").reset_index(drop=True)

    rows = []
    for i, j, k in combinations(range(len(top)), 3):
        w1 = top.iloc[i]
        w2 = top.iloc[j]
        w3 = top.iloc[k]

        n1, n2, n3 = float(w1["n_eff"]), float(w2["n_eff"]), float(w3["n_eff"])
        met = koide_metrics_for_triplet(n1, n2, n3)

        mean_deltaD = float((w1["deltaD"] + w2["deltaD"] + w3["deltaD"]) / 3.0)

        # Same style as previous script: reward strong wells and penalize geometry error.
        score = mean_deltaD / (
            1.0 +
            25.0 * met["koide_error"] +
            0.25 * met["integer_error_10_15_20"]
        )

        rows.append({
            "n1": n1,
            "n2": n2,
            "n3": n3,
            "k1": float(w1["k"]),
            "k2": float(w2["k"]),
            "k3": float(w3["k"]),
            "deltaD1": float(w1["deltaD"]),
            "deltaD2": float(w2["deltaD"]),
            "deltaD3": float(w3["deltaD"]),
            "mean_deltaD": mean_deltaD,
            "score": score,
            **met,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(
        ["koide_error", "integer_error_10_15_20", "score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def summarize_real_region(wells_df, top_n):
    trip = all_triplets_from_wells(wells_df, top_n)
    if trip.empty:
        return None, trip

    best_k = trip.sort_values(
        ["koide_error", "integer_error_10_15_20", "score"],
        ascending=[True, True, False],
    ).iloc[0]

    best_int = trip.sort_values(
        ["integer_error_10_15_20", "koide_error", "score"],
        ascending=[True, True, False],
    ).iloc[0]

    best_score = trip.sort_values(
        ["score"],
        ascending=[False],
    ).iloc[0]

    summary = {
        "best_koide_n1": float(best_k["n1"]),
        "best_koide_n2": float(best_k["n2"]),
        "best_koide_n3": float(best_k["n3"]),
        "best_koide_Q_low": float(best_k["Q_low"]),
        "best_koide_Q_high": float(best_k["Q_high"]),
        "best_koide_Q_mean": float(best_k["Q_mean"]),
        "best_koide_error": float(best_k["koide_error"]),
        "best_koide_integer_error": float(best_k["integer_error_10_15_20"]),
        "best_koide_score": float(best_k["score"]),

        "best_integer_n1": float(best_int["n1"]),
        "best_integer_n2": float(best_int["n2"]),
        "best_integer_n3": float(best_int["n3"]),
        "best_integer_koide_error": float(best_int["koide_error"]),
        "best_integer_error": float(best_int["integer_error_10_15_20"]),
        "best_integer_score": float(best_int["score"]),

        "best_score_n1": float(best_score["n1"]),
        "best_score_n2": float(best_score["n2"]),
        "best_score_n3": float(best_score["n3"]),
        "best_score_koide_error": float(best_score["koide_error"]),
        "best_score_integer_error": float(best_score["integer_error_10_15_20"]),
        "best_score": float(best_score["score"]),
    }

    return summary, trip


# ============================================================
# Null simulation
# ============================================================

def min_errors_for_random_ns(ns_batch, combo_idx):
    """
    ns_batch shape:
        (B, top_n)

    Assumes each row is already sorted ascending.

    Returns:
        min_koide_error per batch row
        min_integer_error per batch row
        joint_min_koide_at_low_integer not used here
    """
    i = combo_idx[:, 0]
    j = combo_idx[:, 1]
    k = combo_idx[:, 2]

    n1 = ns_batch[:, i]
    n2 = ns_batch[:, j]
    n3 = ns_batch[:, k]

    q_low = n1 / n2
    q_high = n3 / (2.0 * n2)

    koide_err = np.sqrt((q_low - Q_KOIDE) ** 2 + (q_high - Q_KOIDE) ** 2)
    int_err = np.sqrt((n1 - 10.0) ** 2 + (n2 - 15.0) ** 2 + (n3 - 20.0) ** 2)

    min_k = np.min(koide_err, axis=1)
    min_i = np.min(int_err, axis=1)

    # Lexicographic best: among triplets with lowest Koide error, integer error matters.
    # For p-values, also compute whether any triplet beats both real errors.
    return min_k, min_i, koide_err, int_err


def run_null_uniform(top_n, n_null, batch_size, rng):
    combo_idx = np.array(list(combinations(range(top_n), 3)), dtype=int)

    min_k_all = []
    min_i_all = []
    pair_k_all = []
    pair_i_all = []

    done = 0
    while done < n_null:
        b = min(batch_size, n_null - done)

        ns = rng.uniform(N_MIN, N_MAX, size=(b, top_n))
        ns.sort(axis=1)

        min_k, min_i, koide_err, int_err = min_errors_for_random_ns(ns, combo_idx)

        # Store pair arrays for joint p-value computation.
        # We only need min values and all pair minima thresholds are checked later
        # using a compressed metric:
        #   min over triplets of max(k_error / real_k, i_error / real_i)
        min_k_all.append(min_k)
        min_i_all.append(min_i)

        # For joint thresholds, store min over triplets of both later is impossible
        # without real threshold, so keep full compressed arrays expensive.
        # Instead return full per-null minima now and compute joint with rerun helper.
        done += b

    return {
        "min_koide_error": np.concatenate(min_k_all),
        "min_integer_error": np.concatenate(min_i_all),
    }


def run_null_empirical(top_n, n_null, batch_size, rng, empirical_pool):
    combo_idx = np.array(list(combinations(range(top_n), 3)), dtype=int)

    min_k_all = []
    min_i_all = []

    empirical_pool = np.asarray(empirical_pool, dtype=float)

    done = 0
    while done < n_null:
        b = min(batch_size, n_null - done)

        ns = rng.choice(empirical_pool, size=(b, top_n), replace=True)
        ns.sort(axis=1)

        min_k, min_i, _, _ = min_errors_for_random_ns(ns, combo_idx)

        min_k_all.append(min_k)
        min_i_all.append(min_i)

        done += b

    return {
        "min_koide_error": np.concatenate(min_k_all),
        "min_integer_error": np.concatenate(min_i_all),
    }


def run_joint_null(top_n, n_null, batch_size, rng, real_kerr, real_ierr, mode, empirical_pool=None):
    """
    Computes p(any random triplet has both:
        koide_error <= real_kerr
        integer_error <= real_ierr)

    This is the strongest "did random wells match both Koide ratio and exact
    (10,15,20)-like integer location?" test.
    """
    combo_idx = np.array(list(combinations(range(top_n), 3)), dtype=int)

    count_both = 0
    count_k_only = 0
    count_i_only = 0

    done = 0
    while done < n_null:
        b = min(batch_size, n_null - done)

        if mode == "uniform":
            ns = rng.uniform(N_MIN, N_MAX, size=(b, top_n))
        elif mode == "empirical":
            if empirical_pool is None:
                raise ValueError("empirical_pool required for empirical mode")
            ns = rng.choice(empirical_pool, size=(b, top_n), replace=True)
        else:
            raise ValueError(f"unknown mode {mode}")

        ns.sort(axis=1)

        i = combo_idx[:, 0]
        j = combo_idx[:, 1]
        k = combo_idx[:, 2]

        n1 = ns[:, i]
        n2 = ns[:, j]
        n3 = ns[:, k]

        q_low = n1 / n2
        q_high = n3 / (2.0 * n2)

        koide_err = np.sqrt((q_low - Q_KOIDE) ** 2 + (q_high - Q_KOIDE) ** 2)
        int_err = np.sqrt((n1 - 10.0) ** 2 + (n2 - 15.0) ** 2 + (n3 - 20.0) ** 2)

        has_k = np.any(koide_err <= real_kerr, axis=1)
        has_i = np.any(int_err <= real_ierr, axis=1)
        has_both_same_triplet = np.any((koide_err <= real_kerr) & (int_err <= real_ierr), axis=1)

        count_k_only += int(np.sum(has_k))
        count_i_only += int(np.sum(has_i))
        count_both += int(np.sum(has_both_same_triplet))

        done += b

    return {
        "p_koide_error": (1.0 + count_k_only) / (n_null + 1.0),
        "p_integer_error": (1.0 + count_i_only) / (n_null + 1.0),
        "p_joint_same_triplet": (1.0 + count_both) / (n_null + 1.0),
    }


# ============================================================
# I/O
# ============================================================

def find_wells_csv(path_arg):
    if path_arg:
        if not os.path.exists(path_arg):
            raise FileNotFoundError(path_arg)
        return path_arg

    for p in DEFAULT_WELLS_CSV_OPTIONS:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "Could not find well_first_wells.csv. Pass --wells path/to/well_first_wells.csv"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wells", default=None, help="Path to well_first_wells.csv")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_WELLS)
    parser.add_argument("--n-null", type=int, default=DEFAULT_N_NULL)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    wells_path = find_wells_csv(args.wells)

    print("=" * 100)
    print("WELL-FIRST KOIDE GEOMETRY PROOF TEST")
    print("=" * 100)
    print(f"[input] wells_csv={wells_path}")
    print(f"[config] top wells per region={args.top}")
    print(f"[config] null draws={args.n_null}")
    print(f"[config] n range from k scan: [{N_MIN:.6f}, {N_MAX:.6f}]")
    print("=" * 100)

    wells = pd.read_csv(wells_path)

    required = {"region", "k", "n_eff", "deltaD"}
    missing = required - set(wells.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    rng = np.random.default_rng(args.seed)

    # Empirical pool from all observed wells.
    empirical_pool = wells["n_eff"].dropna().values

    by_region_rows = []
    triplet_tables = {}

    for region, g in wells.groupby("region"):
        g = g.sort_values("deltaD", ascending=False).copy()

        if len(g) < 3:
            print(f"[skip] {region}: fewer than 3 wells")
            continue

        top_n = min(args.top, len(g))

        real_summary, real_triplets = summarize_real_region(g, top_n)
        if real_summary is None:
            continue

        triplet_tables[region] = real_triplets

        real_kerr = real_summary["best_koide_error"]
        real_ierr = real_summary["best_koide_integer_error"]

        print("\n" + "-" * 100)
        print(f"[region] {region}")
        print(f"  best Koide triplet n=({real_summary['best_koide_n1']:.4f}, "
              f"{real_summary['best_koide_n2']:.4f}, "
              f"{real_summary['best_koide_n3']:.4f})")
        print(f"  Q_low={real_summary['best_koide_Q_low']:.6f}, "
              f"Q_high={real_summary['best_koide_Q_high']:.6f}, "
              f"Q_mean={real_summary['best_koide_Q_mean']:.6f}")
        print(f"  koide_error={real_kerr:.6f}, integer_error={real_ierr:.6f}")

        # Null p-values.
        joint_uniform = run_joint_null(
            top_n=top_n,
            n_null=args.n_null,
            batch_size=args.batch,
            rng=rng,
            real_kerr=real_kerr,
            real_ierr=real_ierr,
            mode="uniform",
        )

        joint_empirical = run_joint_null(
            top_n=top_n,
            n_null=args.n_null,
            batch_size=args.batch,
            rng=rng,
            real_kerr=real_kerr,
            real_ierr=real_ierr,
            mode="empirical",
            empirical_pool=empirical_pool,
        )

        print("  [uniform null]")
        print(f"    p_koide_error       = {joint_uniform['p_koide_error']:.8g}")
        print(f"    p_integer_error     = {joint_uniform['p_integer_error']:.8g}")
        print(f"    p_joint_same_triplet= {joint_uniform['p_joint_same_triplet']:.8g}")

        print("  [empirical well-location null]")
        print(f"    p_koide_error       = {joint_empirical['p_koide_error']:.8g}")
        print(f"    p_integer_error     = {joint_empirical['p_integer_error']:.8g}")
        print(f"    p_joint_same_triplet= {joint_empirical['p_joint_same_triplet']:.8g}")

        row = {
            "region": region,
            "top_wells_used": top_n,
            **real_summary,

            "uniform_p_koide_error": joint_uniform["p_koide_error"],
            "uniform_p_integer_error": joint_uniform["p_integer_error"],
            "uniform_p_joint_same_triplet": joint_uniform["p_joint_same_triplet"],

            "empirical_p_koide_error": joint_empirical["p_koide_error"],
            "empirical_p_integer_error": joint_empirical["p_integer_error"],
            "empirical_p_joint_same_triplet": joint_empirical["p_joint_same_triplet"],
        }

        by_region_rows.append(row)

    by_region = pd.DataFrame(by_region_rows)

    if by_region.empty:
        raise RuntimeError("No region results produced.")

    # Global best region by Koide error and by joint p.
    global_rows = []
    for _, r in by_region.iterrows():
        global_rows.append({
            "region": r["region"],
            "best_koide_error": r["best_koide_error"],
            "best_koide_integer_error": r["best_koide_integer_error"],
            "uniform_p_joint_same_triplet": r["uniform_p_joint_same_triplet"],
            "empirical_p_joint_same_triplet": r["empirical_p_joint_same_triplet"],
            "best_koide_Q_mean": r["best_koide_Q_mean"],
            "best_koide_n1": r["best_koide_n1"],
            "best_koide_n2": r["best_koide_n2"],
            "best_koide_n3": r["best_koide_n3"],
        })

    global_df = pd.DataFrame(global_rows).sort_values(
        ["best_koide_error", "best_koide_integer_error"],
        ascending=[True, True],
    )

    by_region_csv = os.path.join(OUTDIR, "koide_geometry_proof_by_region.csv")
    global_csv = os.path.join(OUTDIR, "koide_geometry_proof_global.csv")
    summary_json = os.path.join(OUTDIR, "koide_geometry_proof_summary.json")

    by_region.to_csv(by_region_csv, index=False)
    global_df.to_csv(global_csv, index=False)

    # Save top triplet tables too.
    triplet_files = {}
    for region, trip in triplet_tables.items():
        safe = region.replace("/", "_").replace("\\", "_").replace(" ", "_")
        path = os.path.join(OUTDIR, f"triplets_{safe}.csv")
        trip.to_csv(path, index=False)
        triplet_files[region] = path

    payload = {
        "test": "well_first_koide_geometry_proof",
        "input_wells_csv": wells_path,
        "top_wells_per_region": args.top,
        "n_null": args.n_null,
        "n_range": [N_MIN, N_MAX],
        "koide_Q": Q_KOIDE,
        "definitions": {
            "Q_low": "n1/n2",
            "Q_high": "n3/(2*n2)",
            "koide_error": "sqrt((Q_low-2/3)^2 + (Q_high-2/3)^2)",
            "integer_error_10_15_20": "sqrt((n1-10)^2 + (n2-15)^2 + (n3-20)^2)",
            "uniform_null": "random well locations uniformly drawn over the original scan n-range",
            "empirical_null": "random well locations sampled from the observed pool of all well n_eff values",
        },
        "region_results": by_region.to_dict(orient="records"),
        "global_order": global_df.to_dict(orient="records"),
        "files": {
            "by_region_csv": by_region_csv,
            "global_csv": global_csv,
            "summary_json": summary_json,
            "triplet_files": triplet_files,
        },
        "interpretation": {
            "small_uniform_p_joint": "The observed triplet is unlikely under random uniform well placement.",
            "small_empirical_p_joint": "The observed triplet is unlikely even given the empirical well-location distribution.",
            "signal_specific": "Requires signal region to be significant while sidebands are not.",
            "sideband_carried": "If sideband has lower error and smaller p than signal, Koide-like geometry is strongest in background/sideband landscape.",
        },
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 100)
    print("GLOBAL ORDER BY RAW-WELL KOIDE ERROR")
    print("=" * 100)
    print(global_df.to_string(index=False))

    print("\nSaved:")
    print(f"  {by_region_csv}")
    print(f"  {global_csv}")
    print(f"  {summary_json}")
    for region, path in triplet_files.items():
        print(f"  {region}: {path}")


if __name__ == "__main__":
    main()
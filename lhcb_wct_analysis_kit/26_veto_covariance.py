#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
20_veto_covariance_koide_test.py

Veto-window covariance test for LHCb B0 -> K*0 mu+ mu- candidate spectra.

Purpose
-------
Test whether Koide-like triplet geometry remains stable when the charmonium
veto windows are varied.

Core idea:
    Raw scan gives frequency wells k_i.
    For each veto scheme, the retained active log-domain length changes:

        Delta ell_A = sum_j ln(q_hi_j / q_lo_j)

    Convert raw k to active-domain winding:

        n_i = k_i * Delta ell_A / (2*pi)

    Then enumerate well triplets and compute:

        Q_low  = n1 / n2
        Q_high = n3 / (2*n2)
        Q_mean = 0.5 * (Q_low + Q_high)

    Koide target:

        Q = 2/3

Outputs
-------
outputs_veto_covariance_koide/
    veto_covariance_wells.csv
    veto_covariance_triplets.csv
    veto_covariance_best_triplets.csv
    veto_covariance_region_summary.csv
    veto_covariance_report.json

Run
---
python 20_veto_covariance_koide_test.py

Requirements
------------
pip install uproot numpy pandas scipy matplotlib
"""

from __future__ import annotations

import os
import glob
import json
import math
import warnings
from dataclasses import dataclass, asdict
from itertools import combinations

import numpy as np
import pandas as pd

try:
    import uproot
except Exception as e:
    raise RuntimeError("Missing uproot. Install with: pip install uproot awkward") from e

try:
    from scipy.stats import gaussian_kde
    from scipy.optimize import minimize
    from scipy.signal import find_peaks
except Exception as e:
    raise RuntimeError("Missing scipy. Install with: pip install scipy") from e

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

OUTDIR = "outputs_veto_covariance_koide"
os.makedirs(OUTDIR, exist_ok=True)

ROOT_PATTERNS = [
    "data/*.dvntuple.root",
    "data/*.root",
]

Q2_MIN = 0.1
Q2_MAX = 19.0

B_SIGNAL = (5230.0, 5330.0)
B_LOW_SB = (5000.0, 5180.0)
B_HIGH_SB = (5380.0, 5600.0)

KST_SIGNAL = (795.9, 995.9)

REGIONS = [
    {
        "region": "B_low",
        "B_window": B_LOW_SB,
        "Kst_window": KST_SIGNAL,
    },
    {
        "region": "signal",
        "B_window": B_SIGNAL,
        "Kst_window": KST_SIGNAL,
    },
    {
        "region": "B_high",
        "B_window": B_HIGH_SB,
        "Kst_window": KST_SIGNAL,
    },
]

# Six veto schemes. Edit these if you want a different covariance ladder.
VETO_SCHEMES = [
    {
        "veto_label": "nominal",
        "jpsi": (8.00, 11.00),
        "psi2s": (12.50, 14.50),
    },
    {
        "veto_label": "narrow",
        "jpsi": (8.40, 10.60),
        "psi2s": (12.80, 14.20),
    },
    {
        "veto_label": "wide",
        "jpsi": (7.80, 11.20),
        "psi2s": (12.30, 14.70),
    },
    {
        "veto_label": "jpsi_wide",
        "jpsi": (7.70, 11.30),
        "psi2s": (12.50, 14.50),
    },
    {
        "veto_label": "psi2s_wide",
        "jpsi": (8.00, 11.00),
        "psi2s": (12.20, 14.80),
    },
    {
        "veto_label": "shifted",
        "jpsi": (8.20, 10.90),
        "psi2s": (12.70, 14.40),
    },
]

# Spectral scan.
K1_FIXED = 7.61054
K_SCAN_MIN = 6.0
K_SCAN_MAX = 32.0
N_K_SCAN = 1301

N_BINS = 240
KDE_BANDWIDTH_SCALE = 1.00

# Amplitude cap for the extra log-cos mode in the Poisson model.
A_MAX = 0.10
ETA_CLIP = 0.5

# Well finding.
MIN_PEAK_PROMINENCE = 1.0
MIN_PEAK_DISTANCE_K = 0.75
MAX_WELLS_FOR_TRIPLETS = 14

# Koide / comb targets.
Q_KOIDE = 2.0 / 3.0
INTEGER_TARGET = np.array([10.0, 15.0, 20.0], dtype=float)

# Useful for reporting how much of the enumerated triplet pool was already near Koide.
Q_NEAR_TOL = 0.02


# ============================================================
# Basic math
# ============================================================

def inv_mass2(px, py, pz, e):
    return e * e - px * px - py * py - pz * pz


def active_intervals_for_veto(veto):
    jlo, jhi = veto["jpsi"]
    plo, phi = veto["psi2s"]

    intervals = [
        (Q2_MIN, min(jlo, Q2_MAX)),
        (max(jhi, Q2_MIN), min(plo, Q2_MAX)),
        (max(phi, Q2_MIN), Q2_MAX),
    ]

    intervals = [(lo, hi) for lo, hi in intervals if hi > lo and lo > 0]
    return intervals


def active_delta_ell(intervals):
    return float(sum(math.log(hi / lo) for lo, hi in intervals))


def in_active_intervals(q2, intervals):
    q2 = np.asarray(q2, dtype=float)
    mask = np.zeros_like(q2, dtype=bool)
    for lo, hi in intervals:
        mask |= (q2 >= lo) & (q2 <= hi)
    return mask


def n_from_k(k, delta_ell_active):
    return np.asarray(k, dtype=float) * float(delta_ell_active) / (2.0 * math.pi)


def k_from_n(n, delta_ell_active):
    return 2.0 * math.pi * np.asarray(n, dtype=float) / float(delta_ell_active)


def poisson_deviance(y, lam):
    y = np.asarray(y, dtype=float)
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-12)

    out = lam - y
    nz = y > 0
    out[nz] += y[nz] * np.log(y[nz] / lam[nz])
    return float(2.0 * np.sum(out))


def ab_from_polar(r, phi):
    return float(r * math.cos(phi)), float(-r * math.sin(phi))


# ============================================================
# ROOT loading
# ============================================================

def find_root_files():
    files = []
    for pattern in ROOT_PATTERNS:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No ROOT files found in ./data/")
    return files


def candidate_branch(keys, options):
    keyset = set(keys)
    for name in options:
        if name in keyset:
            return name
    return None


def find_particle_component(keys, particle_patterns, comp):
    keys_list = list(keys)
    comp_upper = comp.upper()

    exact = []
    for p in particle_patterns:
        exact.extend([
            f"{p}_{comp}",
            f"{p}{comp}",
            f"{p}.{comp}",
        ])

    found = candidate_branch(keys_list, exact)
    if found:
        return found

    for k in keys_list:
        ku = k.upper()
        if not ku.endswith("_" + comp_upper):
            continue
        for p in particle_patterns:
            if p.upper() in ku:
                return k

    return None


def derive_q2_from_muons(tree):
    keys = list(tree.keys())

    plus_patterns = [
        "muplus", "mu_plus", "mup", "mu_p",
        "muplus0", "muplus_0", "MuPlus",
    ]
    minus_patterns = [
        "muminus", "mu_minus", "mum", "mu_m",
        "muminus0", "muminus_0", "MuMinus",
    ]

    pxp = find_particle_component(keys, plus_patterns, "PX")
    pyp = find_particle_component(keys, plus_patterns, "PY")
    pzp = find_particle_component(keys, plus_patterns, "PZ")
    pep = find_particle_component(keys, plus_patterns, "PE")

    pxm = find_particle_component(keys, minus_patterns, "PX")
    pym = find_particle_component(keys, minus_patterns, "PY")
    pzm = find_particle_component(keys, minus_patterns, "PZ")
    pem = find_particle_component(keys, minus_patterns, "PE")

    needed = {
        "mu_plus_PX": pxp,
        "mu_plus_PY": pyp,
        "mu_plus_PZ": pzp,
        "mu_plus_PE": pep,
        "mu_minus_PX": pxm,
        "mu_minus_PY": pym,
        "mu_minus_PZ": pzm,
        "mu_minus_PE": pem,
    }

    missing = [name for name, branch in needed.items() if branch is None]
    if missing:
        print("\n[debug] Missing muon branches:")
        for m in missing:
            print("   ", m)

        print("\n[debug] Branches containing MU:")
        mu_keys = [k for k in keys if "MU" in k.upper()]
        for k in mu_keys[:250]:
            print("   ", k)

        raise RuntimeError("Could not derive q2 from muon four-vectors.")

    branches = [pxp, pyp, pzp, pep, pxm, pym, pzm, pem]
    arr = tree.arrays(branches, library="np")

    Ep = np.asarray(arr[pep], dtype=float)
    px_p = np.asarray(arr[pxp], dtype=float)
    py_p = np.asarray(arr[pyp], dtype=float)
    pz_p = np.asarray(arr[pzp], dtype=float)

    Em = np.asarray(arr[pem], dtype=float)
    px_m = np.asarray(arr[pxm], dtype=float)
    py_m = np.asarray(arr[pym], dtype=float)
    pz_m = np.asarray(arr[pzm], dtype=float)

    E = Ep + Em
    px = px_p + px_m
    py = py_p + py_m
    pz = pz_p + pz_m

    q2_mev2 = inv_mass2(px, py, pz, E)
    q2_gev2 = q2_mev2 / 1.0e6

    return q2_gev2


def first_tree(f):
    preferred = "B0_KstMuMu/DecayTree"

    if preferred in f:
        return f[preferred]

    for key in f.keys(recursive=True):
        obj = f[key]
        if hasattr(obj, "keys") and hasattr(obj, "arrays"):
            if "DecayTree" in key:
                return obj

    for key in f.keys(recursive=True):
        obj = f[key]
        if hasattr(obj, "keys") and hasattr(obj, "arrays"):
            return obj

    raise RuntimeError("No TTree-like object found.")


def load_all_events():
    files = find_root_files()

    direct_q2_candidates = [
        "q2", "Q2", "q2_DTF", "Q2_DTF",
        "mumu_M2", "dimuon_M2",
        "Jpsi_M2", "J_psi_1S_M2", "J_psi_1S_MM2",
    ]

    b_mass_candidates = [
        "B0_M", "B0_MM", "B_M", "B_MM",
    ]

    kst_mass_candidates = [
        "Kst_892_0_M",
        "Kst_892_0_MM",
        "Kst_M",
        "Kst_MM",
        "Kstar_M",
        "Kstar_MM",
        "Kstar0_M",
        "Kstar0_MM",
    ]

    rows = []

    for path in files:
        print(f"[load] {path}")

        with uproot.open(path) as f:
            tree = first_tree(f)
            keys = list(tree.keys())

            q2_branch = candidate_branch(keys, direct_q2_candidates)
            b_branch = candidate_branch(keys, b_mass_candidates)
            kst_branch = candidate_branch(keys, kst_mass_candidates)

            if b_branch is None:
                raise RuntimeError(f"No B mass branch found in {path}")
            if kst_branch is None:
                raise RuntimeError(f"No K* mass branch found in {path}")

            if q2_branch is not None:
                branches = [q2_branch, b_branch, kst_branch]
                arr = tree.arrays(branches, library="np")

                q2 = np.asarray(arr[q2_branch], dtype=float)
                finite = q2[np.isfinite(q2)]
                if len(finite) == 0:
                    continue

                # Convert MeV^2 to GeV^2 if needed.
                if np.nanmedian(finite) > 1e4:
                    q2 = q2 / 1.0e6

                print(f"[q2] using branch {q2_branch}")

            else:
                q2 = derive_q2_from_muons(tree)
                branches = [b_branch, kst_branch]
                arr = tree.arrays(branches, library="np")
                print("[q2] derived from muon four-vectors")

            bm = np.asarray(arr[b_branch], dtype=float)
            km = np.asarray(arr[kst_branch], dtype=float)

            mask = np.isfinite(q2) & np.isfinite(bm) & np.isfinite(km)
            mask &= (q2 >= Q2_MIN) & (q2 <= Q2_MAX)

            df = pd.DataFrame({
                "q2": q2[mask],
                "B_M": bm[mask],
                "Kst_M": km[mask],
                "source_file": os.path.basename(path),
            })

            rows.append(df)

    if not rows:
        raise RuntimeError("No events loaded.")

    out = pd.concat(rows, ignore_index=True)
    print(f"[info] loaded q2-range events: {len(out):,}")
    return out


# ============================================================
# Spectrum and fit
# ============================================================

def select_region(df, region_cfg, intervals):
    blo, bhi = region_cfg["B_window"]
    klo, khi = region_cfg["Kst_window"]

    q2 = df["q2"].to_numpy(float)

    mask = np.isfinite(q2)
    mask &= (df["B_M"].to_numpy(float) >= blo) & (df["B_M"].to_numpy(float) <= bhi)
    mask &= (df["Kst_M"].to_numpy(float) >= klo) & (df["Kst_M"].to_numpy(float) <= khi)
    mask &= in_active_intervals(q2, intervals)

    return df.loc[mask].copy()


def make_histogram(q2, intervals):
    ell = np.log(q2)

    ell_min = math.log(Q2_MIN)
    ell_max = math.log(Q2_MAX)

    counts, edges = np.histogram(ell, bins=N_BINS, range=(ell_min, ell_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    q2_centers = np.exp(centers)

    active = in_active_intervals(q2_centers, intervals)

    return centers[active], counts[active].astype(float), q2_centers[active]


def kde_baseline(ell_centers, counts, bw_scale=KDE_BANDWIDTH_SCALE):
    repeated = np.repeat(ell_centers, np.maximum(counts.astype(int), 0))

    if len(repeated) < 100:
        raise RuntimeError("Too few events for KDE baseline.")

    kde = gaussian_kde(repeated)
    kde.set_bandwidth(kde.factor * bw_scale)

    dens = kde(ell_centers)
    dens = np.maximum(dens, 1e-12)

    baseline = dens / np.sum(dens) * np.sum(counts)
    return np.maximum(baseline, 1e-9)


def nll_eta(theta, y, B, ell, k2=None):
    C = theta[0]
    r1 = theta[1]
    p1 = theta[2]
    a1, b1 = ab_from_polar(r1, p1)

    eta = C + a1 * np.cos(K1_FIXED * ell) + b1 * np.sin(K1_FIXED * ell)

    if k2 is not None:
        r2 = theta[3]
        p2 = theta[4]
        a2, b2 = ab_from_polar(r2, p2)
        eta = eta + a2 * np.cos(k2 * ell) + b2 * np.sin(k2 * ell)

    eta = np.clip(eta, -ETA_CLIP, ETA_CLIP)
    lam = np.maximum(B * np.exp(eta), 1e-12)
    return float(np.sum(lam - y * np.log(lam)))


def fit_base(y, B, ell):
    y = np.asarray(y, dtype=float)
    B = np.maximum(np.asarray(B, dtype=float), 1e-12)
    ell = np.asarray(ell, dtype=float)

    scale = np.sum(y) / max(np.sum(B), 1e-12)
    C0 = float(np.clip(np.log(max(scale, 1e-12)), -ETA_CLIP, ETA_CLIP))

    starts = [
        [C0, 0.0, 0.0],
        [C0, 0.5 * A_MAX, 0.0],
        [C0, A_MAX, 0.0],
        [C0, A_MAX, math.pi / 2],
        [C0, A_MAX, -math.pi / 2],
    ]

    bounds = [
        (-ETA_CLIP, ETA_CLIP),
        (0.0, A_MAX),
        (-math.pi, math.pi),
    ]

    best = None
    for x0 in starts:
        res = minimize(
            nll_eta,
            x0=np.asarray(x0, dtype=float),
            args=(y, B, ell, None),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1500, "ftol": 1e-12, "gtol": 1e-8},
        )
        if best is None or res.fun < best.fun:
            best = res

    theta = np.asarray(best.x, dtype=float)
    C, r1, p1 = theta
    a1, b1 = ab_from_polar(r1, p1)

    eta = C + a1 * np.cos(K1_FIXED * ell) + b1 * np.sin(K1_FIXED * ell)
    eta = np.clip(eta, -ETA_CLIP, ETA_CLIP)
    lam = np.maximum(B * np.exp(eta), 1e-12)

    return {
        "theta": theta,
        "lambda": lam,
        "D": poisson_deviance(y, lam),
        "C": float(C),
        "A1": float(r1),
        "phi1": float(p1),
        "success": bool(best.success),
    }


def fit_extra_mode(y, B, ell, base, k2):
    starts = []

    C0, r10, p10 = base["theta"]
    phase_seeds = [0.0, math.pi / 2, -math.pi / 2, math.pi]
    amp_seeds = [0.0, 0.25 * A_MAX, 0.5 * A_MAX, A_MAX]

    for amp in amp_seeds:
        for ph in phase_seeds:
            starts.append([C0, r10, p10, amp, ph])

    bounds = [
        (-ETA_CLIP, ETA_CLIP),
        (0.0, A_MAX),
        (-math.pi, math.pi),
        (0.0, A_MAX),
        (-math.pi, math.pi),
    ]

    best = None
    for x0 in starts:
        res = minimize(
            nll_eta,
            x0=np.asarray(x0, dtype=float),
            args=(y, B, ell, float(k2)),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1500, "ftol": 1e-12, "gtol": 1e-8},
        )
        if best is None or res.fun < best.fun:
            best = res

    theta = np.asarray(best.x, dtype=float)
    C, r1, p1, r2, p2 = theta

    a1, b1 = ab_from_polar(r1, p1)
    a2, b2 = ab_from_polar(r2, p2)

    eta = (
        C
        + a1 * np.cos(K1_FIXED * ell)
        + b1 * np.sin(K1_FIXED * ell)
        + a2 * np.cos(k2 * ell)
        + b2 * np.sin(k2 * ell)
    )
    eta = np.clip(eta, -ETA_CLIP, ETA_CLIP)
    lam = np.maximum(B * np.exp(eta), 1e-12)

    D = poisson_deviance(y, lam)

    return {
        "D": float(D),
        "deltaD": float(base["D"] - D),
        "A2": float(r2),
        "phi2": float(p2),
        "success": bool(best.success),
    }


def scan_wells(y, B, ell):
    base = fit_base(y, B, ell)

    k_grid = np.linspace(K_SCAN_MIN, K_SCAN_MAX, N_K_SCAN)
    rows = []

    for idx, k2 in enumerate(k_grid):
        fit = fit_extra_mode(y, B, ell, base, k2)
        rows.append({
            "k": float(k2),
            "deltaD": float(fit["deltaD"]),
            "A2": float(fit["A2"]),
            "phi2": float(fit["phi2"]),
            "success": bool(fit["success"]),
        })

        if (idx + 1) % 200 == 0:
            print(f"    k scan {idx + 1}/{len(k_grid)}")

    scan_df = pd.DataFrame(rows)

    dk = float(k_grid[1] - k_grid[0])
    peak_distance = max(1, int(round(MIN_PEAK_DISTANCE_K / dk)))

    peaks, props = find_peaks(
        scan_df["deltaD"].to_numpy(float),
        prominence=MIN_PEAK_PROMINENCE,
        distance=peak_distance,
    )

    wells = scan_df.iloc[peaks].copy()
    if len(wells) == 0:
        # Fallback: keep the global maximum.
        wells = scan_df.sort_values("deltaD", ascending=False).head(1).copy()
        wells["prominence"] = np.nan
    else:
        wells["prominence"] = props.get("prominences", np.full(len(wells), np.nan))

    wells = wells.sort_values("deltaD", ascending=False).head(MAX_WELLS_FOR_TRIPLETS)
    wells = wells.sort_values("k").reset_index(drop=True)
    wells["well_rank_by_k"] = np.arange(len(wells))
    wells["well_rank_by_score"] = wells["deltaD"].rank(ascending=False, method="first").astype(int)

    return scan_df, wells, base


# ============================================================
# Triplet analysis
# ============================================================

def enumerate_triplets(wells, delta_ell_active):
    rows = []

    if len(wells) < 3:
        return pd.DataFrame()

    for i, j, l in combinations(range(len(wells)), 3):
        w = wells.iloc[[i, j, l]].copy().sort_values("k")

        ks = w["k"].to_numpy(float)
        ns = n_from_k(ks, delta_ell_active)

        n1, n2, n3 = ns

        if n2 <= 0:
            continue

        q_low = n1 / n2
        q_high = n3 / (2.0 * n2)
        q_mean = 0.5 * (q_low + q_high)

        koide_error = abs(q_mean - Q_KOIDE)
        q_internal_spread = abs(q_low - q_high)

        integer_error = float(np.sqrt(np.mean((np.sort(ns) - INTEGER_TARGET) ** 2)))

        score = float(np.sum(w["deltaD"].to_numpy(float)))
        min_score = float(np.min(w["deltaD"].to_numpy(float)))
        mean_score = float(np.mean(w["deltaD"].to_numpy(float)))

        rows.append({
            "well_i": int(i),
            "well_j": int(j),
            "well_k": int(l),

            "k1_triplet": float(ks[0]),
            "k2_triplet": float(ks[1]),
            "k3_triplet": float(ks[2]),

            "n1_triplet": float(n1),
            "n2_triplet": float(n2),
            "n3_triplet": float(n3),

            "Q_low": float(q_low),
            "Q_high": float(q_high),
            "Q_mean": float(q_mean),
            "koide_error": float(koide_error),
            "Q_internal_spread": float(q_internal_spread),

            "integer_error_10_15_20": float(integer_error),

            "score": score,
            "mean_score": mean_score,
            "min_score": min_score,
        })

    out = pd.DataFrame(rows)

    if len(out):
        # This intentionally reproduces the selector caveat:
        # it hunts for Q near 2/3 first, then cleaner integer comb, then score.
        out = out.sort_values(
            ["koide_error", "integer_error_10_15_20", "score"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
        out["triplet_selector_rank"] = np.arange(1, len(out) + 1)

    return out


def coefficient_of_variation(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    mean_abs = abs(float(np.mean(x)))
    if mean_abs < 1e-12:
        return np.nan

    return float(np.std(x, ddof=1) / mean_abs) if len(x) > 1 else 0.0


def summarize_region(best_df, all_triplets_df):
    rows = []

    for region, g in best_df.groupby("region"):
        q_vals = g["Q_mean"].to_numpy(float)
        koide_dist = np.abs(q_vals - Q_KOIDE)

        # Stability of selected triplet coordinates across veto schemes.
        n_cols = ["n1_triplet", "n2_triplet", "n3_triplet"]
        k_cols = ["k1_triplet", "k2_triplet", "k3_triplet"]

        n_all = g[n_cols].to_numpy(float).ravel()
        k_all = g[k_cols].to_numpy(float).ravel()

        cv_n = coefficient_of_variation(n_all)
        cv_k = coefficient_of_variation(k_all)
        cv_ratio_n_over_k = cv_n / cv_k if np.isfinite(cv_n) and np.isfinite(cv_k) and cv_k > 0 else np.nan

        near_fracs = []
        for _, row in g.iterrows():
            sub = all_triplets_df[
                (all_triplets_df["region"] == row["region"])
                & (all_triplets_df["veto_label"] == row["veto_label"])
            ]

            if len(sub) == 0:
                continue

            near = np.mean(np.abs(sub["Q_mean"].to_numpy(float) - Q_KOIDE) <= Q_NEAR_TOL)
            near_fracs.append(float(near))

        rows.append({
            "region": region,
            "n_veto_schemes": int(len(g)),

            "Q_mean_mean": float(np.mean(q_vals)),
            "Q_mean_std": float(np.std(q_vals, ddof=1)) if len(q_vals) > 1 else 0.0,
            "Q_mean_min": float(np.min(q_vals)),
            "Q_mean_max": float(np.max(q_vals)),
            "mean_abs_Q_minus_2over3": float(np.mean(koide_dist)),
            "min_abs_Q_minus_2over3": float(np.min(koide_dist)),
            "max_abs_Q_minus_2over3": float(np.max(koide_dist)),

            "koide_error_mean": float(np.mean(g["koide_error"])),
            "koide_error_std": float(np.std(g["koide_error"], ddof=1)) if len(g) > 1 else 0.0,
            "koide_error_cv": coefficient_of_variation(g["koide_error"]),

            "integer_error_10_15_20_mean": float(np.mean(g["integer_error_10_15_20"])),
            "integer_error_10_15_20_min": float(np.min(g["integer_error_10_15_20"])),
            "integer_error_10_15_20_max": float(np.max(g["integer_error_10_15_20"])),

            "CV_n": cv_n,
            "CV_k": cv_k,
            "CV_n_over_CV_k": cv_ratio_n_over_k,
            "n_more_stable_than_k": bool(cv_n < cv_k) if np.isfinite(cv_n) and np.isfinite(cv_k) else False,

            "triplet_pool_near_Q_frac_mean": float(np.mean(near_fracs)) if near_fracs else np.nan,
            "triplet_pool_near_Q_frac_min": float(np.min(near_fracs)) if near_fracs else np.nan,
            "triplet_pool_near_Q_frac_max": float(np.max(near_fracs)) if near_fracs else np.nan,
        })

    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================

def make_plots(best_df, summary_df):
    if len(best_df) == 0:
        return

    plt.figure(figsize=(9, 5))
    for region, g in best_df.groupby("region"):
        plt.plot(
            g["delta_ell_active"],
            g["Q_mean"],
            marker="o",
            linestyle="-",
            label=region,
        )
    plt.axhline(Q_KOIDE, linestyle="--", linewidth=1, label="Q = 2/3")
    plt.xlabel(r"Active log-domain length $\Delta \ell_A$")
    plt.ylabel(r"Selected triplet $Q_{\rm mean}$")
    plt.title("Veto-window covariance: Q stability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "Q_mean_vs_delta_ell.png"), dpi=200)
    plt.close()

    if len(summary_df):
        plt.figure(figsize=(8, 5))
        x = np.arange(len(summary_df))
        width = 0.35
        plt.bar(x - width / 2, summary_df["CV_n"], width, label="CV(n)")
        plt.bar(x + width / 2, summary_df["CV_k"], width, label="CV(k)")
        plt.xticks(x, summary_df["region"].tolist())
        plt.ylabel("Coefficient of variation")
        plt.title("Active-domain n stability vs raw k stability")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "CV_n_vs_CV_k.png"), dpi=200)
        plt.close()


# ============================================================
# Main
# ============================================================

def main():
    df = load_all_events()

    all_wells = []
    all_triplets = []
    best_triplets = []

    for veto in VETO_SCHEMES:
        intervals = active_intervals_for_veto(veto)
        delta_ell = active_delta_ell(intervals)

        print("\n" + "=" * 90)
        print(f"[veto] {veto['veto_label']}")
        print(f"       J/psi={veto['jpsi']} psi2S={veto['psi2s']}")
        print(f"       active intervals={intervals}")
        print(f"       Delta ell_A={delta_ell:.10f}")
        print("=" * 90)

        for region_cfg in REGIONS:
            region = region_cfg["region"]

            print("\n" + "-" * 80)
            print(f"[region] {region}")
            print("-" * 80)

            sub = select_region(df, region_cfg, intervals)
            n_events = len(sub)
            print(f"[select] events={n_events:,}")

            if n_events < 500:
                warnings.warn(f"Skipping {region} / {veto['veto_label']}: too few events.")
                continue

            ell, counts, q2_centers = make_histogram(sub["q2"].to_numpy(float), intervals)
            baseline = kde_baseline(ell, counts)

            scan_df, wells, base = scan_wells(counts, baseline, ell)

            wells = wells.copy()
            wells["region"] = region
            wells["veto_label"] = veto["veto_label"]
            wells["jpsi_lo"] = veto["jpsi"][0]
            wells["jpsi_hi"] = veto["jpsi"][1]
            wells["psi2s_lo"] = veto["psi2s"][0]
            wells["psi2s_hi"] = veto["psi2s"][1]
            wells["delta_ell_active"] = delta_ell
            wells["n_eff"] = n_from_k(wells["k"].to_numpy(float), delta_ell)
            wells["n_events"] = n_events
            wells["D_base"] = float(base["D"])
            wells["A1_base"] = float(base["A1"])

            all_wells.append(wells)

            triplets = enumerate_triplets(wells, delta_ell)
            if len(triplets) == 0:
                warnings.warn(f"No triplets for {region} / {veto['veto_label']}")
                continue

            triplets["region"] = region
            triplets["veto_label"] = veto["veto_label"]
            triplets["jpsi_lo"] = veto["jpsi"][0]
            triplets["jpsi_hi"] = veto["jpsi"][1]
            triplets["psi2s_lo"] = veto["psi2s"][0]
            triplets["psi2s_hi"] = veto["psi2s"][1]
            triplets["delta_ell_active"] = delta_ell
            triplets["n_events"] = n_events
            triplets["n_wells"] = len(wells)
            triplets["n_triplets_total"] = len(triplets)

            q_near_frac = float(np.mean(np.abs(triplets["Q_mean"].to_numpy(float) - Q_KOIDE) <= Q_NEAR_TOL))
            triplets["triplet_pool_near_Q_frac"] = q_near_frac

            all_triplets.append(triplets)

            best = triplets.iloc[0].copy()
            best_triplets.append(best)

            print(
                "[best] "
                f"Q_mean={best['Q_mean']:.6f}, "
                f"|Q-2/3|={best['koide_error']:.6f}, "
                f"n=({best['n1_triplet']:.3f}, {best['n2_triplet']:.3f}, {best['n3_triplet']:.3f}), "
                f"k=({best['k1_triplet']:.3f}, {best['k2_triplet']:.3f}, {best['k3_triplet']:.3f}), "
                f"int_err={best['integer_error_10_15_20']:.3f}, "
                f"near_Q_pool={100*q_near_frac:.1f}%"
            )

    if not all_wells or not all_triplets or not best_triplets:
        raise RuntimeError("No usable results were produced.")

    wells_df = pd.concat(all_wells, ignore_index=True)
    triplets_df = pd.concat(all_triplets, ignore_index=True)
    best_df = pd.DataFrame(best_triplets).reset_index(drop=True)

    summary_df = summarize_region(best_df, triplets_df)

    wells_path = os.path.join(OUTDIR, "veto_covariance_wells.csv")
    triplets_path = os.path.join(OUTDIR, "veto_covariance_triplets.csv")
    best_path = os.path.join(OUTDIR, "veto_covariance_best_triplets.csv")
    summary_path = os.path.join(OUTDIR, "veto_covariance_region_summary.csv")
    report_path = os.path.join(OUTDIR, "veto_covariance_report.json")

    wells_df.to_csv(wells_path, index=False)
    triplets_df.to_csv(triplets_path, index=False)
    best_df.to_csv(best_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    report = {
        "config": {
            "Q2_MIN": Q2_MIN,
            "Q2_MAX": Q2_MAX,
            "K1_FIXED": K1_FIXED,
            "K_SCAN_MIN": K_SCAN_MIN,
            "K_SCAN_MAX": K_SCAN_MAX,
            "N_K_SCAN": N_K_SCAN,
            "N_BINS": N_BINS,
            "KDE_BANDWIDTH_SCALE": KDE_BANDWIDTH_SCALE,
            "A_MAX": A_MAX,
            "Q_KOIDE": Q_KOIDE,
            "INTEGER_TARGET": INTEGER_TARGET.tolist(),
            "Q_NEAR_TOL": Q_NEAR_TOL,
        },
        "veto_schemes": VETO_SCHEMES,
        "regions": REGIONS,
        "outputs": {
            "wells_csv": wells_path,
            "triplets_csv": triplets_path,
            "best_triplets_csv": best_path,
            "summary_csv": summary_path,
        },
        "region_summary": summary_df.to_dict(orient="records"),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    make_plots(best_df, summary_df)

    print("\n" + "=" * 90)
    print("REGION SUMMARY")
    print("=" * 90)

    cols = [
        "region",
        "Q_mean_mean",
        "Q_mean_std",
        "mean_abs_Q_minus_2over3",
        "CV_n",
        "CV_k",
        "CV_n_over_CV_k",
        "n_more_stable_than_k",
        "triplet_pool_near_Q_frac_mean",
        "integer_error_10_15_20_min",
        "integer_error_10_15_20_max",
    ]

    print(summary_df[cols].to_string(index=False))

    print("\nSaved:")
    print(f"  {wells_path}")
    print(f"  {triplets_path}")
    print(f"  {best_path}")
    print(f"  {summary_path}")
    print(f"  {report_path}")
    print(f"  {os.path.join(OUTDIR, 'Q_mean_vs_delta_ell.png')}")
    print(f"  {os.path.join(OUTDIR, 'CV_n_vs_CV_k.png')}")


if __name__ == "__main__":
    main()
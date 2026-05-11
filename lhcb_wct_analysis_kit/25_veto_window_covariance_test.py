#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
25_veto_window_covariance_test.py

Veto-Window Covariance / Active-Domain Invariance Test
-----------------------------------------------------

Purpose:
    Test whether the observed log-periodic structure is invariant in
    active-domain winding space when charmonium veto windows are varied.

Core WCT-specific question:
    If the active support changes, then

        Delta ell_A changes.

    A true active-domain winding structure should remain stable in

        n = k Delta ell_A / (2 pi),

    even if the raw frequency k shifts.

This script runs a well-first scan over several veto-window definitions:

    1. Build active q^2 support.
    2. Compute Delta ell_A.
    3. Scan one extra mode continuously in k.
    4. Find raw wells.
    5. Convert wells to n_eff.
    6. Test whether well triplets remain near:
            (10, 15, 20)
       and whether ratios remain near:
            Q = 2/3.

Outputs:
    outputs_wct_veto_covariance/
        veto_covariance_wells.csv
        veto_covariance_triplets.csv
        veto_covariance_summary.csv
        veto_covariance_summary.json

Decision:
    WCT-supporting:
        k shifts with veto windows, but n_eff and Q remain stable.

    Artifact-like:
        k stays fixed while n_eff drifts, or both k and n_eff drift randomly.
"""

import os
import glob
import json
import math
from dataclasses import dataclass, asdict
from itertools import combinations

import numpy as np
import pandas as pd

try:
    import uproot
except Exception as e:
    raise RuntimeError("Missing uproot. Install with: pip install uproot awkward") from e

try:
    from scipy.optimize import minimize
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks
except Exception as e:
    raise RuntimeError("Missing scipy. Install with: pip install scipy") from e


# ============================================================
# Config
# ============================================================

OUTDIR = "outputs_wct_veto_covariance"
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
        "region": "B_low_sideband_Kst_signal",
        "B_window": B_LOW_SB,
        "Kst_window": KST_SIGNAL,
    },
    {
        "region": "signal_B_signal_Kst",
        "B_window": B_SIGNAL,
        "Kst_window": KST_SIGNAL,
    },
    {
        "region": "B_high_sideband_Kst_signal",
        "B_window": B_HIGH_SB,
        "Kst_window": KST_SIGNAL,
    },
]

# Veto definitions.
# Each entry removes [JPSI_LO, JPSI_HI] and [PSI2S_LO, PSI2S_HI].
# Active intervals become:
#   (Q2_MIN, JPSI_LO), (JPSI_HI, PSI2S_LO), (PSI2S_HI, Q2_MAX)
VETO_SCHEMES = [
    {
        "label": "tight",
        "jpsi": (8.5, 10.5),
        "psi2s": (12.8, 14.2),
    },
    {
        "label": "baseline_wide",
        "jpsi": (8.0, 11.0),
        "psi2s": (12.5, 14.5),
    },
    {
        "label": "wider",
        "jpsi": (7.5, 11.5),
        "psi2s": (12.25, 14.75),
    },
    {
        "label": "very_wide",
        "jpsi": (7.0, 12.0),
        "psi2s": (12.0, 15.0),
    },
    {
        "label": "shift_low",
        "jpsi": (7.8, 10.8),
        "psi2s": (12.3, 14.3),
    },
    {
        "label": "shift_high",
        "jpsi": (8.2, 11.2),
        "psi2s": (12.7, 14.7),
    },
]

K1_FIXED = 7.61054

K_SCAN_MIN = 6.0
K_SCAN_MAX = 36.0
N_K_SCAN = 1501

N_BINS = 240
KDE_BANDWIDTH_SCALE = 1.00

A_MAX = 0.05

MIN_PEAK_PROMINENCE = 1.0
MIN_PEAK_DISTANCE_K = 0.75

TOP_WELLS_FOR_TRIPLETS = 12

KOIDE_Q = 2.0 / 3.0


# ============================================================
# Active-domain helpers
# ============================================================

def active_intervals_from_veto(jpsi, psi2s):
    intervals = [
        (Q2_MIN, float(jpsi[0])),
        (float(jpsi[1]), float(psi2s[0])),
        (float(psi2s[1]), Q2_MAX),
    ]

    clean = []
    for lo, hi in intervals:
        lo = max(lo, Q2_MIN)
        hi = min(hi, Q2_MAX)
        if hi > lo:
            clean.append((lo, hi))

    return clean


def active_delta_ell(intervals):
    return sum(math.log(hi / lo) for lo, hi in intervals)


def in_active_intervals(q2, intervals):
    mask = np.zeros_like(q2, dtype=bool)
    for lo, hi in intervals:
        mask |= (q2 >= lo) & (q2 <= hi)
    return mask


def n_from_k(k, delta_ell):
    return float(k) * float(delta_ell) / (2.0 * math.pi)


def k_from_n(n, delta_ell):
    return 2.0 * math.pi * float(n) / float(delta_ell)


# ============================================================
# ROOT loading
# ============================================================

def find_root_files():
    files = []
    for pat in ROOT_PATTERNS:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No ROOT files found in ./data/")
    return files


def candidate_branch(keys, options):
    keys = set(keys)
    for name in options:
        if name in keys:
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

    needed = [pxp, pyp, pzp, pep, pxm, pym, pzm, pem]
    if any(x is None for x in needed):
        print("\n[debug] branches containing MU:")
        for k in [x for x in keys if "MU" in x.upper()][:250]:
            print("   ", k)
        raise RuntimeError("Could not derive q2 from muon four-vectors.")

    arr = tree.arrays(needed, library="np")

    Ep = np.asarray(arr[pep], dtype=float)
    pxp_v = np.asarray(arr[pxp], dtype=float)
    pyp_v = np.asarray(arr[pyp], dtype=float)
    pzp_v = np.asarray(arr[pzp], dtype=float)

    Em = np.asarray(arr[pem], dtype=float)
    pxm_v = np.asarray(arr[pxm], dtype=float)
    pym_v = np.asarray(arr[pym], dtype=float)
    pzm_v = np.asarray(arr[pzm], dtype=float)

    E = Ep + Em
    px = pxp_v + pxm_v
    py = pyp_v + pym_v
    pz = pzp_v + pzm_v

    q2_mev2 = E * E - px * px - py * py - pz * pz
    q2_gev2 = q2_mev2 / 1.0e6

    return q2_gev2, {
        "mu_plus_PX": pxp,
        "mu_plus_PY": pyp,
        "mu_plus_PZ": pzp,
        "mu_plus_PE": pep,
        "mu_minus_PX": pxm,
        "mu_minus_PY": pym,
        "mu_minus_PZ": pzm,
        "mu_minus_PE": pem,
    }


def load_all_events(files):
    rows = []

    direct_q2_candidates = [
        "q2", "Q2", "q2_DTF", "Q2_DTF",
        "mumu_M2", "dimuon_M2",
        "Jpsi_M2", "J_psi_1S_M2", "J_psi_1S_MM2",
    ]

    b_mass_candidates = ["B0_M", "B0_MM", "B_M", "B_MM"]

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

    for path in files:
        print(f"[load] {path}")

        with uproot.open(path) as f:
            tree = None

            for key in f.keys():
                obj = f[key]
                if hasattr(obj, "keys") and hasattr(obj, "arrays"):
                    tree = obj
                    break

            if tree is None:
                print(f"[warn] no tree found in {path}")
                continue

            keys = list(tree.keys())

            q2_branch = candidate_branch(keys, direct_q2_candidates)
            b_branch = candidate_branch(keys, b_mass_candidates)
            kst_branch = candidate_branch(keys, kst_mass_candidates)

            if b_branch is None:
                raise RuntimeError(f"No B mass branch found in {path}")
            if kst_branch is None:
                raise RuntimeError(f"No K* mass branch found in {path}")

            if q2_branch is not None:
                arr = tree.arrays([q2_branch, b_branch, kst_branch], library="np")
                q2 = np.asarray(arr[q2_branch], dtype=float)
                finite = q2[np.isfinite(q2)]
                if len(finite) == 0:
                    continue
                if np.nanmedian(finite) > 1e4:
                    q2 = q2 / 1.0e6
                print(f"[q2] using branch {q2_branch}")
            else:
                q2, mu_used = derive_q2_from_muons(tree)
                arr = tree.arrays(list(mu_used.values()) + [b_branch, kst_branch], library="np")
                print("[q2] derived from muon four-vectors")

            bm = np.asarray(arr[b_branch], dtype=float)
            km = np.asarray(arr[kst_branch], dtype=float)

            mask = np.isfinite(q2) & np.isfinite(bm) & np.isfinite(km)
            mask &= (q2 >= Q2_MIN) & (q2 <= Q2_MAX)

            rows.append(pd.DataFrame({
                "q2": q2[mask],
                "B_M": bm[mask],
                "Kst_M": km[mask],
                "source_file": os.path.basename(path),
            }))

    if not rows:
        raise RuntimeError("No events loaded.")

    out = pd.concat(rows, ignore_index=True)
    print(f"[info] loaded q2-range events: {len(out):,}")
    return out


# ============================================================
# Selection / histogram / baseline
# ============================================================

def select_region(df, B_window, Kst_window, active_intervals):
    blo, bhi = B_window
    klo, khi = Kst_window

    mask = (df["B_M"] >= blo) & (df["B_M"] <= bhi)
    mask &= (df["Kst_M"] >= klo) & (df["Kst_M"] <= khi)
    mask &= in_active_intervals(df["q2"].values, active_intervals)

    return df.loc[mask].copy()


def make_histogram(q2, active_intervals, n_bins=N_BINS):
    ell = np.log(q2)

    ell_min = math.log(Q2_MIN)
    ell_max = math.log(Q2_MAX)

    counts, edges = np.histogram(ell, bins=n_bins, range=(ell_min, ell_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    q2_centers = np.exp(centers)

    active = in_active_intervals(q2_centers, active_intervals)

    return centers[active], counts[active].astype(float)


def kde_baseline(ell_centers, counts):
    repeated = np.repeat(ell_centers, np.maximum(counts.astype(int), 0))
    if len(repeated) < 100:
        raise RuntimeError("Too few points for KDE baseline.")

    kde = gaussian_kde(repeated)
    kde.set_bandwidth(kde.factor * KDE_BANDWIDTH_SCALE)

    dens = kde(ell_centers)
    dens = np.maximum(dens, 1e-12)

    baseline = dens / dens.sum() * counts.sum()
    return np.maximum(baseline, 1e-9)


# ============================================================
# Poisson fitting
# ============================================================

def basis_matrix(ell, ks):
    cols = [np.ones_like(ell)]

    cols.append(np.cos(K1_FIXED * ell))
    cols.append(np.sin(K1_FIXED * ell))

    for k in ks:
        cols.append(np.cos(float(k) * ell))
        cols.append(np.sin(float(k) * ell))

    return np.vstack(cols).T


def poisson_deviance(y, lam):
    y = np.asarray(y, dtype=float)
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-12)

    out = np.zeros_like(y)
    nz = y > 0
    out[nz] = y[nz] * np.log(y[nz] / lam[nz]) - (y[nz] - lam[nz])
    out[~nz] = lam[~nz]

    return 2.0 * float(np.sum(out))


def fit_poisson_bounded(counts, baseline, ell, ks):
    y = np.asarray(counts, dtype=float)
    B = np.maximum(np.asarray(baseline, dtype=float), 1e-12)
    X = basis_matrix(ell, ks)

    p = X.shape[1]
    beta0 = np.zeros(p)

    bounds = [(None, None)] + [(-A_MAX, A_MAX)] * (p - 1)

    def nll(beta):
        eta = np.clip(X @ beta, -20.0, 20.0)
        lam = B * np.exp(eta)
        return float(np.sum(lam - y * np.log(np.maximum(lam, 1e-12))))

    res = minimize(
        nll,
        beta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-8},
    )

    beta = res.x
    eta = np.clip(X @ beta, -20.0, 20.0)
    lam = B * np.exp(eta)
    dev = poisson_deviance(y, lam)

    A1 = math.sqrt(beta[1] ** 2 + beta[2] ** 2)

    A2 = None
    phi2 = None

    if len(ks) == 1:
        a2 = beta[3]
        b2 = beta[4]
        A2 = math.sqrt(a2 * a2 + b2 * b2)
        phi2 = math.atan2(-b2, a2)

    bound_active = any(abs(v) >= A_MAX - 1e-5 for v in beta[1:])

    return {
        "success": bool(res.success),
        "dev": float(dev),
        "nll": float(res.fun),
        "beta": beta,
        "lambda": lam,
        "A1": float(A1),
        "A2": None if A2 is None else float(A2),
        "phi2": None if phi2 is None else float(phi2),
        "bound_active": bool(bound_active),
        "n_params": int(p),
    }


# ============================================================
# Scan / well / triplet logic
# ============================================================

@dataclass
class WellRow:
    veto_label: str
    region: str
    active_intervals: str
    delta_ell_A: float
    well_rank: int
    k: float
    n_eff: float
    deltaD: float
    prominence: float
    A2: float
    phi2: float
    bound_active: bool
    nearest_integer_n: float
    distance_to_integer: float
    distance_to_n10: float
    distance_to_n15: float
    distance_to_n20: float


@dataclass
class TripletRow:
    veto_label: str
    region: str
    active_intervals: str
    delta_ell_A: float
    n1: float
    n2: float
    n3: float
    k1: float
    k2: float
    k3: float
    deltaD1: float
    deltaD2: float
    deltaD3: float
    Q_low: float
    Q_high: float
    Q_mean: float
    koide_error: float
    integer_error_10_15_20: float
    mean_deltaD: float
    score: float


def scan_continuous_k(ell, counts, baseline, delta_ell):
    base_fit = fit_poisson_bounded(counts, baseline, ell, ks=[])
    D_base = base_fit["dev"]

    k_grid = np.linspace(K_SCAN_MIN, K_SCAN_MAX, N_K_SCAN)

    rows = []
    for idx, k in enumerate(k_grid):
        fit = fit_poisson_bounded(counts, baseline, ell, ks=[float(k)])
        dD = D_base - fit["dev"]

        rows.append({
            "k": float(k),
            "n_eff": n_from_k(k, delta_ell),
            "deltaD": float(dD),
            "A2": float(fit["A2"]) if fit["A2"] is not None else float("nan"),
            "phi2": float(fit["phi2"]) if fit["phi2"] is not None else float("nan"),
            "bound_active": bool(fit["bound_active"]),
            "success": bool(fit["success"]),
        })

        if (idx + 1) % 300 == 0:
            print(f"  scanned {idx+1}/{N_K_SCAN}")

    return pd.DataFrame(rows), base_fit


def find_wells(scan_df):
    y = scan_df["deltaD"].values
    k_grid = scan_df["k"].values

    dk = float(np.median(np.diff(k_grid)))
    min_distance_bins = max(1, int(round(MIN_PEAK_DISTANCE_K / dk)))

    peaks, props = find_peaks(
        y,
        prominence=MIN_PEAK_PROMINENCE,
        distance=min_distance_bins,
    )

    if len(peaks) == 0:
        return []

    prominences = props.get("prominences", np.zeros(len(peaks)))

    order = sorted(range(len(peaks)), key=lambda i: y[peaks[i]], reverse=True)

    wells = []
    for rank, oi in enumerate(order, start=1):
        pidx = int(peaks[oi])
        row = scan_df.iloc[pidx]
        n_eff = float(row["n_eff"])
        nearest_int = round(n_eff)

        wells.append({
            "well_rank": int(rank),
            "k": float(row["k"]),
            "n_eff": n_eff,
            "deltaD": float(row["deltaD"]),
            "prominence": float(prominences[oi]),
            "A2": float(row["A2"]),
            "phi2": float(row["phi2"]),
            "bound_active": bool(row["bound_active"]),
            "nearest_integer_n": float(nearest_int),
            "distance_to_integer": float(abs(n_eff - nearest_int)),
            "distance_to_n10": float(abs(n_eff - 10.0)),
            "distance_to_n15": float(abs(n_eff - 15.0)),
            "distance_to_n20": float(abs(n_eff - 20.0)),
        })

    return wells


def triplets_from_wells(wells, veto_label, region, active_intervals, delta_ell):
    if len(wells) < 3:
        return []

    candidates = sorted(wells[:TOP_WELLS_FOR_TRIPLETS], key=lambda w: w["n_eff"])
    rows = []

    for i, j, k in combinations(range(len(candidates)), 3):
        w1 = candidates[i]
        w2 = candidates[j]
        w3 = candidates[k]

        n1, n2, n3 = w1["n_eff"], w2["n_eff"], w3["n_eff"]
        if n2 <= 0:
            continue

        Q_low = n1 / n2
        Q_high = n3 / (2.0 * n2)
        Q_mean = 0.5 * (Q_low + Q_high)

        koide_error = math.sqrt(
            (Q_low - KOIDE_Q) ** 2 +
            (Q_high - KOIDE_Q) ** 2
        )

        integer_error = math.sqrt(
            (n1 - 10.0) ** 2 +
            (n2 - 15.0) ** 2 +
            (n3 - 20.0) ** 2
        )

        mean_deltaD = (w1["deltaD"] + w2["deltaD"] + w3["deltaD"]) / 3.0

        score = mean_deltaD / (
            1.0 +
            25.0 * koide_error +
            0.25 * integer_error
        )

        rows.append(TripletRow(
            veto_label=veto_label,
            region=region,
            active_intervals=json.dumps(active_intervals),
            delta_ell_A=float(delta_ell),
            n1=float(n1),
            n2=float(n2),
            n3=float(n3),
            k1=float(w1["k"]),
            k2=float(w2["k"]),
            k3=float(w3["k"]),
            deltaD1=float(w1["deltaD"]),
            deltaD2=float(w2["deltaD"]),
            deltaD3=float(w3["deltaD"]),
            Q_low=float(Q_low),
            Q_high=float(Q_high),
            Q_mean=float(Q_mean),
            koide_error=float(koide_error),
            integer_error_10_15_20=float(integer_error),
            mean_deltaD=float(mean_deltaD),
            score=float(score),
        ))

    rows.sort(
        key=lambda r: (
            r.koide_error,
            r.integer_error_10_15_20,
            -r.score,
        )
    )

    return rows


# ============================================================
# Summary logic
# ============================================================

def summarize_covariance(triplets_df):
    rows = []

    for region, rg in triplets_df.groupby("region"):
        for veto_label, vg in rg.groupby("veto_label"):
            best = vg.sort_values(
                ["koide_error", "integer_error_10_15_20", "score"],
                ascending=[True, True, False],
            ).iloc[0]

            rows.append({
                "region": region,
                "veto_label": veto_label,
                "delta_ell_A": float(best["delta_ell_A"]),
                "best_n1": float(best["n1"]),
                "best_n2": float(best["n2"]),
                "best_n3": float(best["n3"]),
                "best_k1": float(best["k1"]),
                "best_k2": float(best["k2"]),
                "best_k3": float(best["k3"]),
                "Q_low": float(best["Q_low"]),
                "Q_high": float(best["Q_high"]),
                "Q_mean": float(best["Q_mean"]),
                "koide_error": float(best["koide_error"]),
                "integer_error_10_15_20": float(best["integer_error_10_15_20"]),
                "score": float(best["score"]),
            })

    summary = pd.DataFrame(rows)

    stability_rows = []

    for region, g in summary.groupby("region"):
        for col in ["best_n1", "best_n2", "best_n3", "Q_mean", "koide_error"]:
            vals = g[col].values.astype(float)
            stability_rows.append({
                "region": region,
                "metric": col,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "range": float(np.max(vals) - np.min(vals)),
                "rel_range": float((np.max(vals) - np.min(vals)) / max(abs(np.mean(vals)), 1e-12)),
            })

    stability = pd.DataFrame(stability_rows)

    return summary, stability


# ============================================================
# Main
# ============================================================

def run():
    print("=" * 100)
    print("VETO-WINDOW COVARIANCE / ACTIVE-DOMAIN INVARIANCE TEST")
    print("=" * 100)
    print(f"[config] K scan: [{K_SCAN_MIN}, {K_SCAN_MAX}], N={N_K_SCAN}")
    print(f"[config] A_MAX={A_MAX}")
    print(f"[config] Veto schemes={len(VETO_SCHEMES)}")
    print("=" * 100)

    files = find_root_files()
    df = load_all_events(files)

    all_wells = []
    all_triplets = []
    scan_summary = []

    for veto in VETO_SCHEMES:
        veto_label = veto["label"]
        active_intervals = active_intervals_from_veto(veto["jpsi"], veto["psi2s"])
        delta_ell = active_delta_ell(active_intervals)

        print("\n" + "#" * 100)
        print(f"[veto] {veto_label}")
        print(f"  jpsi={veto['jpsi']}")
        print(f"  psi2s={veto['psi2s']}")
        print(f"  active_intervals={active_intervals}")
        print(f"  delta_ell_A={delta_ell:.10f}")
        print(f"  k(10,15,20)=({k_from_n(10, delta_ell):.4f}, "
              f"{k_from_n(15, delta_ell):.4f}, "
              f"{k_from_n(20, delta_ell):.4f})")
        print("#" * 100)

        for region_cfg in REGIONS:
            region_name = region_cfg["region"]
            sub = select_region(df, region_cfg["B_window"], region_cfg["Kst_window"], active_intervals)

            print("\n" + "=" * 100)
            print(f"[region] {region_name}")
            print(f"  N_active={len(sub):,}")
            print("=" * 100)

            if len(sub) < 500:
                print("[skip] too few events")
                continue

            ell, counts = make_histogram(sub["q2"].values, active_intervals)

            if counts.sum() < 500:
                print("[skip] too few histogram counts")
                continue

            baseline = kde_baseline(ell, counts)

            scan_df, base_fit = scan_continuous_k(ell, counts, baseline, delta_ell)
            wells = find_wells(scan_df)

            for w in wells:
                all_wells.append(WellRow(
                    veto_label=veto_label,
                    region=region_name,
                    active_intervals=json.dumps(active_intervals),
                    delta_ell_A=float(delta_ell),
                    **w,
                ))

            triplets = triplets_from_wells(
                wells=wells,
                veto_label=veto_label,
                region=region_name,
                active_intervals=active_intervals,
                delta_ell=delta_ell,
            )
            all_triplets.extend(triplets)

            if wells:
                top_well = wells[0]
                print(
                    f"[top well] k={top_well['k']:.4f}, "
                    f"n={top_well['n_eff']:.4f}, "
                    f"DeltaD={top_well['deltaD']:.4f}"
                )

            if triplets:
                best = triplets[0]
                print(
                    f"[best triplet] n=({best.n1:.4f}, {best.n2:.4f}, {best.n3:.4f}), "
                    f"k=({best.k1:.4f}, {best.k2:.4f}, {best.k3:.4f}), "
                    f"Qmean={best.Q_mean:.6f}, "
                    f"epsK={best.koide_error:.6f}, "
                    f"epsInt={best.integer_error_10_15_20:.6f}"
                )

            scan_summary.append({
                "veto_label": veto_label,
                "region": region_name,
                "active_intervals": json.dumps(active_intervals),
                "delta_ell_A": float(delta_ell),
                "N_active": int(len(sub)),
                "hist_counts": int(counts.sum()),
                "D_base": float(base_fit["dev"]),
                "A1": float(base_fit["A1"]),
                "n_wells": int(len(wells)),
                "n_triplets": int(len(triplets)),
            })

    wells_df = pd.DataFrame([asdict(w) for w in all_wells])
    triplets_df = pd.DataFrame([asdict(t) for t in all_triplets])
    scan_summary_df = pd.DataFrame(scan_summary)

    wells_csv = os.path.join(OUTDIR, "veto_covariance_wells.csv")
    triplets_csv = os.path.join(OUTDIR, "veto_covariance_triplets.csv")
    scan_summary_csv = os.path.join(OUTDIR, "veto_covariance_scan_summary.csv")
    best_summary_csv = os.path.join(OUTDIR, "veto_covariance_best_triplets.csv")
    stability_csv = os.path.join(OUTDIR, "veto_covariance_stability.csv")
    summary_json = os.path.join(OUTDIR, "veto_covariance_summary.json")

    wells_df.to_csv(wells_csv, index=False)
    triplets_df.to_csv(triplets_csv, index=False)
    scan_summary_df.to_csv(scan_summary_csv, index=False)

    if not triplets_df.empty:
        best_summary_df, stability_df = summarize_covariance(triplets_df)
    else:
        best_summary_df = pd.DataFrame()
        stability_df = pd.DataFrame()

    best_summary_df.to_csv(best_summary_csv, index=False)
    stability_df.to_csv(stability_csv, index=False)

    payload = {
        "test": "veto_window_covariance_active_domain_invariance",
        "purpose": "Test whether well/triplet structure is stable in n-space under veto-window changes.",
        "Q2_range": [Q2_MIN, Q2_MAX],
        "K_SCAN": [K_SCAN_MIN, K_SCAN_MAX, N_K_SCAN],
        "A_MAX": A_MAX,
        "K1_FIXED": K1_FIXED,
        "veto_schemes": [
            {
                **v,
                "active_intervals": active_intervals_from_veto(v["jpsi"], v["psi2s"]),
                "delta_ell_A": active_delta_ell(active_intervals_from_veto(v["jpsi"], v["psi2s"])),
            }
            for v in VETO_SCHEMES
        ],
        "files": {
            "wells_csv": wells_csv,
            "triplets_csv": triplets_csv,
            "scan_summary_csv": scan_summary_csv,
            "best_summary_csv": best_summary_csv,
            "stability_csv": stability_csv,
            "summary_json": summary_json,
        },
        "interpretation": {
            "stable_n_unstable_k": "WCT-supporting active-domain covariance.",
            "stable_k_unstable_n": "raw-frequency artifact or fixed detector/window frequency.",
            "unstable_both": "weakens WCT-specific interpretation.",
            "stable_Q_not_integer": "ratio geometry survives, but integer winding does not.",
        },
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 100)
    print("VETO COVARIANCE SUMMARY")
    print("=" * 100)

    print("\n[best triplets]")
    if not best_summary_df.empty:
        print(best_summary_df.to_string(index=False))
    else:
        print("No triplets found.")

    print("\n[stability]")
    if not stability_df.empty:
        print(stability_df.to_string(index=False))
    else:
        print("No stability table.")

    print("\nSaved:")
    print(f"  {wells_csv}")
    print(f"  {triplets_csv}")
    print(f"  {scan_summary_csv}")
    print(f"  {best_summary_csv}")
    print(f"  {stability_csv}")
    print(f"  {summary_json}")


if __name__ == "__main__":
    run()
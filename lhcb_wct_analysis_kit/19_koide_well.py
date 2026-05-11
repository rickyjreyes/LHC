#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WCT Well-First Koide Test
-------------------------

Purpose:
    Test whether Koide geometry is already present in the raw continuous
    log-frequency wells before imposing any Koide comb.

This script does NOT fit a Koide comb.

Pipeline:
    1. Load B0 -> K*0 mu+ mu- candidate ROOT files.
    2. Derive q^2 from muon four-vectors if no q2 branch exists.
    3. Select signal and B-sideband regions.
    4. Build q^2 histogram in ell = ln(q^2).
    5. Fit base model:
            lambda_i = B_i exp(C + a1 cos(k1 ell_i) + b1 sin(k1 ell_i))
    6. Scan one extra mode continuously:
            lambda_i = base * exp(a2 cos(k ell_i) + b2 sin(k ell_i))
    7. Find raw wells/local maxima in DeltaD(k).
    8. Convert wells to active-domain windings:
            n(k) = k DeltaEll_A / (2 pi)
    9. Test whether raw well triplets imply Koide:
            Q_low  = n1 / n2
            Q_high = n3 / (2 n2)

For Koide sideband geometry:
    (n1,n2,n3) = (10,15,20)
    Q_low  = 10/15 = 2/3
    Q_high = 20/(2*15) = 2/3

Outputs:
    outputs_wct_well_first_koide/
        well_first_scan_curve.csv
        well_first_wells.csv
        well_first_triplets.csv
        well_first_summary.json
"""

import os
import json
import math
import glob
import warnings
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

try:
    import cupy as cp
    USE_CUPY = True
except Exception:
    cp = np
    USE_CUPY = False

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
# Configuration
# ============================================================

OUTDIR = "outputs_wct_well_first_koide"
os.makedirs(OUTDIR, exist_ok=True)

ROOT_PATTERNS = [
    "data/*.dvntuple.root",
    "data/*.root",
]

Q2_MIN = 0.1
Q2_MAX = 19.0

ACTIVE_INTERVALS = [
    (0.1, 8.0),
    (11.0, 12.5),
    (14.5, 19.0),
]

B_SIGNAL = (5230.0, 5330.0)
B_LOW_SB = (5000.0, 5180.0)
B_HIGH_SB = (5380.0, 5600.0)
KST_SIGNAL = (795.9, 995.9)

REGIONS = [
    {
        "region": "signal_B_signal_Kst",
        "B_window": B_SIGNAL,
        "Kst_window": KST_SIGNAL,
    },
    {
        "region": "B_low_sideband_Kst_signal",
        "B_window": B_LOW_SB,
        "Kst_window": KST_SIGNAL,
    },
    {
        "region": "B_high_sideband_Kst_signal",
        "B_window": B_HIGH_SB,
        "Kst_window": KST_SIGNAL,
    },
]

K1_FIXED = 7.61054

K_SCAN_MIN = 6.0
K_SCAN_MAX = 32.0
N_K_SCAN = 1301

N_BINS = 240
KDE_BANDWIDTH_SCALE = 1.00

A_MAX = 0.10

MIN_PEAK_PROMINENCE = 1.0
MIN_PEAK_DISTANCE_K = 0.75

MAX_WELLS_FOR_TRIPLETS = 12

KOIDE_Q = 2.0 / 3.0

Q_TARGETS = {
    "koide_2over3": 2.0 / 3.0,
    "folded_4over9": 4.0 / 9.0,
    "empirical_0p65": 0.65,
    "quark_like_0p63026": 0.63026,
    "half": 0.5,
}


# ============================================================
# Active-domain helpers
# ============================================================

def active_delta_ell(intervals):
    return sum(math.log(hi / lo) for lo, hi in intervals)


DELTA_ELL_ACTIVE = active_delta_ell(ACTIVE_INTERVALS)


def n_from_k(k):
    return k * DELTA_ELL_ACTIVE / (2.0 * math.pi)


def k_from_n(n):
    return 2.0 * math.pi * n / DELTA_ELL_ACTIVE


def in_active_intervals(q2):
    mask = np.zeros_like(q2, dtype=bool)
    for lo, hi in ACTIVE_INTERVALS:
        mask |= (q2 >= lo) & (q2 <= hi)
    return mask


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


def candidate_branch(tree_or_keys, options):
    if hasattr(tree_or_keys, "keys"):
        keys = set(tree_or_keys.keys())
    else:
        keys = set(tree_or_keys)

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
        print("\n[debug] missing muon branches:")
        for m in missing:
            print("   ", m)

        print("\n[debug] branches containing MU:")
        mu_keys = [k for k in keys if "MU" in k.upper()]
        for k in mu_keys[:250]:
            print("   ", k)

        raise RuntimeError("Could not derive q2 from muons.")

    branches = [pxp, pyp, pzp, pep, pxm, pym, pzm, pem]
    arr = tree.arrays(branches, library="np")

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

    used = {
        "mu_plus_PX": pxp,
        "mu_plus_PY": pyp,
        "mu_plus_PZ": pzp,
        "mu_plus_PE": pep,
        "mu_minus_PX": pxm,
        "mu_minus_PY": pym,
        "mu_minus_PZ": pzm,
        "mu_minus_PE": pem,
    }

    return q2_gev2, used


def load_all_events(files):
    rows = []

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
                branches = [q2_branch, b_branch, kst_branch]
                arr = tree.arrays(branches, library="np")

                q2 = np.asarray(arr[q2_branch], dtype=float)
                finite = q2[np.isfinite(q2)]

                if len(finite) == 0:
                    continue

                if np.nanmedian(finite) > 1e4:
                    q2 = q2 / 1.0e6

                print(f"[q2] using branch {q2_branch}")

            else:
                q2, mu_used = derive_q2_from_muons(tree)
                branches = list(mu_used.values()) + [b_branch, kst_branch]
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
# Selection / histogram / baseline
# ============================================================

def select_region(df, B_window, Kst_window):
    blo, bhi = B_window
    klo, khi = Kst_window

    mask = (df["B_M"] >= blo) & (df["B_M"] <= bhi)
    mask &= (df["Kst_M"] >= klo) & (df["Kst_M"] <= khi)
    mask &= in_active_intervals(df["q2"].values)

    return df.loc[mask].copy()


def make_histogram(q2, n_bins=N_BINS):
    ell = np.log(q2)

    ell_min = math.log(Q2_MIN)
    ell_max = math.log(Q2_MAX)

    counts, edges = np.histogram(ell, bins=n_bins, range=(ell_min, ell_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    q2_centers = np.exp(centers)

    active = in_active_intervals(q2_centers)

    return centers[active], counts[active].astype(float)


def kde_baseline(ell_centers, counts, bw_scale=KDE_BANDWIDTH_SCALE):
    repeated = np.repeat(ell_centers, np.maximum(counts.astype(int), 0))
    if len(repeated) < 100:
        raise RuntimeError("Too few points for KDE baseline.")

    kde = gaussian_kde(repeated)
    kde.set_bandwidth(kde.factor * bw_scale)

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
        cols.append(np.cos(k * ell))
        cols.append(np.sin(k * ell))

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

    A1 = math.sqrt(beta[1] * beta[1] + beta[2] * beta[2])

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
# Well detection and Koide triplet test
# ============================================================

@dataclass
class ScanRow:
    region: str
    k: float
    n_eff: float
    deltaD: float
    A2: float
    phi2: float
    bound_active: bool
    success: bool


@dataclass
class WellRow:
    region: str
    well_rank: int
    peak_index: int
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
    region: str
    k1: float
    k2: float
    k3: float
    n1: float
    n2: float
    n3: float
    deltaD1: float
    deltaD2: float
    deltaD3: float
    Q_low: float
    Q_high: float
    Q_mean: float
    koide_error: float
    target_label_best: str
    target_Q_best: float
    target_error_best: float
    integer_error_10_15_20: float
    score: float


def scan_continuous_k(region_name, ell, counts, baseline):
    base_fit = fit_poisson_bounded(counts, baseline, ell, ks=[])
    D_base = base_fit["dev"]

    print(f"[base] region={region_name} D_base={D_base:.6f} A1={base_fit['A1']:.6f}")

    k_grid = np.linspace(K_SCAN_MIN, K_SCAN_MAX, N_K_SCAN)
    rows = []

    for idx, k in enumerate(k_grid):
        fit = fit_poisson_bounded(counts, baseline, ell, ks=[float(k)])
        dD = D_base - fit["dev"]

        rows.append(ScanRow(
            region=region_name,
            k=float(k),
            n_eff=float(n_from_k(k)),
            deltaD=float(dD),
            A2=float(fit["A2"]) if fit["A2"] is not None else float("nan"),
            phi2=float(fit["phi2"]) if fit["phi2"] is not None else float("nan"),
            bound_active=bool(fit["bound_active"]),
            success=bool(fit["success"]),
        ))

        if (idx + 1) % 200 == 0:
            print(f"  scanned {idx+1}/{N_K_SCAN}")

    return rows


def find_wells(region_name, scan_rows):
    df = pd.DataFrame([asdict(r) for r in scan_rows])
    y = df["deltaD"].values
    k_grid = df["k"].values

    dk = float(np.median(np.diff(k_grid)))
    min_distance_bins = max(1, int(round(MIN_PEAK_DISTANCE_K / dk)))

    peaks, props = find_peaks(
        y,
        prominence=MIN_PEAK_PROMINENCE,
        distance=min_distance_bins,
    )

    well_rows = []
    if len(peaks) == 0:
        return well_rows

    prominences = props.get("prominences", np.zeros(len(peaks)))

    order = sorted(range(len(peaks)), key=lambda i: y[peaks[i]], reverse=True)

    for rank, oi in enumerate(order, start=1):
        pidx = int(peaks[oi])
        row = df.iloc[pidx]
        n_eff = float(row["n_eff"])
        nearest_int = round(n_eff)

        well_rows.append(WellRow(
            region=region_name,
            well_rank=int(rank),
            peak_index=int(pidx),
            k=float(row["k"]),
            n_eff=n_eff,
            deltaD=float(row["deltaD"]),
            prominence=float(prominences[oi]),
            A2=float(row["A2"]),
            phi2=float(row["phi2"]),
            bound_active=bool(row["bound_active"]),
            nearest_integer_n=float(nearest_int),
            distance_to_integer=float(abs(n_eff - nearest_int)),
            distance_to_n10=float(abs(n_eff - 10.0)),
            distance_to_n15=float(abs(n_eff - 15.0)),
            distance_to_n20=float(abs(n_eff - 20.0)),
        ))

    return well_rows


def triplets_from_wells(region_name, well_rows):
    if len(well_rows) < 3:
        return []

    candidates = sorted(well_rows[:MAX_WELLS_FOR_TRIPLETS], key=lambda w: w.n_eff)
    triplets = []

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            for k in range(j + 1, len(candidates)):
                w1 = candidates[i]
                w2 = candidates[j]
                w3 = candidates[k]

                n1, n2, n3 = w1.n_eff, w2.n_eff, w3.n_eff

                if n2 <= 0:
                    continue

                Q_low = n1 / n2
                Q_high = n3 / (2.0 * n2)
                Q_mean = 0.5 * (Q_low + Q_high)

                koide_error = math.sqrt(
                    (Q_low - KOIDE_Q) ** 2 +
                    (Q_high - KOIDE_Q) ** 2
                )

                best_label = None
                best_Q = None
                best_err = float("inf")

                for label, qtar in Q_TARGETS.items():
                    err = math.sqrt((Q_low - qtar) ** 2 + (Q_high - qtar) ** 2)
                    if err < best_err:
                        best_err = err
                        best_label = label
                        best_Q = qtar

                integer_error = math.sqrt(
                    (n1 - 10.0) ** 2 +
                    (n2 - 15.0) ** 2 +
                    (n3 - 20.0) ** 2
                )

                mean_deltaD = (w1.deltaD + w2.deltaD + w3.deltaD) / 3.0

                score = mean_deltaD / (
                    1.0 +
                    25.0 * koide_error +
                    0.25 * integer_error
                )

                triplets.append(TripletRow(
                    region=region_name,
                    k1=float(w1.k),
                    k2=float(w2.k),
                    k3=float(w3.k),
                    n1=float(n1),
                    n2=float(n2),
                    n3=float(n3),
                    deltaD1=float(w1.deltaD),
                    deltaD2=float(w2.deltaD),
                    deltaD3=float(w3.deltaD),
                    Q_low=float(Q_low),
                    Q_high=float(Q_high),
                    Q_mean=float(Q_mean),
                    koide_error=float(koide_error),
                    target_label_best=str(best_label),
                    target_Q_best=float(best_Q),
                    target_error_best=float(best_err),
                    integer_error_10_15_20=float(integer_error),
                    score=float(score),
                ))

    triplets.sort(
        key=lambda r: (
            r.koide_error,
            r.integer_error_10_15_20,
            -r.score
        )
    )

    return triplets


# ============================================================
# Main
# ============================================================

def run():
    print("=" * 100)
    print("WCT WELL-FIRST KOIDE TEST")
    print("=" * 100)
    print(f"[gpu] CuPy available: {USE_CUPY}")
    print(f"[config] active intervals: {ACTIVE_INTERVALS}")
    print(f"[config] Delta ell active = {DELTA_ELL_ACTIVE:.10f}")
    print(f"[config] k scan: [{K_SCAN_MIN}, {K_SCAN_MAX}], N={N_K_SCAN}")
    print(
        f"[config] k(10),k(15),k(20)="
        f"({k_from_n(10):.6f}, {k_from_n(15):.6f}, {k_from_n(20):.6f})"
    )
    print("=" * 100)

    files = find_root_files()
    df = load_all_events(files)

    all_scan_rows = []
    all_well_rows = []
    all_triplet_rows = []

    for region_cfg in REGIONS:
        region_name = region_cfg["region"]
        sub = select_region(df, region_cfg["B_window"], region_cfg["Kst_window"])

        print("\n" + "=" * 100)
        print(f"[region] {region_name}")
        print(f"  B_window={region_cfg['B_window']}")
        print(f"  Kst_window={region_cfg['Kst_window']}")
        print(f"  N_active={len(sub):,}")
        print("=" * 100)

        if len(sub) < 500:
            print("[skip] too few events")
            continue

        ell, counts = make_histogram(sub["q2"].values, N_BINS)

        if counts.sum() < 500:
            print("[skip] too few histogram counts")
            continue

        baseline = kde_baseline(ell, counts, KDE_BANDWIDTH_SCALE)

        scan_rows = scan_continuous_k(region_name, ell, counts, baseline)
        wells = find_wells(region_name, scan_rows)
        triplets = triplets_from_wells(region_name, wells)

        all_scan_rows.extend(scan_rows)
        all_well_rows.extend(wells)
        all_triplet_rows.extend(triplets)

        print("\n[top raw wells]")
        if wells:
            well_df = pd.DataFrame([asdict(w) for w in wells[:10]])
            print(well_df[[
                "well_rank",
                "k",
                "n_eff",
                "deltaD",
                "prominence",
                "nearest_integer_n",
                "distance_to_integer",
                "distance_to_n10",
                "distance_to_n15",
                "distance_to_n20",
                "A2",
                "phi2",
                "bound_active",
            ]].to_string(index=False))
        else:
            print("  no wells found")

        print("\n[best Koide-implying triplets]")
        if triplets:
            trip_df = pd.DataFrame([asdict(t) for t in triplets[:10]])
            print(trip_df[[
                "n1",
                "n2",
                "n3",
                "k1",
                "k2",
                "k3",
                "Q_low",
                "Q_high",
                "Q_mean",
                "koide_error",
                "target_label_best",
                "target_error_best",
                "integer_error_10_15_20",
                "score",
            ]].to_string(index=False))
        else:
            print("  no triplets found")

    scan_df = pd.DataFrame([asdict(r) for r in all_scan_rows])
    wells_df = pd.DataFrame([asdict(w) for w in all_well_rows])
    triplets_df = pd.DataFrame([asdict(t) for t in all_triplet_rows])

    scan_csv = os.path.join(OUTDIR, "well_first_scan_curve.csv")
    wells_csv = os.path.join(OUTDIR, "well_first_wells.csv")
    triplets_csv = os.path.join(OUTDIR, "well_first_triplets.csv")
    summary_json = os.path.join(OUTDIR, "well_first_summary.json")

    scan_df.to_csv(scan_csv, index=False)
    wells_df.to_csv(wells_csv, index=False)
    triplets_df.to_csv(triplets_csv, index=False)

    summary = {
        "test": "wct_well_first_koide_test",
        "purpose": (
            "Find raw continuous scan wells first, then test whether their "
            "geometry implies Koide before comb shaping."
        ),
        "active_intervals_q2": ACTIVE_INTERVALS,
        "delta_ell_active": DELTA_ELL_ACTIVE,
        "k1_fixed": K1_FIXED,
        "k_scan_min": K_SCAN_MIN,
        "k_scan_max": K_SCAN_MAX,
        "n_k_scan": N_K_SCAN,
        "k_expected_n10": k_from_n(10.0),
        "k_expected_n15": k_from_n(15.0),
        "k_expected_n20": k_from_n(20.0),
        "koide_Q": KOIDE_Q,
        "regions": REGIONS,
        "files": {
            "scan_curve_csv": scan_csv,
            "wells_csv": wells_csv,
            "triplets_csv": triplets_csv,
            "summary_json": summary_json,
        },
        "best_wells_by_region": {},
        "best_triplets_by_region": {},
        "interpretation": {
            "koide_found_before_shaping": (
                "Raw wells produce Q_low and Q_high near 2/3 with small "
                "koide_error."
            ),
            "koide_not_raw": (
                "Raw wells do not imply Q=2/3; comb fit may be imposing "
                "Koide geometry."
            ),
            "background_check": (
                "Compare signal and B-sideband regions. If wells/Koide are "
                "stronger in sidebands, source is not isolated signal yield."
            ),
        },
    }

    if not wells_df.empty:
        for region, g in wells_df.groupby("region"):
            gg = g.sort_values("deltaD", ascending=False).head(10)
            summary["best_wells_by_region"][region] = gg.to_dict(orient="records")

    if not triplets_df.empty:
        for region, g in triplets_df.groupby("region"):
            gg = g.sort_values(
                ["koide_error", "integer_error_10_15_20", "score"],
                ascending=[True, True, False],
            ).head(10)
            summary["best_triplets_by_region"][region] = gg.to_dict(orient="records")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 100)
    print("WELL-FIRST KOIDE SUMMARY")
    print("=" * 100)
    print(f"Saved: {scan_csv}")
    print(f"Saved: {wells_csv}")
    print(f"Saved: {triplets_csv}")
    print(f"Saved: {summary_json}")

    if not triplets_df.empty:
        print("\n[global best raw-well Koide triplets]")
        best = triplets_df.sort_values(
            ["koide_error", "integer_error_10_15_20", "score"],
            ascending=[True, True, False],
        ).head(15)

        print(best[[
            "region",
            "n1",
            "n2",
            "n3",
            "Q_low",
            "Q_high",
            "Q_mean",
            "koide_error",
            "target_label_best",
            "integer_error_10_15_20",
            "score",
        ]].to_string(index=False))


if __name__ == "__main__":
    run()
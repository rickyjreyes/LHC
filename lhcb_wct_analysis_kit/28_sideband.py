#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
28_sideband_subtracted_residual_test.py

Sideband-Subtracted Residual Test for the LHCb Open-Data
B0 -> K*0 mu+ mu- Candidate Spectrum

Purpose
-------
This is the cleanest next control for the current LHCb/WCT diagnostic because
it keeps the same q^2 continuum, the same active log-domain support, and the
same k <-> n mapping as the main paper.

It asks whether the high-k / integer-winding / Koide-comb structure survives
after subtracting B-mass sideband shape from the B-signal-window spectrum.

Model
-----
For each log-q^2 bin:

    R_i = N_sig,i - alpha * N_side,i

with

    alpha = sum_i N_sig,i / sum_i N_side,i

and Poisson variance approximation:

    Var(R_i) = N_sig,i + alpha^2 N_side,i.

Since R_i may be negative, the script uses weighted least squares, not Poisson
likelihood, for the sideband-subtracted residual.

It compares:

    base model: constant + fixed k1 mode
    add-one model: base + one tested high-k mode
    comb model: base + three locked modes

Outputs
-------
outputs_sideband_subtracted/
    sideband_subtracted_bins.csv
    sideband_subtracted_scan.csv
    sideband_subtracted_wells.csv
    sideband_subtracted_integer_scan.csv
    sideband_subtracted_comb_tests.csv
    sideband_subtracted_summary.json

Run
---
    python 28_sideband_subtracted_residual_test.py

Optional:
    python 28_sideband_subtracted_residual_test.py --pattern "data/*.root"
    python 28_sideband_subtracted_residual_test.py --n-null 1000
"""

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass, asdict
from itertools import combinations

import numpy as np
import pandas as pd

try:
    import uproot
except Exception as exc:
    raise RuntimeError("Missing uproot. Install with: pip install uproot awkward") from exc

try:
    from scipy.signal import find_peaks
except Exception as exc:
    raise RuntimeError("Missing scipy. Install with: pip install scipy") from exc


# ============================================================
# Pre-registered analysis settings
# ============================================================

OUTDIR = "outputs_sideband_subtracted"
os.makedirs(OUTDIR, exist_ok=True)

DEFAULT_PATTERNS = [
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

K1_FIXED = 7.61054
K_REF = 19.5296

KOIDE_Q = 2.0 / 3.0
A_TARGET_OBSERVED = 1.22828743138222
A_TARGET_GEOM = math.sqrt(3.0 / 2.0)

N_BINS = 240
K_SCAN_MIN = 6.0
K_SCAN_MAX = 32.0
N_K_SCAN = 1301

INTEGER_N_MIN = 10
INTEGER_N_MAX = 22

MIN_PEAK_PROMINENCE = 0.5
MIN_PEAK_DISTANCE_K = 0.75
MAX_WELLS_FOR_TRIPLETS = 12

DEFAULT_N_NULL = 500
RNG_SEED = 271828

# Verdict thresholds. These are deliberately conservative diagnostic thresholds.
P_THRESH_STRONG = 0.01
P_THRESH_WEAK = 0.05
KOIDE_ERR_TOL = 0.025
INTEGER_10_15_20_ERR_TOL = 1.25
A_SCALE_TOL = 0.03


# ============================================================
# Utility functions
# ============================================================

def active_delta_ell(intervals):
    return sum(math.log(hi / lo) for lo, hi in intervals)


DELTA_ELL_ACTIVE = active_delta_ell(ACTIVE_INTERVALS)


def n_from_k(k):
    return k * DELTA_ELL_ACTIVE / (2.0 * math.pi)


def k_from_n(n):
    return 2.0 * math.pi * n / DELTA_ELL_ACTIVE


def in_active_intervals(q2):
    q2 = np.asarray(q2, dtype=float)
    mask = np.zeros_like(q2, dtype=bool)
    for lo, hi in ACTIVE_INTERVALS:
        mask |= (q2 >= lo) & (q2 <= hi)
    return mask


def find_files(pattern=None):
    if pattern:
        files = sorted(glob.glob(pattern))
    else:
        files = []
        for pat in DEFAULT_PATTERNS:
            files.extend(glob.glob(pat))
        files = sorted(set(files))

    if not files:
        raise FileNotFoundError(
            "No ROOT files found. Put B0 -> K*0 mu+mu- ROOT files under data/ "
            "or pass --pattern."
        )
    return files


def candidate_branch(keys, options):
    keys = list(keys)
    exact = set(keys)
    for name in options:
        if name in exact:
            return name

    lower_map = {k.lower(): k for k in keys}
    for name in options:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    for name in options:
        needle = name.lower()
        for key in keys:
            if needle in key.lower():
                return key

    return None


def find_tree(root_file):
    with uproot.open(root_file) as f:
        for key, obj in f.items(recursive=True):
            if hasattr(obj, "arrays") and "DecayTree" in key:
                return key
        for key, obj in f.items(recursive=True):
            if hasattr(obj, "arrays"):
                return key
    raise RuntimeError(f"No TTree found in {root_file}")


def find_particle_component(keys, particle_patterns, comp):
    comp_upper = comp.upper()

    exact = []
    for p in particle_patterns:
        exact.extend([
            f"{p}_{comp}",
            f"{p}{comp}",
            f"{p}.{comp}",
            f"{p}_{comp_upper}",
            f"{p}{comp_upper}",
        ])

    found = candidate_branch(keys, exact)
    if found:
        return found

    for key in keys:
        ku = key.upper()
        if not (ku.endswith("_" + comp_upper) or ku.endswith(comp_upper)):
            continue
        for p in particle_patterns:
            if p.upper() in ku:
                return key

    return None


def derive_q2_from_muons(tree, keys):
    plus_patterns = [
        "muplus", "mu_plus", "mup", "mu_p",
        "muplus0", "muplus_0", "MuPlus",
        "mup_0", "mu1", "muplus_1",
    ]

    minus_patterns = [
        "muminus", "mu_minus", "mum", "mu_m",
        "muminus0", "muminus_0", "MuMinus",
        "mum_0", "mu2", "muminus_1",
    ]

    branches = {
        "pxp": find_particle_component(keys, plus_patterns, "PX"),
        "pyp": find_particle_component(keys, plus_patterns, "PY"),
        "pzp": find_particle_component(keys, plus_patterns, "PZ"),
        "pep": find_particle_component(keys, plus_patterns, "PE"),
        "pxm": find_particle_component(keys, minus_patterns, "PX"),
        "pym": find_particle_component(keys, minus_patterns, "PY"),
        "pzm": find_particle_component(keys, minus_patterns, "PZ"),
        "pem": find_particle_component(keys, minus_patterns, "PE"),
    }

    missing = [k for k, v in branches.items() if v is None]
    if missing:
        print("[debug] missing muon components:", missing)
        print("[debug] mu-like branches:")
        for key in keys:
            if "mu" in key.lower():
                print("  ", key)
        raise RuntimeError("Could not derive q2 from muon four-vectors.")

    arr = tree.arrays(list(branches.values()), library="np")

    Ep = np.asarray(arr[branches["pep"]], dtype=float)
    pxp = np.asarray(arr[branches["pxp"]], dtype=float)
    pyp = np.asarray(arr[branches["pyp"]], dtype=float)
    pzp = np.asarray(arr[branches["pzp"]], dtype=float)

    Em = np.asarray(arr[branches["pem"]], dtype=float)
    pxm = np.asarray(arr[branches["pxm"]], dtype=float)
    pym = np.asarray(arr[branches["pym"]], dtype=float)
    pzm = np.asarray(arr[branches["pzm"]], dtype=float)

    E = Ep + Em
    px = pxp + pxm
    py = pyp + pym
    pz = pzp + pzm

    q2_mev2 = E * E - px * px - py * py - pz * pz
    return q2_mev2 / 1.0e6, branches


def load_all_events(files):
    q2_candidates = [
        "q2", "Q2", "q2_DTF", "Q2_DTF",
        "mumu_M2", "dimuon_M2",
        "Jpsi_M2", "J_psi_1S_M2", "J_psi_1S_MM2",
    ]

    b_mass_candidates = [
        "B0_M", "B0_MM", "B_M", "B_MM",
    ]

    kst_mass_candidates = [
        "Kst_892_0_M", "Kst_892_0_MM",
        "Kst_M", "Kst_MM",
        "Kstar_M", "Kstar_MM",
        "Kstar0_M", "Kstar0_MM",
    ]

    rows = []
    provenance = []

    for path in files:
        print(f"[load] {path}")
        tree_name = find_tree(path)
        with uproot.open(path) as f:
            tree = f[tree_name]
            keys = list(tree.keys())

            q2_branch = candidate_branch(keys, q2_candidates)
            b_branch = candidate_branch(keys, b_mass_candidates)
            kst_branch = candidate_branch(keys, kst_mass_candidates)

            if b_branch is None:
                raise RuntimeError(f"No B0 mass branch found in {path}")
            if kst_branch is None:
                raise RuntimeError(f"No K* mass branch found in {path}")

            if q2_branch:
                arr = tree.arrays([q2_branch, b_branch, kst_branch], library="np")
                q2 = np.asarray(arr[q2_branch], dtype=float)
                finite = q2[np.isfinite(q2)]
                if len(finite) and np.nanmedian(finite) > 1e4:
                    q2 = q2 / 1.0e6
                q2_source = q2_branch
                print(f"[q2] using branch {q2_branch}")
            else:
                q2, used_mu = derive_q2_from_muons(tree, keys)
                arr = tree.arrays([b_branch, kst_branch], library="np")
                q2_source = "derived_from_muon_four_vectors"
                print("[q2] derived from muon four-vectors")

            bm = np.asarray(arr[b_branch], dtype=float)
            km = np.asarray(arr[kst_branch], dtype=float)

            mask = np.isfinite(q2) & np.isfinite(bm) & np.isfinite(km)
            mask &= (q2 >= Q2_MIN) & (q2 <= Q2_MAX)

            sub = pd.DataFrame({
                "q2": q2[mask],
                "B_M": bm[mask],
                "Kst_M": km[mask],
                "source_file": os.path.basename(path),
            })
            rows.append(sub)

            provenance.append({
                "file": path,
                "tree": tree_name,
                "q2_source": q2_source,
                "B_mass_branch": b_branch,
                "Kst_mass_branch": kst_branch,
                "n_loaded_q2_range": int(len(sub)),
            })

    if not rows:
        raise RuntimeError("No rows loaded.")

    df = pd.concat(rows, ignore_index=True)
    print(f"[info] loaded q2-range events: {len(df):,}")
    return df, provenance


def select_region(df, b_window, kst_window):
    blo, bhi = b_window
    klo, khi = kst_window

    mask = (df["B_M"] >= blo) & (df["B_M"] <= bhi)
    mask &= (df["Kst_M"] >= klo) & (df["Kst_M"] <= khi)
    mask &= in_active_intervals(df["q2"].to_numpy())

    return df.loc[mask].copy()


def make_histogram(q2, n_bins=N_BINS):
    ell = np.log(np.asarray(q2, dtype=float))
    ell_min = math.log(Q2_MIN)
    ell_max = math.log(Q2_MAX)

    counts, edges = np.histogram(ell, bins=n_bins, range=(ell_min, ell_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    q2_centers = np.exp(centers)
    active = in_active_intervals(q2_centers)

    return centers[active], q2_centers[active], counts[active].astype(float)


def wls_fit(ell, y, var, ks_extra=None, include_k1=True):
    if ks_extra is None:
        ks_extra = []

    ell = np.asarray(ell, dtype=float)
    y = np.asarray(y, dtype=float)
    var = np.maximum(np.asarray(var, dtype=float), 1.0)

    cols = [np.ones_like(ell)]

    if include_k1:
        cols.append(np.cos(K1_FIXED * ell))
        cols.append(np.sin(K1_FIXED * ell))

    for k in ks_extra:
        cols.append(np.cos(k * ell))
        cols.append(np.sin(k * ell))

    X = np.vstack(cols).T
    w = 1.0 / np.sqrt(var)

    Xw = X * w[:, None]
    yw = y * w

    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    pred = X @ beta
    chi2 = float(np.sum((y - pred) ** 2 / var))

    amps = {}
    idx = 1
    if include_k1:
        amps["A_k1"] = float(math.sqrt(beta[idx] ** 2 + beta[idx + 1] ** 2))
        idx += 2

    for k in ks_extra:
        amp = float(math.sqrt(beta[idx] ** 2 + beta[idx + 1] ** 2))
        phase = float(math.atan2(-beta[idx + 1], beta[idx]))
        amps[f"A_k_{k:.6f}"] = amp
        amps[f"phi_k_{k:.6f}"] = phase
        idx += 2

    return {
        "chi2": chi2,
        "beta": beta,
        "pred": pred,
        "amps": amps,
        "ndof": int(len(y) - len(beta)),
    }


@dataclass
class ScanRow:
    k: float
    n_eff: float
    delta_chi2: float
    amp: float
    phase: float


@dataclass
class WellRow:
    well_rank: int
    peak_index: int
    k: float
    n_eff: float
    delta_chi2: float
    prominence: float
    nearest_integer_n: float
    distance_to_integer: float
    distance_to_n10: float
    distance_to_n15: float
    distance_to_n20: float


@dataclass
class TripletRow:
    k1: float
    k2: float
    k3: float
    n1: float
    n2: float
    n3: float
    delta1: float
    delta2: float
    delta3: float
    Q_low: float
    Q_high: float
    Q_mean: float
    koide_error: float
    integer_error_10_15_20: float
    score: float


def scan_one_mode(ell, y, var, k_grid):
    base = wls_fit(ell, y, var, ks_extra=[], include_k1=True)
    chi2_base = base["chi2"]

    rows = []
    for k in k_grid:
        fit = wls_fit(ell, y, var, ks_extra=[float(k)], include_k1=True)
        delta = chi2_base - fit["chi2"]
        amp = fit["amps"].get(f"A_k_{k:.6f}", float("nan"))
        phase = fit["amps"].get(f"phi_k_{k:.6f}", float("nan"))
        rows.append(ScanRow(
            k=float(k),
            n_eff=float(n_from_k(k)),
            delta_chi2=float(delta),
            amp=float(amp),
            phase=float(phase),
        ))

    return base, rows


def find_wells(scan_rows):
    if not scan_rows:
        return []

    df = pd.DataFrame([asdict(r) for r in scan_rows])
    y = df["delta_chi2"].to_numpy()
    k_grid = df["k"].to_numpy()

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
        row = df.iloc[pidx]
        n_eff = float(row["n_eff"])
        nearest_int = round(n_eff)

        wells.append(WellRow(
            well_rank=int(rank),
            peak_index=pidx,
            k=float(row["k"]),
            n_eff=n_eff,
            delta_chi2=float(row["delta_chi2"]),
            prominence=float(prominences[oi]),
            nearest_integer_n=float(nearest_int),
            distance_to_integer=float(abs(n_eff - nearest_int)),
            distance_to_n10=float(abs(n_eff - 10.0)),
            distance_to_n15=float(abs(n_eff - 15.0)),
            distance_to_n20=float(abs(n_eff - 20.0)),
        ))

    return wells


def triplets_from_wells(wells):
    if len(wells) < 3:
        return []

    candidates = sorted(wells[:MAX_WELLS_FOR_TRIPLETS], key=lambda w: w.n_eff)
    triplets = []

    for w1, w2, w3 in combinations(candidates, 3):
        n1, n2, n3 = w1.n_eff, w2.n_eff, w3.n_eff
        if n2 <= 0:
            continue

        q_low = n1 / n2
        q_high = n3 / (2.0 * n2)
        q_mean = 0.5 * (q_low + q_high)
        koide_error = math.sqrt((q_low - KOIDE_Q) ** 2 + (q_high - KOIDE_Q) ** 2)
        integer_error = math.sqrt(
            (n1 - 10.0) ** 2 +
            (n2 - 15.0) ** 2 +
            (n3 - 20.0) ** 2
        )

        mean_delta = (w1.delta_chi2 + w2.delta_chi2 + w3.delta_chi2) / 3.0
        score = mean_delta / (1.0 + 25.0 * koide_error + 0.25 * integer_error)

        triplets.append(TripletRow(
            k1=float(w1.k),
            k2=float(w2.k),
            k3=float(w3.k),
            n1=float(n1),
            n2=float(n2),
            n3=float(n3),
            delta1=float(w1.delta_chi2),
            delta2=float(w2.delta_chi2),
            delta3=float(w3.delta_chi2),
            Q_low=float(q_low),
            Q_high=float(q_high),
            Q_mean=float(q_mean),
            koide_error=float(koide_error),
            integer_error_10_15_20=float(integer_error),
            score=float(score),
        ))

    triplets.sort(key=lambda r: (r.koide_error, r.integer_error_10_15_20, -r.score))
    return triplets


def comb_fit_delta(ell, y, var, ns):
    ks = [k_from_n(n) for n in ns]
    base = wls_fit(ell, y, var, ks_extra=[], include_k1=True)
    fit = wls_fit(ell, y, var, ks_extra=ks, include_k1=True)
    return base["chi2"] - fit["chi2"], ks, fit


def pure_scale(a_from, b_to):
    x = np.asarray(a_from, dtype=float)
    y = np.asarray(b_to, dtype=float)
    denom = float(np.dot(x, x))
    if denom <= 0:
        return None
    a = float(np.dot(x, y) / denom)
    resid = y - a * x
    rmse = float(np.sqrt(np.mean(resid * resid)))
    return a, rmse, resid.tolist()


def empirical_p(value, null_values):
    null_values = np.asarray(null_values, dtype=float)
    return float((1 + np.sum(null_values >= value)) / (len(null_values) + 1))


def run_nulls(ell, var, k_grid, n_null, rng):
    max_null = []
    kref_null = []
    n15_null = []
    comb_101520_null = []

    k15 = k_from_n(15)
    ns_101520 = [10.0, 15.0, 20.0]

    for j in range(n_null):
        if (j + 1) % max(1, n_null // 10) == 0:
            print(f"  [null] {j+1}/{n_null}")

        y0 = rng.normal(0.0, np.sqrt(np.maximum(var, 1.0)), size=len(var))
        base0, rows0 = scan_one_mode(ell, y0, var, k_grid)
        vals = np.array([r.delta_chi2 for r in rows0], dtype=float)
        max_null.append(float(np.max(vals)))

        # Fixed reference k and fixed n=15.
        d_ref = wls_fit(ell, y0, var, ks_extra=[], include_k1=True)["chi2"] - \
                wls_fit(ell, y0, var, ks_extra=[K_REF], include_k1=True)["chi2"]
        d_n15 = wls_fit(ell, y0, var, ks_extra=[], include_k1=True)["chi2"] - \
                wls_fit(ell, y0, var, ks_extra=[k15], include_k1=True)["chi2"]

        d_comb, _, _ = comb_fit_delta(ell, y0, var, ns_101520)

        kref_null.append(float(d_ref))
        n15_null.append(float(d_n15))
        comb_101520_null.append(float(d_comb))

    return {
        "scanmax": np.array(max_null, dtype=float),
        "kref": np.array(kref_null, dtype=float),
        "n15": np.array(n15_null, dtype=float),
        "comb_101520": np.array(comb_101520_null, dtype=float),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default=None, help='ROOT glob, e.g. "data/*.root"')
    parser.add_argument("--n-null", type=int, default=DEFAULT_N_NULL, help="Number of Gaussian null trials")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    print("=" * 100)
    print("SIDEBAND-SUBTRACTED RESIDUAL TEST")
    print("=" * 100)
    print(f"[config] active intervals = {ACTIVE_INTERVALS}")
    print(f"[config] Delta ell active = {DELTA_ELL_ACTIVE:.10f}")
    print(f"[config] k_ref = {K_REF:.6f}")
    print(f"[config] k(n=10,15,20) = {k_from_n(10):.6f}, {k_from_n(15):.6f}, {k_from_n(20):.6f}")
    print(f"[config] k scan = [{K_SCAN_MIN}, {K_SCAN_MAX}], N={N_K_SCAN}")
    print(f"[config] n_null = {args.n_null}, seed = {args.seed}")
    print("=" * 100)

    files = find_files(args.pattern)
    df, provenance = load_all_events(files)

    sig = select_region(df, B_SIGNAL, KST_SIGNAL)
    low = select_region(df, B_LOW_SB, KST_SIGNAL)
    high = select_region(df, B_HIGH_SB, KST_SIGNAL)

    print("\n[event counts after active support]")
    print(f"  signal: {len(sig):,}")
    print(f"  B-low sideband: {len(low):,}")
    print(f"  B-high sideband: {len(high):,}")

    if len(sig) < 100 or (len(low) + len(high)) < 100:
        raise RuntimeError("Too few events for sideband-subtracted test.")

    ell, q2_centers, h_sig = make_histogram(sig["q2"].to_numpy(), N_BINS)
    _, _, h_low = make_histogram(low["q2"].to_numpy(), N_BINS)
    _, _, h_high = make_histogram(high["q2"].to_numpy(), N_BINS)

    h_side = h_low + h_high
    alpha = float(np.sum(h_sig) / max(np.sum(h_side), 1.0))

    residual = h_sig - alpha * h_side
    variance = h_sig + alpha * alpha * h_side
    variance = np.maximum(variance, 1.0)

    bins_df = pd.DataFrame({
        "ell": ell,
        "q2_center": q2_centers,
        "N_signal": h_sig,
        "N_Blow": h_low,
        "N_Bhigh": h_high,
        "N_side_combined": h_side,
        "alpha": alpha,
        "R_subtracted": residual,
        "variance": variance,
        "z_residual": residual / np.sqrt(variance),
    })
    bins_csv = os.path.join(OUTDIR, "sideband_subtracted_bins.csv")
    bins_df.to_csv(bins_csv, index=False)

    print(f"\n[sideband subtraction]")
    print(f"  alpha = {alpha:.9f}")
    print(f"  sum signal = {np.sum(h_sig):.1f}")
    print(f"  sum side = {np.sum(h_side):.1f}")
    print(f"  sum residual = {np.sum(residual):.6f}")
    print(f"  RMS z residual = {np.sqrt(np.mean((residual / np.sqrt(variance))**2)):.6f}")

    k_grid = np.linspace(K_SCAN_MIN, K_SCAN_MAX, N_K_SCAN)

    print("\n[scan] one-mode sideband-subtracted residual")
    base, scan_rows = scan_one_mode(ell, residual, variance, k_grid)
    scan_df = pd.DataFrame([asdict(r) for r in scan_rows])
    scan_csv = os.path.join(OUTDIR, "sideband_subtracted_scan.csv")
    scan_df.to_csv(scan_csv, index=False)

    best = scan_df.sort_values("delta_chi2", ascending=False).iloc[0].to_dict()
    kref_fit = wls_fit(ell, residual, variance, ks_extra=[K_REF], include_k1=True)
    base_fit = wls_fit(ell, residual, variance, ks_extra=[], include_k1=True)
    delta_kref = base_fit["chi2"] - kref_fit["chi2"]

    k15 = k_from_n(15.0)
    n15_fit = wls_fit(ell, residual, variance, ks_extra=[k15], include_k1=True)
    delta_n15 = base_fit["chi2"] - n15_fit["chi2"]

    print(f"  base chi2 = {base_fit['chi2']:.6f}, ndof = {base_fit['ndof']}")
    print(f"  best k = {best['k']:.6f}, n = {best['n_eff']:.6f}, delta_chi2 = {best['delta_chi2']:.6f}")
    print(f"  k_ref = {K_REF:.6f}, delta_chi2 = {delta_kref:.6f}")
    print(f"  n=15 k = {k15:.6f}, delta_chi2 = {delta_n15:.6f}")

    wells = find_wells(scan_rows)
    wells_df = pd.DataFrame([asdict(w) for w in wells])
    wells_csv = os.path.join(OUTDIR, "sideband_subtracted_wells.csv")
    wells_df.to_csv(wells_csv, index=False)

    print("\n[top wells]")
    if wells:
        print(wells_df.head(12).to_string(index=False))
    else:
        print("  none")

    triplets = triplets_from_wells(wells)
    triplets_df = pd.DataFrame([asdict(t) for t in triplets])
    triplets_csv = os.path.join(OUTDIR, "sideband_subtracted_triplets.csv")
    triplets_df.to_csv(triplets_csv, index=False)

    print("\n[best well-first triplets]")
    if triplets:
        cols = [
            "n1", "n2", "n3", "Q_low", "Q_high", "Q_mean",
            "koide_error", "integer_error_10_15_20", "score",
        ]
        print(triplets_df.head(10)[cols].to_string(index=False))
    else:
        print("  none")

    # Integer scan.
    integer_rows = []
    for n in range(INTEGER_N_MIN, INTEGER_N_MAX + 1):
        k = k_from_n(float(n))
        fit = wls_fit(ell, residual, variance, ks_extra=[k], include_k1=True)
        d = base_fit["chi2"] - fit["chi2"]
        integer_rows.append({
            "n": n,
            "k": k,
            "delta_chi2": d,
            "amp": fit["amps"].get(f"A_k_{k:.6f}", float("nan")),
            "phase": fit["amps"].get(f"phi_k_{k:.6f}", float("nan")),
        })

    integer_df = pd.DataFrame(integer_rows)
    integer_csv = os.path.join(OUTDIR, "sideband_subtracted_integer_scan.csv")
    integer_df.to_csv(integer_csv, index=False)

    # Comb tests.
    comb_specs = [
        ("koide_Q_2_3_true_sideband", [10.0, 15.0, 20.0]),
        ("folded_Q_4_9", [6.6666666667, 15.0, 13.3333333333]),
    ]

    comb_rows = []
    for name, ns in comb_specs:
        d, ks, fit = comb_fit_delta(ell, residual, variance, ns)
        comb_rows.append({
            "comb": name,
            "n_values": ns,
            "k_values": ks,
            "delta_chi2": d,
        })

    comb_df = pd.DataFrame(comb_rows)
    comb_csv = os.path.join(OUTDIR, "sideband_subtracted_comb_tests.csv")
    comb_df.to_csv(comb_csv, index=False)

    print("\n[integer scan]")
    print(integer_df.to_string(index=False))

    print("\n[comb tests]")
    print(comb_df[["comb", "delta_chi2"]].to_string(index=False))

    # Nulls.
    rng = np.random.default_rng(args.seed)
    print("\n[nulls] Gaussian sideband-subtracted residual null")
    nulls = run_nulls(ell, variance, k_grid, args.n_null, rng)

    p_best_scanmax = empirical_p(best["delta_chi2"], nulls["scanmax"])
    p_kref_fixed = empirical_p(delta_kref, nulls["kref"])
    p_n15_fixed = empirical_p(delta_n15, nulls["n15"])

    d_comb_101520 = float(comb_df.loc[comb_df["comb"] == "koide_Q_2_3_true_sideband", "delta_chi2"].iloc[0])
    p_comb_101520 = empirical_p(d_comb_101520, nulls["comb_101520"])

    # Scale if possible from well-first best triplet? Sideband-subtracted has only one residual,
    # so no cross-region scale is defined here. The main question is survival after subtraction.
    best_triplet = None
    if triplets:
        best_triplet = asdict(triplets[0])

    verdict = {
        "sideband_subtracted_survival": {
            "best_scan_delta_chi2": float(best["delta_chi2"]),
            "best_scan_k": float(best["k"]),
            "best_scan_n": float(best["n_eff"]),
            "p_best_scanmax": p_best_scanmax,
            "kref_delta_chi2": float(delta_kref),
            "p_kref_fixed": p_kref_fixed,
            "n15_delta_chi2": float(delta_n15),
            "p_n15_fixed": p_n15_fixed,
            "comb_101520_delta_chi2": d_comb_101520,
            "p_comb_101520": p_comb_101520,
            "best_triplet": best_triplet,
        },
        "interpretation_flags": {
            "best_scan_survives_scanmax_0p05": bool(p_best_scanmax <= P_THRESH_WEAK),
            "kref_survives_fixed_0p05": bool(p_kref_fixed <= P_THRESH_WEAK),
            "n15_survives_fixed_0p05": bool(p_n15_fixed <= P_THRESH_WEAK),
            "comb_101520_survives_0p05": bool(p_comb_101520 <= P_THRESH_WEAK),
            "strong_survival_requires_sideband_subtracted_n15_or_comb": bool(
                (p_n15_fixed <= P_THRESH_WEAK) or (p_comb_101520 <= P_THRESH_WEAK)
            ),
        },
        "caution": (
            "This is a sideband-subtracted WLS residual test. It preserves the same q2 active support "
            "and k-to-n mapping as the main analysis, but it is not a full official background model."
        ),
    }

    summary = {
        "script": "28_sideband_subtracted_residual_test.py",
        "files": files,
        "provenance": provenance,
        "active_intervals": ACTIVE_INTERVALS,
        "delta_ell_active": DELTA_ELL_ACTIVE,
        "windows": {
            "B_SIGNAL": B_SIGNAL,
            "B_LOW_SB": B_LOW_SB,
            "B_HIGH_SB": B_HIGH_SB,
            "KST_SIGNAL": KST_SIGNAL,
        },
        "counts": {
            "signal_active": int(len(sig)),
            "B_low_active": int(len(low)),
            "B_high_active": int(len(high)),
            "hist_signal_sum": float(np.sum(h_sig)),
            "hist_side_sum": float(np.sum(h_side)),
            "alpha": alpha,
        },
        "scan_config": {
            "K1_FIXED": K1_FIXED,
            "K_REF": K_REF,
            "K_SCAN_MIN": K_SCAN_MIN,
            "K_SCAN_MAX": K_SCAN_MAX,
            "N_K_SCAN": N_K_SCAN,
            "N_BINS": N_BINS,
            "n_null": args.n_null,
            "seed": args.seed,
            "k_targets": {
                "n10": k_from_n(10.0),
                "n15": k_from_n(15.0),
                "n20": k_from_n(20.0),
            },
        },
        "outputs": {
            "bins_csv": bins_csv,
            "scan_csv": scan_csv,
            "wells_csv": wells_csv,
            "triplets_csv": triplets_csv,
            "integer_csv": integer_csv,
            "comb_csv": comb_csv,
        },
        "verdict": verdict,
    }

    summary_json = os.path.join(OUTDIR, "sideband_subtracted_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 100)
    print("SIDEBAND-SUBTRACTED VERDICT")
    print("=" * 100)
    print(json.dumps(verdict, indent=2))
    print("\nSaved:")
    for p in [bins_csv, scan_csv, wells_csv, triplets_csv, integer_csv, comb_csv, summary_json]:
        print(" ", p)


if __name__ == "__main__":
    main()

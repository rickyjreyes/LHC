#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Blind Control-Channel / Reconstruction-Control Test for LHCb Log-Winding / Koide Geometry
---------------------------------------------------------------------------------------

Modes
-----
rare_like:
    Uses the same active q^2 support as the B0 -> K*0 mu+ mu- manuscript:
        (0.1, 8.0), (11.0, 12.5), (14.5, 19.0)
    This is the only mode where the active-domain integer targets n={10,15,20}
    map to k inside the original k scan [6,32]. It is a true active-domain
    package test only if the input files are an independent rare-like control
    with continuous q^2 support.

jpsi_peak:
    For B+ -> J/psi(mu+mu-) K+ style ntuples. Because q^2 is concentrated near
    m(J/psi)^2, this mode uses q^2 in (8.0, 11.0). This is a reconstruction /
    J/psi peak sanity check, NOT a direct n=15 active-domain control, because
    Delta ell is much smaller and n={10,15,20} maps to k >> 32.

Outputs
-------
outputs_control_blind/
    control_well_first_scan_curve.csv
    control_wells.csv
    control_triplets.csv
    control_summary.json
    control_blind_verdict.json

Usage
-----
    python 27_control_channel_blind_test.py
    python 27_control_channel_blind_test.py --mode jpsi_peak
    python 27_control_channel_blind_test.py --mode rare_like --pattern "data_control/*.root"
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import uproot
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Missing uproot. Install with: pip install uproot awkward") from exc

try:
    from scipy.optimize import minimize
    from scipy.signal import find_peaks
    from scipy.stats import gaussian_kde
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Missing scipy. Install with: pip install scipy") from exc


# ============================================================
# Fixed settings
# ============================================================

OUTDIR = "outputs_control_blind"
os.makedirs(OUTDIR, exist_ok=True)

DEFAULT_PATTERNS = [
    "data_control/*.dvntuple.root",
    "data_control/*.root",
]

Q2_MIN = 0.1
Q2_MAX = 19.0

# Constants inherited from rare-decay manuscript.
K1_FIXED = 7.61054
KOIDE_Q = 2.0 / 3.0
A_TARGET_OBSERVED = 1.22828743138222
A_TARGET_GEOM = math.sqrt(3.0 / 2.0)

# Scan controls.
K_SCAN_MIN = 6.0
K_SCAN_MAX = 32.0
N_K_SCAN = 1301
N_BINS = 240
KDE_BANDWIDTH_SCALE = 1.00
A_MAX = 0.10

MIN_PEAK_PROMINENCE = 1.0
MIN_PEAK_DISTANCE_K = 0.75
MAX_WELLS_FOR_TRIPLETS = 12

# Blind pass/fail thresholds.
N15_TOL = 0.60
KOIDE_ERR_TOL = 0.025
INTEGER_10_15_20_ERR_TOL = 1.25
A_SCALE_TOL = 0.03
A_GEOM_TOL = 0.03
MIN_REGION_EVENTS = 500


# ============================================================
# Mode configuration
# ============================================================

def build_mode_config(mode: str) -> Dict[str, Any]:
    mode = mode.strip().lower()

    if mode == "rare_like":
        active_intervals = [
            (0.1, 8.0),
            (11.0, 12.5),
            (14.5, 19.0),
        ]
        b_signal = (5230.0, 5330.0)
        b_low_sb = (5000.0, 5180.0)
        b_high_sb = (5380.0, 5600.0)
        kst_signal = (795.9, 995.9)
        regions = [
            {"region": "signal_B_signal_Kst", "B_window": b_signal, "Kst_window": kst_signal},
            {"region": "B_low_sideband_Kst_signal", "B_window": b_low_sb, "Kst_window": kst_signal},
            {"region": "B_high_sideband_Kst_signal", "B_window": b_high_sb, "Kst_window": kst_signal},
        ]
        test_type = "rare_like_active_domain_control"

    elif mode == "jpsi_peak":
        active_intervals = [(8.0, 11.0)]
        # B+ mass regions for B+ -> J/psi K+ control ntuples.
        b_signal = (5220.0, 5350.0)
        b_low_sb = (5050.0, 5180.0)
        b_high_sb = (5380.0, 5600.0)
        # No K* in B+ -> J/psi K+. Dummy Kst_M=900.0 is assigned if no K* branch exists.
        kst_signal = (795.9, 995.9)
        regions = [
            {"region": "signal_Bplus_jpsiK_control", "B_window": b_signal, "Kst_window": kst_signal},
            {"region": "B_low_sideband_jpsiK_control", "B_window": b_low_sb, "Kst_window": kst_signal},
            {"region": "B_high_sideband_jpsiK_control", "B_window": b_high_sb, "Kst_window": kst_signal},
        ]
        test_type = "jpsi_peak_reconstruction_sanity_check"

    else:
        raise ValueError(f"Unknown mode {mode!r}. Use 'rare_like' or 'jpsi_peak'.")

    delta_ell = active_delta_ell(active_intervals)
    integer_targets_valid = all(K_SCAN_MIN <= k_from_n_with_delta(n, delta_ell) <= K_SCAN_MAX for n in [10, 15, 20])

    return {
        "mode": mode,
        "test_type": test_type,
        "active_intervals": active_intervals,
        "regions": regions,
        "delta_ell_active": delta_ell,
        "integer_targets_valid": integer_targets_valid,
    }


# ============================================================
# Helpers
# ============================================================

def active_delta_ell(intervals: Iterable[Tuple[float, float]]) -> float:
    return float(sum(math.log(hi / lo) for lo, hi in intervals))


def n_from_k_with_delta(k: float, delta_ell: float) -> float:
    return float(k * delta_ell / (2.0 * math.pi))


def k_from_n_with_delta(n: float, delta_ell: float) -> float:
    return float(2.0 * math.pi * n / delta_ell)


def in_active_intervals(q2: np.ndarray, active_intervals: List[Tuple[float, float]]) -> np.ndarray:
    q2 = np.asarray(q2, dtype=float)
    mask = np.zeros_like(q2, dtype=bool)
    for lo, hi in active_intervals:
        mask |= (q2 >= lo) & (q2 <= hi)
    return mask


def find_files(pattern: Optional[str] = None) -> List[str]:
    if pattern:
        files = sorted(glob.glob(pattern))
    else:
        files = []
        for pat in DEFAULT_PATTERNS:
            files.extend(glob.glob(pat))
        files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No control ROOT files found. Put files under data_control/ or pass --pattern.")
    return files


def candidate_branch(keys: Iterable[str], options: Iterable[str]) -> Optional[str]:
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


def find_tree(root_file: str) -> str:
    with uproot.open(root_file) as f:
        for key, obj in f.items(recursive=True):
            if hasattr(obj, "arrays") and "DecayTree" in key:
                return key
        for key, obj in f.items(recursive=True):
            if hasattr(obj, "arrays"):
                return key
    raise RuntimeError(f"No TTree found in {root_file}")


def find_particle_component(keys: Iterable[str], particle_patterns: Iterable[str], comp: str) -> Optional[str]:
    keys = list(keys)
    comp_upper = comp.upper()
    exact: List[str] = []
    for p in particle_patterns:
        exact.extend([f"{p}_{comp}", f"{p}{comp}", f"{p}.{comp}", f"{p}_{comp_upper}", f"{p}{comp_upper}"])
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


def derive_q2_from_muons(tree: Any, keys: Iterable[str]) -> Tuple[np.ndarray, Dict[str, str]]:
    plus_patterns = ["muplus", "mu_plus", "mup", "mu_p", "muplus0", "muplus_0", "MuPlus", "mup_0", "mu1", "muplus_1"]
    minus_patterns = ["muminus", "mu_minus", "mum", "mu_m", "muminus0", "muminus_0", "MuMinus", "mum_0", "mu2", "muminus_1"]
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

    branch_values = list(branches.values())
    arr = tree.arrays(branch_values, library="np")
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
    return q2_mev2 / 1.0e6, {k: str(v) for k, v in branches.items()}


def load_all_events(files: List[str]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    q2_candidates = ["q2", "Q2", "q2_DTF", "Q2_DTF", "mumu_M2", "dimuon_M2", "Jpsi_M2", "J_psi_1S_M2", "J_psi_1S_MM2"]
    b_mass_candidates = ["B0_M", "B0_MM", "B_M", "B_MM", "Bplus_M", "Bplus_MM", "Bplus0_M", "Bplus0_MM"]
    kst_mass_candidates = ["Kst_892_0_M", "Kst_892_0_MM", "Kst_M", "Kst_MM", "Kstar_M", "Kstar_MM", "Kstar0_M", "Kstar0_MM"]

    rows: List[pd.DataFrame] = []
    provenance: List[Dict[str, Any]] = []

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
                raise RuntimeError(f"No B mass branch found in {path}")
            if kst_branch is None:
                print(f"[warn] No K* mass branch found in {path}; using dummy Kst_M=900.0 for non-K* control channel.")

            if q2_branch:
                branches_to_read = [q2_branch, b_branch]
                if kst_branch is not None:
                    branches_to_read.append(kst_branch)
                arr = tree.arrays(branches_to_read, library="np")
                q2 = np.asarray(arr[q2_branch], dtype=float)
                finite = q2[np.isfinite(q2)]
                if len(finite) and np.nanmedian(finite) > 1e4:
                    q2 = q2 / 1.0e6
                print(f"[q2] using branch {q2_branch}")
                q2_source = q2_branch
            else:
                q2, _used_mu = derive_q2_from_muons(tree, keys)
                branches_to_read = [b_branch]
                if kst_branch is not None:
                    branches_to_read.append(kst_branch)
                arr = tree.arrays(branches_to_read, library="np")
                print("[q2] derived from muon four-vectors")
                q2_source = "derived_from_muon_four_vectors"

            bm = np.asarray(arr[b_branch], dtype=float)
            if kst_branch is not None:
                km = np.asarray(arr[kst_branch], dtype=float)
            else:
                km = np.full_like(bm, 900.0, dtype=float)

            mask = np.isfinite(q2) & np.isfinite(bm) & np.isfinite(km)
            mask &= (q2 >= Q2_MIN) & (q2 <= Q2_MAX)

            sub = pd.DataFrame({"q2": q2[mask], "B_M": bm[mask], "Kst_M": km[mask], "source_file": os.path.basename(path)})
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


def select_region(df: pd.DataFrame, b_window: Tuple[float, float], kst_window: Tuple[float, float], active_intervals: List[Tuple[float, float]]) -> pd.DataFrame:
    blo, bhi = b_window
    klo, khi = kst_window
    mask = (df["B_M"] >= blo) & (df["B_M"] <= bhi)
    mask &= (df["Kst_M"] >= klo) & (df["Kst_M"] <= khi)
    mask &= in_active_intervals(df["q2"].to_numpy(), active_intervals)
    return df.loc[mask].copy()


def make_histogram(q2: np.ndarray, active_intervals: List[Tuple[float, float]], n_bins: int = N_BINS) -> Tuple[np.ndarray, np.ndarray]:
    ell = np.log(np.asarray(q2, dtype=float))
    ell_min = math.log(Q2_MIN)
    ell_max = math.log(Q2_MAX)
    counts, edges = np.histogram(ell, bins=n_bins, range=(ell_min, ell_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    q2_centers = np.exp(centers)
    active = in_active_intervals(q2_centers, active_intervals)
    return centers[active], counts[active].astype(float)


def kde_baseline(ell_centers: np.ndarray, counts: np.ndarray, bw_scale: float = KDE_BANDWIDTH_SCALE) -> np.ndarray:
    repeated = np.repeat(ell_centers, np.maximum(counts.astype(int), 0))
    if len(repeated) < 100:
        raise RuntimeError("Too few points for KDE baseline.")
    kde = gaussian_kde(repeated)
    kde.set_bandwidth(kde.factor * bw_scale)
    dens = kde(ell_centers)
    dens = np.maximum(dens, 1e-12)
    baseline = dens / dens.sum() * counts.sum()
    return np.maximum(baseline, 1e-9)


def poisson_deviance(y: np.ndarray, lam: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-12)
    out = np.zeros_like(y)
    nz = y > 0
    out[nz] = y[nz] * np.log(y[nz] / lam[nz]) - (y[nz] - lam[nz])
    out[~nz] = lam[~nz]
    return 2.0 * float(np.sum(out))


def basis_matrix(ell: np.ndarray, ks: List[float]) -> np.ndarray:
    cols = [np.ones_like(ell), np.cos(K1_FIXED * ell), np.sin(K1_FIXED * ell)]
    for k in ks:
        cols.append(np.cos(k * ell))
        cols.append(np.sin(k * ell))
    return np.vstack(cols).T


def fit_poisson_bounded(counts: np.ndarray, baseline: np.ndarray, ell: np.ndarray, ks: List[float]) -> Dict[str, Any]:
    y = np.asarray(counts, dtype=float)
    B = np.maximum(np.asarray(baseline, dtype=float), 1e-12)
    X = basis_matrix(ell, ks)
    p = X.shape[1]
    beta0 = np.zeros(p)
    bounds = [(None, None)] + [(-A_MAX, A_MAX)] * (p - 1)

    def nll(beta: np.ndarray) -> float:
        eta = np.clip(X @ beta, -20.0, 20.0)
        lam = B * np.exp(eta)
        return float(np.sum(lam - y * np.log(np.maximum(lam, 1e-12))))

    res = minimize(nll, beta0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-8})
    beta = res.x
    eta = np.clip(X @ beta, -20.0, 20.0)
    lam = B * np.exp(eta)
    dev = poisson_deviance(y, lam)
    A1 = math.sqrt(beta[1] * beta[1] + beta[2] * beta[2]) if p >= 3 else float("nan")
    A2 = float("nan")
    phi2 = float("nan")
    if len(ks) == 1:
        a2 = beta[3]
        b2 = beta[4]
        A2 = math.sqrt(a2 * a2 + b2 * b2)
        phi2 = math.atan2(-b2, a2)
    bound_active = any(abs(v) >= A_MAX - 1e-5 for v in beta[1:])
    return {"success": bool(res.success), "dev": float(dev), "A1": float(A1), "A2": float(A2), "phi2": float(phi2), "bound_active": bool(bound_active), "lambda": lam, "beta": beta}


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
    integer_error_10_15_20: float
    score: float


def scan_continuous_k(region_name: str, ell: np.ndarray, counts: np.ndarray, baseline: np.ndarray, delta_ell: float) -> List[ScanRow]:
    base_fit = fit_poisson_bounded(counts, baseline, ell, ks=[])
    d_base = base_fit["dev"]
    print(f"[base] {region_name} D_base={d_base:.6f}, A1={base_fit['A1']:.6f}")
    k_grid = np.linspace(K_SCAN_MIN, K_SCAN_MAX, N_K_SCAN)
    rows: List[ScanRow] = []
    for i, k in enumerate(k_grid):
        fit = fit_poisson_bounded(counts, baseline, ell, ks=[float(k)])
        dD = d_base - fit["dev"]
        rows.append(ScanRow(region_name, float(k), n_from_k_with_delta(k, delta_ell), float(dD), float(fit["A2"]), float(fit["phi2"]), bool(fit["bound_active"]), bool(fit["success"])))
        if (i + 1) % 250 == 0:
            print(f"  scanned {i + 1}/{N_K_SCAN}")
    return rows


def find_wells(region_name: str, scan_rows: List[ScanRow]) -> List[WellRow]:
    if not scan_rows:
        return []
    df = pd.DataFrame([asdict(r) for r in scan_rows])
    y = df["deltaD"].to_numpy()
    k_grid = df["k"].to_numpy()
    dk = float(np.median(np.diff(k_grid)))
    min_distance_bins = max(1, int(round(MIN_PEAK_DISTANCE_K / dk)))
    peaks, props = find_peaks(y, prominence=MIN_PEAK_PROMINENCE, distance=min_distance_bins)
    if len(peaks) == 0:
        return []
    prominences = props.get("prominences", np.zeros(len(peaks)))
    order = sorted(range(len(peaks)), key=lambda i: y[peaks[i]], reverse=True)
    wells: List[WellRow] = []
    for rank, oi in enumerate(order, start=1):
        pidx = int(peaks[oi])
        row = df.iloc[pidx]
        n_eff = float(row["n_eff"])
        nearest_int = round(n_eff)
        wells.append(WellRow(
            region=region_name,
            well_rank=int(rank),
            peak_index=pidx,
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
    return wells


def triplets_from_wells(region_name: str, wells: List[WellRow]) -> List[TripletRow]:
    if len(wells) < 3:
        return []
    candidates = sorted(wells[:MAX_WELLS_FOR_TRIPLETS], key=lambda w: w.n_eff)
    triplets: List[TripletRow] = []
    for w1, w2, w3 in combinations(candidates, 3):
        n1, n2, n3 = w1.n_eff, w2.n_eff, w3.n_eff
        if n2 <= 0:
            continue
        q_low = n1 / n2
        q_high = n3 / (2.0 * n2)
        q_mean = 0.5 * (q_low + q_high)
        koide_error = math.sqrt((q_low - KOIDE_Q) ** 2 + (q_high - KOIDE_Q) ** 2)
        integer_error = math.sqrt((n1 - 10.0) ** 2 + (n2 - 15.0) ** 2 + (n3 - 20.0) ** 2)
        mean_dD = (w1.deltaD + w2.deltaD + w3.deltaD) / 3.0
        score = mean_dD / (1.0 + 25.0 * koide_error + 0.25 * integer_error)
        triplets.append(TripletRow(region_name, float(w1.k), float(w2.k), float(w3.k), float(n1), float(n2), float(n3), float(w1.deltaD), float(w2.deltaD), float(w3.deltaD), float(q_low), float(q_high), float(q_mean), float(koide_error), float(integer_error), float(score)))
    triplets.sort(key=lambda r: (r.koide_error, r.integer_error_10_15_20, -r.score))
    return triplets


def best_triplet_for_region(triplets_df: pd.DataFrame, region: str) -> Optional[Dict[str, Any]]:
    if triplets_df.empty:
        return None
    sub = triplets_df[triplets_df["region"] == region].copy()
    if sub.empty:
        return None
    sub = sub.sort_values(["koide_error", "integer_error_10_15_20", "score"], ascending=[True, True, False])
    return sub.iloc[0].to_dict()


def pure_scale(a_from: List[float], b_to: List[float]) -> Optional[Tuple[float, float, List[float]]]:
    x = np.asarray(a_from, dtype=float)
    y = np.asarray(b_to, dtype=float)
    denom = float(np.dot(x, x))
    if denom <= 0:
        return None
    a = float(np.dot(x, y) / denom)
    resid = y - a * x
    rmse = float(np.sqrt(np.mean(resid * resid)))
    return a, rmse, resid.tolist()


def blind_verdict(wells_df: pd.DataFrame, triplets_df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    mode = cfg["mode"]
    regions = cfg["regions"]
    region_names = [r["region"] for r in regions]
    signal_region = region_names[0]
    b_low_region = region_names[1]
    b_high_region = region_names[2]
    integer_targets_valid = bool(cfg["integer_targets_valid"])
    package_verdict_valid = bool(mode == "rare_like" and integer_targets_valid)

    verdict: Dict[str, Any] = {
        "control_mode": mode,
        "control_test_type": cfg["test_type"],
        "integer_targets_valid_in_scan": integer_targets_valid,
        "package_reproduction_verdict_is_valid": package_verdict_valid,
        "pre_registered_target": {
            "a_observed_from_rare_decay": A_TARGET_OBSERVED,
            "a_geom_sqrt_3_over_2": A_TARGET_GEOM,
            "koide_Q": KOIDE_Q,
            "n_targets": [10, 15, 20],
        },
        "criteria": {},
        "overall": {},
    }

    c1_regions: Dict[str, Any] = {}
    if not wells_df.empty:
        for region, g in wells_df.groupby("region"):
            top = g.sort_values("deltaD", ascending=False).head(10)
            best_dist = float(top["distance_to_n15"].min())
            c1_regions[region] = {"best_distance_to_n15_among_top10": best_dist, "passes": bool(integer_targets_valid and best_dist <= N15_TOL)}
    verdict["criteria"]["c1_n15_present"] = c1_regions

    c23_regions: Dict[str, Any] = {}
    for region in region_names:
        best = best_triplet_for_region(triplets_df, region)
        if best is None:
            c23_regions[region] = {"available": False, "passes_koide_error": False, "passes_integer_10_15_20": False}
            continue
        c23_regions[region] = {
            "available": True,
            "n_triplet": [best["n1"], best["n2"], best["n3"]],
            "Q_low": best["Q_low"],
            "Q_high": best["Q_high"],
            "Q_mean": best["Q_mean"],
            "koide_error": best["koide_error"],
            "integer_error_10_15_20": best["integer_error_10_15_20"],
            "passes_koide_error": bool(best["koide_error"] <= KOIDE_ERR_TOL),
            "passes_integer_10_15_20": bool(integer_targets_valid and best["integer_error_10_15_20"] <= INTEGER_10_15_20_ERR_TOL),
        }
    verdict["criteria"]["c2_c3_best_triplet_geometry"] = c23_regions

    b_low = best_triplet_for_region(triplets_df, b_low_region)
    sig = best_triplet_for_region(triplets_df, signal_region)
    scale_info: Dict[str, Any] = {
        "available": False,
        "control_mode": mode,
        "signal_region": signal_region,
        "b_low_region": b_low_region,
        "passes_a_1p2283": False,
        "passes_sqrt_3_over_2": False,
    }
    if b_low is not None and sig is not None:
        x = [b_low["n1"], b_low["n2"], b_low["n3"]]
        y = [sig["n1"], sig["n2"], sig["n3"]]
        s = pure_scale(x, y)
        if s:
            a, rmse, resid = s
            scale_info = {
                "available": True,
                "control_mode": mode,
                "signal_region": signal_region,
                "b_low_region": b_low_region,
                "B_low_triplet": x,
                "signal_triplet": y,
                "a_scale": a,
                "rmse": rmse,
                "residuals": resid,
                "distance_to_rare_decay_a": abs(a - A_TARGET_OBSERVED),
                "distance_to_sqrt_3_over_2": abs(a - A_TARGET_GEOM),
                "passes_a_1p2283": bool(package_verdict_valid and abs(a - A_TARGET_OBSERVED) <= A_SCALE_TOL),
                "passes_sqrt_3_over_2": bool(package_verdict_valid and abs(a - A_TARGET_GEOM) <= A_GEOM_TOL),
            }
    verdict["criteria"]["c4_Blow_to_signal_scale"] = scale_info

    n15_pass_count = sum(int(v.get("passes", False)) for v in c1_regions.values())
    koide_pass_count = sum(int(v.get("passes_koide_error", False)) for v in c23_regions.values())
    integer_pass_count = sum(int(v.get("passes_integer_10_15_20", False)) for v in c23_regions.values())
    scale_pass = bool(scale_info.get("passes_a_1p2283", False) or scale_info.get("passes_sqrt_3_over_2", False))
    reproduced_score = int(n15_pass_count >= 1) + int(koide_pass_count >= 1) + int(integer_pass_count >= 1) + int(scale_pass)

    if not package_verdict_valid:
        interpretation = (
            "This mode is a reconstruction / peak-shape sanity check, not a valid active-domain package verdict. "
            "The rare-decay integer targets do not lie in the scanned k range, so n=15, (10,15,20), and scale-package pass/fail should be read as not applicable."
        )
    elif reproduced_score >= 3:
        interpretation = (
            "Control channel reproduces much of the rare-decay package. This favors generic candidate-spectrum, "
            "detector/reconstruction, or broader active-domain geometry over rare-decay specificity."
        )
    else:
        interpretation = (
            "Control channel fails to reproduce the rare-decay package. This supports channel specificity, "
            "but is not by itself a discovery claim."
        )

    verdict["overall"] = {
        "control_mode": mode,
        "control_test_type": cfg["test_type"],
        "integer_targets_valid_in_scan": integer_targets_valid,
        "package_reproduction_verdict_is_valid": package_verdict_valid,
        "n15_pass_count": int(n15_pass_count),
        "koide_pass_count": int(koide_pass_count),
        "integer_10_15_20_pass_count": int(integer_pass_count),
        "scale_pass": bool(scale_pass),
        "reproduced_score_out_of_4": int(reproduced_score),
        "control_reproduces_rare_decay_package": bool(package_verdict_valid and reproduced_score >= 3),
        "interpretation": interpretation,
    }
    return verdict


def run(pattern: Optional[str] = None, mode: str = "jpsi_peak") -> None:
    cfg = build_mode_config(mode)
    active_intervals = cfg["active_intervals"]
    delta_ell = cfg["delta_ell_active"]

    print("=" * 100)
    print("BLIND CONTROL-CHANNEL TEST")
    print("=" * 100)
    print(f"[config] control_mode = {cfg['mode']}")
    print(f"[config] control_test_type = {cfg['test_type']}")
    print(f"[config] active intervals: {active_intervals}")
    print(f"[config] Delta ell active = {delta_ell:.10f}")
    print(f"[config] k(10), k(15), k(20) = {k_from_n_with_delta(10, delta_ell):.6f}, {k_from_n_with_delta(15, delta_ell):.6f}, {k_from_n_with_delta(20, delta_ell):.6f}")
    print(f"[config] k_scan_range = [{K_SCAN_MIN}, {K_SCAN_MAX}]")
    for ntarget in [10, 15, 20]:
        kt = k_from_n_with_delta(ntarget, delta_ell)
        if not (K_SCAN_MIN <= kt <= K_SCAN_MAX):
            print(f"[warn] n={ntarget} maps to k={kt:.3f}, outside scan range [{K_SCAN_MIN}, {K_SCAN_MAX}]. Integer-winding tests are not valid in this mode.")
    print(f"[config] integer_targets_valid_in_scan = {cfg['integer_targets_valid']}")
    print(f"[config] target a rare-decay = {A_TARGET_OBSERVED:.9f}")
    print(f"[config] target sqrt(3/2) = {A_TARGET_GEOM:.9f}")
    print("=" * 100)

    files = find_files(pattern)
    df, provenance = load_all_events(files)

    all_scan: List[ScanRow] = []
    all_wells: List[WellRow] = []
    all_triplets: List[TripletRow] = []
    region_counts: Dict[str, int] = {}

    for region_cfg in cfg["regions"]:
        region_name = region_cfg["region"]
        sub = select_region(df, region_cfg["B_window"], region_cfg["Kst_window"], active_intervals)
        region_counts[region_name] = int(len(sub))

        print("\n" + "=" * 100)
        print(f"[region] {region_name}")
        print(f"  B window: {region_cfg['B_window']}")
        print(f"  K* window: {region_cfg['Kst_window']}")
        print(f"  N active: {len(sub):,}")
        print("=" * 100)

        if len(sub) < MIN_REGION_EVENTS:
            print("[skip] too few events")
            continue

        ell, counts = make_histogram(sub["q2"].to_numpy(), active_intervals, N_BINS)
        if counts.sum() < MIN_REGION_EVENTS:
            print("[skip] too few binned counts")
            continue

        baseline = kde_baseline(ell, counts, KDE_BANDWIDTH_SCALE)
        scan = scan_continuous_k(region_name, ell, counts, baseline, delta_ell)
        wells = find_wells(region_name, scan)
        triplets = triplets_from_wells(region_name, wells)

        all_scan.extend(scan)
        all_wells.extend(wells)
        all_triplets.extend(triplets)

        print("\n[top wells]")
        if wells:
            print(pd.DataFrame([asdict(w) for w in wells[:10]])[["well_rank", "k", "n_eff", "deltaD", "distance_to_n15", "distance_to_n10", "distance_to_n20", "A2", "bound_active"]].to_string(index=False))
        else:
            print("  none")

        print("\n[best triplets]")
        if triplets:
            print(pd.DataFrame([asdict(t) for t in triplets[:10]])[["n1", "n2", "n3", "Q_low", "Q_high", "Q_mean", "koide_error", "integer_error_10_15_20", "score"]].to_string(index=False))
        else:
            print("  none")

    scan_df = pd.DataFrame([asdict(r) for r in all_scan])
    wells_df = pd.DataFrame([asdict(w) for w in all_wells])
    triplets_df = pd.DataFrame([asdict(t) for t in all_triplets])

    scan_csv = os.path.join(OUTDIR, "control_well_first_scan_curve.csv")
    wells_csv = os.path.join(OUTDIR, "control_wells.csv")
    triplets_csv = os.path.join(OUTDIR, "control_triplets.csv")
    summary_json = os.path.join(OUTDIR, "control_summary.json")
    verdict_json = os.path.join(OUTDIR, "control_blind_verdict.json")

    scan_df.to_csv(scan_csv, index=False)
    wells_df.to_csv(wells_csv, index=False)
    triplets_df.to_csv(triplets_csv, index=False)

    verdict = blind_verdict(wells_df, triplets_df, cfg)
    summary = {
        "script": "27_control_channel_blind_test.py",
        "purpose": "Blind control-channel / reconstruction-control test of log-winding / Koide package.",
        "control_mode": cfg["mode"],
        "control_test_type": cfg["test_type"],
        "integer_targets_valid_in_scan": bool(cfg["integer_targets_valid"]),
        "files": files,
        "provenance": provenance,
        "region_counts": region_counts,
        "active_intervals": active_intervals,
        "delta_ell_active": delta_ell,
        "k1_fixed": K1_FIXED,
        "k_scan": [K_SCAN_MIN, K_SCAN_MAX, N_K_SCAN],
        "k_targets": {"n10": k_from_n_with_delta(10.0, delta_ell), "n15": k_from_n_with_delta(15.0, delta_ell), "n20": k_from_n_with_delta(20.0, delta_ell)},
        "targets": {"koide_Q": KOIDE_Q, "a_rare_decay": A_TARGET_OBSERVED, "a_geom_sqrt_3_over_2": A_TARGET_GEOM},
        "outputs": {"scan_csv": scan_csv, "wells_csv": wells_csv, "triplets_csv": triplets_csv, "verdict_json": verdict_json},
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(verdict_json, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)

    print("\n" + "=" * 100)
    print("CONTROL-CHANNEL BLIND VERDICT")
    print("=" * 100)
    print(json.dumps(verdict["overall"], indent=2))
    print("\nSaved:")
    print(" ", scan_csv)
    print(" ", wells_csv)
    print(" ", triplets_csv)
    print(" ", summary_json)
    print(" ", verdict_json)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default=None, help='Optional ROOT glob, e.g. "data_jpsi/*.root". Default: data_control/*.root')
    parser.add_argument("--mode", default="jpsi_peak", choices=["rare_like", "jpsi_peak"], help="Control mode to run.")
    args = parser.parse_args()
    run(args.pattern, args.mode)


if __name__ == "__main__":
    main()

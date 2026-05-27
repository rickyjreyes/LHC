#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Milestone 3A: Dynamic Scale-Phase Invariant Test
================================================

Purpose
-------
Test whether the sideband-subtracted mobile Koide-like triplet follows a
coherent trajectory when the subtraction strength is varied.

This directly tests the hypothesis:

    The absolute active-domain n-values may move, but the motion is governed
    by a coherent scale/phase law rather than arbitrary optimizer freedom.

Core ladder
-----------
For each lambda in a subtraction ladder:

    R_i(lambda) = N_sig,i - lambda * alpha * N_side,i

where:

    alpha = sum_i N_sig,i / sum_i N_side,i

At lambda = 0:
    no sideband subtraction.

At lambda = 1:
    nominal sideband subtraction.

At lambda > 1:
    over-subtraction stress test.

For each lambda, the script extracts:
    - best one-mode scan peak
    - top wells
    - best mobile scaled-Koide triplet
    - fitted scale a in n ≈ a*(10,15,20)
    - Koide ratio error
    - phase values at the triplet modes
    - score/strength

Then it computes a trajectory statistic comparing:
    - smoothness of a(lambda)
    - smoothness of phase(lambda)
    - stability of Q_mean(lambda)
    - continuity of selected triplet branch
    - mean mobile score

Null
----
The same lambda ladder is run on Gaussian residual nulls:

    y0_i(lambda) ~ Normal(0, sqrt(Var_i(lambda)))

The empirical p-value is:

    p_traj = P_null(T_traj_null >= T_traj_observed)

Outputs
-------
outputs_milestone3_lambda_trajectory/
    lambda_trajectory.csv
    lambda_wells.csv
    lambda_triplets.csv
    null_trajectory_scores.csv
    milestone3_summary.json
    milestone3_report.txt
    optional plots

Run
---
Fast test:
    python milestone3_dynamic_scale_phase_cupy.py --pattern "data/*.root" --n-null 100 --gpu --plot

Main run:
    python milestone3_dynamic_scale_phase_cupy.py --pattern "data/*.root" --n-null 1000 --gpu --plot

More detailed lambda grid:
    python milestone3_dynamic_scale_phase_cupy.py --lambda-min 0 --lambda-max 1.25 --lambda-steps 26 --n-null 1000 --gpu --plot
"""

import argparse
import glob
import json
import math
import os
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Backend
# ============================================================

def get_backend(use_gpu: bool):
    if not use_gpu:
        return np, False
    try:
        import cupy as cp
        _ = cp.zeros(1)
        return cp, True
    except Exception as exc:
        print(f"[warn] CuPy unavailable; using NumPy. Reason: {exc}")
        return np, False


def to_numpy(x):
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    return np.asarray(x)


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    outdir: str = "outputs_milestone3_lambda_trajectory"

    q2_min: float = 0.1
    q2_max: float = 19.0

    jpsi_low: float = 8.0
    jpsi_high: float = 11.0
    psi2s_low: float = 12.5
    psi2s_high: float = 14.5

    b_signal_low: float = 5230.0
    b_signal_high: float = 5330.0
    b_low_sideband_low: float = 5000.0
    b_low_sideband_high: float = 5180.0
    b_high_sideband_low: float = 5380.0
    b_high_sideband_high: float = 5600.0

    kst_low: float = 795.9
    kst_high: float = 995.9

    n_bins: int = 240

    k1_fixed: float = 7.61054
    k_ref: float = 19.5296
    k_min: float = 6.0
    k_max: float = 32.0
    k_steps: int = 1301

    koide_q: float = 2.0 / 3.0

    min_peak_prominence: float = 0.5
    min_peak_distance_k: float = 0.75
    max_wells_for_triplets: int = 12

    lambda_min: float = 0.0
    lambda_max: float = 1.25
    lambda_steps: int = 26

    n_null: int = 1000
    seed: int = 314159

    # Mobile scaled-Koide score weights.
    mobile_q_weight: float = 25.0
    mobile_scale_weight: float = 0.25
    mobile_integer_weight: float = 0.0

    # Trajectory score weights.
    traj_q_weight: float = 25.0
    traj_scale_jump_weight: float = 10.0
    traj_phase_jump_weight: float = 1.0
    traj_branch_jump_weight: float = 0.25
    traj_score_weight: float = 1.0

    p_weak: float = 0.05
    p_strong: float = 0.01

    eps: float = 1e-12


# ============================================================
# Domain maps
# ============================================================

def active_intervals(cfg: Config) -> List[Tuple[float, float]]:
    return [
        (cfg.q2_min, cfg.jpsi_low),
        (cfg.jpsi_high, cfg.psi2s_low),
        (cfg.psi2s_high, cfg.q2_max),
    ]


def active_delta_ell(cfg: Config) -> float:
    return float(sum(math.log(hi / lo) for lo, hi in active_intervals(cfg)))


def n_from_k(k: float, cfg: Config) -> float:
    return float(k * active_delta_ell(cfg) / (2.0 * math.pi))


def k_from_n(n: float, cfg: Config) -> float:
    return float(2.0 * math.pi * n / active_delta_ell(cfg))


def in_active_intervals(q2: np.ndarray, cfg: Config) -> np.ndarray:
    q2 = np.asarray(q2, dtype=float)
    mask = np.zeros_like(q2, dtype=bool)
    for lo, hi in active_intervals(cfg):
        mask |= (q2 >= lo) & (q2 <= hi)
    return mask


def wrapped_phase_diff(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    return np.angle(np.exp(1j * np.diff(phi)))


# ============================================================
# ROOT loading
# ============================================================

def find_files(pattern: Optional[str]) -> List[str]:
    if pattern:
        files = sorted(glob.glob(pattern))
    else:
        files = sorted(set(glob.glob("data/*.dvntuple.root") + glob.glob("data/*.root")))
    if not files:
        raise FileNotFoundError("No ROOT files found. Use --pattern \"data/*.root\" or put files under data/.")
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
    import uproot
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
    exact = []
    for p in particle_patterns:
        exact.extend([
            f"{p}_{comp}", f"{p}{comp}", f"{p}.{comp}",
            f"{p}_{comp_upper}", f"{p}{comp_upper}",
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
    return q2_mev2 / 1.0e6, {k: str(v) for k, v in branches.items()}


def load_all_events(files: List[str], cfg: Config) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    import uproot

    q2_candidates = ["q2", "Q2", "q2_DTF", "Q2_DTF", "mumu_M2", "dimuon_M2", "Jpsi_M2", "J_psi_1S_M2", "J_psi_1S_MM2"]
    b_mass_candidates = ["B0_M", "B0_MM", "B_M", "B_MM"]
    kst_mass_candidates = [
        "Kst_892_0_M", "Kst_892_0_MM",
        "Kst_M", "Kst_MM",
        "Kstar_M", "Kstar_MM", "Kstar0_M", "Kstar0_MM",
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
                raise RuntimeError(f"No B0/B mass branch found in {path}")
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
                q2, _ = derive_q2_from_muons(tree, keys)
                arr = tree.arrays([b_branch, kst_branch], library="np")
                q2_source = "derived_from_muon_four_vectors"
                print("[q2] derived from muon four-vectors")

            bm = np.asarray(arr[b_branch], dtype=float)
            km = np.asarray(arr[kst_branch], dtype=float)

            mask = np.isfinite(q2) & np.isfinite(bm) & np.isfinite(km)
            mask &= (q2 >= cfg.q2_min) & (q2 <= cfg.q2_max)

            rows.append(pd.DataFrame({
                "q2": q2[mask],
                "B_M": bm[mask],
                "Kst_M": km[mask],
                "source_file": os.path.basename(path),
            }))

            provenance.append({
                "file": path,
                "tree": tree_name,
                "q2_source": q2_source,
                "B_mass_branch": b_branch,
                "Kst_mass_branch": kst_branch,
                "n_loaded_q2_range": int(np.sum(mask)),
            })

    df = pd.concat(rows, ignore_index=True)
    print(f"[info] loaded q2-range events: {len(df):,}")
    return df, provenance


# ============================================================
# Selection and histograms
# ============================================================

def select_region(df: pd.DataFrame, b_window: Tuple[float, float], cfg: Config) -> pd.DataFrame:
    blo, bhi = b_window
    mask = (df["B_M"] >= blo) & (df["B_M"] <= bhi)
    mask &= (df["Kst_M"] >= cfg.kst_low) & (df["Kst_M"] <= cfg.kst_high)
    mask &= in_active_intervals(df["q2"].to_numpy(), cfg)
    return df.loc[mask].copy()


def make_histogram(q2: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ell = np.log(np.asarray(q2, dtype=float))
    ell_min = math.log(cfg.q2_min)
    ell_max = math.log(cfg.q2_max)

    counts, edges = np.histogram(ell, bins=cfg.n_bins, range=(ell_min, ell_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    q2_centers = np.exp(centers)
    active = in_active_intervals(q2_centers, cfg)

    return centers[active], q2_centers[active], counts[active].astype(float)


# ============================================================
# WLS scan
# ============================================================

def wls_fit_np(ell: np.ndarray, y: np.ndarray, var: np.ndarray, ks_extra: List[float], cfg: Config, include_k1: bool = True):
    ell = np.asarray(ell, dtype=float)
    y = np.asarray(y, dtype=float)
    var = np.maximum(np.asarray(var, dtype=float), 1.0)

    cols = [np.ones_like(ell)]
    if include_k1:
        cols.append(np.cos(cfg.k1_fixed * ell))
        cols.append(np.sin(cfg.k1_fixed * ell))

    for k in ks_extra:
        cols.append(np.cos(float(k) * ell))
        cols.append(np.sin(float(k) * ell))

    X = np.vstack(cols).T
    w = 1.0 / np.sqrt(var)
    Xw = X * w[:, None]
    yw = y * w

    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    pred = X @ beta
    chi2 = float(np.sum((y - pred) ** 2 / var))

    amps = {}
    phases = {}
    idx = 1
    if include_k1:
        amps["A_k1"] = float(math.hypot(beta[idx], beta[idx + 1]))
        phases["phi_k1"] = float(math.atan2(-beta[idx + 1], beta[idx]))
        idx += 2

    for k in ks_extra:
        amps[f"A_k_{k:.6f}"] = float(math.hypot(beta[idx], beta[idx + 1]))
        phases[f"phi_k_{k:.6f}"] = float(math.atan2(-beta[idx + 1], beta[idx]))
        idx += 2

    return {
        "chi2": chi2,
        "beta": beta,
        "pred": pred,
        "amps": amps,
        "phases": phases,
        "ndof": int(len(y) - len(beta)),
    }


def wls_delta_for_many_k_gpu(ell_np, y_np, var_np, k_grid_np, cfg: Config, xp):
    base = wls_fit_np(ell_np, y_np, var_np, [], cfg, include_k1=True)

    y = xp.asarray(y_np, dtype=float)
    ell = xp.asarray(ell_np, dtype=float)
    var = xp.maximum(xp.asarray(var_np, dtype=float), 1.0)
    pred0 = xp.asarray(base["pred"], dtype=float)
    r = (y - pred0) / xp.sqrt(var)

    k_grid = xp.asarray(k_grid_np, dtype=float).reshape(-1, 1)
    ell_row = ell.reshape(1, -1)

    inv_sqrt_var = 1.0 / xp.sqrt(var)
    c = xp.cos(k_grid * ell_row) * inv_sqrt_var.reshape(1, -1)
    s = xp.sin(k_grid * ell_row) * inv_sqrt_var.reshape(1, -1)

    rr = r.reshape(1, -1)

    cc = xp.sum(c * c, axis=1) + cfg.eps
    ss = xp.sum(s * s, axis=1) + cfg.eps
    cs = xp.sum(c * s, axis=1)
    cy = xp.sum(c * rr, axis=1)
    sy = xp.sum(s * rr, axis=1)

    det = cc * ss - cs * cs + cfg.eps
    a = (ss * cy - cs * sy) / det
    b = (-cs * cy + cc * sy) / det

    delta = a * cy + b * sy
    amp = xp.sqrt(a * a + b * b)
    phase = xp.arctan2(-b, a)

    return base, to_numpy(delta), to_numpy(amp), to_numpy(phase)


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
    phase: float
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
    phase1: float
    phase2: float
    phase3: float
    Q_low: float
    Q_high: float
    Q_mean: float
    koide_error: float
    integer_error_10_15_20: float
    mobile_score: float
    a_scale: float
    scale_error: float
    mean_delta: float
    mean_phase: float


def scan_one_mode(ell, y, var, k_grid, cfg: Config, xp):
    base, deltas, amps, phases = wls_delta_for_many_k_gpu(ell, y, var, k_grid, cfg, xp)
    rows = [
        ScanRow(
            k=float(k),
            n_eff=n_from_k(float(k), cfg),
            delta_chi2=float(d),
            amp=float(a),
            phase=float(ph),
        )
        for k, d, a, ph in zip(k_grid, deltas, amps, phases)
    ]
    return base, rows


def find_wells(scan_rows: List[ScanRow], cfg: Config) -> List[WellRow]:
    try:
        from scipy.signal import find_peaks
    except Exception as exc:
        raise RuntimeError("Missing scipy. Install with: pip install scipy") from exc

    if not scan_rows:
        return []

    df = pd.DataFrame([asdict(r) for r in scan_rows])
    y = df["delta_chi2"].to_numpy()
    k_grid = df["k"].to_numpy()

    dk = float(np.median(np.diff(k_grid)))
    min_dist_bins = max(1, int(round(cfg.min_peak_distance_k / dk)))

    peaks, props = find_peaks(y, prominence=cfg.min_peak_prominence, distance=min_dist_bins)
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
            phase=float(row["phase"]),
            nearest_integer_n=float(nearest_int),
            distance_to_integer=float(abs(n_eff - nearest_int)),
            distance_to_n10=float(abs(n_eff - 10.0)),
            distance_to_n15=float(abs(n_eff - 15.0)),
            distance_to_n20=float(abs(n_eff - 20.0)),
        ))

    return wells


def scaled_koide_geometry(ns: List[float], cfg: Config) -> Dict[str, float]:
    n = np.array(sorted(ns), dtype=float)
    t = np.array([10.0, 15.0, 20.0], dtype=float)

    if len(n) != 3:
        return {"a_scale": np.nan, "scale_error": np.nan, "koide_error": np.nan, "q_low": np.nan, "q_high": np.nan, "q_mean": np.nan}

    a = float(np.dot(n, t) / np.dot(t, t))
    resid = n - a * t
    scale_error = float(np.sqrt(np.mean(resid * resid)))

    n1, n2, n3 = n
    q_low = float(n1 / n2)
    q_high = float(n3 / (2.0 * n2))
    q_mean = float(0.5 * (q_low + q_high))
    koide_error = float(math.sqrt((q_low - cfg.koide_q) ** 2 + (q_high - cfg.koide_q) ** 2))
    integer_error = float(np.linalg.norm(n - t))

    return {
        "a_scale": a,
        "scale_error": scale_error,
        "koide_error": koide_error,
        "q_low": q_low,
        "q_high": q_high,
        "q_mean": q_mean,
        "integer_error_10_15_20": integer_error,
    }


def triplets_from_wells(wells: List[WellRow], cfg: Config) -> List[TripletRow]:
    if len(wells) < 3:
        return []

    candidates = sorted(wells[:cfg.max_wells_for_triplets], key=lambda w: w.n_eff)
    triplets = []

    for w1, w2, w3 in combinations(candidates, 3):
        ns = [w1.n_eff, w2.n_eff, w3.n_eff]
        geom = scaled_koide_geometry(ns, cfg)

        mean_delta = float((w1.delta_chi2 + w2.delta_chi2 + w3.delta_chi2) / 3.0)
        denom = (
            1.0
            + cfg.mobile_q_weight * geom["koide_error"]
            + cfg.mobile_scale_weight * geom["scale_error"]
            + cfg.mobile_integer_weight * geom["integer_error_10_15_20"]
        )
        mobile_score = float(mean_delta / max(denom, cfg.eps))

        phases = np.array([w1.phase, w2.phase, w3.phase], dtype=float)
        mean_phase = float(np.angle(np.mean(np.exp(1j * phases))))

        triplets.append(TripletRow(
            k1=float(w1.k), k2=float(w2.k), k3=float(w3.k),
            n1=float(w1.n_eff), n2=float(w2.n_eff), n3=float(w3.n_eff),
            delta1=float(w1.delta_chi2), delta2=float(w2.delta_chi2), delta3=float(w3.delta_chi2),
            phase1=float(w1.phase), phase2=float(w2.phase), phase3=float(w3.phase),
            Q_low=float(geom["q_low"]),
            Q_high=float(geom["q_high"]),
            Q_mean=float(geom["q_mean"]),
            koide_error=float(geom["koide_error"]),
            integer_error_10_15_20=float(geom["integer_error_10_15_20"]),
            mobile_score=mobile_score,
            a_scale=float(geom["a_scale"]),
            scale_error=float(geom["scale_error"]),
            mean_delta=mean_delta,
            mean_phase=mean_phase,
        ))

    triplets.sort(key=lambda r: r.mobile_score, reverse=True)
    return triplets


# ============================================================
# Lambda analysis
# ============================================================

def analyze_one_lambda(lam: float, ell, h_sig, h_side, alpha, cfg: Config, xp, k_grid: np.ndarray, y_override=None, var_override=None):
    if y_override is None:
        y = h_sig - lam * alpha * h_side
    else:
        y = y_override

    if var_override is None:
        var = np.maximum(h_sig + (lam * alpha) ** 2 * h_side, 1.0)
    else:
        var = np.maximum(var_override, 1.0)

    base, scan_rows = scan_one_mode(ell, y, var, k_grid, cfg, xp)
    wells = find_wells(scan_rows, cfg)
    triplets = triplets_from_wells(wells, cfg)

    if triplets:
        best_mobile = triplets[0]
        best_mobile_dict = asdict(best_mobile)
    else:
        best_mobile = None
        best_mobile_dict = {}

    best_scan = max(scan_rows, key=lambda r: r.delta_chi2)

    out = {
        "lambda": float(lam),
        "base_chi2": float(base["chi2"]),
        "best_scan_k": float(best_scan.k),
        "best_scan_n": float(best_scan.n_eff),
        "best_scan_delta": float(best_scan.delta_chi2),
        "n_wells": int(len(wells)),
        "n_triplets": int(len(triplets)),
    }

    if best_mobile:
        out.update({
            "mobile_score": float(best_mobile.mobile_score),
            "a_scale": float(best_mobile.a_scale),
            "scale_error": float(best_mobile.scale_error),
            "koide_error": float(best_mobile.koide_error),
            "Q_mean": float(best_mobile.Q_mean),
            "Q_low": float(best_mobile.Q_low),
            "Q_high": float(best_mobile.Q_high),
            "triplet_n1": float(best_mobile.n1),
            "triplet_n2": float(best_mobile.n2),
            "triplet_n3": float(best_mobile.n3),
            "triplet_k1": float(best_mobile.k1),
            "triplet_k2": float(best_mobile.k2),
            "triplet_k3": float(best_mobile.k3),
            "delta1": float(best_mobile.delta1),
            "delta2": float(best_mobile.delta2),
            "delta3": float(best_mobile.delta3),
            "mean_delta": float(best_mobile.mean_delta),
            "phase1": float(best_mobile.phase1),
            "phase2": float(best_mobile.phase2),
            "phase3": float(best_mobile.phase3),
            "mean_phase": float(best_mobile.mean_phase),
        })
    else:
        for key in [
            "mobile_score", "a_scale", "scale_error", "koide_error", "Q_mean", "Q_low", "Q_high",
            "triplet_n1", "triplet_n2", "triplet_n3", "triplet_k1", "triplet_k2", "triplet_k3",
            "delta1", "delta2", "delta3", "mean_delta", "phase1", "phase2", "phase3", "mean_phase",
        ]:
            out[key] = float("nan")

    return out, scan_rows, wells, triplets


def trajectory_score(traj_df: pd.DataFrame, cfg: Config) -> Dict[str, float]:
    """
    Trajectory statistic.

    Plain terms:
        Rewards high mobile score and Q stability.
        Penalizes rough/jumpy a(lambda), phase(lambda), and branch location.

    This is not a discovery statistic by itself. It is a coherence score whose
    null distribution is computed by the same lambda ladder.
    """
    df = traj_df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mobile_score", "a_scale", "koide_error", "Q_mean", "mean_phase", "triplet_n2"])

    if len(df) < 4:
        return {
            "T_traj": float("-inf"),
            "mean_mobile_score": float("nan"),
            "mean_koide_error": float("nan"),
            "a_roughness": float("nan"),
            "phase_roughness": float("nan"),
            "branch_roughness": float("nan"),
            "n_points": int(len(df)),
        }

    a = df["a_scale"].to_numpy(float)
    phi = df["mean_phase"].to_numpy(float)
    n2 = df["triplet_n2"].to_numpy(float)
    score = df["mobile_score"].to_numpy(float)
    epsK = df["koide_error"].to_numpy(float)

    da = np.diff(a)
    dphi = wrapped_phase_diff(phi)
    dn2 = np.diff(n2)

    a_rough = float(np.sqrt(np.mean(da * da)))
    phi_rough = float(np.sqrt(np.mean(dphi * dphi)))
    branch_rough = float(np.sqrt(np.mean(dn2 * dn2)))

    mean_score = float(np.mean(score))
    mean_epsK = float(np.mean(epsK))
    median_score = float(np.median(score))

    # Coherence statistic.
    numerator = cfg.traj_score_weight * mean_score
    denominator = (
        1.0
        + cfg.traj_q_weight * mean_epsK
        + cfg.traj_scale_jump_weight * a_rough
        + cfg.traj_phase_jump_weight * phi_rough
        + cfg.traj_branch_jump_weight * branch_rough
    )
    T = float(numerator / max(denominator, cfg.eps))

    return {
        "T_traj": T,
        "mean_mobile_score": mean_score,
        "median_mobile_score": median_score,
        "mean_koide_error": mean_epsK,
        "a_roughness": a_rough,
        "phase_roughness": phi_rough,
        "branch_roughness": branch_rough,
        "n_points": int(len(df)),
        "a_min": float(np.min(a)),
        "a_max": float(np.max(a)),
        "a_mean": float(np.mean(a)),
        "Q_mean_mean": float(np.mean(df["Q_mean"].to_numpy(float))),
    }


def empirical_p_ge(value: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=float)
    return float((1.0 + np.sum(null_values >= value)) / (len(null_values) + 1.0))


def run_lambda_ladder(ell, h_sig, h_side, alpha, cfg: Config, xp, k_grid: np.ndarray, lambdas: np.ndarray):
    traj_rows = []
    all_wells = []
    all_triplets = []

    for lam in lambdas:
        row, scan_rows, wells, triplets = analyze_one_lambda(lam, ell, h_sig, h_side, alpha, cfg, xp, k_grid)
        traj_rows.append(row)

        for w in wells:
            d = asdict(w)
            d["lambda"] = float(lam)
            all_wells.append(d)

        for t in triplets:
            d = asdict(t)
            d["lambda"] = float(lam)
            all_triplets.append(d)

    traj_df = pd.DataFrame(traj_rows)
    wells_df = pd.DataFrame(all_wells)
    triplets_df = pd.DataFrame(all_triplets)

    score = trajectory_score(traj_df, cfg)
    return traj_df, wells_df, triplets_df, score


def run_null_trajectories(ell, h_sig, h_side, alpha, cfg: Config, xp, k_grid: np.ndarray, lambdas: np.ndarray):
    rng = np.random.default_rng(cfg.seed)

    null_rows = []

    print(f"[null] running {cfg.n_null} lambda-trajectory nulls")
    for j in range(cfg.n_null):
        if (j + 1) % max(1, cfg.n_null // 10) == 0:
            print(f"  [null] {j + 1}/{cfg.n_null}")

        traj_rows = []
        for lam in lambdas:
            var = np.maximum(h_sig + (lam * alpha) ** 2 * h_side, 1.0)
            y0 = rng.normal(0.0, np.sqrt(var), size=len(var))
            row, _, _, _ = analyze_one_lambda(lam, ell, h_sig, h_side, alpha, cfg, xp, k_grid, y_override=y0, var_override=var)
            traj_rows.append(row)

        traj_df = pd.DataFrame(traj_rows)
        sc = trajectory_score(traj_df, cfg)
        sc["null_index"] = int(j)
        null_rows.append(sc)

    return pd.DataFrame(null_rows)


# ============================================================
# Plots
# ============================================================

def make_plots(outdir: Path, traj_df: pd.DataFrame, null_df: pd.DataFrame, obs_score: Dict[str, float]):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable; skipping plots. Reason: {exc}")
        return

    # a(lambda)
    plt.figure(figsize=(10, 5))
    plt.plot(traj_df["lambda"], traj_df["a_scale"], marker="o")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$a$ in $n \approx a(10,15,20)$")
    plt.title("Mobile Koide scale trajectory")
    plt.tight_layout()
    plt.savefig(outdir / "lambda_scale_a.png", dpi=160)
    plt.close()

    # Q(lambda)
    plt.figure(figsize=(10, 5))
    plt.plot(traj_df["lambda"], traj_df["Q_mean"], marker="o")
    plt.axhline(2.0/3.0, linestyle="--", linewidth=1)
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$Q_{\rm mean}$")
    plt.title("Koide ratio trajectory")
    plt.tight_layout()
    plt.savefig(outdir / "lambda_Q_mean.png", dpi=160)
    plt.close()

    # score(lambda)
    plt.figure(figsize=(10, 5))
    plt.plot(traj_df["lambda"], traj_df["mobile_score"], marker="o")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("mobile scaled-Koide score")
    plt.title("Mobile score trajectory")
    plt.tight_layout()
    plt.savefig(outdir / "lambda_mobile_score.png", dpi=160)
    plt.close()

    # phase(lambda)
    plt.figure(figsize=(10, 5))
    plt.plot(traj_df["lambda"], traj_df["mean_phase"], marker="o")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("mean triplet phase")
    plt.title("Mean phase trajectory")
    plt.tight_layout()
    plt.savefig(outdir / "lambda_mean_phase.png", dpi=160)
    plt.close()

    # null score
    if "T_traj" in null_df.columns:
        plt.figure(figsize=(10, 5))
        plt.hist(null_df["T_traj"].dropna().to_numpy(), bins=60, alpha=0.8)
        plt.axvline(obs_score["T_traj"], linestyle="--", linewidth=2)
        plt.xlabel(r"$T_{\rm traj}$")
        plt.ylabel("null count")
        plt.title("Trajectory null distribution")
        plt.tight_layout()
        plt.savefig(outdir / "trajectory_null.png", dpi=160)
        plt.close()


# ============================================================
# Main
# ============================================================

def run(cfg: Config, pattern: Optional[str], use_gpu: bool, plot: bool):
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    xp, used_gpu = get_backend(use_gpu)

    print("=" * 100)
    print("MILESTONE 3A: DYNAMIC SCALE-PHASE INVARIANT TEST")
    print("=" * 100)
    print(f"[backend] {'CuPy' if used_gpu else 'NumPy'}")
    print(f"[config] active intervals = {active_intervals(cfg)}")
    print(f"[config] Delta ell active = {active_delta_ell(cfg):.10f}")
    print(f"[config] lambda range = [{cfg.lambda_min}, {cfg.lambda_max}], steps={cfg.lambda_steps}")
    print(f"[config] k scan = [{cfg.k_min}, {cfg.k_max}], steps={cfg.k_steps}")
    print(f"[config] n_null = {cfg.n_null}, seed={cfg.seed}")
    print("=" * 100)

    files = find_files(pattern)
    df, provenance = load_all_events(files, cfg)

    sig = select_region(df, (cfg.b_signal_low, cfg.b_signal_high), cfg)
    low = select_region(df, (cfg.b_low_sideband_low, cfg.b_low_sideband_high), cfg)
    high = select_region(df, (cfg.b_high_sideband_low, cfg.b_high_sideband_high), cfg)

    print("\n[event counts after active support]")
    print(f"  signal:          {len(sig):,}")
    print(f"  B-low sideband:  {len(low):,}")
    print(f"  B-high sideband: {len(high):,}")

    if len(sig) < 100 or (len(low) + len(high)) < 100:
        raise RuntimeError("Too few events for trajectory test.")

    ell, q2_centers, h_sig = make_histogram(sig["q2"].to_numpy(), cfg)
    _, _, h_low = make_histogram(low["q2"].to_numpy(), cfg)
    _, _, h_high = make_histogram(high["q2"].to_numpy(), cfg)

    h_side = h_low + h_high
    alpha = float(np.sum(h_sig) / max(np.sum(h_side), 1.0))

    lambdas = np.linspace(cfg.lambda_min, cfg.lambda_max, cfg.lambda_steps)
    k_grid = np.linspace(cfg.k_min, cfg.k_max, cfg.k_steps)

    print("\n[sideband scale]")
    print(f"  alpha = {alpha:.9f}")
    print(f"  sum signal = {np.sum(h_sig):.1f}")
    print(f"  sum side = {np.sum(h_side):.1f}")

    print("\n[observed] running lambda ladder")
    traj_df, wells_df, triplets_df, obs_score = run_lambda_ladder(ell, h_sig, h_side, alpha, cfg, xp, k_grid, lambdas)

    traj_csv = outdir / "lambda_trajectory.csv"
    wells_csv = outdir / "lambda_wells.csv"
    triplets_csv = outdir / "lambda_triplets.csv"

    traj_df.to_csv(traj_csv, index=False)
    wells_df.to_csv(wells_csv, index=False)
    triplets_df.to_csv(triplets_csv, index=False)

    print("\n[observed trajectory]")
    print(traj_df[[
        "lambda", "mobile_score", "a_scale", "scale_error", "koide_error", "Q_mean",
        "triplet_n1", "triplet_n2", "triplet_n3", "mean_phase"
    ]].to_string(index=False))

    print("\n[observed trajectory score]")
    print(json.dumps(obs_score, indent=2))

    null_df = run_null_trajectories(ell, h_sig, h_side, alpha, cfg, xp, k_grid, lambdas)
    null_csv = outdir / "null_trajectory_scores.csv"
    null_df.to_csv(null_csv, index=False)

    p_traj = empirical_p_ge(obs_score["T_traj"], null_df["T_traj"].to_numpy(float))

    verdict = {
        "T_traj": float(obs_score["T_traj"]),
        "p_traj": float(p_traj),
        "observed_score_components": obs_score,
        "lambda_min": float(cfg.lambda_min),
        "lambda_max": float(cfg.lambda_max),
        "lambda_steps": int(cfg.lambda_steps),
        "flags": {
            "trajectory_survives_0p05": bool(p_traj <= cfg.p_weak),
            "trajectory_survives_0p01": bool(p_traj <= cfg.p_strong),
            "milestone3_pass": bool(p_traj <= cfg.p_weak),
        },
    }

    if verdict["flags"]["trajectory_survives_0p01"]:
        interpretation = (
            "Milestone 3A strong pass: the mobile scaled-Koide branch follows a coherent "
            "lambda-trajectory at p <= 0.01 under the matched trajectory null."
        )
    elif verdict["flags"]["trajectory_survives_0p05"]:
        interpretation = (
            "Milestone 3A pass: the mobile scaled-Koide branch follows a coherent "
            "lambda-trajectory at p <= 0.05 under the matched trajectory null."
        )
    else:
        interpretation = (
            "Milestone 3A not established: the observed mobile scaled-Koide trajectory is "
            "not rare under the matched lambda-trajectory null."
        )

    summary = {
        "script": "milestone3_dynamic_scale_phase_cupy.py",
        "purpose": "Dynamic lambda-ladder scale/phase invariant test.",
        "backend": "CuPy" if used_gpu else "NumPy",
        "files": files,
        "provenance": provenance,
        "config": asdict(cfg),
        "active_intervals": active_intervals(cfg),
        "delta_ell_active": active_delta_ell(cfg),
        "counts": {
            "signal_active_events": int(len(sig)),
            "B_low_active_events": int(len(low)),
            "B_high_active_events": int(len(high)),
            "hist_signal_sum": float(np.sum(h_sig)),
            "hist_side_sum": float(np.sum(h_side)),
            "alpha": float(alpha),
        },
        "outputs": {
            "trajectory_csv": str(traj_csv),
            "wells_csv": str(wells_csv),
            "triplets_csv": str(triplets_csv),
            "null_csv": str(null_csv),
        },
        "verdict": verdict,
        "interpretation": interpretation,
    }

    summary_json = outdir / "milestone3_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = outdir / "milestone3_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Milestone 3A: Dynamic Scale-Phase Invariant Test\n")
        f.write("================================================\n\n")
        f.write(f"backend: {'CuPy' if used_gpu else 'NumPy'}\n")
        f.write(f"Delta ell active: {active_delta_ell(cfg):.10f}\n")
        f.write(f"alpha: {alpha:.9f}\n")
        f.write(f"lambda range: [{cfg.lambda_min}, {cfg.lambda_max}], steps={cfg.lambda_steps}\n\n")
        f.write("Observed trajectory score\n")
        f.write("-------------------------\n")
        for k, v in obs_score.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\np_traj: {p_traj}\n\n")
        f.write("Flags\n")
        f.write("-----\n")
        for k, v in verdict["flags"].items():
            f.write(f"{k}: {v}\n")
        f.write("\nInterpretation\n")
        f.write("--------------\n")
        f.write(interpretation + "\n")

    if plot:
        make_plots(outdir, traj_df, null_df, obs_score)

    print("\n" + "=" * 100)
    print("MILESTONE 3A VERDICT")
    print("=" * 100)
    print(json.dumps(verdict, indent=2))
    print("\nInterpretation:")
    print(interpretation)

    print("\nSaved:")
    for p in [traj_csv, wells_csv, triplets_csv, null_csv, summary_json, report_path]:
        print(" ", p)


def parse_args():
    p = argparse.ArgumentParser(description="Milestone 3A dynamic scale-phase invariant lambda-ladder test.")

    p.add_argument("--pattern", default=None, help='ROOT glob, e.g. "data/*.root"')
    p.add_argument("--outdir", default="outputs_milestone3_lambda_trajectory")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--plot", action="store_true")

    p.add_argument("--n-null", type=int, default=1000)
    p.add_argument("--seed", type=int, default=314159)

    p.add_argument("--lambda-min", type=float, default=0.0)
    p.add_argument("--lambda-max", type=float, default=1.25)
    p.add_argument("--lambda-steps", type=int, default=26)

    p.add_argument("--n-bins", type=int, default=240)
    p.add_argument("--k-min", type=float, default=6.0)
    p.add_argument("--k-max", type=float, default=32.0)
    p.add_argument("--k-steps", type=int, default=1301)

    p.add_argument("--b-signal-low", type=float, default=5230.0)
    p.add_argument("--b-signal-high", type=float, default=5330.0)
    p.add_argument("--b-low-sideband-low", type=float, default=5000.0)
    p.add_argument("--b-low-sideband-high", type=float, default=5180.0)
    p.add_argument("--b-high-sideband-low", type=float, default=5380.0)
    p.add_argument("--b-high-sideband-high", type=float, default=5600.0)

    p.add_argument("--mobile-q-weight", type=float, default=25.0)
    p.add_argument("--mobile-scale-weight", type=float, default=0.25)
    p.add_argument("--mobile-integer-weight", type=float, default=0.0)

    p.add_argument("--traj-q-weight", type=float, default=25.0)
    p.add_argument("--traj-scale-jump-weight", type=float, default=10.0)
    p.add_argument("--traj-phase-jump-weight", type=float, default=1.0)
    p.add_argument("--traj-branch-jump-weight", type=float, default=0.25)
    p.add_argument("--traj-score-weight", type=float, default=1.0)

    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config(
        outdir=args.outdir,
        n_null=args.n_null,
        seed=args.seed,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        lambda_steps=args.lambda_steps,
        n_bins=args.n_bins,
        k_min=args.k_min,
        k_max=args.k_max,
        k_steps=args.k_steps,
        b_signal_low=args.b_signal_low,
        b_signal_high=args.b_signal_high,
        b_low_sideband_low=args.b_low_sideband_low,
        b_low_sideband_high=args.b_low_sideband_high,
        b_high_sideband_low=args.b_high_sideband_low,
        b_high_sideband_high=args.b_high_sideband_high,
        mobile_q_weight=args.mobile_q_weight,
        mobile_scale_weight=args.mobile_scale_weight,
        mobile_integer_weight=args.mobile_integer_weight,
        traj_q_weight=args.traj_q_weight,
        traj_scale_jump_weight=args.traj_scale_jump_weight,
        traj_phase_jump_weight=args.traj_phase_jump_weight,
        traj_branch_jump_weight=args.traj_branch_jump_weight,
        traj_score_weight=args.traj_score_weight,
    )

    run(cfg, pattern=args.pattern, use_gpu=args.gpu, plot=args.plot)


if __name__ == "__main__":
    main()

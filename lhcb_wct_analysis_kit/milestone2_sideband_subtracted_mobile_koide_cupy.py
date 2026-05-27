#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Milestone 2: Sideband-Subtracted Signal Isolation Test
=====================================================

Purpose
-------
Test whether the log-periodic / active-domain winding / Koide-comb structure
survives after subtracting B-mass sideband shape from the B-signal-window
candidate spectrum.

This is the practical next step after the smooth-Poisson global null.

Core subtraction
----------------
For each log-q2 bin i:

    R_i = N_sig,i - alpha * N_side,i

where

    N_side,i = N_Blow,i + N_Bhigh,i
    alpha = sum_i N_sig,i / sum_i N_side,i

Variance approximation:

    Var(R_i) = N_sig,i + alpha^2 N_side,i

Because R_i can be negative, the fit uses weighted least squares rather than
Poisson likelihood.

Main outputs
------------
outputs_milestone2_sideband/
    sideband_subtracted_bins.csv
    sideband_subtracted_scan.csv
    sideband_subtracted_wells.csv
    sideband_subtracted_triplets.csv
    sideband_subtracted_integer_scan.csv
    sideband_subtracted_comb_tests.csv
    null_scores.csv
    milestone2_summary.json
    milestone2_report.txt
    optional plots

Run
---
    python milestone2_sideband_subtracted_cupy.py --pattern "data/*.root" --n-null 1000 --gpu --plot

Fast test:
    python milestone2_sideband_subtracted_cupy.py --pattern "data/*.root" --n-null 100 --gpu

Interpretation
--------------
This updated test separates two hypotheses:

1. fixed-branch survival:
       fixed n=15 or fixed (10,15,20) survives sideband subtraction.

2. mobile-ratio survival:
       the absolute n-location moves, but a scaled Koide-like triplet
       a*(10,15,20) remains unusually sharp and strong.

A Milestone 2 fixed-branch upgrade occurs if:

    p_n15_fixed <= 0.05
        or
    p_comb_101520 <= 0.05

A mobile-Koide upgrade occurs if:

    p_mobile_scaled_koide <= 0.05

This is the correct test when the n-values move but the ratio geometry may survive.
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
    outdir: str = "outputs_milestone2_sideband"

    q2_min: float = 0.1
    q2_max: float = 19.0

    # Charm-veto active support.
    jpsi_low: float = 8.0
    jpsi_high: float = 11.0
    psi2s_low: float = 12.5
    psi2s_high: float = 14.5

    # B0/K* windows.
    b_signal_low: float = 5230.0
    b_signal_high: float = 5330.0
    b_low_sideband_low: float = 5000.0
    b_low_sideband_high: float = 5180.0
    b_high_sideband_low: float = 5380.0
    b_high_sideband_high: float = 5600.0

    kst_low: float = 795.9
    kst_high: float = 995.9

    # Histogram/scan.
    n_bins: int = 240
    k1_fixed: float = 7.61054
    k_ref: float = 19.5296
    k_min: float = 6.0
    k_max: float = 32.0
    k_steps: int = 1301

    # Integer scan.
    n_min: int = 10
    n_max: int = 22

    # Koide targets.
    koide_q: float = 2.0 / 3.0

    # Well-first.
    min_peak_prominence: float = 0.5
    min_peak_distance_k: float = 0.75
    max_wells_for_triplets: int = 12

    # Nulls.
    n_null: int = 1000
    seed: int = 271828

    # Verdict thresholds.
    p_weak: float = 0.05
    p_strong: float = 0.01

    # Mobile scaled-Koide statistic:
    # triplet n is compared to a*(10,15,20), where a is fitted.
    mobile_q_weight: float = 25.0
    mobile_scale_weight: float = 0.25
    mobile_integer_weight: float = 0.0

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


# ============================================================
# ROOT loading helpers
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
    plus_patterns = [
        "muplus", "mu_plus", "mup", "mu_p", "muplus0", "muplus_0",
        "MuPlus", "mup_0", "mu1", "muplus_1",
    ]
    minus_patterns = [
        "muminus", "mu_minus", "mum", "mu_m", "muminus0", "muminus_0",
        "MuMinus", "mum_0", "mu2", "muminus_1",
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
    return q2_mev2 / 1.0e6, {k: str(v) for k, v in branches.items()}


def load_all_events(files: List[str], cfg: Config) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    import uproot

    q2_candidates = [
        "q2", "Q2", "q2_DTF", "Q2_DTF",
        "mumu_M2", "dimuon_M2", "Jpsi_M2", "J_psi_1S_M2", "J_psi_1S_MM2",
    ]
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
                q2, used_mu = derive_q2_from_muons(tree, keys)
                arr = tree.arrays([b_branch, kst_branch], library="np")
                q2_source = "derived_from_muon_four_vectors"
                print("[q2] derived from muon four-vectors")

            bm = np.asarray(arr[b_branch], dtype=float)
            km = np.asarray(arr[kst_branch], dtype=float)

            mask = np.isfinite(q2) & np.isfinite(bm) & np.isfinite(km)
            mask &= (q2 >= cfg.q2_min) & (q2 <= cfg.q2_max)

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
        raise RuntimeError("No event rows loaded.")

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
# Weighted least squares on residual
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
    idx = 1
    if include_k1:
        amps["A_k1"] = float(math.hypot(beta[idx], beta[idx + 1]))
        idx += 2

    for k in ks_extra:
        amps[f"A_k_{k:.6f}"] = float(math.hypot(beta[idx], beta[idx + 1]))
        amps[f"phi_k_{k:.6f}"] = float(math.atan2(-beta[idx + 1], beta[idx]))
        idx += 2

    return {
        "chi2": chi2,
        "beta": beta,
        "pred": pred,
        "amps": amps,
        "ndof": int(len(y) - len(beta)),
    }


def wls_delta_for_many_k_gpu(ell_np, y_np, var_np, k_grid_np, cfg: Config, xp):
    """
    Fast vectorized scan for add-one mode over many k.
    Uses exact weighted residual projection after base model is fit with NumPy.

    For each candidate pair cos(k ell), sin(k ell), computes weighted LS
    improvement over base by solving the 2x2 residual projection problem.
    """
    base = wls_fit_np(ell_np, y_np, var_np, [], cfg, include_k1=True)
    base_chi2 = base["chi2"]

    # Weighted residual after base model.
    y = xp.asarray(y_np, dtype=float)
    ell = xp.asarray(ell_np, dtype=float)
    var = xp.maximum(xp.asarray(var_np, dtype=float), 1.0)
    pred0 = xp.asarray(base["pred"], dtype=float)
    r = (y - pred0) / xp.sqrt(var)

    k_grid = xp.asarray(k_grid_np, dtype=float).reshape(-1, 1)
    ell_row = ell.reshape(1, -1)

    c = xp.cos(k_grid * ell_row) / xp.sqrt(var).reshape(1, -1)
    s = xp.sin(k_grid * ell_row) / xp.sqrt(var).reshape(1, -1)

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


def scan_one_mode(ell, y, var, k_grid, cfg: Config, xp):
    base, deltas, amps, phases = wls_delta_for_many_k_gpu(ell, y, var, k_grid, cfg, xp)
    rows = []
    for k, d, a, ph in zip(k_grid, deltas, amps, phases):
        rows.append(ScanRow(
            k=float(k),
            n_eff=n_from_k(float(k), cfg),
            delta_chi2=float(d),
            amp=float(a),
            phase=float(ph),
        ))
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

    peaks, props = find_peaks(
        y,
        prominence=cfg.min_peak_prominence,
        distance=min_dist_bins,
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


def triplets_from_wells(wells: List[WellRow], cfg: Config) -> List[TripletRow]:
    if len(wells) < 3:
        return []

    candidates = sorted(wells[:cfg.max_wells_for_triplets], key=lambda w: w.n_eff)
    triplets = []

    for w1, w2, w3 in combinations(candidates, 3):
        n1, n2, n3 = w1.n_eff, w2.n_eff, w3.n_eff
        if n2 <= 0:
            continue

        q_low = n1 / n2
        q_high = n3 / (2.0 * n2)
        q_mean = 0.5 * (q_low + q_high)
        koide_error = math.sqrt((q_low - cfg.koide_q) ** 2 + (q_high - cfg.koide_q) ** 2)

        integer_error = math.sqrt((n1 - 10.0) ** 2 + (n2 - 15.0) ** 2 + (n3 - 20.0) ** 2)

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



def scaled_koide_geometry(ns: List[float], cfg: Config) -> Dict[str, float]:
    """
    Compute mobile scaled-Koide geometry for a triplet.

    Plain definitions:
        ns:
            Three active-domain winding locations.

        a:
            Best-fit scale mapping (10,15,20) to ns.

        scale_error:
            Root-mean-square residual after fitting a.

        koide_error:
            Ratio error independent of absolute placement.

    Symbolic form:
        a* = <n,t>/<t,t>, t=(10,15,20)
        eps_a = sqrt(mean((n-a*t)^2))
        eps_K = sqrt((n1/n2-2/3)^2 + (n3/(2 n2)-2/3)^2)
    """
    if len(ns) != 3:
        return {
            "a_scale": float("nan"),
            "scale_error": float("nan"),
            "koide_error": float("nan"),
            "q_low": float("nan"),
            "q_high": float("nan"),
            "q_mean": float("nan"),
            "scaled_integer_error": float("nan"),
        }

    n = np.array(sorted(ns), dtype=float)
    t = np.array([10.0, 15.0, 20.0], dtype=float)

    denom = float(np.dot(t, t))
    a = float(np.dot(n, t) / denom) if denom > 0 else float("nan")
    resid = n - a * t
    scale_error = float(np.sqrt(np.mean(resid * resid)))

    n1, n2, n3 = n
    if n2 <= 0:
        q_low = q_high = q_mean = koide_error = float("nan")
    else:
        q_low = float(n1 / n2)
        q_high = float(n3 / (2.0 * n2))
        q_mean = float(0.5 * (q_low + q_high))
        koide_error = float(math.sqrt((q_low - cfg.koide_q) ** 2 + (q_high - cfg.koide_q) ** 2))

    # This is absolute distance to unscaled (10,15,20), retained only as a diagnostic.
    integer_error = float(np.linalg.norm(n - t))

    return {
        "a_scale": a,
        "scale_error": scale_error,
        "koide_error": koide_error,
        "q_low": q_low,
        "q_high": q_high,
        "q_mean": q_mean,
        "scaled_integer_error": integer_error,
    }


def mobile_scaled_koide_score_from_triplet(triplet: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """
    Mobile scaled-Koide score.

    The statistic rewards:
        - strong mean triplet power
        - small Koide ratio error
        - small scale residual relative to a*(10,15,20)

    It does NOT require n2=15.
    """
    ns = [float(triplet["n1"]), float(triplet["n2"]), float(triplet["n3"])]
    geom = scaled_koide_geometry(ns, cfg)
    mean_delta = float((triplet["delta1"] + triplet["delta2"] + triplet["delta3"]) / 3.0)

    denom = (
        1.0
        + cfg.mobile_q_weight * geom["koide_error"]
        + cfg.mobile_scale_weight * geom["scale_error"]
        + cfg.mobile_integer_weight * geom["scaled_integer_error"]
    )
    score = float(mean_delta / max(denom, cfg.eps))

    out = dict(triplet)
    out.update(geom)
    out["mobile_scaled_koide_score"] = score
    out["mobile_mean_delta"] = mean_delta
    return out


def best_mobile_scaled_koide_from_wells(wells: List[WellRow], cfg: Config) -> Optional[Dict[str, Any]]:
    """
    Search all top-well triplets for the best mobile scaled-Koide score.
    This is the moving-n version of the sideband-subtracted Koide test.
    """
    triplets = triplets_from_wells(wells, cfg)
    if not triplets:
        return None

    scored = [mobile_scaled_koide_score_from_triplet(asdict(t), cfg) for t in triplets]
    scored.sort(key=lambda d: d["mobile_scaled_koide_score"], reverse=True)
    return scored[0]


def mobile_scaled_koide_score_from_scan_rows(scan_rows: List[ScanRow], cfg: Config) -> Optional[Dict[str, Any]]:
    wells = find_wells(scan_rows, cfg)
    return best_mobile_scaled_koide_from_wells(wells, cfg)



def delta_for_fixed_k(ell, y, var, k, cfg: Config) -> float:
    base = wls_fit_np(ell, y, var, [], cfg, include_k1=True)
    fit = wls_fit_np(ell, y, var, [float(k)], cfg, include_k1=True)
    return float(base["chi2"] - fit["chi2"])


def comb_fit_delta(ell, y, var, ns, cfg: Config):
    ks = [k_from_n(float(n), cfg) for n in ns]
    base = wls_fit_np(ell, y, var, [], cfg, include_k1=True)
    fit = wls_fit_np(ell, y, var, ks, cfg, include_k1=True)
    return float(base["chi2"] - fit["chi2"]), ks, fit


def empirical_p_ge(value: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=float)
    return float((1.0 + np.sum(null_values >= value)) / (len(null_values) + 1.0))


def run_nulls(ell, var, k_grid, cfg: Config, xp):
    """
    Gaussian residual null for sideband-subtracted residual.
    """
    rng = np.random.default_rng(cfg.seed)

    max_null = []
    kref_null = []
    n15_null = []
    comb_101520_null = []
    folded_449_null = []
    mobile_scaled_koide_null = []
    mobile_scaled_koide_a_null = []
    mobile_scaled_koide_error_null = []

    k15 = k_from_n(15.0, cfg)

    print(f"[null] running {cfg.n_null} Gaussian residual nulls")

    for j in range(cfg.n_null):
        if (j + 1) % max(1, cfg.n_null // 10) == 0:
            print(f"  [null] {j + 1}/{cfg.n_null}")

        y0 = rng.normal(0.0, np.sqrt(np.maximum(var, 1.0)), size=len(var))

        _, rows0 = scan_one_mode(ell, y0, var, k_grid, cfg, xp)
        vals = np.array([r.delta_chi2 for r in rows0], dtype=float)
        max_null.append(float(np.max(vals)))

        mobile0 = mobile_scaled_koide_score_from_scan_rows(rows0, cfg)
        if mobile0 is None:
            mobile_scaled_koide_null.append(float("-inf"))
            mobile_scaled_koide_a_null.append(float("nan"))
            mobile_scaled_koide_error_null.append(float("nan"))
        else:
            mobile_scaled_koide_null.append(float(mobile0["mobile_scaled_koide_score"]))
            mobile_scaled_koide_a_null.append(float(mobile0["a_scale"]))
            mobile_scaled_koide_error_null.append(float(mobile0["koide_error"]))

        kref_null.append(delta_for_fixed_k(ell, y0, var, cfg.k_ref, cfg))
        n15_null.append(delta_for_fixed_k(ell, y0, var, k15, cfg))
        comb_101520_null.append(comb_fit_delta(ell, y0, var, [10.0, 15.0, 20.0], cfg)[0])
        folded_449_null.append(comb_fit_delta(ell, y0, var, [6.6666666667, 15.0, 13.3333333333], cfg)[0])

    return {
        "scanmax": np.array(max_null, dtype=float),
        "kref": np.array(kref_null, dtype=float),
        "n15": np.array(n15_null, dtype=float),
        "comb_101520": np.array(comb_101520_null, dtype=float),
        "folded_449": np.array(folded_449_null, dtype=float),
        "mobile_scaled_koide": np.array(mobile_scaled_koide_null, dtype=float),
        "mobile_scaled_koide_a": np.array(mobile_scaled_koide_a_null, dtype=float),
        "mobile_scaled_koide_error": np.array(mobile_scaled_koide_error_null, dtype=float),
    }


# ============================================================
# Plots
# ============================================================

def make_plots(outdir: Path, bins_df: pd.DataFrame, scan_df: pd.DataFrame, summary: Dict[str, Any], cfg: Config):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable; skipping plots. Reason: {exc}")
        return

    plt.figure(figsize=(10, 5))
    plt.errorbar(
        bins_df["q2_center"],
        bins_df["R_subtracted"],
        yerr=np.sqrt(bins_df["variance"]),
        fmt="o",
        markersize=3,
        linewidth=1,
    )
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel(r"$q^2$")
    plt.ylabel(r"$R_i=N_{\rm sig}-\alpha N_{\rm side}$")
    plt.title("Sideband-subtracted residual")
    plt.tight_layout()
    plt.savefig(outdir / "sideband_subtracted_residual.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(scan_df["k"], scan_df["delta_chi2"], linewidth=1.5)
    plt.axvline(cfg.k_ref, linestyle="--", linewidth=1, label="k_ref")
    plt.axvline(k_from_n(15.0, cfg), linestyle=":", linewidth=1.5, label="n=15")
    plt.xlabel("k")
    plt.ylabel(r"$\Delta \chi^2$")
    plt.title("Sideband-subtracted one-mode scan")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "sideband_subtracted_scan.png", dpi=160)
    plt.close()


# ============================================================
# Main analysis
# ============================================================

def run(cfg: Config, pattern: Optional[str], use_gpu: bool, plot: bool):
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    xp, used_gpu = get_backend(use_gpu)

    print("=" * 100)
    print("MILESTONE 2: SIDEBAND-SUBTRACTED SIGNAL ISOLATION")
    print("=" * 100)
    print(f"[backend] {'CuPy' if used_gpu else 'NumPy'}")
    print(f"[config] active intervals = {active_intervals(cfg)}")
    print(f"[config] Delta ell active = {active_delta_ell(cfg):.10f}")
    print(f"[config] k_ref = {cfg.k_ref:.6f}")
    print(f"[config] k(n=10,15,20) = {k_from_n(10,cfg):.6f}, {k_from_n(15,cfg):.6f}, {k_from_n(20,cfg):.6f}")
    print(f"[config] scan k=[{cfg.k_min}, {cfg.k_max}], steps={cfg.k_steps}")
    print(f"[config] n_null={cfg.n_null}, seed={cfg.seed}")
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
        raise RuntimeError("Too few events for sideband-subtracted test.")

    ell, q2_centers, h_sig = make_histogram(sig["q2"].to_numpy(), cfg)
    _, _, h_low = make_histogram(low["q2"].to_numpy(), cfg)
    _, _, h_high = make_histogram(high["q2"].to_numpy(), cfg)

    h_side = h_low + h_high
    alpha = float(np.sum(h_sig) / max(np.sum(h_side), 1.0))

    residual = h_sig - alpha * h_side
    variance = np.maximum(h_sig + alpha * alpha * h_side, 1.0)
    z = residual / np.sqrt(variance)

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
        "z_residual": z,
    })
    bins_csv = outdir / "sideband_subtracted_bins.csv"
    bins_df.to_csv(bins_csv, index=False)

    print("\n[sideband subtraction]")
    print(f"  alpha = {alpha:.9f}")
    print(f"  sum signal = {np.sum(h_sig):.1f}")
    print(f"  sum side = {np.sum(h_side):.1f}")
    print(f"  sum residual = {np.sum(residual):.6f}")
    print(f"  RMS z residual = {np.sqrt(np.mean(z*z)):.6f}")

    k_grid = np.linspace(cfg.k_min, cfg.k_max, cfg.k_steps)

    print("\n[scan] sideband-subtracted residual")
    base_fit, scan_rows = scan_one_mode(ell, residual, variance, k_grid, cfg, xp)
    scan_df = pd.DataFrame([asdict(r) for r in scan_rows])
    scan_csv = outdir / "sideband_subtracted_scan.csv"
    scan_df.to_csv(scan_csv, index=False)

    best = scan_df.sort_values("delta_chi2", ascending=False).iloc[0].to_dict()

    delta_kref = delta_for_fixed_k(ell, residual, variance, cfg.k_ref, cfg)
    k15 = k_from_n(15.0, cfg)
    delta_n15 = delta_for_fixed_k(ell, residual, variance, k15, cfg)

    wells = find_wells(scan_rows, cfg)
    wells_df = pd.DataFrame([asdict(w) for w in wells])
    wells_csv = outdir / "sideband_subtracted_wells.csv"
    wells_df.to_csv(wells_csv, index=False)

    triplets = triplets_from_wells(wells, cfg)
    triplets_df = pd.DataFrame([asdict(t) for t in triplets])
    triplets_csv = outdir / "sideband_subtracted_triplets.csv"
    triplets_df.to_csv(triplets_csv, index=False)

    print(f"  base chi2 = {base_fit['chi2']:.6f}, ndof = {base_fit['ndof']}")
    print(f"  best k = {best['k']:.6f}, n = {best['n_eff']:.6f}, delta_chi2 = {best['delta_chi2']:.6f}")
    print(f"  k_ref = {cfg.k_ref:.6f}, delta_chi2 = {delta_kref:.6f}")
    print(f"  n=15 k = {k15:.6f}, delta_chi2 = {delta_n15:.6f}")

    print("\n[top wells]")
    if not wells_df.empty:
        print(wells_df.head(12).to_string(index=False))
    else:
        print("  none")

    print("\n[best well-first triplets]")
    if not triplets_df.empty:
        cols = ["n1", "n2", "n3", "Q_low", "Q_high", "Q_mean", "koide_error", "integer_error_10_15_20", "score"]
        print(triplets_df.head(10)[cols].to_string(index=False))
    else:
        print("  none")

    # Integer scan.
    integer_rows = []
    for n in range(cfg.n_min, cfg.n_max + 1):
        k = k_from_n(float(n), cfg)
        d = delta_for_fixed_k(ell, residual, variance, k, cfg)
        fit = wls_fit_np(ell, residual, variance, [k], cfg, include_k1=True)
        integer_rows.append({
            "n": int(n),
            "k": float(k),
            "delta_chi2": float(d),
            "amp": float(fit["amps"].get(f"A_k_{k:.6f}", np.nan)),
            "phase": float(fit["amps"].get(f"phi_k_{k:.6f}", np.nan)),
        })

    integer_df = pd.DataFrame(integer_rows)
    integer_csv = outdir / "sideband_subtracted_integer_scan.csv"
    integer_df.to_csv(integer_csv, index=False)

    # Comb tests.
    comb_specs = [
        ("koide_Q_2_3_true_sideband", [10.0, 15.0, 20.0]),
        ("folded_Q_4_9", [6.6666666667, 15.0, 13.3333333333]),
    ]
    comb_rows = []
    for name, ns in comb_specs:
        d, ks, fit = comb_fit_delta(ell, residual, variance, ns, cfg)
        comb_rows.append({
            "comb": name,
            "n_values": ns,
            "k_values": ks,
            "delta_chi2": float(d),
        })
    comb_df = pd.DataFrame(comb_rows)
    comb_csv = outdir / "sideband_subtracted_comb_tests.csv"
    comb_df.to_csv(comb_csv, index=False)

    print("\n[integer scan]")
    print(integer_df.to_string(index=False))
    print("\n[comb tests]")
    print(comb_df[["comb", "delta_chi2"]].to_string(index=False))

    # Null tests.
    nulls = run_nulls(ell, variance, k_grid, cfg, xp)
    null_df = pd.DataFrame({
        "scanmax": nulls["scanmax"],
        "kref": nulls["kref"],
        "n15": nulls["n15"],
        "comb_101520": nulls["comb_101520"],
        "folded_449": nulls["folded_449"],
        "mobile_scaled_koide": nulls["mobile_scaled_koide"],
        "mobile_scaled_koide_a": nulls["mobile_scaled_koide_a"],
        "mobile_scaled_koide_error": nulls["mobile_scaled_koide_error"],
    })
    null_csv = outdir / "null_scores.csv"
    null_df.to_csv(null_csv, index=False)

    d_comb_101520 = float(comb_df.loc[comb_df["comb"] == "koide_Q_2_3_true_sideband", "delta_chi2"].iloc[0])
    d_folded_449 = float(comb_df.loc[comb_df["comb"] == "folded_Q_4_9", "delta_chi2"].iloc[0])

    p_best_scanmax = empirical_p_ge(best["delta_chi2"], nulls["scanmax"])
    p_kref_fixed = empirical_p_ge(delta_kref, nulls["kref"])
    p_n15_fixed = empirical_p_ge(delta_n15, nulls["n15"])
    p_comb_101520 = empirical_p_ge(d_comb_101520, nulls["comb_101520"])
    p_folded_449 = empirical_p_ge(d_folded_449, nulls["folded_449"])

    best_triplet = asdict(triplets[0]) if triplets else None
    best_mobile_scaled_koide = best_mobile_scaled_koide_from_wells(wells, cfg)
    mobile_score_obs = (
        float(best_mobile_scaled_koide["mobile_scaled_koide_score"])
        if best_mobile_scaled_koide is not None
        else float("-inf")
    )
    p_mobile_scaled_koide = empirical_p_ge(mobile_score_obs, nulls["mobile_scaled_koide"])

    verdict = {
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

        "folded_449_delta_chi2": d_folded_449,
        "p_folded_449": p_folded_449,

        "best_triplet": best_triplet,

        "best_mobile_scaled_koide": best_mobile_scaled_koide,
        "mobile_scaled_koide_score": mobile_score_obs,
        "p_mobile_scaled_koide": p_mobile_scaled_koide,

        "flags": {
            "best_scan_survives_0p05": bool(p_best_scanmax <= cfg.p_weak),
            "kref_survives_0p05": bool(p_kref_fixed <= cfg.p_weak),
            "n15_survives_0p05": bool(p_n15_fixed <= cfg.p_weak),
            "comb_101520_survives_0p05": bool(p_comb_101520 <= cfg.p_weak),
            "folded_449_survives_0p05": bool(p_folded_449 <= cfg.p_weak),
            "mobile_scaled_koide_survives_0p05": bool(p_mobile_scaled_koide <= cfg.p_weak),
            "mobile_scaled_koide_survives_0p01": bool(p_mobile_scaled_koide <= cfg.p_strong),

            "milestone2_fixed_branch_upgrade": bool((p_n15_fixed <= cfg.p_weak) or (p_comb_101520 <= cfg.p_weak)),
            "milestone2_mobile_ratio_upgrade": bool(p_mobile_scaled_koide <= cfg.p_weak),
            "milestone2_strong_upgrade": bool(((p_n15_fixed <= cfg.p_weak) or (p_comb_101520 <= cfg.p_weak)) or (p_mobile_scaled_koide <= cfg.p_weak)),
            "milestone2_structural_survival": bool(p_best_scanmax <= cfg.p_weak),
        },
    }

    if verdict["flags"]["milestone2_fixed_branch_upgrade"]:
        interpretation = (
            "Milestone 2 fixed-branch upgrade: the sideband-subtracted residual preserves "
            "a locked n=15 or fixed Q=2/3 comb component at p <= 0.05."
        )
    elif verdict["flags"]["milestone2_mobile_ratio_upgrade"]:
        interpretation = (
            "Milestone 2 mobile-ratio upgrade: the fixed n=15 / fixed (10,15,20) branch is not isolated, "
            "but a moving scaled-Koide triplet a*(10,15,20) is significant at p <= 0.05."
        )
    elif verdict["flags"]["milestone2_structural_survival"]:
        interpretation = (
            "Mixed Milestone 2 result: a log-frequency structure survives sideband subtraction, "
            "but neither the fixed branch nor the mobile scaled-Koide ratio is isolated at p <= 0.05."
        )
    else:
        interpretation = (
            "Milestone 2 not established under this test: the sideband-subtracted residual does not preserve "
            "fixed-branch, scanmax, or mobile scaled-Koide structure at p <= 0.05."
        )

    summary = {
        "script": "milestone2_sideband_subtracted_cupy.py",
        "purpose": "Sideband-subtracted signal-isolation test for LHCb log-winding / Koide structure.",
        "backend": "CuPy" if used_gpu else "NumPy",
        "files": files,
        "provenance": provenance,
        "config": asdict(cfg),
        "active_intervals": active_intervals(cfg),
        "delta_ell_active": active_delta_ell(cfg),
        "k_targets": {
            "n10": k_from_n(10.0, cfg),
            "n15": k_from_n(15.0, cfg),
            "n20": k_from_n(20.0, cfg),
        },
        "counts": {
            "signal_active_events": int(len(sig)),
            "B_low_active_events": int(len(low)),
            "B_high_active_events": int(len(high)),
            "hist_signal_sum": float(np.sum(h_sig)),
            "hist_side_sum": float(np.sum(h_side)),
            "alpha": float(alpha),
            "residual_sum": float(np.sum(residual)),
            "rms_z_residual": float(np.sqrt(np.mean(z * z))),
        },
        "outputs": {
            "bins_csv": str(bins_csv),
            "scan_csv": str(scan_csv),
            "wells_csv": str(wells_csv),
            "triplets_csv": str(triplets_csv),
            "integer_csv": str(integer_csv),
            "comb_csv": str(comb_csv),
            "null_csv": str(null_csv),
        },
        "verdict": verdict,
        "interpretation": interpretation,
    }

    summary_json = outdir / "milestone2_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = outdir / "milestone2_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Milestone 2: Sideband-Subtracted Signal Isolation\n")
        f.write("=================================================\n\n")
        f.write(f"backend: {'CuPy' if used_gpu else 'NumPy'}\n")
        f.write(f"Delta ell active: {active_delta_ell(cfg):.10f}\n")
        f.write(f"alpha: {alpha:.9f}\n\n")
        f.write("Counts\n")
        f.write("------\n")
        f.write(f"signal active events: {len(sig)}\n")
        f.write(f"B-low active events: {len(low)}\n")
        f.write(f"B-high active events: {len(high)}\n")
        f.write(f"hist signal sum: {np.sum(h_sig):.1f}\n")
        f.write(f"hist side sum: {np.sum(h_side):.1f}\n")
        f.write(f"residual sum: {np.sum(residual):.6f}\n\n")

        f.write("Observed statistics\n")
        f.write("-------------------\n")
        for key, val in verdict.items():
            if key not in ["flags", "best_triplet", "best_mobile_scaled_koide"]:
                f.write(f"{key}: {val}\n")

        f.write("\nFlags\n")
        f.write("-----\n")
        for key, val in verdict["flags"].items():
            f.write(f"{key}: {val}\n")

        f.write("\nInterpretation\n")
        f.write("--------------\n")
        f.write(interpretation + "\n")

    if plot:
        make_plots(outdir, bins_df, scan_df, summary, cfg)

    print("\n" + "=" * 100)
    print("MILESTONE 2 VERDICT")
    print("=" * 100)
    print(json.dumps(verdict, indent=2))
    print("\nInterpretation:")
    print(interpretation)

    print("\nSaved:")
    for p in [bins_csv, scan_csv, wells_csv, triplets_csv, integer_csv, comb_csv, null_csv, summary_json, report_path]:
        print(" ", p)


def parse_args():
    p = argparse.ArgumentParser(description="Milestone 2 sideband-subtracted signal-isolation test.")

    p.add_argument("--pattern", default=None, help='ROOT glob, e.g. "data/*.root"')
    p.add_argument("--outdir", default="outputs_milestone2_sideband")
    p.add_argument("--gpu", action="store_true", help="Use CuPy if available.")
    p.add_argument("--plot", action="store_true", help="Create plots.")

    p.add_argument("--n-null", type=int, default=1000)
    p.add_argument("--seed", type=int, default=271828)

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

    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config(
        outdir=args.outdir,
        n_null=args.n_null,
        seed=args.seed,
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
    )

    run(cfg, pattern=args.pattern, use_gpu=args.gpu, plot=args.plot)


if __name__ == "__main__":
    main()

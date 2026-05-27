#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Global look-elsewhere null test for log-periodic / WCT-style residual scans.

Purpose
-------
This script estimates how often a smooth non-oscillatory candidate spectrum,
with Poisson counting noise, can produce WCT-like log-domain structure after
being passed through the same broad search pipeline.

It is designed for the LHCb open-data B0 -> K*0 mu+ mu- candidate-spectrum
analysis.

Core tested objects
-------------------
q2:
    Momentum-transfer coordinate.

ell:
    Logarithmic coordinate, ell = ln(q2).

active domain A:
    q2 regions retained after charmonium vetoes.

Delta ell_A:
    Total active log-domain support.

k:
    Log-frequency of cosine residual component.

n:
    Active-domain winding coordinate, n = k Delta ell_A / (2 pi).

Q:
    Koide-style triplet ratio, with target Q_l = 2/3.

Outputs
-------
outputs_global_null_look_elsewhere/
    observed_scores.json
    toy_scores.csv
    global_pvalues.json
    scan_config.json
    summary.txt

Input expectations
------------------
The input CSV should contain either:

Option A:
    q2, observed, baseline

Option B:
    q2, count

If only count is present, a smooth baseline is estimated by Gaussian/KDE-like
smoothing in ell-space.

CuPy
----
Use --gpu to use CuPy. If CuPy is unavailable, the script falls back to NumPy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ============================================================
# Backend selection
# ============================================================

def get_backend(use_gpu: bool):
    """
    Return xp = cupy or numpy.

    xp:
        Array library used for vectorized computation.
    """
    if not use_gpu:
        return np, False

    try:
        import cupy as cp
        _ = cp.zeros(1)
        return cp, True
    except Exception as exc:
        print(f"[warn] CuPy unavailable; falling back to NumPy. Reason: {exc}")
        return np, False


def to_numpy(x):
    """
    Convert CuPy or NumPy array to NumPy array.
    """
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
class ScanConfig:
    q2_min: float = 0.1
    q2_max: float = 19.0

    # Nominal widened charmonium vetoes
    jpsi_low: float = 8.0
    jpsi_high: float = 11.0
    psi2s_low: float = 12.5
    psi2s_high: float = 14.5

    # Frequency scan
    k_min: float = 8.0
    k_max: float = 30.0
    k_steps: int = 2201

    # Integer active-domain scan
    n_min: int = 6
    n_max: int = 24

    # Koide comb
    n0: float = 15.0
    q_target: float = 2.0 / 3.0
    q_alt: float = 4.0 / 9.0

    # Well-first scan
    top_wells: int = 12
    min_peak_separation_k: float = 0.45

    # Null generation
    n_toys: int = 5000
    seed: int = 12345

    # Smoothing if baseline absent
    smooth_sigma_bins: float = 4.0

    # Score weights
    w_scan: float = 1.0
    w_n15: float = 1.0
    w_koide: float = 1.0
    w_well: float = 1.0

    # Numerical floor
    eps: float = 1e-9


# ============================================================
# Data loading and preparation
# ============================================================

def load_input_csv(path: Path) -> pd.DataFrame:
    """
    Load input CSV and normalize expected column names.

    Accepted columns:
        q2
        count or observed or yield
        baseline or expected or smooth
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    # Normalize names
    lower = {c.lower().strip(): c for c in df.columns}

    q2_col = None
    for cand in ["q2", "q_2", "q2_center", "q2_bin_center", "bin_center"]:
        if cand in lower:
            q2_col = lower[cand]
            break

    count_col = None
    for cand in ["observed", "count", "counts", "yield", "n", "y"]:
        if cand in lower:
            count_col = lower[cand]
            break

    baseline_col = None
    for cand in ["baseline", "expected", "smooth", "kde", "mu", "base"]:
        if cand in lower:
            baseline_col = lower[cand]
            break

    if q2_col is None:
        raise ValueError(f"Could not find q2 column in {df.columns.tolist()}")
    if count_col is None:
        raise ValueError(f"Could not find count/observed/yield column in {df.columns.tolist()}")

    out = pd.DataFrame()
    out["q2"] = pd.to_numeric(df[q2_col], errors="coerce")
    out["observed"] = pd.to_numeric(df[count_col], errors="coerce")

    if baseline_col is not None:
        out["baseline"] = pd.to_numeric(df[baseline_col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["q2", "observed"])
    out = out.sort_values("q2").reset_index(drop=True)

    return out


def active_mask_np(q2: np.ndarray, cfg: ScanConfig) -> np.ndarray:
    """
    Active mask outside charmonium vetoes.

    q2:
        Momentum-transfer bin centers.

    returns:
        Boolean mask for active analysis support.
    """
    q2 = np.asarray(q2)
    base = (q2 >= cfg.q2_min) & (q2 <= cfg.q2_max)
    veto_jpsi = (q2 >= cfg.jpsi_low) & (q2 <= cfg.jpsi_high)
    veto_psi2s = (q2 >= cfg.psi2s_low) & (q2 <= cfg.psi2s_high)
    return base & (~veto_jpsi) & (~veto_psi2s)


def delta_ell_active(cfg: ScanConfig) -> float:
    """
    Total retained active support in ell = ln(q2).

    For active intervals:
        [q2_min, jpsi_low]
        [jpsi_high, psi2s_low]
        [psi2s_high, q2_max]
    """
    return (
        math.log(cfg.jpsi_low / cfg.q2_min)
        + math.log(cfg.psi2s_low / cfg.jpsi_high)
        + math.log(cfg.q2_max / cfg.psi2s_high)
    )


def gaussian_smooth_np(y: np.ndarray, sigma_bins: float) -> np.ndarray:
    """
    Simple Gaussian smoothing without SciPy.

    y:
        Input count vector.

    sigma_bins:
        Gaussian width in bin units.

    returns:
        Smoothed vector.
    """
    y = np.asarray(y, dtype=float)
    radius = max(3, int(math.ceil(5 * sigma_bins)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(y, (radius, radius), mode="reflect")
    smoothed = np.convolve(padded, kernel, mode="same")[radius:-radius]
    return np.maximum(smoothed, 1e-6)


def prepare_data(df: pd.DataFrame, cfg: ScanConfig) -> Dict[str, np.ndarray]:
    """
    Prepare q2, ell, observed counts, baseline, and active mask.

    If baseline is missing, estimate one by smoothing in q2-bin order.
    """
    q2 = df["q2"].to_numpy(dtype=float)
    observed = df["observed"].to_numpy(dtype=float)

    keep = np.isfinite(q2) & np.isfinite(observed) & (q2 > 0)
    q2 = q2[keep]
    observed = observed[keep]

    if "baseline" in df.columns:
        baseline = df["baseline"].to_numpy(dtype=float)[keep]
        baseline = np.where(np.isfinite(baseline), baseline, np.nan)
        if np.any(~np.isfinite(baseline)) or np.any(baseline <= 0):
            baseline = gaussian_smooth_np(observed, cfg.smooth_sigma_bins)
    else:
        baseline = gaussian_smooth_np(observed, cfg.smooth_sigma_bins)

    baseline = np.maximum(baseline, cfg.eps)

    mask = active_mask_np(q2, cfg)
    q2 = q2[mask]
    observed = observed[mask]
    baseline = baseline[mask]
    ell = np.log(q2)

    # Sort by ell
    order = np.argsort(ell)
    return {
        "q2": q2[order],
        "ell": ell[order],
        "observed": observed[order],
        "baseline": baseline[order],
    }


# ============================================================
# Residual and fitting utilities
# ============================================================

def poisson_standardized_residual(y, mu, xp, eps: float):
    """
    Standardized residual for Poisson-like counts.

    r_i = (y_i - mu_i) / sqrt(mu_i + eps)
    """
    return (y - mu) / xp.sqrt(mu + eps)


def center_weighted(v, xp):
    """
    Remove mean from vector.
    """
    return v - xp.mean(v)


def scan_cosine_power(
    ell,
    residual,
    k_grid,
    xp,
    eps: float = 1e-12,
):
    """
    Compute cosine/sine least-squares power for each k.

    Model:
        residual ~ a cos(k ell) + b sin(k ell)

    For each k, score approximates explained squared residual:
        score = projection_power = beta^T X^T y

    This is a fast proxy for a two-parameter sinusoidal residual scan.

    Returns:
        scores array shaped [len(k_grid)]
    """
    ell = ell.reshape(1, -1)
    k = k_grid.reshape(-1, 1)

    phase = k * ell
    c = xp.cos(phase)
    s = xp.sin(phase)

    y = residual.reshape(1, -1)

    cc = xp.sum(c * c, axis=1) + eps
    ss = xp.sum(s * s, axis=1) + eps
    cs = xp.sum(c * s, axis=1)

    cy = xp.sum(c * y, axis=1)
    sy = xp.sum(s * y, axis=1)

    det = cc * ss - cs * cs + eps

    # beta = inv([[cc,cs],[cs,ss]]) @ [cy,sy]
    a = (ss * cy - cs * sy) / det
    b = (-cs * cy + cc * sy) / det

    # explained power = beta dot X'y
    score = a * cy + b * sy

    return score


def best_scan_score(ell, y, mu, cfg: ScanConfig, xp):
    """
    Continuous k scan score.

    Returns:
        best_k, best_score, score_at_ref, k_grid, scores
    """
    residual = poisson_standardized_residual(y, mu, xp, cfg.eps)
    residual = center_weighted(residual, xp)

    k_grid = xp.linspace(cfg.k_min, cfg.k_max, cfg.k_steps)
    scores = scan_cosine_power(ell, residual, k_grid, xp, cfg.eps)

    idx = int(to_numpy(xp.argmax(scores)))
    best_k = float(to_numpy(k_grid[idx]))
    best_score = float(to_numpy(scores[idx]))

    return best_k, best_score, k_grid, scores


def integer_winding_scan(ell, y, mu, cfg: ScanConfig, xp):
    """
    Scan integer active-domain winding frequencies.

    k_n = 2 pi n / Delta ell_A
    """
    d_ell = delta_ell_active(cfg)
    n_values_np = np.arange(cfg.n_min, cfg.n_max + 1)
    k_values_np = 2.0 * np.pi * n_values_np / d_ell

    n_values = xp.asarray(n_values_np, dtype=float)
    k_values = xp.asarray(k_values_np, dtype=float)

    residual = poisson_standardized_residual(y, mu, xp, cfg.eps)
    residual = center_weighted(residual, xp)

    scores = scan_cosine_power(ell, residual, k_values, xp, cfg.eps)
    idx = int(to_numpy(xp.argmax(scores)))

    # n=15 score if inside range
    n15_score = np.nan
    if cfg.n_min <= 15 <= cfg.n_max:
        i15 = int(15 - cfg.n_min)
        n15_score = float(to_numpy(scores[i15]))

    return {
        "best_n": int(n_values_np[idx]),
        "best_k": float(k_values_np[idx]),
        "best_score": float(to_numpy(scores[idx])),
        "n15_score": float(n15_score),
        "n_values": n_values_np,
        "k_values": k_values_np,
        "scores": to_numpy(scores),
    }


def comb_score_for_q(ell, y, mu, q_value: float, cfg: ScanConfig, xp):
    """
    Score a Koide-style comb:
        (n-, n0, n+) = n0 * (Q, 1, 2Q)

    Frequencies:
        k_i = 2 pi n_i / Delta ell_A

    Model:
        residual ~ sum_i [a_i cos(k_i ell) + b_i sin(k_i ell)]

    Score:
        explained least-squares power.
    """
    d_ell = delta_ell_active(cfg)
    n_triplet = np.array([cfg.n0 * q_value, cfg.n0, cfg.n0 * 2.0 * q_value], dtype=float)
    k_triplet = 2.0 * np.pi * n_triplet / d_ell

    residual = poisson_standardized_residual(y, mu, xp, cfg.eps)
    residual = center_weighted(residual, xp)

    ell_1 = ell.reshape(1, -1)
    cols = []
    for k in k_triplet:
        phase = float(k) * ell_1
        cols.append(xp.cos(phase).reshape(-1))
        cols.append(xp.sin(phase).reshape(-1))

    X = xp.stack(cols, axis=1)  # [N, 6]
    yy = residual.reshape(-1, 1)

    XtX = X.T @ X + cfg.eps * xp.eye(X.shape[1])
    Xty = X.T @ yy

    beta = xp.linalg.solve(XtX, Xty)
    explained = float(to_numpy((beta.T @ Xty)[0, 0]))

    return {
        "q": float(q_value),
        "n_triplet": n_triplet.tolist(),
        "k_triplet": k_triplet.tolist(),
        "score": explained,
    }


def koide_comb_scan(ell, y, mu, cfg: ScanConfig, xp):
    """
    Compare Q=2/3 Class I and Q=4/9 folded Class III combs.

    Returns both scores and ratio.
    """
    q23 = comb_score_for_q(ell, y, mu, cfg.q_target, cfg, xp)
    q49 = comb_score_for_q(ell, y, mu, cfg.q_alt, cfg, xp)

    ratio_49_over_23 = q49["score"] / max(q23["score"], cfg.eps)

    return {
        "q23_score": q23["score"],
        "q49_score": q49["score"],
        "q49_over_q23": float(ratio_49_over_23),
        "q23": q23,
        "q49": q49,
    }


# ============================================================
# Well-first triplet logic
# ============================================================

def find_local_peaks_np(k_grid: np.ndarray, scores: np.ndarray, min_sep_k: float, top_n: int):
    """
    Find local maxima in score(k), enforce approximate separation in k,
    and return top peaks.

    Returns:
        List of (k, score)
    """
    k_grid = np.asarray(k_grid)
    scores = np.asarray(scores)

    if len(scores) < 3:
        return []

    candidates = []
    for i in range(1, len(scores) - 1):
        if scores[i] > scores[i - 1] and scores[i] >= scores[i + 1]:
            candidates.append((float(k_grid[i]), float(scores[i])))

    candidates.sort(key=lambda x: x[1], reverse=True)

    selected = []
    for k, sc in candidates:
        if all(abs(k - k0) >= min_sep_k for k0, _ in selected):
            selected.append((k, sc))
        if len(selected) >= top_n:
            break

    return selected


def triplet_koide_error(ns: Tuple[float, float, float], q_target: float):
    """
    Compute Koide-like sideband ratio error for ordered triplet n1 < n2 < n3.

    Q_low  = n1 / n2
    Q_high = n3 / (2 n2)
    eps_K  = sqrt((Q_low-Q)^2 + (Q_high-Q)^2)
    Q_mean = (Q_low + Q_high)/2
    """
    n1, n2, n3 = sorted(ns)
    q_low = n1 / n2
    q_high = n3 / (2.0 * n2)
    eps_k = math.sqrt((q_low - q_target) ** 2 + (q_high - q_target) ** 2)
    q_mean = 0.5 * (q_low + q_high)
    return eps_k, q_mean, q_low, q_high


def best_well_triplet_from_scan(k_grid_np, scores_np, cfg: ScanConfig):
    """
    From continuous scan scores, find top local wells, convert k -> n,
    and find best Koide-like triplet.
    """
    peaks = find_local_peaks_np(
        k_grid_np,
        scores_np,
        min_sep_k=cfg.min_peak_separation_k,
        top_n=cfg.top_wells,
    )

    d_ell = delta_ell_active(cfg)

    wells = []
    for k, sc in peaks:
        n_eff = k * d_ell / (2.0 * np.pi)
        wells.append({"k": k, "n": n_eff, "score": sc})

    if len(wells) < 3:
        return {
            "eps_k": np.inf,
            "q_mean": np.nan,
            "triplet_n": [],
            "triplet_k": [],
            "wells": wells,
        }

    best = None
    m = len(wells)
    for i in range(m):
        for j in range(i + 1, m):
            for l in range(j + 1, m):
                ns = (wells[i]["n"], wells[j]["n"], wells[l]["n"])
                eps_k, q_mean, q_low, q_high = triplet_koide_error(ns, cfg.q_target)
                if best is None or eps_k < best["eps_k"]:
                    ks = (wells[i]["k"], wells[j]["k"], wells[l]["k"])
                    best = {
                        "eps_k": eps_k,
                        "q_mean": q_mean,
                        "q_low": q_low,
                        "q_high": q_high,
                        "triplet_n": sorted(ns),
                        "triplet_k": sorted(ks),
                        "wells": wells,
                    }

    return best


# ============================================================
# Combined scoring
# ============================================================

def compute_all_scores(y_np: np.ndarray, mu_np: np.ndarray, ell_np: np.ndarray, cfg: ScanConfig, xp):
    """
    Compute all observed or toy scores.

    y_np:
        Observed/toy counts in active domain.

    mu_np:
        Smooth null baseline.

    ell_np:
        Log q2 bin centers in active domain.

    returns:
        Flat dictionary of scalar scores and key diagnostics.
    """
    y = xp.asarray(y_np, dtype=float)
    mu = xp.asarray(mu_np, dtype=float)
    ell = xp.asarray(ell_np, dtype=float)

    best_k, best_scan, k_grid, scores = best_scan_score(ell, y, mu, cfg, xp)

    int_scan = integer_winding_scan(ell, y, mu, cfg, xp)
    comb = koide_comb_scan(ell, y, mu, cfg, xp)

    k_grid_np = to_numpy(k_grid)
    scores_np = to_numpy(scores)
    well = best_well_triplet_from_scan(k_grid_np, scores_np, cfg)

    # WCT-like combined score.
    # Larger means more WCT-like under this diagnostic.
    #
    # Components:
    #   scan strength
    #   n=15 strength
    #   Koide Q=2/3 comb strength
    #   inverse Koide-error for best raw-well triplet
    #
    # Well score is bounded by eps to avoid explosion.
    well_score = 1.0 / max(well["eps_k"], 1e-6)

    total_score = (
        cfg.w_scan * best_scan
        + cfg.w_n15 * int_scan["n15_score"]
        + cfg.w_koide * comb["q23_score"]
        + cfg.w_well * well_score
    )

    out = {
        "score_total": float(total_score),
        "scan_best_k": float(best_k),
        "scan_best_score": float(best_scan),

        "integer_best_n": int(int_scan["best_n"]),
        "integer_best_k": float(int_scan["best_k"]),
        "integer_best_score": float(int_scan["best_score"]),
        "integer_n15_score": float(int_scan["n15_score"]),

        "comb_q23_score": float(comb["q23_score"]),
        "comb_q49_score": float(comb["q49_score"]),
        "comb_q49_over_q23": float(comb["q49_over_q23"]),

        "well_eps_k": float(well["eps_k"]),
        "well_q_mean": float(well["q_mean"]) if np.isfinite(well["q_mean"]) else np.nan,
        "well_score_inv_eps": float(well_score),
        "well_triplet_n": well["triplet_n"],
        "well_triplet_k": well["triplet_k"],
    }

    return out


def empirical_pvalue_ge(toy_values: np.ndarray, observed_value: float):
    """
    Empirical p-value for large-is-more-extreme statistic.

    Uses +1 correction:
        p = (1 + count(toy >= obs)) / (N + 1)
    """
    toy_values = np.asarray(toy_values, dtype=float)
    return float((1.0 + np.sum(toy_values >= observed_value)) / (len(toy_values) + 1.0))


def empirical_pvalue_le(toy_values: np.ndarray, observed_value: float):
    """
    Empirical p-value for small-is-more-extreme statistic.

    Uses +1 correction:
        p = (1 + count(toy <= obs)) / (N + 1)
    """
    toy_values = np.asarray(toy_values, dtype=float)
    return float((1.0 + np.sum(toy_values <= observed_value)) / (len(toy_values) + 1.0))


# ============================================================
# Toy generation
# ============================================================

def run_toys(data: Dict[str, np.ndarray], cfg: ScanConfig, xp, used_gpu: bool, outdir: Path):
    """
    Run toy Poisson null ensemble.

    Null:
        y_toy_i ~ Poisson(mu_i)

    The baseline mu_i is held fixed to observed smooth baseline.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    ell = data["ell"]
    observed = data["observed"]
    mu = np.maximum(data["baseline"], cfg.eps)

    print("[info] computing observed scores...")
    observed_scores = compute_all_scores(observed, mu, ell, cfg, xp)

    rng = np.random.default_rng(cfg.seed)

    toy_rows = []
    print(f"[info] running {cfg.n_toys} toys using {'CuPy' if used_gpu else 'NumPy'} backend...")

    for t in range(cfg.n_toys):
        y_toy = rng.poisson(mu).astype(float)
        scores = compute_all_scores(y_toy, mu, ell, cfg, xp)
        scores["toy"] = t
        toy_rows.append(scores)

        if (t + 1) % max(1, cfg.n_toys // 20) == 0:
            print(f"[progress] {t + 1}/{cfg.n_toys}")

    toy_df = pd.DataFrame(toy_rows)

    # p-values
    pvals = {
        "p_global_total_score": empirical_pvalue_ge(
            toy_df["score_total"].to_numpy(), observed_scores["score_total"]
        ),
        "p_scan_best_score": empirical_pvalue_ge(
            toy_df["scan_best_score"].to_numpy(), observed_scores["scan_best_score"]
        ),
        "p_integer_n15_score": empirical_pvalue_ge(
            toy_df["integer_n15_score"].to_numpy(), observed_scores["integer_n15_score"]
        ),
        "p_comb_q23_score": empirical_pvalue_ge(
            toy_df["comb_q23_score"].to_numpy(), observed_scores["comb_q23_score"]
        ),
        "p_well_eps_k_low": empirical_pvalue_le(
            toy_df["well_eps_k"].to_numpy(), observed_scores["well_eps_k"]
        ),
    }

    # Save
    with open(outdir / "observed_scores.json", "w", encoding="utf-8") as f:
        json.dump(observed_scores, f, indent=2)

    with open(outdir / "global_pvalues.json", "w", encoding="utf-8") as f:
        json.dump(pvals, f, indent=2)

    with open(outdir / "scan_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    toy_df.to_csv(outdir / "toy_scores.csv", index=False)

    with open(outdir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Global look-elsewhere toy-null summary\n")
        f.write("======================================\n\n")
        f.write(f"backend: {'CuPy' if used_gpu else 'NumPy'}\n")
        f.write(f"n_toys: {cfg.n_toys}\n")
        f.write(f"seed: {cfg.seed}\n")
        f.write(f"Delta ell_A: {delta_ell_active(cfg):.10f}\n\n")

        f.write("Observed scores\n")
        f.write("---------------\n")
        for k, v in observed_scores.items():
            f.write(f"{k}: {v}\n")

        f.write("\nGlobal empirical p-values\n")
        f.write("-------------------------\n")
        for k, v in pvals.items():
            f.write(f"{k}: {v}\n")

    return observed_scores, toy_df, pvals


# ============================================================
# Optional plotting
# ============================================================

def make_plots(outdir: Path):
    """
    Generate simple diagnostic plots if matplotlib is available.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable; skipping plots. Reason: {exc}")
        return

    toy_path = outdir / "toy_scores.csv"
    obs_path = outdir / "observed_scores.json"

    if not toy_path.exists() or not obs_path.exists():
        return

    toy = pd.read_csv(toy_path)
    with open(obs_path, "r", encoding="utf-8") as f:
        obs = json.load(f)

    plot_specs = [
        ("score_total", "Global WCT-like total score", "score_total_null.png"),
        ("scan_best_score", "Best continuous high-k scan score", "scan_best_score_null.png"),
        ("integer_n15_score", "n=15 integer-winding score", "integer_n15_score_null.png"),
        ("comb_q23_score", "Q=2/3 Koide-comb score", "comb_q23_score_null.png"),
        ("well_eps_k", "Best raw-well Koide error", "well_eps_k_null.png"),
    ]

    for col, title, fname in plot_specs:
        if col not in toy.columns or col not in obs:
            continue

        plt.figure(figsize=(8, 5))
        plt.hist(toy[col].dropna().to_numpy(), bins=60, alpha=0.8)
        plt.axvline(obs[col], linestyle="--", linewidth=2)
        plt.title(title)
        plt.xlabel(col)
        plt.ylabel("toy count")
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=160)
        plt.close()


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="CuPy/NumPy global look-elsewhere toy-null test for WCT log-periodic residual scans."
    )

    p.add_argument("--input", required=True, help="Input CSV: q2 plus observed/count and optional baseline.")
    p.add_argument("--out", default="outputs_global_null_look_elsewhere", help="Output directory.")
    p.add_argument("--n-toys", type=int, default=5000, help="Number of Poisson toy null datasets.")
    p.add_argument("--seed", type=int, default=12345, help="Random seed.")
    p.add_argument("--gpu", action="store_true", help="Use CuPy GPU backend if available.")

    p.add_argument("--q2-min", type=float, default=0.1)
    p.add_argument("--q2-max", type=float, default=19.0)

    p.add_argument("--jpsi-low", type=float, default=8.0)
    p.add_argument("--jpsi-high", type=float, default=11.0)
    p.add_argument("--psi2s-low", type=float, default=12.5)
    p.add_argument("--psi2s-high", type=float, default=14.5)

    p.add_argument("--k-min", type=float, default=8.0)
    p.add_argument("--k-max", type=float, default=30.0)
    p.add_argument("--k-steps", type=int, default=2201)

    p.add_argument("--n-min", type=int, default=6)
    p.add_argument("--n-max", type=int, default=24)
    p.add_argument("--top-wells", type=int, default=12)

    p.add_argument("--smooth-sigma-bins", type=float, default=4.0)

    p.add_argument("--plot", action="store_true", help="Create diagnostic plots.")

    return p.parse_args()


def main():
    args = parse_args()

    cfg = ScanConfig(
        q2_min=args.q2_min,
        q2_max=args.q2_max,
        jpsi_low=args.jpsi_low,
        jpsi_high=args.jpsi_high,
        psi2s_low=args.psi2s_low,
        psi2s_high=args.psi2s_high,
        k_min=args.k_min,
        k_max=args.k_max,
        k_steps=args.k_steps,
        n_min=args.n_min,
        n_max=args.n_max,
        top_wells=args.top_wells,
        n_toys=args.n_toys,
        seed=args.seed,
        smooth_sigma_bins=args.smooth_sigma_bins,
    )

    xp, used_gpu = get_backend(args.gpu)

    input_path = Path(args.input)
    outdir = Path(args.out)

    print(f"[load] {input_path}")
    df = load_input_csv(input_path)
    data = prepare_data(df, cfg)

    print(f"[data] active bins: {len(data['q2'])}")
    print(f"[data] q2 range: {data['q2'].min():.4f} .. {data['q2'].max():.4f}")
    print(f"[data] ell range: {data['ell'].min():.4f} .. {data['ell'].max():.4f}")
    print(f"[data] Delta ell_A config: {delta_ell_active(cfg):.10f}")
    print(f"[backend] {'CuPy' if used_gpu else 'NumPy'}")

    observed_scores, toy_df, pvals = run_toys(data, cfg, xp, used_gpu, outdir)

    if args.plot:
        make_plots(outdir)

    print("\n[done] observed scores")
    for k, v in observed_scores.items():
        print(f"  {k}: {v}")

    print("\n[done] global p-values")
    for k, v in pvals.items():
        print(f"  {k}: {v}")

    print(f"\n[outputs] {outdir.resolve()}")


if __name__ == "__main__":
    main()
"""
09d_two_mode_kde_baseline_polar_cupy.py

Two-mode bounded Poisson log-cos test.

Goal:
    Determine whether the high-k WCT-like mode near k ~= 19.53 survives
    after accounting for the dominant bounded-Poisson mask mode near k ~= 7.61.

Model 0:
    lambda_i = B_i * exp(C + a1 cos(k1 ell_i) + b1 sin(k1 ell_i))

Model 1:
    lambda_i = B_i * exp(C
                         + a1 cos(k1 ell_i) + b1 sin(k1 ell_i)
                         + a2 cos(k2 ell_i) + b2 sin(k2 ell_i))

where:
    ell_i = ln(q2_i)
    k1 fixed = 7.61054
    k2 scanned

Bounds:
    sqrt(a1^2 + b1^2) <= A1_MAX
    sqrt(a2^2 + b2^2) <= A2_MAX

Score:
    DeltaD_add(k2) = D_base(k1) - D_two(k1,k2)

Null:
    Generate Poisson samples from fitted base model:
        N_null ~ Poisson(lambda_base)
    For each null:
        refit base model
        scan two-mode model
        record max DeltaD_add

Outputs:
    outputs_logcos_poisson_twomode/two_mode_summary.json
    outputs_logcos_poisson_twomode/two_mode_scan_mask.csv
    outputs_logcos_poisson_twomode/two_mode_null_mask.csv
    outputs_logcos_poisson_twomode/two_mode_scan_mask.png
    outputs_logcos_poisson_twomode/two_mode_fit_mask.png
"""

from __future__ import annotations

import glob
import json
import importlib.metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.optimize import minimize
from scipy.stats import gaussian_kde


# =============================================================================
# CuPy import with metadata workaround
# =============================================================================

_orig_distributions = importlib_metadata.distributions

def _safe_distributions(*args, **kwargs):
    dists = []
    for d in _orig_distributions(*args, **kwargs):
        try:
            if getattr(d, "metadata", None) is not None:
                dists.append(d)
        except Exception:
            pass
    return dists

importlib_metadata.distributions = _safe_distributions

import cupy as cp

GPU_AVAILABLE = True
USE_CUPY = True
NULL_BATCH = 512


# =============================================================================
# Config
# =============================================================================

DATA_GLOB = "data/*.root"
TREE_NAME = "B0_KstMuMu/DecayTree"

OUT_DIR = Path("outputs_logcos_poisson_twomode_kde_polar")
OUT_DIR.mkdir(exist_ok=True, parents=True)

Q2_MIN = 0.1
Q2_MAX = 19.0

B0_M_MIN = 5230.0
B0_M_MAX = 5330.0

KST_M_MIN = 795.9
KST_M_MAX = 995.9

# Wider vetoes for resonance-tail stress test.
JPSI_VETO = (8.0, 11.0)
PSI2S_VETO = (12.5, 14.5)

# Q2_BINS = 60
SMOOTH_WINDOW = 11
SMOOTH_POLY = 3

RUN_MODE = "mask"
BASELINE_MODE = "kde"  # options: "kde", "savgol"
KDE_BW_METHOD = "scott"  # scipy gaussian_kde bw_method
KDE_BANDWIDTH_SCALE = 1.50  # multiply scipy KDE bandwidth

# Dominant bounded-Poisson mode from previous run.
K1_FIXED = 7.61054

# WCT/reference high-k mode from residual scan.
REFERENCE_K2 = 19.5296
REFERENCE_TOL = 0.75

# Scan second mode.
# Local high-k scan window. Wider than 09c to detect edge-running.
K2_MIN = 18.0
K2_MAX = 24.0
N_K2 = 601
K_EDGE_TOL = 0.05

# Amplitude bounds.
# A1_MAX = 0.50
# A2_MAX = 0.50

A1_MAX = 0.10
A2_MAX = 0.10
ETA_CLIP = 0.2
Q2_BINS = 60

# ETA_CLIP = 1.0

IRLS_ITERS = 10
IRLS_RIDGE = 1e-8

NULL_N = 5000
SEED = 12345


# =============================================================================
# Data loading
# =============================================================================

def inv_mass2(px, py, pz, e):
    return e**2 - px**2 - py**2 - pz**2


def in_veto_q2(q2):
    q2 = np.asarray(q2)
    jpsi = (q2 >= JPSI_VETO[0]) & (q2 <= JPSI_VETO[1])
    psi2s = (q2 >= PSI2S_VETO[0]) & (q2 <= PSI2S_VETO[1])
    return jpsi | psi2s


def load_events():
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError(f"No ROOT files match {DATA_GLOB}")

    required = [
        "B0_M",
        "muplus_PX", "muplus_PY", "muplus_PZ", "muplus_PE",
        "muminus_PX", "muminus_PY", "muminus_PZ", "muminus_PE",
        "Kplus_PX", "Kplus_PY", "Kplus_PZ", "Kplus_PE",
        "piminus_PX", "piminus_PY", "piminus_PZ", "piminus_PE",
    ]
    optional = ["Kst_892_0_M", "Kst_M"]

    chunks = []

    for path in files:
        print(f"[load] {path}")
        with uproot.open(path) as f:
            if TREE_NAME in f:
                tree_name = TREE_NAME
            else:
                tree_name = next(k for k in f.keys(recursive=True) if "DecayTree" in k)

            t = f[tree_name]
            keys = set(t.keys())
            branches = [b for b in required + optional if b in keys]
            arr = t.arrays(branches, library="pd")
            chunks.append(arr)

    df = pd.concat(chunks, ignore_index=True)

    q_px = df["muplus_PX"] + df["muminus_PX"]
    q_py = df["muplus_PY"] + df["muminus_PY"]
    q_pz = df["muplus_PZ"] + df["muminus_PZ"]
    q_e = df["muplus_PE"] + df["muminus_PE"]

    df["q2"] = inv_mass2(q_px, q_py, q_pz, q_e) / 1e6

    if "Kst_892_0_M" in df.columns:
        df["Kst_mass"] = df["Kst_892_0_M"]
    elif "Kst_M" in df.columns:
        df["Kst_mass"] = df["Kst_M"]
    else:
        k_px = df["Kplus_PX"] + df["piminus_PX"]
        k_py = df["Kplus_PY"] + df["piminus_PY"]
        k_pz = df["Kplus_PZ"] + df["piminus_PZ"]
        k_e = df["Kplus_PE"] + df["piminus_PE"]
        m2 = inv_mass2(k_px, k_py, k_pz, k_e)
        df["Kst_mass"] = np.sqrt(np.maximum(m2, 0.0))

    sel = np.isfinite(df["q2"])
    sel &= (df["q2"] >= Q2_MIN) & (df["q2"] <= Q2_MAX)
    sel &= (df["B0_M"] >= B0_M_MIN) & (df["B0_M"] <= B0_M_MAX)
    sel &= (df["Kst_mass"] >= KST_M_MIN) & (df["Kst_mass"] <= KST_M_MAX)

    return df.loc[sel].copy()


# =============================================================================
# Binning / baseline
# =============================================================================

def make_savgol_baseline(counts):
    n = len(counts)
    window = min(SMOOTH_WINDOW, n if n % 2 else n - 1)

    min_win = SMOOTH_POLY + 2
    if min_win % 2 == 0:
        min_win += 1

    window = max(window, min_win)

    baseline = savgol_filter(
        counts.astype(float),
        window_length=window,
        polyorder=SMOOTH_POLY,
    )
    return np.maximum(baseline, 1e-9)


def make_kde_baseline(q2_values, centers, edges, veto):
    """
    Resonance-aware empirical baseline.

    The KDE is fit only on events outside the widened charmonium vetoes, then
    evaluated at all q2 bin centers. This avoids training the smooth baseline on
    the veto peaks/tails that created the Savitzky-Golay wall.
    """
    q2_values = np.asarray(q2_values, dtype=float)
    train = q2_values[
        np.isfinite(q2_values)
        & (q2_values >= Q2_MIN)
        & (q2_values <= Q2_MAX)
        & (~in_veto_q2(q2_values))
    ]

    if len(train) < 100:
        raise RuntimeError(f"Too few events for KDE baseline after vetoes: {len(train)}")

    kde = gaussian_kde(train, bw_method=KDE_BW_METHOD)
    kde.set_bandwidth(kde.factor * KDE_BANDWIDTH_SCALE)
    dens = kde.evaluate(centers)
    bin_width = float(edges[1] - edges[0])
    baseline = dens * len(train) * bin_width

    # Do not allow exact zeros in the veto valleys or extrapolated tails.
    return np.maximum(baseline, 1e-9)


def make_binned_counts(q2_values, mode="mask"):
    counts, edges = np.histogram(q2_values, bins=Q2_BINS, range=(Q2_MIN, Q2_MAX))
    centers = 0.5 * (edges[:-1] + edges[1:])
    veto = in_veto_q2(centers)

    if BASELINE_MODE == "kde":
        baseline = make_kde_baseline(q2_values, centers, edges, veto)
    elif BASELINE_MODE == "savgol":
        baseline = make_savgol_baseline(counts)
    else:
        raise ValueError(f"Unknown BASELINE_MODE={BASELINE_MODE}")

    if mode == "mask":
        keep = ~veto
    elif mode == "old":
        keep = np.ones_like(veto, dtype=bool)
    elif mode == "inpaint":
        keep = np.ones_like(veto, dtype=bool)
    else:
        raise ValueError(mode)

    N = counts[keep].astype(float)
    B = baseline[keep].astype(float)
    q2 = centers[keep].astype(float)
    ell = np.log(q2)

    scale = np.sum(N) / max(np.sum(B), 1e-12)
    B = np.maximum(B * scale, 1e-9)

    return {
        "centers": centers,
        "counts_all": counts,
        "baseline_all": baseline,
        "veto_all": veto,
        "keep": keep,
        "q2": q2,
        "ell": ell.astype(float),
        "N": N.astype(float),
        "B": B.astype(float),
        "baseline_mode": BASELINE_MODE,
        "kde_bw_method": KDE_BW_METHOD if BASELINE_MODE == "kde" else None,
    }


# =============================================================================
# Deviance
# =============================================================================

def poisson_deviance_np(N, lam):
    N = np.asarray(N, dtype=float)
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-12)

    term = lam - N
    nz = N > 0
    term[nz] += N[nz] * np.log(N[nz] / lam[nz])
    return float(2.0 * np.sum(term))


def gaussian_sigma_from_p(p):
    try:
        from scipy.special import erfcinv
        return float(np.sqrt(2.0) * erfcinv(2.0 * p))
    except Exception:
        return None


def p_value(real, null_vals):
    return float((1 + np.sum(null_vals >= real)) / (1 + len(null_vals)))


# =============================================================================
# CPU exact bounded fits
# =============================================================================

def _safe_log_scale(N, B) -> float:
    """Null normalization C seed, clipped to the bounded C range."""
    c0 = float(np.log(max(np.sum(N), 1e-12) / max(np.sum(B), 1e-12)))
    return float(np.clip(c0, -ETA_CLIP, ETA_CLIP))


def _ab_from_polar(r: float, phi: float):
    """Use same phase convention as the old beta form: phi = atan2(-b, a)."""
    return float(r * np.cos(phi)), float(-r * np.sin(phi))


def _poisson_nll_from_eta(N, B, eta):
    lam = np.maximum(B * np.exp(eta), 1e-12)
    return float(np.sum(lam - N * np.log(lam)))


def _polar_base_nll(theta, N, B, ell, k1):
    C, r1, phi1 = theta
    a1, b1 = _ab_from_polar(r1, phi1)
    eta = C + a1 * np.cos(k1 * ell) + b1 * np.sin(k1 * ell)
    return _poisson_nll_from_eta(N, B, eta)


def _polar_two_nll(theta, N, B, ell, k1, k2):
    C, r1, phi1, r2, phi2 = theta
    a1, b1 = _ab_from_polar(r1, phi1)
    a2, b2 = _ab_from_polar(r2, phi2)
    eta = (
        C
        + a1 * np.cos(k1 * ell) + b1 * np.sin(k1 * ell)
        + a2 * np.cos(k2 * ell) + b2 * np.sin(k2 * ell)
    )
    return _poisson_nll_from_eta(N, B, eta)


def _best_minimize(fun, starts, bounds, args):
    """Run bounded L-BFGS-B from several deterministic starts and keep the lowest NLL."""
    best = None
    for x0 in starts:
        res = minimize(
            fun,
            x0=np.asarray(x0, dtype=float),
            args=args,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
        )
        if best is None or float(res.fun) < float(best.fun):
            best = res
    return best


def fit_base_cpu_bounded(N, B, ell, k1):
    """
    Polar bounded base fit. This removes the SLSQP circular constraint.

    Parameters:
        C in [-ETA_CLIP, ETA_CLIP]
        A1=r1 in [0, A1_MAX]
        phi1 in [-pi, pi]
    """
    N = np.asarray(N, dtype=float)
    B = np.maximum(np.asarray(B, dtype=float), 1e-12)
    ell = np.asarray(ell, dtype=float)

    C0 = _safe_log_scale(N, B)
    starts = []
    for r in (0.0, 0.5 * A1_MAX, A1_MAX):
        for ph in (0.0, 0.5 * np.pi, -0.5 * np.pi, np.pi):
            starts.append([C0, r, ph])

    bounds = [(-ETA_CLIP, ETA_CLIP), (0.0, A1_MAX), (-np.pi, np.pi)]
    res = _best_minimize(_polar_base_nll, starts, bounds, (N, B, ell, k1))

    C, r1, phi1 = res.x
    a1, b1 = _ab_from_polar(r1, phi1)
    eta = C + a1 * np.cos(k1 * ell) + b1 * np.sin(k1 * ell)
    lam = np.maximum(B * np.exp(eta), 1e-12)
    D = poisson_deviance_np(N, lam)

    A1 = float(r1)
    return {
        "k1": float(k1),
        "D_base": float(D),
        "C": float(C),
        "a1": float(a1),
        "b1": float(b1),
        "A1": A1,
        "phi1": float(phi1),
        "lambda_base": lam,
        "success": bool(res.success),
        "message": str(res.message),
        "n_iter": int(getattr(res, "nit", -1)),
        "optimizer": "polar_LBFGSB",
        "amplitude1_bound_active": bool(abs(A1 - A1_MAX) <= 1e-5),
    }


def fit_two_cpu_bounded(N, B, ell, k1, k2):
    """
    Polar bounded two-mode fit. This removes both SLSQP circular constraints.

    Parameters:
        C in [-ETA_CLIP, ETA_CLIP]
        A1=r1 in [0, A1_MAX]
        phi1 in [-pi, pi]
        A2=r2 in [0, A2_MAX]
        phi2 in [-pi, pi]
    """
    N = np.asarray(N, dtype=float)
    B = np.maximum(np.asarray(B, dtype=float), 1e-12)
    ell = np.asarray(ell, dtype=float)

    base = fit_base_cpu_bounded(N, B, ell, k1)
    C0 = float(np.clip(base["C"], -ETA_CLIP, ETA_CLIP))
    r10 = float(np.clip(base["A1"], 0.0, A1_MAX))
    ph10 = float(np.clip(base["phi1"], -np.pi, np.pi))

    starts = []
    # Include bound-active starts because prior SLSQP solutions consistently found the cap.
    for r2 in (0.0, 0.5 * A2_MAX, A2_MAX):
        for ph2 in (0.0, 0.5 * np.pi, -0.5 * np.pi, np.pi):
            starts.append([C0, r10, ph10, r2, ph2])
    # Extra opposite-phase starts for mode 1 in case the joint fit wants to rotate both modes.
    for ph1 in (0.0, 0.5 * np.pi, -0.5 * np.pi, np.pi):
        starts.append([C0, A1_MAX, ph1, A2_MAX, 0.0])
        starts.append([C0, A1_MAX, ph1, A2_MAX, np.pi])

    bounds = [
        (-ETA_CLIP, ETA_CLIP),
        (0.0, A1_MAX),
        (-np.pi, np.pi),
        (0.0, A2_MAX),
        (-np.pi, np.pi),
    ]
    res = _best_minimize(_polar_two_nll, starts, bounds, (N, B, ell, k1, k2))

    C, r1, phi1, r2, phi2 = res.x
    a1, b1 = _ab_from_polar(r1, phi1)
    a2, b2 = _ab_from_polar(r2, phi2)

    eta = (
        C
        + a1 * np.cos(k1 * ell) + b1 * np.sin(k1 * ell)
        + a2 * np.cos(k2 * ell) + b2 * np.sin(k2 * ell)
    )
    lam = np.maximum(B * np.exp(eta), 1e-12)
    D = poisson_deviance_np(N, lam)

    A1 = float(r1)
    A2 = float(r2)
    return {
        "k1": float(k1),
        "k2": float(k2),
        "D_two": float(D),
        "C": float(C),
        "a1": float(a1),
        "b1": float(b1),
        "A1": A1,
        "phi1": float(phi1),
        "a2": float(a2),
        "b2": float(b2),
        "A2": A2,
        "phi2": float(phi2),
        "lambda_two": lam,
        "success": bool(res.success),
        "message": str(res.message),
        "n_iter": int(getattr(res, "nit", -1)),
        "optimizer": "polar_LBFGSB",
        "amplitude1_bound_active": bool(abs(A1 - A1_MAX) <= 1e-5),
        "amplitude2_bound_active": bool(abs(A2 - A2_MAX) <= 1e-5),
    }


# =============================================================================
# GPU projected Newton fits
# =============================================================================

def make_gpu_cache(ell, B, k1, k2_grid):
    ell_g = cp.asarray(ell, dtype=cp.float64)
    B_g = cp.asarray(B, dtype=cp.float64)
    k2_g = cp.asarray(k2_grid, dtype=cp.float64)

    one = cp.ones_like(ell_g)
    c1 = cp.cos(k1 * ell_g)
    s1 = cp.sin(k1 * ell_g)

    phase2 = k2_g[:, None] * ell_g[None, :]
    c2 = cp.cos(phase2)
    s2 = cp.sin(phase2)

    X_base = cp.stack([one, c1, s1], axis=1)  # N x 3

    one_k = cp.ones_like(c2)
    c1_k = cp.broadcast_to(c1[None, :], c2.shape)
    s1_k = cp.broadcast_to(s1[None, :], c2.shape)

    X_two = cp.stack([one_k, c1_k, s1_k, c2, s2], axis=2)  # K x N x 5

    return {
        "B_g": B_g,
        "k2_g": k2_g,
        "k2_grid": np.asarray(k2_grid, dtype=float),
        "X_base": X_base,
        "X_two": X_two,
    }


def poisson_deviance_gpu_batch(N_batch, lam_batch):
    lam = cp.maximum(lam_batch, 1e-12)
    N = N_batch
    term = lam - N
    nz = N > 0
    term = cp.where(nz, term + N * cp.log(cp.maximum(N, 1e-12) / lam), term)
    return 2.0 * cp.sum(term, axis=1)


def project_base(beta):
    ab = beta[:, 1:3]
    A = cp.sqrt(cp.sum(ab * ab, axis=1))
    scale = cp.minimum(1.0, A1_MAX / cp.maximum(A, 1e-12))
    beta[:, 1] *= scale
    beta[:, 2] *= scale
    return beta


def project_two(beta):
    # beta: K x B x 5
    A1 = cp.sqrt(beta[:, :, 1] ** 2 + beta[:, :, 2] ** 2)
    s1 = cp.minimum(1.0, A1_MAX / cp.maximum(A1, 1e-12))
    beta[:, :, 1] *= s1
    beta[:, :, 2] *= s1

    A2 = cp.sqrt(beta[:, :, 3] ** 2 + beta[:, :, 4] ** 2)
    s2 = cp.minimum(1.0, A2_MAX / cp.maximum(A2, 1e-12))
    beta[:, :, 3] *= s2
    beta[:, :, 4] *= s2

    return beta


def gpu_fit_base_batch(N_batch_cpu, B, ell, k1, cache):
    """
    Fit base model for each batch sample.

    Returns:
        D_base shape B
        beta_base shape B x 3
        lambda_base shape B x N
    """
    N_batch = cp.asarray(N_batch_cpu, dtype=cp.float64)
    if N_batch.ndim == 1:
        N_batch = N_batch[None, :]

    B_g = cache["B_g"]
    X = cache["X_base"]  # N x 3

    Bsz = int(N_batch.shape[0])
    beta = cp.zeros((Bsz, 3), dtype=cp.float64)

    eye = cp.eye(3, dtype=cp.float64)[None, :, :]

    for _ in range(IRLS_ITERS):
        eta = cp.einsum("np,bp->bn", X, beta)
        eta = cp.clip(eta, -ETA_CLIP, ETA_CLIP)
        mu = B_g[None, :] * cp.exp(eta)

        resid = mu - N_batch
        grad = cp.einsum("np,bn->bp", X, resid)

        H = cp.einsum("np,bn,nq->bpq", X, mu, X)
        H = H + IRLS_RIDGE * eye

        step = cp.linalg.solve(H, grad[:, :, None])[:, :, 0]
        beta = beta - step
        beta = project_base(beta)

    eta = cp.einsum("np,bp->bn", X, beta)
    eta = cp.clip(eta, -ETA_CLIP, ETA_CLIP)
    lam = B_g[None, :] * cp.exp(eta)
    D = poisson_deviance_gpu_batch(N_batch, lam)

    return {
        "D_base": D,
        "beta_base": beta,
        "lambda_base": lam,
    }


def gpu_scan_two_batch(N_batch_cpu, B, ell, k1, k2_grid, cache):
    """
    For each batch sample:
        fit base model
        scan two-mode model over all k2
        return max DeltaD_add = D_base - D_two_best
    """
    N_batch = cp.asarray(N_batch_cpu, dtype=cp.float64)
    if N_batch.ndim == 1:
        N_batch = N_batch[None, :]

    B_g = cache["B_g"]
    X = cache["X_two"]  # K x N x 5
    k2_g = cache["k2_g"]

    Bsz = int(N_batch.shape[0])
    Ksz = int(X.shape[0])

    base = gpu_fit_base_batch(cp.asnumpy(N_batch), B, ell, k1, cache)
    D_base = base["D_base"]  # B

    beta = cp.zeros((Ksz, Bsz, 5), dtype=cp.float64)

    eye = cp.eye(5, dtype=cp.float64)[None, None, :, :]

    for _ in range(IRLS_ITERS):
        eta = cp.einsum("knp,kbp->kbn", X, beta)
        eta = cp.clip(eta, -ETA_CLIP, ETA_CLIP)
        mu = B_g[None, None, :] * cp.exp(eta)

        resid = mu - N_batch[None, :, :]
        grad = cp.einsum("knp,kbn->kbp", X, resid)

        H = cp.einsum("knp,kbn,knq->kbpq", X, mu, X)
        H = H + IRLS_RIDGE * eye

        step = cp.linalg.solve(H, grad[..., None])[..., 0]
        beta = beta - step
        beta = project_two(beta)

    eta = cp.einsum("knp,kbp->kbn", X, beta)
    eta = cp.clip(eta, -ETA_CLIP, ETA_CLIP)
    mu = B_g[None, None, :] * cp.exp(eta)

    lam = cp.maximum(mu, 1e-12)
    N_exp = N_batch[None, :, :]

    term = lam - N_exp
    nz = N_exp > 0
    term = cp.where(
        nz,
        term + N_exp * cp.log(cp.maximum(N_exp, 1e-12) / lam),
        term,
    )
    D_two = 2.0 * cp.sum(term, axis=2)  # K x B

    delta_add = D_base[None, :] - D_two

    best_idx = cp.argmax(delta_add, axis=0)
    best_delta = delta_add[best_idx, cp.arange(Bsz)]
    best_k2 = k2_g[best_idx]

    return {
        "D_base": D_base,
        "D_two": D_two,
        "delta_add": delta_add,
        "beta_two": beta,
        "best_idx": cp.asnumpy(best_idx),
        "best_delta": cp.asnumpy(best_delta),
        "best_k2": cp.asnumpy(best_k2),
        "cache": cache,
    }


# =============================================================================
# Run
# =============================================================================
def run_two_mode_test(q2_values, rng):
    data = make_binned_counts(q2_values, mode=RUN_MODE)

    N = data["N"]
    B = data["B"]
    ell = data["ell"]

    # -------------------------------------------------------------------------
    # Exact CPU base fit
    # -------------------------------------------------------------------------

    base_exact = fit_base_cpu_bounded(N, B, ell, K1_FIXED)
    D_base_exact = float(base_exact["D_base"])

    # -------------------------------------------------------------------------
    # Exact CPU local scan around the WCT/reference high-k window
    #
    # This replaces the broken/flat GPU real-data scan.
    # The GPU path is still used later for the null bootstrap.
    # -------------------------------------------------------------------------

    exact_k2_grid = np.linspace(K2_MIN, K2_MAX, N_K2)

    exact_rows = []

    print(f"[exact] scanning k2 window [{exact_k2_grid[0]}, {exact_k2_grid[-1]}]")

    for idx, k2 in enumerate(exact_k2_grid):
        r = fit_two_cpu_bounded(N, B, ell, K1_FIXED, float(k2))
        delta = D_base_exact - r["D_two"]

        exact_rows.append({
            "k2": float(k2),
            "deltaD_add_exact": float(delta),
            "D_base_exact": float(D_base_exact),
            "D_two_exact": float(r["D_two"]),

            "C": float(r["C"]),

            "a1": float(r["a1"]),
            "b1": float(r["b1"]),
            "A1": float(r["A1"]),
            "phi1": float(r["phi1"]),

            "a2": float(r["a2"]),
            "b2": float(r["b2"]),
            "A2": float(r["A2"]),
            "phi2": float(r["phi2"]),

            "success": bool(r["success"]),
            "message": str(r["message"]),
            "n_iter": int(r.get("n_iter", -1)),

            "amplitude1_bound_active": bool(r["amplitude1_bound_active"]),
            "amplitude2_bound_active": bool(r["amplitude2_bound_active"]),
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(exact_k2_grid):
            print(f"  exact scan {idx + 1}/{len(exact_k2_grid)}")

    scan_df = pd.DataFrame(exact_rows)

    best_idx = int(scan_df["deltaD_add_exact"].idxmax())
    best_k2_exact = float(scan_df.loc[best_idx, "k2"])

    best_two = fit_two_cpu_bounded(N, B, ell, K1_FIXED, best_k2_exact)
    ref_two = fit_two_cpu_bounded(N, B, ell, K1_FIXED, REFERENCE_K2)

    delta_best = D_base_exact - best_two["D_two"]
    delta_ref = D_base_exact - ref_two["D_two"]

    near_reference = abs(best_two["k2"] - REFERENCE_K2) <= REFERENCE_TOL
    edge_limited = (
        abs(best_two["k2"] - exact_k2_grid[0]) <= K_EDGE_TOL
        or abs(best_two["k2"] - exact_k2_grid[-1]) <= K_EDGE_TOL
    )

    print(
        f"[real] base k1={K1_FIXED:.5f}, "
        f"D_base={base_exact['D_base']:.4f}, "
        f"A1={base_exact['A1']:.4f}, "
        f"success={base_exact.get('success')}"
    )
    print(
        f"[real] best local k2={best_two['k2']:.5f}, "
        f"ΔD_add={delta_best:.4f}, "
        f"A1={best_two['A1']:.4f}, "
        f"A2={best_two['A2']:.4f}, "
        f"near_ref={near_reference}, "
        f"edge={edge_limited}, "
        f"success={best_two.get('success')}"
    )
    print(
        f"[ref ] k2={REFERENCE_K2:.5f}, "
        f"ΔD_add_ref={delta_ref:.4f}, "
        f"A1={ref_two['A1']:.4f}, "
        f"A2={ref_two['A2']:.4f}, "
        f"success={ref_two.get('success')}"
    )

    # -------------------------------------------------------------------------
    # GPU null bootstrap over the SAME local k2 window
    #
    # Null:
    #   N_null ~ Poisson(lambda_base)
    #
    # For each null:
    #   fit base + scan second mode over exact_k2_grid window.
    # -------------------------------------------------------------------------

    k2_grid = exact_k2_grid
    cache = make_gpu_cache(ell, B, K1_FIXED, k2_grid)

    lam_base = base_exact["lambda_base"]

    null_best = np.empty(NULL_N, dtype=float)
    null_best_k2 = np.empty(NULL_N, dtype=float)
    null_ref = np.empty(NULL_N, dtype=float)

    ref_idx = int(np.argmin(np.abs(k2_grid - REFERENCE_K2)))

    n_done = 0
    while n_done < NULL_N:
        bsz = min(NULL_BATCH, NULL_N - n_done)

        Y = rng.poisson(
            lam=lam_base[None, :],
            size=(bsz, len(N))
        ).astype(np.float64)

        out_null = gpu_scan_two_batch(Y, B, ell, K1_FIXED, k2_grid, cache)

        best_delta = out_null["best_delta"]
        best_k2 = out_null["best_k2"]
        ref_delta = cp.asnumpy(out_null["delta_add"][ref_idx, :])

        null_best[n_done:n_done + bsz] = best_delta
        null_best_k2[n_done:n_done + bsz] = best_k2
        null_ref[n_done:n_done + bsz] = ref_delta

        n_done += bsz
        if n_done % 500 == 0 or n_done == NULL_N:
            print(f"  null {n_done}/{NULL_N}")

    p_best = p_value(delta_best, null_best)
    z_best = gaussian_sigma_from_p(p_best)

    # Conservative: compare reference k2 against scan-max null over 18..21.
    p_ref = p_value(delta_ref, null_best)
    z_ref = gaussian_sigma_from_p(p_ref)

    # Also compute a less-conservative fixed-k reference p-value.
    p_ref_fixed = p_value(delta_ref, null_ref)
    z_ref_fixed = gaussian_sigma_from_p(p_ref_fixed)

    # -------------------------------------------------------------------------
    # Baseline-wall diagnostics
    # -------------------------------------------------------------------------

    real_scan_min = float(scan_df["deltaD_add_exact"].min())
    real_scan_median = float(scan_df["deltaD_add_exact"].median())
    real_scan_max = float(scan_df["deltaD_add_exact"].max())
    scan_success_fraction = float(scan_df["success"].mean())
    scan_A1_bound_fraction = float(scan_df["amplitude1_bound_active"].mean())
    scan_A2_bound_fraction = float(scan_df["amplitude2_bound_active"].mean())

    # Computed later after null is available, but defined here for clarity.
    null_best_p99 = float(np.percentile(null_best, 99))
    null_best_max = float(np.max(null_best))
    wall_ratio_vs_null_p99 = float(real_scan_min / max(null_best_p99, 1e-12))
    wall_ratio_vs_null_max = float(real_scan_min / max(null_best_max, 1e-12))
    baseline_mismatch_wall = bool(
        real_scan_min > 10.0 * max(null_best_p99, 1e-12)
        and scan_A2_bound_fraction > 0.80
    )

    # -------------------------------------------------------------------------
    # Save exact scan + null
    # -------------------------------------------------------------------------

    scan_path = OUT_DIR / f"two_mode_scan_{RUN_MODE}.csv"
    scan_df.to_csv(scan_path, index=False)

    null_path = OUT_DIR / f"two_mode_null_{RUN_MODE}.csv"
    pd.DataFrame({
        "null_best_deltaD_add": null_best,
        "null_best_k2": null_best_k2,
        "null_reference_deltaD_add": null_ref,
    }).to_csv(null_path, index=False)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(scan_df["k2"], scan_df["deltaD_add_exact"], lw=1.5)
    ax.axvline(
        best_two["k2"],
        color="red",
        ls="--",
        label=f"best local k2={best_two['k2']:.4f}"
    )
    ax.axvline(
        REFERENCE_K2,
        color="black",
        ls=":",
        label=f"ref k2={REFERENCE_K2:.4f}"
    )
    ax.set_xlabel(r"$k_2$")
    ax.set_ylabel(r"$\Delta D_{\rm add}$ exact")
    ax.set_title(
        f"Exact two-mode bounded Poisson scan | "
        f"p_best={p_best:.4g}, p_ref={p_ref:.4g}"
    )
    ax.legend()
    fig.tight_layout()

    scan_png = OUT_DIR / f"two_mode_scan_{RUN_MODE}.png"
    fig.savefig(scan_png, dpi=160)
    plt.close(fig)

    q2 = data["q2"]

    lam_base_fit = base_exact["lambda_base"]
    lam_best_two = best_two["lambda_two"]
    lam_ref_two = ref_two["lambda_two"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.step(q2, N, where="mid", label="counts")
    ax.plot(q2, B, lw=1.2, label=f"baseline ({BASELINE_MODE})")
    ax.plot(q2, lam_base_fit, lw=1.8, label=f"base k1={K1_FIXED:.3f}")
    ax.plot(q2, lam_best_two, lw=1.8, label=f"two-mode best k2={best_two['k2']:.3f}")
    ax.plot(q2, lam_ref_two, lw=1.4, ls="--", label=f"two-mode ref k2={REFERENCE_K2:.3f}")
    ax.set_xlabel(r"$q^2$")
    ax.set_ylabel("counts / bin")
    ax.set_title("Bounded Poisson: base vs two-mode")
    ax.legend()
    fig.tight_layout()

    fit_png = OUT_DIR / f"two_mode_fit_{RUN_MODE}.png"
    fig.savefig(fit_png, dpi=160)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Verdict
    # -------------------------------------------------------------------------

    if p_ref <= 0.05:
        label = "REFERENCE_K_SURVIVES_AFTER_LOW_K"
        reason = (
            f"Adding k2={REFERENCE_K2} after base k1={K1_FIXED} beats the local "
            f"scan-max parametric null over the configured local k2 window with p={p_ref:.5f}."
        )
    elif p_best <= 0.05 and near_reference:
        label = "BEST_SECOND_MODE_MATCHES_REFERENCE"
        reason = (
            f"Best local second mode k2={best_two['k2']:.5f} is near reference "
            f"and significant."
        )
    elif p_best <= 0.05:
        label = "SECOND_MODE_SIGNIFICANT_DIFFERENT_K"
        reason = (
            f"Second mode is significant in local window, but best k2={best_two['k2']:.5f} "
            f"is not near reference k2={REFERENCE_K2}."
        )
    else:
        label = "NO_SIGNIFICANT_SECOND_MODE"
        reason = (
            f"After fitting base k1={K1_FIXED}, no second mode beats the local "
            f"parametric null over the configured local k2 window."
        )

    if baseline_mismatch_wall:
        label = label + "__BASELINE_MISMATCH_WALL"
        reason += (
            f" WARNING: the minimum real ΔD across the scan is {real_scan_min:.3g}, "
            f"which is {wall_ratio_vs_null_p99:.1f}x the null p99; this indicates "
            "a broad baseline-mismatch wall rather than a localized k peak."
        )

    result = {
        "test": "two_mode_kde_baseline_polar_exact_local_scan",
        "mode": RUN_MODE,
        "model_base": "B exp(C + a1 cos(k1 ln q2) + b1 sin(k1 ln q2))",
        "model_two": "B exp(C + mode1 + a2 cos(k2 ln q2) + b2 sin(k2 ln q2))",

        "k1_fixed": K1_FIXED,
        "reference_k2": REFERENCE_K2,
        "reference_tol": REFERENCE_TOL,

        "k2_scan": [float(exact_k2_grid[0]), float(exact_k2_grid[-1])],
        "n_k2": int(len(exact_k2_grid)),

        "A1_MAX": A1_MAX,
        "A2_MAX": A2_MAX,
        "eta_clip": ETA_CLIP,

        "null_n": NULL_N,
        "seed": SEED,

        "gpu": {
            "cupy_available": bool(GPU_AVAILABLE),
            "use_cupy": bool(USE_CUPY),
            "using_gpu": bool(USE_CUPY and GPU_AVAILABLE),
            "null_batch": int(NULL_BATCH),
            "gpu_used_for_null_only": True,
            "real_scan": "exact_cpu_polar_LBFGSB",
        },

        "data": {
            "data_glob": DATA_GLOB,
            "tree": TREE_NAME,
            "q2_bins": Q2_BINS,
            "baseline_mode": str(data.get("baseline_mode")),
            "kde_bw_method": data.get("kde_bw_method"),
            "vetoes": {"JPSI": list(JPSI_VETO), "PSI2S": list(PSI2S_VETO)},
            "n_fit_bins": int(len(N)),
            "n_events_after_mass_cuts_before_charmonium_handling": int(len(q2_values)),
        },

        "base": {
            **{k: v for k, v in base_exact.items() if k != "lambda_base"},
        },

        "best_two": {
            **{k: v for k, v in best_two.items() if k != "lambda_two"},
            "deltaD_add": float(delta_best),
            "p_scan_max_null": p_best,
            "z_scan_max_null": z_best,
            "near_reference_k2": bool(near_reference),
            "delta_from_reference_k2": float(best_two["k2"] - REFERENCE_K2),
            "edge_limited": bool(edge_limited),
        },

        "reference_two": {
            **{k: v for k, v in ref_two.items() if k != "lambda_two"},
            "deltaD_add": float(delta_ref),
            "p_vs_local_scan_max_null": p_ref,
            "z_vs_local_scan_max_null": z_ref,
            "p_vs_fixed_reference_null": p_ref_fixed,
            "z_vs_fixed_reference_null": z_ref_fixed,
        },

        "null": {
            "best_deltaD_mean": float(np.mean(null_best)),
            "best_deltaD_p95": float(np.percentile(null_best, 95)),
            "best_deltaD_p99": float(np.percentile(null_best, 99)),

            "reference_deltaD_mean": float(np.mean(null_ref)),
            "reference_deltaD_p95": float(np.percentile(null_ref, 95)),
            "reference_deltaD_p99": float(np.percentile(null_ref, 99)),
        },

        "diagnostics": {
            "real_scan_min_deltaD": real_scan_min,
            "real_scan_median_deltaD": real_scan_median,
            "real_scan_max_deltaD": real_scan_max,
            "scan_success_fraction": scan_success_fraction,
            "scan_A1_bound_fraction": scan_A1_bound_fraction,
            "scan_A2_bound_fraction": scan_A2_bound_fraction,
            "null_best_p99": null_best_p99,
            "null_best_max": null_best_max,
            "wall_ratio_vs_null_p99": wall_ratio_vs_null_p99,
            "wall_ratio_vs_null_max": wall_ratio_vs_null_max,
            "baseline_mismatch_wall": baseline_mismatch_wall,
        },

        "files": {
            "scan_csv": str(scan_path),
            "null_csv": str(null_path),
            "scan_png": str(scan_png),
            "fit_png": str(fit_png),
        },

        "verdict": {
            "label": label,
            "reason": reason,
        },
    }

    out_path = OUT_DIR / "two_mode_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 70)
    print("TWO-MODE BOUNDED POISSON VERDICT")
    print("=" * 70)
    print(json.dumps(result["verdict"], indent=2))

    print("\nResult:")
    print(
        f"base k1={K1_FIXED:.5f}, "
        f"D_base={base_exact['D_base']:.4f}, "
        f"A1={base_exact['A1']:.4f}, "
        f"success={base_exact.get('success')}"
    )
    print(
        f"best local k2={best_two['k2']:.5f}, "
        f"ΔD_add={delta_best:.4f}, "
        f"p_best={p_best:.5f}, "
        f"z_best={z_best}, "
        f"A2={best_two['A2']:.4f}, "
        f"near_ref={near_reference}, "
        f"success={best_two.get('success')}"
    )
    print(
        f"ref k2={REFERENCE_K2:.5f}, "
        f"ΔD_add_ref={delta_ref:.4f}, "
        f"p_ref_scanmax={p_ref:.5f}, "
        f"z_ref_scanmax={z_ref}, "
        f"p_ref_fixed={p_ref_fixed:.5f}, "
        f"z_ref_fixed={z_ref_fixed}, "
        f"A2_ref={ref_two['A2']:.4f}, "
        f"success={ref_two.get('success')}"
    )

    print(f"\nSaved: {out_path}")

# =============================================================================
# Main
# =============================================================================

def main():
    print(f"[gpu] CuPy available: {GPU_AVAILABLE}")
    print(f"[gpu] USE_CUPY: {USE_CUPY}")
    print(f"[gpu] using GPU: {USE_CUPY and GPU_AVAILABLE}")
    print(f"[gpu] NULL_BATCH: {NULL_BATCH}")
    print(f"[config] mode={RUN_MODE}")
    print(f"[config] k1_fixed={K1_FIXED}")
    print(f"[config] reference_k2={REFERENCE_K2}")
    print(f"[config] k2_scan=[{K2_MIN}, {K2_MAX}], N_K2={N_K2}")
    print(f"[config] A1_MAX={A1_MAX}, A2_MAX={A2_MAX}")
    print(f"[config] baseline_mode={BASELINE_MODE}, vetoes JPSI={JPSI_VETO}, PSI2S={PSI2S_VETO}")
    print(f"[config] KDE_BANDWIDTH_SCALE={KDE_BANDWIDTH_SCALE}")

    rng = np.random.default_rng(SEED)

    df = load_events()
    q2_values = df["q2"].to_numpy()

    print(f"[info] selected events before charmonium handling: {len(q2_values):,}")

    run_two_mode_test(q2_values, rng)


if __name__ == "__main__":
    main()

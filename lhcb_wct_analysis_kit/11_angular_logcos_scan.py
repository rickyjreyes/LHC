"""
11_angular_logcos_scan.py

Angular log-cos scan for derived B0 -> K*0 mu+ mu- angles.

CuPy update:
    GPU-accelerates the k-scan and batched shuffle-null scans.

Main speedup:
    Instead of doing 5000 nulls x 2200 k values in Python loops,
    this batches null permutations and solves all k fits with GPU matrix algebra.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cupy as cp
    GPU_AVAILABLE = True
except Exception:
    cp = None
    GPU_AVAILABLE = False


# =============================================================================
# Config
# =============================================================================

INPUT_PARQUET = Path("outputs_angles/angles.parquet")
INPUT_CSV = Path("outputs_angles/angles.csv")

OUT_DIR = Path("outputs_angular_logcos")
OUT_DIR.mkdir(exist_ok=True, parents=True)

Q2_MIN = 0.1
Q2_MAX = 19.0

JPSI_VETO = (8.68, 10.09)
PSI2S_VETO = (12.86, 14.18)

N_Q2_BINS = 48
MIN_EVENTS_PER_BIN = 50

# Extended lower bound because prior angular best modes hit k=2.0 floor.
# K_MIN = 0.5
# K_MAX = 30.0
# N_K = 2200
K_EDGE_TOL = 0.05

K_MIN = 0.05
K_MAX = 5.0
N_K = 2000

NULL_N = 5000
SEED = 12345

# GPU controls.
USE_CUPY = True
NULL_BATCH = 512

REFERENCE_K = 19.53
REFERENCE_TOL = 0.50


# =============================================================================
# Data helpers
# =============================================================================

def load_angles() -> pd.DataFrame:
    if INPUT_PARQUET.exists():
        print(f"[load] {INPUT_PARQUET}")
        return pd.read_parquet(INPUT_PARQUET)

    if INPUT_CSV.exists():
        print(f"[load] {INPUT_CSV}")
        return pd.read_csv(INPUT_CSV)

    raise FileNotFoundError(
        "Missing outputs_angles/angles.parquet and outputs_angles/angles.csv"
    )


def in_veto(q2: np.ndarray) -> np.ndarray:
    return (
        ((q2 >= JPSI_VETO[0]) & (q2 <= JPSI_VETO[1]))
        | ((q2 >= PSI2S_VETO[0]) & (q2 <= PSI2S_VETO[1]))
    )


def safe_mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.mean(x))


def safe_sem(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def build_moments(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "q2",
        "cosThetaL",
        "cosThetaK",
        "phi",
        "passes_signal_selection",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required angle columns: {missing}")

    d = df.copy()

    d = d[d["passes_signal_selection"].astype(bool)]
    d = d[np.isfinite(d["q2"])]
    d = d[(d["q2"] >= Q2_MIN) & (d["q2"] <= Q2_MAX)]
    d = d[~in_veto(d["q2"].to_numpy())]

    print(f"[select] events after signal selection and veto: {len(d):,}")

    cL = d["cosThetaL"].to_numpy(float)
    cK = d["cosThetaK"].to_numpy(float)
    phi = d["phi"].to_numpy(float)

    cL = np.clip(cL, -1.0, 1.0)
    cK = np.clip(cK, -1.0, 1.0)

    sL = np.sqrt(np.maximum(1.0 - cL * cL, 0.0))
    sK = np.sqrt(np.maximum(1.0 - cK * cK, 0.0))

    sin2thetaK = 2.0 * sK * cK

    d["moment_cosThetaL"] = cL
    d["moment_cosThetaK"] = cK
    d["moment_cosPhi"] = np.cos(phi)
    d["moment_sinPhi"] = np.sin(phi)
    d["moment_cos2Phi"] = np.cos(2.0 * phi)
    d["moment_sin2Phi"] = np.sin(2.0 * phi)

    # P'_5-sensitive proxy.
    d["moment_m5_proxy"] = sin2thetaK * sL * np.cos(phi)

    # Shape moments.
    d["moment_cosThetaL2"] = cL * cL
    d["moment_cosThetaK2"] = cK * cK

    q2_edges = np.linspace(Q2_MIN, Q2_MAX, N_Q2_BINS + 1)

    rows = []
    for lo, hi in zip(q2_edges[:-1], q2_edges[1:]):
        b = d[(d["q2"] >= lo) & (d["q2"] < hi)]
        n = len(b)

        if n < MIN_EVENTS_PER_BIN:
            continue

        q2_center = 0.5 * (lo + hi)
        ell = np.log(q2_center)

        row = {
            "q2_lo": float(lo),
            "q2_hi": float(hi),
            "q2_center": float(q2_center),
            "ell": float(ell),
            "n": int(n),
        }

        for col in [
            "moment_cosThetaL",
            "moment_cosThetaK",
            "moment_cosPhi",
            "moment_sinPhi",
            "moment_cos2Phi",
            "moment_sin2Phi",
            "moment_m5_proxy",
            "moment_cosThetaL2",
            "moment_cosThetaK2",
        ]:
            x = b[col].to_numpy(float)
            row[col] = safe_mean(x)
            row[col + "_sem"] = safe_sem(x)

        rows.append(row)

    moments = pd.DataFrame(rows)
    if len(moments) < 8:
        raise RuntimeError(
            f"Too few usable q2 bins: {len(moments)}. "
            "Lower MIN_EVENTS_PER_BIN or N_Q2_BINS."
        )

    moments.to_csv(OUT_DIR / "angular_moments.csv", index=False)

    print(f"[moments] usable bins: {len(moments)}")
    print(f"[save] {OUT_DIR / 'angular_moments.csv'}")

    return moments


# =============================================================================
# CPU fallback fitting
# =============================================================================

def weighted_lstsq(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    sw = np.sqrt(np.maximum(w, 0.0))
    Xw = X * sw[:, None]
    yw = y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ beta
    chi2 = float(np.sum(w * resid * resid))
    return beta, chi2


def fit_for_k_cpu(ell: np.ndarray, y: np.ndarray, w: np.ndarray, k: float) -> dict:
    X0 = np.ones((len(ell), 1))
    beta0, chi0 = weighted_lstsq(X0, y, w)

    X1 = np.column_stack([
        np.ones_like(ell),
        np.cos(k * ell),
        np.sin(k * ell),
    ])
    beta1, chi1 = weighted_lstsq(X1, y, w)

    C, a, b = beta1
    A = float(np.sqrt(a * a + b * b))
    phi = float(np.arctan2(-b, a))

    return {
        "k": float(k),
        "delta_chi2": float(chi0 - chi1),
        "chi2_null": float(chi0),
        "chi2_cos": float(chi1),
        "C": float(C),
        "a": float(a),
        "b": float(b),
        "A": A,
        "phi": phi,
    }


def scan_k_cpu(ell: np.ndarray, y: np.ndarray, w: np.ndarray, k_grid: np.ndarray):
    rows = [fit_for_k_cpu(ell, y, w, k) for k in k_grid]
    best = max(rows, key=lambda r: r["delta_chi2"])
    return rows, best


# =============================================================================
# CuPy batched fitting
# =============================================================================

def make_gpu_cache(ell: np.ndarray, w: np.ndarray, k_grid: np.ndarray):
    """
    Precompute the weighted normal-equation matrices for all k.

    Model:
        y = C + a cos(k ell) + b sin(k ell)
    """
    if not (USE_CUPY and GPU_AVAILABLE):
        return None

    ell_g = cp.asarray(ell, dtype=cp.float64)
    w_g = cp.asarray(w, dtype=cp.float64)
    k_g = cp.asarray(k_grid, dtype=cp.float64)

    phase = k_g[:, None] * ell_g[None, :]
    c = cp.cos(phase)
    s = cp.sin(phase)
    one = cp.ones_like(c)

    def dotw(a, b):
        return cp.sum(w_g[None, :] * a * b, axis=1)

    A00 = dotw(one, one)
    A01 = dotw(one, c)
    A02 = dotw(one, s)
    A11 = dotw(c, c)
    A12 = dotw(c, s)
    A22 = dotw(s, s)

    A = cp.stack([
        cp.stack([A00, A01, A02], axis=1),
        cp.stack([A01, A11, A12], axis=1),
        cp.stack([A02, A12, A22], axis=1),
    ], axis=1)

    invA = cp.linalg.inv(A)

    return {
        "ell_g": ell_g,
        "w_g": w_g,
        "k_g": k_g,
        "k_grid": k_grid,
        "c": c,
        "s": s,
        "one": one,
        "invA": invA,
        "sumw": cp.sum(w_g),
    }


def gpu_scan_batch(y_batch: np.ndarray, cache: dict):
    """
    Batched GPU scan.

    y_batch shape:
        (B, N)

    Returns:
        best_delta: shape (B,)
        best_k: shape (B,)
        best_idx: shape (B,)
        optional arrays for beta/chi if needed.
    """
    Y = cp.asarray(y_batch, dtype=cp.float64)
    if Y.ndim == 1:
        Y = Y[None, :]

    w_g = cache["w_g"]
    c = cache["c"]
    s = cache["s"]
    invA = cache["invA"]
    k_g = cache["k_g"]
    sumw = cache["sumw"]

    Bsz = Y.shape[0]
    Ksz = c.shape[0]

    WY = Y * w_g[None, :]

    yWy = cp.sum(WY * Y, axis=1)
    sumWy = cp.sum(WY, axis=1)
    chi0 = yWy - (sumWy * sumWy) / sumw

    # X^T W y for each k and each batch item.
    B0 = cp.broadcast_to(sumWy[None, :], (Ksz, Bsz))
    B1 = c @ WY.T
    B2 = s @ WY.T

    Bvec = cp.stack([B0, B1, B2], axis=2)  # K x B x 3

    beta = cp.einsum("kij,kbj->kbi", invA, Bvec)

    # Weighted least-squares identity:
    # chi2_cos = y^T W y - beta^T X^T W y
    beta_dot_B = cp.sum(beta * Bvec, axis=2)
    chi1 = yWy[None, :] - beta_dot_B

    dchi = chi0[None, :] - chi1

    best_idx = cp.argmax(dchi, axis=0)
    best_delta = dchi[best_idx, cp.arange(Bsz)]
    best_k = k_g[best_idx]

    return {
        "best_delta": cp.asnumpy(best_delta),
        "best_k": cp.asnumpy(best_k),
        "best_idx": cp.asnumpy(best_idx),
        "dchi": dchi,
        "chi0": chi0,
        "chi1": chi1,
        "beta": beta,
    }


def scan_k_gpu(ell: np.ndarray, y: np.ndarray, w: np.ndarray, k_grid: np.ndarray):
    """
    Real-data scan using GPU. Returns rows and best dict.
    """
    cache = make_gpu_cache(ell, w, k_grid)
    if cache is None:
        return scan_k_cpu(ell, y, w, k_grid)

    out = gpu_scan_batch(y[None, :], cache)

    dchi = cp.asnumpy(out["dchi"][:, 0])
    chi1 = cp.asnumpy(out["chi1"][:, 0])
    chi0 = float(cp.asnumpy(out["chi0"])[0])
    beta = cp.asnumpy(out["beta"][:, 0, :])

    best_idx = int(out["best_idx"][0])
    C_best, a_best, b_best = beta[best_idx]
    k_best = float(k_grid[best_idx])
    chi1_best = float(chi1[best_idx])
    dchi_best = float(dchi[best_idx])

    rows = []
    for i, k in enumerate(k_grid):
        C, a, b = beta[i]
        A = float(np.sqrt(a * a + b * b))
        phi = float(np.arctan2(-b, a))
        rows.append({
            "k": float(k),
            "delta_chi2": float(dchi[i]),
            "chi2_null": chi0,
            "chi2_cos": float(chi1[i]),
            "C": float(C),
            "a": float(a),
            "b": float(b),
            "A": A,
            "phi": phi,
        })

    best = {
        "k": k_best,
        "delta_chi2": dchi_best,
        "chi2_null": chi0,
        "chi2_cos": chi1_best,
        "C": float(C_best),
        "a": float(a_best),
        "b": float(b_best),
        "A": float(np.sqrt(a_best * a_best + b_best * b_best)),
        "phi": float(np.arctan2(-b_best, a_best)),
    }

    return rows, best


# =============================================================================
# Null scan
# =============================================================================

def run_null_cpu(
    ell: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    k_grid: np.ndarray,
    rng: np.random.Generator,
    n_null: int,
):
    vals = np.empty(n_null)
    ks = np.empty(n_null)

    for i in range(n_null):
        y_null = rng.permutation(y)
        _, best = scan_k_cpu(ell, y_null, w, k_grid)
        vals[i] = best["delta_chi2"]
        ks[i] = best["k"]

        if (i + 1) % 500 == 0:
            print(f"  null {i + 1}/{n_null}")

    return vals, ks


def run_null_gpu(
    ell: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    k_grid: np.ndarray,
    rng: np.random.Generator,
    n_null: int,
):
    cache = make_gpu_cache(ell, w, k_grid)
    if cache is None:
        return run_null_cpu(ell, y, w, k_grid, rng, n_null)

    vals = np.empty(n_null)
    ks = np.empty(n_null)

    n_done = 0
    while n_done < n_null:
        bsz = min(NULL_BATCH, n_null - n_done)

        # Generate shuffled nulls on CPU, then send one batch to GPU.
        Y = np.empty((bsz, len(y)), dtype=np.float64)
        for j in range(bsz):
            Y[j, :] = rng.permutation(y)

        out = gpu_scan_batch(Y, cache)

        vals[n_done:n_done + bsz] = out["best_delta"]
        ks[n_done:n_done + bsz] = out["best_k"]

        n_done += bsz

        if n_done % 500 == 0 or n_done == n_null:
            print(f"  null {n_done}/{n_null}")

    return vals, ks


def run_null(
    ell: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    k_grid: np.ndarray,
    rng: np.random.Generator,
    n_null: int,
):
    if USE_CUPY and GPU_AVAILABLE:
        return run_null_gpu(ell, y, w, k_grid, rng, n_null)

    return run_null_cpu(ell, y, w, k_grid, rng, n_null)


# =============================================================================
# Moment preparation / plotting
# =============================================================================

def p_value(real: float, null_vals: np.ndarray) -> float:
    return float((1 + np.sum(null_vals >= real)) / (1 + len(null_vals)))


def gaussian_sigma_from_p(p: float):
    try:
        from scipy.special import erfcinv
        return float(np.sqrt(2.0) * erfcinv(2.0 * p))
    except Exception:
        return None


def prepare_series(moments: pd.DataFrame, moment_col: str):
    sem_col = moment_col + "_sem"

    d = moments[["ell", moment_col, sem_col, "n"]].copy()
    d = d[np.isfinite(d["ell"])]
    d = d[np.isfinite(d[moment_col])]

    ell = d["ell"].to_numpy(float)
    y = d[moment_col].to_numpy(float)

    y = y - np.mean(y)

    if sem_col in d.columns:
        sem = d[sem_col].to_numpy(float)
        good = np.isfinite(sem) & (sem > 0)
        if np.sum(good) >= max(5, len(sem) // 2):
            w = np.ones_like(y)
            w[good] = 1.0 / (sem[good] ** 2)
            w[~good] = np.nanmedian(w[good])
        else:
            w = d["n"].to_numpy(float)
    else:
        w = d["n"].to_numpy(float)

    w = w / np.nanmedian(w)

    return ell, y, w


def plot_result(
    moment_col: str,
    ell: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    scan_df: pd.DataFrame,
    best: dict,
    p: float,
):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(scan_df["k"], scan_df["delta_chi2"], lw=1.6)
    ax.axvline(best["k"], color="red", ls="--", label=f"k*={best['k']:.3f}")
    ax.axvline(REFERENCE_K, color="black", ls=":", label=f"yield k≈{REFERENCE_K}")
    ax.axvline(K_MIN, color="gray", ls=":", lw=0.9, label=f"K_MIN={K_MIN}")
    ax.axvline(K_MAX, color="gray", ls=":", lw=0.9, label=f"K_MAX={K_MAX}")
    ax.set_xlabel(r"$k_\ell$")
    ax.set_ylabel(r"$\Delta \chi^2$")
    ax.set_title(f"Angular log-cos scan: {moment_col} | p={p:.4g}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"angular_logcos_scan_{moment_col}.png", dpi=160)
    plt.close(fig)

    ell_plot = np.linspace(float(np.min(ell)), float(np.max(ell)), 600)
    y_fit = (
        best["C"]
        + best["a"] * np.cos(best["k"] * ell_plot)
        + best["b"] * np.sin(best["k"] * ell_plot)
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(ell, y, s=30, label="moment bins")
    ax.plot(ell_plot, y_fit, color="red", lw=1.8, label="best cos fit")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel(r"$\ell=\ln(q^2)$")
    ax.set_ylabel(f"{moment_col} residual")
    ax.set_title(f"{moment_col}: cos({best['k']:.3f} ell + phi)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"angular_logcos_fit_{moment_col}.png", dpi=160)
    plt.close(fig)


# =============================================================================
# Per-moment run
# =============================================================================

def run_moment(moments: pd.DataFrame, moment_col: str, rng: np.random.Generator) -> dict:
    print("\n" + "=" * 80)
    print(f"[moment] {moment_col}")
    print("=" * 80)

    ell, y, w = prepare_series(moments, moment_col)
    k_grid = np.linspace(K_MIN, K_MAX, N_K)

    if USE_CUPY and GPU_AVAILABLE:
        rows, best = scan_k_gpu(ell, y, w, k_grid)
    else:
        rows, best = scan_k_cpu(ell, y, w, k_grid)

    scan_df = pd.DataFrame(rows)
    scan_path = OUT_DIR / f"angular_logcos_scan_{moment_col}.csv"
    scan_df.to_csv(scan_path, index=False)

    print(
        f"[real] k={best['k']:.4f}, "
        f"period={2*np.pi/best['k']:.4f}, "
        f"Δχ²={best['delta_chi2']:.4f}, "
        f"A={best['A']:.4f}"
    )

    null_vals, null_ks = run_null(ell, y, w, k_grid, rng, NULL_N)

    p = p_value(best["delta_chi2"], null_vals)
    z = gaussian_sigma_from_p(p)

    null_path = OUT_DIR / f"angular_logcos_null_{moment_col}.csv"
    pd.DataFrame({
        "null_delta_chi2": null_vals,
        "null_best_k": null_ks,
    }).to_csv(null_path, index=False)

    near_reference = abs(best["k"] - REFERENCE_K) <= REFERENCE_TOL

    edge_limited = (
        abs(best["k"] - K_MIN) <= K_EDGE_TOL
        or abs(best["k"] - K_MAX) <= K_EDGE_TOL
    )

    plot_result(moment_col, ell, y, w, scan_df, best, p)

    result = {
        "moment": moment_col,
        "n_bins": int(len(ell)),
        "best": {
            **best,
            "period": float(2.0 * np.pi / best["k"]),
            "p_shuffle": p,
            "z_shuffle_one_sided": z,
            "null_delta_chi2_mean": float(np.mean(null_vals)),
            "null_delta_chi2_p95": float(np.percentile(null_vals, 95)),
            "null_delta_chi2_p99": float(np.percentile(null_vals, 99)),
            "near_reference_k": bool(near_reference),
            "delta_from_reference_k": float(best["k"] - REFERENCE_K),
            "edge_limited": bool(edge_limited),
            "distance_from_k_min": float(best["k"] - K_MIN),
            "distance_from_k_max": float(K_MAX - best["k"]),
        },
        "files": {
            "scan_csv": str(scan_path),
            "null_csv": str(null_path),
            "scan_png": str(OUT_DIR / f"angular_logcos_scan_{moment_col}.png"),
            "fit_png": str(OUT_DIR / f"angular_logcos_fit_{moment_col}.png"),
        },
    }

    print(
        f"[null] p={p:.5f}, z={z}, "
        f"near_yield_k={near_reference}, "
        f"delta_k={best['k'] - REFERENCE_K:.4f}, "
        f"edge_limited={edge_limited}"
    )

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"[gpu] CuPy available: {GPU_AVAILABLE}")
    print(f"[gpu] USE_CUPY: {USE_CUPY}")
    print(f"[gpu] using GPU: {USE_CUPY and GPU_AVAILABLE}")
    print(f"[gpu] NULL_BATCH: {NULL_BATCH}")

    rng = np.random.default_rng(SEED)

    df = load_angles()
    moments = build_moments(df)

    moment_cols = [
        "moment_cosThetaL",
        "moment_cosThetaK",
        "moment_cosPhi",
        "moment_sinPhi",
        "moment_cos2Phi",
        "moment_sin2Phi",
        "moment_m5_proxy",
        "moment_cosThetaL2",
        "moment_cosThetaK2",
    ]

    results = {}
    for col in moment_cols:
        results[col] = run_moment(moments, col, rng)

    passed = []
    near_yield = []
    edge_limited_passed = []

    for col, obj in results.items():
        b = obj["best"]

        if b["p_shuffle"] <= 0.05:
            passed.append(col)

            if b.get("edge_limited", False):
                edge_limited_passed.append(col)

        if b["p_shuffle"] <= 0.05 and b["near_reference_k"]:
            near_yield.append(col)

    if len(near_yield) >= 1:
        label = "ANGULAR_LOGCOS_MATCHES_YIELD_MODE"
        reason = (
            "At least one angular moment passes p<=0.05 and lands near the "
            f"yield reference k={REFERENCE_K}."
        )
    elif len(passed) >= 1:
        label = "ANGULAR_LOGCOS_PRESENT_DIFFERENT_K"
        reason = (
            "At least one angular moment passes p<=0.05, but not near the "
            f"yield reference k={REFERENCE_K}."
        )
    else:
        label = "NO_SIGNIFICANT_ANGULAR_LOGCOS"
        reason = "No angular moment beats the shuffle null at p<=0.05."

    if edge_limited_passed:
        reason += (
            " Some significant moments are edge-limited, so rerun with a wider "
            "k range if needed."
        )

    summary = {
        "test": "angular_logcos_scan",
        "input": str(INPUT_PARQUET if INPUT_PARQUET.exists() else INPUT_CSV),
        "gpu": {
            "cupy_available": bool(GPU_AVAILABLE),
            "use_cupy": bool(USE_CUPY),
            "using_gpu": bool(USE_CUPY and GPU_AVAILABLE),
            "null_batch": int(NULL_BATCH),
        },
        "q2_range": [Q2_MIN, Q2_MAX],
        "vetoes": {
            "JPSI": JPSI_VETO,
            "PSI2S": PSI2S_VETO,
        },
        "n_q2_bins": N_Q2_BINS,
        "min_events_per_bin": MIN_EVENTS_PER_BIN,
        "k_scan": [K_MIN, K_MAX],
        "n_k": N_K,
        "k_edge_tol": K_EDGE_TOL,
        "null_n": NULL_N,
        "seed": SEED,
        "reference_k": REFERENCE_K,
        "reference_tol": REFERENCE_TOL,
        "n_moment_bins": int(len(moments)),
        "moments": results,
        "verdict": {
            "label": label,
            "reason": reason,
            "passed_p_le_0_05": passed,
            "passed_and_near_yield_k": near_yield,
            "passed_but_edge_limited": edge_limited_passed,
        },
    }

    out_path = OUT_DIR / "angular_logcos_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("ANGULAR LOG-COS VERDICT")
    print("=" * 80)
    print(json.dumps(summary["verdict"], indent=2))

    print("\nBest modes:")
    for col, obj in results.items():
        b = obj["best"]
        print(
            f"{col:22s} "
            f"k={b['k']:.4f} "
            f"period={b['period']:.4f} "
            f"Δχ²={b['delta_chi2']:.3f} "
            f"A={b['A']:.4f} "
            f"p={b['p_shuffle']:.5f} "
            f"z={b['z_shuffle_one_sided']} "
            f"near_yield={b['near_reference_k']} "
            f"edge_limited={b.get('edge_limited', False)}"
        )

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
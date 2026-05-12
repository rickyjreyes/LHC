#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WCT Sideband-Subtracted Comb Test
---------------------------------

Purpose:
    Test whether the Koide/trig log-winding comb remains after subtracting
    a B-mass sideband background template.

Signal:
    B in [5230, 5330], K* in [795.9, 995.9]

Background templates:
    B low sideband  [5000, 5180], K* signal
    B high sideband [5380, 5600], K* signal

Template:
    B_temp(ell) = alpha * B_low(ell) + beta * B_high(ell)
    alpha,beta >= 0 fitted to signal histogram.

Residual:
    R(ell) = Y_signal(ell) - B_temp(ell)

Null/base residual model:
    R_base = C + a1 cos(k1 ell) + b1 sin(k1 ell)

WCT alternatives:
    R_alt = R_base + comb modes

Test statistic:
    DeltaChi2 = chi2_base - chi2_alt

Bootstrap:
    Generate Poisson pseudo-signal from fitted sideband-template + base residual.
    Refit alpha,beta and base/alt models.
    Compute p = P_null(DeltaChi2_null >= DeltaChi2_real)

This is a sideband-subtracted diagnostic, not a final LHC likelihood.
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
    from scipy.optimize import minimize, lsq_linear
except Exception as e:
    raise RuntimeError("Missing scipy. Install with: pip install scipy") from e


# ============================================================
# Config
# ============================================================

OUTDIR = "outputs_wct_sideband_subtracted"
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

K1_FIXED = 7.61054
N0 = 15
REFERENCE_K2 = 19.5296

N_BINS = 240
NULL_N = 5000
RNG_SEED = 1661

A_MAX = 0.25  # residual WLS coefficients can be looser than Poisson log-amplitudes

WCT_MODELS = [
    {
        "label": "koide_Q_2over3",
        "description": "spin-1/2 true sideband Koide comb",
        "Qs": [2.0 / 3.0],
    },
    {
        "label": "folded_Q_4over9",
        "description": "spin-3/2 folded/subcentral trig comb",
        "Qs": [4.0 / 9.0],
    },
    {
        "label": "combined_Q_2over3_plus_4over9",
        "description": "combined Koide sideband + folded trig branches",
        "Qs": [2.0 / 3.0, 4.0 / 9.0],
    },
]


# ============================================================
# Winding map
# ============================================================

def active_delta_ell(intervals):
    return sum(math.log(hi / lo) for lo, hi in intervals)


DELTA_ELL_ACTIVE = active_delta_ell(ACTIVE_INTERVALS)


def k_from_n(n):
    return 2.0 * math.pi * n / DELTA_ELL_ACTIVE


def comb_from_Q(Q, n0=N0):
    return np.array([n0 * Q, n0, n0 * 2.0 * Q], dtype=float)


def comb_k_from_Q(Q, n0=N0):
    ns = comb_from_Q(Q, n0)
    ks = np.array([k_from_n(n) for n in ns], dtype=float)
    return ns, ks


def ks_for_model(model):
    all_ns = []
    all_ks = []

    for Q in model["Qs"]:
        ns, ks = comb_k_from_Q(Q, N0)
        all_ns.extend(ns.tolist())
        all_ks.extend(ks.tolist())

    out_ns = []
    out_ks = []
    seen = set()

    for n, k in zip(all_ns, all_ks):
        key = round(k, 12)
        if key not in seen:
            seen.add(key)
            out_ns.append(float(n))
            out_ks.append(float(k))

    return np.array(out_ns), np.array(out_ks)


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

    b_mass_candidates = ["B0_M", "B0_MM", "B_M", "B_MM"]

    kst_mass_candidates = [
        "Kst_892_0_M", "Kst_892_0_MM",
        "Kst_M", "Kst_MM",
        "Kstar_M", "Kstar_MM",
        "Kstar0_M", "Kstar0_MM",
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

    df = pd.concat(rows, ignore_index=True)
    print(f"[info] loaded events in q2 range: {len(df):,}")

    return df


# ============================================================
# Region selection / histograms
# ============================================================

def select_region(df, B_window, Kst_window):
    blo, bhi = B_window
    klo, khi = Kst_window

    mask = (df["B_M"] >= blo) & (df["B_M"] <= bhi)
    mask &= (df["Kst_M"] >= klo) & (df["Kst_M"] <= khi)

    sub = df.loc[mask].copy()
    active = in_active_intervals(sub["q2"].values)
    sub = sub.loc[active].copy()

    return sub


def make_histogram(q2, n_bins=N_BINS):
    ell = np.log(q2)

    ell_min = math.log(Q2_MIN)
    ell_max = math.log(Q2_MAX)

    counts, edges = np.histogram(ell, bins=n_bins, range=(ell_min, ell_max))
    centers = 0.5 * (edges[:-1] + edges[1:])

    q2_centers = np.exp(centers)
    active = in_active_intervals(q2_centers)

    return centers[active], counts[active].astype(float)


# ============================================================
# Sideband template subtraction
# ============================================================

def fit_sideband_template(Y, L, H):
    """
    Fit Y ~= alpha L + beta H with alpha,beta >= 0.

    Weighted least squares with variance approx Y+1.
    """
    Y = np.asarray(Y, dtype=float)
    L = np.asarray(L, dtype=float)
    H = np.asarray(H, dtype=float)

    X = np.vstack([L, H]).T
    sigma = np.sqrt(np.maximum(Y, 1.0))
    Xw = X / sigma[:, None]
    Yw = Y / sigma

    res = lsq_linear(Xw, Yw, bounds=(0.0, np.inf), lsmr_tol="auto")

    alpha, beta = res.x
    Bhat = alpha * L + beta * H

    return float(alpha), float(beta), Bhat


def residual_variance(Y, L, H, alpha, beta):
    """
    Approx variance of R = Y - alpha L - beta H.
    Poisson independent approximation:
        Var(R) = Var(Y) + alpha^2 Var(L) + beta^2 Var(H)
               ≈ Y + alpha^2 L + beta^2 H
    """
    var = Y + alpha * alpha * L + beta * beta * H
    return np.maximum(var, 1.0)


# ============================================================
# Residual weighted LS fitting
# ============================================================

def basis_matrix(ell, ks):
    cols = [np.ones_like(ell)]

    # low-k nuisance
    cols.append(np.cos(K1_FIXED * ell))
    cols.append(np.sin(K1_FIXED * ell))

    for k in ks:
        cols.append(np.cos(k * ell))
        cols.append(np.sin(k * ell))

    return np.vstack(cols).T


def fit_residual_wls(R, var, ell, ks):
    """
    Weighted LS:
        R = X beta + noise
    with coefficient bounds on sinusoidal terms.
    """
    X = basis_matrix(ell, ks)
    y = np.asarray(R, dtype=float)
    sigma = np.sqrt(np.maximum(var, 1.0))

    Xw = X / sigma[:, None]
    yw = y / sigma

    p = X.shape[1]
    bounds_lo = np.full(p, -np.inf)
    bounds_hi = np.full(p, np.inf)

    # leave intercept free, bound sinusoidal coefficients
    bounds_lo[1:] = -A_MAX
    bounds_hi[1:] = A_MAX

    res = lsq_linear(
        Xw,
        yw,
        bounds=(bounds_lo, bounds_hi),
        lsmr_tol="auto",
        max_iter=2000,
    )

    beta = res.x
    pred = X @ beta

    chi2_val = float(np.sum(((y - pred) ** 2) / np.maximum(var, 1.0)))

    amps = []
    # low-k
    A1 = math.sqrt(beta[1] ** 2 + beta[2] ** 2)

    offset = 3
    for j in range(len(ks)):
        a = beta[offset + 2 * j]
        b = beta[offset + 2 * j + 1]
        amps.append(math.sqrt(a * a + b * b))

    bound_active = any(abs(v) >= A_MAX - 1e-8 for v in beta[1:])

    return {
        "success": bool(res.success),
        "chi2": chi2_val,
        "beta": beta,
        "pred": pred,
        "A1": float(A1),
        "amps": [float(x) for x in amps],
        "any_bound_active": bool(bound_active),
        "n_params": int(p),
    }


def fit_base_and_alt(Y, L, H, ell, ks):
    alpha, beta, Bhat = fit_sideband_template(Y, L, H)
    R = Y - Bhat
    var = residual_variance(Y, L, H, alpha, beta)

    base = fit_residual_wls(R, var, ell, ks=[])
    alt = fit_residual_wls(R, var, ell, ks=ks)

    delta = base["chi2"] - alt["chi2"]

    return {
        "alpha": alpha,
        "beta": beta,
        "Bhat": Bhat,
        "R": R,
        "var": var,
        "base": base,
        "alt": alt,
        "delta_chi2": float(delta),
    }


def make_null_signal_counts(rng, L, H, fit_result):
    """
    Generate null signal counts from:
        lambda_null = Bhat + base residual prediction
    clipped positive.
    """
    Bhat = fit_result["Bhat"]
    pred_base = fit_result["base"]["pred"]

    lam = Bhat + pred_base
    lam = np.maximum(lam, 1e-6)

    return rng.poisson(lam)


# ============================================================
# Results
# ============================================================

@dataclass
class SubtractedResult:
    model_label: str
    model_description: str
    n_values: str
    k_values: str
    N_signal: int
    N_B_low: int
    N_B_high: int
    alpha_low: float
    beta_high: float
    chi2_base: float
    chi2_alt: float
    delta_chi2: float
    p_bootstrap: float
    null_mean: float
    null_std: float
    null_95: float
    null_99: float
    null_999: float
    base_A1: float
    alt_amps: str
    alt_bound_active: bool
    base_success: bool
    alt_success: bool


# ============================================================
# Main
# ============================================================

def run():
    print("=" * 96)
    print("WCT SIDEBAND-SUBTRACTED COMB TEST")
    print("=" * 96)
    print(f"[gpu] CuPy available: {USE_CUPY}")
    print(f"[config] active intervals: {ACTIVE_INTERVALS}")
    print(f"[config] Delta ell active = {DELTA_ELL_ACTIVE:.10f}")
    print(f"[config] n0={N0}, k1={K1_FIXED}, ref k2={REFERENCE_K2}")
    print(f"[config] NULL_N={NULL_N}")
    print("=" * 96)

    for model in WCT_MODELS:
        ns, ks = ks_for_model(model)
        print(f"[model] {model['label']}")
        print(f"    n={ns}")
        print(f"    k={ks}")

    files = find_root_files()
    df = load_all_events(files)

    sig = select_region(df, B_SIGNAL, KST_SIGNAL)
    low = select_region(df, B_LOW_SB, KST_SIGNAL)
    high = select_region(df, B_HIGH_SB, KST_SIGNAL)

    print("\n[event counts]")
    print(f"  signal active: {len(sig):,}")
    print(f"  B low sideband active: {len(low):,}")
    print(f"  B high sideband active: {len(high):,}")

    if len(sig) < 500 or len(low) < 500 or len(high) < 500:
        raise RuntimeError("Too few events in signal or sidebands for subtraction test.")

    ell, Y = make_histogram(sig["q2"].values, N_BINS)
    ell_l, L = make_histogram(low["q2"].values, N_BINS)
    ell_h, H = make_histogram(high["q2"].values, N_BINS)

    # Same binning by construction.
    if len(Y) != len(L) or len(Y) != len(H):
        raise RuntimeError("Histogram length mismatch.")

    rng = np.random.default_rng(RNG_SEED)

    results = []
    null_rows = []

    for model in WCT_MODELS:
        ns, ks = ks_for_model(model)

        print("\n" + "=" * 96)
        print(f"[test] {model['label']}")
        print("=" * 96)

        real_fit = fit_base_and_alt(Y, L, H, ell, ks)
        d_real = real_fit["delta_chi2"]

        print(f"[template] alpha_low={real_fit['alpha']:.6g}, beta_high={real_fit['beta']:.6g}")
        print(f"[real] chi2_base={real_fit['base']['chi2']:.6f}")
        print(f"[real] chi2_alt ={real_fit['alt']['chi2']:.6f}")
        print(f"[real] DeltaChi2={d_real:.6f}")
        print(f"[real] alt amps={real_fit['alt']['amps']}, bound={real_fit['alt']['any_bound_active']}")

        null_deltas = []

        for j in range(NULL_N):
            Y_null = make_null_signal_counts(rng, L, H, real_fit)

            try:
                null_fit = fit_base_and_alt(Y_null, L, H, ell, ks)
                null_deltas.append(null_fit["delta_chi2"])
            except Exception:
                # Rare pathological bootstrap failure; skip.
                continue

            if (j + 1) % 500 == 0:
                print(f"  null {j+1}/{NULL_N}")

        null_deltas = np.asarray(null_deltas, dtype=float)

        if len(null_deltas) < max(100, NULL_N // 2):
            warnings.warn("Many null fits failed; p-value may be unstable.")

        p_boot = (1.0 + np.sum(null_deltas >= d_real)) / (len(null_deltas) + 1.0)

        print(f"[p] bootstrap={p_boot:.8f}")
        print(
            f"[null] mean={np.mean(null_deltas):.4f}, std={np.std(null_deltas):.4f}, "
            f"q99={np.quantile(null_deltas, 0.99):.4f}"
        )

        results.append(SubtractedResult(
            model_label=model["label"],
            model_description=model["description"],
            n_values=json.dumps([float(x) for x in ns]),
            k_values=json.dumps([float(x) for x in ks]),
            N_signal=int(Y.sum()),
            N_B_low=int(L.sum()),
            N_B_high=int(H.sum()),
            alpha_low=float(real_fit["alpha"]),
            beta_high=float(real_fit["beta"]),
            chi2_base=float(real_fit["base"]["chi2"]),
            chi2_alt=float(real_fit["alt"]["chi2"]),
            delta_chi2=float(d_real),
            p_bootstrap=float(p_boot),
            null_mean=float(np.mean(null_deltas)),
            null_std=float(np.std(null_deltas)),
            null_95=float(np.quantile(null_deltas, 0.95)),
            null_99=float(np.quantile(null_deltas, 0.99)),
            null_999=float(np.quantile(null_deltas, 0.999)),
            base_A1=float(real_fit["base"]["A1"]),
            alt_amps=json.dumps([float(x) for x in real_fit["alt"]["amps"]]),
            alt_bound_active=bool(real_fit["alt"]["any_bound_active"]),
            base_success=bool(real_fit["base"]["success"]),
            alt_success=bool(real_fit["alt"]["success"]),
        ))

        for val in null_deltas:
            null_rows.append({
                "model_label": model["label"],
                "null_delta_chi2": float(val),
                "real_delta_chi2": float(d_real),
            })

    result_df = pd.DataFrame([asdict(r) for r in results])
    null_df = pd.DataFrame(null_rows)

    summary_csv = os.path.join(OUTDIR, "sideband_subtracted_summary.csv")
    null_csv = os.path.join(OUTDIR, "sideband_subtracted_null.csv")
    summary_json = os.path.join(OUTDIR, "sideband_subtracted_summary.json")

    result_df.to_csv(summary_csv, index=False)
    null_df.to_csv(null_csv, index=False)

    payload = {
        "test": "wct_sideband_subtracted_comb_test",
        "method": "fit B sideband template, subtract template, WLS comb test on residual",
        "active_intervals_q2": ACTIVE_INTERVALS,
        "delta_ell_active": DELTA_ELL_ACTIVE,
        "n0": N0,
        "k1_fixed": K1_FIXED,
        "reference_k2": REFERENCE_K2,
        "signal_window_B": B_SIGNAL,
        "B_low_sideband": B_LOW_SB,
        "B_high_sideband": B_HIGH_SB,
        "Kst_signal_window": KST_SIGNAL,
        "null_n": NULL_N,
        "wct_models": [
            {
                **m,
                "n_values": ks_for_model(m)[0].tolist(),
                "k_values": ks_for_model(m)[1].tolist(),
            }
            for m in WCT_MODELS
        ],
        "results": result_df.to_dict(orient="records"),
        "files": {
            "summary_csv": summary_csv,
            "null_csv": null_csv,
            "summary_json": summary_json,
        },
        "interpretation": {
            "small_p": "comb remains after B-sideband template subtraction",
            "large_p": "comb is explained by B-sideband/background template plus low-k residual",
            "bound_active": "amplitude bound is stressed; repeat cap ladder",
        },
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 96)
    print("SIDEBAND-SUBTRACTED SUMMARY")
    print("=" * 96)
    print(result_df.to_string(index=False))

    print(f"\nSaved: {summary_csv}")
    print(f"Saved: {null_csv}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    run()
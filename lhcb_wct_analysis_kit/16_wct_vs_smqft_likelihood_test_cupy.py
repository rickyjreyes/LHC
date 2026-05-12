#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WCT vs SM/QFT-like Likelihood Test
----------------------------------

Goal:
    Compare a smooth SM/QFT-like yield null against WCT active-domain
    log-winding comb alternatives on LHCb open-data B0 -> K*0 mu+ mu- candidates.

Null model H0:
    lambda_i = B_i exp(C + a1 cos(k1 ell_i) + b1 sin(k1 ell_i))

    This represents:
        smooth repaired KDE baseline
        + dominant low-frequency nuisance rail k1

Alternative H1:
    H0 + WCT comb terms.

    Tested WCT alternatives:
        1. Koide sideband comb:
           Q = 2/3, n = (10,15,20)

        2. Folded trig comb:
           Q = 4/9, n = (6.667,15,13.333)

        3. Combined WCT two-branch model:
           Q = 2/3 comb + Q = 4/9 comb

Test statistic:
    DeltaD = D_null - D_alt

Parametric bootstrap p-value:
    Generate Poisson pseudo-data from fitted H0.
    Refit H0 and H1 to each pseudo-dataset.
    p = P_null(DeltaD_null >= DeltaD_real)

Also reports:
    AIC, BIC
    likelihood-ratio style p-value using chi-square approximation
    bootstrap p-value, preferred

Interpretation:
    Small p means the smooth SM/QFT-like null is insufficient relative to the
    tested WCT comb model under this pipeline.

    This is NOT a final Standard Model exclusion test.
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
    from scipy.stats import gaussian_kde, chi2
except Exception as e:
    raise RuntimeError("Missing scipy. Install with: pip install scipy") from e

try:
    from scipy.optimize import minimize
except Exception as e:
    raise RuntimeError("Missing scipy optimize. Install with: pip install scipy") from e


# ============================================================
# Configuration
# ============================================================

OUTDIR = "outputs_wct_vs_smqft"
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
KST_SIGNAL = (795.9, 995.9)

K1_FIXED = 7.61054
N0 = 15
REFERENCE_K2 = 19.5296

KDE_BANDWIDTH_SCALES = [0.50, 0.75, 1.00, 1.25, 1.50]

N_BINS = 240
NULL_N = 5000
RNG_SEED = 1551

A_MAX = 0.10

WCT_MODELS = [
    {
        "label": "WCT_Koide_sideband_Q_2over3",
        "description": "spin-1/2 true sideband Koide comb",
        "Qs": [2.0 / 3.0],
    },
    {
        "label": "WCT_folded_Q_4over9",
        "description": "spin-3/2 folded/subcentral trig comb",
        "Qs": [4.0 / 9.0],
    },
    {
        "label": "WCT_combined_Q_2over3_plus_4over9",
        "description": "combined Koide sideband + folded trig branches",
        "Qs": [2.0 / 3.0, 4.0 / 9.0],
    },
]


# ============================================================
# Winding maps
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

    # Deduplicate exact repeated k values, preserving order.
    out_ns = []
    out_ks = []
    seen = set()
    for n, k in zip(all_ns, all_ks):
        key = round(k, 12)
        if key not in seen:
            seen.add(key)
            out_ns.append(float(n))
            out_ks.append(float(k))

    return np.array(out_ns, dtype=float), np.array(out_ks, dtype=float)


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

    exact_candidates = []
    for p in particle_patterns:
        exact_candidates.extend([
            f"{p}_{comp}",
            f"{p}{comp}",
            f"{p}.{comp}",
        ])

    found = candidate_branch(keys_list, exact_candidates)
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
        "muplus0", "muplus_0",
        "MuPlus",
    ]

    minus_patterns = [
        "muminus", "mu_minus", "mum", "mu_m",
        "muminus0", "muminus_0",
        "MuMinus",
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
        print("\n[debug] Could not derive q2. Missing:")
        for m in missing:
            print("   ", m)

        print("\n[debug] Branches containing MU:")
        mu_keys = [k for k in keys if "MU" in k.upper()]
        for k in mu_keys[:250]:
            print("   ", k)

        raise RuntimeError("Could not find muon branches. Send printed MU branch block.")

    branches = [pxp, pyp, pzp, pep, pxm, pym, pzm, pem]
    arr = tree.arrays(branches, library="np")

    Ep = np.asarray(arr[pep], dtype=float)
    px_p = np.asarray(arr[pxp], dtype=float)
    py_p = np.asarray(arr[pyp], dtype=float)
    pz_p = np.asarray(arr[pzp], dtype=float)

    Em = np.asarray(arr[pem], dtype=float)
    px_m = np.asarray(arr[pxm], dtype=float)
    py_m = np.asarray(arr[pym], dtype=float)
    pz_m = np.asarray(arr[pzm], dtype=float)

    E = Ep + Em
    px = px_p + px_m
    py = py_p + py_m
    pz = pz_p + pz_m

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


def load_signal_q2(files):
    q2_values = []

    direct_q2_candidates = [
        "q2", "Q2", "q2_DTF", "Q2_DTF",
        "mumu_M2", "dimuon_M2",
        "Jpsi_M2", "J_psi_1S_M2", "J_psi_1S_MM2",
    ]

    b_mass_candidates = [
        "B0_M", "B0_MM", "B_M", "B_MM"
    ]

    kstar_mass_candidates = [
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
            kst_branch = candidate_branch(keys, kstar_mass_candidates)

            if b_branch is None:
                raise RuntimeError(f"No B mass branch found in {path}.")
            if kst_branch is None:
                raise RuntimeError(f"No K* mass branch found in {path}.")

            if q2_branch is not None:
                branches = [q2_branch, b_branch, kst_branch]
                arr = tree.arrays(branches, library="np")

                q2 = np.asarray(arr[q2_branch], dtype=float)
                finite = q2[np.isfinite(q2)]

                if len(finite) == 0:
                    print("[warn] direct q2 branch has no finite values")
                    continue

                if np.nanmedian(finite) > 1e4:
                    q2 = q2 / 1.0e6

                print(f"[q2] using direct branch: {q2_branch}")

            else:
                q2, mu_used = derive_q2_from_muons(tree)
                branches = list(mu_used.values()) + [b_branch, kst_branch]
                arr = tree.arrays(branches, library="np")
                print("[q2] derived from muon four-vectors")

            bm = np.asarray(arr[b_branch], dtype=float)
            km = np.asarray(arr[kst_branch], dtype=float)

            mask = np.isfinite(q2) & np.isfinite(bm) & np.isfinite(km)
            mask &= (q2 >= Q2_MIN) & (q2 <= Q2_MAX)
            mask &= (bm >= B_SIGNAL[0]) & (bm <= B_SIGNAL[1])
            mask &= (km >= KST_SIGNAL[0]) & (km <= KST_SIGNAL[1])
            mask &= in_active_intervals(q2)

            q2_values.append(q2[mask])

    if not q2_values:
        raise RuntimeError("No signal q2 values loaded.")

    q2_all = np.concatenate(q2_values)
    print(f"[info] selected signal active events: {len(q2_all):,}")

    if len(q2_all) < 1000:
        warnings.warn("Small sample. Check branches/cuts.")

    return q2_all


# ============================================================
# Histogram / baseline
# ============================================================

def make_histogram(q2, n_bins=N_BINS):
    ell = np.log(q2)

    ell_min = math.log(Q2_MIN)
    ell_max = math.log(Q2_MAX)

    counts, edges = np.histogram(ell, bins=n_bins, range=(ell_min, ell_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    q2_centers = np.exp(centers)

    active = in_active_intervals(q2_centers)
    counts = counts[active].astype(float)
    centers = centers[active]

    return centers, counts


def kde_baseline(ell_centers, counts, bandwidth_scale):
    repeated = np.repeat(ell_centers, np.maximum(counts.astype(int), 0))

    if len(repeated) < 100:
        raise RuntimeError("Too few repeated points for KDE baseline.")

    kde = gaussian_kde(repeated)
    kde.set_bandwidth(kde.factor * bandwidth_scale)

    dens = kde(ell_centers)
    dens = np.maximum(dens, 1e-12)

    baseline = dens / dens.sum() * counts.sum()
    baseline = np.maximum(baseline, 1e-9)

    return baseline


# ============================================================
# Poisson fitting
# ============================================================

def basis_matrix(ell, ks):
    cols = [np.ones_like(ell)]

    # Low-frequency nuisance / SM-like flexible rail
    cols.append(np.cos(K1_FIXED * ell))
    cols.append(np.sin(K1_FIXED * ell))

    for k in ks:
        cols.append(np.cos(k * ell))
        cols.append(np.sin(k * ell))

    return np.vstack(cols).T


def unpack_amplitudes(beta, n_modes):
    A1 = math.sqrt(beta[1] ** 2 + beta[2] ** 2)

    amps = []
    offset = 3
    for j in range(n_modes):
        a = beta[offset + 2 * j]
        b = beta[offset + 2 * j + 1]
        amps.append(math.sqrt(a * a + b * b))

    return A1, amps


def poisson_deviance(y, lam):
    y = np.asarray(y, dtype=float)
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-12)

    out = np.zeros_like(y, dtype=float)
    nonzero = y > 0

    out[nonzero] = y[nonzero] * np.log(y[nonzero] / lam[nonzero]) - (y[nonzero] - lam[nonzero])
    out[~nonzero] = lam[~nonzero]

    return 2.0 * float(np.sum(out))


def fit_poisson_bounded(counts, baseline, X, n_modes):
    y = np.asarray(counts, dtype=float)
    B = np.maximum(np.asarray(baseline, dtype=float), 1e-12)
    X = np.asarray(X, dtype=float)

    p = X.shape[1]
    beta0 = np.zeros(p, dtype=float)

    bounds = [(None, None)] + [(-A_MAX, A_MAX)] * (p - 1)

    def nll(beta):
        eta = X @ beta
        eta = np.clip(eta, -20.0, 20.0)
        lam = B * np.exp(eta)
        return float(np.sum(lam - y * np.log(np.maximum(lam, 1e-12))))

    result = minimize(
        nll,
        beta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-8},
    )

    beta = result.x
    eta = np.clip(X @ beta, -20.0, 20.0)
    lam = B * np.exp(eta)

    dev = poisson_deviance(y, lam)
    A1, amps = unpack_amplitudes(beta, n_modes)
    bound_active = any(abs(v) >= A_MAX - 1e-5 for v in beta[1:])

    return {
        "success": bool(result.success),
        "nll": float(result.fun),
        "dev": float(dev),
        "beta": beta,
        "lambda": lam,
        "n_params": int(p),
        "A1": float(A1),
        "amps": [float(a) for a in amps],
        "any_bound_active": bool(bound_active),
    }


def fit_model(counts, baseline, ell, ks):
    X = basis_matrix(ell, ks=ks)
    return fit_poisson_bounded(counts, baseline, X, n_modes=len(ks))


def make_null_counts_from_H0(rng, baseline, H0_fit, ell):
    X0 = basis_matrix(ell, ks=[])
    beta = H0_fit["beta"]

    eta = np.clip(X0 @ beta, -20.0, 20.0)
    lam = np.maximum(baseline * np.exp(eta), 1e-12)

    return rng.poisson(lam)


def aic(nll, k):
    return 2.0 * k + 2.0 * nll


def bic(nll, k, nobs):
    return k * math.log(max(nobs, 1)) + 2.0 * nll


# ============================================================
# Result schema
# ============================================================

@dataclass
class ModelCompareResult:
    KDE_BANDWIDTH_SCALE: float
    model_label: str
    model_description: str
    n_modes_added: int
    n_values: str
    k_values: str
    N_events: int
    N_bins_used: int
    D_H0: float
    D_H1: float
    deltaD: float
    dof_added: int
    p_chi2_approx: float
    p_bootstrap_H0: float
    AIC_H0: float
    AIC_H1: float
    delta_AIC_H1_minus_H0: float
    BIC_H0: float
    BIC_H1: float
    delta_BIC_H1_minus_H0: float
    H0_A1: float
    H1_amps: str
    H1_any_bound_active: bool
    H0_success: bool
    H1_success: bool


# ============================================================
# Main test
# ============================================================

def run():
    print("=" * 96)
    print("WCT VS SM/QFT-LIKE LIKELIHOOD TEST")
    print("=" * 96)
    print(f"[gpu] CuPy available: {USE_CUPY}")
    print(f"[config] active intervals: {ACTIVE_INTERVALS}")
    print(f"[config] Delta ell active = {DELTA_ELL_ACTIVE:.10f}")
    print(f"[config] n0 = {N0}")
    print(f"[config] K1_FIXED = {K1_FIXED}")
    print(f"[config] reference_k2 = {REFERENCE_K2}")
    print(f"[config] NULL_N = {NULL_N}")
    print("=" * 96)

    for model in WCT_MODELS:
        ns, ks = ks_for_model(model)
        print(f"[WCT model] {model['label']}")
        print(f"    n={ns}")
        print(f"    k={ks}")

    files = find_root_files()
    q2 = load_signal_q2(files)

    ell_centers, counts = make_histogram(q2, N_BINS)
    nobs_bins = int(np.sum(counts > 0))
    n_events = int(counts.sum())

    rng = np.random.default_rng(RNG_SEED)

    results = []
    null_rows = []

    for bw in KDE_BANDWIDTH_SCALES:
        print("\n" + "=" * 96)
        print(f"[bandwidth] KDE_BANDWIDTH_SCALE={bw:.2f}")
        print("=" * 96)

        baseline = kde_baseline(ell_centers, counts, bw)

        # H0 = SM/QFT-like repaired smooth model + low-k nuisance.
        H0 = fit_model(counts, baseline, ell_centers, ks=[])
        D0 = H0["dev"]

        print(f"[H0 SM/QFT-like] D={D0:.6f}, nll={H0['nll']:.6f}, A1={H0['A1']:.6f}, success={H0['success']}")

        for model in WCT_MODELS:
            ns, ks = ks_for_model(model)

            H1 = fit_model(counts, baseline, ell_centers, ks=ks)
            D1 = H1["dev"]
            dD = D0 - D1

            dof_added = 2 * len(ks)

            p_chi = float(chi2.sf(max(dD, 0.0), dof_added))

            # AIC/BIC
            AIC0 = aic(H0["nll"], H0["n_params"])
            AIC1 = aic(H1["nll"], H1["n_params"])
            BIC0 = bic(H0["nll"], H0["n_params"], nobs_bins)
            BIC1 = bic(H1["nll"], H1["n_params"], nobs_bins)

            print("\n[real fit]")
            print(f"  model={model['label']}")
            print(f"  D_H1={D1:.6f}, DeltaD={dD:.6f}, dof_added={dof_added}, p_chi2={p_chi:.6g}")
            print(f"  amps={H1['amps']}, bound={H1['any_bound_active']}")

            # Bootstrap under H0.
            null_deltas = []

            for j in range(NULL_N):
                y_null = make_null_counts_from_H0(rng, baseline, H0, ell_centers)

                H0_null = fit_model(y_null, baseline, ell_centers, ks=[])
                H1_null = fit_model(y_null, baseline, ell_centers, ks=ks)

                dD_null = H0_null["dev"] - H1_null["dev"]
                null_deltas.append(dD_null)

                if (j + 1) % 500 == 0:
                    print(f"    null {j+1}/{NULL_N}")

            null_deltas = np.asarray(null_deltas, dtype=float)
            p_boot = (1.0 + np.sum(null_deltas >= dD)) / (len(null_deltas) + 1.0)

            print(f"  p_bootstrap_H0={p_boot:.8f}")

            results.append(ModelCompareResult(
                KDE_BANDWIDTH_SCALE=float(bw),
                model_label=model["label"],
                model_description=model["description"],
                n_modes_added=int(len(ks)),
                n_values=json.dumps([float(x) for x in ns]),
                k_values=json.dumps([float(x) for x in ks]),
                N_events=n_events,
                N_bins_used=nobs_bins,
                D_H0=float(D0),
                D_H1=float(D1),
                deltaD=float(dD),
                dof_added=int(dof_added),
                p_chi2_approx=float(p_chi),
                p_bootstrap_H0=float(p_boot),
                AIC_H0=float(AIC0),
                AIC_H1=float(AIC1),
                delta_AIC_H1_minus_H0=float(AIC1 - AIC0),
                BIC_H0=float(BIC0),
                BIC_H1=float(BIC1),
                delta_BIC_H1_minus_H0=float(BIC1 - BIC0),
                H0_A1=float(H0["A1"]),
                H1_amps=json.dumps([float(x) for x in H1["amps"]]),
                H1_any_bound_active=bool(H1["any_bound_active"]),
                H0_success=bool(H0["success"]),
                H1_success=bool(H1["success"]),
            ))

            null_rows.append({
                "KDE_BANDWIDTH_SCALE": float(bw),
                "model_label": model["label"],
                "real_deltaD": float(dD),
                "null_deltaD_mean": float(np.mean(null_deltas)),
                "null_deltaD_std": float(np.std(null_deltas)),
                "null_deltaD_95": float(np.quantile(null_deltas, 0.95)),
                "null_deltaD_99": float(np.quantile(null_deltas, 0.99)),
                "null_deltaD_999": float(np.quantile(null_deltas, 0.999)),
                "p_bootstrap_H0": float(p_boot),
                "NULL_N": int(NULL_N),
            })

    result_df = pd.DataFrame([asdict(r) for r in results])
    null_df = pd.DataFrame(null_rows)

    summary_csv = os.path.join(OUTDIR, "wct_vs_smqft_summary.csv")
    null_csv = os.path.join(OUTDIR, "wct_vs_smqft_null.csv")
    summary_json = os.path.join(OUTDIR, "wct_vs_smqft_summary.json")

    result_df.to_csv(summary_csv, index=False)
    null_df.to_csv(null_csv, index=False)

    rollup = []
    if not result_df.empty:
        for label, g in result_df.groupby("model_label"):
            rollup.append({
                "model_label": label,
                "min_p_bootstrap_H0": float(g["p_bootstrap_H0"].min()),
                "median_p_bootstrap_H0": float(g["p_bootstrap_H0"].median()),
                "max_deltaD": float(g["deltaD"].max()),
                "mean_deltaD": float(g["deltaD"].mean()),
                "mean_delta_AIC_H1_minus_H0": float(g["delta_AIC_H1_minus_H0"].mean()),
                "mean_delta_BIC_H1_minus_H0": float(g["delta_BIC_H1_minus_H0"].mean()),
                "bound_active_count": int(g["H1_any_bound_active"].sum()),
                "n_rows": int(len(g)),
            })

    payload = {
        "test": "wct_vs_smqft_like_likelihood_test",
        "null_H0": "SM/QFT-like repaired KDE baseline plus low-k nuisance mode k1",
        "alternative_H1": "WCT active-domain log-winding comb terms",
        "important_limitation": (
            "This compares WCT combs against a smooth SM/QFT-like empirical null, not against "
            "a full Standard Model/QFT amplitude calculation with official covariance, efficiencies, "
            "acceptance, backgrounds, and hadronic uncertainties."
        ),
        "active_intervals_q2": ACTIVE_INTERVALS,
        "delta_ell_active": DELTA_ELL_ACTIVE,
        "n0": N0,
        "k1_fixed": K1_FIXED,
        "reference_k2": REFERENCE_K2,
        "kde_bandwidth_scales": KDE_BANDWIDTH_SCALES,
        "null_n": NULL_N,
        "wct_models": [
            {
                **m,
                "n_values": ks_for_model(m)[0].tolist(),
                "k_values": ks_for_model(m)[1].tolist(),
            }
            for m in WCT_MODELS
        ],
        "rollup": rollup,
        "files": {
            "summary_csv": summary_csv,
            "null_csv": null_csv,
            "summary_json": summary_json,
        },
        "interpretation": {
            "small_p_bootstrap_H0": (
                "The smooth SM/QFT-like empirical null rarely generates a WCT-sized comb improvement."
            ),
            "negative_delta_AIC_or_BIC": (
                "The WCT alternative is preferred even after parameter penalty."
            ),
            "bound_active": (
                "If true, amplitudes hit imposed cap; repeat with amplitude-cap ladder."
            ),
        },
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 96)
    print("WCT VS SM/QFT-LIKE SUMMARY")
    print("=" * 96)

    if rollup:
        print(pd.DataFrame(rollup).to_string(index=False))

    print(f"\nSaved: {summary_csv}")
    print(f"Saved: {null_csv}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    run()
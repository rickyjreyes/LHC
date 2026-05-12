#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WCT Koide / Trig Comb Scan
--------------------------

Purpose:
    Test whether repaired LHCb yield residuals prefer a Koide-ratio
    active-domain log-winding comb.

Comb model:
    n0 fixed, Q scanned.
    (n_minus, n0, n_plus) = n0 * (Q, 1, 2Q)

For Koide charged leptons:
    Q = 2/3
    n0 = 15
    comb = (10, 15, 20)

Model:
    lambda_i = B_i * exp(eta_i)

    eta_i = C
          + a1 cos(k1 ell_i) + b1 sin(k1 ell_i)
          + sum_j [a_j cos(k_j ell_i) + b_j sin(k_j ell_i)]

Uses:
    CuPy if available.
    NumPy fallback if CuPy unavailable.

Inputs expected:
    ROOT files in ./data/*.root or ./data/*.dvntuple.root

Outputs:
    outputs_wct_koide_comb/
        koide_comb_summary.csv
        koide_comb_summary.json
        koide_comb_null.csv
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
    from scipy.stats import gaussian_kde
except Exception as e:
    raise RuntimeError("Missing scipy. Install with: pip install scipy") from e

try:
    from scipy.optimize import minimize
except Exception as e:
    raise RuntimeError("Missing scipy optimize. Install with: pip install scipy") from e


# ============================================================
# Configuration
# ============================================================

OUTDIR = "outputs_wct_koide_comb"
os.makedirs(OUTDIR, exist_ok=True)

ROOT_PATTERNS = [
    "data/*.dvntuple.root",
    "data/*.root",
]

# q^2 domain and vetoes
Q2_MIN = 0.1
Q2_MAX = 19.0

ACTIVE_INTERVALS = [
    (0.1, 8.0),
    (11.0, 12.5),
    (14.5, 19.0),
]

# mass windows, matching prior scripts
B_MASS_MIN = 5230.0
B_MASS_MAX = 5330.0
KSTAR_MASS_MIN = 795.9
KSTAR_MASS_MAX = 995.9

# fixed low-frequency rail
K1_FIXED = 7.61054

# reference high-k value
REFERENCE_K2 = 19.5296

# central active-domain winding
N0 = 15

# tested Q table
Q_TABLE = [
    ("threshold_Q1", 1.0),
    ("near_075", 0.75),
    ("koide_lepton", 2.0 / 3.0),
    ("low_bw_empirical", 0.65),
    ("quark_like", 0.63026),
    ("spin1_Qhalf", 0.5),
    ("spin32_Q4over9", 4.0 / 9.0),
    ("neutrino_like_Qthird", 1.0 / 3.0),
]

# optional dense scan around Koide
DENSE_Q_SCAN = np.round(np.linspace(0.55, 0.75, 41), 6)

KDE_BANDWIDTH_SCALES = [0.50, 0.75, 1.00, 1.25, 1.50]

N_BINS = 240
NULL_N = 5000
RNG_SEED = 1337

# amplitude cap for every sinusoidal mode
A_MAX = 0.10

# If true, include dense scan around Koide in addition to named table.
INCLUDE_DENSE_Q_SCAN = True


# ============================================================
# Utilities
# ============================================================

def xp_to_np(x):
    if USE_CUPY:
        return cp.asnumpy(x)
    return np.asarray(x)


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


def in_active_intervals(q2):
    mask = np.zeros_like(q2, dtype=bool)
    for lo, hi in ACTIVE_INTERVALS:
        mask |= (q2 >= lo) & (q2 <= hi)
    return mask


def find_root_files():
    files = []
    for pat in ROOT_PATTERNS:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No ROOT files found in ./data/")
    return files


def candidate_branch(tree, options):
    keys = set(tree.keys())
    for name in options:
        if name in keys:
            return name
    return None


def load_q2_from_root_files(files):
    """
    Robust-ish loader.

    It tries common branch names first. If your local trees differ,
    update branch_options below.
    """

    q2_values = []

    branch_options = {
        "q2": [
            "q2",
            "Q2",
            "J_psi_1S_MM2",
            "mumu_M2",
            "dimuon_M2",
        ],
        "B_M": [
            "B0_M",
            "B_M",
            "B_MM",
            "B0_MM",
        ],
        "Kst_M": [
            "Kst_M",
            "Kstar_M",
            "Kst0_M",
            "Kstar0_M",
            "Kst_MM",
            "Kstar_MM",
        ],
    }

    for path in files:
        print(f"[load] {path}")

        with uproot.open(path) as f:
            tree = None

            # Find first TTree-like object
            for key in f.keys():
                obj = f[key]
                if hasattr(obj, "keys") and hasattr(obj, "arrays"):
                    tree = obj
                    break

            if tree is None:
                print(f"[warn] no tree found in {path}")
                continue

            q2_branch = candidate_branch(tree, branch_options["q2"])
            b_branch = candidate_branch(tree, branch_options["B_M"])
            kst_branch = candidate_branch(tree, branch_options["Kst_M"])

            if q2_branch is None:
                raise RuntimeError(
                    f"Could not find q2 branch in {path}. "
                    f"Available branches include: {list(tree.keys())[:40]}"
                )

            branches = [q2_branch]
            if b_branch:
                branches.append(b_branch)
            if kst_branch:
                branches.append(kst_branch)

            arr = tree.arrays(branches, library="np")

            q2 = np.asarray(arr[q2_branch], dtype=float)

            # If branch is dimuon mass squared in MeV^2, convert to GeV^2.
            # If already GeV^2, leave unchanged.
            finite_med = np.nanmedian(q2[np.isfinite(q2)])
            if finite_med > 1e4:
                q2 = q2 / 1.0e6

            mask = np.isfinite(q2)
            mask &= (q2 >= Q2_MIN) & (q2 <= Q2_MAX)

            if b_branch:
                bm = np.asarray(arr[b_branch], dtype=float)
                mask &= np.isfinite(bm)
                mask &= (bm >= B_MASS_MIN) & (bm <= B_MASS_MAX)

            if kst_branch:
                km = np.asarray(arr[kst_branch], dtype=float)
                mask &= np.isfinite(km)
                mask &= (km >= KSTAR_MASS_MIN) & (km <= KSTAR_MASS_MAX)

            q2_values.append(q2[mask])

    if not q2_values:
        raise RuntimeError("No q2 values loaded.")

    q2_all = np.concatenate(q2_values)
    print(f"[info] selected events before active intervals: {len(q2_all):,}")

    active = in_active_intervals(q2_all)
    q2_active = q2_all[active]

    print(f"[info] selected events after active intervals: {len(q2_active):,}")

    if len(q2_active) < 1000:
        warnings.warn("Small active sample. Check branch names and cuts.")

    return q2_active

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
    """
    Find four-vector component branches for final-state particles.

    Examples expected:
        muplus_PX, muplus_PY, muplus_PZ, muplus_PE
        muminus_PX, muminus_PY, muminus_PZ, muminus_PE

    But this also searches by substring if exact names differ.
    """
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
    """
    Derive q^2 from mu+ and mu- four-vectors.

    q^2 = (E+ + E-)^2 - |p+ + p-|^2

    LHCb momenta are usually MeV, so q^2 is converted:
        MeV^2 -> GeV^2 by /1e6
    """
    keys = list(tree.keys())

    plus_patterns = [
        "muplus", "mu_plus", "mup", "mu_p",
        "muplus0", "muplus_0",
        "muplus_TRUE", "mu_plus_TRUE",
        "MuPlus",
    ]

    minus_patterns = [
        "muminus", "mu_minus", "mum", "mu_m",
        "muminus0", "muminus_0",
        "muminus_TRUE", "mu_minus_TRUE",
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
        for k in mu_keys[:200]:
            print("   ", k)

        raise RuntimeError(
            "Could not find muon four-vector branches needed to derive q2. "
            "Send the printed MU branch block."
        )

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


def load_q2_from_root_files(files):
    """
    Loads or derives q^2.

    Priority:
        1. Use direct q2-like branch if present.
        2. Else derive q2 from mu+ and mu- four-vectors.
    """

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

            if q2_branch is not None:
                branches = [q2_branch]

                if b_branch:
                    branches.append(b_branch)
                if kst_branch:
                    branches.append(kst_branch)

                arr = tree.arrays(branches, library="np")
                q2 = np.asarray(arr[q2_branch], dtype=float)

                finite = q2[np.isfinite(q2)]
                if len(finite) == 0:
                    print("[warn] direct q2 branch has no finite values")
                    continue

                finite_med = np.nanmedian(finite)
                if finite_med > 1e4:
                    q2 = q2 / 1.0e6

                print(f"[q2] using direct branch: {q2_branch}")

            else:
                q2, mu_used = derive_q2_from_muons(tree)

                branches = list(mu_used.values())
                if b_branch:
                    branches.append(b_branch)
                if kst_branch:
                    branches.append(kst_branch)

                arr = tree.arrays(branches, library="np")

                print("[q2] derived from muon four-vectors:")
                for name, branch in mu_used.items():
                    print(f"      {name}: {branch}")

            mask = np.isfinite(q2)
            mask &= (q2 >= Q2_MIN) & (q2 <= Q2_MAX)

            if b_branch:
                bm = np.asarray(arr[b_branch], dtype=float)
                mask &= np.isfinite(bm)
                mask &= (bm >= B_MASS_MIN) & (bm <= B_MASS_MAX)
            else:
                print("[warn] no B mass branch found; skipping B mass cut")

            if kst_branch:
                km = np.asarray(arr[kst_branch], dtype=float)
                mask &= np.isfinite(km)
                mask &= (km >= KSTAR_MASS_MIN) & (km <= KSTAR_MASS_MAX)
            else:
                print("[warn] no K* mass branch found; skipping K* mass cut")

            q2_values.append(q2[mask])

    if not q2_values:
        raise RuntimeError("No q2 values loaded.")

    q2_all = np.concatenate(q2_values)

    print(f"[info] selected events before active intervals: {len(q2_all):,}")

    active = in_active_intervals(q2_all)
    q2_active = q2_all[active]

    print(f"[info] selected events after active intervals: {len(q2_active):,}")

    if len(q2_active) < 1000:
        warnings.warn("Small active sample. Check branches/cuts.")

    return q2_active

def make_histogram(q2, n_bins=N_BINS):
    ell = np.log(q2)

    # Active intervals are disconnected; histogram over full ell range,
    # then keep bins whose centers lie in active intervals.
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
    """
    KDE baseline on event-level approximation from histogram counts.

    For speed and reproducibility, expand counts only approximately.
    For large counts this is fine; for exact work use unbinned q2 events.
    """

    repeated = np.repeat(ell_centers, np.maximum(counts.astype(int), 0))
    if len(repeated) < 100:
        raise RuntimeError("Too few repeated points for KDE baseline.")

    kde = gaussian_kde(repeated)
    kde.set_bandwidth(kde.factor * bandwidth_scale)

    dens = kde(ell_centers)
    dens = np.maximum(dens, 1e-12)

    # Normalize to total counts in active histogram
    baseline = dens / dens.sum() * counts.sum()
    baseline = np.maximum(baseline, 1e-9)

    return baseline


def basis_matrix(ell, ks):
    """
    Columns:
        intercept,
        cos(k1 ell), sin(k1 ell),
        cos(k_j ell), sin(k_j ell) for all comb ks
    """
    cols = [np.ones_like(ell)]
    cols.append(np.cos(K1_FIXED * ell))
    cols.append(np.sin(K1_FIXED * ell))

    for k in ks:
        cols.append(np.cos(k * ell))
        cols.append(np.sin(k * ell))

    return np.vstack(cols).T


def unpack_amplitudes(beta, n_comb_modes):
    """
    beta:
        [C, a1, b1, a_minus, b_minus, a0, b0, a_plus, b_plus]
    """
    A1 = math.sqrt(beta[1] ** 2 + beta[2] ** 2)
    comb_As = []
    offset = 3
    for j in range(n_comb_modes):
        a = beta[offset + 2 * j]
        b = beta[offset + 2 * j + 1]
        comb_As.append(math.sqrt(a * a + b * b))
    return A1, comb_As


def poisson_deviance(y, lam):
    y = np.asarray(y, dtype=float)
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-12)

    out = np.zeros_like(y, dtype=float)
    nonzero = y > 0
    out[nonzero] = y[nonzero] * np.log(y[nonzero] / lam[nonzero]) - (y[nonzero] - lam[nonzero])
    out[~nonzero] = lam[~nonzero]
    return 2.0 * float(np.sum(out))


def fit_poisson_bounded(counts, baseline, X, n_comb_modes):
    """
    Minimize negative Poisson log likelihood.

    lambda = baseline * exp(X beta)

    Bounds:
        Intercept free.
        All sine/cos coefficients bounded by [-A_MAX, A_MAX].
    """

    y = np.asarray(counts, dtype=float)
    B = np.maximum(np.asarray(baseline, dtype=float), 1e-12)
    X = np.asarray(X, dtype=float)

    p = X.shape[1]
    beta0 = np.zeros(p, dtype=float)

    bounds = [(None, None)] + [(-A_MAX, A_MAX)] * (p - 1)

    def nll(beta):
        eta = X @ beta
        # clip exponent for stability
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

    A1, comb_As = unpack_amplitudes(beta, n_comb_modes)
    bound_active = any(abs(v) >= A_MAX - 1e-5 for v in beta[1:])

    return {
        "success": bool(result.success),
        "dev": float(dev),
        "nll": float(result.fun),
        "beta": beta,
        "A1": float(A1),
        "comb_As": [float(x) for x in comb_As],
        "any_bound_active": bool(bound_active),
    }


def fit_base_model(counts, baseline, ell):
    X = basis_matrix(ell, ks=[])
    return fit_poisson_bounded(counts, baseline, X, n_comb_modes=0)


def fit_comb_model(counts, baseline, ell, ks):
    X = basis_matrix(ell, ks=ks)
    return fit_poisson_bounded(counts, baseline, X, n_comb_modes=len(ks))


def make_null_counts_from_base(rng, baseline, base_fit, ell):
    X_base = basis_matrix(ell, ks=[])
    beta = base_fit["beta"]
    eta = np.clip(X_base @ beta, -20.0, 20.0)
    lam = np.maximum(baseline * np.exp(eta), 1e-12)
    return rng.poisson(lam)


# ============================================================
# Main scan
# ============================================================

@dataclass
class CombResult:
    KDE_BANDWIDTH_SCALE: float
    label: str
    Q: float
    n_minus: float
    n0: float
    n_plus: float
    k_minus: float
    k0: float
    k_plus: float
    deltaD: float
    p_vs_model_scanmax_null: float
    p_vs_fixed_model_null: float
    A_minus: float
    A0: float
    A_plus: float
    any_A_bound_active: bool
    success: bool


def build_q_tests():
    tests = []

    for label, q in Q_TABLE:
        tests.append((label, float(q)))

    if INCLUDE_DENSE_Q_SCAN:
        existing = {round(q, 6) for _, q in tests}
        for q in DENSE_Q_SCAN:
            q = float(q)
            if round(q, 6) not in existing:
                tests.append((f"dense_Q_{q:.6f}", q))

    # sort descending Q for readable table
    tests = sorted(tests, key=lambda x: x[1], reverse=True)
    return tests


def run():
    print("=" * 88)
    print("WCT KOIDE / TRIG COMB SCAN")
    print("=" * 88)
    print(f"[gpu] CuPy available: {USE_CUPY}")
    print(f"[config] active intervals: {ACTIVE_INTERVALS}")
    print(f"[config] Delta ell active = {DELTA_ELL_ACTIVE:.10f}")
    print(f"[config] n0 = {N0}")
    print(f"[config] Koide Q=2/3 comb n = {comb_from_Q(2/3, N0)}")
    print(f"[config] Koide comb k = {comb_k_from_Q(2/3, N0)[1]}")
    print(f"[config] K1_FIXED = {K1_FIXED}")
    print(f"[config] reference_k2 = {REFERENCE_K2}")
    print(f"[config] NULL_N = {NULL_N}")
    print("=" * 88)

    files = find_root_files()
    q2 = load_q2_from_root_files(files)
    ell_centers, counts = make_histogram(q2, N_BINS)

    q_tests = build_q_tests()
    rng = np.random.default_rng(RNG_SEED)

    all_results = []
    null_rows = []

    for bw in KDE_BANDWIDTH_SCALES:
        print("\n" + "=" * 88)
        print(f"[bandwidth] KDE_BANDWIDTH_SCALE={bw:.2f}")
        print("=" * 88)

        baseline = kde_baseline(ell_centers, counts, bw)
        base_fit = fit_base_model(counts, baseline, ell_centers)
        D_base = base_fit["dev"]

        print(f"[base] D_base={D_base:.6f}, A1={base_fit['A1']:.6f}, success={base_fit['success']}")

        observed = []

        for label, Q in q_tests:
            ns, ks = comb_k_from_Q(Q, N0)
            fit = fit_comb_model(counts, baseline, ell_centers, ks)
            deltaD = D_base - fit["dev"]

            comb_As = fit["comb_As"]
            row = {
                "KDE_BANDWIDTH_SCALE": bw,
                "label": label,
                "Q": Q,
                "n_minus": ns[0],
                "n0": ns[1],
                "n_plus": ns[2],
                "k_minus": ks[0],
                "k0": ks[1],
                "k_plus": ks[2],
                "deltaD": float(deltaD),
                "A_minus": comb_As[0],
                "A0": comb_As[1],
                "A_plus": comb_As[2],
                "any_A_bound_active": fit["any_bound_active"],
                "success": fit["success"],
            }
            observed.append(row)

        # Null distributions:
        # 1. fixed-model null per tested Q
        # 2. scanmax null across all Q models
        fixed_null = {label: [] for label, _ in q_tests}
        scanmax_null = []

        for j in range(NULL_N):
            y_null = make_null_counts_from_base(rng, baseline, base_fit, ell_centers)
            base_null_fit = fit_base_model(y_null, baseline, ell_centers)
            D0_null = base_null_fit["dev"]

            null_deltas = []

            for label, Q in q_tests:
                _, ks = comb_k_from_Q(Q, N0)
                fit_null = fit_comb_model(y_null, baseline, ell_centers, ks)
                dD_null = D0_null - fit_null["dev"]
                fixed_null[label].append(dD_null)
                null_deltas.append(dD_null)

            scanmax = max(null_deltas)
            scanmax_null.append(scanmax)

            if (j + 1) % 500 == 0:
                print(f"  null {j+1}/{NULL_N}")

        scanmax_null = np.asarray(scanmax_null, dtype=float)

        for row in observed:
            label = row["label"]
            obs = row["deltaD"]
            fixed_arr = np.asarray(fixed_null[label], dtype=float)

            p_fixed = (1.0 + np.sum(fixed_arr >= obs)) / (len(fixed_arr) + 1.0)
            p_scan = (1.0 + np.sum(scanmax_null >= obs)) / (len(scanmax_null) + 1.0)

            row["p_vs_fixed_model_null"] = float(p_fixed)
            row["p_vs_model_scanmax_null"] = float(p_scan)

            all_results.append(CombResult(**row))

            null_rows.append({
                "KDE_BANDWIDTH_SCALE": bw,
                "label": label,
                "Q": row["Q"],
                "obs_deltaD": obs,
                "p_vs_fixed_model_null": p_fixed,
                "p_vs_model_scanmax_null": p_scan,
                "null_fixed_mean": float(np.mean(fixed_arr)),
                "null_fixed_std": float(np.std(fixed_arr)),
                "null_scanmax_mean": float(np.mean(scanmax_null)),
                "null_scanmax_std": float(np.std(scanmax_null)),
            })

        # Print best row
        obs_df = pd.DataFrame(observed)
        best = obs_df.sort_values("deltaD", ascending=False).iloc[0]
        koide = obs_df[np.isclose(obs_df["Q"], 2.0 / 3.0, atol=1e-9)].iloc[0]

        print("\n[best Q]")
        print(best[[
            "label", "Q", "n_minus", "n0", "n_plus",
            "k_minus", "k0", "k_plus",
            "deltaD", "A_minus", "A0", "A_plus",
            "any_A_bound_active",
        ]].to_string())

        print("\n[Koide Q=2/3]")
        print(koide[[
            "label", "Q", "n_minus", "n0", "n_plus",
            "k_minus", "k0", "k_plus",
            "deltaD", "A_minus", "A0", "A_plus",
            "any_A_bound_active",
        ]].to_string())

    # Save outputs
    result_df = pd.DataFrame([asdict(r) for r in all_results])
    null_df = pd.DataFrame(null_rows)

    result_csv = os.path.join(OUTDIR, "koide_comb_summary.csv")
    null_csv = os.path.join(OUTDIR, "koide_comb_null.csv")
    result_json = os.path.join(OUTDIR, "koide_comb_summary.json")

    result_df.to_csv(result_csv, index=False)
    null_df.to_csv(null_csv, index=False)

    best_by_bw = (
        result_df.sort_values(["KDE_BANDWIDTH_SCALE", "deltaD"], ascending=[True, False])
        .groupby("KDE_BANDWIDTH_SCALE")
        .head(1)
        .reset_index(drop=True)
    )

    koide_rows = result_df[np.isclose(result_df["Q"], 2.0 / 3.0, atol=1e-9)].copy()

    payload = {
        "test": "wct_koide_trig_active_domain_comb_scan",
        "active_intervals_q2": ACTIVE_INTERVALS,
        "delta_ell_active": DELTA_ELL_ACTIVE,
        "n0": N0,
        "k1_fixed": K1_FIXED,
        "reference_k2": REFERENCE_K2,
        "koide_Q": 2.0 / 3.0,
        "koide_comb_n": comb_from_Q(2.0 / 3.0, N0).tolist(),
        "koide_comb_k": comb_k_from_Q(2.0 / 3.0, N0)[1].tolist(),
        "kde_bandwidth_scales": KDE_BANDWIDTH_SCALES,
        "null_n": NULL_N,
        "best_Q_by_bandwidth": best_by_bw.to_dict(orient="records"),
        "koide_Q_rows": koide_rows.to_dict(orient="records"),
        "files": {
            "summary_csv": result_csv,
            "null_csv": null_csv,
            "summary_json": result_json,
        },
        "interpretation_note": (
            "A pass means the active-domain log-winding comb geometry improves the yield model "
            "under repaired KDE baseline machinery. This does not by itself prove WCT, new physics, "
            "or absence of Standard Model / detector explanations."
        ),
    }

    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 88)
    print("KOIDE / TRIG COMB SUMMARY")
    print("=" * 88)
    print("\nBest Q by bandwidth:")
    print(best_by_bw[[
        "KDE_BANDWIDTH_SCALE", "label", "Q",
        "n_minus", "n0", "n_plus",
        "deltaD", "p_vs_model_scanmax_null",
        "p_vs_fixed_model_null",
        "A_minus", "A0", "A_plus",
        "any_A_bound_active",
    ]].to_string(index=False))

    print("\nKoide Q=2/3 rows:")
    print(koide_rows[[
        "KDE_BANDWIDTH_SCALE", "label", "Q",
        "n_minus", "n0", "n_plus",
        "deltaD", "p_vs_model_scanmax_null",
        "p_vs_fixed_model_null",
        "A_minus", "A0", "A_plus",
        "any_A_bound_active",
    ]].to_string(index=False))

    print(f"\nSaved: {result_csv}")
    print(f"Saved: {null_csv}")
    print(f"Saved: {result_json}")


if __name__ == "__main__":
    run()
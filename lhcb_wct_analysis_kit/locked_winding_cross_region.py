#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Locked Winding Cross-Region Test
--------------------------------

Purpose:
    Test whether fixed WCT/Koide/folded-trig branches explain the q^2 spectra
    across B-mass regions without floating k.

This is stricter than a free scan.

Input:
    ROOT files in ./data/*.root or ./data/*.dvntuple.root

Output:
    outputs_wct_locked_winding_cross_region/
        locked_winding_region_summary.csv
        locked_winding_model_comparison.csv
        locked_winding_summary.json

Core hypotheses:

    H0:
        KDE baseline + low-k nuisance mode k1

    H1 locked branch:
        H0 + fixed branch modes k_i

Branches tested:
    1. Koide exact sideband:
        n = (10,15,20)

    2. Folded trig branch:
        n = (6.667,15,13.333)

    3. B-low well-first triplet:
        n = (9.8141,14.5614,19.0196)

    4. Signal well-first triplet:
        n = (12.1421,17.6350,23.5082)

    5. Scaled B-low -> signal:
        n = 1.228287 * (9.8141,14.5614,19.0196)

Interpretation:
    If fixed branches win without floating k, WCT-specific odds rise.

    If only free-k scans work, but locked branches fail, the structure is less
    WCT-specific and more likely broad candidate-spectrum structure.
"""

import os
import glob
import json
import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

try:
    import uproot
except Exception as e:
    raise RuntimeError("Missing uproot. Install with: pip install uproot awkward") from e

try:
    from scipy.optimize import minimize
    from scipy.stats import gaussian_kde
except Exception as e:
    raise RuntimeError("Missing scipy. Install with: pip install scipy") from e


# ============================================================
# Config
# ============================================================

OUTDIR = "outputs_wct_locked_winding_cross_region"
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

REGIONS = [
    {
        "region": "B_low_sideband_Kst_signal",
        "B_window": B_LOW_SB,
        "Kst_window": KST_SIGNAL,
    },
    {
        "region": "signal_B_signal_Kst",
        "B_window": B_SIGNAL,
        "Kst_window": KST_SIGNAL,
    },
    {
        "region": "B_high_sideband_Kst_signal",
        "B_window": B_HIGH_SB,
        "Kst_window": KST_SIGNAL,
    },
]

K1_FIXED = 7.61054
N_BINS = 240
KDE_BANDWIDTH_SCALE = 1.00
A_MAX = 0.10

NULL_N = 3000
RNG_SEED = 23001

DELTA_ELL_ACTIVE = sum(math.log(hi / lo) for lo, hi in ACTIVE_INTERVALS)

SCALE_BLOW_TO_SIGNAL = 1.2282873602374818

BRANCHES = [
    {
        "label": "koide_exact_10_15_20",
        "class": "true_sideband_Q_2over3",
        "n": [10.0, 15.0, 20.0],
    },
    {
        "label": "folded_trig_Q_4over9",
        "class": "folded_subcentral_Q_4over9",
        "n": [6.6666666667, 15.0, 13.3333333333],
    },
    {
        "label": "well_first_Blow",
        "class": "raw_well_triplet",
        "n": [9.81412076, 14.56141638, 19.01961387],
    },
    {
        "label": "well_first_signal",
        "class": "raw_well_triplet",
        "n": [12.14212149, 17.63498598, 23.50824274],
    },
    {
        "label": "scaled_Blow_to_signal",
        "class": "scaled_raw_well_triplet",
        "n": [
            SCALE_BLOW_TO_SIGNAL * 9.81412076,
            SCALE_BLOW_TO_SIGNAL * 14.56141638,
            SCALE_BLOW_TO_SIGNAL * 19.01961387,
        ],
    },
]


# ============================================================
# Helpers
# ============================================================

def k_from_n(n):
    return 2.0 * math.pi * float(n) / DELTA_ELL_ACTIVE


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


def candidate_branch(keys, options):
    keys = set(keys)
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

    needed = [pxp, pyp, pzp, pep, pxm, pym, pzm, pem]
    if any(x is None for x in needed):
        print("\n[debug] branches containing MU:")
        for k in [x for x in keys if "MU" in x.upper()][:250]:
            print("   ", k)
        raise RuntimeError("Could not derive q2 from muon four-vectors.")

    arr = tree.arrays(needed, library="np")

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

    return q2_gev2, {
        "mu_plus_PX": pxp,
        "mu_plus_PY": pyp,
        "mu_plus_PZ": pzp,
        "mu_plus_PE": pep,
        "mu_minus_PX": pxm,
        "mu_minus_PY": pym,
        "mu_minus_PZ": pzm,
        "mu_minus_PE": pem,
    }


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
                arr = tree.arrays([q2_branch, b_branch, kst_branch], library="np")
                q2 = np.asarray(arr[q2_branch], dtype=float)
                finite = q2[np.isfinite(q2)]
                if len(finite) == 0:
                    continue
                if np.nanmedian(finite) > 1e4:
                    q2 = q2 / 1.0e6
                print(f"[q2] using branch {q2_branch}")
            else:
                q2, mu_used = derive_q2_from_muons(tree)
                arr = tree.arrays(list(mu_used.values()) + [b_branch, kst_branch], library="np")
                print("[q2] derived from muon four-vectors")

            bm = np.asarray(arr[b_branch], dtype=float)
            km = np.asarray(arr[kst_branch], dtype=float)

            mask = np.isfinite(q2) & np.isfinite(bm) & np.isfinite(km)
            mask &= (q2 >= Q2_MIN) & (q2 <= Q2_MAX)

            rows.append(pd.DataFrame({
                "q2": q2[mask],
                "B_M": bm[mask],
                "Kst_M": km[mask],
                "source_file": os.path.basename(path),
            }))

    if not rows:
        raise RuntimeError("No events loaded.")

    df = pd.concat(rows, ignore_index=True)
    print(f"[info] loaded q2-range events: {len(df):,}")
    return df


def select_region(df, B_window, Kst_window):
    blo, bhi = B_window
    klo, khi = Kst_window

    mask = (df["B_M"] >= blo) & (df["B_M"] <= bhi)
    mask &= (df["Kst_M"] >= klo) & (df["Kst_M"] <= khi)
    mask &= in_active_intervals(df["q2"].values)

    return df.loc[mask].copy()


def make_histogram(q2, n_bins=N_BINS):
    ell = np.log(q2)
    ell_min = math.log(Q2_MIN)
    ell_max = math.log(Q2_MAX)

    counts, edges = np.histogram(ell, bins=n_bins, range=(ell_min, ell_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    q2_centers = np.exp(centers)

    active = in_active_intervals(q2_centers)

    return centers[active], counts[active].astype(float)


def kde_baseline(ell_centers, counts):
    repeated = np.repeat(ell_centers, np.maximum(counts.astype(int), 0))
    if len(repeated) < 100:
        raise RuntimeError("Too few points for KDE baseline.")

    kde = gaussian_kde(repeated)
    kde.set_bandwidth(kde.factor * KDE_BANDWIDTH_SCALE)

    dens = kde(ell_centers)
    dens = np.maximum(dens, 1e-12)
    baseline = dens / dens.sum() * counts.sum()

    return np.maximum(baseline, 1e-9)


# ============================================================
# Poisson model
# ============================================================

def basis_matrix(ell, ks):
    cols = [np.ones_like(ell)]

    # nuisance low-k rail
    cols.append(np.cos(K1_FIXED * ell))
    cols.append(np.sin(K1_FIXED * ell))

    for k in ks:
        cols.append(np.cos(float(k) * ell))
        cols.append(np.sin(float(k) * ell))

    return np.vstack(cols).T


def poisson_deviance(y, lam):
    y = np.asarray(y, dtype=float)
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-12)

    out = np.zeros_like(y)
    nz = y > 0
    out[nz] = y[nz] * np.log(y[nz] / lam[nz]) - (y[nz] - lam[nz])
    out[~nz] = lam[~nz]

    return 2.0 * float(np.sum(out))


def fit_poisson_bounded(counts, baseline, ell, ks):
    y = np.asarray(counts, dtype=float)
    B = np.maximum(np.asarray(baseline, dtype=float), 1e-12)
    X = basis_matrix(ell, ks)

    p = X.shape[1]
    beta0 = np.zeros(p)

    bounds = [(None, None)] + [(-A_MAX, A_MAX)] * (p - 1)

    def nll(beta):
        eta = np.clip(X @ beta, -20.0, 20.0)
        lam = B * np.exp(eta)
        return float(np.sum(lam - y * np.log(np.maximum(lam, 1e-12))))

    res = minimize(
        nll,
        beta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-8},
    )

    beta = res.x
    eta = np.clip(X @ beta, -20.0, 20.0)
    lam = B * np.exp(eta)
    dev = poisson_deviance(y, lam)

    A1 = math.sqrt(beta[1] ** 2 + beta[2] ** 2)

    amps = []
    phases = []
    if len(ks) > 0:
        offset = 3
        for j in range(len(ks)):
            a = beta[offset + 2 * j]
            b = beta[offset + 2 * j + 1]
            amps.append(float(math.sqrt(a * a + b * b)))
            phases.append(float(math.atan2(-b, a)))

    bound_active = any(abs(v) >= A_MAX - 1e-5 for v in beta[1:])

    return {
        "success": bool(res.success),
        "dev": float(dev),
        "nll": float(res.fun),
        "beta": beta,
        "lambda": lam,
        "A1": float(A1),
        "amps": amps,
        "phases": phases,
        "bound_active": bool(bound_active),
        "n_params": int(p),
    }


def make_null_counts(rng, baseline, base_fit, ell):
    X0 = basis_matrix(ell, ks=[])
    beta = base_fit["beta"]
    eta = np.clip(X0 @ beta, -20.0, 20.0)
    lam = np.maximum(baseline * np.exp(eta), 1e-12)
    return rng.poisson(lam)


# ============================================================
# Result objects
# ============================================================

@dataclass
class BranchResult:
    region: str
    branch_label: str
    branch_class: str
    n_values: str
    k_values: str
    N_events: int
    D_base: float
    D_branch: float
    deltaD: float
    p_fixed_branch: float
    null_mean: float
    null_std: float
    null_95: float
    null_99: float
    A1: float
    amps: str
    phases: str
    bound_active: bool
    success_base: bool
    success_branch: bool


# ============================================================
# Main
# ============================================================

def run():
    print("=" * 100)
    print("LOCKED WINDING CROSS-REGION TEST")
    print("=" * 100)
    print(f"[config] Delta ell active = {DELTA_ELL_ACTIVE:.10f}")
    print(f"[config] K1_FIXED = {K1_FIXED}")
    print(f"[config] A_MAX = {A_MAX}")
    print(f"[config] NULL_N = {NULL_N}")
    print("=" * 100)

    for b in BRANCHES:
        ks = [k_from_n(n) for n in b["n"]]
        print(f"[branch] {b['label']}")
        print(f"    n = {b['n']}")
        print(f"    k = {[round(x, 6) for x in ks]}")

    files = find_root_files()
    df = load_all_events(files)

    rng = np.random.default_rng(RNG_SEED)

    rows = []
    region_summary = []

    for region_cfg in REGIONS:
        region_name = region_cfg["region"]
        sub = select_region(df, region_cfg["B_window"], region_cfg["Kst_window"])

        print("\n" + "=" * 100)
        print(f"[region] {region_name}")
        print(f"  B_window={region_cfg['B_window']}")
        print(f"  Kst_window={region_cfg['Kst_window']}")
        print(f"  N_active={len(sub):,}")
        print("=" * 100)

        if len(sub) < 500:
            print("[skip] too few events")
            continue

        ell, counts = make_histogram(sub["q2"].values)
        baseline = kde_baseline(ell, counts)

        base_fit = fit_poisson_bounded(counts, baseline, ell, ks=[])
        D_base = base_fit["dev"]

        print(f"[base] D_base={D_base:.6f}, A1={base_fit['A1']:.6f}")

        best_branch = None

        for branch in BRANCHES:
            n_vals = [float(x) for x in branch["n"]]
            k_vals = [k_from_n(x) for x in n_vals]

            fit = fit_poisson_bounded(counts, baseline, ell, ks=k_vals)
            D_branch = fit["dev"]
            deltaD = D_base - D_branch

            null_deltas = []
            for j in range(NULL_N):
                y_null = make_null_counts(rng, baseline, base_fit, ell)
                base_null = fit_poisson_bounded(y_null, baseline, ell, ks=[])
                branch_null = fit_poisson_bounded(y_null, baseline, ell, ks=k_vals)
                null_deltas.append(base_null["dev"] - branch_null["dev"])

                if (j + 1) % 1000 == 0:
                    print(f"  null {j+1}/{NULL_N} {region_name} {branch['label']}")

            null_deltas = np.asarray(null_deltas, dtype=float)
            p_fixed = (1.0 + np.sum(null_deltas >= deltaD)) / (len(null_deltas) + 1.0)

            row = BranchResult(
                region=region_name,
                branch_label=branch["label"],
                branch_class=branch["class"],
                n_values=json.dumps(n_vals),
                k_values=json.dumps([float(x) for x in k_vals]),
                N_events=int(counts.sum()),
                D_base=float(D_base),
                D_branch=float(D_branch),
                deltaD=float(deltaD),
                p_fixed_branch=float(p_fixed),
                null_mean=float(np.mean(null_deltas)),
                null_std=float(np.std(null_deltas)),
                null_95=float(np.quantile(null_deltas, 0.95)),
                null_99=float(np.quantile(null_deltas, 0.99)),
                A1=float(base_fit["A1"]),
                amps=json.dumps(fit["amps"]),
                phases=json.dumps(fit["phases"]),
                bound_active=bool(fit["bound_active"]),
                success_base=bool(base_fit["success"]),
                success_branch=bool(fit["success"]),
            )

            rows.append(row)

            print(
                f"[{branch['label']}] "
                f"DeltaD={deltaD:.4f}, p={p_fixed:.6g}, "
                f"amps={fit['amps']}, bound={fit['bound_active']}"
            )

            if best_branch is None or deltaD > best_branch["deltaD"]:
                best_branch = {
                    "branch_label": branch["label"],
                    "deltaD": float(deltaD),
                    "p_fixed_branch": float(p_fixed),
                }

        region_summary.append({
            "region": region_name,
            "N_events": int(counts.sum()),
            "D_base": float(D_base),
            "best_locked_branch": best_branch["branch_label"] if best_branch else None,
            "best_deltaD": best_branch["deltaD"] if best_branch else None,
            "best_p": best_branch["p_fixed_branch"] if best_branch else None,
        })

    result_df = pd.DataFrame([asdict(r) for r in rows])
    summary_df = pd.DataFrame(region_summary)

    result_csv = os.path.join(OUTDIR, "locked_winding_model_comparison.csv")
    summary_csv = os.path.join(OUTDIR, "locked_winding_region_summary.csv")
    summary_json = os.path.join(OUTDIR, "locked_winding_summary.json")

    result_df.to_csv(result_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    payload = {
        "test": "locked_winding_cross_region",
        "purpose": "Test fixed WCT/Koide/folded/well-first branches across B-mass regions without floating k.",
        "active_intervals": ACTIVE_INTERVALS,
        "delta_ell_active": DELTA_ELL_ACTIVE,
        "k1_fixed": K1_FIXED,
        "A_MAX": A_MAX,
        "NULL_N": NULL_N,
        "branches": [
            {
                **b,
                "k": [k_from_n(n) for n in b["n"]],
            }
            for b in BRANCHES
        ],
        "region_summary": region_summary,
        "files": {
            "model_comparison_csv": result_csv,
            "region_summary_csv": summary_csv,
            "summary_json": summary_json,
        },
        "interpretation": {
            "fixed_branch_significant": "A branch remains explanatory without floating k.",
            "same_branch_across_regions": "Evidence for a shared locked spectral skeleton.",
            "different_branches_by_region": "Evidence for region-dependent projection of candidate-spectrum geometry.",
            "sideband_stronger_than_signal": "Not signal-specific rare-decay evidence.",
        },
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 100)
    print("LOCKED WINDING CROSS-REGION SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("\nSaved:")
    print(f"  {result_csv}")
    print(f"  {summary_csv}")
    print(f"  {summary_json}")


if __name__ == "__main__":
    run()
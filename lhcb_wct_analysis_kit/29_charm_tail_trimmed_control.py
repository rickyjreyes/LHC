#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
30_charm_trimmed_control.py

Corrected charm-removal control for LHCb open-data B0 -> K*0 mu+ mu-.

This script does NOT fit or subtract charm tails. It removes/vetoes the
J/psi and psi(2S) q2 windows before any spectral test, then runs the same
log-winding / Koide diagnostics on the remaining active continuum.

It also runs a sideband-subtracted test on the same charm-trimmed support:
    R_i = N_sig,i - alpha * (N_Blow,i + N_Bhigh,i)

Run:
    python 30_charm_trimmed_control.py
    python 30_charm_trimmed_control.py --n-null 1000
"""

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass, asdict
from itertools import combinations

import numpy as np
import pandas as pd
import uproot
from scipy.signal import find_peaks

OUTDIR = "outputs_charm_trimmed_control"
os.makedirs(OUTDIR, exist_ok=True)

DEFAULT_PATTERNS = ["data/*.dvntuple.root", "data/*.root"]

Q2_MIN = 0.1
Q2_MAX = 19.0
CHARM_WINDOWS = {"Jpsi": (8.0, 11.0), "psi2S": (12.5, 14.5)}
ACTIVE_INTERVALS = [(0.1, 8.0), (11.0, 12.5), (14.5, 19.0)]

B_SIGNAL = (5230.0, 5330.0)
B_LOW_SB = (5000.0, 5180.0)
B_HIGH_SB = (5380.0, 5600.0)
KST_SIGNAL = (795.9, 995.9)
REGIONS = [
    ("signal_B_signal_Kst", B_SIGNAL, KST_SIGNAL),
    ("B_low_sideband_Kst_signal", B_LOW_SB, KST_SIGNAL),
    ("B_high_sideband_Kst_signal", B_HIGH_SB, KST_SIGNAL),
]

K1_FIXED = 7.61054
K_REF = 19.5296
KOIDE_Q = 2.0 / 3.0

N_BINS = 240
K_SCAN_MIN = 6.0
K_SCAN_MAX = 32.0
N_K_SCAN = 1301
INTEGER_N_MIN = 10
INTEGER_N_MAX = 22
MIN_PEAK_PROMINENCE = 0.5
MIN_PEAK_DISTANCE_K = 0.75
MAX_WELLS_FOR_TRIPLETS = 12
DEFAULT_N_NULL = 500
RNG_SEED = 314159
P_THRESH = 0.05


def active_delta_ell(intervals):
    return sum(math.log(hi / lo) for lo, hi in intervals)

DELTA_ELL_ACTIVE = active_delta_ell(ACTIVE_INTERVALS)


def n_from_k(k):
    return k * DELTA_ELL_ACTIVE / (2.0 * math.pi)


def k_from_n(n):
    return 2.0 * math.pi * n / DELTA_ELL_ACTIVE


def in_intervals(q2, intervals):
    q2 = np.asarray(q2, dtype=float)
    mask = np.zeros_like(q2, dtype=bool)
    for lo, hi in intervals:
        mask |= (q2 >= lo) & (q2 <= hi)
    return mask


def in_active_intervals(q2):
    return in_intervals(q2, ACTIVE_INTERVALS)


def find_files(pattern=None):
    if pattern:
        files = sorted(glob.glob(pattern))
    else:
        files = []
        for pat in DEFAULT_PATTERNS:
            files.extend(glob.glob(pat))
        files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No ROOT files found. Put ROOT files under data/ or pass --pattern.")
    return files


def candidate_branch(keys, options):
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


def find_tree(root_file):
    with uproot.open(root_file) as f:
        for key, obj in f.items(recursive=True):
            if hasattr(obj, "arrays") and "DecayTree" in key:
                return key
        for key, obj in f.items(recursive=True):
            if hasattr(obj, "arrays"):
                return key
    raise RuntimeError(f"No TTree found in {root_file}")


def find_particle_component(keys, particle_patterns, comp):
    comp_upper = comp.upper()
    exact = []
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


def derive_q2_from_muons(tree, keys):
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
        for key in keys:
            if "mu" in key.lower():
                print("  ", key)
        raise RuntimeError("Could not derive q2 from muon four-vectors.")
    arr = tree.arrays(list(branches.values()), library="np")
    E = np.asarray(arr[branches["pep"]], float) + np.asarray(arr[branches["pem"]], float)
    px = np.asarray(arr[branches["pxp"]], float) + np.asarray(arr[branches["pxm"]], float)
    py = np.asarray(arr[branches["pyp"]], float) + np.asarray(arr[branches["pym"]], float)
    pz = np.asarray(arr[branches["pzp"]], float) + np.asarray(arr[branches["pzm"]], float)
    return (E * E - px * px - py * py - pz * pz) / 1.0e6, branches


def load_all_events(files):
    q2_candidates = ["q2", "Q2", "q2_DTF", "Q2_DTF", "mumu_M2", "dimuon_M2", "Jpsi_M2", "J_psi_1S_M2", "J_psi_1S_MM2"]
    b_mass_candidates = ["B0_M", "B0_MM", "B_M", "B_MM"]
    kst_mass_candidates = ["Kst_892_0_M", "Kst_892_0_MM", "Kst_M", "Kst_MM", "Kstar_M", "Kstar_MM", "Kstar0_M", "Kstar0_MM"]
    rows, provenance = [], []
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
            mask &= (q2 >= Q2_MIN) & (q2 <= Q2_MAX)
            rows.append(pd.DataFrame({"q2": q2[mask], "B_M": bm[mask], "Kst_M": km[mask], "source_file": os.path.basename(path)}))
            provenance.append({"file": path, "tree": tree_name, "q2_source": q2_source, "B_mass_branch": b_branch, "Kst_mass_branch": kst_branch, "n_loaded_q2_range": int(np.sum(mask))})
    df = pd.concat(rows, ignore_index=True)
    print(f"[info] loaded q2-range events: {len(df):,}")
    return df, provenance


def select_region(df, b_window, kst_window, active_only=False):
    blo, bhi = b_window
    klo, khi = kst_window
    mask = (df["B_M"] >= blo) & (df["B_M"] <= bhi)
    mask &= (df["Kst_M"] >= klo) & (df["Kst_M"] <= khi)
    mask &= (df["q2"] >= Q2_MIN) & (df["q2"] <= Q2_MAX)
    if active_only:
        mask &= in_active_intervals(df["q2"].to_numpy())
    return df.loc[mask].copy()


def make_active_histogram(q2, n_bins=N_BINS):
    ell = np.log(np.asarray(q2, dtype=float))
    edges = np.linspace(math.log(Q2_MIN), math.log(Q2_MAX), n_bins + 1)
    counts, edges = np.histogram(ell, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    q2_centers = np.exp(centers)
    active = in_active_intervals(q2_centers)
    return centers[active], q2_centers[active], counts[active].astype(float)


def wls_fit(ell, y, var, ks_extra=None, include_k1=True):
    if ks_extra is None:
        ks_extra = []
    ell = np.asarray(ell, dtype=float)
    y = np.asarray(y, dtype=float)
    var = np.maximum(np.asarray(var, dtype=float), 1.0)
    cols = [np.ones_like(ell)]
    if include_k1:
        cols += [np.cos(K1_FIXED * ell), np.sin(K1_FIXED * ell)]
    for k in ks_extra:
        cols += [np.cos(k * ell), np.sin(k * ell)]
    X = np.vstack(cols).T
    w = 1.0 / np.sqrt(var)
    beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    pred = X @ beta
    chi2 = float(np.sum((y - pred) ** 2 / var))
    amps = {}
    idx = 1
    if include_k1:
        amps["A_k1"] = float(math.hypot(beta[idx], beta[idx+1])); idx += 2
    for k in ks_extra:
        amps[f"A_k_{k:.6f}"] = float(math.hypot(beta[idx], beta[idx+1]))
        amps[f"phi_k_{k:.6f}"] = float(math.atan2(-beta[idx+1], beta[idx]))
        idx += 2
    return {"chi2": chi2, "beta": beta, "pred": pred, "amps": amps, "ndof": int(len(y) - len(beta))}

@dataclass
class ScanRow:
    region: str; k: float; n_eff: float; delta_chi2: float; amp: float; phase: float
@dataclass
class WellRow:
    region: str; well_rank: int; peak_index: int; k: float; n_eff: float; delta_chi2: float; prominence: float; nearest_integer_n: float; distance_to_integer: float; distance_to_n10: float; distance_to_n15: float; distance_to_n20: float
@dataclass
class TripletRow:
    region: str; k1: float; k2: float; k3: float; n1: float; n2: float; n3: float; delta1: float; delta2: float; delta3: float; Q_low: float; Q_high: float; Q_mean: float; koide_error: float; integer_error_10_15_20: float; score: float


def scan_one_mode(region, ell, y, var, k_grid):
    base = wls_fit(ell, y, var, [], True)
    rows = []
    for k in k_grid:
        fit = wls_fit(ell, y, var, [float(k)], True)
        d = base["chi2"] - fit["chi2"]
        rows.append(ScanRow(region, float(k), float(n_from_k(k)), float(d), float(fit["amps"].get(f"A_k_{k:.6f}", np.nan)), float(fit["amps"].get(f"phi_k_{k:.6f}", np.nan))))
    return base, rows


def find_wells(region, scan_rows):
    if not scan_rows: return []
    df = pd.DataFrame([asdict(r) for r in scan_rows])
    y = df["delta_chi2"].to_numpy(); k_grid = df["k"].to_numpy()
    dk = float(np.median(np.diff(k_grid)))
    peaks, props = find_peaks(y, prominence=MIN_PEAK_PROMINENCE, distance=max(1, int(round(MIN_PEAK_DISTANCE_K / dk))))
    if len(peaks) == 0: return []
    prominences = props.get("prominences", np.zeros(len(peaks)))
    order = sorted(range(len(peaks)), key=lambda i: y[peaks[i]], reverse=True)
    wells = []
    for rank, oi in enumerate(order, 1):
        pidx = int(peaks[oi]); row = df.iloc[pidx]
        n_eff = float(row["n_eff"]); nearest = round(n_eff)
        wells.append(WellRow(region, rank, pidx, float(row["k"]), n_eff, float(row["delta_chi2"]), float(prominences[oi]), float(nearest), float(abs(n_eff-nearest)), float(abs(n_eff-10)), float(abs(n_eff-15)), float(abs(n_eff-20))))
    return wells


def triplets_from_wells(region, wells):
    if len(wells) < 3: return []
    candidates = sorted(wells[:MAX_WELLS_FOR_TRIPLETS], key=lambda w: w.n_eff)
    triplets = []
    for w1, w2, w3 in combinations(candidates, 3):
        n1, n2, n3 = w1.n_eff, w2.n_eff, w3.n_eff
        if n2 <= 0: continue
        q_low = n1 / n2; q_high = n3 / (2*n2); q_mean = 0.5*(q_low+q_high)
        koide_error = math.sqrt((q_low-KOIDE_Q)**2 + (q_high-KOIDE_Q)**2)
        integer_error = math.sqrt((n1-10)**2 + (n2-15)**2 + (n3-20)**2)
        mean_delta = (w1.delta_chi2 + w2.delta_chi2 + w3.delta_chi2) / 3.0
        score = mean_delta / (1.0 + 25.0*koide_error + 0.25*integer_error)
        triplets.append(TripletRow(region, w1.k, w2.k, w3.k, n1, n2, n3, w1.delta_chi2, w2.delta_chi2, w3.delta_chi2, q_low, q_high, q_mean, koide_error, integer_error, score))
    triplets.sort(key=lambda r: (r.koide_error, r.integer_error_10_15_20, -r.score))
    return triplets


def comb_fit_delta(ell, y, var, ns):
    ks = [k_from_n(float(n)) for n in ns]
    base = wls_fit(ell, y, var, [], True)
    fit = wls_fit(ell, y, var, ks, True)
    return base["chi2"] - fit["chi2"], ks, fit


def empirical_p(value, null_values):
    null_values = np.asarray(null_values, dtype=float)
    return float((1 + np.sum(null_values >= value)) / (len(null_values) + 1))


def run_nulls(ell, var, k_grid, n_null, rng):
    max_null=[]; kref_null=[]; n15_null=[]; comb_null=[]; folded_null=[]
    k15 = k_from_n(15)
    for j in range(n_null):
        if (j+1) % max(1, n_null//10) == 0: print(f"  [null] {j+1}/{n_null}")
        y0 = rng.normal(0, np.sqrt(np.maximum(var, 1.0)), len(var))
        _, rows0 = scan_one_mode("null", ell, y0, var, k_grid)
        max_null.append(float(max(r.delta_chi2 for r in rows0)))
        b0 = wls_fit(ell, y0, var, [], True)
        kref_null.append(float(b0["chi2"] - wls_fit(ell, y0, var, [K_REF], True)["chi2"]))
        n15_null.append(float(b0["chi2"] - wls_fit(ell, y0, var, [k15], True)["chi2"]))
        comb_null.append(float(comb_fit_delta(ell, y0, var, [10,15,20])[0]))
        folded_null.append(float(comb_fit_delta(ell, y0, var, [6.6666666667,15,13.3333333333])[0]))
    return {"scanmax":np.array(max_null), "kref":np.array(kref_null), "n15":np.array(n15_null), "comb_101520":np.array(comb_null), "folded_449":np.array(folded_null)}


def analyze_spectrum(region, ell, q2, counts, n_null, rng, prefix="charm_trimmed"):
    var = np.maximum(counts, 1.0)
    k_grid = np.linspace(K_SCAN_MIN, K_SCAN_MAX, N_K_SCAN)
    base, scan_rows = scan_one_mode(region, ell, counts, var, k_grid)
    scan_df = pd.DataFrame([asdict(r) for r in scan_rows])
    scan_csv = os.path.join(OUTDIR, f"{prefix}_scan_{region}.csv"); scan_df.to_csv(scan_csv, index=False)
    best = scan_df.sort_values("delta_chi2", ascending=False).iloc[0].to_dict()
    base_fit = wls_fit(ell, counts, var, [], True)
    delta_kref = base_fit["chi2"] - wls_fit(ell, counts, var, [K_REF], True)["chi2"]
    delta_n15 = base_fit["chi2"] - wls_fit(ell, counts, var, [k_from_n(15)], True)["chi2"]
    wells = find_wells(region, scan_rows)
    wells_df = pd.DataFrame([asdict(w) for w in wells])
    wells_csv = os.path.join(OUTDIR, f"{prefix}_wells_{region}.csv"); wells_df.to_csv(wells_csv, index=False)
    triplets = triplets_from_wells(region, wells)
    triplets_df = pd.DataFrame([asdict(t) for t in triplets])
    triplets_csv = os.path.join(OUTDIR, f"{prefix}_triplets_{region}.csv"); triplets_df.to_csv(triplets_csv, index=False)
    integer_rows=[]
    for n in range(INTEGER_N_MIN, INTEGER_N_MAX+1):
        k = k_from_n(float(n)); f = wls_fit(ell, counts, var, [k], True)
        integer_rows.append({"region":region,"n":n,"k":k,"delta_chi2":base_fit["chi2"]-f["chi2"],"amp":f["amps"].get(f"A_k_{k:.6f}", np.nan),"phase":f["amps"].get(f"phi_k_{k:.6f}", np.nan)})
    integer_csv = os.path.join(OUTDIR, f"{prefix}_integer_{region}.csv"); pd.DataFrame(integer_rows).to_csv(integer_csv, index=False)
    comb_rows=[]
    for cname, ns in [("koide_Q_2_3_true_sideband", [10,15,20]), ("folded_Q_4_9", [6.6666666667,15,13.3333333333])]:
        d, ks, _ = comb_fit_delta(ell, counts, var, ns)
        comb_rows.append({"region":region,"comb":cname,"n_values":ns,"k_values":ks,"delta_chi2":d})
    comb_df = pd.DataFrame(comb_rows)
    comb_csv = os.path.join(OUTDIR, f"{prefix}_comb_{region}.csv"); comb_df.to_csv(comb_csv, index=False)
    print(f"\n[{region}]")
    print(f"  count sum = {np.sum(counts):.1f}")
    print(f"  best scan k={best['k']:.4f}, n={best['n_eff']:.4f}, dchi2={best['delta_chi2']:.4f}")
    print(f"  k_ref dchi2={delta_kref:.4f}; n15 dchi2={delta_n15:.4f}")
    print("  comb tests:"); print(comb_df[["comb","delta_chi2"]].to_string(index=False))
    if not wells_df.empty:
        print("  top wells:"); print(wells_df.head(8)[["well_rank","k","n_eff","delta_chi2","distance_to_n15","distance_to_n20"]].to_string(index=False))
    if not triplets_df.empty:
        print("  best triplets:"); print(triplets_df.head(5)[["n1","n2","n3","Q_low","Q_high","Q_mean","koide_error","integer_error_10_15_20","score"]].to_string(index=False))
    print("  nulls:")
    nulls = run_nulls(ell, var, k_grid, n_null, rng)
    d_comb = float(comb_df.loc[comb_df["comb"]=="koide_Q_2_3_true_sideband", "delta_chi2"].iloc[0])
    d_folded = float(comb_df.loc[comb_df["comb"]=="folded_Q_4_9", "delta_chi2"].iloc[0])
    p_best = empirical_p(best["delta_chi2"], nulls["scanmax"])
    p_kref = empirical_p(delta_kref, nulls["kref"])
    p_n15 = empirical_p(delta_n15, nulls["n15"])
    p_comb = empirical_p(d_comb, nulls["comb_101520"])
    p_folded = empirical_p(d_folded, nulls["folded_449"])
    print(f"  p_best={p_best:.5f}, p_kref={p_kref:.5f}, p_n15={p_n15:.5f}, p_comb={p_comb:.5f}, p_folded={p_folded:.5f}")
    return {"region":region,"count_sum":float(np.sum(counts)),"scan":{"best_k":float(best["k"]),"best_n":float(best["n_eff"]),"best_delta_chi2":float(best["delta_chi2"]),"p_best_scanmax":p_best,"kref_delta_chi2":float(delta_kref),"p_kref_fixed":p_kref,"n15_delta_chi2":float(delta_n15),"p_n15_fixed":p_n15},"comb":{"comb_101520_delta_chi2":d_comb,"p_comb_101520":p_comb,"folded_449_delta_chi2":d_folded,"p_folded_449":p_folded},"best_triplet":asdict(triplets[0]) if triplets else None,"interpretation_flags":{"best_scan_survives_0p05":bool(p_best<=P_THRESH),"kref_survives_0p05":bool(p_kref<=P_THRESH),"n15_survives_0p05":bool(p_n15<=P_THRESH),"comb_101520_survives_0p05":bool(p_comb<=P_THRESH),"folded_449_survives_0p05":bool(p_folded<=P_THRESH)},"outputs":{"scan_csv":scan_csv,"wells_csv":wells_csv,"triplets_csv":triplets_csv,"integer_csv":integer_csv,"comb_csv":comb_csv}}


def analyze_sideband_subtracted(region_hists, n_null, rng):
    ell = region_hists["signal_B_signal_Kst"]["ell"]
    q2 = region_hists["signal_B_signal_Kst"]["q2"]
    h_sig = region_hists["signal_B_signal_Kst"]["counts"]
    h_low = region_hists["B_low_sideband_Kst_signal"]["counts"]
    h_high = region_hists["B_high_sideband_Kst_signal"]["counts"]
    h_side = h_low + h_high
    alpha = float(np.sum(h_sig) / max(np.sum(h_side), 1.0))
    residual = h_sig - alpha * h_side
    var = np.maximum(h_sig + alpha * alpha * h_side, 1.0)
    bins_csv = os.path.join(OUTDIR, "charm_trimmed_sideband_bins.csv")
    pd.DataFrame({"ell":ell,"q2_center":q2,"N_signal":h_sig,"N_Blow":h_low,"N_Bhigh":h_high,"N_side":h_side,"alpha":alpha,"R_subtracted":residual,"variance":var,"z_residual":residual/np.sqrt(var)}).to_csv(bins_csv, index=False)
    result = analyze_spectrum("sideband_subtracted_charm_trimmed", ell, q2, residual, n_null, rng, prefix="charm_trimmed_sideband")
    result["alpha"] = alpha
    result["signal_sum"] = float(np.sum(h_sig))
    result["side_sum"] = float(np.sum(h_side))
    result["residual_sum"] = float(np.sum(residual))
    result["rms_z_residual"] = float(np.sqrt(np.mean((residual / np.sqrt(var))**2)))
    result["bins_csv"] = bins_csv
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default=None, help='ROOT glob, e.g. "data/*.root"')
    parser.add_argument("--n-null", type=int, default=DEFAULT_N_NULL)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()
    print("="*100)
    print("CHARM-TRIMMED CONTROL: CUT FIRST, THEN TEST")
    print("="*100)
    print(f"[config] charm windows removed before scan = {CHARM_WINDOWS}")
    print(f"[config] active intervals = {ACTIVE_INTERVALS}")
    print(f"[config] Delta ell active = {DELTA_ELL_ACTIVE:.10f}")
    print(f"[config] k_ref = {K_REF:.6f}")
    print(f"[config] k(n=10,15,20) = {k_from_n(10):.6f}, {k_from_n(15):.6f}, {k_from_n(20):.6f}")
    print(f"[config] n_null = {args.n_null}, seed = {args.seed}")
    print("="*100)
    files = find_files(args.pattern)
    df, provenance = load_all_events(files)
    rng = np.random.default_rng(args.seed)
    region_results=[]; region_hists={}
    for region_name, b_window, kst_window in REGIONS:
        region_df = select_region(df, b_window, kst_window, active_only=True)
        print(f"\n[select] {region_name}: active charm-trimmed events = {len(region_df):,}")
        ell, q2_centers, counts = make_active_histogram(region_df["q2"].to_numpy(), N_BINS)
        region_hists[region_name] = {"ell":ell,"q2":q2_centers,"counts":counts}
        bins_csv = os.path.join(OUTDIR, f"charm_trimmed_bins_{region_name}.csv")
        pd.DataFrame({"region":region_name,"ell":ell,"q2_center":q2_centers,"counts":counts}).to_csv(bins_csv, index=False)
        result = analyze_spectrum(region_name, ell, q2_centers, counts, args.n_null, rng)
        result["event_count_active"] = int(len(region_df))
        result["bins_csv"] = bins_csv
        region_results.append(result)
    sideband_result = analyze_sideband_subtracted(region_hists, args.n_null, rng)
    summary = {"script":"30_charm_trimmed_control.py","purpose":"Remove charm windows before any spectral test; no charm fitting or subtraction.","files":files,"provenance":provenance,"config":{"Q2_MIN":Q2_MIN,"Q2_MAX":Q2_MAX,"CHARM_WINDOWS_REMOVED":CHARM_WINDOWS,"ACTIVE_INTERVALS":ACTIVE_INTERVALS,"DELTA_ELL_ACTIVE":DELTA_ELL_ACTIVE,"B_SIGNAL":B_SIGNAL,"B_LOW_SB":B_LOW_SB,"B_HIGH_SB":B_HIGH_SB,"KST_SIGNAL":KST_SIGNAL,"K1_FIXED":K1_FIXED,"K_REF":K_REF,"k_targets":{"n10":k_from_n(10.0),"n15":k_from_n(15.0),"n20":k_from_n(20.0)},"N_BINS":N_BINS,"K_SCAN_MIN":K_SCAN_MIN,"K_SCAN_MAX":K_SCAN_MAX,"N_K_SCAN":N_K_SCAN,"n_null":args.n_null,"seed":args.seed},"region_results":region_results,"sideband_subtracted_result":sideband_result,"interpretation":"Corrected charm-removal control: charm regions are cut before testing. No fitted charm yield is subtracted, so the test cannot overcount resonance leakage. Per-region results test the charm-trimmed candidate spectra; sideband-subtracted result tests survival after removing shared sideband-like structure on the same support."}
    summary_json = os.path.join(OUTDIR, "charm_trimmed_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f: json.dump(summary, f, indent=2)
    print("\n" + "="*100)
    print("CHARM-TRIMMED CONTROL SUMMARY")
    print("="*100)
    for r in region_results:
        print(f"{r['region']}: p_best={r['scan']['p_best_scanmax']:.4f}, p_kref={r['scan']['p_kref_fixed']:.4f}, p_n15={r['scan']['p_n15_fixed']:.4f}, p_comb={r['comb']['p_comb_101520']:.4f}, p_folded={r['comb']['p_folded_449']:.4f}")
    s = sideband_result
    print(f"sideband_subtracted_charm_trimmed: p_best={s['scan']['p_best_scanmax']:.4f}, p_kref={s['scan']['p_kref_fixed']:.4f}, p_n15={s['scan']['p_n15_fixed']:.4f}, p_comb={s['comb']['p_comb_101520']:.4f}, p_folded={s['comb']['p_folded_449']:.4f}")
    print("\nSaved:")
    print(" ", summary_json)

if __name__ == "__main__":
    main()

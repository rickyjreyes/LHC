#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
31_cms_locked_frequency_test.py

Fixed-template CMS -> LHCb cross-experiment test.

This stage does NOT scan frequency or phase. It maps the CMS dimuon template

    A cos[omega_m ln(m / 1 GeV) - phi_CMS]

onto the LHCb variable ell = ln(q2 / 1 GeV^2), q2 = m^2, giving

    k_LHCb = omega_m / 2
    phi_LHCb = -phi_CMS

under this repository's convention A cos(k ell + phi).

The amplitude is not frozen because CMS and this LHCb pipeline use different
residual/rate normalizations. Only the CMS-derived frequency, phase, and
positive sign are locked.

Scientific status:
    - request-48 has already been studied elsewhere in this repository, so a
      result on it is a retrospective cross-experiment fixed-template test,
      not a pristine blind replication.
    - the empirical null below keeps the observed-data KDE baseline fixed.
      It is an implemented calibration diagnostic, not an end-to-end detector
      or background-model systematic calibration.

See CMS_LOCKED_LHCB_FREEZE_2026-08-31.md.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import uproot
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, gaussian_kde, norm


# -----------------------------------------------------------------------------
# Frozen CMS -> LHCb template
# -----------------------------------------------------------------------------

CMS_OMEGA_M = 7.025825825825827
CMS_DISCOVERY_PHASE_RAD = -0.1889538223

K_CMS_Q2 = CMS_OMEGA_M / 2.0
PHI_LHCB_RAD = -CMS_DISCOVERY_PHASE_RAD

# Stage-09d-compatible LHCb selection/baseline.
Q2_MIN = 0.1
Q2_MAX = 19.0
B0_M_MIN = 5230.0
B0_M_MAX = 5330.0
KST_M_MIN = 795.9
KST_M_MAX = 995.9
JPSI_VETO = (8.0, 11.0)
PSI2S_VETO = (12.5, 14.5)
Q2_BINS = 60
KDE_BANDWIDTH_SCALE = 1.50
A_MAX = 0.10
ETA_CLIP = 0.20

TREE_NAME = "B0_KstMuMu/DecayTree"
SEED = 20260831
DEFAULT_NULL_N = 10_000

OUT_DIR = Path("outputs_cms_locked_lhcb")

REMOTE_BASE = (
    "https://opendata-lhcb-ntupling-service.app.cern.ch/api/requests/48/"
    "outputs/real-production"
)

RUN_FILES = {
    "00382466": [
        "00382466_00000001_1.dvntuple.root",
        "00382466_00000002_1.dvntuple.root",
        "00382466_00000003_1.dvntuple.root",
    ],
    "00382467": [
        "00382467_00000001_1.dvntuple.root",
        "00382467_00000002_1.dvntuple.root",
        "00382467_00000003_1.dvntuple.root",
    ],
}

MUON_BRANCHES = [
    "B0_M",
    "muplus_PX", "muplus_PY", "muplus_PZ", "muplus_PE",
    "muminus_PX", "muminus_PY", "muminus_PZ", "muminus_PE",
]


def in_veto_q2(q2: np.ndarray) -> np.ndarray:
    q2 = np.asarray(q2, dtype=float)
    return (
        ((q2 >= JPSI_VETO[0]) & (q2 <= JPSI_VETO[1]))
        | ((q2 >= PSI2S_VETO[0]) & (q2 <= PSI2S_VETO[1]))
    )


def active_delta_ell() -> float:
    intervals = [
        (Q2_MIN, JPSI_VETO[0]),
        (JPSI_VETO[1], PSI2S_VETO[0]),
        (PSI2S_VETO[1], Q2_MAX),
    ]
    return float(sum(math.log(b / a) for a, b in intervals))


DELTA_ELL_A = active_delta_ell()
CMS_ACTIVE_WINDING = K_CMS_Q2 * DELTA_ELL_A / (2.0 * math.pi)


def derive_q2(arr: dict[str, np.ndarray]) -> np.ndarray:
    e = np.asarray(arr["muplus_PE"], float) + np.asarray(arr["muminus_PE"], float)
    px = np.asarray(arr["muplus_PX"], float) + np.asarray(arr["muminus_PX"], float)
    py = np.asarray(arr["muplus_PY"], float) + np.asarray(arr["muminus_PY"], float)
    pz = np.asarray(arr["muplus_PZ"], float) + np.asarray(arr["muminus_PZ"], float)
    return (e * e - px * px - py * py - pz * pz) / 1.0e6


def _select_kst_branch(tree) -> str:
    keys = set(tree.keys())
    for name in ("Kst_892_0_M", "Kst_M"):
        if name in keys:
            return name
    raise KeyError("No supported K* mass branch found (Kst_892_0_M or Kst_M)")


def _stream_one_file(url_or_path: str, *, step_size: str) -> tuple[np.ndarray, dict]:
    pieces: list[np.ndarray] = []
    n_seen = 0
    n_selected = 0

    with uproot.open(url_or_path, timeout=300) as f:
        if TREE_NAME in f:
            tree = f[TREE_NAME]
        else:
            decay_trees = [k for k in f.keys(recursive=True) if "DecayTree" in k]
            if not decay_trees:
                raise KeyError(f"{url_or_path}: no DecayTree found")
            tree = f[decay_trees[0]]

        kst_branch = _select_kst_branch(tree)
        branches = MUON_BRANCHES + [kst_branch]
        missing = [b for b in branches if b not in tree.keys()]
        if missing:
            raise KeyError(f"{url_or_path}: missing branches {missing}")

        for arr in tree.iterate(branches, step_size=step_size, library="np"):
            n_chunk = len(arr["B0_M"])
            n_seen += n_chunk

            q2 = derive_q2(arr)
            bm = np.asarray(arr["B0_M"], float)
            km = np.asarray(arr[kst_branch], float)

            keep = np.isfinite(q2) & np.isfinite(bm) & np.isfinite(km)
            keep &= (q2 >= Q2_MIN) & (q2 <= Q2_MAX)
            keep &= (bm >= B0_M_MIN) & (bm <= B0_M_MAX)
            keep &= (km >= KST_M_MIN) & (km <= KST_M_MAX)

            chosen = np.asarray(q2[keep], dtype=np.float64)
            if chosen.size:
                pieces.append(chosen)
                n_selected += int(chosen.size)

    if not pieces:
        raise RuntimeError(f"{url_or_path}: no selected q2 events")

    return np.concatenate(pieces), {
        "source": url_or_path,
        "entries_seen": int(n_seen),
        "selected_pre_veto": int(n_selected),
    }


def load_request48(run_groups: Iterable[str], *, step_size: str) -> tuple[np.ndarray, list[dict]]:
    pieces: list[np.ndarray] = []
    provenance: list[dict] = []

    for group in run_groups:
        for filename in RUN_FILES[group]:
            url = f"{REMOTE_BASE}/{filename}"
            print(f"[remote] {group}: {filename}", flush=True)
            q2, meta = _stream_one_file(url, step_size=step_size)
            meta["run_group"] = group
            pieces.append(q2)
            provenance.append(meta)
            print(
                f"         entries={meta['entries_seen']:,} "
                f"selected_pre_veto={meta['selected_pre_veto']:,}",
                flush=True,
            )

    return np.concatenate(pieces), provenance


def load_local(pattern: str, *, step_size: str) -> tuple[np.ndarray, list[dict]]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No local ROOT files match {pattern!r}")

    pieces: list[np.ndarray] = []
    provenance: list[dict] = []
    for path in files:
        print(f"[local] {path}", flush=True)
        q2, meta = _stream_one_file(path, step_size=step_size)
        pieces.append(q2)
        provenance.append(meta)
    return np.concatenate(pieces), provenance


def make_binned_model(q2_values: np.ndarray) -> dict[str, np.ndarray]:
    q2_values = np.asarray(q2_values, dtype=float)

    counts, edges = np.histogram(q2_values, bins=Q2_BINS, range=(Q2_MIN, Q2_MAX))
    centers = 0.5 * (edges[:-1] + edges[1:])
    veto_centers = in_veto_q2(centers)

    kde_train = q2_values[
        np.isfinite(q2_values)
        & (q2_values >= Q2_MIN)
        & (q2_values <= Q2_MAX)
        & (~in_veto_q2(q2_values))
    ]
    if kde_train.size < 100:
        raise RuntimeError("Too few active events for KDE baseline")

    kde = gaussian_kde(kde_train, bw_method="scott")
    kde.set_bandwidth(kde.factor * KDE_BANDWIDTH_SCALE)
    dens = kde.evaluate(centers)
    bin_width = float(edges[1] - edges[0])
    baseline_all = np.maximum(dens * len(kde_train) * bin_width, 1e-12)

    keep = ~veto_centers
    N = counts[keep].astype(float)
    B = baseline_all[keep].astype(float)
    q2 = centers[keep].astype(float)
    ell = np.log(q2)

    # Match the stage-09d convention by normalizing the smooth baseline to the
    # active observed count before the locked modulation fit.
    B *= np.sum(N) / max(np.sum(B), 1e-12)
    B = np.maximum(B, 1e-12)

    return {
        "N": N,
        "B": B,
        "q2": q2,
        "ell": ell,
        "counts_all": counts.astype(float),
        "centers_all": centers,
        "baseline_all": baseline_all,
        "keep": keep,
    }


def poisson_nll(N: np.ndarray, lam: np.ndarray) -> float:
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-12)
    N = np.asarray(N, dtype=float)
    return float(np.sum(lam - N * np.log(lam)))


def fit_at_amplitude(
    N: np.ndarray,
    B: np.ndarray,
    wave: np.ndarray,
    amplitude: float,
) -> tuple[float, float, np.ndarray]:
    shape = B * np.exp(np.clip(amplitude * wave, -ETA_CLIP, ETA_CLIP))
    c_hat = math.log(max(np.sum(N), 1e-12) / max(np.sum(shape), 1e-12))
    c_hat = float(np.clip(c_hat, -ETA_CLIP, ETA_CLIP))
    lam = np.maximum(shape * math.exp(c_hat), 1e-12)
    return poisson_nll(N, lam), c_hat, lam


def fit_locked_template(N: np.ndarray, B: np.ndarray, ell: np.ndarray) -> dict:
    wave = np.cos(K_CMS_Q2 * ell + PHI_LHCB_RAD)

    nll0, c0, lam0 = fit_at_amplitude(N, B, wave, 0.0)

    def objective(a: float) -> float:
        return fit_at_amplitude(N, B, wave, float(a))[0]

    opt = minimize_scalar(
        objective,
        bounds=(0.0, A_MAX),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 500},
    )

    candidates = [
        (0.0, *fit_at_amplitude(N, B, wave, 0.0)),
        (float(opt.x), *fit_at_amplitude(N, B, wave, float(opt.x))),
        (A_MAX, *fit_at_amplitude(N, B, wave, A_MAX)),
    ]
    # tuple = (A, nll, C, lambda)
    best = min(candidates, key=lambda item: item[1])
    a_hat, nll1, c1, lam1 = best

    q = max(0.0, 2.0 * (nll0 - nll1))
    p_chernoff = 1.0 if q <= 0.0 else float(0.5 * chi2.sf(q, 1))
    z_chernoff = None
    if 0.0 < p_chernoff < 1.0:
        z_chernoff = float(norm.isf(p_chernoff))

    return {
        "A_hat": float(a_hat),
        "C_null": float(c0),
        "C_alt": float(c1),
        "nll_null": float(nll0),
        "nll_alt": float(nll1),
        "q_locked": float(q),
        "p_chernoff_diagnostic": float(p_chernoff),
        "z_chernoff_diagnostic": z_chernoff,
        "lambda_null": lam0,
        "lambda_alt": lam1,
        "wave": wave,
        "amplitude_bound_active": bool(abs(float(a_hat) - A_MAX) <= 1e-6),
    }


def empirical_null(
    B: np.ndarray,
    ell: np.ndarray,
    lambda_null: np.ndarray,
    *,
    n_null: int,
    rng: np.random.Generator,
) -> np.ndarray:
    q_null = np.empty(n_null, dtype=float)
    for i in range(n_null):
        pseudo = rng.poisson(lambda_null)
        q_null[i] = fit_locked_template(pseudo.astype(float), B, ell)["q_locked"]
        if (i + 1) % max(1, min(1000, n_null // 10 or 1)) == 0:
            print(f"[null] {i + 1:,}/{n_null:,}", flush=True)
    return q_null


def analyze(
    q2_values: np.ndarray,
    provenance: list[dict],
    *,
    label: str,
    n_null: int,
) -> dict:
    model = make_binned_model(q2_values)
    N = model["N"]
    B = model["B"]
    ell = model["ell"]

    fit = fit_locked_template(N, B, ell)
    rng = np.random.default_rng(SEED)

    null_q = np.empty(0, dtype=float)
    exceedances = None
    p_empirical = None
    if n_null > 0:
        null_q = empirical_null(
            B,
            ell,
            fit["lambda_null"],
            n_null=n_null,
            rng=rng,
        )
        exceedances = int(np.count_nonzero(null_q >= fit["q_locked"]))
        p_empirical = float((exceedances + 1) / (n_null + 1))

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    if n_null > 0:
        pd.DataFrame({"q_locked_null": null_q}).to_csv(
            OUT_DIR / f"{label}_cms_locked_null.csv", index=False
        )

    summary = {
        "test": "CMS_to_LHCb_fixed_frequency_fixed_phase_positive_amplitude",
        "classification": "retrospective_cross_experiment_fixed_template",
        "label": label,
        "source_provenance": provenance,
        "selection": {
            "q2_range_GeV2": [Q2_MIN, Q2_MAX],
            "B0_mass_MeV": [B0_M_MIN, B0_M_MAX],
            "Kst_mass_MeV": [KST_M_MIN, KST_M_MAX],
            "Jpsi_veto_GeV2": list(JPSI_VETO),
            "psi2S_veto_GeV2": list(PSI2S_VETO),
            "q2_bins": Q2_BINS,
            "kde_bandwidth_scale": KDE_BANDWIDTH_SCALE,
        },
        "frozen_template": {
            "cms_omega_m": CMS_OMEGA_M,
            "cms_discovery_phase_rad": CMS_DISCOVERY_PHASE_RAD,
            "lhcb_k_q2": K_CMS_Q2,
            "lhcb_phase_rad": PHI_LHCB_RAD,
            "phase_convention": "A*cos(k*ln(q2/1_GeV2)+phi)",
            "amplitude_constraint": [0.0, A_MAX],
            "active_delta_ell": DELTA_ELL_A,
            "active_winding_n": CMS_ACTIVE_WINDING,
            "frequency_scanned": False,
            "phase_scanned": False,
            "sign_scanned": False,
        },
        "counts": {
            "selected_pre_veto": int(len(q2_values)),
            "selected_active": int(np.count_nonzero(~in_veto_q2(q2_values))),
            "binned_active_count": int(np.sum(N)),
            "active_bins": int(len(N)),
        },
        "result": {
            "A_hat": fit["A_hat"],
            "C_null": fit["C_null"],
            "C_alt": fit["C_alt"],
            "q_locked": fit["q_locked"],
            "p_chernoff_diagnostic": fit["p_chernoff_diagnostic"],
            "z_chernoff_diagnostic": fit["z_chernoff_diagnostic"],
            "amplitude_bound_active": fit["amplitude_bound_active"],
            "empirical_null_trials": int(n_null),
            "empirical_exceedances": exceedances,
            "empirical_add_one_p": p_empirical,
        },
        "limitations": [
            "request-48 has prior analysis history in this repository; this is not a pristine blind replication",
            "the empirical null keeps the observed-data KDE baseline fixed rather than refitting the baseline end-to-end",
            "the LHCb B0->K*0 mu+mu- candidate spectrum is not the same inclusive dimuon observable used by CMS",
            "a positive result does not establish WCT causation or a discovery-grade physical significance",
        ],
        "seed": SEED,
    }

    out_path = OUT_DIR / f"{label}_cms_locked_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nCMS -> LHCb locked-template result")
    print("----------------------------------")
    print(f"label             : {label}")
    print(f"k(q2) frozen      : {K_CMS_Q2:.15f}")
    print(f"phase frozen      : {PHI_LHCB_RAD:.10f} rad")
    print(f"active winding n  : {CMS_ACTIVE_WINDING:.9f}")
    print(f"A_hat             : {fit['A_hat']:.8f}")
    print(f"q_locked          : {fit['q_locked']:.8f}")
    print(f"Chernoff p (diag) : {fit['p_chernoff_diagnostic']:.6g}")
    if fit["z_chernoff_diagnostic"] is not None:
        print(f"Chernoff Z (diag) : {fit['z_chernoff_diagnostic']:.6f}")
    if p_empirical is not None:
        print(f"empirical p       : {p_empirical:.6g} "
              f"({exceedances}/{n_null} exceedances; add-one)")
    print(f"summary           : {out_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen CMS-derived waveform test on LHCb q2 data."
    )
    parser.add_argument(
        "--source",
        choices=["request48", "local"],
        default="request48",
        help="Use public request-48 streaming data or local data/*.root files.",
    )
    parser.add_argument(
        "--sample",
        choices=["combined", "00382466", "00382467", "all"],
        default="combined",
        help="request-48 run-group selection. 'all' runs both groups and combined.",
    )
    parser.add_argument(
        "--data-glob",
        default="data/*.root",
        help="Local ROOT glob used with --source local.",
    )
    parser.add_argument(
        "--n-null",
        type=int,
        default=DEFAULT_NULL_N,
        help="Poisson fixed-baseline null trials; use 0 for fit only.",
    )
    parser.add_argument(
        "--step-size",
        default="100 MB",
        help="uproot streaming chunk size.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the frozen constants without reading data.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_null < 0:
        raise SystemExit("--n-null must be >= 0")

    print("Frozen CMS -> LHCb mapping")
    print(f"CMS omega_m       = {CMS_OMEGA_M:.15f}")
    print(f"CMS phase         = {CMS_DISCOVERY_PHASE_RAD:.10f} rad")
    print(f"LHCb k(q2)        = {K_CMS_Q2:.15f}")
    print(f"LHCb phase        = {PHI_LHCB_RAD:.10f} rad")
    print(f"active winding n  = {CMS_ACTIVE_WINDING:.9f}")
    print("frequency scan    = disabled")
    print("phase scan        = disabled")
    print("sign scan         = disabled (A >= 0)")

    if args.dry_run:
        return 0

    if args.source == "local":
        q2, provenance = load_local(args.data_glob, step_size=args.step_size)
        analyze(q2, provenance, label="local_combined", n_null=args.n_null)
        return 0

    if args.sample == "all":
        cache: dict[str, tuple[np.ndarray, list[dict]]] = {}
        for group in ("00382466", "00382467"):
            cache[group] = load_request48([group], step_size=args.step_size)
            analyze(
                cache[group][0],
                cache[group][1],
                label=group,
                n_null=args.n_null,
            )
        q2 = np.concatenate([cache["00382466"][0], cache["00382467"][0]])
        provenance = cache["00382466"][1] + cache["00382467"][1]
        analyze(q2, provenance, label="request48_combined", n_null=args.n_null)
        return 0

    groups = list(RUN_FILES) if args.sample == "combined" else [args.sample]
    q2, provenance = load_request48(groups, step_size=args.step_size)
    label = "request48_combined" if args.sample == "combined" else args.sample
    analyze(q2, provenance, label=label, n_null=args.n_null)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

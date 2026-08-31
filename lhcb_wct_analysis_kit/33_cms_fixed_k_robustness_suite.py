#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
33_cms_fixed_k_robustness_suite.py

Post-unblinding robustness suite for the exact CMS-mapped LHCb frequency.

The frequency is never scanned:

    k = omega_CMS / 2 = 3.512912912912913...

The phase remains free through the two fixed-frequency quadratures

    a cos(k ell) + b sin(k ell),   ell = ln(q2 / 1 GeV^2).

The prespecified robustness grid is documented in
CMS_FIXED_K_ROBUSTNESS_PLAN_2026-08-31.md.

This suite evaluates:
    * run-group split (00382466, 00382467, combined),
    * KDE bandwidth ladder,
    * q2-binning ladder,
    * charmonium-veto perturbations,
    * Chebyshev log-rate background degrees 2..6,
    * nominal KDE refit bootstrap,
    * degree-4 Chebyshev refit bootstrap.

All results are exploratory because the request-48 data were already opened by
stages 31 and 32. No output from this script is a discovery significance.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, gaussian_kde


HERE = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


base = _load_module("cms_locked_stage31", HERE / "31_cms_locked_frequency_test.py")
diag = _load_module("cms_free_phase_stage32", HERE / "32_cms_fixed_frequency_free_phase_diagnostic.py")


K_FIXED = float(base.K_CMS_Q2)
CMS_PHASE = float(base.PHI_LHCB_RAD)
SEED = 20260831

Q2_MIN = float(base.Q2_MIN)
Q2_MAX = float(base.Q2_MAX)
NOMINAL_JPSI = tuple(float(x) for x in base.JPSI_VETO)
NOMINAL_PSI2S = tuple(float(x) for x in base.PSI2S_VETO)

NOMINAL_BINS = 60
NOMINAL_KDE_SCALE = 1.50

KDE_SCALES = (0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50)
BIN_COUNTS = (48, 60, 72, 90, 120)
CHEB_DEGREES = (2, 3, 4, 5, 6)

VETO_VARIANTS = {
    "nominal": ((8.00, 11.00), (12.50, 14.50)),
    "narrow": ((8.25, 10.75), (12.75, 14.25)),
    "wide": ((7.75, 11.25), (12.25, 14.75)),
    "shift_down": ((7.75, 10.75), (12.25, 14.25)),
    "shift_up": ((8.25, 11.25), (12.75, 14.75)),
}

OUT_DIR = Path("outputs_cms_fixed_k_robustness")

# Wider descriptive-fit bounds than stage 32. The primary score statistic does
# not use these bounds. Hitting them is reported and is not silently ignored.
FIT_C_BOUND = 3.0
FIT_AB_BOUND = 1.0


def wrap_phase(x: float) -> float:
    return float((x + math.pi) % (2.0 * math.pi) - math.pi)


def in_veto(q2: np.ndarray, jpsi: tuple[float, float], psi2s: tuple[float, float]) -> np.ndarray:
    q2 = np.asarray(q2, dtype=float)
    return (
        ((q2 >= jpsi[0]) & (q2 <= jpsi[1]))
        | ((q2 >= psi2s[0]) & (q2 <= psi2s[1]))
    )


def normalize_baseline(N: np.ndarray, B: np.ndarray) -> np.ndarray:
    B = np.maximum(np.asarray(B, dtype=float), 1e-12)
    total = float(np.sum(N))
    return B * total / max(float(np.sum(B)), 1e-12)


def make_kde_model(
    q2_values: np.ndarray,
    *,
    bins: int,
    bandwidth_scale: float,
    jpsi: tuple[float, float],
    psi2s: tuple[float, float],
) -> dict[str, Any]:
    q2_values = np.asarray(q2_values, dtype=float)
    counts, edges = np.histogram(q2_values, bins=bins, range=(Q2_MIN, Q2_MAX))
    centers = 0.5 * (edges[:-1] + edges[1:])

    active_events = (
        np.isfinite(q2_values)
        & (q2_values >= Q2_MIN)
        & (q2_values <= Q2_MAX)
        & (~in_veto(q2_values, jpsi, psi2s))
    )
    train = q2_values[active_events]
    if train.size < 100:
        raise RuntimeError(f"Too few active events for KDE baseline: {train.size}")

    kde = gaussian_kde(train, bw_method="scott")
    kde.set_bandwidth(kde.factor * float(bandwidth_scale))
    dens = kde.evaluate(centers)
    bin_width = float(edges[1] - edges[0])
    baseline_all = np.maximum(dens * train.size * bin_width, 1e-12)

    keep = ~in_veto(centers, jpsi, psi2s)
    N = counts[keep].astype(float)
    B = normalize_baseline(N, baseline_all[keep])
    q2 = centers[keep].astype(float)
    ell = np.log(q2)

    return {
        "family": "kde",
        "N": N,
        "B": B,
        "q2": q2,
        "ell": ell,
        "counts_all": counts.astype(float),
        "centers_all": centers,
        "edges": edges,
        "keep": keep,
        "kde": kde,
        "kde_train": train,
        "bins": int(bins),
        "bandwidth_scale": float(bandwidth_scale),
        "jpsi": jpsi,
        "psi2s": psi2s,
    }


def _cheb_design(ell: np.ndarray, degree: int) -> np.ndarray:
    ell = np.asarray(ell, dtype=float)
    lo = math.log(Q2_MIN)
    hi = math.log(Q2_MAX)
    x = 2.0 * (ell - lo) / (hi - lo) - 1.0
    return np.polynomial.chebyshev.chebvander(x, degree)


def fit_cheb_counts(N: np.ndarray, ell: np.ndarray, degree: int) -> dict[str, Any]:
    N = np.asarray(N, dtype=float)
    ell = np.asarray(ell, dtype=float)
    X = _cheb_design(ell, degree)

    mean_count = max(float(np.mean(N)), 1e-9)
    beta0 = np.zeros(degree + 1, dtype=float)
    beta0[0] = math.log(mean_count)

    def nll(beta: np.ndarray) -> float:
        eta = np.clip(X @ beta, -30.0, 30.0)
        lam = np.maximum(np.exp(eta), 1e-12)
        return float(np.sum(lam - N * np.log(lam)))

    result = minimize(
        nll,
        x0=beta0,
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
    )
    eta = np.clip(X @ result.x, -30.0, 30.0)
    B = normalize_baseline(N, np.exp(eta))
    return {
        "B": B,
        "beta": np.asarray(result.x, dtype=float),
        "success": bool(result.success),
        "message": str(result.message),
    }


def make_cheb_model(
    q2_values: np.ndarray,
    *,
    bins: int,
    degree: int,
    jpsi: tuple[float, float],
    psi2s: tuple[float, float],
) -> dict[str, Any]:
    q2_values = np.asarray(q2_values, dtype=float)
    counts, edges = np.histogram(q2_values, bins=bins, range=(Q2_MIN, Q2_MAX))
    centers = 0.5 * (edges[:-1] + edges[1:])
    keep = ~in_veto(centers, jpsi, psi2s)
    N = counts[keep].astype(float)
    q2 = centers[keep].astype(float)
    ell = np.log(q2)
    fit = fit_cheb_counts(N, ell, degree)
    return {
        "family": "chebyshev",
        "N": N,
        "B": fit["B"],
        "q2": q2,
        "ell": ell,
        "counts_all": counts.astype(float),
        "centers_all": centers,
        "edges": edges,
        "keep": keep,
        "bins": int(bins),
        "degree": int(degree),
        "cheb_success": fit["success"],
        "cheb_message": fit["message"],
        "jpsi": jpsi,
        "psi2s": psi2s,
    }


def poisson_fit_wide(N: np.ndarray, B: np.ndarray, ell: np.ndarray) -> dict[str, Any]:
    N = np.asarray(N, dtype=float)
    B = np.maximum(np.asarray(B, dtype=float), 1e-12)
    ell = np.asarray(ell, dtype=float)

    cosx = np.cos(K_FIXED * ell)
    sinx = np.sin(K_FIXED * ell)
    c0, lam0 = diag.null_rate_from_baseline(N, B)
    nll0 = float(np.sum(lam0 - N * np.log(lam0)))

    score = diag.efficient_score_components(N, lam0, ell)
    x0 = np.array(
        [
            float(np.clip(c0, -FIT_C_BOUND, FIT_C_BOUND)),
            float(np.clip(score["a_score"], -0.25, 0.25)),
            float(np.clip(score["b_score"], -0.25, 0.25)),
        ]
    )

    def nll(theta: np.ndarray) -> float:
        C, a, b = theta
        eta = C + a * cosx + b * sinx
        lam = np.maximum(B * np.exp(np.clip(eta, -30.0, 30.0)), 1e-12)
        return float(np.sum(lam - N * np.log(lam)))

    result = minimize(
        nll,
        x0=x0,
        method="L-BFGS-B",
        bounds=[
            (-FIT_C_BOUND, FIT_C_BOUND),
            (-FIT_AB_BOUND, FIT_AB_BOUND),
            (-FIT_AB_BOUND, FIT_AB_BOUND),
        ],
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-9, "maxls": 50},
    )

    C, a, b = [float(v) for v in result.x]
    A = float(math.hypot(a, b))
    phi = float(math.atan2(-b, a)) if A > 0.0 else 0.0
    q_lrt = float(max(0.0, 2.0 * (nll0 - float(result.fun))))

    return {
        "success": bool(result.success),
        "message": str(result.message),
        "C": C,
        "a": a,
        "b": b,
        "A": A,
        "phi": phi,
        "q_lrt": q_lrt,
        "p_chi2_2_lrt_diagnostic": float(chi2.sf(q_lrt, 2)),
        "a_bound_active": bool(abs(abs(a) - FIT_AB_BOUND) <= 1e-6),
        "b_bound_active": bool(abs(abs(b) - FIT_AB_BOUND) <= 1e-6),
    }


def evaluate_model(
    model: dict[str, Any],
    *,
    n_null: int,
    seed: int,
) -> dict[str, Any]:
    N = np.asarray(model["N"], dtype=float)
    B = np.asarray(model["B"], dtype=float)
    ell = np.asarray(model["ell"], dtype=float)

    c0, lam0 = diag.null_rate_from_baseline(N, B)
    score = diag.efficient_score_components(N, lam0, ell)
    q_score = float(score["q_score"])

    null_q = np.empty(0, dtype=float)
    exceedances = None
    p_emp = None
    if n_null > 0:
        rng = np.random.default_rng(seed)
        null_q = diag.multinomial_score_null(
            total=int(np.sum(N)),
            lam0=lam0,
            X_centered=score["X_centered"],
            I_inv=score["I_inv"],
            n_null=n_null,
            rng=rng,
            batch_size=min(1000, max(1, n_null)),
        )
        exceedances = int(np.count_nonzero(null_q >= q_score))
        p_emp = float((exceedances + 1) / (n_null + 1))

    fit = poisson_fit_wide(N, B, ell)
    phi_fit = float(fit["phi"])

    return {
        "active_count": int(np.sum(N)),
        "active_bins": int(len(N)),
        "q_score": q_score,
        "p_chi2_2_diagnostic": float(chi2.sf(q_score, 2)),
        "score_A": float(score["A_score"]),
        "score_phi_rad": float(score["phi_score"]),
        "empirical_trials": int(n_null),
        "empirical_exceedances": exceedances,
        "empirical_add_one_p": p_emp,
        "fit_A": float(fit["A"]),
        "fit_phi_rad": phi_fit,
        "fit_phase_delta_cms_rad": wrap_phase(phi_fit - CMS_PHASE),
        "fit_phase_delta_cms_deg": float(math.degrees(wrap_phase(phi_fit - CMS_PHASE))),
        "fit_q_lrt": float(fit["q_lrt"]),
        "fit_success": bool(fit["success"]),
        "fit_a_bound_active": bool(fit["a_bound_active"]),
        "fit_b_bound_active": bool(fit["b_bound_active"]),
        "null_q": null_q,
    }


def load_q2_by_group(data_glob: str, *, step_size: str) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    files = sorted(glob.glob(data_glob))
    if not files:
        raise FileNotFoundError(f"No local ROOT files match {data_glob!r}")

    group_pieces: dict[str, list[np.ndarray]] = {"00382466": [], "00382467": []}
    provenance: list[dict[str, Any]] = []

    for path in files:
        name = Path(path).name
        group = None
        for candidate in group_pieces:
            if name.startswith(candidate + "_"):
                group = candidate
                break
        if group is None:
            print(f"[skip] unrecognized request-48 filename: {path}", flush=True)
            continue

        print(f"[load] {group}: {path}", flush=True)
        q2, meta = base._stream_one_file(path, step_size=step_size)
        group_pieces[group].append(np.asarray(q2, dtype=float))
        meta["run_group"] = group
        provenance.append(meta)

    out: dict[str, np.ndarray] = {}
    for group, pieces in group_pieces.items():
        if not pieces:
            raise RuntimeError(f"No files loaded for required run group {group}")
        out[group] = np.concatenate(pieces)
    out["combined"] = np.concatenate([out["00382466"], out["00382467"]])
    return out, provenance


def scenario_row(
    *,
    category: str,
    label: str,
    sample: str,
    family: str,
    parameter: str,
    bins: int,
    jpsi: tuple[float, float],
    psi2s: tuple[float, float],
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "category": category,
        "label": label,
        "sample": sample,
        "background_family": family,
        "parameter": parameter,
        "bins": int(bins),
        "jpsi_low": float(jpsi[0]),
        "jpsi_high": float(jpsi[1]),
        "psi2s_low": float(psi2s[0]),
        "psi2s_high": float(psi2s[1]),
        "k_fixed": K_FIXED,
        "frequency_scanned": False,
        "active_count": result["active_count"],
        "active_bins": result["active_bins"],
        "q_score": result["q_score"],
        "p_chi2_2_diagnostic": result["p_chi2_2_diagnostic"],
        "score_A": result["score_A"],
        "score_phi_rad": result["score_phi_rad"],
        "empirical_trials": result["empirical_trials"],
        "empirical_exceedances": result["empirical_exceedances"],
        "empirical_add_one_p": result["empirical_add_one_p"],
        "fit_A": result["fit_A"],
        "fit_phi_rad": result["fit_phi_rad"],
        "fit_phase_delta_cms_rad": result["fit_phase_delta_cms_rad"],
        "fit_phase_delta_cms_deg": result["fit_phase_delta_cms_deg"],
        "fit_q_lrt": result["fit_q_lrt"],
        "fit_success": result["fit_success"],
        "fit_a_bound_active": result["fit_a_bound_active"],
        "fit_b_bound_active": result["fit_b_bound_active"],
    }


def sample_active_from_kde(
    kde: gaussian_kde,
    *,
    total: int,
    jpsi: tuple[float, float],
    psi2s: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    have = 0
    while have < total:
        need = total - have
        draw_n = max(1000, int(math.ceil(need * 1.5)))
        vals = np.asarray(kde.resample(size=draw_n, seed=rng)).reshape(-1)
        keep = (
            np.isfinite(vals)
            & (vals >= Q2_MIN)
            & (vals <= Q2_MAX)
            & (~in_veto(vals, jpsi, psi2s))
        )
        accepted = vals[keep]
        if accepted.size:
            accepted = accepted[:need]
            pieces.append(accepted)
            have += int(accepted.size)
    return np.concatenate(pieces)


def kde_refit_bootstrap(
    observed_model: dict[str, Any],
    *,
    n_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_trials <= 0:
        return np.empty(0, dtype=float)

    observed_q = evaluate_model(observed_model, n_null=0, seed=SEED)["q_score"]
    total = int(np.sum(observed_model["N"]))
    out = np.empty(n_trials, dtype=float)

    for i in range(n_trials):
        pseudo_q2 = sample_active_from_kde(
            observed_model["kde"],
            total=total,
            jpsi=observed_model["jpsi"],
            psi2s=observed_model["psi2s"],
            rng=rng,
        )
        pseudo_model = make_kde_model(
            pseudo_q2,
            bins=int(observed_model["bins"]),
            bandwidth_scale=float(observed_model["bandwidth_scale"]),
            jpsi=observed_model["jpsi"],
            psi2s=observed_model["psi2s"],
        )
        N = pseudo_model["N"]
        B = pseudo_model["B"]
        ell = pseudo_model["ell"]
        _, lam0 = diag.null_rate_from_baseline(N, B)
        out[i] = diag.efficient_score_components(N, lam0, ell)["q_score"]
        step = max(1, n_trials // 10)
        if (i + 1) % step == 0 or i + 1 == n_trials:
            print(f"[kde-refit-null] {i + 1:,}/{n_trials:,}", flush=True)

    print(f"[kde-refit-null] observed q={observed_q:.8f}", flush=True)
    return out


def cheb_refit_bootstrap(
    observed_model: dict[str, Any],
    *,
    degree: int,
    n_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_trials <= 0:
        return np.empty(0, dtype=float)

    N_obs = np.asarray(observed_model["N"], dtype=float)
    B_obs = np.asarray(observed_model["B"], dtype=float)
    ell = np.asarray(observed_model["ell"], dtype=float)
    total = int(np.sum(N_obs))
    probs = B_obs / np.sum(B_obs)
    out = np.empty(n_trials, dtype=float)

    for i in range(n_trials):
        pseudo = rng.multinomial(total, probs).astype(float)
        fit = fit_cheb_counts(pseudo, ell, degree)
        B_refit = fit["B"]
        _, lam0 = diag.null_rate_from_baseline(pseudo, B_refit)
        out[i] = diag.efficient_score_components(pseudo, lam0, ell)["q_score"]
        step = max(1, n_trials // 10)
        if (i + 1) % step == 0 or i + 1 == n_trials:
            print(f"[cheb-refit-null] {i + 1:,}/{n_trials:,}", flush=True)

    return out


def empirical_from_null(observed_q: float, null_q: np.ndarray) -> dict[str, Any]:
    if null_q.size == 0:
        return {"trials": 0, "exceedances": None, "add_one_p": None}
    exc = int(np.count_nonzero(null_q >= observed_q))
    return {
        "trials": int(null_q.size),
        "exceedances": exc,
        "add_one_p": float((exc + 1) / (null_q.size + 1)),
    }


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    q2_by_group, provenance = load_q2_by_group(args.data_glob, step_size=args.step_size)
    combined = q2_by_group["combined"]

    rows: list[dict[str, Any]] = []
    scenario_index = 0

    def run_one(
        *,
        category: str,
        label: str,
        sample: str,
        model: dict[str, Any],
        parameter: str,
    ) -> dict[str, Any]:
        nonlocal scenario_index
        scenario_index += 1
        print("\n" + "=" * 88)
        print(f"[{scenario_index:02d}] {category}: {label}")
        print(f"sample={sample} family={model['family']} parameter={parameter}")
        print("=" * 88)
        result = evaluate_model(
            model,
            n_null=args.n_null,
            seed=SEED + scenario_index,
        )
        row = scenario_row(
            category=category,
            label=label,
            sample=sample,
            family=model["family"],
            parameter=parameter,
            bins=int(model["bins"]),
            jpsi=model["jpsi"],
            psi2s=model["psi2s"],
            result=result,
        )
        rows.append(row)
        print(
            f"q={row['q_score']:.6f} p_emp={row['empirical_add_one_p']} "
            f"phase={row['fit_phi_rad']:.6f} rad deltaCMS={row['fit_phase_delta_cms_deg']:.2f} deg"
        )
        return result

    # 1. Run-group split.
    nominal_models: dict[str, dict[str, Any]] = {}
    for sample in ("00382466", "00382467", "combined"):
        model = make_kde_model(
            q2_by_group[sample],
            bins=NOMINAL_BINS,
            bandwidth_scale=NOMINAL_KDE_SCALE,
            jpsi=NOMINAL_JPSI,
            psi2s=NOMINAL_PSI2S,
        )
        nominal_models[sample] = model
        run_one(
            category="run_group",
            label=f"{sample}_nominal",
            sample=sample,
            model=model,
            parameter="kde_scale=1.50",
        )

    # 2. KDE bandwidth ladder.
    for bw in KDE_SCALES:
        model = make_kde_model(
            combined,
            bins=NOMINAL_BINS,
            bandwidth_scale=bw,
            jpsi=NOMINAL_JPSI,
            psi2s=NOMINAL_PSI2S,
        )
        run_one(
            category="kde_bandwidth",
            label=f"combined_kde_bw_{bw:.2f}",
            sample="combined",
            model=model,
            parameter=f"kde_scale={bw:.2f}",
        )

    # 3. Binning ladder.
    for bins in BIN_COUNTS:
        model = make_kde_model(
            combined,
            bins=bins,
            bandwidth_scale=NOMINAL_KDE_SCALE,
            jpsi=NOMINAL_JPSI,
            psi2s=NOMINAL_PSI2S,
        )
        run_one(
            category="binning",
            label=f"combined_bins_{bins}",
            sample="combined",
            model=model,
            parameter=f"bins={bins}",
        )

    # 4. Veto perturbations.
    for name, (jpsi, psi2s) in VETO_VARIANTS.items():
        model = make_kde_model(
            combined,
            bins=NOMINAL_BINS,
            bandwidth_scale=NOMINAL_KDE_SCALE,
            jpsi=jpsi,
            psi2s=psi2s,
        )
        run_one(
            category="veto",
            label=f"combined_veto_{name}",
            sample="combined",
            model=model,
            parameter=name,
        )

    # 5. Alternative Chebyshev background degrees.
    cheb_models: dict[int, dict[str, Any]] = {}
    for degree in CHEB_DEGREES:
        model = make_cheb_model(
            combined,
            bins=NOMINAL_BINS,
            degree=degree,
            jpsi=NOMINAL_JPSI,
            psi2s=NOMINAL_PSI2S,
        )
        cheb_models[degree] = model
        run_one(
            category="background_family",
            label=f"combined_cheb_deg_{degree}",
            sample="combined",
            model=model,
            parameter=f"chebyshev_degree={degree}",
        )

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    table = pd.DataFrame(rows)
    table_path = OUT_DIR / "fixed_k_robustness_table.csv"
    table.to_csv(table_path, index=False)

    # Refit-baseline calibrations are deliberately separate from the grid's
    # fixed-baseline empirical p-values.
    nominal = nominal_models["combined"]
    nominal_eval = evaluate_model(nominal, n_null=0, seed=SEED)
    q_nominal = float(nominal_eval["q_score"])

    kde_refit_q = kde_refit_bootstrap(
        nominal,
        n_trials=args.n_kde_refit,
        rng=np.random.default_rng(SEED + 100_000),
    )
    kde_refit_emp = empirical_from_null(q_nominal, kde_refit_q)
    if kde_refit_q.size:
        pd.DataFrame({"q_score_null": kde_refit_q}).to_csv(
            OUT_DIR / "nominal_kde_refit_null.csv", index=False
        )

    cheb4 = cheb_models[4]
    cheb4_eval = evaluate_model(cheb4, n_null=0, seed=SEED)
    q_cheb4 = float(cheb4_eval["q_score"])
    cheb_refit_q = cheb_refit_bootstrap(
        cheb4,
        degree=4,
        n_trials=args.n_cheb_refit,
        rng=np.random.default_rng(SEED + 200_000),
    )
    cheb_refit_emp = empirical_from_null(q_cheb4, cheb_refit_q)
    if cheb_refit_q.size:
        pd.DataFrame({"q_score_null": cheb_refit_q}).to_csv(
            OUT_DIR / "cheb4_refit_null.csv", index=False
        )

    summary = {
        "test": "CMS_to_LHCb_fixed_k_post_unblinding_robustness_suite",
        "classification": "exploratory_post_unblinding_prespecified_robustness_grid",
        "frequency": {
            "cms_omega_m": float(base.CMS_OMEGA_M),
            "lhcb_k_q2": K_FIXED,
            "frequency_scanned": False,
            "phase_free": True,
            "cms_reference_phase_rad": CMS_PHASE,
        },
        "plan": "CMS_FIXED_K_ROBUSTNESS_PLAN_2026-08-31.md",
        "data_glob": args.data_glob,
        "source_provenance": provenance,
        "fixed_baseline_trials_per_scenario": int(args.n_null),
        "scenario_count": int(len(rows)),
        "scenario_table": str(table_path),
        "refit_calibrations": {
            "nominal_kde": {
                "observed_q_score": q_nominal,
                **kde_refit_emp,
                "method": "continuous_KDE_active_domain_generation_then_KDE_refit",
                "bandwidth_scale": NOMINAL_KDE_SCALE,
            },
            "chebyshev_degree_4": {
                "observed_q_score": q_cheb4,
                **cheb_refit_emp,
                "method": "multinomial_generation_then_degree4_chebyshev_refit",
            },
        },
        "guardrails": [
            "Request-48 was already unblinded before this suite; these are robustness diagnostics, not a new prospective replication.",
            "The frequency is fixed in every scenario; no frequency search is performed.",
            "Fixed-baseline and refit-baseline empirical p-values are analysis-model calibrations, not detector/systematics-calibrated physical significances.",
            "Zero empirical exceedances define only the Monte Carlo resolution floor.",
            "No CMS and LHCb p-values or Z values are combined.",
        ],
        "seed": SEED,
    }

    summary_path = OUT_DIR / "fixed_k_robustness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("FIXED-k ROBUSTNESS SUITE COMPLETE")
    print("=" * 88)
    print(f"k fixed               : {K_FIXED:.15f}")
    print(f"scenarios             : {len(rows)}")
    print(f"grid null/scenario    : {args.n_null}")
    print(
        "KDE refit null       : "
        f"{kde_refit_emp['add_one_p']} "
        f"({kde_refit_emp['exceedances']}/{kde_refit_emp['trials']} exceedances)"
    )
    print(
        "Cheb4 refit null     : "
        f"{cheb_refit_emp['add_one_p']} "
        f"({cheb_refit_emp['exceedances']}/{cheb_refit_emp['trials']} exceedances)"
    )
    print(f"table                 : {table_path}")
    print(f"summary               : {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the prespecified exploratory fixed-CMS-k robustness suite on local LHCb request-48 files."
    )
    p.add_argument(
        "--data-glob",
        default="data/*.root",
        help="Local request-48 ROOT glob. Default: data/*.root",
    )
    p.add_argument(
        "--step-size",
        default="100 MB",
        help="uproot chunk size. Default: 100 MB",
    )
    p.add_argument(
        "--n-null",
        type=int,
        default=10_000,
        help="Fixed-baseline multinomial null trials per grid point. Default: 10000",
    )
    p.add_argument(
        "--n-kde-refit",
        type=int,
        default=500,
        help="Nominal KDE generate-and-refit bootstrap trials. Default: 500",
    )
    p.add_argument(
        "--n-cheb-refit",
        type=int,
        default=1000,
        help="Degree-4 Chebyshev refit bootstrap trials. Default: 1000",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the frozen grid without reading data.",
    )
    return p.parse_args()


def print_plan() -> None:
    print("CMS-mapped fixed-k LHCb robustness suite")
    print(f"k fixed              = {K_FIXED:.15f}")
    print("frequency scan       = disabled")
    print("phase                = free")
    print("classification       = exploratory post-unblinding")
    print(f"run groups           = 00382466, 00382467, combined")
    print(f"KDE scales           = {', '.join(f'{x:.2f}' for x in KDE_SCALES)}")
    print(f"bin counts           = {', '.join(str(x) for x in BIN_COUNTS)}")
    print(f"veto variants        = {', '.join(VETO_VARIANTS)}")
    print(f"Chebyshev degrees    = {', '.join(str(x) for x in CHEB_DEGREES)}")


def main() -> int:
    args = parse_args()
    if args.n_null < 0 or args.n_kde_refit < 0 or args.n_cheb_refit < 0:
        raise SystemExit("null trial counts must be >= 0")

    print_plan()
    if args.dry_run:
        return 0

    run_suite(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

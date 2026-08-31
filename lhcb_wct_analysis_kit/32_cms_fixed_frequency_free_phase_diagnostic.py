#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
32_cms_fixed_frequency_free_phase_diagnostic.py

Exploratory post-unblinding diagnostic following the preregistered failure of
31_cms_locked_frequency_test.py on the combined request-48 LHCb sample.

Question:
    Is the CMS-derived log frequency present in this LHCb B0 -> K*0 mu+ mu-
    spectrum if phase is allowed to differ?

The frequency remains fixed at the exact CMS -> q2 mapping

    k = omega_CMS / 2 = 3.512912912912913...

but the two quadratures are free:

    eta_i = C + a cos(k ell_i) + b sin(k ell_i)

with

    A = sqrt(a^2 + b^2)
    phi = atan2(-b, a)

so the repository convention remains

    A cos(k ell + phi).

This script deliberately DOES NOT scan frequency. It is not a prospective
replication because the request-48 data have already been inspected by the
locked-template test. Its classification is therefore exploratory,
post-unblinding, fixed-frequency/free-phase.

Primary statistic:
    A two-quadrature efficient score statistic at the single fixed frequency,
    with the overall normalization projected out. The empirical null is
    multinomial conditional on the observed active total and the frozen smooth
    baseline shape. This makes the null a shape-only calibration and avoids
    spending 10,000 trials re-optimizing a nuisance normalization.

Secondary descriptive fit:
    A bounded Poisson log-link fit in (C, a, b), used to report the fitted
    amplitude and phase. Its likelihood-ratio value is descriptive; the
    empirical p-value reported by this script calibrates the score statistic.

The original stage-31 frozen result must remain unchanged and should be
reported alongside this diagnostic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "31_cms_locked_frequency_test.py"

_spec = importlib.util.spec_from_file_location("cms_locked_stage31", BASE_PATH)
base = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(base)


K_FIXED = float(base.K_CMS_Q2)
CMS_PHASE = float(base.PHI_LHCB_RAD)
ETA_CLIP = float(base.ETA_CLIP)
SEED = 20260831
DEFAULT_NULL_N = 10_000

# These are numerical safety bounds on the two free quadrature coefficients,
# not a transferred CMS amplitude prediction. The empirical null uses exactly
# the same statistic, and the observed fit reports whether either bound is hit.
AB_BOUND = 0.20

OUT_DIR = Path("outputs_cms_fixed_frequency_free_phase")


def wrap_phase(x: float) -> float:
    return float((x + math.pi) % (2.0 * math.pi) - math.pi)


def null_rate_from_baseline(N: np.ndarray, B: np.ndarray) -> tuple[float, np.ndarray]:
    """Profile the single normalization nuisance under the smooth null."""
    c0 = math.log(max(float(np.sum(N)), 1e-12) / max(float(np.sum(B)), 1e-12))
    c0 = float(np.clip(c0, -ETA_CLIP, ETA_CLIP))
    lam0 = np.maximum(B * math.exp(c0), 1e-12)
    return c0, lam0


def efficient_score_components(
    N: np.ndarray,
    lam0: np.ndarray,
    ell: np.ndarray,
) -> dict:
    """
    Two-quadrature score test with normalization projected out.

    For X=[cos(k ell), sin(k ell)], the nuisance intercept is removed by
    lambda-weighted centering. Under the fixed-baseline Poisson null,

        q = U^T I^{-1} U

    is the local two-degree-of-freedom score statistic.
    """
    N = np.asarray(N, dtype=float)
    lam0 = np.asarray(lam0, dtype=float)
    ell = np.asarray(ell, dtype=float)

    x = np.column_stack((np.cos(K_FIXED * ell), np.sin(K_FIXED * ell)))
    wsum = max(float(np.sum(lam0)), 1e-12)
    means = np.sum(lam0[:, None] * x, axis=0) / wsum
    xc = x - means

    resid = N - lam0
    U = xc.T @ resid
    I = xc.T @ (lam0[:, None] * xc)
    I_inv = np.linalg.pinv(I, rcond=1e-12)
    beta_score = I_inv @ U
    q_score = float(max(0.0, U @ I_inv @ U))

    a_score = float(beta_score[0])
    b_score = float(beta_score[1])
    A_score = float(math.hypot(a_score, b_score))
    phi_score = float(math.atan2(-b_score, a_score)) if A_score > 0 else 0.0

    return {
        "q_score": q_score,
        "U": U,
        "I": I,
        "I_inv": I_inv,
        "X_centered": xc,
        "a_score": a_score,
        "b_score": b_score,
        "A_score": A_score,
        "phi_score": phi_score,
    }


def poisson_fit_free_phase(N: np.ndarray, B: np.ndarray, ell: np.ndarray) -> dict:
    """Convex bounded Poisson fit at the single frozen frequency."""
    N = np.asarray(N, dtype=float)
    B = np.maximum(np.asarray(B, dtype=float), 1e-12)
    ell = np.asarray(ell, dtype=float)

    cosx = np.cos(K_FIXED * ell)
    sinx = np.sin(K_FIXED * ell)

    def nll(theta: np.ndarray) -> float:
        C, a, b = theta
        eta = C + a * cosx + b * sinx
        lam = np.maximum(B * np.exp(eta), 1e-12)
        return float(np.sum(lam - N * np.log(lam)))

    c0, lam0 = null_rate_from_baseline(N, B)
    nll0 = float(np.sum(lam0 - N * np.log(lam0)))

    result = minimize(
        nll,
        x0=np.array([c0, 0.0, 0.0], dtype=float),
        method="L-BFGS-B",
        bounds=[(-ETA_CLIP, ETA_CLIP), (-AB_BOUND, AB_BOUND), (-AB_BOUND, AB_BOUND)],
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9, "maxls": 50},
    )

    C, a, b = [float(v) for v in result.x]
    nll1 = float(result.fun)
    q_lrt = float(max(0.0, 2.0 * (nll0 - nll1)))
    A = float(math.hypot(a, b))
    phi = float(math.atan2(-b, a)) if A > 0 else 0.0

    return {
        "success": bool(result.success),
        "message": str(result.message),
        "C": C,
        "a": a,
        "b": b,
        "A": A,
        "phi": phi,
        "nll_null": nll0,
        "nll_alt": nll1,
        "q_lrt": q_lrt,
        "p_chi2_2_lrt_diagnostic": float(chi2.sf(q_lrt, 2)),
        "a_bound_active": bool(abs(abs(a) - AB_BOUND) <= 1e-6),
        "b_bound_active": bool(abs(abs(b) - AB_BOUND) <= 1e-6),
    }


def multinomial_score_null(
    *,
    total: int,
    lam0: np.ndarray,
    X_centered: np.ndarray,
    I_inv: np.ndarray,
    n_null: int,
    rng: np.random.Generator,
    batch_size: int = 1000,
) -> np.ndarray:
    """Fast shape-only empirical null conditional on the observed total."""
    probs = np.asarray(lam0, dtype=float)
    probs = probs / np.sum(probs)
    expected = float(total) * probs

    # Recompute the conditional information at the observed fixed total.
    I_cond = X_centered.T @ (expected[:, None] * X_centered)
    I_cond_inv = np.linalg.pinv(I_cond, rcond=1e-12)

    out = np.empty(n_null, dtype=float)
    done = 0
    while done < n_null:
        m = min(batch_size, n_null - done)
        pseudo = rng.multinomial(total, probs, size=m).astype(float)
        resid = pseudo - expected[None, :]
        U = resid @ X_centered
        q = np.einsum("bi,ij,bj->b", U, I_cond_inv, U)
        out[done : done + m] = np.maximum(q, 0.0)
        done += m
        if done % max(1, min(1000, n_null // 10 or 1)) == 0 or done == n_null:
            print(f"[null] {done:,}/{n_null:,}", flush=True)
    return out


def analyze_local(data_glob: str, *, n_null: int, step_size: str, label: str) -> dict:
    q2_values, provenance = base.load_local(data_glob, step_size=step_size)
    model = base.make_binned_model(q2_values)
    N = model["N"].astype(float)
    B = model["B"].astype(float)
    ell = model["ell"].astype(float)

    c0, lam0 = null_rate_from_baseline(N, B)
    score = efficient_score_components(N, lam0, ell)
    fit = poisson_fit_free_phase(N, B, ell)

    q_score = float(score["q_score"])
    p_chi2 = float(chi2.sf(q_score, 2))

    null_q = np.empty(0, dtype=float)
    exceedances = None
    p_emp = None
    if n_null > 0:
        rng = np.random.default_rng(SEED)
        null_q = multinomial_score_null(
            total=int(np.sum(N)),
            lam0=lam0,
            X_centered=score["X_centered"],
            I_inv=score["I_inv"],
            n_null=n_null,
            rng=rng,
        )
        exceedances = int(np.count_nonzero(null_q >= q_score))
        p_emp = float((exceedances + 1) / (n_null + 1))

    phi_fit = float(fit["phi"])
    phase_delta_cms = wrap_phase(phi_fit - CMS_PHASE)
    phase_delta_opposite = wrap_phase(phi_fit - wrap_phase(CMS_PHASE + math.pi))

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    if n_null > 0:
        pd.DataFrame({"q_score_null": null_q}).to_csv(
            OUT_DIR / f"{label}_fixed_k_free_phase_null.csv", index=False
        )

    summary = {
        "test": "CMS_to_LHCb_fixed_frequency_free_phase_two_quadrature",
        "classification": "exploratory_post_unblinding_fixed_frequency_free_phase",
        "label": label,
        "source_provenance": provenance,
        "frozen_frequency": {
            "cms_omega_m": float(base.CMS_OMEGA_M),
            "lhcb_k_q2": K_FIXED,
            "frequency_scanned": False,
            "phase_free": True,
            "cms_reference_phase_lhcb_convention_rad": CMS_PHASE,
        },
        "selection": {
            "q2_range_GeV2": [float(base.Q2_MIN), float(base.Q2_MAX)],
            "B0_mass_MeV": [float(base.B0_M_MIN), float(base.B0_M_MAX)],
            "Kst_mass_MeV": [float(base.KST_M_MIN), float(base.KST_M_MAX)],
            "Jpsi_veto_GeV2": list(base.JPSI_VETO),
            "psi2S_veto_GeV2": list(base.PSI2S_VETO),
            "q2_bins": int(base.Q2_BINS),
            "kde_bandwidth_scale": float(base.KDE_BANDWIDTH_SCALE),
        },
        "counts": {
            "selected_pre_veto": int(len(q2_values)),
            "selected_active": int(np.count_nonzero(~base.in_veto_q2(q2_values))),
            "binned_active_count": int(np.sum(N)),
            "active_bins": int(len(N)),
        },
        "primary_score_result": {
            "q_score": q_score,
            "p_chi2_2_diagnostic": p_chi2,
            "a_score": float(score["a_score"]),
            "b_score": float(score["b_score"]),
            "A_score": float(score["A_score"]),
            "phi_score_rad": float(score["phi_score"]),
            "empirical_null": "multinomial_conditional_on_observed_active_total_fixed_baseline_shape",
            "empirical_trials": int(n_null),
            "empirical_exceedances": exceedances,
            "empirical_add_one_p": p_emp,
        },
        "secondary_poisson_fit": {
            **fit,
            "phase_delta_to_cms_rad": phase_delta_cms,
            "phase_delta_to_cms_deg": float(math.degrees(phase_delta_cms)),
            "phase_delta_to_opposite_cms_rad": phase_delta_opposite,
            "phase_delta_to_opposite_cms_deg": float(math.degrees(phase_delta_opposite)),
            "quadrature_coefficient_bound": AB_BOUND,
        },
        "interpretation_guardrails": [
            "This is exploratory because request-48 was already unblinded by the fixed-phase stage-31 test.",
            "A small empirical p-value here would show structure at the CMS-derived frequency in this LHCb channel, not a prospective replication.",
            "Phase compatibility with CMS is descriptive after unblinding and must not be promoted to a preregistered result.",
            "A null result would weigh against recurrence of the CMS frequency itself in this exclusive B0->K*0 mu+mu- spectrum.",
            "The empirical null conditions on the observed active total and keeps the smooth KDE baseline shape fixed; it is not an end-to-end detector/systematics calibration.",
        ],
        "seed": SEED,
    }

    out_path = OUT_DIR / f"{label}_fixed_k_free_phase_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nCMS-frequency / free-phase diagnostic")
    print("-------------------------------------")
    print(f"label               : {label}")
    print(f"k(q2) fixed         : {K_FIXED:.15f}")
    print("frequency scan      : disabled")
    print("phase               : free")
    print(f"score q (2 quad)    : {q_score:.8f}")
    print(f"chi2_2 p (diag)     : {p_chi2:.6g}")
    if p_emp is not None:
        print(f"empirical p         : {p_emp:.6g} ({exceedances}/{n_null} exceedances; add-one)")
    print(f"Poisson A_hat       : {fit['A']:.8f}")
    print(f"Poisson phase_hat   : {phi_fit:.10f} rad")
    print(f"phase - CMS         : {phase_delta_cms:.10f} rad ({math.degrees(phase_delta_cms):.3f} deg)")
    print(f"phase - opposite    : {phase_delta_opposite:.10f} rad ({math.degrees(phase_delta_opposite):.3f} deg)")
    print(f"Poisson q_LRT       : {fit['q_lrt']:.8f}")
    print(f"summary             : {out_path}")

    if fit["a_bound_active"] or fit["b_bound_active"]:
        print("WARNING             : a/b numerical safety bound active; inspect fit before interpretation")

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exploratory fixed-CMS-frequency, free-phase diagnostic on local LHCb request-48 ROOT files."
    )
    p.add_argument(
        "--data-glob",
        default="data/*.root",
        help="Local ROOT glob. Default: data/*.root",
    )
    p.add_argument(
        "--n-null",
        type=int,
        default=DEFAULT_NULL_N,
        help="Conditional multinomial empirical-null trials. Default: 10000",
    )
    p.add_argument(
        "--step-size",
        default="100 MB",
        help="uproot chunk size used while reading local ROOT files.",
    )
    p.add_argument(
        "--label",
        default="local_combined",
        help="Output label. Change this for run-group subsets.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fixed frequency and classification without reading data.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_null < 0:
        raise SystemExit("--n-null must be >= 0")

    print("CMS-frequency / free-phase diagnostic setup")
    print(f"CMS omega_m         = {base.CMS_OMEGA_M:.15f}")
    print(f"LHCb k(q2) fixed    = {K_FIXED:.15f}")
    print(f"CMS ref phase       = {CMS_PHASE:.10f} rad")
    print("frequency scan      = disabled")
    print("phase               = free")
    print("classification      = exploratory post-unblinding")

    if args.dry_run:
        return 0

    analyze_local(
        args.data_glob,
        n_null=args.n_null,
        step_size=args.step_size,
        label=args.label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
34_cms_fixed_k_crossrun_background_test.py

Cross-run phase-prediction / background-discrimination test at the exact
CMS-mapped LHCb frequency.

This is post-unblinding.  It does not scan frequency and is not a prospective
replication.  The prespecified protocol is frozen in
CMS_FIXED_K_CROSSRUN_BACKGROUND_TEST_PLAN_2026-08-31.md.

For each Chebyshev log-rate background degree 2..6:

  1. fit run group A with background + free fixed-k quadratures;
  2. freeze the fitted phase and test it in B while refitting B's background
     and a nonnegative target amplitude;
  3. reverse A/B;
  4. sum the two directional likelihood-ratio statistics;
  5. calibrate the complete two-way procedure with paired pseudoexperiments in
     which both training phases and both target backgrounds are refit.

The fixed frequency is

    k = omega_CMS / 2 = 3.512912912912913...

No result from this script is a detector/systematics-calibrated discovery
significance.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2


HERE = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


rob = _load_module(
    "cms_fixed_k_stage33", HERE / "33_cms_fixed_k_robustness_suite.py"
)

K_FIXED = float(rob.K_FIXED)
Q2_MIN = float(rob.Q2_MIN)
Q2_MAX = float(rob.Q2_MAX)
JPSI = tuple(float(x) for x in rob.NOMINAL_JPSI)
PSI2S = tuple(float(x) for x in rob.NOMINAL_PSI2S)
BINS = int(rob.NOMINAL_BINS)
SEED = 20260831
DEGREES = (2, 3, 4, 5, 6)

GROUP_A = "00382466"
GROUP_B = "00382467"

OUT_DIR = Path("outputs_cms_fixed_k_crossrun_background")

# Wide numerical safety bounds.  Fits report if a signal coefficient reaches
# a bound.  These bounds are not part of a transferred CMS amplitude claim.
BETA_BOUND = 50.0
QUAD_BOUND = 5.0
AMP_BOUND = 5.0


def wrap_phase(x: float) -> float:
    return float((x + math.pi) % (2.0 * math.pi) - math.pi)


def poisson_nll(N: np.ndarray, eta: np.ndarray) -> float:
    eta = np.clip(np.asarray(eta, dtype=float), -30.0, 30.0)
    lam = np.maximum(np.exp(eta), 1e-12)
    N = np.asarray(N, dtype=float)
    return float(np.sum(lam - N * np.log(lam)))


def build_group_hist(q2_values: np.ndarray) -> dict[str, np.ndarray]:
    q2_values = np.asarray(q2_values, dtype=float)
    counts, edges = np.histogram(q2_values, bins=BINS, range=(Q2_MIN, Q2_MAX))
    centers = 0.5 * (edges[:-1] + edges[1:])
    keep = ~rob.in_veto(centers, JPSI, PSI2S)
    N = counts[keep].astype(float)
    q2 = centers[keep].astype(float)
    ell = np.log(q2)
    return {
        "N": N,
        "q2": q2,
        "ell": ell,
        "centers_all": centers,
        "counts_all": counts.astype(float),
        "keep": keep,
    }


def cheb_design(ell: np.ndarray, degree: int) -> np.ndarray:
    ell = np.asarray(ell, dtype=float)
    lo = math.log(Q2_MIN)
    hi = math.log(Q2_MAX)
    x = 2.0 * (ell - lo) / (hi - lo) - 1.0
    return np.polynomial.chebyshev.chebvander(x, degree)


def fit_null(
    N: np.ndarray,
    ell: np.ndarray,
    degree: int,
    *,
    start_beta: np.ndarray | None = None,
) -> dict[str, Any]:
    N = np.asarray(N, dtype=float)
    X = cheb_design(ell, degree)

    if start_beta is None:
        beta0 = np.zeros(degree + 1, dtype=float)
        beta0[0] = math.log(max(float(np.mean(N)), 1e-9))
    else:
        beta0 = np.asarray(start_beta, dtype=float).copy()

    def objective(beta: np.ndarray) -> float:
        return poisson_nll(N, X @ beta)

    result = minimize(
        objective,
        x0=beta0,
        method="L-BFGS-B",
        bounds=[(-BETA_BOUND, BETA_BOUND)] * (degree + 1),
        options={"maxiter": 3000, "ftol": 1e-11, "gtol": 1e-8, "maxls": 50},
    )

    beta = np.asarray(result.x, dtype=float)
    eta = X @ beta
    lam = np.maximum(np.exp(np.clip(eta, -30.0, 30.0)), 1e-12)
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "beta": beta,
        "nll": float(result.fun),
        "lambda": lam,
        "bound_active": bool(np.any(np.abs(beta) >= BETA_BOUND - 1e-6)),
    }


def fit_free_phase(
    N: np.ndarray,
    ell: np.ndarray,
    degree: int,
    *,
    null_fit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Jointly fit Chebyshev background plus both fixed-k quadratures."""
    N = np.asarray(N, dtype=float)
    ell = np.asarray(ell, dtype=float)
    X = cheb_design(ell, degree)
    c = np.cos(K_FIXED * ell)
    s = np.sin(K_FIXED * ell)

    if null_fit is None:
        null_fit = fit_null(N, ell, degree)
    theta0 = np.concatenate([null_fit["beta"], np.array([0.0, 0.0])])

    def objective(theta: np.ndarray) -> float:
        beta = theta[: degree + 1]
        a, b = theta[degree + 1 : degree + 3]
        return poisson_nll(N, X @ beta + a * c + b * s)

    bounds = [(-BETA_BOUND, BETA_BOUND)] * (degree + 1) + [
        (-QUAD_BOUND, QUAD_BOUND),
        (-QUAD_BOUND, QUAD_BOUND),
    ]
    result = minimize(
        objective,
        x0=theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 4000, "ftol": 1e-11, "gtol": 1e-8, "maxls": 50},
    )

    theta = np.asarray(result.x, dtype=float)
    beta = theta[: degree + 1]
    a = float(theta[degree + 1])
    b = float(theta[degree + 2])
    A = float(math.hypot(a, b))
    phi = float(math.atan2(-b, a)) if A > 0.0 else 0.0
    q_free = float(max(0.0, 2.0 * (null_fit["nll"] - float(result.fun))))

    return {
        "success": bool(result.success),
        "message": str(result.message),
        "beta": beta,
        "a": a,
        "b": b,
        "A": A,
        "phi": phi,
        "nll": float(result.fun),
        "q_free": q_free,
        "p_chi2_2_diagnostic": float(chi2.sf(q_free, 2)),
        "quad_bound_active": bool(
            abs(a) >= QUAD_BOUND - 1e-6 or abs(b) >= QUAD_BOUND - 1e-6
        ),
    }


def fit_phase_locked_target(
    N: np.ndarray,
    ell: np.ndarray,
    degree: int,
    phase: float,
    *,
    null_fit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Refit target background and a nonnegative amplitude at a phase predicted
    from the other run group.
    """
    N = np.asarray(N, dtype=float)
    ell = np.asarray(ell, dtype=float)
    X = cheb_design(ell, degree)
    wave = np.cos(K_FIXED * ell + float(phase))

    if null_fit is None:
        null_fit = fit_null(N, ell, degree)

    def objective(theta: np.ndarray) -> float:
        beta = theta[: degree + 1]
        amp = float(theta[degree + 1])
        return poisson_nll(N, X @ beta + amp * wave)

    # A small positive start avoids numerical sticking exactly on the boundary.
    theta0 = np.concatenate([null_fit["beta"], np.array([0.01])])
    bounds = [(-BETA_BOUND, BETA_BOUND)] * (degree + 1) + [(0.0, AMP_BOUND)]
    result = minimize(
        objective,
        x0=theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 4000, "ftol": 1e-11, "gtol": 1e-8, "maxls": 50},
    )

    # Explicitly compare with the exact nested A=0 solution so an optimizer
    # artifact can never make the alternative look worse than the null.
    nll_alt = min(float(result.fun), float(null_fit["nll"]))
    if float(null_fit["nll"]) <= float(result.fun):
        amp = 0.0
        beta = np.asarray(null_fit["beta"], dtype=float)
    else:
        amp = float(result.x[degree + 1])
        beta = np.asarray(result.x[: degree + 1], dtype=float)

    q = float(max(0.0, 2.0 * (null_fit["nll"] - nll_alt)))
    p_chernoff = 1.0 if q <= 0.0 else float(0.5 * chi2.sf(q, 1))

    return {
        "success": bool(result.success),
        "message": str(result.message),
        "beta": beta,
        "A": amp,
        "nll": nll_alt,
        "q": q,
        "p_chernoff_diagnostic": p_chernoff,
        "amp_bound_active": bool(amp >= AMP_BOUND - 1e-6),
    }


def observed_for_degree(
    N_A: np.ndarray,
    N_B: np.ndarray,
    ell: np.ndarray,
    degree: int,
) -> dict[str, Any]:
    null_A = fit_null(N_A, ell, degree)
    null_B = fit_null(N_B, ell, degree)
    free_A = fit_free_phase(N_A, ell, degree, null_fit=null_A)
    free_B = fit_free_phase(N_B, ell, degree, null_fit=null_B)

    # A phase predicts B; B phase predicts A.
    test_B_from_A = fit_phase_locked_target(
        N_B, ell, degree, free_A["phi"], null_fit=null_B
    )
    test_A_from_B = fit_phase_locked_target(
        N_A, ell, degree, free_B["phi"], null_fit=null_A
    )

    phase_delta = wrap_phase(float(free_A["phi"]) - float(free_B["phi"]))
    q_joint = float(test_B_from_A["q"] + test_A_from_B["q"])

    return {
        "degree": int(degree),
        "null_A": null_A,
        "null_B": null_B,
        "free_A": free_A,
        "free_B": free_B,
        "test_B_from_A": test_B_from_A,
        "test_A_from_B": test_A_from_B,
        "phase_delta_rad": phase_delta,
        "phase_delta_deg": float(math.degrees(phase_delta)),
        "q_joint": q_joint,
    }


def paired_null(
    *,
    N_A: np.ndarray,
    N_B: np.ndarray,
    ell: np.ndarray,
    degree: int,
    null_A_obs: dict[str, Any],
    null_B_obs: dict[str, Any],
    n_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Calibrate the complete two-way phase-prediction procedure under null."""
    if n_trials <= 0:
        return np.empty(0, dtype=float)

    total_A = int(np.sum(N_A))
    total_B = int(np.sum(N_B))
    probs_A = np.asarray(null_A_obs["lambda"], dtype=float)
    probs_B = np.asarray(null_B_obs["lambda"], dtype=float)
    probs_A /= np.sum(probs_A)
    probs_B /= np.sum(probs_B)

    out = np.empty(n_trials, dtype=float)
    step = max(1, n_trials // 10)

    for i in range(n_trials):
        pseudo_A = rng.multinomial(total_A, probs_A).astype(float)
        pseudo_B = rng.multinomial(total_B, probs_B).astype(float)

        null_A = fit_null(pseudo_A, ell, degree)
        null_B = fit_null(pseudo_B, ell, degree)
        free_A = fit_free_phase(pseudo_A, ell, degree, null_fit=null_A)
        free_B = fit_free_phase(pseudo_B, ell, degree, null_fit=null_B)

        B_from_A = fit_phase_locked_target(
            pseudo_B, ell, degree, free_A["phi"], null_fit=null_B
        )
        A_from_B = fit_phase_locked_target(
            pseudo_A, ell, degree, free_B["phi"], null_fit=null_A
        )
        out[i] = float(B_from_A["q"] + A_from_B["q"])

        if (i + 1) % step == 0 or i + 1 == n_trials:
            print(
                f"[degree {degree} paired-null] {i + 1:,}/{n_trials:,}",
                flush=True,
            )

    return out


def empirical_p(observed_q: float, null_q: np.ndarray) -> dict[str, Any]:
    if null_q.size == 0:
        return {"trials": 0, "exceedances": None, "add_one_p": None}
    exc = int(np.count_nonzero(null_q >= float(observed_q)))
    return {
        "trials": int(null_q.size),
        "exceedances": exc,
        "add_one_p": float((exc + 1) / (null_q.size + 1)),
    }


def compact_row(obs: dict[str, Any], emp: dict[str, Any]) -> dict[str, Any]:
    return {
        "degree": obs["degree"],
        "k_fixed": K_FIXED,
        "frequency_scanned": False,
        "A_free_group_A": float(obs["free_A"]["A"]),
        "phi_group_A_rad": float(obs["free_A"]["phi"]),
        "q_free_group_A": float(obs["free_A"]["q_free"]),
        "A_free_group_B": float(obs["free_B"]["A"]),
        "phi_group_B_rad": float(obs["free_B"]["phi"]),
        "q_free_group_B": float(obs["free_B"]["q_free"]),
        "phase_delta_A_minus_B_rad": float(obs["phase_delta_rad"]),
        "phase_delta_A_minus_B_deg": float(obs["phase_delta_deg"]),
        "q_B_given_phase_A": float(obs["test_B_from_A"]["q"]),
        "A_B_given_phase_A": float(obs["test_B_from_A"]["A"]),
        "q_A_given_phase_B": float(obs["test_A_from_B"]["q"]),
        "A_A_given_phase_B": float(obs["test_A_from_B"]["A"]),
        "q_joint": float(obs["q_joint"]),
        "empirical_trials": emp["trials"],
        "empirical_exceedances": emp["exceedances"],
        "empirical_add_one_p": emp["add_one_p"],
        "free_A_quad_bound_active": bool(obs["free_A"]["quad_bound_active"]),
        "free_B_quad_bound_active": bool(obs["free_B"]["quad_bound_active"]),
        "target_B_amp_bound_active": bool(obs["test_B_from_A"]["amp_bound_active"]),
        "target_A_amp_bound_active": bool(obs["test_A_from_B"]["amp_bound_active"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    q2_by_group, provenance = rob.load_q2_by_group(
        args.data_glob, step_size=args.step_size
    )

    hist_A = build_group_hist(q2_by_group[GROUP_A])
    hist_B = build_group_hist(q2_by_group[GROUP_B])
    N_A = hist_A["N"]
    N_B = hist_B["N"]
    ell_A = hist_A["ell"]
    ell_B = hist_B["ell"]
    if not np.allclose(ell_A, ell_B, rtol=0.0, atol=1e-14):
        raise RuntimeError("Run-group active-bin coordinates do not match")
    ell = ell_A

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}

    for degree in DEGREES:
        print("\n" + "=" * 88)
        print(f"CHEBYSHEV DEGREE {degree}: observed cross-run phase prediction")
        print("=" * 88)

        obs = observed_for_degree(N_A, N_B, ell, degree)
        print(
            f"phi_A={obs['free_A']['phi']:.8f}  "
            f"phi_B={obs['free_B']['phi']:.8f}  "
            f"delta={obs['phase_delta_deg']:.3f} deg"
        )
        print(
            f"q(B|phi_A)={obs['test_B_from_A']['q']:.8f}  "
            f"q(A|phi_B)={obs['test_A_from_B']['q']:.8f}  "
            f"q_joint={obs['q_joint']:.8f}"
        )

        rng = np.random.default_rng(SEED + 10_000 * degree)
        null_q = paired_null(
            N_A=N_A,
            N_B=N_B,
            ell=ell,
            degree=degree,
            null_A_obs=obs["null_A"],
            null_B_obs=obs["null_B"],
            n_trials=args.n_null,
            rng=rng,
        )
        emp = empirical_p(obs["q_joint"], null_q)
        print(
            f"paired empirical p={emp['add_one_p']} "
            f"({emp['exceedances']}/{emp['trials']} exceedances)"
        )

        if null_q.size:
            pd.DataFrame({"q_joint_null": null_q}).to_csv(
                OUT_DIR / f"degree_{degree}_paired_null.csv", index=False
            )

        row = compact_row(obs, emp)
        rows.append(row)
        detail[str(degree)] = {
            "observed": {
                "phase_A_rad": float(obs["free_A"]["phi"]),
                "phase_B_rad": float(obs["free_B"]["phi"]),
                "phase_delta_rad": float(obs["phase_delta_rad"]),
                "phase_delta_deg": float(obs["phase_delta_deg"]),
                "free_A_amplitude": float(obs["free_A"]["A"]),
                "free_B_amplitude": float(obs["free_B"]["A"]),
                "free_A_q": float(obs["free_A"]["q_free"]),
                "free_B_q": float(obs["free_B"]["q_free"]),
                "q_B_given_phase_A": float(obs["test_B_from_A"]["q"]),
                "A_B_given_phase_A": float(obs["test_B_from_A"]["A"]),
                "q_A_given_phase_B": float(obs["test_A_from_B"]["q"]),
                "A_A_given_phase_B": float(obs["test_A_from_B"]["A"]),
                "q_joint": float(obs["q_joint"]),
            },
            "paired_null": emp,
        }

    table = pd.DataFrame(rows)
    table_path = OUT_DIR / "crossrun_background_degree_table.csv"
    table.to_csv(table_path, index=False)

    summary = {
        "test": "CMS_fixed_k_crossrun_phase_prediction_background_refit",
        "classification": "exploratory_post_unblinding_prespecified_crossrun_robustness",
        "plan": "CMS_FIXED_K_CROSSRUN_BACKGROUND_TEST_PLAN_2026-08-31.md",
        "frequency": {
            "cms_omega_m": float(rob.base.CMS_OMEGA_M),
            "lhcb_k_q2": K_FIXED,
            "frequency_scanned": False,
        },
        "groups": {"A": GROUP_A, "B": GROUP_B},
        "selection": {
            "q2_range_GeV2": [Q2_MIN, Q2_MAX],
            "Jpsi_veto_GeV2": list(JPSI),
            "psi2S_veto_GeV2": list(PSI2S),
            "bins": BINS,
        },
        "background_degrees": list(DEGREES),
        "paired_null_trials_per_degree": int(args.n_null),
        "counts": {
            "group_A_active": int(np.sum(N_A)),
            "group_B_active": int(np.sum(N_B)),
            "active_bins": int(len(ell)),
        },
        "results": detail,
        "table": str(table_path),
        "source_provenance": provenance,
        "guardrails": [
            "Request-48 was already unblinded before this test.",
            "The frequency is fixed and never scanned.",
            "Training phase is estimated in one run group and transferred to the other.",
            "The target smooth background and target nonnegative amplitude are refit for each directional test.",
            "Every paired null pseudoexperiment re-estimates both training phases and refits both target backgrounds.",
            "Empirical p-values calibrate this analysis model, not detector/reconstruction/physics systematics.",
            "Zero exceedances are only Monte Carlo resolution floors.",
            "No CMS and LHCb p-values or Z values are combined.",
        ],
        "seed": SEED,
    }

    summary_path = OUT_DIR / "crossrun_background_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("CROSS-RUN FIXED-k BACKGROUND TEST COMPLETE")
    print("=" * 88)
    print(f"k fixed            : {K_FIXED:.15f}")
    print(f"degrees            : {', '.join(str(x) for x in DEGREES)}")
    print(f"paired null/degree : {args.n_null}")
    print(f"table              : {table_path}")
    print(f"summary            : {summary_path}")
    print("\nDegree summary")
    for row in rows:
        print(
            f"d={row['degree']}  q_joint={row['q_joint']:.6f}  "
            f"phase_delta={row['phase_delta_A_minus_B_deg']:.3f} deg  "
            f"p_emp={row['empirical_add_one_p']}"
        )

    return summary


def print_plan() -> None:
    print("CMS fixed-k cross-run background discrimination test")
    print(f"k fixed              = {K_FIXED:.15f}")
    print("frequency scan       = disabled")
    print(f"run group A          = {GROUP_A}")
    print(f"run group B          = {GROUP_B}")
    print(f"Chebyshev degrees    = {', '.join(str(x) for x in DEGREES)}")
    print("directions           = A phase -> B and B phase -> A")
    print("target background    = refit at same degree")
    print("target amplitude     = nonnegative, phase frozen from other group")
    print("paired null          = refit phases + target backgrounds every pseudo-pair")
    print("classification       = exploratory post-unblinding")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fixed-CMS-k cross-run phase prediction with refit Chebyshev backgrounds."
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
        default=1000,
        help="Paired refit-null pseudoexperiments per degree. Default: 1000",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the frozen setup without reading data.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_null < 0:
        raise SystemExit("--n-null must be >= 0")
    print_plan()
    if args.dry_run:
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

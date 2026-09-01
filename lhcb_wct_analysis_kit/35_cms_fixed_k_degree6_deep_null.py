#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
35_cms_fixed_k_degree6_deep_null.py

Deep paired-null calibration of the weakest / most flexible prespecified
background case from stage 34: degree-6 Chebyshev background at the exact
CMS-mapped LHCb frequency.

The statistical procedure is unchanged from stage 34.  Only the number of
paired pseudoexperiments is increased, by default from 1,000 to 10,000.

See CMS_FIXED_K_DEG6_DEEP_NULL_PLAN_2026-08-31.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TARGET = HERE / "34_cms_fixed_k_crossrun_background_test.py"

_spec = importlib.util.spec_from_file_location("cms_crossrun_stage34", TARGET)
s34 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(s34)


DEGREE = 6
DEFAULT_NULL_N = 10_000
OUT_DIR = Path("outputs_cms_fixed_k_degree6_deep_null")
PLAN = "CMS_FIXED_K_DEG6_DEEP_NULL_PLAN_2026-08-31.md"


def zero_exceedance_upper(n: int, alpha: float) -> float | None:
    """Exact one-sided binomial upper bound for zero successes in n trials."""
    if n <= 0:
        return None
    return float(1.0 - alpha ** (1.0 / float(n)))


def run(args: argparse.Namespace) -> dict:
    q2_by_group, provenance = s34.rob.load_q2_by_group(
        args.data_glob, step_size=args.step_size
    )

    hist_A = s34.build_group_hist(q2_by_group[s34.GROUP_A])
    hist_B = s34.build_group_hist(q2_by_group[s34.GROUP_B])

    N_A = hist_A["N"]
    N_B = hist_B["N"]
    ell_A = hist_A["ell"]
    ell_B = hist_B["ell"]
    if not np.allclose(ell_A, ell_B, rtol=0.0, atol=1e-14):
        raise RuntimeError("Run-group active-bin coordinates do not match")
    ell = ell_A

    print("\nObserved degree-6 cross-run statistic")
    print("-------------------------------------")
    obs = s34.observed_for_degree(N_A, N_B, ell, DEGREE)
    print(f"phi_A             : {obs['free_A']['phi']:.10f} rad")
    print(f"phi_B             : {obs['free_B']['phi']:.10f} rad")
    print(f"phase difference  : {obs['phase_delta_deg']:.6f} deg")
    print(f"q(B|phi_A)        : {obs['test_B_from_A']['q']:.10f}")
    print(f"q(A|phi_B)        : {obs['test_A_from_B']['q']:.10f}")
    print(f"q_joint           : {obs['q_joint']:.10f}")

    rng = np.random.default_rng(s34.SEED + 10_000 * DEGREE)
    null_q = s34.paired_null(
        N_A=N_A,
        N_B=N_B,
        ell=ell,
        degree=DEGREE,
        null_A_obs=obs["null_A"],
        null_B_obs=obs["null_B"],
        n_trials=args.n_null,
        rng=rng,
    )
    emp = s34.empirical_p(obs["q_joint"], null_q)

    upper95 = None
    upper99 = None
    if emp["exceedances"] == 0 and emp["trials"] > 0:
        upper95 = zero_exceedance_upper(emp["trials"], 0.05)
        upper99 = zero_exceedance_upper(emp["trials"], 0.01)

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    if null_q.size:
        pd.DataFrame({"q_joint_null": null_q}).to_csv(
            OUT_DIR / "degree6_deep_paired_null.csv", index=False
        )

    summary = {
        "test": "CMS_fixed_k_degree6_crossrun_deep_paired_null",
        "classification": "exploratory_post_unblinding_deep_tail_calibration",
        "plan": PLAN,
        "frequency": {
            "cms_omega_m": float(s34.rob.base.CMS_OMEGA_M),
            "lhcb_k_q2": float(s34.K_FIXED),
            "frequency_scanned": False,
        },
        "background": {
            "family": "Chebyshev_log_rate",
            "degree": DEGREE,
        },
        "groups": {
            "A": s34.GROUP_A,
            "B": s34.GROUP_B,
        },
        "selection": {
            "q2_range_GeV2": [float(s34.Q2_MIN), float(s34.Q2_MAX)],
            "Jpsi_veto_GeV2": list(s34.JPSI),
            "psi2S_veto_GeV2": list(s34.PSI2S),
            "bins": int(s34.BINS),
        },
        "observed": {
            "phi_A_rad": float(obs["free_A"]["phi"]),
            "phi_B_rad": float(obs["free_B"]["phi"]),
            "phase_delta_rad": float(obs["phase_delta_rad"]),
            "phase_delta_deg": float(obs["phase_delta_deg"]),
            "q_B_given_phase_A": float(obs["test_B_from_A"]["q"]),
            "q_A_given_phase_B": float(obs["test_A_from_B"]["q"]),
            "q_joint": float(obs["q_joint"]),
        },
        "paired_null": {
            **emp,
            "zero_exceedance_exact_one_sided_95_upper": upper95,
            "zero_exceedance_exact_one_sided_99_upper": upper99,
        },
        "source_provenance": provenance,
        "guardrails": [
            "This is post-unblinding and not a prospective replication.",
            "The model and statistic are unchanged from the degree-6 stage-34 test; only null depth is increased.",
            "The frequency is fixed and never scanned.",
            "The background degree is fixed at 6 because it was the most flexible / weakest prespecified stage-34 stress test.",
            "Every pseudoexperiment refits both training phases and both target backgrounds.",
            "The empirical tail calibrates this analysis model, not detector/reconstruction/physics systematics.",
            "The add-one floor is not a physical p-value estimate and must not be converted into discovery sigma.",
            "No CMS and LHCb p-values or Z values are combined.",
        ],
        "seed": int(s34.SEED + 10_000 * DEGREE),
    }

    summary_path = OUT_DIR / "degree6_deep_null_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDegree-6 deep paired-null result")
    print("--------------------------------")
    print(f"k fixed            : {s34.K_FIXED:.15f}")
    print(f"degree             : {DEGREE}")
    print(f"q_joint observed   : {obs['q_joint']:.10f}")
    print(
        f"empirical p        : {emp['add_one_p']} "
        f"({emp['exceedances']}/{emp['trials']} exceedances; add-one)"
    )
    if upper95 is not None:
        print(f"zero-exc 95% upper : {upper95:.10g}")
        print(f"zero-exc 99% upper : {upper99:.10g}")
    print(f"summary            : {summary_path}")
    return summary


def print_plan() -> None:
    print("CMS fixed-k degree-6 deep paired-null calibration")
    print(f"k fixed              = {s34.K_FIXED:.15f}")
    print("frequency scan       = disabled")
    print(f"Chebyshev degree     = {DEGREE}")
    print(f"run group A          = {s34.GROUP_A}")
    print(f"run group B          = {s34.GROUP_B}")
    print("statistic            = stage-34 two-way phase-transfer q_joint")
    print("paired null          = full phase/background refit every pseudo-pair")
    print("default null trials  = 10000")
    print("classification       = exploratory post-unblinding deep-tail calibration")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deep degree-6 paired-null calibration for the fixed-CMS-k LHCb cross-run test."
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
        default=DEFAULT_NULL_N,
        help="Degree-6 paired pseudoexperiments. Default: 10000",
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

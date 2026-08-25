#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
31_run_group_triplet_dilation.py

Independent run-group test of the "moving ratio" interpretation.

Stage 30 established bidirectional out-of-sample prediction of the residual
family while the independently preferred high-k coordinate moved.  This stage
asks whether the more stable object is a three-well geometry related between
run groups by one multiplicative dilation rather than one fixed k or n.

The well scan deliberately reuses the stage-19 implementation and its exact
repository definition

    Q_low  = n1 / n2
    Q_high = n3 / (2 n2)

with n proportional to k on a fixed active support.  Two triplets are reported:

  * strength: the three strongest raw wells (no Koide target in the selector)
  * koide: minimum stage-19 Koide error among the top-N raw wells, but WITHOUT
    the old (10,15,20) integer-position tie-break

For each independently selected pair, fit

    k_B ~= a k_A

and report scale error, normalized shape error, Q changes, and phase coherence.
A combinatorial ranking among all top-N triplet pairs is also reported.  That
ranking is descriptive/conditional, NOT a full event-level global null and,
for the Koide-selected pair, does not correct for selection on Q~2/3.

n = k DeltaEll_A/(2 pi) remains domain-dependent.  Here both run groups use the
same support, so k and n have identical ratios and dilation factors.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs_run_group_triplet_dilation"
CACHE = OUT / "selected_q2_cache.npz"


def load_numbered(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Reuse the exact event selection / q2 streaming from stage 30 and the exact
# well-first spectral scan / Q definition from stage 19.
hold = load_numbered("stage30_holdout", "30_run_group_holdout.py")
well = load_numbered("stage19_well", "19_koide_well.py")


def ordered_triplet(rows, selector: str) -> dict:
    rows = sorted(rows, key=lambda r: float(r.k))
    k = np.array([r.k for r in rows], dtype=float)
    n = np.array([r.n_eff for r in rows], dtype=float)
    d = np.array([r.deltaD for r in rows], dtype=float)
    phi = np.array([r.phi2 for r in rows], dtype=float)
    q_low = float(n[0] / n[1])
    q_high = float(n[2] / (2.0 * n[1]))
    q_mean = 0.5 * (q_low + q_high)
    qerr = math.sqrt((q_low - well.KOIDE_Q) ** 2 + (q_high - well.KOIDE_Q) ** 2)
    return {
        "selector": selector,
        "k": k, "n": n, "deltaD": d, "phi": phi,
        "mean_deltaD": float(np.mean(d)),
        "Q_low": q_low, "Q_high": q_high, "Q_mean": q_mean,
        "koide_error": float(qerr),
        "shape_low_mid": float(k[0] / k[1]),
        "shape_high_mid": float(k[2] / k[1]),
    }


def enumerate_triplets(wells, top_n: int) -> list[dict]:
    top = sorted(wells, key=lambda r: float(r.deltaD), reverse=True)[:top_n]
    return [ordered_triplet(c, "enumerated") for c in itertools.combinations(top, 3)]


def select_strength(wells) -> dict:
    if len(wells) < 3:
        raise RuntimeError("Fewer than three wells")
    rows = sorted(wells, key=lambda r: float(r.deltaD), reverse=True)[:3]
    return ordered_triplet(rows, "top3_strength")


def select_koide(triplets: list[dict]) -> dict:
    # Deliberately no distance-to-(10,15,20) term: coordinates are allowed to move.
    t = min(triplets, key=lambda x: (x["koide_error"], -x["mean_deltaD"]))
    out = dict(t)
    out["selector"] = "min_koide_error_no_integer_lock"
    return out


def wrap_phase(x):
    return (np.asarray(x, float) + np.pi) % (2.0 * np.pi) - np.pi


def dilation_metrics(A: dict, B: dict) -> dict:
    kA, kB = np.asarray(A["k"], float), np.asarray(B["k"], float)
    a = float(np.dot(kA, kB) / np.dot(kA, kA))
    pred = a * kA
    rmse = float(np.sqrt(np.mean((kB - pred) ** 2)))
    rel = float(rmse / np.mean(kB))

    log_ratio = np.log(kB) - np.log(kA)
    loga = float(np.mean(log_ratio))
    log_rmse = float(np.sqrt(np.mean((log_ratio - loga) ** 2)))

    shapeA = np.array([A["shape_low_mid"], A["shape_high_mid"]])
    shapeB = np.array([B["shape_low_mid"], B["shape_high_mid"]])
    dphi = wrap_phase(np.asarray(B["phi"]) - np.asarray(A["phi"]))
    z = np.mean(np.exp(1j * dphi))

    return {
        "scale_a": a,
        "scale_rmse_k": rmse,
        "scale_relative_rmse": rel,
        "scale_predicted_k_B": pred.tolist(),
        "log_scale_a": float(np.exp(loga)),
        "log_scale_rmse": log_rmse,
        "shape_distance": float(np.linalg.norm(shapeB - shapeA)),
        "delta_Q_low": float(B["Q_low"] - A["Q_low"]),
        "delta_Q_high": float(B["Q_high"] - A["Q_high"]),
        "delta_Q_mean": float(B["Q_mean"] - A["Q_mean"]),
        "phase_coherence": float(abs(z)),
        "mean_phase_offset": float(np.angle(z)),
        "delta_phi": dphi.tolist(),
    }


def combinatorial_reference(allA: list[dict], allB: list[dict], obs: dict) -> dict:
    scale, shape, phase = [], [], []
    for A in allA:
        for B in allB:
            m = dilation_metrics(A, B)
            scale.append(m["scale_relative_rmse"])
            shape.append(m["shape_distance"])
            phase.append(m["phase_coherence"])
    scale, shape, phase = map(np.asarray, (scale, shape, phase))
    return {
        "reference_type": "conditional_combinatorial_triplet_pair_ranking",
        "n_pairs": int(scale.size),
        "p_scale_relative_rmse": float(np.mean(scale <= obs["scale_relative_rmse"])),
        "p_shape_distance": float(np.mean(shape <= obs["shape_distance"])),
        "p_phase_coherence": float(np.mean(phase >= obs["phase_coherence"])),
        "median_scale_relative_rmse": float(np.median(scale)),
        "median_shape_distance": float(np.median(shape)),
        "median_phase_coherence": float(np.median(phase)),
        "warning": (
            "Triplet-pair ranking only: not an event-level/global null. "
            "For the Koide-selected pair it does not correct for selection on Q~2/3."
        ),
    }


def serial(t: dict, group: str) -> dict:
    return {
        "run_group": group, "selector": t["selector"],
        "k1": float(t["k"][0]), "k2": float(t["k"][1]), "k3": float(t["k"][2]),
        "n1": float(t["n"][0]), "n2": float(t["n"][1]), "n3": float(t["n"][2]),
        "deltaD1": float(t["deltaD"][0]), "deltaD2": float(t["deltaD"][1]), "deltaD3": float(t["deltaD"][2]),
        "phi1": float(t["phi"][0]), "phi2": float(t["phi"][1]), "phi3": float(t["phi"][2]),
        "mean_deltaD": float(t["mean_deltaD"]),
        "Q_low": float(t["Q_low"]), "Q_high": float(t["Q_high"]), "Q_mean": float(t["Q_mean"]),
        "koide_error": float(t["koide_error"]),
        "shape_low_mid": float(t["shape_low_mid"]), "shape_high_mid": float(t["shape_high_mid"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-size", default="100 MB")
    ap.add_argument("--top-wells", type=int, default=12)
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    provenance = []
    groups = list(hold.RUN_FILES)
    if args.reuse_cache and CACHE.exists():
        print(f"[cache] loading {CACHE}")
        z = np.load(CACHE)
        q2 = {g: np.asarray(z[g], float) for g in groups}
    else:
        q2 = {}
        for g in groups:
            q2[g], p = hold.stream_selected_q2(g, step_size=args.step_size)
            provenance.extend(p)
        np.savez_compressed(CACHE, **q2)
        print(f"[cache] saved {CACHE}")

    scans, wells_by_group = [], {}
    all_wells = []
    for g in groups:
        print("\n" + "=" * 90)
        print(f"INDEPENDENT STAGE-19 WELL SCAN: {g}")
        print("=" * 90)
        ell, counts = well.make_histogram(q2[g], well.N_BINS)
        baseline = well.kde_baseline(ell, counts, well.KDE_BANDWIDTH_SCALE)
        sr = well.scan_continuous_k(g, ell, counts, baseline)
        wr = well.find_wells(g, sr)
        scans.extend(sr); all_wells.extend(wr); wells_by_group[g] = wr
        print(f"[wells] {g}: {len(wr)}")

    scan_df = pd.DataFrame([asdict(x) for x in scans]).rename(columns={"region": "run_group"})
    wells_df = pd.DataFrame([asdict(x) for x in all_wells]).rename(columns={"region": "run_group"})
    scan_df.to_csv(OUT / "run_group_scan_curve.csv", index=False)
    wells_df.to_csv(OUT / "run_group_wells.csv", index=False)

    gA, gB = groups
    allA = enumerate_triplets(wells_by_group[gA], args.top_wells)
    allB = enumerate_triplets(wells_by_group[gB], args.top_wells)
    sA, sB = select_strength(wells_by_group[gA]), select_strength(wells_by_group[gB])
    qA, qB = select_koide(allA), select_koide(allB)
    sm, qm = dilation_metrics(sA, sB), dilation_metrics(qA, qB)

    triplet_rows = [serial(sA, gA), serial(sB, gB), serial(qA, gA), serial(qB, gB)]
    pd.DataFrame(triplet_rows).to_csv(OUT / "run_group_triplets.csv", index=False)

    summary = {
        "test": "independent_run_group_triplet_dilation",
        "status": "COMPLETE",
        "question": "Is the repeatable object a fixed spectral coordinate or a triplet geometry that moves by coherent dilation?",
        "configuration": {
            "active_intervals": well.ACTIVE_INTERVALS,
            "delta_ell_active": well.DELTA_ELL_ACTIVE,
            "k1_fixed": well.K1_FIXED,
            "k_scan": [well.K_SCAN_MIN, well.K_SCAN_MAX, well.N_K_SCAN],
            "n_bins": well.N_BINS,
            "kde_bandwidth_scale": well.KDE_BANDWIDTH_SCALE,
            "top_wells": args.top_wells,
            "koide_Q": well.KOIDE_Q,
        },
        "event_counts": {
            g: {
                "selected_pre_veto": int(len(q2[g])),
                "active": int(np.count_nonzero(well.in_active_intervals(q2[g]))),
                "n_wells": int(len(wells_by_group[g])),
            } for g in groups
        },
        "strength_triplets": {
            gA: serial(sA, gA), gB: serial(sB, gB),
            "dilation": sm,
            "combinatorial_reference": combinatorial_reference(allA, allB, sm),
            "interpretation": "Target-independent: triplets are the three strongest raw wells only."
        },
        "koide_triplets": {
            gA: serial(qA, gA), gB: serial(qB, gB),
            "dilation": qm,
            "combinatorial_reference": combinatorial_reference(allA, allB, qm),
            "interpretation": (
                "Stage-19 Q geometry with the integer-position lock removed. "
                "Selector favors Q~2/3, so its reference p-values are descriptive."
            ),
        },
        "guardrails": [
            "No fixed n=15 or fixed (10,15,20) coordinates are required.",
            "n is domain-dependent; same support here makes k and n ratios equivalent.",
            "Strength selector does not use Koide geometry.",
            "Koide selector matches stage 19: Q_low=n1/n2 and Q_high=n3/(2*n2).",
            "Combinatorial rankings are not event-level/global null p-values.",
            "Phase coherence is diagnostic only and is not a pass/fail criterion."
        ],
        "provenance": provenance,
    }
    (OUT / "run_group_dilation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("RUN-GROUP MOVING-TRIPLET SUMMARY")
    print("=" * 90)
    for label, A, B, m in [("STRENGTH", sA, sB, sm), ("KOIDE", qA, qB, qm)]:
        print(f"[{label}] {gA} k={np.round(A['k'],4).tolist()} Q=({A['Q_low']:.6f},{A['Q_high']:.6f})")
        print(f"[{label}] {gB} k={np.round(B['k'],4).tolist()} Q=({B['Q_low']:.6f},{B['Q_high']:.6f})")
        print(f"[{label}] a={m['scale_a']:.8f} rel_RMSE={m['scale_relative_rmse']:.6g} shape={m['shape_distance']:.6g} phase_C={m['phase_coherence']:.6g}")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()

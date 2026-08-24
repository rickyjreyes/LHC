"""
30_run_group_holdout.py

Strict run-group holdout test for the LHCb request-48 B0 -> K*0 mu+ mu- sample.

Question
--------
Can a high-k residual mode discovered using one run group predict the other run
group with no frequency, amplitude, or phase re-selection on the held-out data?

Design
------
1. Stream the six public request-48 ROOT files directly from the LHCb Open Data
   Ntupling Service. Only the branches needed for q2, B0 mass and K* mass are
   read.
2. Reproduce the stage-09d signal selection and widened charmonium vetoes.
3. On TRAIN only, build the stage-09d KDE baseline, fit the fixed low mode
   k1=7.61054, scan k2 in [18,24], and freeze the best two-mode fit.
4. Evaluate the frozen train base shape and frozen train two-mode shape on TEST.
   No held-out frequency scan, no Koide selector, and no held-out phase or
   amplitude fit is allowed.
5. Calibrate the held-out log-likelihood-ratio with a multinomial null generated
   from the frozen TRAIN base shape, conditional on the observed TEST count.
6. Repeat in both directions: 00382466 -> 00382467 and reverse.

This is an internal cross-run replication test. A positive result demonstrates
predictive reproducibility across the two run groups; it does not establish a
physical origin or WCT uniquely.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
from scipy.optimize import minimize
from scipy.stats import gaussian_kde, norm


# -----------------------------------------------------------------------------
# Locked analysis configuration: stage 09d
# -----------------------------------------------------------------------------
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
K1_FIXED = 7.61054
K2_MIN = 18.0
K2_MAX = 24.0
N_K2 = 601
A1_MAX = 0.10
A2_MAX = 0.10
ETA_CLIP = 0.20
SEED = 20260823
DEFAULT_NULL_N = 100_000
TREE_NAME = "B0_KstMuMu/DecayTree"

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

BRANCHES = [
    "B0_M",
    "Kst_892_0_M",
    "muplus_PX", "muplus_PY", "muplus_PZ", "muplus_PE",
    "muminus_PX", "muminus_PY", "muminus_PZ", "muminus_PE",
]

OUT_DIR = Path("outputs_run_group_holdout")


def in_veto_q2(q2: np.ndarray) -> np.ndarray:
    q2 = np.asarray(q2, dtype=float)
    return (
        ((q2 >= JPSI_VETO[0]) & (q2 <= JPSI_VETO[1]))
        | ((q2 >= PSI2S_VETO[0]) & (q2 <= PSI2S_VETO[1]))
    )


def active_delta_ell() -> float:
    intervals = [(Q2_MIN, JPSI_VETO[0]),
                 (JPSI_VETO[1], PSI2S_VETO[0]),
                 (PSI2S_VETO[1], Q2_MAX)]
    return float(sum(math.log(b / a) for a, b in intervals))


DELTA_ELL_A = active_delta_ell()


def derive_q2(arr: dict[str, np.ndarray]) -> np.ndarray:
    e = np.asarray(arr["muplus_PE"], float) + np.asarray(arr["muminus_PE"], float)
    px = np.asarray(arr["muplus_PX"], float) + np.asarray(arr["muminus_PX"], float)
    py = np.asarray(arr["muplus_PY"], float) + np.asarray(arr["muminus_PY"], float)
    pz = np.asarray(arr["muplus_PZ"], float) + np.asarray(arr["muminus_PZ"], float)
    return (e * e - px * px - py * py - pz * pz) / 1.0e6


def stream_selected_q2(run_group: str, *, step_size: str = "100 MB") -> tuple[np.ndarray, list[dict]]:
    pieces: list[np.ndarray] = []
    provenance: list[dict] = []

    for filename in RUN_FILES[run_group]:
        url = f"{REMOTE_BASE}/{filename}"
        print(f"[remote] {run_group}: {filename}", flush=True)
        n_seen = 0
        n_selected = 0

        # Opening only ROOT metadata first makes branch failures explicit before a
        # long iteration starts.
        with uproot.open(url, timeout=300) as f:
            tree = f[TREE_NAME]
            missing = [b for b in BRANCHES if b not in tree.keys()]
            if missing:
                raise KeyError(f"{filename}: missing branches {missing}")

            for arr in tree.iterate(BRANCHES, step_size=step_size, library="np"):
                n_chunk = len(arr["B0_M"])
                n_seen += n_chunk
                q2 = derive_q2(arr)
                bm = np.asarray(arr["B0_M"], float)
                km = np.asarray(arr["Kst_892_0_M"], float)

                keep = np.isfinite(q2) & np.isfinite(bm) & np.isfinite(km)
                keep &= (q2 >= Q2_MIN) & (q2 <= Q2_MAX)
                keep &= (bm >= B0_M_MIN) & (bm <= B0_M_MAX)
                keep &= (km >= KST_M_MIN) & (km <= KST_M_MAX)

                chosen = np.asarray(q2[keep], dtype=np.float64)
                if chosen.size:
                    pieces.append(chosen)
                    n_selected += int(chosen.size)

        provenance.append({
            "run_group": run_group,
            "file": filename,
            "entries_seen": int(n_seen),
            "selected_pre_veto": int(n_selected),
        })
        print(f"         entries={n_seen:,} selected_pre_veto={n_selected:,}", flush=True)

    if not pieces:
        raise RuntimeError(f"No selected q2 events for run group {run_group}")

    q2 = np.concatenate(pieces)
    print(
        f"[group] {run_group}: selected_pre_veto={len(q2):,}; "
        f"active={np.count_nonzero(~in_veto_q2(q2)):,}",
        flush=True,
    )
    return q2, provenance


def make_binned_counts(q2_values: np.ndarray) -> dict[str, np.ndarray]:
    q2_values = np.asarray(q2_values, dtype=float)
    counts, edges = np.histogram(q2_values, bins=Q2_BINS, range=(Q2_MIN, Q2_MAX))
    centers = 0.5 * (edges[:-1] + edges[1:])
    veto = in_veto_q2(centers)

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
    baseline = np.maximum(dens * len(kde_train) * bin_width, 1e-9)

    keep = ~veto
    N = counts[keep].astype(float)
    B = baseline[keep].astype(float)
    q2 = centers[keep].astype(float)
    ell = np.log(q2)

    B *= np.sum(N) / max(np.sum(B), 1e-12)
    B = np.maximum(B, 1e-12)

    return {
        "N": N,
        "B": B,
        "q2": q2,
        "ell": ell,
        "counts_all": counts.astype(float),
        "centers_all": centers,
        "keep": keep,
    }


def _ab_from_polar(r: float, phi: float) -> tuple[float, float]:
    # Same convention as stage 09d.
    return float(r * np.cos(phi)), float(-r * np.sin(phi))


def _poisson_nll(N: np.ndarray, B: np.ndarray, eta: np.ndarray) -> float:
    eta = np.clip(eta, -ETA_CLIP, ETA_CLIP)
    lam = np.maximum(B * np.exp(eta), 1e-12)
    return float(np.sum(lam - N * np.log(lam)))


def poisson_deviance(N: np.ndarray, lam: np.ndarray) -> float:
    N = np.asarray(N, float)
    lam = np.maximum(np.asarray(lam, float), 1e-12)
    term = lam - N
    nz = N > 0
    term[nz] += N[nz] * np.log(N[nz] / lam[nz])
    return float(2.0 * np.sum(term))


def base_nll(theta, N, B, ell):
    C, r1, phi1 = theta
    a1, b1 = _ab_from_polar(r1, phi1)
    eta = C + a1 * np.cos(K1_FIXED * ell) + b1 * np.sin(K1_FIXED * ell)
    return _poisson_nll(N, B, eta)


def two_nll(theta, N, B, ell, k2):
    C, r1, phi1, r2, phi2 = theta
    a1, b1 = _ab_from_polar(r1, phi1)
    a2, b2 = _ab_from_polar(r2, phi2)
    eta = (
        C
        + a1 * np.cos(K1_FIXED * ell) + b1 * np.sin(K1_FIXED * ell)
        + a2 * np.cos(k2 * ell) + b2 * np.sin(k2 * ell)
    )
    return _poisson_nll(N, B, eta)


def fit_base(N: np.ndarray, B: np.ndarray, ell: np.ndarray) -> dict:
    c0 = float(np.clip(np.log(max(np.sum(N), 1e-12) / max(np.sum(B), 1e-12)),
                       -ETA_CLIP, ETA_CLIP))
    bounds = [(-ETA_CLIP, ETA_CLIP), (0.0, A1_MAX), (-np.pi, np.pi)]
    starts = []
    for r in (0.0, 0.5 * A1_MAX, A1_MAX):
        for ph in (0.0, 0.5 * np.pi, -0.5 * np.pi, np.pi):
            starts.append([c0, r, ph])

    best = None
    for x0 in starts:
        res = minimize(base_nll, x0=np.asarray(x0), args=(N, B, ell),
                       method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 1500, "ftol": 1e-11, "gtol": 1e-8})
        if best is None or float(res.fun) < float(best.fun):
            best = res

    C, r1, phi1 = map(float, best.x)
    a1, b1 = _ab_from_polar(r1, phi1)
    eta = C + a1 * np.cos(K1_FIXED * ell) + b1 * np.sin(K1_FIXED * ell)
    lam = np.maximum(B * np.exp(np.clip(eta, -ETA_CLIP, ETA_CLIP)), 1e-12)
    return {
        "C": C, "A1": r1, "phi1": phi1, "a1": a1, "b1": b1,
        "lambda": lam, "D": poisson_deviance(N, lam),
        "success": bool(best.success),
        "bound1": bool(abs(r1 - A1_MAX) <= 1e-5),
    }


def fit_two_at_k(N, B, ell, k2: float, base: dict,
                 warm: np.ndarray | None = None, robust: bool = False) -> dict:
    bounds = [
        (-ETA_CLIP, ETA_CLIP), (0.0, A1_MAX), (-np.pi, np.pi),
        (0.0, A2_MAX), (-np.pi, np.pi),
    ]
    starts: list[list[float] | np.ndarray] = []
    if warm is not None:
        starts.append(np.asarray(warm, float))
    starts.append([base["C"], base["A1"], base["phi1"], 0.0, 0.0])
    starts.append([base["C"], base["A1"], base["phi1"], 0.5 * A2_MAX, 0.0])
    if robust:
        for r2 in (0.5 * A2_MAX, A2_MAX):
            for ph2 in (0.0, 0.5 * np.pi, -0.5 * np.pi, np.pi):
                starts.append([base["C"], base["A1"], base["phi1"], r2, ph2])

    best = None
    for x0 in starts:
        res = minimize(two_nll, x0=np.asarray(x0), args=(N, B, ell, float(k2)),
                       method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 1200, "ftol": 1e-10, "gtol": 1e-8})
        if best is None or float(res.fun) < float(best.fun):
            best = res

    C, r1, phi1, r2, phi2 = map(float, best.x)
    a1, b1 = _ab_from_polar(r1, phi1)
    a2, b2 = _ab_from_polar(r2, phi2)
    eta = (
        C + a1 * np.cos(K1_FIXED * ell) + b1 * np.sin(K1_FIXED * ell)
        + a2 * np.cos(float(k2) * ell) + b2 * np.sin(float(k2) * ell)
    )
    lam = np.maximum(B * np.exp(np.clip(eta, -ETA_CLIP, ETA_CLIP)), 1e-12)
    return {
        "theta": np.asarray(best.x, float),
        "k2": float(k2), "C": C,
        "A1": r1, "phi1": phi1, "a1": a1, "b1": b1,
        "A2": r2, "phi2": phi2, "a2": a2, "b2": b2,
        "lambda": lam, "D": poisson_deviance(N, lam),
        "success": bool(best.success),
        "bound1": bool(abs(r1 - A1_MAX) <= 1e-5),
        "bound2": bool(abs(r2 - A2_MAX) <= 1e-5),
    }


def discover_train_model(q2_train: np.ndarray) -> tuple[dict, dict, pd.DataFrame, dict]:
    spec = make_binned_counts(q2_train)
    N, B, ell = spec["N"], spec["B"], spec["ell"]
    base = fit_base(N, B, ell)

    grid = np.linspace(K2_MIN, K2_MAX, N_K2)
    rows = []
    warm = None
    best_fast = None
    for j, k2 in enumerate(grid):
        fit = fit_two_at_k(N, B, ell, float(k2), base, warm=warm, robust=False)
        warm = fit["theta"]
        delta_d = float(base["D"] - fit["D"])
        rows.append((float(k2), delta_d, fit["A2"], fit["phi2"]))
        if best_fast is None or delta_d > best_fast[0]:
            best_fast = (delta_d, float(k2), fit)
        if (j + 1) % 100 == 0:
            print(f"[scan] {j+1}/{len(grid)} k2={k2:.3f} best_DeltaD={best_fast[0]:.3f}", flush=True)

    # Refit the winning frequency with a broader deterministic multistart.
    k_best = best_fast[1]
    two = fit_two_at_k(N, B, ell, k_best, base, warm=best_fast[2]["theta"], robust=True)
    scan = pd.DataFrame(rows, columns=["k2", "deltaD", "A2_fast", "phi2_fast"])
    return base, two, scan, spec


def normalized_shape(lam: np.ndarray) -> np.ndarray:
    lam = np.maximum(np.asarray(lam, float), 1e-300)
    return lam / np.sum(lam)


def conditional_logscore(N: np.ndarray, p: np.ndarray) -> float:
    p = np.maximum(np.asarray(p, float), 1e-300)
    return float(np.dot(np.asarray(N, float), np.log(p)))


def fixed_shape_null_pvalue(observed: float, p_null: np.ndarray, p_alt: np.ndarray,
                            n_events: int, n_null: int, rng: np.random.Generator) -> tuple[float, float, float]:
    log_ratio = np.log(np.maximum(p_alt, 1e-300)) - np.log(np.maximum(p_null, 1e-300))
    exceed = 0
    vals_sum = 0.0
    vals_sq = 0.0
    done = 0
    batch = 5000
    while done < n_null:
        m = min(batch, n_null - done)
        toys = rng.multinomial(n_events, p_null, size=m)
        stats = toys @ log_ratio
        exceed += int(np.count_nonzero(stats >= observed))
        vals_sum += float(np.sum(stats))
        vals_sq += float(np.dot(stats, stats))
        done += m
    mean = vals_sum / n_null
    var = max(vals_sq / n_null - mean * mean, 0.0)
    sd = math.sqrt(var)
    p = float((1 + exceed) / (1 + n_null))
    return p, mean, sd


def holdout_direction(train_group: str, test_group: str, q2_by_group: dict[str, np.ndarray],
                      n_null: int, rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    print("\n" + "=" * 88)
    print(f"STRICT HOLDOUT {train_group} -> {test_group}")
    print("=" * 88, flush=True)

    q2_train = q2_by_group[train_group]
    q2_test = q2_by_group[test_group]

    base, two, scan, train_spec = discover_train_model(q2_train)
    test_spec = make_binned_counts(q2_test)

    # Both groups use identical fixed bins and veto mask.
    if not np.allclose(train_spec["q2"], test_spec["q2"]):
        raise RuntimeError("Train/test retained bin centers differ")

    p_base = normalized_shape(base["lambda"])
    p_alt = normalized_shape(two["lambda"])
    y_test = test_spec["N"]

    ll_base = conditional_logscore(y_test, p_base)
    ll_alt = conditional_logscore(y_test, p_alt)
    delta_ll = float(ll_alt - ll_base)

    p_emp, null_mean, null_sd = fixed_shape_null_pvalue(
        delta_ll, p_base, p_alt, int(np.sum(y_test)), n_null, rng
    )
    z_emp = float(norm.isf(p_emp)) if 0.0 < p_emp < 1.0 else None

    train_ll_base = conditional_logscore(train_spec["N"], p_base)
    train_ll_alt = conditional_logscore(train_spec["N"], p_alt)

    result = {
        "train_group": train_group,
        "test_group": test_group,
        "train_selected_pre_veto": int(len(q2_train)),
        "test_selected_pre_veto": int(len(q2_test)),
        "train_active_events": int(np.count_nonzero(~in_veto_q2(q2_train))),
        "test_active_events": int(np.count_nonzero(~in_veto_q2(q2_test))),
        "train_hist_active_sum": int(np.sum(train_spec["N"])),
        "test_hist_active_sum": int(np.sum(test_spec["N"])),
        "k1_fixed": K1_FIXED,
        "k2_discovered_train": float(two["k2"]),
        "n2_active_coordinate": float(two["k2"] * DELTA_ELL_A / (2.0 * np.pi)),
        "train_deltaD": float(base["D"] - two["D"]),
        "train_delta_conditional_logL": float(train_ll_alt - train_ll_base),
        "A1_train": float(two["A1"]),
        "phi1_train": float(two["phi1"]),
        "A2_train": float(two["A2"]),
        "phi2_train": float(two["phi2"]),
        "A1_bound_active": bool(two["bound1"]),
        "A2_bound_active": bool(two["bound2"]),
        "test_logL_base_frozen": ll_base,
        "test_logL_alt_frozen": ll_alt,
        "test_delta_logL_frozen": delta_ll,
        "test_delta_logL_per_event": float(delta_ll / max(np.sum(y_test), 1.0)),
        "null_toys": int(n_null),
        "null_delta_logL_mean": float(null_mean),
        "null_delta_logL_sd": float(null_sd),
        "empirical_fixed_holdout_p": p_emp,
        "empirical_fixed_holdout_z_one_sided": z_emp,
        "predicts_holdout": bool(delta_ll > 0.0 and p_emp <= 0.05),
    }

    print(json.dumps(result, indent=2), flush=True)
    return result, scan


def circular_distance(a: float, b: float) -> float:
    return float(abs(np.angle(np.exp(1j * (a - b)))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null", type=int, default=DEFAULT_NULL_N)
    ap.add_argument("--step-size", default="100 MB")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    q2_by_group = {}
    provenance = []
    for g in ("00382466", "00382467"):
        q2, prov = stream_selected_q2(g, step_size=args.step_size)
        q2_by_group[g] = q2
        provenance.extend(prov)

    r_ab, scan_ab = holdout_direction("00382466", "00382467", q2_by_group, args.n_null, rng)
    r_ba, scan_ba = holdout_direction("00382467", "00382466", q2_by_group, args.n_null, rng)

    dk = abs(r_ab["k2_discovered_train"] - r_ba["k2_discovered_train"])
    dphi = circular_distance(r_ab["phi2_train"], r_ba["phi2_train"])
    both = bool(r_ab["predicts_holdout"] and r_ba["predicts_holdout"])
    one = bool(r_ab["predicts_holdout"] or r_ba["predicts_holdout"])
    if both:
        verdict = "BIDIRECTIONAL_STRICT_HOLDOUT_PASS"
    elif one:
        verdict = "MIXED_ONE_DIRECTION_ONLY"
    else:
        verdict = "STRICT_HOLDOUT_NOT_REPRODUCED"

    summary = {
        "test": "strict_cross_run_high_k_holdout",
        "status": verdict,
        "interpretation_scope": (
            "Internal run-group predictive replication only; does not establish physical origin "
            "or uniquely validate WCT."
        ),
        "selection": {
            "q2": [Q2_MIN, Q2_MAX],
            "B0_M": [B0_M_MIN, B0_M_MAX],
            "Kst_M": [KST_M_MIN, KST_M_MAX],
            "Jpsi_veto": list(JPSI_VETO),
            "psi2S_veto": list(PSI2S_VETO),
            "q2_bins": Q2_BINS,
            "KDE_bandwidth_scale": KDE_BANDWIDTH_SCALE,
            "k1_fixed": K1_FIXED,
            "k2_scan": [K2_MIN, K2_MAX, N_K2],
            "A1_max": A1_MAX,
            "A2_max": A2_MAX,
            "delta_ell_active": DELTA_ELL_A,
        },
        "directions": [r_ab, r_ba],
        "cross_discovery_compatibility": {
            "abs_delta_k2": float(dk),
            "circular_delta_phi2": float(dphi),
        },
        "provenance": provenance,
    }

    (out / "holdout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([r_ab, r_ba]).to_csv(out / "holdout_results.csv", index=False)
    pd.DataFrame(provenance).to_csv(out / "event_provenance.csv", index=False)
    scan_ab.to_csv(out / "scan_00382466_train.csv", index=False)
    scan_ba.to_csv(out / "scan_00382467_train.csv", index=False)

    print("\n" + "=" * 88)
    print("FINAL STRICT HOLDOUT VERDICT")
    print("=" * 88)
    print(verdict)
    print(f"abs delta k2 = {dk:.6g}")
    print(f"circular delta phi2 = {dphi:.6g} rad")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()

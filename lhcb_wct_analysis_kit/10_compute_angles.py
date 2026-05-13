"""
10_compute_angles_cupy.py

GPU-optional CuPy update for computing angular variables for:
    B0 -> K*0(-> K+ pi-) mu+ mu-

Outputs:
    outputs_angles/angles.parquet
    outputs_angles/angles.csv
    outputs_angles/angle_summary.json
    outputs_angles/angle_distributions.png

Computed:
    q2, Kst_mass, cosThetaL, cosThetaK, phi

Convention:
    cosThetaL = angle between mu+ and opposite B direction in dimuon rest frame
    cosThetaK = angle between K+ and opposite B direction in K* rest frame
    phi       = signed angle between Kπ and μμ decay planes in B rest frame

Important boost convention:
    E' = gamma(E - beta · p)
    boost into parent rest frame uses beta = parent_vec / parent_E, not -beta.
"""

from __future__ import annotations

import json
import glob
import importlib.metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# CuPy import with safe fallback
# =============================================================================

GPU_AVAILABLE = False
cp = None

try:
    # Workaround for environments with broken package metadata entries.
    _orig_distributions = importlib_metadata.distributions

    def _safe_distributions(*args, **kwargs):
        dists = []
        for d in _orig_distributions(*args, **kwargs):
            try:
                if getattr(d, "metadata", None) is not None:
                    dists.append(d)
            except Exception:
                pass
        return dists

    importlib_metadata.distributions = _safe_distributions

    import cupy as cp  # type: ignore
    GPU_AVAILABLE = True
except Exception:
    cp = None
    GPU_AVAILABLE = False

USE_CUPY = True


def xp_module():
    return cp if (USE_CUPY and GPU_AVAILABLE) else np


def to_numpy(x):
    if GPU_AVAILABLE and cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


# =============================================================================
# Config
# =============================================================================

DATA_GLOB = "data/*.root"
TREE_NAME = "B0_KstMuMu/DecayTree"

OUT_DIR = Path("outputs_angles")
OUT_DIR.mkdir(exist_ok=True, parents=True)

Q2_MIN = 0.1
Q2_MAX = 19.0

B0_M_MIN = 5230.0
B0_M_MAX = 5330.0

KST_M_MIN = 795.9
KST_M_MAX = 995.9

JPSI_VETO = (8.68, 10.09)
PSI2S_VETO = (12.86, 14.18)

SAVE_CSV = True
SAVE_PARQUET = True

# Rest checks require small GPU->CPU reductions. Keep True for validation.
DO_REST_CHECKS = True


# =============================================================================
# Four-vector tools, GPU-aware
# =============================================================================

def make_p4(df: pd.DataFrame, prefix: str):
    xp = xp_module()
    required = [
        f"{prefix}_PE",
        f"{prefix}_PX",
        f"{prefix}_PY",
        f"{prefix}_PZ",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing four-vector columns for {prefix}: {missing}")

    arr = np.column_stack([
        df[f"{prefix}_PE"].to_numpy(dtype=np.float64),
        df[f"{prefix}_PX"].to_numpy(dtype=np.float64),
        df[f"{prefix}_PY"].to_numpy(dtype=np.float64),
        df[f"{prefix}_PZ"].to_numpy(dtype=np.float64),
    ])
    return xp.asarray(arr, dtype=xp.float64)


def p3(p4):
    return p4[:, 1:4]


def mass2(p4):
    xp = xp_module()
    return p4[:, 0] ** 2 - xp.sum(p4[:, 1:4] ** 2, axis=1)


def mass(p4):
    xp = xp_module()
    return xp.sqrt(xp.maximum(mass2(p4), 0.0))


def norm3(v):
    xp = xp_module()
    return xp.sqrt(xp.sum(v * v, axis=1))


def unit3(v, eps: float = 1e-30):
    xp = xp_module()
    n = norm3(v)
    return v / xp.maximum(n[:, None], eps)


def dot3(a, b):
    xp = xp_module()
    return xp.sum(a * b, axis=1)


def cross3(a, b):
    xp = xp_module()
    return xp.cross(a, b)


def clip_cos(x):
    xp = xp_module()
    return xp.clip(x, -1.0, 1.0)


def boost(p4, beta):
    """
    Lorentz boost four-vectors by velocity beta.

    p4 order: [E, px, py, pz]

    Convention:
        E' = gamma(E - beta · p)
        p' = p + [((gamma - 1)(beta·p)/beta²) - gamma E] beta
    """
    xp = xp_module()
    E = p4[:, 0]
    p = p4[:, 1:4]

    b2 = xp.sum(beta * beta, axis=1)
    b2 = xp.minimum(b2, 1.0 - 1e-15)

    gamma = 1.0 / xp.sqrt(1.0 - b2)
    bp = xp.sum(beta * p, axis=1)

    nz = b2 > 1e-30
    factor = xp.zeros_like(E)
    factor = xp.where(
        nz,
        ((gamma - 1.0) * bp / xp.maximum(b2, 1e-30)) - gamma * E,
        0.0,
    )

    p_prime = p + factor[:, None] * beta
    E_prime = gamma * (E - bp)

    return xp.column_stack([E_prime, p_prime])


def boost_to_rest(p4, parent):
    beta = p3(parent) / xp_module().maximum(parent[:, 0:1], 1e-30)
    return boost(p4, beta)


def check_rest_boost(parent, name: str) -> None:
    boosted = boost_to_rest(parent, parent)
    rest_p = to_numpy(norm3(boosted[:, 1:4]))
    rest_e = to_numpy(boosted[:, 0])
    print(f"[check] {name:7s} rest |p| median = {np.median(rest_p):.6e}")
    print(f"[check] {name:7s} rest |p| p95    = {np.percentile(rest_p, 95):.6e}")
    print(f"[check] {name:7s} rest E median   = {np.median(rest_e):.6e}")


# =============================================================================
# Data loading
# =============================================================================

def choose_tree(file_handle, preferred: str) -> str:
    if preferred in file_handle:
        return preferred
    for key in file_handle.keys(recursive=True):
        if key.split(";")[0] == preferred:
            return key
    for key in file_handle.keys(recursive=True):
        if "DecayTree" in key:
            return key
    raise RuntimeError("No DecayTree found")


def load_dataframe() -> pd.DataFrame:
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError(f"No ROOT files match {DATA_GLOB}")

    required = [
        "B0_M",
        "B0_PX", "B0_PY", "B0_PZ", "B0_PE",
        "Kplus_PX", "Kplus_PY", "Kplus_PZ", "Kplus_PE",
        "piminus_PX", "piminus_PY", "piminus_PZ", "piminus_PE",
        "muplus_PX", "muplus_PY", "muplus_PZ", "muplus_PE",
        "muminus_PX", "muminus_PY", "muminus_PZ", "muminus_PE",
    ]

    optional = [
        "Kst_892_0_M",
        "Kst_M",
        "runNumber",
        "eventNumber",
        "Polarity",
        "nCandidate",
        "totCandidates",
    ]

    chunks = []
    missing_by_file = {}

    for path in files:
        with uproot.open(path) as f:
            tree_name = choose_tree(f, TREE_NAME)
            tree = f[tree_name]
            keys = set(tree.keys())

            missing = [b for b in required if b not in keys]
            if missing:
                missing_by_file[path] = missing
                print(f"[warn] skipping {path}, missing required: {missing}")
                continue

            branches = required + [b for b in optional if b in keys]

            print(f"[load] {path}")
            print(f"[load] tree = {tree_name}")
            print(f"[load] branches = {len(branches)}")

            arr = tree.arrays(branches, library="pd")
            arr["source_file"] = Path(path).name
            chunks.append(arr)

    if not chunks:
        raise RuntimeError(f"No usable ROOT files. Missing by file: {missing_by_file}")

    return pd.concat(chunks, ignore_index=True)


# =============================================================================
# Angle computation
# =============================================================================

def compute_angles(df: pd.DataFrame) -> pd.DataFrame:
    xp = xp_module()

    K = make_p4(df, "Kplus")
    pi = make_p4(df, "piminus")
    mup = make_p4(df, "muplus")
    mum = make_p4(df, "muminus")

    # Use reconstructed B from final-state four-vectors.
    B = K + pi + mup + mum
    q = mup + mum
    Kst = K + pi

    if DO_REST_CHECKS:
        check_rest_boost(q, "dimuon")
        check_rest_boost(Kst, "Kstar")
        check_rest_boost(B, "B_reco")

    q2 = mass2(q) / 1e6

    if "Kst_892_0_M" in df.columns:
        Kst_mass = xp.asarray(df["Kst_892_0_M"].to_numpy(dtype=np.float64))
    elif "Kst_M" in df.columns:
        Kst_mass = xp.asarray(df["Kst_M"].to_numpy(dtype=np.float64))
    else:
        Kst_mass = mass(Kst)

    # cosThetaL: mu+ vs opposite B direction in dimuon rest frame.
    mup_qrf = boost_to_rest(mup, q)
    B_qrf = boost_to_rest(B, q)
    v_mu = p3(mup_qrf)
    v_B_opp = -p3(B_qrf)
    cosThetaL = clip_cos(
        dot3(v_mu, v_B_opp) / xp.maximum(norm3(v_mu) * norm3(v_B_opp), 1e-30)
    )

    # cosThetaK: K+ vs opposite B direction in K* rest frame.
    K_krf = boost_to_rest(K, Kst)
    B_krf = boost_to_rest(B, Kst)
    v_K = p3(K_krf)
    v_B_opp_k = -p3(B_krf)
    cosThetaK = clip_cos(
        dot3(v_K, v_B_opp_k) / xp.maximum(norm3(v_K) * norm3(v_B_opp_k), 1e-30)
    )

    # phi: signed angle between Kpi and mumu decay planes in B rest frame.
    K_brf = boost_to_rest(K, B)
    pi_brf = boost_to_rest(pi, B)
    mup_brf = boost_to_rest(mup, B)
    mum_brf = boost_to_rest(mum, B)
    q_brf = boost_to_rest(q, B)

    k_vec = p3(K_brf)
    pi_vec = p3(pi_brf)
    mup_vec = p3(mup_brf)
    mum_vec = p3(mum_brf)
    q_vec = p3(q_brf)

    n_K = unit3(cross3(k_vec, pi_vec))
    n_L = unit3(cross3(mup_vec, mum_vec))
    q_hat = unit3(q_vec)

    cos_phi = clip_cos(dot3(n_K, n_L))
    sin_phi = dot3(q_hat, cross3(n_K, n_L))
    phi = xp.arctan2(sin_phi, cos_phi)

    q2_np = to_numpy(q2)
    kst_np = to_numpy(Kst_mass)
    cosL_np = to_numpy(cosThetaL)
    cosK_np = to_numpy(cosThetaK)
    phi_np = to_numpy(phi)

    out = pd.DataFrame({
        "q2": q2_np,
        "Kst_mass": kst_np,
        "B0_M": df["B0_M"].to_numpy(dtype=np.float64),
        "cosThetaL": cosL_np,
        "cosThetaK": cosK_np,
        "phi": phi_np,
        "source_file": df["source_file"].astype(str).to_numpy(),
    })

    for col in ["runNumber", "eventNumber", "Polarity", "nCandidate", "totCandidates"]:
        if col in df.columns:
            out[col] = df[col].to_numpy()

    out["finite_angles"] = (
        np.isfinite(out["q2"])
        & np.isfinite(out["Kst_mass"])
        & np.isfinite(out["B0_M"])
        & np.isfinite(out["cosThetaL"])
        & np.isfinite(out["cosThetaK"])
        & np.isfinite(out["phi"])
    )

    out["in_q2_range"] = (out["q2"] >= Q2_MIN) & (out["q2"] <= Q2_MAX)
    out["in_B0_window"] = (out["B0_M"] >= B0_M_MIN) & (out["B0_M"] <= B0_M_MAX)
    out["in_Kst_window"] = (out["Kst_mass"] >= KST_M_MIN) & (out["Kst_mass"] <= KST_M_MAX)

    jpsi = (out["q2"] >= JPSI_VETO[0]) & (out["q2"] <= JPSI_VETO[1])
    psi2s = (out["q2"] >= PSI2S_VETO[0]) & (out["q2"] <= PSI2S_VETO[1])
    out["in_charmonium_veto"] = jpsi | psi2s

    out["passes_signal_selection"] = (
        out["finite_angles"]
        & out["in_q2_range"]
        & out["in_B0_window"]
        & out["in_Kst_window"]
        & (~out["in_charmonium_veto"])
    )

    return out


# =============================================================================
# Diagnostics
# =============================================================================

def summarize_angles(out: pd.DataFrame) -> dict:
    sel = out["passes_signal_selection"]

    def stats(col: str, mask=None):
        x = out[col].to_numpy(dtype=float) if mask is None else out.loc[mask, col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return None
        return {
            "n": int(len(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            "min": float(np.min(x)),
            "p01": float(np.percentile(x, 1)),
            "p05": float(np.percentile(x, 5)),
            "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)),
            "max": float(np.max(x)),
        }

    sig_cosL = stats("cosThetaL", sel)
    sig_cosK = stats("cosThetaK", sel)
    sig_phi = stats("phi", sel)

    angle_health = {
        "cosThetaL_has_spread": bool(sig_cosL and sig_cosL["std"] > 0.05),
        "cosThetaK_has_spread": bool(sig_cosK and sig_cosK["std"] > 0.05),
        "phi_has_spread": bool(sig_phi and sig_phi["std"] > 0.5),
        "cosThetaL_range_ok": bool(sig_cosL and sig_cosL["min"] >= -1.000001 and sig_cosL["max"] <= 1.000001),
        "cosThetaK_range_ok": bool(sig_cosK and sig_cosK["min"] >= -1.000001 and sig_cosK["max"] <= 1.000001),
        "phi_range_ok": bool(sig_phi and sig_phi["min"] >= -np.pi - 1e-6 and sig_phi["max"] <= np.pi + 1e-6),
    }
    angle_health["angles_usable"] = bool(
        angle_health["cosThetaL_has_spread"]
        and angle_health["cosThetaK_has_spread"]
        and angle_health["phi_has_spread"]
        and angle_health["cosThetaL_range_ok"]
        and angle_health["cosThetaK_range_ok"]
        and angle_health["phi_range_ok"]
    )

    return {
        "n_total": int(len(out)),
        "n_finite_angles": int(out["finite_angles"].sum()),
        "n_in_q2_range": int(out["in_q2_range"].sum()),
        "n_in_B0_window": int(out["in_B0_window"].sum()),
        "n_in_Kst_window": int(out["in_Kst_window"].sum()),
        "n_in_charmonium_veto": int(out["in_charmonium_veto"].sum()),
        "n_passes_signal_selection": int(sel.sum()),
        "gpu": {
            "cupy_available": bool(GPU_AVAILABLE),
            "use_cupy": bool(USE_CUPY),
            "using_gpu": bool(USE_CUPY and GPU_AVAILABLE),
        },
        "angle_health": angle_health,
        "angle_stats_all": {
            "q2": stats("q2"),
            "Kst_mass": stats("Kst_mass"),
            "B0_M": stats("B0_M"),
            "cosThetaL": stats("cosThetaL"),
            "cosThetaK": stats("cosThetaK"),
            "phi": stats("phi"),
        },
        "angle_stats_signal_selection": {
            "q2": stats("q2", sel),
            "Kst_mass": stats("Kst_mass", sel),
            "B0_M": stats("B0_M", sel),
            "cosThetaL": stats("cosThetaL", sel),
            "cosThetaK": stats("cosThetaK", sel),
            "phi": stats("phi", sel),
        },
        "convention": {
            "cosThetaL": "mu+ vs -B direction in dimuon rest frame",
            "cosThetaK": "K+ vs -B direction in K* rest frame",
            "phi": "signed angle between Kpi and mumu planes in B rest frame",
        },
        "warning": "For LHCb-publication P5' comparison, validate CP/flavor angle convention.",
    }


def plot_distributions(out: pd.DataFrame) -> None:
    sel = out["passes_signal_selection"]

    fig, ax = plt.subplots(2, 3, figsize=(13, 7))

    ax[0, 0].hist(out.loc[sel, "q2"], bins=60)
    ax[0, 0].set_title("q² selected")
    ax[0, 0].set_xlabel("q² [GeV²]")

    ax[0, 1].hist(out.loc[sel, "Kst_mass"], bins=60)
    ax[0, 1].set_title("K* mass selected")
    ax[0, 1].set_xlabel("MeV")

    ax[0, 2].hist(out.loc[sel, "B0_M"], bins=60)
    ax[0, 2].set_title("B0 mass selected")
    ax[0, 2].set_xlabel("MeV")

    ax[1, 0].hist(out.loc[sel, "cosThetaL"], bins=60, range=(-1, 1))
    ax[1, 0].set_title("cosThetaL")
    ax[1, 0].set_xlabel("cosThetaL")

    ax[1, 1].hist(out.loc[sel, "cosThetaK"], bins=60, range=(-1, 1))
    ax[1, 1].set_title("cosThetaK")
    ax[1, 1].set_xlabel("cosThetaK")

    ax[1, 2].hist(out.loc[sel, "phi"], bins=60, range=(-np.pi, np.pi))
    ax[1, 2].set_title("phi")
    ax[1, 2].set_xlabel("rad")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "angle_distributions.png", dpi=160)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print(f"[gpu] CuPy available: {GPU_AVAILABLE}")
    print(f"[gpu] USE_CUPY: {USE_CUPY}")
    print(f"[gpu] using GPU: {USE_CUPY and GPU_AVAILABLE}")

    print("[load] ROOT files")
    df = load_dataframe()
    print(f"[load] rows: {len(df):,}")

    print("[compute] angles")
    out = compute_angles(df)

    print("[save] outputs")

    if SAVE_PARQUET:
        try:
            out.to_parquet(OUT_DIR / "angles.parquet", index=False)
            print(f"[save] {OUT_DIR / 'angles.parquet'}")
        except Exception as e:
            print(f"[warn] parquet save failed: {e}")

    if SAVE_CSV:
        out.to_csv(OUT_DIR / "angles.csv", index=False)
        print(f"[save] {OUT_DIR / 'angles.csv'}")

    summary = summarize_angles(out)
    with open(OUT_DIR / "angle_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {OUT_DIR / 'angle_summary.json'}")

    plot_distributions(out)
    print(f"[save] {OUT_DIR / 'angle_distributions.png'}")

    print("\n" + "=" * 80)
    print("ANGLE COMPUTATION SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved folder: {OUT_DIR}")

    if not summary["angle_health"]["angles_usable"]:
        print("\nWARNING: angles are not usable yet. Check rest-frame boost diagnostics and angle spreads.")
    else:
        print("\nOK: derived angles appear usable for angular log-cos / P5-proxy tests.")


if __name__ == "__main__":
    main()

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq


def find_root_files(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No ROOT files found for pattern: {pattern}")
    return files


def find_ttrees(root_file):
    trees = []
    with uproot.open(root_file) as f:
        for key, obj in f.items(recursive=True):
            if isinstance(obj, uproot.behaviors.TTree.TTree):
                trees.append(key)
    return trees


def choose_tree(root_file, preferred=None):
    trees = find_ttrees(root_file)
    if not trees:
        raise RuntimeError(f"No TTrees found in {root_file}")
    if preferred and preferred in trees:
        return preferred
    for t in trees:
        if "DecayTree" in t:
            return t
    return trees[0]


def branch_candidates(branches, terms):
    terms = [t.lower() for t in terms]
    out = []
    for b in branches:
        bl = b.lower()
        if all(t in bl for t in terms):
            out.append(b)
    return out


def resolve_branch(branches, preferred, required=False):
    if preferred in branches:
        return preferred
    low_map = {b.lower(): b for b in branches}
    if preferred.lower() in low_map:
        return low_map[preferred.lower()]
    hits = [b for b in branches if preferred.lower() in b.lower()]
    if hits:
        return hits[0]
    if required:
        raise KeyError(f"Missing required branch: {preferred}")
    return None


def load_dataframe(files_pattern, tree_name, requested_branches):
    files = find_root_files(files_pattern)
    dfs = []
    used_tree = None
    missing_by_file = {}

    for path in files:
        with uproot.open(path) as f:
            auto_tree = tree_name if tree_name in f else choose_tree(path, preferred=tree_name)
            used_tree = auto_tree
            tree = f[auto_tree]
            available = list(tree.keys())

            use = []
            missing = []
            rename = {}
            for b in requested_branches:
                rb = resolve_branch(available, b, required=False)
                if rb:
                    use.append(rb)
                    if rb != b:
                        rename[rb] = b
                else:
                    missing.append(b)

            if not use:
                raise RuntimeError(f"No requested branches found in {path}")

            arr = tree.arrays(use, library="pd")
            if rename:
                arr = arr.rename(columns=rename)
            arr["source_file"] = str(path)
            dfs.append(arr)
            missing_by_file[str(path)] = missing

    return pd.concat(dfs, ignore_index=True), used_tree, missing_by_file


def inv_mass2(px, py, pz, e):
    return e**2 - px**2 - py**2 - pz**2


def add_q2(df, muplus_prefix="muplus", muminus_prefix="muminus"):
    needed = [
        f"{muplus_prefix}_PX", f"{muplus_prefix}_PY", f"{muplus_prefix}_PZ", f"{muplus_prefix}_PE",
        f"{muminus_prefix}_PX", f"{muminus_prefix}_PY", f"{muminus_prefix}_PZ", f"{muminus_prefix}_PE",
    ]
    missing = [x for x in needed if x not in df.columns]
    if missing:
        raise KeyError(f"Cannot compute q2. Missing: {missing}")

    q_px = df[f"{muplus_prefix}_PX"] + df[f"{muminus_prefix}_PX"]
    q_py = df[f"{muplus_prefix}_PY"] + df[f"{muminus_prefix}_PY"]
    q_pz = df[f"{muplus_prefix}_PZ"] + df[f"{muminus_prefix}_PZ"]
    q_e = df[f"{muplus_prefix}_PE"] + df[f"{muminus_prefix}_PE"]

    out = df.copy()
    out["q2"] = inv_mass2(q_px, q_py, q_pz, q_e) / 1e6
    return out


def basic_selection(df, q2_min=0.1, q2_max=19.0, b0_m_min=5100, b0_m_max=5600,
                    kst_m_min=750, kst_m_max=1100, jpsi_veto=(8.0, 11.0),
                    psi2s_veto=(12.5, 15.0)):
    out = df.copy()
    out = out[np.isfinite(out["q2"])]
    out = out[(out["q2"] > q2_min) & (out["q2"] < q2_max)]

    if "B0_M" in out.columns:
        out = out[(out["B0_M"] > b0_m_min) & (out["B0_M"] < b0_m_max)]

    if "Kst_M" in out.columns:
        out = out[(out["Kst_M"] > kst_m_min) & (out["Kst_M"] < kst_m_max)]

    if jpsi_veto:
        lo, hi = jpsi_veto
        out = out[~((out["q2"] > lo) & (out["q2"] < hi))]

    if psi2s_veto:
        lo, hi = psi2s_veto
        out = out[~((out["q2"] > lo) & (out["q2"] < hi))]

    return out


def make_q2_spectrum(df, bins=60, q2_min=0.1, q2_max=19.0, weight_col=None):
    weights = df[weight_col].to_numpy() if weight_col and weight_col in df.columns else None
    counts, edges = np.histogram(df["q2"].to_numpy(), bins=bins, range=(q2_min, q2_max), weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if weights is None:
        err = np.sqrt(np.maximum(counts, 1.0))
    else:
        sumw2, _ = np.histogram(df["q2"].to_numpy(), bins=bins, range=(q2_min, q2_max), weights=weights**2)
        err = np.sqrt(np.maximum(sumw2, 1e-12))

    return centers, counts.astype(float), err, edges


def smooth_residual(x, y, window=11, poly=3):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < poly + 3:
        baseline = np.maximum(np.mean(y), 1e-9) * np.ones_like(y)
        return baseline, y / baseline - 1.0

    window = min(window, n if n % 2 == 1 else n - 1)
    min_win = poly + 2
    if min_win % 2 == 0:
        min_win += 1
    window = max(window, min_win)
    if window > n:
        window = n if n % 2 == 1 else n - 1

    baseline = savgol_filter(y, window_length=window, polyorder=poly)
    baseline = np.maximum(baseline, 1e-9)
    return baseline, y / baseline - 1.0


def log_fft_scan(q2, residual, n_grid=512):
    q2 = np.asarray(q2, dtype=float)
    residual = np.asarray(residual, dtype=float)
    mask = (q2 > 0) & np.isfinite(residual)
    q2 = q2[mask]
    residual = residual[mask]

    ell = np.log(q2)
    order = np.argsort(ell)
    ell = ell[order]
    residual = residual[order]

    ell_grid = np.linspace(ell.min(), ell.max(), n_grid)
    r_grid = interp1d(ell, residual, kind="linear", fill_value="extrapolate")(ell_grid)

    r_grid = r_grid - np.mean(r_grid)
    rw = r_grid * np.hanning(len(r_grid))

    d_ell = ell_grid[1] - ell_grid[0]
    amp = np.abs(rfft(rw))
    k = 2 * np.pi * rfftfreq(n_grid, d=d_ell)
    return ell_grid, r_grid, k, amp


def peak_report(k, amp, kmin=2.0, kmax=30.0):
    mask = (k >= kmin) & (k <= kmax)
    kk = np.asarray(k)[mask]
    aa = np.asarray(amp)[mask]
    if len(kk) == 0:
        return None
    med = float(np.median(aa))
    idx = int(np.argmax(aa))
    return {
        "k_peak": float(kk[idx]),
        "amp_peak": float(aa[idx]),
        "median_amp": med,
        "snr_like": float(aa[idx] / med) if med > 0 else float("inf"),
    }


def save_json(obj, path):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

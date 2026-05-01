import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq
from scipy.stats import gaussian_kde


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
            # Newer uproot can return an awkward Record array even with library='pd'
            # in some configurations. Coerce to DataFrame.
            if not isinstance(arr, pd.DataFrame):
                import awkward as ak
                arr = ak.to_dataframe(arr)
                # Strip multi-index if present (single-jagged scalar branches)
                if isinstance(arr.index, pd.MultiIndex):
                    arr = arr.reset_index(drop=True)
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


def add_kst_mass(df, kplus_prefix="Kplus", piminus_prefix="piminus"):
    """
    Compute K*(892) candidate mass from K+ pi- four-vectors when Kst_M
    is not present as a branch. Writes Kst_M in MeV/c^2 to match LHCb
    convention used by basic_selection (which expects a 750-1100 window).
    """
    needed = [
        f"{kplus_prefix}_PX", f"{kplus_prefix}_PY", f"{kplus_prefix}_PZ", f"{kplus_prefix}_PE",
        f"{piminus_prefix}_PX", f"{piminus_prefix}_PY", f"{piminus_prefix}_PZ", f"{piminus_prefix}_PE",
    ]
    missing = [x for x in needed if x not in df.columns]
    if missing:
        # Cannot compute. Leave df untouched so basic_selection can skip the cut.
        return df

    px = df[f"{kplus_prefix}_PX"] + df[f"{piminus_prefix}_PX"]
    py = df[f"{kplus_prefix}_PY"] + df[f"{piminus_prefix}_PY"]
    pz = df[f"{kplus_prefix}_PZ"] + df[f"{piminus_prefix}_PZ"]
    e  = df[f"{kplus_prefix}_PE"] + df[f"{piminus_prefix}_PE"]

    m2 = inv_mass2(px, py, pz, e)
    out = df.copy()
    # Avoid sqrt of small-negative due to floating point
    out["Kst_M"] = np.sqrt(np.maximum(m2, 0.0))
    out["Kst_M_computed"] = True
    return out


def basic_selection(df, q2_min=0.1, q2_max=19.0, b0_m_min=5100, b0_m_max=5600,
                    kst_m_min=792.0, kst_m_max=992.0, jpsi_veto=(8.0, 11.0),
                    psi2s_veto=(12.5, 15.0), require_kst=True, verbose=True):
    """
    Apply selection cuts and (optionally) print a per-cut event count.

    require_kst=True (default): raises ValueError if Kst_M column is absent.
    Kst_M can be computed from K+ pi- four-vectors via add_kst_mass() before
    calling this function. The K*(892) cut defines the channel; skipping it
    silently means the sample is not B0 -> K*0 mu+ mu-.

    Returns the selected DataFrame. Cutflow is also attached as
    out.attrs['cutflow'] = list of (name, n_remaining) tuples.
    """
    cutflow = []

    def _step(name, frame):
        n = int(len(frame))
        cutflow.append((name, n))
        if verbose:
            print(f"  cut {name:20s}: {n:8d}")
        return frame

    if verbose:
        print("--- cutflow ---")

    out = df.copy()
    out = _step("raw", out)

    out = out[np.isfinite(out["q2"])]
    out = _step("finite_q2", out)

    out = out[(out["q2"] > q2_min) & (out["q2"] < q2_max)]
    out = _step("q2_range", out)

    if "B0_M" in out.columns:
        out = out[(out["B0_M"] > b0_m_min) & (out["B0_M"] < b0_m_max)]
        out = _step("B0_M", out)
    else:
        if verbose:
            print("  cut B0_M               : SKIPPED (branch missing)")

    if "Kst_M" in out.columns:
        out = out[(out["Kst_M"] > kst_m_min) & (out["Kst_M"] < kst_m_max)]
        out = _step("Kst_M", out)
    else:
        if require_kst:
            raise ValueError(
                "Kst_M missing. Compute it from Kplus+piminus four-vectors "
                "via add_kst_mass(df) before calling basic_selection. "
                "Without the K*(892) mass cut the channel is not B0 -> K*0 mu+ mu-. "
                "Pass require_kst=False to bypass for diagnostic only."
            )
        if verbose:
            print("  cut Kst_M              : SKIPPED (branch missing, require_kst=False)")

    if jpsi_veto:
        lo, hi = jpsi_veto
        out = out[~((out["q2"] > lo) & (out["q2"] < hi))]
        out = _step("J/psi_veto", out)

    if psi2s_veto:
        lo, hi = psi2s_veto
        out = out[~((out["q2"] > lo) & (out["q2"] < hi))]
        out = _step("psi2S_veto", out)

    out = _step("final", out)
    out.attrs["cutflow"] = cutflow
    return out


def select_with_mode(df, mode, q2_min=0.1, q2_max=19.0,
                     b0_m_min=5100, b0_m_max=5600,
                     kst_m_min=792.0, kst_m_max=992.0,
                     jpsi_veto=(8.0, 11.0), psi2s_veto=(12.5, 15.0),
                     verbose=False):
    """
    Selection wrapper for the cut sweep. Modes:
        "raw_q2" : q^2 range only (no B0 mass, no K*, no veto)
        "loose"  : q^2 range + B0 mass
        "medium" : q^2 range + B0 mass + K*(892) mass
        "tight"  : medium + charmonium vetoes

    require_kst is implicit in the mode (medium/tight require Kst_M to be present).
    """
    if mode == "raw_q2":
        return basic_selection(
            df, q2_min=q2_min, q2_max=q2_max,
            b0_m_min=-1e9, b0_m_max=1e9,
            kst_m_min=-1e9, kst_m_max=1e9,
            jpsi_veto=None, psi2s_veto=None,
            require_kst=False, verbose=verbose,
        )
    if mode == "loose":
        return basic_selection(
            df, q2_min=q2_min, q2_max=q2_max,
            b0_m_min=b0_m_min, b0_m_max=b0_m_max,
            kst_m_min=-1e9, kst_m_max=1e9,
            jpsi_veto=None, psi2s_veto=None,
            require_kst=False, verbose=verbose,
        )
    if mode == "medium":
        return basic_selection(
            df, q2_min=q2_min, q2_max=q2_max,
            b0_m_min=b0_m_min, b0_m_max=b0_m_max,
            kst_m_min=kst_m_min, kst_m_max=kst_m_max,
            jpsi_veto=None, psi2s_veto=None,
            require_kst=True, verbose=verbose,
        )
    if mode == "tight":
        return basic_selection(
            df, q2_min=q2_min, q2_max=q2_max,
            b0_m_min=b0_m_min, b0_m_max=b0_m_max,
            kst_m_min=kst_m_min, kst_m_max=kst_m_max,
            jpsi_veto=jpsi_veto, psi2s_veto=psi2s_veto,
            require_kst=True, verbose=verbose,
        )
    raise ValueError(f"Unknown selection mode: {mode}. "
                     "Use raw_q2 | loose | medium | tight.")


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


# =============================================================================
# LFT (log-Fourier transform) primitives
# Bin uniformly in ell = ln(q^2 / Q2_REF) and FFT the residual directly.
# This is the correct test for log-periodic structure R = A cos(k_ell ell + phi).
# =============================================================================

def make_lft_spectrum(df, q2_ref=1.0, q2_min=0.1, q2_max=19.0, n_bins=60):
    """
    Bin events uniformly in ell = ln(q^2/Q2_REF). Returns (ell_centers, counts,
    poisson_err, ell_edges, q2_ref). counts are floats, err = sqrt(N) (Poisson).
    """
    q2 = df["q2"].to_numpy()
    q2 = q2[(q2 > 0) & np.isfinite(q2)]
    ell = np.log(q2 / q2_ref)

    ell_min = np.log(q2_min / q2_ref)
    ell_max = np.log(q2_max / q2_ref)
    ell_edges = np.linspace(ell_min, ell_max, n_bins + 1)
    ell_centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])

    counts, _ = np.histogram(ell, bins=ell_edges)
    counts = counts.astype(float)
    err = np.sqrt(np.maximum(counts, 1.0))
    return ell_centers, counts, err, ell_edges, q2_ref


def lft_baseline(ell_centers, counts, ell_edges, ell_events=None,
                 mode="kde", floor=0.5, savgol_window=11, savgol_poly=3,
                 kde_bw=None):
    """
    Build a baseline N_smooth(ell) via one of three modes:
      "savgol" : Savitzky-Golay over counts (legacy).
      "floor"  : np.maximum(savgol(counts), floor) - Poisson-safe.
      "kde"    : Gaussian KDE on ell_events, scaled to expected counts per bin.

    "kde" requires ell_events (the per-event ell values). "savgol" and "floor"
    operate on binned counts only.
    """
    counts = np.asarray(counts, dtype=float)
    n = len(counts)

    if mode == "kde":
        if ell_events is None or len(ell_events) < 5:
            # fall back gracefully
            mode = "floor"

    if mode == "savgol":
        win = min(savgol_window, n if n % 2 == 1 else n - 1)
        min_win = savgol_poly + 2
        if min_win % 2 == 0:
            min_win += 1
        win = max(win, min_win)
        baseline = savgol_filter(counts, window_length=win, polyorder=savgol_poly)
        baseline = np.maximum(baseline, 1e-9)
        return baseline

    if mode == "floor":
        win = min(savgol_window, n if n % 2 == 1 else n - 1)
        min_win = savgol_poly + 2
        if min_win % 2 == 0:
            min_win += 1
        win = max(win, min_win)
        smooth = savgol_filter(counts, window_length=win, polyorder=savgol_poly)
        return np.maximum(smooth, float(floor))

    if mode == "kde":
        kde = gaussian_kde(ell_events, bw_method=kde_bw)
        density = kde(ell_centers)
        bin_widths = np.diff(ell_edges)
        baseline = density * len(ell_events) * bin_widths
        return np.maximum(baseline, 1e-9)

    raise ValueError(f"Unknown baseline mode: {mode}")


def lft_residual(counts, baseline):
    """
    Poisson-normalized residual:
        R_i = (N_i - B_i) / sqrt(B_i + 1)
    Bounded for empty bins (does not saturate at -1 like the ratio form).
    """
    counts = np.asarray(counts, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    return (counts - baseline) / np.sqrt(baseline + 1.0)


def lft_fft(ell_centers, residual, window="hanning", subtract_mean=True):
    """
    FFT the residual directly on the uniform ell grid. No interpolation,
    no zero-padding beyond the natural rfft length. Returns (k_ell, amp).

    The Hanning window suppresses leakage from the ell-boundary
    discontinuity (the residual is not periodic in ell).
    """
    r = np.asarray(residual, dtype=float)
    if subtract_mean:
        r = r - np.mean(r)

    n = len(r)
    if window == "hanning":
        w = np.hanning(n)
    elif window in (None, "none", "rect"):
        w = np.ones(n)
    else:
        raise ValueError(f"Unknown window: {window}")

    rw = r * w
    d_ell = float(ell_centers[1] - ell_centers[0])
    amp = np.abs(rfft(rw))
    freq_cycles = rfftfreq(n, d=d_ell)
    k_ell = 2.0 * np.pi * freq_cycles
    return k_ell, amp


# =============================================================================
# FFT (linear-q^2) primitives - artifact / control test parallel to LFT.
# Bin uniformly in q^2 and FFT the residual. If a peak appears at the same
# location in both LFT (k_ell) and FFT (k_q2) spaces, the structure is more
# likely a bin/cut artifact than log-periodic physics.
# =============================================================================

def make_q2_linear_spectrum(df, q2_min=0.1, q2_max=19.0, n_bins=60):
    """
    Bin events uniformly in linear q^2. Returns (q2_centers, counts,
    poisson_err, q2_edges). Twin of make_lft_spectrum but in linear space.
    """
    q2 = df["q2"].to_numpy()
    q2 = q2[(q2 > 0) & np.isfinite(q2)]
    edges = np.linspace(q2_min, q2_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(q2, bins=edges)
    counts = counts.astype(float)
    err = np.sqrt(np.maximum(counts, 1.0))
    return centers, counts, err, edges


def fft_baseline(q2_centers, counts, q2_edges, q2_events=None,
                 mode="kde", floor=0.5, savgol_window=11, savgol_poly=3,
                 kde_bw=None):
    """
    Build a baseline N_smooth(q^2) for the linear FFT scan. Mirrors lft_baseline.
    """
    counts = np.asarray(counts, dtype=float)
    n = len(counts)

    if mode == "kde":
        if q2_events is None or len(q2_events) < 5:
            mode = "floor"

    if mode == "savgol":
        win = min(savgol_window, n if n % 2 == 1 else n - 1)
        min_win = savgol_poly + 2
        if min_win % 2 == 0:
            min_win += 1
        win = max(win, min_win)
        baseline = savgol_filter(counts, window_length=win, polyorder=savgol_poly)
        return np.maximum(baseline, 1e-9)

    if mode == "floor":
        win = min(savgol_window, n if n % 2 == 1 else n - 1)
        min_win = savgol_poly + 2
        if min_win % 2 == 0:
            min_win += 1
        win = max(win, min_win)
        smooth = savgol_filter(counts, window_length=win, polyorder=savgol_poly)
        return np.maximum(smooth, float(floor))

    if mode == "kde":
        kde = gaussian_kde(q2_events, bw_method=kde_bw)
        density = kde(q2_centers)
        bin_widths = np.diff(q2_edges)
        baseline = density * len(q2_events) * bin_widths
        return np.maximum(baseline, 1e-9)

    raise ValueError(f"Unknown baseline mode: {mode}")


def fft_residual(counts, baseline):
    """Poisson-normalized residual on the linear-q^2 grid. Same as lft_residual."""
    counts = np.asarray(counts, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    return (counts - baseline) / np.sqrt(baseline + 1.0)


def linear_fft(q2_centers, residual, window="hanning", subtract_mean=True):
    """
    FFT the linear-q^2 residual. Returns (k_q2, amp). Twin of lft_fft.
    Note: k_q2 has units of 1/GeV^2, so it is NOT directly comparable in
    magnitude to k_ell (which is dimensionless). The artifact test compares
    *patterns of peaks* across the two domains, not the numerical k values.
    """
    r = np.asarray(residual, dtype=float)
    if subtract_mean:
        r = r - np.mean(r)

    n = len(r)
    if window == "hanning":
        w = np.hanning(n)
    elif window in (None, "none", "rect"):
        w = np.ones(n)
    else:
        raise ValueError(f"Unknown window: {window}")

    rw = r * w
    d_q2 = float(q2_centers[1] - q2_centers[0])
    amp = np.abs(rfft(rw))
    freq_cycles = rfftfreq(n, d=d_q2)
    k_q2 = 2.0 * np.pi * freq_cycles
    return k_q2, amp


def lft_segments(df, segments, q2_ref=1.0, n_bins_per_segment=None,
                 baseline_mode="kde", baseline_floor=0.5):
    """
    Segmented LFT: split q^2 into the listed (q2_lo, q2_hi) intervals
    (typically the regions between charmonium vetoes), bin each in ell,
    compute residual + FFT per segment. Returns a list of dicts:
        [{ "segment": (q2_lo, q2_hi), "ell_centers": ..., "counts": ...,
           "baseline": ..., "residual": ..., "k_ell": ..., "amp": ... }, ...]

    Per-segment k_ell grids generally differ; combining them requires care.
    For now, return them separately so the caller can decide how to merge.
    """
    out = []
    q2 = df["q2"].to_numpy()
    q2 = q2[(q2 > 0) & np.isfinite(q2)]

    for (q2_lo, q2_hi) in segments:
        m = (q2 >= q2_lo) & (q2 < q2_hi)
        q2_seg = q2[m]
        if len(q2_seg) < 5:
            out.append({
                "segment": (float(q2_lo), float(q2_hi)),
                "n_events": int(len(q2_seg)),
                "skipped": True,
            })
            continue

        if n_bins_per_segment is None:
            n_bins = max(8, int(np.ceil(np.log(q2_hi / q2_lo) * 12)))
        else:
            n_bins = n_bins_per_segment

        ell_events = np.log(q2_seg / q2_ref)
        ell_min = np.log(q2_lo / q2_ref)
        ell_max = np.log(q2_hi / q2_ref)
        ell_edges = np.linspace(ell_min, ell_max, n_bins + 1)
        ell_centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])
        counts, _ = np.histogram(ell_events, bins=ell_edges)
        counts = counts.astype(float)

        baseline = lft_baseline(ell_centers, counts, ell_edges,
                                ell_events=ell_events,
                                mode=baseline_mode, floor=baseline_floor)
        residual = lft_residual(counts, baseline)
        k_ell, amp = lft_fft(ell_centers, residual)

        out.append({
            "segment": (float(q2_lo), float(q2_hi)),
            "n_events": int(len(q2_seg)),
            "ell_centers": ell_centers,
            "counts": counts,
            "baseline": baseline,
            "residual": residual,
            "k_ell": k_ell,
            "amp": amp,
            "skipped": False,
        })
    return out


def null_bootstrap_residual(residual, n_trials, ell_centers, rng,
                            band_min, band_max, snr_min=3.0,
                            prominence_min=1.0, kmin=2.0, kmax=None):
    """
    Shuffle the residual N times, FFT each, run global_peak_report, count how
    often a significant peak lands in [band_min, band_max] under the null
    (no log-periodic structure). Returns false-positive rate and per-trial
    records. Used to calibrate whether observed in-band peaks are surprising.
    """
    rows = []
    n_in_band = 0
    n_significant = 0

    for i in range(n_trials):
        shuffled = rng.permutation(residual)
        k_ell, amp = lft_fft(ell_centers, shuffled)
        rep = global_peak_report(k_ell, amp, band_min, band_max,
                                 kmin=kmin, kmax=kmax,
                                 snr_min=snr_min, prominence_min=prominence_min)
        if rep is None:
            rows.append({"iter": i, "passed": False, "in_band": False})
            continue
        n_significant += 1
        if rep["in_band"]:
            n_in_band += 1
        rows.append({
            "iter": i,
            "passed": True,
            "in_band": rep["in_band"],
            "k_peak": rep["k_peak"],
            "snr_like": rep["snr_like"],
        })

    summary = {
        "n_trials": int(n_trials),
        "n_significant": int(n_significant),
        "n_in_band": int(n_in_band),
        "false_positive_rate_significance": float(n_significant / n_trials),
        "false_positive_rate_in_band": float(n_in_band / n_trials),
    }
    return summary, rows


def peak_report(k, amp, kmin=2.0, kmax=30.0, snr_min=3.0, prominence_min=1.0):
    """
    Return the most significant peak in [kmin, kmax] only if it is:
      (i)  a local maximum with prominence >= prominence_min * robust_sigma, AND
      (ii) (amp_peak - median) / robust_sigma >= snr_min.
    Where robust_sigma = 1.4826 * MAD of amplitudes in the search window.
    Returns None if no peak passes both gates.

    snr_min and prominence_min defaults match config.SNR_MIN, config.PROMINENCE_MIN.
    """
    k = np.asarray(k)
    amp = np.asarray(amp)
    mask = (k >= kmin) & (k <= kmax) & np.isfinite(amp)
    kk = k[mask]
    aa = amp[mask]
    if len(aa) < 5:
        return None

    med = float(np.median(aa))
    mad = float(np.median(np.abs(aa - med))) + 1e-12
    robust_sigma = 1.4826 * mad

    if robust_sigma < 1e-6:
        return None  # protection against degenerate spectra (e.g. all zeros)

    peaks_idx, props = find_peaks(aa, prominence=prominence_min * robust_sigma)
    if len(peaks_idx) == 0:
        return None

    snr_arr = (aa[peaks_idx] - med) / robust_sigma
    good = snr_arr >= snr_min
    if not np.any(good):
        return None

    surviving_peaks = peaks_idx[good]
    surviving_snr = snr_arr[good]
    surviving_prom = props["prominences"][good]

    best_local = int(np.argmax(surviving_snr))
    best = surviving_peaks[best_local]

    return {
        "k_peak": float(kk[best]),
        "amp_peak": float(aa[best]),
        "median_amp": med,
        "robust_sigma": float(robust_sigma),
        "snr_like": float(surviving_snr[best_local]),
        "prominence": float(surviving_prom[best_local]),
    }


def global_peak_report(k, amp, band_min, band_max, kmin=2.0, kmax=None,
                       snr_min=3.0, prominence_min=1.0):
    """
    Search the FULL k range [kmin, kmax] (not just the WCT band), find the
    most significant peak, and report whether it lands inside [band_min, band_max].

    Honest stability test: if a real signal exists in the band, the GLOBAL
    argmax will land there without being told to. Restricting the search to
    the band before taking argmax is question-begging.

    Returns dict with peak info plus 'in_band' bool, or None if no peak
    passes the significance gate anywhere in [kmin, kmax].
    """
    k = np.asarray(k)
    amp = np.asarray(amp)
    if kmax is None:
        kmax = float(k.max())

    rep = peak_report(k, amp, kmin=kmin, kmax=kmax,
                      snr_min=snr_min, prominence_min=prominence_min)
    if rep is None:
        return None

    rep["in_band"] = bool(band_min <= rep["k_peak"] <= band_max)
    rep["band"] = [float(band_min), float(band_max)]
    return rep


def save_json(obj, path):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

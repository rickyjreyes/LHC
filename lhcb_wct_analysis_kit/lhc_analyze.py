# analyze_lhcb_kst_mumu.py
# pip install uproot awkward numpy pandas scipy matplotlib

import glob
import numpy as np
import pandas as pd
import uproot
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt


# -----------------------------
# 1. Load ROOT ntuples
# -----------------------------

FILES = "data/*.root"          # change if needed
TREE = "B0_KstMuMu/DecayTree"  # likely tree path; inspect if fails


def inspect_file(path):
    with uproot.open(path) as f:
        print(f.keys())
        for k in f.keys():
            print(k)


def load(files_pattern=FILES, tree_name=TREE):
    files = glob.glob(files_pattern)
    if not files:
        raise FileNotFoundError("No ROOT files found.")

    branches = [
        # B candidate
        "B0_M", "B0_PX", "B0_PY", "B0_PZ", "B0_PE",

        # Muons
        "muplus_PX", "muplus_PY", "muplus_PZ", "muplus_PE",
        "muminus_PX", "muminus_PY", "muminus_PZ", "muminus_PE",

        # K/pi
        "Kplus_PX", "Kplus_PY", "Kplus_PZ", "Kplus_PE",
        "piminus_PX", "piminus_PY", "piminus_PZ", "piminus_PE",

        # optional if present
        "Kst_M",
        "B0_ENDVERTEX_CHI2", "B0_ENDVERTEX_NDOF",
        "B0_IPCHI2_OWNPV", "B0_DIRA_OWNPV",
    ]

    dfs = []
    for path in files:
        with uproot.open(path) as f:
            if tree_name not in f:
                print(f"\nTree not found in {path}. Available:")
                print(f.keys())
                raise KeyError(tree_name)

            tree = f[tree_name]
            available = set(tree.keys())
            use = [b for b in branches if b in available]

            arr = tree.arrays(use, library="pd")
            arr["source_file"] = path
            dfs.append(arr)

    return pd.concat(dfs, ignore_index=True)


# -----------------------------
# 2. Derived kinematics
# -----------------------------

def inv_mass2(px, py, pz, e):
    return e**2 - px**2 - py**2 - pz**2


def add_q2(df):
    q_px = df["muplus_PX"] + df["muminus_PX"]
    q_py = df["muplus_PY"] + df["muminus_PY"]
    q_pz = df["muplus_PZ"] + df["muminus_PZ"]
    q_e  = df["muplus_PE"] + df["muminus_PE"]

    q2_mev2 = inv_mass2(q_px, q_py, q_pz, q_e)
    df["q2"] = q2_mev2 / 1e6  # MeV^2 -> GeV^2
    return df


def basic_selection(df):
    out = df.copy()

    # physical q^2 window used in B -> K* mu mu analyses
    out = out[(out["q2"] > 0.1) & (out["q2"] < 19.0)]

    # B mass window, adjust after plotting
    if "B0_M" in out:
        out = out[(out["B0_M"] > 5100) & (out["B0_M"] < 5600)]

    # K* mass window if branch exists
    if "Kst_M" in out:
        out = out[(out["Kst_M"] > 750) & (out["Kst_M"] < 1100)]

    # veto charmonium regions
    out = out[~((out["q2"] > 8.0) & (out["q2"] < 11.0))]    # J/psi
    out = out[~((out["q2"] > 12.5) & (out["q2"] < 15.0))]   # psi(2S)

    return out


# -----------------------------
# 3. q^2 spectrum + residual
# -----------------------------

def make_q2_spectrum(df, bins=60, q2_min=0.1, q2_max=19.0):
    counts, edges = np.histogram(df["q2"], bins=bins, range=(q2_min, q2_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    err = np.sqrt(np.maximum(counts, 1))
    return centers, counts.astype(float), err


def smooth_residual(x, y, window=11, poly=3):
    # window must be odd and <= len(y)
    window = min(window, len(y) - (1 - len(y) % 2))
    if window % 2 == 0:
        window -= 1
    window = max(window, poly + 2 + ((poly + 2) % 2 == 0))

    baseline = savgol_filter(y, window_length=window, polyorder=poly)
    baseline = np.maximum(baseline, 1e-9)
    residual = y / baseline - 1.0
    return baseline, residual


# -----------------------------
# 4. WCT log-q^2 FFT scan
# -----------------------------

def log_fft_scan(q2, residual, n_grid=512):
    mask = (q2 > 0) & np.isfinite(residual)
    q2 = q2[mask]
    residual = residual[mask]

    ell = np.log(q2)
    order = np.argsort(ell)
    ell = ell[order]
    residual = residual[order]

    ell_grid = np.linspace(ell.min(), ell.max(), n_grid)
    interp = interp1d(ell, residual, kind="linear", fill_value="extrapolate")
    r_grid = interp(ell_grid)

    # remove mean and window
    r_grid = r_grid - np.mean(r_grid)
    win = np.hanning(len(r_grid))
    rw = r_grid * win

    d_ell = ell_grid[1] - ell_grid[0]
    amp = np.abs(rfft(rw))
    freq_cycles = rfftfreq(n_grid, d=d_ell)

    # Convert cycles per log-unit to angular log-frequency k_l
    k = 2 * np.pi * freq_cycles

    return ell_grid, r_grid, k, amp


def peak_report(k, amp, kmin=2, kmax=30):
    mask = (k >= kmin) & (k <= kmax)
    kk = k[mask]
    aa = amp[mask]

    if len(kk) == 0:
        return None

    med = np.median(aa)
    idx = np.argmax(aa)
    return {
        "k_peak": kk[idx],
        "amp_peak": aa[idx],
        "median_amp": med,
        "snr_like": aa[idx] / med if med > 0 else np.inf,
    }


# -----------------------------
# 5. Robustness test
# -----------------------------

def bootstrap_fft(df, n_boot=200, bins=60):
    peaks = []

    for _ in range(n_boot):
        sample = df.sample(len(df), replace=True)
        q2, y, err = make_q2_spectrum(sample, bins=bins)
        _, r = smooth_residual(q2, y)
        _, _, k, amp = log_fft_scan(q2, r)

        rep = peak_report(k, amp, 8, 20)
        if rep:
            peaks.append(rep["k_peak"])

    return np.array(peaks)


# -----------------------------
# 6. Plots
# -----------------------------

def plot_all(q2, y, baseline, residual, k, amp, out_prefix="lhcb"):
    plt.figure(figsize=(8, 5))
    plt.errorbar(q2, y, yerr=np.sqrt(np.maximum(y, 1)), fmt="o", ms=3, label="data")
    plt.plot(q2, baseline, label="smooth baseline")
    plt.xlabel(r"$q^2$ [GeV$^2$]")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_q2_spectrum.png", dpi=200)

    plt.figure(figsize=(8, 5))
    plt.axhline(0, lw=1)
    plt.plot(q2, residual, "o-", ms=3)
    plt.xlabel(r"$q^2$ [GeV$^2$]")
    plt.ylabel(r"$R = N/N_{\rm smooth}-1$")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_residual.png", dpi=200)

    plt.figure(figsize=(8, 5))
    mask = (k > 0) & (k < 40)
    plt.plot(k[mask], amp[mask])
    plt.axvspan(8, 20, alpha=0.15, label="WCT target band")
    plt.xlabel(r"log-frequency $k_\ell$")
    plt.ylabel("FFT amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_log_fft.png", dpi=200)


# -----------------------------
# 7. Main
# -----------------------------

def main():
    df = load()
    print("Loaded events:", len(df))

    df = add_q2(df)
    df_sel = basic_selection(df)
    print("Selected events:", len(df_sel))

    q2, y, err = make_q2_spectrum(df_sel, bins=60)
    baseline, residual = smooth_residual(q2, y, window=11, poly=3)

    ell_grid, r_grid, k, amp = log_fft_scan(q2, residual)

    rep_all = peak_report(k, amp, 2, 30)
    rep_wct = peak_report(k, amp, 8, 20)

    print("\nPeak report [2,30]:", rep_all)
    print("Peak report [8,20] WCT band:", rep_wct)

    peaks = bootstrap_fft(df_sel, n_boot=200, bins=60)
    if len(peaks):
        print("\nBootstrap WCT-band peak:")
        print("mean k =", np.mean(peaks))
        print("std k  =", np.std(peaks))
        print("5-95%  =", np.percentile(peaks, [5, 95]))

    plot_all(q2, y, baseline, residual, k, amp)

    out = pd.DataFrame({
        "q2_center": q2,
        "counts": y,
        "baseline": baseline,
        "residual": residual,
        "err": err,
    })
    out.to_csv("lhcb_q2_residuals.csv", index=False)

    print("\nSaved:")
    print("lhcb_q2_residuals.csv")
    print("lhcb_q2_spectrum.png")
    print("lhcb_residual.png")
    print("lhcb_log_fft.png")


if __name__ == "__main__":
    main()
"""
FFT control scan in LINEAR q^2.

Bin events uniformly in q^2 (NOT log q^2), build the same Poisson-normalized
residual against a KDE baseline, and FFT the result. This is the artifact /
diagnostic test that runs in parallel to 02_lft_wct_scan.py.

Decision rule (combine with the LFT result):
    LFT peak in WCT band, FFT clean       -> log-periodic candidate (WCT-like)
    LFT peak in WCT band, FFT same peak   -> bin / cut / selection artifact
    No LFT peak                           -> no detection

Note on units: k_q2 has units 1/GeV^2, k_ell is dimensionless. Numerical
values are NOT directly comparable - the test compares whether each spectrum
has a significant peak at all, and whether their structures co-vary across
the cut sweep (see 05_cut_sweep.py).

Outputs:
    outputs/fft_residuals.csv     (q2-binned counts, baseline, residual)
    outputs/fft_power.csv         (k_q2, amplitude)
    outputs/fft_spectrum.png      (counts vs q^2 with baseline)
    outputs/fft_residual.png      (Poisson-normalized residual vs q^2)
    outputs/fft_power.png         (FFT amplitude in linear-q^2 frequency)
    outputs/fft_summary.json      (peaks, gates, diagnostic flag)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *
from lhcb_utils import (
    load_dataframe, add_q2, add_kst_mass, basic_selection,
    make_q2_linear_spectrum, fft_baseline, fft_residual, linear_fft,
    peak_report, save_json,
)

REQUESTED = [
    "B0_M",
    "muplus_PX", "muplus_PY", "muplus_PZ", "muplus_PE",
    "muminus_PX", "muminus_PY", "muminus_PZ", "muminus_PE",
    "Kplus_PX", "Kplus_PY", "Kplus_PZ", "Kplus_PE",
    "piminus_PX", "piminus_PY", "piminus_PZ", "piminus_PE",
    "Kst_M",
]


def get_kst_window():
    if KST_MODE == "tight":
        return KST_M_MIN, KST_M_MAX
    return KST_M_MIN_LOOSE, KST_M_MAX_LOOSE


def make_plots(centers, counts, err, baseline, residual, k_q2, amp):
    OUT_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.errorbar(centers, counts, yerr=err, fmt="o", ms=3, label="data")
    plt.plot(centers, baseline, label=f"baseline ({BASELINE_MODE})")
    plt.xlabel(r"$q^2$ [GeV$^2$]")
    plt.ylabel("Counts per linear $q^2$ bin")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fft_spectrum.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.axhline(0, lw=1, color="k")
    plt.plot(centers, residual, "o-", ms=3)
    plt.xlabel(r"$q^2$ [GeV$^2$]")
    plt.ylabel(r"$R = (N - B) / \sqrt{B + 1}$")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fft_residual.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    mask = k_q2 > 0
    plt.plot(k_q2[mask], amp[mask])
    plt.xlabel(r"linear-$q^2$ frequency $k_{q^2}$ [1/GeV$^2$]")
    plt.ylabel("FFT amplitude")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fft_power.png", dpi=200)
    plt.close()


def main():
    df, used_tree, missing = load_dataframe(FILES_GLOB, TREE_NAME, REQUESTED)
    print("Tree used:", used_tree)
    print("Loaded events:", len(df))

    df = add_q2(df)
    if "Kst_M" not in df.columns:
        df = add_kst_mass(df)

    kst_min, kst_max = get_kst_window()
    apply_jpsi  = JPSI_VETO  if VETO_MODE in ("mask", "segment_lft") else None
    apply_psi2s = PSI2S_VETO if VETO_MODE in ("mask", "segment_lft") else None

    df_sel = basic_selection(
        df, q2_min=Q2_MIN, q2_max=Q2_MAX,
        b0_m_min=B0_M_MIN, b0_m_max=B0_M_MAX,
        kst_m_min=kst_min, kst_m_max=kst_max,
        jpsi_veto=apply_jpsi, psi2s_veto=apply_psi2s,
        require_kst=True, verbose=False,
    )
    print(f"Selected events: {len(df_sel)}")

    diagnostic_only = len(df_sel) < MIN_EVENTS_FOR_LFT
    if diagnostic_only:
        print(f"\n*** DIAGNOSTIC ONLY: selected events ({len(df_sel)}) < "
              f"MIN_EVENTS_FOR_LFT ({MIN_EVENTS_FOR_LFT}) ***")

    centers, counts, err, edges = make_q2_linear_spectrum(
        df_sel, q2_min=Q2_MIN, q2_max=Q2_MAX, n_bins=N_FFT_BINS
    )
    q2_events = df_sel["q2"].to_numpy()
    baseline = fft_baseline(
        centers, counts, edges, q2_events=q2_events,
        mode=BASELINE_MODE, floor=BASELINE_FLOOR,
        savgol_window=SMOOTH_WINDOW, savgol_poly=SMOOTH_POLY,
    )
    residual = fft_residual(counts, baseline)
    k_q2, amp = linear_fft(centers, residual)

    # Search the full positive-frequency range. Linear-q^2 FFT has no
    # natural "WCT band" - that band is meaningful only in log space.
    rep = peak_report(k_q2, amp, kmin=0.5, kmax=float(k_q2.max()),
                      snr_min=SNR_MIN, prominence_min=PROMINENCE_MIN)
    print(f"\nLinear-q^2 FFT peak (full range): {rep}")

    pd.DataFrame({
        "q2_center": centers,
        "counts": counts,
        "err": err,
        "baseline": baseline,
        "residual": residual,
    }).to_csv(OUT_DIR / "fft_residuals.csv", index=False)
    pd.DataFrame({"k_q2": k_q2, "amp": amp}).to_csv(OUT_DIR / "fft_power.csv", index=False)
    make_plots(centers, counts, err, baseline, residual, k_q2, amp)

    summary = {
        "tree": used_tree,
        "baseline_mode": BASELINE_MODE,
        "kst_mode": KST_MODE,
        "veto_mode": VETO_MODE,
        "n_fft_bins": int(N_FFT_BINS),
        "loaded_events": int(len(df)),
        "selected_events": int(len(df_sel)),
        "diagnostic_only": diagnostic_only,
        "min_events_for_fft": MIN_EVENTS_FOR_LFT,
        "snr_min": SNR_MIN,
        "prominence_min": PROMINENCE_MIN,
        "k_q2_units": "1/GeV^2",
        "k_q2_max": float(k_q2.max()),
        "linear_fft_peak": rep,
        "missing_branches": missing,
        "cutflow": df_sel.attrs.get("cutflow", []),
        "interpretation_note": (
            "Compare to lft_summary.json. If the LFT shows a WCT-band peak "
            "and this FFT shows no significant peak, the structure is "
            "log-periodic and WCT-consistent. If both show significant peaks, "
            "the structure is more likely a binning / cut / selection artifact."
        ),
    }
    save_json(summary, OUT_DIR / "fft_summary.json")
    print("\nSaved outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()

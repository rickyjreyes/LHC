"""
LFT (log-Fourier transform) scan for log-periodic residuals in B0 -> K*0 mu+ mu-.

Tests for: R(ell) = A * cos(k_ell * ell + phi), with ell = ln(q^2 / Q2_REF).
Bins uniformly in ell (NOT q^2). FFT applied directly to the Poisson-normalized
residual on the ell grid.

Outputs:
    outputs/lft_residuals.csv     (ell-binned counts, baseline, residual)
    outputs/lft_power.csv         (k_ell, amplitude)
    outputs/lft_spectrum.png      (counts vs ell with baseline)
    outputs/lft_residual.png      (Poisson-normalized residual vs ell)
    outputs/lft_power.png         (FFT amplitude with WCT band shaded)
    outputs/lft_summary.json      (peaks, cutflow, gates passed)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *
from lhcb_utils import (
    load_dataframe, add_q2, add_kst_mass, basic_selection,
    make_lft_spectrum, lft_baseline, lft_residual, lft_fft, lft_segments,
    peak_report, global_peak_report, save_json,
)

REQUESTED = [
    "B0_M", "B0_PX", "B0_PY", "B0_PZ", "B0_PE",
    "muplus_PX", "muplus_PY", "muplus_PZ", "muplus_PE",
    "muminus_PX", "muminus_PY", "muminus_PZ", "muminus_PE",
    "Kplus_PX", "Kplus_PY", "Kplus_PZ", "Kplus_PE",
    "piminus_PX", "piminus_PY", "piminus_PZ", "piminus_PE",
    "Kst_M", "eventNumber", "runNumber", "cosThetaL", "cosThetaK", "phi",
]


def get_kst_window():
    if KST_MODE == "tight":
        return KST_M_MIN, KST_M_MAX
    if KST_MODE == "loose":
        return KST_M_MIN_LOOSE, KST_M_MAX_LOOSE
    raise ValueError(f"Unknown KST_MODE: {KST_MODE}")


def make_plots(ell_centers, counts, err, baseline, residual, k_ell, amp, q2_ref):
    OUT_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.errorbar(ell_centers, counts, yerr=err, fmt="o", ms=3, label="data")
    plt.plot(ell_centers, baseline, label=f"baseline ({BASELINE_MODE})")
    plt.xlabel(rf"$\ell = \ln(q^2 / {q2_ref}\,\mathrm{{GeV}}^2)$")
    plt.ylabel("Counts per ell-bin")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "lft_spectrum.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.axhline(0, lw=1, color="k")
    plt.plot(ell_centers, residual, "o-", ms=3)
    plt.xlabel(rf"$\ell = \ln(q^2 / {q2_ref}\,\mathrm{{GeV}}^2)$")
    plt.ylabel(r"$R = (N - B) / \sqrt{B + 1}$")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "lft_residual.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    mask = (k_ell > 0) & (k_ell < 40)
    plt.plot(k_ell[mask], amp[mask])
    plt.axvspan(WCT_K_MIN, WCT_K_MAX, alpha=0.15, label="WCT target band")
    plt.xlabel(r"log-frequency $k_\ell$")
    plt.ylabel("FFT amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "lft_power.png", dpi=200)
    plt.close()


def run_continuous_lft(df_sel):
    """Standard LFT: bin all selected events in ell, FFT continuous spectrum."""
    ec, counts, err, edges, q2_ref = make_lft_spectrum(
        df_sel, q2_ref=Q2_REF, q2_min=Q2_MIN, q2_max=Q2_MAX, n_bins=N_LFT_BINS
    )
    ell_events = np.log(df_sel["q2"].to_numpy() / q2_ref)
    baseline = lft_baseline(
        ec, counts, edges, ell_events=ell_events,
        mode=BASELINE_MODE, floor=BASELINE_FLOOR,
        savgol_window=SMOOTH_WINDOW, savgol_poly=SMOOTH_POLY,
    )
    residual = lft_residual(counts, baseline)
    k_ell, amp = lft_fft(ec, residual)
    return ec, counts, err, baseline, residual, k_ell, amp, q2_ref


def run_segmented_lft(df_sel):
    """Segmented LFT: split at veto edges, FFT each segment separately."""
    segments = [
        (Q2_MIN, JPSI_VETO[0]),
        (JPSI_VETO[1], PSI2S_VETO[0]),
        (PSI2S_VETO[1], Q2_MAX),
    ]
    return lft_segments(
        df_sel, segments, q2_ref=Q2_REF,
        baseline_mode=BASELINE_MODE, baseline_floor=BASELINE_FLOOR,
    )


def main():
    df, used_tree, missing = load_dataframe(FILES_GLOB, TREE_NAME, REQUESTED)
    print("Tree used:", used_tree)
    print("Loaded events:", len(df))

    df = add_q2(df)
    if "Kst_M" not in df.columns:
        df = add_kst_mass(df)
        if "Kst_M" in df.columns:
            print("Kst_M not in branches; computed from K+ pi- four-vectors.")
        else:
            print("WARNING: Kst_M cannot be computed (K+/pi- four-vectors missing).")

    kst_min, kst_max = get_kst_window()
    print(f"K* window ({KST_MODE}): [{kst_min}, {kst_max}] MeV")

    apply_jpsi  = JPSI_VETO  if VETO_MODE in ("mask", "segment_lft") else None
    apply_psi2s = PSI2S_VETO if VETO_MODE in ("mask", "segment_lft") else None

    df_sel = basic_selection(
        df, q2_min=Q2_MIN, q2_max=Q2_MAX,
        b0_m_min=B0_M_MIN, b0_m_max=B0_M_MAX,
        kst_m_min=kst_min, kst_m_max=kst_max,
        jpsi_veto=apply_jpsi, psi2s_veto=apply_psi2s,
        require_kst=True, verbose=True,
    )
    print(f"Selected events: {len(df_sel)}")

    diagnostic_only = len(df_sel) < MIN_EVENTS_FOR_LFT
    if diagnostic_only:
        print(f"\n*** DIAGNOSTIC ONLY: selected events ({len(df_sel)}) < "
              f"MIN_EVENTS_FOR_LFT ({MIN_EVENTS_FOR_LFT}) ***")
        print("    LFT results below are NOT a valid WCT detection test.")

    if VETO_MODE == "segment_lft":
        seg_results = run_segmented_lft(df_sel)
        seg_summary = []
        for s in seg_results:
            entry = {"segment": s["segment"], "n_events": s["n_events"],
                     "skipped": s["skipped"]}
            if not s["skipped"]:
                rep = global_peak_report(
                    s["k_ell"], s["amp"], WCT_K_MIN, WCT_K_MAX,
                    kmin=2.0, snr_min=SNR_MIN, prominence_min=PROMINENCE_MIN,
                )
                entry["peak"] = rep
            seg_summary.append(entry)
        save_json({
            "tree": used_tree,
            "veto_mode": VETO_MODE,
            "loaded_events": int(len(df)),
            "selected_events": int(len(df_sel)),
            "diagnostic_only": diagnostic_only,
            "segments": seg_summary,
            "missing_branches": missing,
        }, OUT_DIR / "lft_summary.json")
        print("Segmented LFT done. See outputs/lft_summary.json")
        return

    # Continuous LFT
    ec, counts, err, baseline, residual, k_ell, amp, q2_ref = run_continuous_lft(df_sel)

    rep_band = peak_report(k_ell, amp, WCT_K_MIN, WCT_K_MAX,
                          snr_min=SNR_MIN, prominence_min=PROMINENCE_MIN)
    rep_global = global_peak_report(k_ell, amp, WCT_K_MIN, WCT_K_MAX,
                                    kmin=2.0, snr_min=SNR_MIN,
                                    prominence_min=PROMINENCE_MIN)

    print(f"\nWCT-band peak (snr_min={SNR_MIN}): {rep_band}")
    print(f"Global peak (in_band={rep_global['in_band'] if rep_global else None}): {rep_global}")

    pd.DataFrame({
        "ell_center": ec,
        "counts": counts,
        "err": err,
        "baseline": baseline,
        "residual": residual,
    }).to_csv(OUT_DIR / "lft_residuals.csv", index=False)
    pd.DataFrame({"k_ell": k_ell, "amp": amp}).to_csv(OUT_DIR / "lft_power.csv", index=False)
    make_plots(ec, counts, err, baseline, residual, k_ell, amp, q2_ref)

    summary = {
        "tree": used_tree,
        "veto_mode": VETO_MODE,
        "baseline_mode": BASELINE_MODE,
        "kst_mode": KST_MODE,
        "loaded_events": int(len(df)),
        "selected_events": int(len(df_sel)),
        "diagnostic_only": diagnostic_only,
        "min_events_for_lft": MIN_EVENTS_FOR_LFT,
        "wct_band": [WCT_K_MIN, WCT_K_MAX],
        "snr_min": SNR_MIN,
        "prominence_min": PROMINENCE_MIN,
        "wct_band_peak": rep_band,
        "global_peak": rep_global,
        "missing_branches": missing,
        "cutflow": df_sel.attrs.get("cutflow", []),
    }
    save_json(summary, OUT_DIR / "lft_summary.json")
    print("\nSaved outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()

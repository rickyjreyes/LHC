"""
Cut sweep: run the LFT and FFT scans across four selection modes and report
peak stability. Identifies artifacts by checking whether a candidate peak
survives multiple cut definitions.

Modes (defined in lhcb_utils.select_with_mode):
    raw_q2  : q^2 range only
    loose   : + B0 mass window
    medium  : + K*(892) mass window
    tight   : + charmonium vetoes (J/psi, psi(2S))

Decision rules:
    Peak appears only in one mode and disappears under tighter cuts
        -> likely a fragile artifact tied to that specific selection.
    Peak appears in raw_q2 only, then vanishes once B0 cut is applied
        -> background / combinatorial structure, not signal physics.
    Peak persists across loose/medium/tight in LFT but NOT in FFT
        -> log-periodic and WCT-consistent (worth deeper investigation).
    Peak persists across modes in BOTH LFT and FFT
        -> likely binning / instrumental / kinematic structure, not WCT.

Outputs:
    outputs/cut_sweep.csv        (one row per mode with all peak metrics)
    outputs/cut_sweep.json       (same data, structured for downstream)
"""
import numpy as np
import pandas as pd

from config import *
from lhcb_utils import (
    load_dataframe, add_q2, add_kst_mass, select_with_mode,
    make_lft_spectrum, lft_baseline, lft_residual, lft_fft,
    make_q2_linear_spectrum, fft_baseline, fft_residual, linear_fft,
    peak_report, global_peak_report, save_json,
)

REQUESTED = [
    "B0_M",
    "muplus_PX", "muplus_PY", "muplus_PZ", "muplus_PE",
    "muminus_PX", "muminus_PY", "muminus_PZ", "muminus_PE",
    "Kplus_PX", "Kplus_PY", "Kplus_PZ", "Kplus_PE",
    "piminus_PX", "piminus_PY", "piminus_PZ", "piminus_PE",
    "Kst_M",
]

MODES = ["raw_q2", "loose", "medium", "tight"]


def get_kst_window():
    if KST_MODE == "tight":
        return KST_M_MIN, KST_M_MAX
    return KST_M_MIN_LOOSE, KST_M_MAX_LOOSE


def run_lft(df_sel):
    if len(df_sel) < 5:
        return None, None
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
    wct_band = peak_report(k_ell, amp, WCT_K_MIN, WCT_K_MAX,
                           snr_min=SNR_MIN, prominence_min=PROMINENCE_MIN)
    glb = global_peak_report(k_ell, amp, WCT_K_MIN, WCT_K_MAX,
                             kmin=2.0, snr_min=SNR_MIN,
                             prominence_min=PROMINENCE_MIN)
    return wct_band, glb


def run_fft(df_sel):
    if len(df_sel) < 5:
        return None
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
    return peak_report(k_q2, amp, kmin=0.5, kmax=float(k_q2.max()),
                       snr_min=SNR_MIN, prominence_min=PROMINENCE_MIN)


def fmt_peak(p, key):
    if p is None:
        return None
    return p.get(key)


def main():
    df, used_tree, missing = load_dataframe(FILES_GLOB, TREE_NAME, REQUESTED)
    df = add_q2(df)
    if "Kst_M" not in df.columns:
        df = add_kst_mass(df)

    kst_min, kst_max = get_kst_window()

    rows = []
    print(f"{'mode':<10} {'n':>6}  {'lft_band_k':>11} {'lft_band_snr':>13}  "
          f"{'lft_glb_k':>10} {'lft_glb_inband':>15}  {'fft_k':>8} {'fft_snr':>9}")
    print("-" * 110)

    for mode in MODES:
        try:
            df_sel = select_with_mode(
                df, mode,
                q2_min=Q2_MIN, q2_max=Q2_MAX,
                b0_m_min=B0_M_MIN, b0_m_max=B0_M_MAX,
                kst_m_min=kst_min, kst_m_max=kst_max,
                jpsi_veto=JPSI_VETO, psi2s_veto=PSI2S_VETO,
                verbose=False,
            )
        except ValueError as e:
            # medium/tight require Kst_M; skip if not available
            print(f"{mode:<10} skipped: {e}")
            continue

        n = len(df_sel)
        diagnostic = n < MIN_EVENTS_FOR_LFT

        wct_band, glb = run_lft(df_sel)
        fft_peak = run_fft(df_sel)

        row = {
            "mode": mode,
            "n_selected": int(n),
            "diagnostic_only": diagnostic,
            "lft_wct_band_k": fmt_peak(wct_band, "k_peak"),
            "lft_wct_band_snr": fmt_peak(wct_band, "snr_like"),
            "lft_global_k": fmt_peak(glb, "k_peak"),
            "lft_global_snr": fmt_peak(glb, "snr_like"),
            "lft_global_in_band": glb["in_band"] if glb else None,
            "fft_k_q2": fmt_peak(fft_peak, "k_peak"),
            "fft_snr": fmt_peak(fft_peak, "snr_like"),
        }
        rows.append(row)

        def s(v, w, fmt="{:>{w}.3f}"):
            if v is None:
                return f"{'-':>{w}}"
            if isinstance(v, bool):
                return f"{str(v):>{w}}"
            return fmt.format(v, w=w)

        print(f"{mode:<10} {n:>6}  "
              f"{s(row['lft_wct_band_k'], 11)} {s(row['lft_wct_band_snr'], 13)}  "
              f"{s(row['lft_global_k'], 10)} {s(row['lft_global_in_band'], 15)}  "
              f"{s(row['fft_k_q2'], 8)} {s(row['fft_snr'], 9)}")

    # Stability assessment
    lft_band_ks = [r["lft_wct_band_k"] for r in rows
                   if r["lft_wct_band_k"] is not None]
    fft_ks = [r["fft_k_q2"] for r in rows if r["fft_k_q2"] is not None]

    if len(lft_band_ks) >= 2:
        lft_k_arr = np.array(lft_band_ks)
        lft_persist = float(np.std(lft_k_arr))
    else:
        lft_persist = None

    interpretation = []
    n_lft_band = len(lft_band_ks)
    n_fft = len(fft_ks)
    n_modes_attempted = len(rows)

    # Modes where LFT WCT-band and FFT *both* fired.
    co_occurrence_modes = [
        r["mode"] for r in rows
        if r["lft_wct_band_k"] is not None and r["fft_k_q2"] is not None
    ]
    n_co = len(co_occurrence_modes)

    if n_lft_band == 0:
        interpretation.append("No LFT WCT-band peak in any cut mode.")
    elif n_lft_band == 1:
        which = next(r["mode"] for r in rows if r["lft_wct_band_k"] is not None)
        interpretation.append(
            f"LFT WCT-band peak appears ONLY in mode '{which}'. "
            "Likely fragile / cut-specific artifact."
        )
    elif n_lft_band >= 2 and n_co == 0:
        interpretation.append(
            f"LFT WCT-band peak persists across {n_lft_band} cut modes; "
            "FFT shows no co-occurring peak in any of those modes. "
            "Pattern is consistent with log-periodic structure (WCT-like)."
        )
    elif n_lft_band >= 2 and n_co >= n_lft_band:
        interpretation.append(
            "LFT WCT-band peaks AND linear-q^2 FFT peaks BOTH appear in "
            "the same cut modes. Co-occurrence suggests a binning / "
            "instrumental structure rather than WCT."
        )
    elif n_lft_band >= 2 and 0 < n_co < n_lft_band:
        interpretation.append(
            f"LFT WCT-band peak persists in {n_lft_band} modes; "
            f"FFT co-occurs in {n_co} of them. Mixed evidence: the LFT "
            "signal is more robust than any FFT artifact, but check the "
            "FFT-co-occurring modes for selection-induced structure."
        )
    else:
        interpretation.append(
            f"Mixed: LFT in {n_lft_band}/{n_modes_attempted} modes, "
            f"FFT in {n_fft}/{n_modes_attempted}, co-occurring in {n_co}. "
            "Inconclusive without more statistics."
        )

    pd.DataFrame(rows).to_csv(OUT_DIR / "cut_sweep.csv", index=False)
    save_json({
        "modes": rows,
        "n_modes_attempted": n_modes_attempted,
        "lft_band_persistence_count": n_lft_band,
        "fft_persistence_count": n_fft,
        "co_occurrence_modes": co_occurrence_modes,
        "co_occurrence_count": n_co,
        "lft_band_k_std_across_modes": lft_persist,
        "interpretation": interpretation,
        "settings": {
            "wct_band": [WCT_K_MIN, WCT_K_MAX],
            "wct_k_target": WCT_K_TARGET,
            "snr_min": SNR_MIN,
            "prominence_min": PROMINENCE_MIN,
            "min_events_for_lft": MIN_EVENTS_FOR_LFT,
            "n_lft_bins": N_LFT_BINS,
            "n_fft_bins": N_FFT_BINS,
        },
    }, OUT_DIR / "cut_sweep.json")

    print("\n--- Interpretation ---")
    for line in interpretation:
        print(line)


if __name__ == "__main__":
    main()

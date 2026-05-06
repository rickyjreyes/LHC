"""
Bootstrap stability + null calibration for the LFT scan.

Two complementary tests:
  1. Event-resampling bootstrap: resample selected events with replacement,
     re-bin in ell, refit baseline, recompute residual + FFT, run the
     significance-gated global_peak_report. Reports:
       - fraction of bootstraps where ANY significant peak exists
       - of those, fraction landing in the WCT band
     Restricting argmax to the WCT band before peak selection (legacy) is
     question-begging and is reported only as a labeled sanity number.

  2. Null bootstrap: shuffle the observed residual N times, run the same
     significance gate, report the false-positive rate. This calibrates
     whether observed in-band peaks are surprising under the no-structure null.
"""
import numpy as np
import pandas as pd

from config import *
from lhcb_utils import (
    load_dataframe, add_q2, add_kst_mass, basic_selection,
    make_lft_spectrum, lft_baseline, lft_residual, lft_fft,
    peak_report, global_peak_report, null_bootstrap_residual,
    save_json,
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


def one_lft_scan(df):
    """Bin in ell, build baseline + residual, FFT, return (k_ell, amp, ec, residual)."""
    ec, counts, err, edges, q2_ref = make_lft_spectrum(
        df, q2_ref=Q2_REF, q2_min=Q2_MIN, q2_max=Q2_MAX, n_bins=N_LFT_BINS
    )
    ell_events = np.log(df["q2"].to_numpy() / q2_ref)
    baseline = lft_baseline(
        ec, counts, edges, ell_events=ell_events,
        mode=BASELINE_MODE, floor=BASELINE_FLOOR,
        savgol_window=SMOOTH_WINDOW, savgol_poly=SMOOTH_POLY,
    )
    residual = lft_residual(counts, baseline)
    k_ell, amp = lft_fft(ec, residual)
    return k_ell, amp, ec, residual


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    df, used_tree, missing = load_dataframe(FILES_GLOB, TREE_NAME, REQUESTED)
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
    n = len(df_sel)
    print(f"Selected events: {n}")
    diagnostic_only = n < MIN_EVENTS_FOR_LFT
    if diagnostic_only:
        print(f"*** DIAGNOSTIC ONLY (n={n} < MIN_EVENTS_FOR_LFT={MIN_EVENTS_FOR_LFT}) ***")

    # ---- 1. Event-resampling bootstrap ----
    band_rows = []      # legacy: argmax restricted to WCT band before selection
    global_rows = []    # honest: significance-gated global search
    n_global_passed  = 0
    n_global_in_band = 0

    for i in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        sample = df_sel.iloc[idx]
        try:
            k_ell, amp, _, _ = one_lft_scan(sample)
        except Exception:
            continue

        band_rep = peak_report(k_ell, amp, WCT_K_MIN, WCT_K_MAX,
                              snr_min=SNR_MIN, prominence_min=PROMINENCE_MIN)
        global_rep = global_peak_report(k_ell, amp, WCT_K_MIN, WCT_K_MAX,
                                        kmin=2.0, snr_min=SNR_MIN,
                                        prominence_min=PROMINENCE_MIN)
        if band_rep:
            band_rep["iter"] = i
            band_rows.append(band_rep)
        if global_rep:
            global_rep["iter"] = i
            global_rows.append(global_rep)
            n_global_passed += 1
            if global_rep["in_band"]:
                n_global_in_band += 1

    band_df = pd.DataFrame(band_rows)
    global_df = pd.DataFrame(global_rows)
    band_df.to_csv(OUT_DIR / "bootstrap_peaks.csv", index=False)
    global_df.to_csv(OUT_DIR / "bootstrap_peaks_global.csv", index=False)

    # ---- 2. Null bootstrap on the observed residual ----
    null_summary = None
    if n >= 5:
        try:
            _, _, ec_obs, residual_obs = one_lft_scan(df_sel)
            null_summary, null_rows = null_bootstrap_residual(
                residual_obs, NULL_BOOTSTRAP_N, ec_obs,
                np.random.default_rng(RANDOM_SEED + 1),
                WCT_K_MIN, WCT_K_MAX,
                snr_min=SNR_MIN, prominence_min=PROMINENCE_MIN,
                kmin=2.0,
            )
            pd.DataFrame(null_rows).to_csv(OUT_DIR / "null_bootstrap.csv", index=False)
        except Exception as e:
            print(f"Null bootstrap failed: {e}")

    summary = {
        "n_bootstrap": int(BOOTSTRAP_N),
        "n_selected_events": int(n),
        "diagnostic_only": diagnostic_only,
        "wct_band": [float(WCT_K_MIN), float(WCT_K_MAX)],
        "snr_min": float(SNR_MIN),
        "prominence_min": float(PROMINENCE_MIN),
        "band_restricted_argmax": {
            "note": "Legacy. Argmax restricted to WCT band BEFORE selection. "
                    "Carries no stability information; included only for "
                    "comparison to old runs.",
            "n_returned": int(len(band_df)),
        },
        "global_search_significance_gated": {
            "note": "Honest stability test. Searches full k range, requires "
                    "local-max + (amp-med)/robust_sigma >= snr_min.",
            "n_passed_significance": int(n_global_passed),
            "fraction_passed_significance": float(n_global_passed / BOOTSTRAP_N),
            "fraction_in_band_given_passed": float(
                n_global_in_band / n_global_passed if n_global_passed > 0 else 0.0
            ),
            "fraction_in_band_overall": float(n_global_in_band / BOOTSTRAP_N),
        },
        "null_residual_shuffle": null_summary,
    }

    if len(band_df):
        summary["band_restricted_argmax"].update({
            "k_mean": float(band_df["k_peak"].mean()),
            "k_std": float(band_df["k_peak"].std()),
            "snr_like_mean": float(band_df["snr_like"].mean()),
        })
    if len(global_df):
        summary["global_search_significance_gated"].update({
            "k_mean": float(global_df["k_peak"].mean()),
            "k_std": float(global_df["k_peak"].std()),
            "k_p05": float(global_df["k_peak"].quantile(0.05)),
            "k_p95": float(global_df["k_peak"].quantile(0.95)),
            "snr_like_mean": float(global_df["snr_like"].mean()),
            "snr_like_median": float(global_df["snr_like"].median()),
        })

    save_json(summary, OUT_DIR / "bootstrap_summary.json")
    print("\nBootstrap summary:")
    print(f"  bootstraps with significant global peak : "
          f"{summary['global_search_significance_gated']['fraction_passed_significance']:.1%}")
    print(f"  of those, in WCT band                   : "
          f"{summary['global_search_significance_gated']['fraction_in_band_given_passed']:.1%}")
    if null_summary:
        print(f"  null shuffle FP rate (significance)     : "
              f"{null_summary['false_positive_rate_significance']:.1%}")
        print(f"  null shuffle FP rate (in-band)          : "
              f"{null_summary['false_positive_rate_in_band']:.1%}")


if __name__ == "__main__":
    main()

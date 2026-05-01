"""
Stability check on the raw_q2 cut-sweep peak (k_ell ~ 15.57, SNR ~ 3.32).

The test sample is too small to populate the strict B0 -> K*0 mu+ mu-
selection (only 5 events survive). Only raw_q2 (1504 events, no B0 mass
cut, no K*(892) cut, no charmonium veto) has enough stats for an LFT.
That selection is combinatorial-dominated and is NOT the WCT physics
channel, but we can still ask: is the k = 15.57 peak stable under
event resampling, or does it move around / vanish?

A genuine spectral feature should:
    * survive event-resampling bootstrap (same peak across draws)
    * not be matched by a residual-shuffle null (no in-band FPs)
    * not show a co-occurring linear-q^2 FFT peak

If any of those fail in raw_q2, the peak is artifact, not WCT.
"""
import json
import numpy as np
import pandas as pd

from config import (
    FILES_GLOB, TREE_NAME, OUT_DIR, Q2_REF, Q2_MIN, Q2_MAX,
    N_LFT_BINS, N_FFT_BINS, BASELINE_MODE, BASELINE_FLOOR,
    SMOOTH_WINDOW, SMOOTH_POLY,
    WCT_K_MIN, WCT_K_MAX, SNR_MIN, PROMINENCE_MIN,
    BOOTSTRAP_N, NULL_BOOTSTRAP_N, RANDOM_SEED,
)
from lhcb_utils import (
    load_dataframe, add_q2,
    select_with_mode,
    make_lft_spectrum, lft_baseline, lft_residual, lft_fft,
    make_q2_linear_spectrum, fft_baseline, fft_residual, linear_fft,
    global_peak_report, save_json,
)

REQUESTED = [
    "B0_M",
    "muplus_PX", "muplus_PY", "muplus_PZ", "muplus_PE",
    "muminus_PX", "muminus_PY", "muminus_PZ", "muminus_PE",
]


def lft_pipeline(df_sel):
    ec, counts, err, edges, q2_ref = make_lft_spectrum(
        df_sel, q2_ref=Q2_REF, q2_min=Q2_MIN, q2_max=Q2_MAX, n_bins=N_LFT_BINS,
    )
    ell_events = np.log(df_sel["q2"].to_numpy() / q2_ref)
    baseline = lft_baseline(
        ec, counts, edges, ell_events=ell_events,
        mode=BASELINE_MODE, floor=BASELINE_FLOOR,
        savgol_window=SMOOTH_WINDOW, savgol_poly=SMOOTH_POLY,
    )
    residual = lft_residual(counts, baseline)
    k_ell, amp = lft_fft(ec, residual)
    return ec, residual, k_ell, amp


def fft_pipeline(df_sel):
    qc, counts, err, edges = make_q2_linear_spectrum(
        df_sel, q2_min=Q2_MIN, q2_max=Q2_MAX, n_bins=N_FFT_BINS,
    )
    q2_events = df_sel["q2"].to_numpy()
    baseline = fft_baseline(
        qc, counts, edges, q2_events=q2_events,
        mode=BASELINE_MODE, floor=BASELINE_FLOOR,
        savgol_window=SMOOTH_WINDOW, savgol_poly=SMOOTH_POLY,
    )
    residual = fft_residual(counts, baseline)
    k_q2, amp = linear_fft(qc, residual)
    return k_q2, amp


def main():
    df, used_tree, _ = load_dataframe(FILES_GLOB, TREE_NAME, REQUESTED)
    df = add_q2(df)
    df_sel = select_with_mode(df, "raw_q2", q2_min=Q2_MIN, q2_max=Q2_MAX, verbose=False)
    n = len(df_sel)
    print(f"raw_q2 selected events: {n}")

    # --- nominal LFT and FFT ---
    ec, residual, k_ell, amp_lft = lft_pipeline(df_sel)
    nominal_lft = global_peak_report(
        k_ell, amp_lft, WCT_K_MIN, WCT_K_MAX,
        kmin=2.0, snr_min=SNR_MIN, prominence_min=PROMINENCE_MIN,
    )
    print("Nominal LFT global peak:", nominal_lft)

    k_q2, amp_fft = fft_pipeline(df_sel)
    nominal_fft = global_peak_report(
        k_q2, amp_fft, WCT_K_MIN, WCT_K_MAX,  # band reused, just for in_band flag
        kmin=0.5, snr_min=SNR_MIN, prominence_min=PROMINENCE_MIN,
    )
    print("Nominal linear-q^2 FFT global peak:", nominal_fft)

    rng = np.random.default_rng(RANDOM_SEED)

    # --- event-resampling bootstrap ---
    bs_rows = []
    n_passed = 0
    n_in_band = 0
    k_peaks = []
    for i in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        df_bs = df_sel.iloc[idx].reset_index(drop=True)
        try:
            _, _, k, amp = lft_pipeline(df_bs)
        except Exception:
            bs_rows.append({"iter": i, "passed": False, "in_band": False})
            continue
        rep = global_peak_report(
            k, amp, WCT_K_MIN, WCT_K_MAX,
            kmin=2.0, snr_min=SNR_MIN, prominence_min=PROMINENCE_MIN,
        )
        if rep is None:
            bs_rows.append({"iter": i, "passed": False, "in_band": False})
            continue
        n_passed += 1
        if rep["in_band"]:
            n_in_band += 1
        k_peaks.append(rep["k_peak"])
        bs_rows.append({
            "iter": i, "passed": True, "in_band": rep["in_band"],
            "k_peak": rep["k_peak"], "snr_like": rep["snr_like"],
        })

    bs_summary = {
        "n_trials": BOOTSTRAP_N,
        "n_passed_significance": n_passed,
        "fraction_passed_significance": n_passed / BOOTSTRAP_N,
        "n_in_band": n_in_band,
        "fraction_in_band_overall": n_in_band / BOOTSTRAP_N,
        "fraction_in_band_given_passed": (n_in_band / n_passed) if n_passed else 0.0,
        "k_peak_mean": float(np.mean(k_peaks)) if k_peaks else None,
        "k_peak_std": float(np.std(k_peaks)) if k_peaks else None,
        "k_peak_p05": float(np.percentile(k_peaks, 5)) if k_peaks else None,
        "k_peak_p95": float(np.percentile(k_peaks, 95)) if k_peaks else None,
    }

    # --- residual-shuffle null ---
    null_passed = 0
    null_in_band = 0
    for i in range(NULL_BOOTSTRAP_N):
        shuffled = rng.permutation(residual)
        k_n, amp_n = lft_fft(ec, shuffled)
        rep = global_peak_report(
            k_n, amp_n, WCT_K_MIN, WCT_K_MAX,
            kmin=2.0, snr_min=SNR_MIN, prominence_min=PROMINENCE_MIN,
        )
        if rep is None:
            continue
        null_passed += 1
        if rep["in_band"]:
            null_in_band += 1

    null_summary = {
        "n_trials": NULL_BOOTSTRAP_N,
        "n_passed_significance": null_passed,
        "false_positive_rate_significance": null_passed / NULL_BOOTSTRAP_N,
        "n_in_band": null_in_band,
        "false_positive_rate_in_band": null_in_band / NULL_BOOTSTRAP_N,
    }

    out = {
        "mode": "raw_q2",
        "selected_events": int(n),
        "selection_note": (
            "raw_q2 has no B0 mass cut, no K*(892) cut, and no charmonium veto. "
            "It is combinatorial-dominated and is NOT the B0 -> K*0 mu+ mu- "
            "physics channel. Used here only because it is the only selection "
            "with enough stats for an LFT in this test sample."
        ),
        "wct_band": [WCT_K_MIN, WCT_K_MAX],
        "snr_min": SNR_MIN,
        "prominence_min": PROMINENCE_MIN,
        "nominal_lft_global_peak": nominal_lft,
        "nominal_linear_fft_global_peak": nominal_fft,
        "event_resampling_bootstrap": bs_summary,
        "residual_shuffle_null": null_summary,
    }
    save_json(out, OUT_DIR / "raw_q2_bootstrap.json")
    pd.DataFrame(bs_rows).to_csv(OUT_DIR / "raw_q2_bootstrap_peaks.csv", index=False)
    print("\nWrote outputs/raw_q2_bootstrap.json and outputs/raw_q2_bootstrap_peaks.csv")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

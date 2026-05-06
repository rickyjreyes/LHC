import numpy as np
import pandas as pd

from config import *
from lhcb_utils import (
    load_dataframe, add_q2, basic_selection, make_q2_spectrum,
    smooth_residual, log_fft_scan, peak_report, save_json
)

REQUESTED = [
    "B0_M",
    "muplus_PX", "muplus_PY", "muplus_PZ", "muplus_PE",
    "muminus_PX", "muminus_PY", "muminus_PZ", "muminus_PE",
    "Kst_M",
]

def one_scan(df):
    q2, counts, err, edges = make_q2_spectrum(df, bins=Q2_BINS, q2_min=Q2_MIN, q2_max=Q2_MAX)
    baseline, residual = smooth_residual(q2, counts, window=SMOOTH_WINDOW, poly=SMOOTH_POLY)
    _, _, k, amp = log_fft_scan(q2, residual, n_grid=LOG_GRID_N)
    return peak_report(k, amp, WCT_K_MIN, WCT_K_MAX)

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    df, used_tree, missing = load_dataframe(FILES_GLOB, TREE_NAME, REQUESTED)
    df = add_q2(df)
    df_sel = basic_selection(
        df, q2_min=Q2_MIN, q2_max=Q2_MAX,
        b0_m_min=B0_M_MIN, b0_m_max=B0_M_MAX,
        kst_m_min=KST_M_MIN, kst_m_max=KST_M_MAX,
        jpsi_veto=JPSI_VETO, psi2s_veto=PSI2S_VETO,
    )

    rows = []
    n = len(df_sel)
    print("Selected events:", n)

    for i in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        sample = df_sel.iloc[idx]
        rep = one_scan(sample)
        if rep:
            rep["iter"] = i
            rows.append(rep)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "bootstrap_peaks.csv", index=False)

    summary = {}
    if len(out):
        summary = {
            "n_bootstrap": int(len(out)),
            "k_mean": float(out["k_peak"].mean()),
            "k_std": float(out["k_peak"].std()),
            "k_p05": float(out["k_peak"].quantile(0.05)),
            "k_p95": float(out["k_peak"].quantile(0.95)),
            "snr_like_mean": float(out["snr_like"].mean()),
            "snr_like_median": float(out["snr_like"].median()),
        }

    save_json(summary, OUT_DIR / "bootstrap_summary.json")
    print("\\nBootstrap summary:")
    print(summary)

if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt

from config import *
from lhcb_utils import (
    load_dataframe, add_q2, basic_selection, make_q2_spectrum,
    smooth_residual, log_fft_scan, peak_report, save_json
)

REQUESTED = [
    "B0_M", "B0_PX", "B0_PY", "B0_PZ", "B0_PE",
    "muplus_PX", "muplus_PY", "muplus_PZ", "muplus_PE",
    "muminus_PX", "muminus_PY", "muminus_PZ", "muminus_PE",
    "Kplus_PX", "Kplus_PY", "Kplus_PZ", "Kplus_PE",
    "piminus_PX", "piminus_PY", "piminus_PZ", "piminus_PE",
    "Kst_M", "eventNumber", "runNumber", "cosThetaL", "cosThetaK", "phi",
]

def plot_all(q2, counts, err, baseline, residual, k, amp):
    OUT_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.errorbar(q2, counts, yerr=err, fmt="o", ms=3, label="data")
    plt.plot(q2, baseline, label="smooth baseline")
    plt.xlabel(r"$q^2$ [GeV$^2$]")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "q2_spectrum.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.axhline(0, lw=1)
    plt.plot(q2, residual, "o-", ms=3)
    plt.xlabel(r"$q^2$ [GeV$^2$]")
    plt.ylabel(r"$R=N/N_{smooth}-1$")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "residual.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    mask = (k > 0) & (k < 40)
    plt.plot(k[mask], amp[mask])
    plt.axvspan(WCT_K_MIN, WCT_K_MAX, alpha=0.15, label="WCT target band")
    plt.xlabel(r"log-frequency $k_l$")
    plt.ylabel("FFT amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "log_fft.png", dpi=200)
    plt.close()

def main():
    df, used_tree, missing = load_dataframe(FILES_GLOB, TREE_NAME, REQUESTED)
    print("Tree used:", used_tree)
    print("Loaded events:", len(df))

    df = add_q2(df)
    df_sel = basic_selection(
        df, q2_min=Q2_MIN, q2_max=Q2_MAX,
        b0_m_min=B0_M_MIN, b0_m_max=B0_M_MAX,
        kst_m_min=KST_M_MIN, kst_m_max=KST_M_MAX,
        jpsi_veto=JPSI_VETO, psi2s_veto=PSI2S_VETO,
    )
    print("Selected events:", len(df_sel))

    q2, counts, err, edges = make_q2_spectrum(df_sel, bins=Q2_BINS, q2_min=Q2_MIN, q2_max=Q2_MAX)
    baseline, residual = smooth_residual(q2, counts, window=SMOOTH_WINDOW, poly=SMOOTH_POLY)
    ell_grid, r_grid, k, amp = log_fft_scan(q2, residual, n_grid=LOG_GRID_N)

    rep_all = peak_report(k, amp, 2, 30)
    rep_wct = peak_report(k, amp, WCT_K_MIN, WCT_K_MAX)

    print("\\nPeak report [2,30]:", rep_all)
    print(f"Peak report WCT band [{WCT_K_MIN},{WCT_K_MAX}]:", rep_wct)

    pd.DataFrame({
        "q2_center": q2,
        "counts": counts,
        "err": err,
        "baseline": baseline,
        "residual": residual,
    }).to_csv(OUT_DIR / "q2_residuals.csv", index=False)

    pd.DataFrame({"k_l": k, "amp": amp}).to_csv(OUT_DIR / "log_fft.csv", index=False)
    plot_all(q2, counts, err, baseline, residual, k, amp)

    summary = {
        "tree": used_tree,
        "loaded_events": int(len(df)),
        "selected_events": int(len(df_sel)),
        "peak_2_30": rep_all,
        "peak_wct_band": rep_wct,
        "missing": missing,
    }
    save_json(summary, OUT_DIR / "analysis_summary.json")
    print("\\nSaved outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()

# Optional control-channel comparison.
# Expected folders:
#   data_signal/*.root
#   data_control/*.root

from config import *
from lhcb_utils import (
    load_dataframe, add_q2, basic_selection, make_q2_spectrum,
    smooth_residual, log_fft_scan, peak_report
)

REQUESTED = [
    "B0_M",
    "muplus_PX", "muplus_PY", "muplus_PZ", "muplus_PE",
    "muminus_PX", "muminus_PY", "muminus_PZ", "muminus_PE",
    "Kst_M",
]

def scan(pattern):
    df, tree, missing = load_dataframe(pattern, TREE_NAME, REQUESTED)
    df = add_q2(df)
    df_sel = basic_selection(df)
    q2, counts, err, edges = make_q2_spectrum(df_sel, bins=Q2_BINS)
    baseline, residual = smooth_residual(q2, counts)
    _, _, k, amp = log_fft_scan(q2, residual)
    return peak_report(k, amp, WCT_K_MIN, WCT_K_MAX), len(df_sel)

def main():
    sig_rep, sig_n = scan("data_signal/*.root")
    ctrl_rep, ctrl_n = scan("data_control/*.root")

    print("Signal events:", sig_n)
    print("Signal WCT-band peak:", sig_rep)
    print("\\nControl events:", ctrl_n)
    print("Control WCT-band peak:", ctrl_rep)

    if sig_rep and ctrl_rep:
        dk = abs(sig_rep["k_peak"] - ctrl_rep["k_peak"])
        print("\\nDelta k signal-control:", dk)
        if dk < 1.0:
            print("WARNING: similar control peak. Possible detector/systematic artifact.")
        else:
            print("Control does not match signal peak.")

if __name__ == "__main__":
    main()

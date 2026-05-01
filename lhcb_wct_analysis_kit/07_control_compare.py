"""
Control-channel comparison.

Expected directory layout:
    data_signal/*.root    (B0 -> K* mu mu signal candidates)
    data_control/*.root   (B0 -> J/psi K* with J/psi -> mu mu, same final state)

Runs the LFT scan on both. WCT criterion requires:
    - significant peak in [WCT_K_MIN, WCT_K_MAX] in signal channel
    - NO matching peak (within DELTA_K_MATCH) in control channel
If both channels show the same peak, it is detector/selection structure,
not new physics.
"""
import numpy as np

from config import *
from lhcb_utils import (
    load_dataframe, add_q2, add_kst_mass, basic_selection,
    make_lft_spectrum, lft_baseline, lft_residual, lft_fft,
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

DELTA_K_MATCH = 1.0  # |k_signal - k_control| below this is a "matching" peak


def get_kst_window():
    if KST_MODE == "tight":
        return KST_M_MIN, KST_M_MAX
    return KST_M_MIN_LOOSE, KST_M_MAX_LOOSE


def scan_channel(pattern, label, apply_charm_vetoes=True):
    """Run continuous LFT on one channel. Returns dict with peak info + diagnostics."""
    print(f"\n--- {label}: {pattern} ---")
    try:
        df, used_tree, missing = load_dataframe(pattern, TREE_NAME, REQUESTED)
    except FileNotFoundError as e:
        return {"label": label, "n": 0, "peak": None, "error": str(e)}

    df = add_q2(df)
    if "Kst_M" not in df.columns:
        df = add_kst_mass(df)

    kst_min, kst_max = get_kst_window()
    apply_jpsi  = JPSI_VETO  if apply_charm_vetoes else None
    apply_psi2s = PSI2S_VETO if apply_charm_vetoes else None

    df_sel = basic_selection(
        df, q2_min=Q2_MIN, q2_max=Q2_MAX,
        b0_m_min=B0_M_MIN, b0_m_max=B0_M_MAX,
        kst_m_min=kst_min, kst_m_max=kst_max,
        jpsi_veto=apply_jpsi, psi2s_veto=apply_psi2s,
        require_kst=True, verbose=True,
    )
    n = len(df_sel)
    print(f"{label} selected events: {n}")

    if n < 5:
        return {"label": label, "n": n, "peak": None, "diagnostic_only": True,
                "warning": "Too few events for LFT."}

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

    peak = global_peak_report(k_ell, amp, WCT_K_MIN, WCT_K_MAX,
                              kmin=2.0, snr_min=SNR_MIN,
                              prominence_min=PROMINENCE_MIN)
    diagnostic_only = n < MIN_EVENTS_FOR_LFT
    return {
        "label": label,
        "n": int(n),
        "peak": peak,
        "diagnostic_only": diagnostic_only,
        "missing_branches": missing,
    }


def main():
    sig = scan_channel("data_signal/*.root", "signal", apply_charm_vetoes=True)
    # For B0 -> J/psi K*, do NOT apply the J/psi veto (it would remove the channel).
    ctrl = scan_channel("data_control/*.root", "control_JpsiKst", apply_charm_vetoes=False)

    verdict = {}
    sig_peak = sig.get("peak")
    ctrl_peak = ctrl.get("peak")

    if sig.get("error") or ctrl.get("error"):
        verdict["status"] = "missing_data"
        verdict["note"] = (f"signal_error={sig.get('error')}, "
                           f"control_error={ctrl.get('error')}")
    elif not sig_peak:
        verdict["status"] = "no_signal_peak"
        verdict["note"] = "No significant peak in WCT band in signal channel."
    elif not ctrl_peak:
        verdict["status"] = "signal_only"
        verdict["note"] = "Signal has WCT-band peak, control does not. WCT-consistent."
        verdict["signal_k_peak"] = sig_peak["k_peak"]
    else:
        dk = abs(sig_peak["k_peak"] - ctrl_peak["k_peak"])
        verdict["delta_k_signal_control"] = float(dk)
        if sig_peak["in_band"] and ctrl_peak["in_band"] and dk < DELTA_K_MATCH:
            verdict["status"] = "control_matches"
            verdict["note"] = ("Control channel has matching peak in WCT band. "
                               "Likely detector/selection artifact, NOT WCT.")
        else:
            verdict["status"] = "signal_distinct_from_control"
            verdict["note"] = ("Signal peak does not match control. "
                               "Consistent with WCT-like structure.")
        verdict["signal_k_peak"] = sig_peak["k_peak"]
        verdict["control_k_peak"] = ctrl_peak["k_peak"]

    if sig.get("diagnostic_only") or ctrl.get("diagnostic_only"):
        verdict["caveat"] = ("One or both channels below MIN_EVENTS_FOR_LFT. "
                             "Treat verdict as diagnostic only.")

    print("\n=== Verdict ===")
    print(verdict)

    save_json({
        "signal": sig,
        "control": ctrl,
        "verdict": verdict,
        "delta_k_match_threshold": DELTA_K_MATCH,
    }, OUT_DIR / "control_compare.json")


if __name__ == "__main__":
    main()

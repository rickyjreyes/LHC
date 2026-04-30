# 07_sensitivity_scan.py

import numpy as np
import pandas as pd

from config import *
from lhcb_utils import (
    make_q2_spectrum,
    smooth_residual,
    log_fft_scan,
    peak_report
)


def inject_signal(q2, counts, A, k, phi):
    mod = 1 + A * np.cos(k * np.log(q2) + phi)
    return counts * mod


def run_injection(q2, counts, A, k_true, trials=50):
    recovered = []

    for _ in range(trials):
        phi = np.random.uniform(0, 2*np.pi)

        y = inject_signal(q2, counts, A, k_true, phi)

        baseline, residual = smooth_residual(q2, y)
        _, _, k, amp = log_fft_scan(q2, residual)

        rep = peak_report(k, amp, 2, 30)

        if rep:
            k_hat = rep["k_peak"]
            snr = rep["snr_like"]

            if abs(k_hat - k_true) < 1.0 and snr > 3:
                recovered.append(1)
            else:
                recovered.append(0)
        else:
            recovered.append(0)

    return np.mean(recovered)


def sensitivity_grid(q2, counts):
    k_vals = np.linspace(5, 25, 10)
    A_vals = np.linspace(0.01, 0.15, 10)

    results = []

    for k in k_vals:
        for A in A_vals:
            p = run_injection(q2, counts, A, k)
            results.append({
                "k": k,
                "A": A,
                "recovery_prob": p
            })
            print(f"k={k:.2f}, A={A:.3f} -> {p:.2f}")

    return pd.DataFrame(results)


def main():
    # load real spectrum from previous run
    df = pd.read_csv("outputs/q2_residuals.csv")

    q2 = df["q2_center"].values
    counts = df["counts"].values

    res = sensitivity_grid(q2, counts)
    res.to_csv("outputs/sensitivity_scan.csv", index=False)


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Charm-strength sweep.

Runs the strong mimicry test across multiple charm-tail strengths and noise levels.
This checks whether WCT begins falsely fitting charm tails only under certain regimes.

Default grid:
    charm_scale = [0.15, 0.25, 0.35, 0.50, 0.75]
    sigma       = [0.10, 0.15, 0.20]

Use lower n-mc first for speed, then raise.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run(cmd):
    print("\n$", " ".join(str(x) for x in cmd), flush=True)
    subprocess.check_call([str(x) for x in cmd])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="mimicry_sweep_results")
    p.add_argument("--n-mc", type=int, default=200)
    p.add_argument("--target-k", type=float, default=11.7)
    p.add_argument("--k-window", type=float, default=1.0)
    p.add_argument("--charm-scales", nargs="+", type=float, default=[0.15, 0.25, 0.35, 0.50, 0.75])
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.10, 0.15, 0.20])
    args = p.parse_args()

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    for charm_scale in args.charm_scales:
        for sigma in args.sigmas:
            tag = f"charm_{charm_scale:.2f}_sigma_{sigma:.2f}".replace(".", "p")
            sub = root / tag

            run([
                sys.executable,
                "12_run_strong_mimicry_test.py",
                "--out-dir", sub,
                "--n-mc", args.n_mc,
                "--target-k", args.target_k,
                "--k-window", args.k_window,
                "--charm-scale", charm_scale,
                "--sigma", sigma,
            ])

            summary = json.loads((sub / "mc_false_positive_summary.json").read_text(encoding="utf-8"))
            scan_stats = json.loads((sub / "scan_stats.json").read_text(encoding="utf-8"))

            rows.append({
                "charm_scale": charm_scale,
                "sigma": sigma,
                "n_mc": args.n_mc,
                "best_k_single": scan_stats["best_k"],
                "delta_chi2_vs_charm_single": scan_stats["delta_chi2_vs_charm"],
                "false_positive_rate_vs_charm": summary["false_positive_rate_vs_charm"],
                "false_positive_rate_vs_charm_and_k": summary["false_positive_rate_vs_charm_and_k"],
                "wct_preferred_over_charm_by_aic_rate": summary["wct_preferred_over_charm_by_aic_rate"],
                "wct_preferred_over_charm_by_bic_rate": summary["wct_preferred_over_charm_by_bic_rate"],
                "result_dir": str(sub),
            })

    df = pd.DataFrame(rows)
    out_csv = root / "mimicry_sweep_summary.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== Sweep summary ===")
    print(df.to_string(index=False))
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()

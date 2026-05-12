#!/usr/bin/env python
"""
Strong Hadronic Mimicry Injection Test

This is the next test after the 20-run smoke test.

It performs:
1. Generate a charm-only fake dataset.
2. Run WCT-vs-charm scan.
3. Run N Monte Carlo charm-only injections.
4. Compute false-positive rates:
   - WCT beats constant
   - WCT beats charm
   - WCT beats charm AND lands near target k
   - WCT preferred over charm by AIC/BIC
5. Write a PASS/FAIL report.

Default target:
    k_target = 11.7
    n_mc = 1000

Interpretation:
    PASS if WCT does not falsely discover charm-only data.
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


def verdict(summary, max_fp_charm, max_fp_charm_k, max_bic_rate):
    fp_charm = summary["false_positive_rate_vs_charm"]
    fp_charm_k = summary["false_positive_rate_vs_charm_and_k"]
    bic_rate = summary["wct_preferred_over_charm_by_bic_rate"]

    passed = (
        fp_charm <= max_fp_charm and
        fp_charm_k <= max_fp_charm_k and
        bic_rate <= max_bic_rate
    )

    if passed:
        return "PASS"
    return "FAIL"


def make_report(out_dir, summary, scan_stats, fit_table, args):
    status = verdict(
        summary,
        max_fp_charm=args.max_fp_charm,
        max_fp_charm_k=args.max_fp_charm_k,
        max_bic_rate=args.max_bic_rate,
    )

    report = []
    report.append("# Strong Hadronic Mimicry Injection Test Report\n")
    report.append(f"Verdict: **{status}**\n")
    report.append("\n## Configuration\n")
    report.append(f"- Monte Carlo injections: `{args.n_mc}`\n")
    report.append(f"- Target k: `{args.target_k}`\n")
    report.append(f"- k-window: `±{args.k_window}`\n")
    report.append(f"- k scan range: `{args.k_min}` to `{args.k_max}`\n")
    report.append(f"- Charm fake bins: `{args.n_bins}`\n")
    report.append(f"- Injection sigma: `{args.sigma}`\n")
    report.append(f"- Charm scale: `{args.charm_scale}`\n")

    report.append("\n## Main single-dataset fit\n")
    report.append(f"- delta chi2 vs constant: `{scan_stats['delta_chi2_vs_const']}`\n")
    report.append(f"- delta chi2 vs charm: `{scan_stats['delta_chi2_vs_charm']}`\n")
    report.append(f"- best WCT k: `{scan_stats['best_k']}`\n")
    report.append(f"- WCT AIC minus charm AIC: `{scan_stats['wct_aic_minus_charm_aic']}`\n")
    report.append(f"- WCT BIC minus charm BIC: `{scan_stats['wct_bic_minus_charm_bic']}`\n")

    report.append("\n## Monte Carlo false-positive rates\n")
    for k, v in summary.items():
        report.append(f"- `{k}`: `{v}`\n")

    report.append("\n## Pass criteria\n")
    report.append(f"- false_positive_rate_vs_charm <= `{args.max_fp_charm}`\n")
    report.append(f"- false_positive_rate_vs_charm_and_k <= `{args.max_fp_charm_k}`\n")
    report.append(f"- wct_preferred_over_charm_by_bic_rate <= `{args.max_bic_rate}`\n")

    report.append("\n## Interpretation\n")
    if status == "PASS":
        report.append(
            "WCT did not repeatedly mistake charm-only Breit-Wigner tail pseudo-data "
            "for the target log-periodic signal. This reduces the charm-mimicry failure mode.\n"
        )
    else:
        report.append(
            "WCT produced too many charm-only false positives. The ansatz is too flexible "
            "or the null model is insufficient. Do not proceed to discovery claims until fixed.\n"
        )

    report.append("\n## Fit comparison table\n\n")
    report.append(fit_table.to_markdown(index=False))
    report.append("\n")

    (out_dir / "STRONG_MIMICRY_REPORT.md").write_text("".join(report), encoding="utf-8")
    print(f"\nWrote report: {out_dir / 'STRONG_MIMICRY_REPORT.md'}")
    print(f"Verdict: {status}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="strong_mimicry_results")
    p.add_argument("--n-mc", type=int, default=1000)
    p.add_argument("--target-k", type=float, default=11.7)
    p.add_argument("--k-window", type=float, default=1.0)
    p.add_argument("--k-min", type=float, default=2.0)
    p.add_argument("--k-max", type=float, default=25.0)
    p.add_argument("--n-bins", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--charm-scale", type=float, default=0.35)

    # Strict pass criteria. Change these only if intentionally loosening.
    p.add_argument("--max-fp-charm", type=float, default=0.01)
    p.add_argument("--max-fp-charm-k", type=float, default=0.002)
    p.add_argument("--max-bic-rate", type=float, default=0.05)

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fake_csv = out_dir / "fake_charm.csv"

    run([
        sys.executable,
        "07_inject_charm_mimicry.py",
        "--out", fake_csv,
        "--n-bins", args.n_bins,
        "--seed", args.seed,
        "--sigma", args.sigma,
        "--charm-scale", args.charm_scale,
    ])

    run([
        sys.executable,
        "08_scan_wct_on_injections.py",
        "--input", fake_csv,
        "--n-mc", args.n_mc,
        "--target-k", args.target_k,
        "--k-window", args.k_window,
        "--k-min", args.k_min,
        "--k-max", args.k_max,
        "--out-dir", out_dir,
    ])

    summary_path = out_dir / "mc_false_positive_summary.json"
    scan_path = out_dir / "scan_stats.json"
    fit_path = out_dir / "fit_comparison.csv"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scan_stats = json.loads(scan_path.read_text(encoding="utf-8"))
    fit_table = pd.read_csv(fit_path)

    make_report(out_dir, summary, scan_stats, fit_table, args)


if __name__ == "__main__":
    main()

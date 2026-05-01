"""
Run the full LHCb WCT analysis pipeline.

HARD scripts must succeed (selection / spectrum / bootstrap / cut sweep).
SOFT scripts may fail without aborting (control comparison if data folders
are absent, P5 placeholder if angles are missing, sensitivity scan).
"""
import subprocess
import sys

HARD_SCRIPTS = [
    "00_inspect_root.py",
    "01_check_branches.py",
    "02_lft_wct_scan.py",          # primary WCT test (log-Fourier on ell)
    "03_fft_control_scan.py",      # artifact diagnostic (linear-q^2 FFT)
    "04_bootstrap_lft.py",         # event-resampling + null shuffle
    "05_cut_sweep.py",             # raw_q2 / loose / medium / tight
    "04_angle_branch_report.py",   # diagnostic for P5 readiness
]

SOFT_SCRIPTS = [
    "07_control_compare.py",       # data_signal/ vs data_control/
    "06_p5_placeholder.py",        # disabled until angles present
]


def run(script, hard):
    print("\n" + "=" * 80)
    print(f"RUNNING [{'HARD' if hard else 'SOFT'}] {script}")
    print("=" * 80)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        msg = f"\n{'FAILED' if hard else 'WARNING'}: {script} exited with code {result.returncode}"
        print(msg)
        if hard:
            sys.exit(result.returncode)


def main():
    for s in HARD_SCRIPTS:
        run(s, hard=True)
    for s in SOFT_SCRIPTS:
        run(s, hard=False)
    print("\nAll done. Check outputs/")


if __name__ == "__main__":
    main()

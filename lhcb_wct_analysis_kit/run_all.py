import subprocess
import sys

SCRIPTS = [
    "00_inspect_root.py",
    "01_check_branches.py",
    "02_q2_wct_scan.py",
    "03_bootstrap_scan.py",
    "04_angle_branch_report.py",
    "06_p5_placeholder.py",
]

def run(script):
    print("\n" + "=" * 80)
    print("RUNNING", script)
    print("=" * 80)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\nFAILED: {script}")
        sys.exit(result.returncode)

def main():
    for s in SCRIPTS:
        run(s)
    print("\nAll done. Check outputs/")

if __name__ == "__main__":
    main()

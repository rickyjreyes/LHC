#!/usr/bin/env python
"""
Runs a complete demo:
1. generate fake charm-only data
2. fit constant/charm/WCT models
3. run a small MC false-positive test
"""

import subprocess
import sys
from pathlib import Path

cmds = [
    [sys.executable, "07_inject_charm_mimicry.py", "--out", "fake_charm.csv", "--n-bins", "16", "--seed", "1"],
    [sys.executable, "08_scan_wct_on_injections.py", "--input", "fake_charm.csv", "--n-mc", "20", "--target-k", "11.7", "--out-dir", "demo_results"],
]

for cmd in cmds:
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd)

print("\nDone. See fake_charm.csv and demo_results/")

# Conservative P5 angular diagnostic.
# This is not a publication-grade P5 likelihood fit.
# P5_READY = False until cosThetaL, cosThetaK, phi exist as branches
# (or are reconstructed from four-vectors). Per spec, this script remains
# a no-op when angles are absent.

import pandas as pd
import matplotlib.pyplot as plt

from config import *
from lhcb_utils import load_dataframe, add_q2, add_kst_mass, basic_selection

REQUESTED = [
    "B0_M",
    "muplus_PX", "muplus_PY", "muplus_PZ", "muplus_PE",
    "muminus_PX", "muminus_PY", "muminus_PZ", "muminus_PE",
    "Kplus_PX", "Kplus_PY", "Kplus_PZ", "Kplus_PE",
    "piminus_PX", "piminus_PY", "piminus_PZ", "piminus_PE",
    "Kst_M",
    "cosThetaL", "cosThetaK", "phi",
]

def main():
    df, tree, missing = load_dataframe(FILES_GLOB, TREE_NAME, REQUESTED)

    # Bail early if angles are missing - no need to run selection.
    needed = ["cosThetaL", "cosThetaK", "phi"]
    missing_angles = [x for x in needed if x not in df.columns]
    if missing_angles:
        print("P5_READY = False")
        print("Missing angle branches:", missing_angles)
        print("Reconstruct angles from four-vectors before running P5 fit, "
              "or see 04_angle_branch_report.py for available angle-like branches.")
        return

    df = add_q2(df)
    if "Kst_M" not in df.columns:
        df = add_kst_mass(df)
    df = basic_selection(df, require_kst=True, verbose=False)

    OUT_DIR.mkdir(exist_ok=True)

    for col in needed:
        plt.figure(figsize=(7, 4))
        plt.hist(df[col], bins=50)
        plt.xlabel(col)
        plt.ylabel("Events")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{col}_hist.png", dpi=200)
        plt.close()

    qbins = [0.1, 2.0, 4.0, 6.0, 8.0, 11.0, 12.5, 15.0, 19.0]
    df["q2_bin"] = pd.cut(df["q2"], qbins)
    summary = df.groupby("q2_bin")[needed].agg(["mean", "std", "count"])
    summary.to_csv(OUT_DIR / "angle_q2_summary.csv")

    print(f"Saved angle diagnostics in {OUT_DIR}")

if __name__ == "__main__":
    main()

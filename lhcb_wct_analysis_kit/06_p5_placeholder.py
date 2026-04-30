# Conservative P5 angular diagnostic.
# This is not a publication-grade P5 likelihood fit.

import pandas as pd
import matplotlib.pyplot as plt

from config import *
from lhcb_utils import load_dataframe, add_q2, basic_selection

REQUESTED = [
    "B0_M",
    "muplus_PX", "muplus_PY", "muplus_PZ", "muplus_PE",
    "muminus_PX", "muminus_PY", "muminus_PZ", "muminus_PE",
    "Kst_M",
    "cosThetaL", "cosThetaK", "phi",
]

def main():
    df, tree, missing = load_dataframe(FILES_GLOB, TREE_NAME, REQUESTED)
    df = add_q2(df)
    df = basic_selection(df)

    needed = ["cosThetaL", "cosThetaK", "phi"]
    missing_angles = [x for x in needed if x not in df.columns]
    if missing_angles:
        print("Missing angle branches:", missing_angles)
        print("P5 fit not ready. Use 04_angle_branch_report.py or compute angles from 4-vectors.")
        return

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

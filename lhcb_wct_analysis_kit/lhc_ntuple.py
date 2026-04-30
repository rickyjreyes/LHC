# check_lhcb_ntuple_branches.py
# pip install uproot pandas

import glob
import uproot

FILES = "data/*.root"

REQUIRED_FOR_Q2_SCAN = [
    "muplus_PX", "muplus_PY", "muplus_PZ", "muplus_PE",
    "muminus_PX", "muminus_PY", "muminus_PZ", "muminus_PE",
    "B0_M",
]

REQUIRED_FOR_P5 = [
    "cosThetaL",
    "cosThetaK",
    "phi",
]

USEFUL_EXTRA = [
    "Kplus_PX", "Kplus_PY", "Kplus_PZ", "Kplus_PE",
    "piminus_PX", "piminus_PY", "piminus_PZ", "piminus_PE",
    "Kst_M",
    "B0_PX", "B0_PY", "B0_PZ", "B0_PE",
    "eventNumber",
    "runNumber",
]


def find_trees(root_file):
    trees = []
    with uproot.open(root_file) as f:
        for key, obj in f.items(recursive=True):
            if isinstance(obj, uproot.behaviors.TTree.TTree):
                trees.append(key)
    return trees


def branch_match(branches, target):
    """
    Exact match first. If unavailable, find close names containing target fragments.
    """
    if target in branches:
        return target

    t = target.lower()
    candidates = [b for b in branches if t in b.lower()]
    return candidates


def inspect():
    files = glob.glob(FILES)
    if not files:
        raise FileNotFoundError("Put downloaded ROOT files in ./data first.")

    path = files[0]
    print(f"\nInspecting: {path}")

    trees = find_trees(path)
    print("\nAvailable trees:")
    for t in trees:
        print("  ", t)

    if not trees:
        raise RuntimeError("No TTrees found.")

    tree_name = trees[0]
    print(f"\nUsing tree: {tree_name}")

    with uproot.open(path) as f:
        tree = f[tree_name]
        branches = list(tree.keys())

    print(f"\nNumber of branches: {len(branches)}")

    print("\n=== Required for q^2 scan ===")
    q2_ok = True
    for b in REQUIRED_FOR_Q2_SCAN:
        m = branch_match(branches, b)
        if m == b:
            print(f"OK      {b}")
        else:
            print(f"MISSING {b}   close={m[:10] if isinstance(m, list) else m}")
            q2_ok = False

    print("\n=== Required for P'_5 angular analysis ===")
    p5_ok = True
    for b in REQUIRED_FOR_P5:
        m = branch_match(branches, b)
        if m == b:
            print(f"OK      {b}")
        else:
            print(f"MISSING {b}   close={m[:10] if isinstance(m, list) else m}")
            p5_ok = False

    print("\n=== Useful extra branches ===")
    for b in USEFUL_EXTRA:
        m = branch_match(branches, b)
        if m == b:
            print(f"OK      {b}")
        else:
            print(f"MISSING {b}   close={m[:10] if isinstance(m, list) else m}")

    print("\n=== Verdict ===")
    if q2_ok:
        print("q^2 spectral / WCT log-frequency scan: READY")
    else:
        print("q^2 spectral / WCT log-frequency scan: NOT READY")

    if p5_ok:
        print("P'_5 angular analysis: READY")
    else:
        print("P'_5 angular analysis: NOT READY unless angles are computed from 4-vectors.")

    print("\nSuggested tree path for analysis script:")
    print(tree_name)


if __name__ == "__main__":
    inspect()
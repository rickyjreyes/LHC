from config import FILES_GLOB, TREE_NAME, OUT_DIR
from lhcb_utils import find_root_files, choose_tree, resolve_branch, save_json
import uproot

REQUIRED_FOR_Q2_SCAN = [
    "muplus_PX", "muplus_PY", "muplus_PZ", "muplus_PE",
    "muminus_PX", "muminus_PY", "muminus_PZ", "muminus_PE",
    "B0_M",
]

REQUIRED_FOR_P5 = ["cosThetaL", "cosThetaK", "phi"]

USEFUL_EXTRA = [
    "Kplus_PX", "Kplus_PY", "Kplus_PZ", "Kplus_PE",
    "piminus_PX", "piminus_PY", "piminus_PZ", "piminus_PE",
    "Kst_M", "B0_PX", "B0_PY", "B0_PZ", "B0_PE",
    "eventNumber", "runNumber", "nCandidate", "totCandidates",
]

def status(branches, names):
    return {name: resolve_branch(branches, name, required=False) for name in names}

def print_group(title, results):
    print(f"\n=== {title} ===")
    ok = True
    for name, found in results.items():
        if found:
            print(f"OK      {name} -> {found}")
        else:
            print(f"MISSING {name}")
            ok = False
    return ok

def main():
    files = find_root_files(FILES_GLOB)
    path = files[0]
    tree_name = choose_tree(path, preferred=TREE_NAME)

    with uproot.open(path) as f:
        tree = f[tree_name]
        branches = list(tree.keys())

    q2 = status(branches, REQUIRED_FOR_Q2_SCAN)
    p5 = status(branches, REQUIRED_FOR_P5)
    extra = status(branches, USEFUL_EXTRA)

    print(f"File: {path}")
    print(f"Tree: {tree_name}")
    print(f"Branch count: {len(branches)}")

    q2_ok = print_group("Required for q2 scan", q2)
    p5_ok = print_group("Required for P5 angular analysis", p5)
    print_group("Useful extras", extra)

    verdict = {
        "file": path,
        "tree": tree_name,
        "q2_ready": q2_ok,
        "p5_ready": p5_ok,
        "q2_branches": q2,
        "p5_branches": p5,
        "extra_branches": extra,
    }
    save_json(verdict, OUT_DIR / "branch_report.json")

    print("\n=== Verdict ===")
    print("q2 WCT scan:", "READY" if q2_ok else "NOT READY")
    print("P5 angular analysis:", "READY" if p5_ok else "NOT READY unless angles are computed from 4-vectors")
    print(f"\nSaved: {OUT_DIR / 'branch_report.json'}")
    print(f"\nUse this TREE_NAME in config.py if needed:\n{tree_name}")

if __name__ == "__main__":
    main()

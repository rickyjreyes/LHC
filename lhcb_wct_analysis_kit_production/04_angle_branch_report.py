from config import FILES_GLOB, TREE_NAME, OUT_DIR
from lhcb_utils import find_root_files, choose_tree, branch_candidates, save_json
import uproot

ANGLE_TERMS = [["costheta"], ["theta"], ["phi"], ["angle"], ["hel"]]

def main():
    files = find_root_files(FILES_GLOB)
    path = files[0]
    tree_name = choose_tree(path, preferred=TREE_NAME)

    with uproot.open(path) as f:
        tree = f[tree_name]
        branches = list(tree.keys())

    report = {}
    print(f"Tree: {tree_name}")
    print("\nPossible angle branches:")

    for terms in ANGLE_TERMS:
        hits = branch_candidates(branches, terms)
        report["+".join(terms)] = hits
        print(f"\nTerms {terms}:")
        for h in hits[:50]:
            print(" ", h)

    save_json(report, OUT_DIR / "angle_branch_candidates.json")
    print(f"\nSaved: {OUT_DIR / 'angle_branch_candidates.json'}")

if __name__ == "__main__":
    main()

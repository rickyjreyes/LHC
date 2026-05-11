import uproot
from config import FILES_GLOB
from lhcb_utils import find_root_files, find_ttrees

def main():
    files = find_root_files(FILES_GLOB)
    path = files[0]
    print(f"Inspecting first ROOT file: {path}")

    with uproot.open(path) as f:
        print("\nTop-level keys:")
        for k in f.keys():
            print(" ", k)

    trees = find_ttrees(path)
    print("\nTTrees found:")
    for t in trees:
        print(" ", t)

    if trees:
        with uproot.open(path) as f:
            tree = f[trees[0]]
            print(f"\nFirst tree: {trees[0]}")
            print(f"Branches ({len(tree.keys())}):")
            for b in tree.keys():
                print(" ", b)

if __name__ == "__main__":
    main()

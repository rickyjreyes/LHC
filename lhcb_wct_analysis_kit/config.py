from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

FILES_GLOB = str(DATA_DIR / "*.root")
TREE_NAME = "B0_KstMuMu/DecayTree"

Q2_MIN = 0.1
Q2_MAX = 19.0

B0_M_MIN = 5100
B0_M_MAX = 5600
KST_M_MIN = 750
KST_M_MAX = 1100

JPSI_VETO = (8.0, 11.0)
PSI2S_VETO = (12.5, 15.0)

Q2_BINS = 60
LOG_GRID_N = 512
SMOOTH_WINDOW = 11
SMOOTH_POLY = 3

WCT_K_MIN = 8.0
WCT_K_MAX = 20.0

BOOTSTRAP_N = 200
RANDOM_SEED = 12345

from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

FILES_GLOB = str(DATA_DIR / "*.root")
TREE_NAME = "B0_KstMuMu/DecayTree"

# Reference q^2 scale for ell = ln(q^2 / Q2_REF). 1 GeV^2 keeps ell dimensionless.
Q2_REF = 1.0

Q2_MIN = 0.1
Q2_MAX = 19.0

B0_M_MIN = 5100
B0_M_MAX = 5600

# K*(892) window. Use loose-mode constants below for diagnostic exploration.
KST_M_MIN = 792.0
KST_M_MAX = 992.0
KST_M_MIN_LOOSE = 700.0
KST_M_MAX_LOOSE = 1100.0
KST_MODE = "tight"   # "tight" | "loose"

# Charmonium vetoes. Mode controls how the LFT handles the gaps:
#   "none"        -> diagnostic only, no veto applied
#   "mask"        -> drop events in veto windows, FFT continuous spectrum
#   "segment_lft" -> split log-spectrum at veto edges, FFT each segment, combine
JPSI_VETO  = (8.0, 11.0)
PSI2S_VETO = (12.5, 15.0)
VETO_MODE  = "mask"

# LFT binning. N_LFT_BINS is the canonical name; Q2_BINS kept for back-compat.
N_LFT_BINS = 60
Q2_BINS    = N_LFT_BINS

# FFT (linear-q^2) binning for the artifact/control scan.
# Same default as N_LFT_BINS so power spectra are directly comparable.
N_FFT_BINS = 60

LOG_GRID_N    = 512
SMOOTH_WINDOW = 11
SMOOTH_POLY   = 3

# Baseline mode
#   "savgol" -> Savitzky-Golay over ell-binned counts (legacy)
#   "floor"  -> max(smoothed_counts, BASELINE_FLOOR), Poisson-safe
#   "kde"    -> Gaussian KDE on ell, scaled to expected counts per bin
BASELINE_MODE  = "kde"
BASELINE_FLOOR = 0.5

# WCT log-frequency band of interest
WCT_K_MIN    = 8.0
WCT_K_MAX    = 20.0
WCT_K_TARGET = 12.0   # central WCT prediction (informational, not used as a cut)

# Significance gates for peak_report / global_peak_report
SNR_MIN        = 3.0
PROMINENCE_MIN = 1.0   # in units of 1.4826 * MAD (robust sigma)

# Bootstrap configuration
BOOTSTRAP_N        = 200
RANDOM_SEED        = 12345
NULL_BOOTSTRAP_N   = 200   # shuffle-residual null trials for false-positive rate
MIN_EVENTS_FOR_LFT = 100   # below this, results labeled diagnostic only

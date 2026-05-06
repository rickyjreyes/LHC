# LHCb WCT Analysis Kit

Target:

B0 -> (K*(892)0 -> K+ pi-) mu+ mu-

Put ROOT files in:

data/*.root

Install:

pip install uproot awkward numpy pandas scipy matplotlib

Run:

python run_all.py

Main outputs:

outputs/branch_report.json
outputs/q2_residuals.csv
outputs/log_fft.csv
outputs/q2_spectrum.png
outputs/residual.png
outputs/log_fft.png
outputs/bootstrap_peaks.csv
outputs/bootstrap_summary.json

Interpretation:

- q2 scan ready if muon four-vectors and B0_M exist.
- P5 angular analysis ready only if cosThetaL, cosThetaK, phi exist.
- WCT candidate requires a stable peak in k_l in [8,20], high SNR-like, and no matching control-channel artifact.

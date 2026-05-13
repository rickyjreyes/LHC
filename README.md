# LHCb / WCT Analysis Kit

This repository contains a reproducible analysis pipeline for studying log-periodic residual structure, active-domain winding, Koide-like comb geometry, sideband behavior, charm-veto stability, and control-channel behavior in open-data `B0 -> K*0 mu+ mu-` candidate spectra.

The project focuses on one central question:

> Does the selected LHCb candidate spectrum contain stable structure in logarithmic momentum-transfer space, and does that structure organize naturally in active-domain winding variables?

The main log-domain variable is:

```text
ell = ln(q2)
```

where `q2` is the dimuon invariant mass squared.

The tested residual family is:

```text
A cos(k ell + phi)
```

where `A` is amplitude, `k` is log-frequency, and `phi` is phase.

The analysis kit includes ROOT-file inspection, branch validation, q2 reconstruction, KDE-baseline repair, bounded-Poisson log-cos tests, active-domain integer-winding scans, Koide/trig comb scans, sideband-subtracted tests, charm-trimmed continuum tests, veto-window covariance tests, and optional angular reconstruction diagnostics.

---

## Repository target

Primary decay channel:

```text
B0 -> K*(892)0 mu+ mu-
K*(892)0 -> K+ pi-
```

Primary input format:

```text
LHCb ROOT / dvntuple ROOT files
```

Primary analysis variable:

```text
q2 = m(mu+ mu-)^2
ell = ln(q2)
```

Primary spectral map:

```text
n = k Delta ell_A / (2 pi)
```

where `Delta ell_A` is the retained active log-domain length after charmonium vetoes.

---

## Directory layout

Expected input folders:

```text
data/                  Main B0 -> K*0 mu+ mu- ROOT files
data_control/          Optional independent control-channel ROOT files
```

Generated output folders include:

```text
outputs/
outputs_run_all/
outputs_angles/
outputs_angular_logcos/
outputs_logcos_poisson_twomode_kde_polar/
outputs_wct_integer_winding/
outputs_wct_koide_comb/
outputs_wct_vs_smqft/
outputs_wct_sideband_subtracted/
outputs_sideband_subtracted/
outputs_charm_trimmed_control/
outputs_wct_well_first_koide/
outputs_wct_well_proof/
outputs_wct_cross_region_scaling/
outputs_wct_cross_region_stability/
outputs_wct_locked_branch_amplitude_ladder/
outputs_wct_veto_covariance/
outputs_control_blind/
```

---

## Installation

Create and activate a Python environment, then install the required packages:

```bash
pip install uproot awkward numpy pandas scipy matplotlib
```

For GPU-accelerated scripts, install CuPy matching your CUDA version. Example:

```bash
pip install cupy-cuda12x
```

If CuPy is not available, several scripts either fall back to CPU mode or report that GPU acceleration is unavailable.

---

## Quick start

Inspect the pipeline plan without running stages:

```bash
python run_all.py --dry-run
```

Run a reduced diagnostic pass:

```bash
python run_all.py --fast --continue-on-error
```

Run the default yield-side pipeline:

```bash
python run_all.py --continue-on-error
```

Run the full pipeline including optional and overlap stages:

```bash
python run_all.py --full --controls --continue-on-error
```

Run selected stages only:

```bash
python run_all.py --only 09d,12,13 --continue-on-error
```

Run sideband and charm-trimmed controls:

```bash
python run_all.py --only 28,29 --n-null 100 --continue-on-error
```

Run stronger null statistics for stages 28 and 29:

```bash
python run_all.py --only 28,29 --n-null 1000 --continue-on-error
```

---

## Windows UTF-8 note

On Windows PowerShell, set UTF-8 before long runs to avoid encoding crashes on symbols such as `≈`, `Δ`, and `θ`:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
python run_all.py --continue-on-error
```

The current runner also sets UTF-8 for subprocesses, but setting the environment explicitly is useful for long scripted runs.

---

## Pipeline overview

The pipeline is organized into five layers:

1. Intake and branch validation
2. Yield-side log-cos and winding tests
3. Koide/trig comb and WCT-vs-smooth comparisons
4. Sideband, charm, control, and veto-covariance tests
5. Optional angular reconstruction and angular log-cos diagnostics

---

## Intake stages

| Key | Script | Purpose |
|---:|---|---|
| `00` | `00_inspect_root.py` | Inspect the first ROOT file and list available trees and branches. |
| `01` | `01_check_branches.py` | Check q2, four-vector, and angular branch readiness. |
| `04` | `04_angle_branch_report.py` | Search for direct angular branch candidates. |

Typical use:

```bash
python run_all.py --only 00,01,04 --continue-on-error
```

---

## Yield-side repaired log-cos pipeline

| Key | Script | Purpose |
|---:|---|---|
| `09d` | `09d_two_mode_kde_baseline_polar_cupy.py` | Final repaired KDE-baseline bounded-Poisson two-mode scan. |
| `12` | `12_wct_integer_winding_scan.py` | Discrete active-domain integer-winding scan. |
| `13` | `13_wct_koide_trig_comb_scan_cupy.py` | Koide/trig comb scan over active-domain winding ratios. |
| `16` | `16_wct_vs_smqft_likelihood_test_cupy.py` | Compare smooth SM/QFT-like nulls against WCT comb alternatives. |
| `17` | `17_wct_sideband_subtracted_comb_test_cupy.py` | Sideband-subtracted WCT comb diagnostic. |

Typical use:

```bash
python run_all.py --only 09d,12,13,16,17 --continue-on-error
```

---

## Well-first and cross-region diagnostics

| Key | Script | Purpose |
|---:|---|---|
| `19` | `19_koide_well.py` | Find raw spectral wells before imposing a Koide comb. |
| `20` | `20_koide_proof.py` | Null/proof-style test of Koide-like geometry in raw wells. |
| `21` | `21_cross_region_scaling_phase_test.py` | Test cross-region dilation and phase coherence. |
| `22` | `22_cross_regional_stability_test.py` | Sweep top-well count and region-pair direction stability. |
| `24` | `24_locked_branch_amplitude.py` | Locked-branch amplitude-cap ladder. |

Typical use:

```bash
python run_all.py --only 19,20,21,22,24 --continue-on-error
```

---

## Veto covariance and active-domain invariance

| Key | Script | Purpose |
|---:|---|---|
| `25` | `25_veto_window_covariance_test.py` | Main veto-window covariance / active-domain invariance test. |
| `26` | `26_veto_covariance.py` | Alternate/report-style veto covariance implementation. |

Typical use:

```bash
python run_all.py --only 25 --continue-on-error
```

Stage `25` is the preferred active-domain covariance test. Stage `26` is useful as an alternate implementation or comparison report.

---

## Control-channel and blind-control stages

| Key | Script | Purpose |
|---:|---|---|
| `27` | `27_control_channel_blind_test.py` | Blind control-channel or reconstruction-control test. |
| `28` | `28_sideband.py` | Sideband-subtracted residual WLS control using the same active support and k-to-n map. |
| `29` | `29_charm_tail_subtraction_control.py` | Charm-trimmed continuum control: remove J/psi and psi(2S), then test the remaining continuum. |

Typical use:

```bash
python run_all.py --only 27,28,29 --controls --continue-on-error
```

For stronger null statistics:

```bash
python run_all.py --only 28,29 --n-null 1000 --continue-on-error
```

---

## Angular pipeline

The angular stages derive angular observables from final-state four-vectors and then scan angular moments for log-cos structure.

| Key | Script | Purpose |
|---:|---|---|
| `10` | `10_compute_angles.py` | Compute q2, K* mass, cosThetaL, cosThetaK, and phi from four-vectors. |
| `11` | `11_angular_logcos_scan.py` | Angular moment / P5-proxy log-cos scan. |

Run angular stages:

```bash
python run_all.py --angular --continue-on-error
```

Or run them directly:

```bash
python run_all.py --only 10,11 --continue-on-error
```

Expected angular outputs:

```text
outputs_angles/angles.parquet
outputs_angles/angles.csv
outputs_angles/angle_summary.json
outputs_angles/angle_distributions.png
outputs_angular_logcos/angular_moments.csv
```

---

## Main outputs

The exact outputs depend on the selected stages. Common outputs include:

```text
outputs/branch_report.json
outputs/angle_branch_candidates.json

outputs_logcos_poisson_twomode_kde_polar/two_mode_summary.json
outputs_logcos_poisson_twomode_kde_polar/two_mode_scan_mask.csv
outputs_logcos_poisson_twomode_kde_polar/two_mode_null_mask.csv
outputs_logcos_poisson_twomode_kde_polar/two_mode_scan_mask.png
outputs_logcos_poisson_twomode_kde_polar/two_mode_fit_mask.png

outputs_wct_integer_winding/integer_winding_summary.csv
outputs_wct_integer_winding/integer_winding_summary.json
outputs_wct_integer_winding/integer_winding_null.csv
outputs_wct_integer_winding/integer_winding_scores.png

outputs_wct_koide_comb/koide_comb_summary.csv
outputs_wct_koide_comb/koide_comb_summary.json
outputs_wct_koide_comb/koide_comb_null.csv

outputs_wct_vs_smqft/wct_vs_smqft_summary.csv
outputs_wct_vs_smqft/wct_vs_smqft_summary.json

outputs_wct_sideband_subtracted/sideband_subtracted_summary.csv
outputs_wct_sideband_subtracted/sideband_subtracted_summary.json

outputs_wct_well_first_koide/well_first_scan_curve.csv
outputs_wct_well_first_koide/well_first_wells.csv
outputs_wct_well_first_koide/well_first_triplets.csv
outputs_wct_well_first_koide/well_first_summary.json

outputs_wct_cross_region_scaling/cross_region_scaling_summary.json
outputs_wct_cross_region_stability/stability_table.csv
outputs_wct_cross_region_stability/stability_summary.json

outputs_wct_veto_covariance/veto_covariance_wells.csv
outputs_wct_veto_covariance/veto_covariance_triplets.csv
outputs_wct_veto_covariance/veto_covariance_summary.csv
outputs_wct_veto_covariance/veto_covariance_summary.json

outputs_control_blind/control_summary.json
outputs_control_blind/control_blind_verdict.json
```

---

## Active-domain winding

The central frequency-to-winding map is:

```text
n = k Delta ell_A / (2 pi)
```

For the widened-veto active support,

```text
A = (0.1, 8.0) union (11.0, 12.5) union (14.5, 19.0)
```

the active log-domain length is:

```text
Delta ell_A = ln(8.0 / 0.1) + ln(12.5 / 11.0) + ln(19.0 / 14.5)
            = 4.7801503359
```

The corresponding integer grid is:

```text
k_n = 2 pi n / Delta ell_A
```

Useful reference values:

| n | k_n |
|---:|---:|
| 10 | 13.1443 |
| 12 | 15.7732 |
| 14 | 18.4021 |
| 15 | 19.7165 |
| 16 | 21.0309 |
| 18 | 23.6598 |
| 20 | 26.2887 |

---

## Koide / trig comb model

The comb scan tests active-domain winding triplets of the form:

```text
(n_minus, n0, n_plus) = n0 * (Q, 1, 2Q)
```

with central winding:

```text
n0 = 15
```

For the charged-lepton Koide value:

```text
Q = 2 / 3
```

the active-domain comb becomes:

```text
(n_minus, n0, n_plus) = (10, 15, 20)
```

Other scanned values include folded, spin-like, quark-like, and dense local values around the Koide region.

---

## Charmonium vetoes

The main yield-side pipeline uses widened charmonium vetoes:

```text
J/psi:  8.0  <= q2 <= 11.0
psi(2S): 12.5 <= q2 <= 14.5
```

These vetoes define the active support used in the integer-winding and comb scans.

The veto-covariance stage varies these windows and checks whether structure is more stable in active-domain winding `n` than in raw frequency `k`.

---

## Data requirements

The minimal q2 yield-side pipeline requires the muon four-vector branches:

```text
muplus_PX
muplus_PY
muplus_PZ
muplus_PE
muminus_PX
muminus_PY
muminus_PZ
muminus_PE
```

Mass-window selection uses:

```text
B0_M
Kst_M
```

or compatible branch names discovered by the scripts.

The angular pipeline additionally requires final-state four-vectors for:

```text
B0
K+
pi-
mu+
mu-
```

or enough final-state information to reconstruct the B0 candidate.

---

## Configuration

The base configuration lives in:

```text
config.py
```

Common settings include:

```text
DATA_DIR
OUT_DIR
FILES_GLOB
TREE_NAME
Q2_MIN
Q2_MAX
B0_M_MIN
B0_M_MAX
KST_M_MIN
KST_M_MAX
JPSI_VETO
PSI2S_VETO
Q2_BINS
WCT_K_MIN
WCT_K_MAX
BOOTSTRAP_N
RANDOM_SEED
```

Most later-stage scripts also contain local configuration blocks for scan ranges, null counts, amplitude caps, bandwidth ladders, and output folders.

---

## Common commands

Inspect ROOT structure:

```bash
python 00_inspect_root.py
```

Check required branches:

```bash
python 01_check_branches.py
```

Run the repaired two-mode KDE-baseline scan:

```bash
python 09d_two_mode_kde_baseline_polar_cupy.py
```

Run integer-winding scan:

```bash
python 12_wct_integer_winding_scan.py
```

Run Koide/trig comb scan:

```bash
python 13_wct_koide_trig_comb_scan_cupy.py
```

Run WCT-vs-smooth likelihood comparison:

```bash
python 16_wct_vs_smqft_likelihood_test_cupy.py
```

Run sideband-subtracted comb diagnostic:

```bash
python 17_wct_sideband_subtracted_comb_test_cupy.py
```

Run well-first Koide scan:

```bash
python 19_koide_well.py
```

Run veto-window covariance test:

```bash
python 25_veto_window_covariance_test.py
```

Run blind control-channel test:

```bash
python 27_control_channel_blind_test.py
```

Run angular reconstruction:

```bash
python 10_compute_angles.py
```

Run angular log-cos scan:

```bash
python 11_angular_logcos_scan.py
```

---

## Suggested run order

For a clean full analysis pass:

```bash
python run_all.py --dry-run
python run_all.py --fast --continue-on-error
python run_all.py --continue-on-error
```

Then run extended controls:

```bash
python run_all.py --full --controls --continue-on-error
```

For a focused paper-output pass:

```bash
python run_all.py --only 00,01,04,09d,12,13,16,17,19,20,21,22,24,25,27,28,29 --controls --continue-on-error
```

For angular diagnostics:

```bash
python run_all.py --only 10,11 --continue-on-error
```


## Project status

This repository is the working analysis kit for the LHCb / WCT log-domain candidate-spectrum study.

The current emphasis is on:

```text
reproducibility
control tests
active-domain winding stability
Koide/trig comb diagnostics
sideband and charm-continuum behavior
```

The default pipeline is yield-side because those stages are the most developed and directly tied to the current manuscript outputs. Angular diagnostics are included as an opt-in extension.

---

## Citation / related manuscript

Related manuscript title:

```text
Log-Spectral Structure and Koide-Like Winding Geometry
in Open-Data B0 -> K*0 mu+ mu- Candidate Spectra
```

Author:

```text
Richard J. Reyes
```

Original release:

```text
May 9, 2026
```

---

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## Contact

Richard J. Reyes

```text
rickyjreyes@gmail.com
```

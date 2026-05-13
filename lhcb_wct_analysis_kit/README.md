# LHCb WCT Analysis Kit

This directory contains the current analysis pipeline for testing structured log-domain residual behavior in open-data `B0 -> (K*(892)0 -> K+ pi-) mu+ mu-` candidate spectra.

The main runner is:

```bash
python run_all.py
```

The default runner is focused on the non-angular paper-grade pipeline. Angular stages are opt-in because they require separate validation before being treated as final analysis results.

---

## Input data

Place the main LHCb ROOT files in:

```text
data/*.root
data/*.dvntuple.root
```

Optional independent control-channel ROOT files go in:

```text
data_control/*.root
data_control/*.dvntuple.root
```

Most stages will skip automatically if the required ROOT files are not present.

---

## Install

Core Python packages:

```bash
pip install uproot awkward numpy pandas scipy matplotlib
```

GPU-accelerated stages use CuPy when available:

```bash
pip install cupy-cuda12x
```

Use the CuPy package matching your CUDA installation. CPU fallback behavior depends on the individual script.

---

## Quick start

Check the execution plan without running stages:

```bash
python run_all.py --dry-run
```

Run a reduced smoke/diagnostic pass:

```bash
python run_all.py --fast --continue-on-error
```

Run the default non-angular paper-grade pipeline:

```bash
python run_all.py --continue-on-error
```

Run the full pipeline, including optional overlap/control stages:

```bash
python run_all.py --full --controls --continue-on-error
```

Run only the sideband and charm-trimmed controls:

```bash
python run_all.py --only 28,29 --n-null 500 --continue-on-error
```

---

## Pipeline stages

### Intake and readiness

| Key | Script | Purpose |
|---:|---|---|
| `00` | `00_inspect_root.py` | Inspect the first ROOT file and tree layout. |
| `01` | `01_check_branches.py` | Check q2, four-vector, and angular branch readiness. |
| `03` | `03_bootstrap_scan.py` | Legacy/bootstrap diagnostic on yield-side log-FFT peaks. |
| `04` | `04_angle_branch_report.py` | Search for direct angular branch candidates. |

### Yield-side repaired log-cos / winding pipeline

| Key | Script | Purpose |
|---:|---|---|
| `09d` | `09d_two_mode_kde_baseline_polar_cupy.py` | Repaired KDE-baseline bounded-Poisson two-mode scan. |
| `12` | `12_wct_integer_winding_scan.py` | Discrete active-domain integer-winding scan. |
| `13` | `13_wct_koide_trig_comb_scan_cupy.py` | Koide/trig comb scan over active-domain winding ratios. |
| `16` | `16_wct_vs_smqft_likelihood_test_cupy.py` | Compare smooth empirical SM/QFT-like null against WCT comb alternatives. |
| `17` | `17_wct_sideband_subtracted_comb_test_cupy.py` | Sideband-subtracted WCT comb diagnostic. |
| `28` | `28_sideband.py` | Sideband-subtracted residual WLS control using the same active support and k-to-n mapping. |
| `29` | `29_charm_tail_trimmed_control.py` | Charm-trimmed control: remove J/psi and psi(2S) windows first, then test the continuum. |

### Angular pipeline, opt-in

| Key | Script | Purpose |
|---:|---|---|
| `10` | `10_compute_angles.py` | Compute derived q2, K* mass, cosThetaL, cosThetaK, and phi from four-vectors. |
| `11` | `11_angular_logcos_scan.py` | Angular moment / P5-proxy log-cos scan. |

Run angular stages only after confirming the local implementations are the intended final versions:

```bash
python run_all.py --angular --continue-on-error
python run_all.py --only 10,11 --continue-on-error
```

### Well-first and cross-region diagnostics

| Key | Script | Purpose |
|---:|---|---|
| `19` | `19_koide_well.py` | Find raw spectral wells before imposing any Koide comb. |
| `20` | `20_koide_proof.py` | Null/proof-style test of Koide-like geometry in raw wells. |
| `21` | `21_cross_region_scaling_phase_test.py` | Cross-region dilation and phase-coherence test. |
| `22` | `22_cross_regional_stability_test.py` | Stability sweep over top-well counts and region-pair directions. |
| `24` | `24_locked_branch_amplitude.py` | Locked-branch amplitude-cap ladder. |
| `lwcr` | `locked_winding_cross_region.py` | Locked-winding cross-region test without floating k. |

### Veto covariance / active-domain invariance

| Key | Script | Purpose |
|---:|---|---|
| `25` | `25_veto_window_covariance_test.py` | Main veto-window covariance / active-domain invariance test. |
| `26` | `26_veto_covariance.py` | Optional alternate/report-style veto covariance implementation. |

Stage `26` overlaps with `25`, so it is skipped by default. Use `--full` or `--only 26` to run it.

### Controls

| Key | Script | Purpose |
|---:|---|---|
| `05` | `05_control_compare.py` | Legacy signal/control FFT comparison; kept for provenance. |
| `27` | `27_control_channel_blind_test.py` | Blind control-channel / reconstruction-control test. |

Control stages are skipped unless `--controls` is passed or `data_control/` contains ROOT files.

---

## Main output directories

Generated output directories include:

```text
outputs/
outputs_run_all/
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
outputs_wct_locked_winding_cross_region/
outputs_wct_veto_covariance/
outputs_control_blind/
```

Every `run_all.py` run writes logs and a manifest under:

```text
outputs_run_all/run_manifest.json
outputs_run_all/<stage>_<script_name>.log
```

---

## Recommended clean rebuild sequence

```bash
python run_all.py --dry-run
python run_all.py --only 00,01,04 --continue-on-error
python run_all.py --only 09d,12,13,16,17 --continue-on-error
python run_all.py --only 28,29 --n-null 500 --continue-on-error
python run_all.py --only 19,20,21,22,24,25 --continue-on-error
python run_all.py --controls --only 27 --control-mode jpsi_peak --continue-on-error
```

For stronger null statistics on stages 28 and 29, increase `--n-null` after the smoke/rebuild sequence is working.


# LHCb / WCT Analysis Kit

This repository contains the working analysis pipeline for testing log-periodic residual structure, active-domain winding, Koide-like comb geometry, sideband controls, charm-trimmed controls, and veto-window covariance in open-data `B0 -> K*0 mu+ mu-` candidate spectra.

The main orchestrator is:

```bash
python run_all.py
```

The default runner is intentionally focused on the **non-angular paper-grade pipeline**. Angular stages `10` and `11` are opt-in because local working copies of those files have varied and should be verified before being treated as final.

---

## Directory layout

Expected input data:

```text
data/                  Main LHCb ROOT files
data_control/          Optional independent control-channel ROOT files
```

Generated output folders are created by the scripts, for example:

```text
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
outputs_wct_veto_covariance/
outputs_control_blind/
```

---

## Quick start

First check the plan without running anything:

```bash
python run_all.py --dry-run
```

Run a reduced diagnostic pass:

```bash
python run_all.py --fast --continue-on-error
```

Run the default non-angular paper-grade pipeline:

```bash
python run_all.py --continue-on-error
```

Run the full pipeline including optional overlap/alternate stages:

```bash
python run_all.py --full --controls --continue-on-error
```

Run only the new sideband and charm-trimmed controls:

```bash
python run_all.py --only 28,29 --n-null 100 --continue-on-error
```

For stronger null statistics on stages 28 and 29:

```bash
python run_all.py --only 28,29 --n-null 1000 --continue-on-error
```

---

## Windows UTF-8 note

On Windows PowerShell, set UTF-8 before long runs to avoid crashes on symbols such as `≈`, `Δ`, and `θ`:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
python run_all.py --continue-on-error
```

The current `run_all.py` also sets UTF-8 for subprocesses internally, but setting the environment explicitly is still recommended.

---

## Pipeline stages

### Intake

| Key | Script | Purpose |
|---:|---|---|
| `00` | `00_inspect_root.py` | Inspect first ROOT file and tree layout. |
| `01` | `01_check_branches.py` | Check q², four-vector, and angular branch readiness. |
| `04` | `04_angle_branch_report.py` | Search for possible direct angular branch candidates. |

### Yield-side repaired log-cos / winding pipeline

| Key | Script | Purpose |
|---:|---|---|
| `09d` | `09d_two_mode_kde_baseline_polar_cupy.py` | Final repaired KDE-baseline bounded-Poisson two-mode scan. |
| `12` | `12_wct_integer_winding_scan.py` | Discrete active-domain integer-winding scan. |
| `13` | `13_wct_koide_trig_comb_scan_cupy.py` | Koide/trig comb scan over active-domain winding ratios. |
| `16` | `16_wct_vs_smqft_likelihood_test_cupy.py` | Compare smooth SM/QFT-like null against WCT comb alternatives. |
| `17` | `17_wct_sideband_subtracted_comb_test_cupy.py` | Sideband-subtracted WCT comb diagnostic. |
| `28` | `28_sideband.py` | Sideband-subtracted residual WLS control using the same active support and k↔n mapping. |
| `29` | `29_charm_tail_subtraction_control.py` | Charm-trimmed control: cut J/ψ and ψ(2S) first, then test the continuum. |

### Angular pipeline, opt-in

| Key | Script | Purpose |
|---:|---|---|
| `10` | `10_compute_angles.py` | Compute derived q², K* mass, cosThetaL, cosThetaK, and phi from four-vectors. |
| `11` | `11_angular_logcos_scan.py` | Angular moment / P5-proxy log-cos scan. |

Run angular stages only after confirming the local `10` and `11` files are the intended final implementations:

```bash
python run_all.py --angular --continue-on-error
# or
python run_all.py --only 10,11 --continue-on-error
```

### Well-first and cross-region diagnostics

| Key | Script | Purpose |
|---:|---|---|
| `19` | `19_koide_well.py` | Find raw spectral wells before imposing any Koide comb. |
| `20` | `20_koide_proof.py` | Null/proof-style test of Koide-like geometry in raw wells. |
| `21` | `21_cross_region_scaling_phase_test.py` | Test cross-region dilation and phase coherence. |
| `22` | `22_cross_regional_stability_test.py` | Stability sweep over top-well counts and region pair directions. |
| `24` | `24_locked_branch_amplitude.py` | Locked-branch amplitude-cap ladder. |

### Veto covariance / active-domain invariance

| Key | Script | Purpose |
|---:|---|---|
| `25` | `25_veto_window_covariance_test.py` | Main veto-window covariance / active-domain invariance test. |
| `26` | `26_veto_covariance.py` | Optional alternate/report-style veto covariance implementation. |

Stage `26` overlaps with `25`, so it is skipped by default. Use `--full` or `--only 26` to run it.

### Controls

| Key | Script | Purpose |
|---:|---|---|
| `05` | `05_control_compare.py` | Legacy signal/control FFT comparison. Kept for provenance. |
| `27` | `27_control_channel_blind_test.py` | Blind control-channel / reconstruction-control test. |

Control stages are skipped unless `--controls` is passed or `data_control/` contains ROOT files.

---

## Important interpretation guardrails

This project is a candidate-spectrum diagnostic pipeline, not an official LHCb measurement.

The strongest defensible reading is:

> The open-data candidate spectrum contains structured log-domain residual behavior that survives several stress tests, including repaired baselines, active-domain winding tests, sideband controls, charm-trimmed controls, cross-region diagnostics, and veto-window covariance checks.

Do **not** overstate the current result as:

> A confirmed signal-specific rare-decay discovery or a Standard Model exclusion.

The paper-grade wording should remain close to:

> structured candidate-spectrum log-domain with scale-coupled signal and sideband projections, not a signal-specific rare-decay discovery.

---

## Recommended run sequence

For a clean local rebuild:

```bash
python run_all.py --dry-run
python run_all.py --only 00,01,04 --continue-on-error
python run_all.py --only 09d,12,13,16,17 --continue-on-error
python run_all.py --only 28,29 --n-null 500 --continue-on-error
python run_all.py --only 19,20,21,22,24,25 --continue-on-error
python run_all.py --controls --only 27 --control-mode jpsi_peak --continue-on-error
```

For a quick smoke test:

```bash
python run_all.py --fast --continue-on-error
```

For the complete default non-angular chain:

```bash
python run_all.py --continue-on-error
```

---

## Output manifest and logs

Each `run_all.py` run writes logs and a manifest under:

```text
outputs_run_all/
```

Key files:

```text
outputs_run_all/run_manifest.json
outputs_run_all/<stage>_<script_name>.log
```

Use the manifest to see which stages ran, failed, or were skipped.

---

## Archive recommendations

Move superseded, duplicate, or scratch files to `archive/` so the main repo view contains only current pipeline files.

Recommended archive candidates:

```text
01_veto_aware_spectral.py
02_control_compare.py
03_bootstrap_diagnostic.py
06_p5_placeholder.py
07_ladder_ratio_test.py
07_sensitivity_scan.py
09_direct_logcos.py
09b_direct_logcos_poisson_cupy.py
09c_two_mode_bounded_poisson_cupy.py
09c_two_mode_bounded_poisson_polar_cupy.py
14_wct_koide_sideband_control_scan_cupy.py
15_wct_koide_comb_cupy.py
lhc_ntuple.py
```

Do not delete these permanently. They are useful provenance, but should not be mixed with the final-stage scripts.

`10_compute_angles.py` and `11_angular_logcos_scan.py` should not be archived automatically. Keep them if verified, but treat them as opt-in until the local versions are confirmed.

---

## Current main files to keep visible

```text
00_inspect_root.py
01_check_branches.py
04_angle_branch_report.py
09d_two_mode_kde_baseline_polar_cupy.py
12_wct_integer_winding_scan.py
13_wct_koide_trig_comb_scan_cupy.py
16_wct_vs_smqft_likelihood_test_cupy.py
17_wct_sideband_subtracted_comb_test_cupy.py
19_koide_well.py
20_koide_proof.py
21_cross_region_scaling_phase_test.py
22_cross_regional_stability_test.py
24_locked_branch_amplitude.py
25_veto_window_covariance_test.py
26_veto_covariance.py
27_control_channel_blind_test.py
28_sideband.py
29_charm_tail_subtraction_control.py
config.py
lhcb_utils.py
run_all.py
```

Optional, if verified:

```text
10_compute_angles.py
11_angular_logcos_scan.py
```

---

## Naming note for stage 29

The file name `29_charm_tail_subtraction_control.py` is slightly misleading if the script is the corrected charm-trimmed control. The intended method is:

1. Remove the charmonium windows first.
2. Run the same log-winding / Koide diagnostics on the remaining active continuum.
3. Optionally test sideband-subtracted survival on that same trimmed support.

A clearer future name would be:

```text
29_charm_trimmed_control.py
```

If renamed, update the stage entry in `run_all.py`.

---

## Requirements

Core Python packages used across the pipeline:

```bash
pip install numpy pandas scipy matplotlib uproot awkward
```

GPU-accelerated scripts use CuPy when available:

```bash
pip install cupy-cuda12x
```

Use the CuPy package matching your CUDA version.


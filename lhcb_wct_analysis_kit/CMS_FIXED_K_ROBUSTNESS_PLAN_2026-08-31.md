# CMS-mapped fixed-k LHCb robustness plan — 2026-08-31

## Status

This is a **post-unblinding robustness plan**. The combined LHCb request-48 sample has already been inspected in stages 31 and 32. Therefore this suite is not a blind replication and cannot restore prospective status.

The purpose is narrower: determine whether the strong stage-32 response at the exact CMS-mapped frequency survives prespecified changes in sample split, smooth-background construction, KDE bandwidth, charmonium vetoes, and binning, without allowing a frequency search.

## Frozen frequency

The CMS source frequency remains

```text
omega_m = 7.025825825825827
```

and the exact LHCb q2 mapping remains

```text
k = omega_m / 2
  = 3.5129129129129133
```

Every test in this suite uses this single frequency. Frequency scanning is prohibited.

The phase remains free in the two-quadrature diagnostic:

```text
eta = C + a cos(k ell) + b sin(k ell)
ell = ln(q2 / 1 GeV^2)
```

with

```text
A = sqrt(a^2 + b^2)
phi = atan2(-b, a)
```

## Nominal selection

```text
channel: B0 -> K*(892)0 mu+ mu-
q2: 0.1 to 19.0 GeV^2
B0 mass: 5230 to 5330 MeV
K* mass: 795.9 to 995.9 MeV
J/psi veto: 8.0 to 11.0 GeV^2
psi(2S) veto: 12.5 to 14.5 GeV^2
bins: 60
KDE bandwidth scale: 1.50 x scipy Scott bandwidth
```

## Prespecified robustness grid

### 1. Run-group split

Evaluate all of the following without selecting the better result:

```text
00382466 only
00382467 only
combined
```

The nominal 60-bin, KDE-scale-1.50, nominal-veto model is used for this comparison.

### 2. KDE bandwidth ladder

On the combined sample with nominal vetoes and 60 bins:

```text
0.75
1.00
1.25
1.50
1.75
2.00
2.50
```

No bandwidth is selected after inspection. The full ladder is reported.

### 3. Binning ladder

On the combined sample with nominal vetoes and KDE scale 1.50:

```text
48 bins
60 bins
72 bins
90 bins
120 bins
```

The full ladder is reported.

### 4. Charmonium-veto perturbations

On the combined sample with 60 bins and KDE scale 1.50:

```text
nominal:
  J/psi   8.00 to 11.00
  psi2S  12.50 to 14.50

narrow:
  J/psi   8.25 to 10.75
  psi2S  12.75 to 14.25

wide:
  J/psi   7.75 to 11.25
  psi2S  12.25 to 14.75

shift_down:
  J/psi   7.75 to 10.75
  psi2S  12.25 to 14.25

shift_up:
  J/psi   8.25 to 11.25
  psi2S  12.75 to 14.75
```

These are mask robustness checks, not new signal definitions.

### 5. Alternative smooth backgrounds

In addition to the unbinned KDE baseline, evaluate Poisson-fitted Chebyshev log-rate backgrounds on the active bins at degrees:

```text
2
3
4
5
6
```

The Chebyshev family is intentionally capable of absorbing broad smooth structure. Survival across higher degrees is therefore a stronger stress test, but no individual degree is privileged after inspection.

## Primary statistic

For each scenario, use the fixed-frequency two-quadrature efficient score statistic with the intercept/nuisance normalization projected out:

```text
q = U^T I^-1 U
```

The reported phase and amplitude are descriptive. The primary robustness quantity is the fixed-k score and its empirical calibration.

## Fixed-baseline empirical null

For each grid point, generate multinomial pseudo-count spectra conditional on the observed active total using the fitted smooth baseline probabilities for that scenario.

Default:

```text
10,000 trials per scenario
seed = 20260831
```

Report the exceedance count and add-one p-value

```text
p = (r + 1) / (N + 1)
```

A zero-exceedance result is reported only as the Monte Carlo floor, not as an estimate of the true tail probability.

## Refit-baseline nulls

The suite also includes two explicitly refitted-background calibrations on the combined nominal sample:

1. **KDE refit bootstrap** — generate pseudo-events from the fitted nominal active-domain KDE, refit a KDE with the same bandwidth scale to each pseudo-sample, rebuild the binned smooth baseline, then recompute the fixed-k score.
2. **Chebyshev refit bootstrap** — generate multinomial pseudo-counts from a fitted degree-4 Chebyshev baseline, refit the degree-4 Chebyshev coefficients in every pseudoexperiment, then recompute the fixed-k score.

Default:

```text
500 refit trials for KDE
1000 refit trials for Chebyshev degree 4
```

These calibrations still do not model detector, trigger, reconstruction, acceptance, or Standard Model physics systematics. They test background-fit reuse more directly than the fixed-baseline null.

## Interpretation rule

The stage-32 observation is considered **robust within this exploratory LHCb analysis** only if the fixed-k response remains qualitatively strong across both run groups and does not disappear under reasonable bandwidth, binning, veto, and smooth-background variations.

A response that exists only for the nominal KDE choice, one run group, one binning, or one veto configuration is treated as evidence for analysis-induced structure rather than a stable cross-sample spectral feature.

No CMS and LHCb p-values or Z values are combined.

No result from this suite is discovery-grade physical significance.

## Implementation

```text
33_cms_fixed_k_robustness_suite.py
```

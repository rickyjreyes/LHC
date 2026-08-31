# CMS -> LHCb fixed-template freeze — 2026-08-31

## Purpose

This freeze defines a direct cross-experiment test of the CMS dimuon log-periodic
candidate in the existing LHCb `B0 -> K*(892)0 mu+ mu-` analysis framework.

The test is intentionally separate from the repository's existing high-`k`,
integer-winding, Koide/trig-comb, and cross-region analyses. It does **not**
reinterpret those results and it does **not** scan for a new LHCb optimum.

The implementation is:

```text
31_cms_locked_frequency_test.py
```

No result from that script is recorded in this freeze document.

## CMS source values

The frozen CMS candidate values are taken from the CMS analysis record in
`rickyjreyes/wct-cms`, where the candidate was identified in the Run2016H
discovery file and then tested at fixed frequency in later CMS samples.

Frozen values:

```text
CMS omega_m = 7.025825825825827
CMS discovery-file phase = -0.1889538223 rad
CMS phase convention = A cos[omega_m ln(m / 1 GeV) - phi_CMS]
```

The original discovery-file phase is used rather than a phase re-estimated from
a later CMS replication so that the LHCb phase is not chosen using LHCb output
or a post-hoc cross-sample average.

The CMS amplitude is **not** transferred. CMS and this LHCb pipeline use
different residual/rate normalizations, so an amplitude equality would not be
a justified cross-experiment prediction.

## Exact variable mapping

The LHCb repository uses

```text
q2 = m(mu+ mu-)^2
ell = ln(q2 / 1 GeV^2)
```

Since

```text
q2 = m^2
ln(q2 / 1 GeV^2) = 2 ln(m / 1 GeV)
```

the CMS waveform maps as

```text
A cos[omega_m ln(m / 1 GeV) - phi_CMS]
=
A cos[(omega_m / 2) ln(q2 / 1 GeV^2) - phi_CMS].
```

Therefore the frozen LHCb frequency is

```text
k_CMS_to_LHCb = omega_m / 2
              = 3.5129129129129133
```

The existing bounded-Poisson code uses the convention

```text
A cos(k ell + phi_LHCb)
```

with

```text
a = A cos(phi_LHCb)
b = -A sin(phi_LHCb).
```

Thus the frozen phase is

```text
phi_LHCb = -phi_CMS
         = +0.1889538223 rad.
```

No frequency-origin or phase-origin adjustment is required because the numeric
`q2` variable is in `GeV^2` and the reference is `1 GeV^2`.

## Frozen sign

The alternative is directional:

```text
A >= 0.
```

The sign is not allowed to flip after the data are inspected. If the best fit
would require the opposite sign, the locked positive-amplitude test should
return `A_hat = 0` or otherwise fail to improve the null.

## LHCb selection

The first implementation uses the same basic selection and KDE baseline family
as stage `09d`:

```text
channel: B0 -> K*(892)0 mu+ mu-
q2 range: 0.1 <= q2 <= 19.0 GeV^2
B0 mass: 5230 <= B0_M <= 5330 MeV
K* mass: 795.9 <= Kst_M <= 995.9 MeV
J/psi veto: 8.0 <= q2 <= 11.0 GeV^2
psi(2S) veto: 12.5 <= q2 <= 14.5 GeV^2
q2 bins: 60
KDE: scipy gaussian_kde, Scott bandwidth x 1.50
```

The retained active log-domain length is

```text
Delta ell_A = 4.780150335923678
```

so the CMS-derived template corresponds to the active-domain winding

```text
n_CMS = k_CMS_to_LHCb Delta ell_A / (2 pi)
      = 2.672569886096363.
```

This is distinct from the repository's previously studied `n = 10, 15, 20`
Koide/trig-comb structure.

## Frozen statistical model

Let `B_i` be the stage-09d-style KDE baseline in the active bins and

```text
ell_i = ln(q2_i / 1 GeV^2)
w_i = cos(k_CMS_to_LHCb ell_i + phi_LHCb).
```

Null:

```text
lambda_i(H0) = B_i exp(C).
```

Locked alternative:

```text
lambda_i(H1) = B_i exp(C + A w_i)
0 <= A <= 0.10.
```

Only `C` and the non-negative amplitude `A` are fitted.

The primary statistic is

```text
q_locked = 2 [log L(H1) - log L(H0)].
```

The following are forbidden in the primary test:

- scanning or re-optimizing `k`;
- scanning or re-optimizing phase;
- selecting the sign after inspection;
- substituting a nearby LHCb peak for the frozen CMS frequency;
- replacing the discovery-file CMS phase with a better-matching CMS or LHCb phase after inspection.

A one-sided Chernoff `0.5 chi-square_1` tail is reported only as an analytic
diagnostic.

## Empirical null

The script can generate Poisson pseudoexperiments from the fitted locked-test
null and refit the null normalization and locked positive amplitude on every
pseudoexperiment.

The default is:

```text
N_null = 10,000
seed = 20260831
```

The empirical p-value uses the add-one convention:

```text
p = (exceedances + 1) / (N_null + 1).
```

This null keeps the observed-data KDE baseline fixed. It therefore does **not**
constitute an end-to-end background-model, detector, trigger, or reconstruction
calibration and must not be labeled a discovery-grade significance.

## Data status and interpretation

The public LHCb request-48 `B0 -> K*0 mu+ mu-` files have already been studied
by other analyses in this repository. Therefore a result on request-48 is
classified as:

```text
retrospective cross-experiment fixed-template test
```

rather than a pristine blind replication.

That distinction does not make the test useless. The exact
`k = 3.5129129129129133`, fixed-phase, fixed-sign statistic was imported from
the independent CMS analysis rather than selected by scanning for a new LHCb
optimum. A successful result would therefore be evidence worth following up
with a genuinely untouched LHCb run, channel, or release.

The strongest future version is:

1. commit this statistic and all selection/background choices;
2. identify an LHCb sample not previously inspected with this statistic;
3. run the frozen frequency, phase, and sign once;
4. preserve the first output;
5. then perform predeclared background/systematic sensitivity tests;
6. only after model adequacy is established, deepen empirical-tail calibration.

## Commands

Inspect the freeze without touching data:

```bash
python 31_cms_locked_frequency_test.py --dry-run
```

Run the combined public request-48 sample:

```bash
python 31_cms_locked_frequency_test.py \
  --source request48 \
  --sample combined \
  --n-null 10000
```

Run each request-48 run group and then the combined sample:

```bash
python 31_cms_locked_frequency_test.py \
  --source request48 \
  --sample all \
  --n-null 10000
```

Run local ROOT files:

```bash
python 31_cms_locked_frequency_test.py \
  --source local \
  --data-glob "data/*.root" \
  --n-null 10000
```

## Claim boundary

A positive request-48 result would **not** by itself establish:

- a globally calibrated `>5 sigma` anomaly;
- WCT causation;
- independence from LHCb acceptance/reconstruction/background effects;
- equivalence between the exclusive LHCb `B0 -> K*0 mu+ mu-` spectrum and the
  inclusive CMS dimuon spectrum.

It would establish only what the test directly supports: whether the
CMS-derived fixed waveform has predictive power in this LHCb candidate spectrum
under the frozen implementation above.

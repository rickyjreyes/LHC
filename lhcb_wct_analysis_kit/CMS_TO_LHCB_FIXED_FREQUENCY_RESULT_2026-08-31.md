# CMS -> LHCb fixed-frequency result — 2026-08-31

## Scope

This record documents the first direct use of the CMS-derived log frequency in the existing LHCb `B0 -> K*(892)0 mu+ mu-` request-48 analysis.

The CMS source frequency is

```text
omega_m = 7.025825825825827
```

and the exact mapping to the LHCb variable

```text
ell = ln(q2 / 1 GeV^2), q2 = m(mu+mu-)^2
```

is

```text
k_CMS_to_LHCb = omega_m / 2
              = 3.5129129129129133
```

No LHCb frequency scan was used in either test below.

This result belongs in the LHC repository because the tested data are LHCb data. The CMS repository remains the source record for the original CMS candidate frequency and its CMS cross-period replications.

---

## Test 1 — frozen frequency + frozen phase + frozen sign

Implementation:

```text
31_cms_locked_frequency_test.py
```

Freeze document:

```text
CMS_LOCKED_LHCB_FREEZE_2026-08-31.md
```

Frozen LHCb-convention template:

```text
A cos(k ell + phi)
k = 3.5129129129129133
phi = +0.1889538223 rad
A >= 0
```

Frequency scan: disabled.

Phase scan: disabled.

Sign scan: disabled.

Combined local request-48 result:

```text
A_hat             = 0.00000000
q_locked          = 0.00000000
Chernoff p (diag) = 1
empirical p       = 1
empirical null    = 10000 / 10000 exceedances
```

### Verdict

**FAIL for the exact CMS-derived waveform transfer.**

The constrained best-fit amplitude is zero. The exact combination of CMS-derived frequency, CMS-derived phase, and positive sign does not improve the LHCb request-48 spectrum under the implemented stage-31 model.

This does not invalidate the CMS H/H2/G cross-period result. The LHCb sample is an exclusive `B0 -> K*0 mu+ mu-` candidate spectrum rather than the inclusive CMS dimuon observable.

---

## Test 2 — frozen frequency + free phase

Implementation:

```text
32_cms_fixed_frequency_free_phase_diagnostic.py
```

Classification:

```text
exploratory_post_unblinding_fixed_frequency_free_phase
```

The stage-31 result had already unblinded the combined request-48 sample, so this second test is explicitly exploratory. It must not be promoted to a prospective replication.

The frequency remained exactly frozen:

```text
k = 3.5129129129129133
```

with the two quadratures free:

```text
eta_i = C + a cos(k ell_i) + b sin(k ell_i)
```

and

```text
A = sqrt(a^2 + b^2)
phi = atan2(-b, a)
```

Frequency scan: disabled.

Phase: free.

Combined local request-48 result:

```text
score q (2 quadratures) = 398.52848531
chi2_2 p (diagnostic)   = 2.88828e-87
empirical trials        = 10000
empirical exceedances   = 0
empirical add-one p     = 9.999e-05

Poisson A_hat           = 0.21759750
Poisson phase_hat       = -1.9757308469 rad
phase - CMS             = -2.1646846692 rad = -124.027 deg
phase - opposite CMS    = +0.9769079844 rad = +55.973 deg
Poisson q_LRT           = 385.21460687
```

The Poisson fit emitted:

```text
WARNING: a/b numerical safety bound active; inspect fit before interpretation
```

At least one free quadrature coefficient reached the numerical safety bound. Therefore the reported `A_hat` and exact fitted phase are boundary-limited descriptive estimates and must not be treated as final amplitude or phase measurements.

The primary two-quadrature score statistic is not defined by that amplitude bound. The empirical calibration used a conditional multinomial null at the same fixed frequency and fixed smooth baseline shape.

### Verdict

**Strong fixed-frequency response under the implemented exploratory null, but not CMS phase replication.**

The exact CMS-mapped frequency produces a very large two-quadrature response in the combined LHCb request-48 spectrum when phase is allowed to vary. In the 10,000-trial implemented conditional multinomial null, zero trials exceeded the observed score, giving only the Monte Carlo resolution statement

```text
p_add-one = 1 / 10001 ~= 9.999e-05
```

This is not an estimate that the true physical p-value is `~1e-4`, and the analytic `chi2_2` value is not a discovery significance.

---

## Combined interpretation

The two tests together distinguish the hypotheses:

```text
same frequency + same phase + same sign    -> FAIL
same exact frequency, phase allowed free  -> STRONG EXPLORATORY RESPONSE
```

Therefore the stage-31 failure should not be summarized as "the CMS frequency is absent from LHCb." The more precise statement is:

> The exact CMS waveform does not transfer unchanged into the LHCb `B0 -> K*0 mu+ mu-` spectrum, but the exact CMS-mapped log frequency has a strong two-quadrature response under the implemented fixed-baseline exploratory diagnostic when phase is free.

The fitted phase is not compatible with the CMS reference phase, and its numerical value is not yet stable because the secondary Poisson fit reached a quadrature safety bound.

---

## What this does not establish

This result does **not** establish any of the following:

- a discovery-grade cross-experiment anomaly;
- a globally calibrated physical significance;
- detector/systematics independence;
- phase universality;
- amplitude universality;
- WCT causation;
- replication in an observable equivalent to the inclusive CMS dimuon spectrum.

The empirical stage-32 null conditions on the observed active event total and keeps the KDE baseline shape fixed. It is not an end-to-end detector, reconstruction, acceptance, background-family, or Standard Model systematic calibration.

---

## Required next robustness tests

The next tests must keep the frequency frozen at

```text
k = 3.5129129129129133
```

and should not perform a new frequency search.

Priority checks:

1. Run-group split: `00382466` and `00382467` independently, reporting score, amplitude, and phase without selecting the better group.
2. KDE bandwidth ladder with the same fixed frequency.
3. Alternative smooth background families at the same fixed frequency.
4. Charmonium-veto perturbation / mask-covariance checks at the same fixed frequency.
5. Refit-baseline pseudoexperiments rather than keeping the observed KDE shape fixed.
6. If an untouched LHCb sample or channel is available, freeze the fixed-frequency statistic before opening it.
7. Highest-value external test: an inclusive opposite-sign dimuon spectrum from a different detector, especially ATLAS, with the CMS frequency frozen before inspection.

Do not combine CMS and LHCb p-values or sigma values without a separately frozen joint statistic and joint null model.

---

## Current scientific status

A defensible one-sentence status is:

> **The exact CMS-derived waveform fails to transfer unchanged to LHCb request-48, but the exact CMS-mapped log frequency shows a strong exploratory fixed-frequency/free-phase response in the same LHCb spectrum under the implemented fixed-baseline null; systematic and independent-observable tests are still required.**

# CMS fixed-k degree-6 deep paired-null plan — 2026-08-31

## Status

This is a post-unblinding tail-calibration plan following the completed stage-34 cross-run background-discrimination test.

Stage 34 tested Chebyshev background degrees 2 through 6 at the exact fixed CMS-mapped LHCb frequency and obtained zero paired-null exceedances in 1,000 pseudoexperiments at every degree.

Degree 6 is selected for deeper calibration because it is the most flexible and weakest prespecified background stress test:

```text
phase difference = 51.103 deg
q_joint          = 51.77310339
paired null      = 0 / 1000 exceedances
```

No model component is changed for this deeper run.

## Frozen model

```text
k = 3.512912912912913
frequency scan = disabled
Chebyshev degree = 6
run group A = 00382466
run group B = 00382467
bins = 60
J/psi veto = 8.0 to 11.0 GeV^2
psi(2S) veto = 12.5 to 14.5 GeV^2
```

For the observed statistic:

1. Fit A with degree-6 Chebyshev background plus free fixed-k quadratures.
2. Transfer A's fitted phase to B.
3. Refit B's degree-6 background and a nonnegative target amplitude at the transferred phase.
4. Fit B with degree-6 Chebyshev background plus free fixed-k quadratures.
5. Transfer B's fitted phase to A.
6. Refit A's degree-6 background and a nonnegative target amplitude at the transferred phase.
7. Sum the two directional likelihood-ratio statistics.

The observed stage-34 value is expected to reproduce numerically as

```text
q_joint ~= 51.77310339
```

apart from negligible optimizer tolerance differences.

## Paired null

Default deep calibration:

```text
10,000 paired pseudoexperiments
seed = 20260831 + 60000
```

Every pseudoexperiment must repeat the full procedure:

- generate pseudo A and pseudo B from their fitted degree-6 smooth null probabilities conditional on the observed active totals;
- refit each degree-6 null background;
- refit each free fixed-k training phase;
- transfer the training phase to the opposite pseudo-run;
- refit the target degree-6 background and nonnegative amplitude;
- compute q_joint.

Report

```text
r = number of q_joint,null >= q_joint,obs
p_add-one = (r + 1) / (N + 1)
```

Also report exact one-sided zero-exceedance binomial upper bounds when r = 0:

```text
95% upper = 1 - 0.05^(1/N)
99% upper = 1 - 0.01^(1/N)
```

## Interpretation

This is an analysis-model tail calibration only. It does not model detector, trigger, reconstruction, acceptance, or Standard Model systematic uncertainties.

A zero-exceedance result does not establish that the true physical p-value equals the add-one floor and must not be converted into a discovery sigma.

No frequency, phase, background degree, binning, or veto tuning is permitted after inspection of this run.

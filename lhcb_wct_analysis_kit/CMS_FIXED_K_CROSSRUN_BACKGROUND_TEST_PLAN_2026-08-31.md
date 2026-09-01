# CMS-mapped fixed-k cross-run background discrimination plan — 2026-08-31

## Status

This is a **post-unblinding, prespecified robustness test**. Request-48 has already been opened in stages 31–33. The purpose is to address the main surviving ambiguity from stage 33: whether the strong response at the exact CMS-mapped frequency is a stable cross-run component or a feature that flexible smooth backgrounds can absorb.

No result from this stage is a prospective replication or a discovery significance.

## Frozen frequency

The CMS source value remains

```text
omega_m = 7.025825825825827
```

and the LHCb q2 mapping remains

```text
k = omega_m / 2 = 3.5129129129129133
```

Frequency scanning is prohibited.

## Data split

Use the two request-48 run groups exactly as fixed independent partitions:

```text
A = 00382466
B = 00382467
```

Both directions are always reported:

```text
A predicts phase -> test B
B predicts phase -> test A
```

No direction may be selected after inspection.

## Nominal selection

Use the same stage-33 nominal event selection and histogram support:

```text
q2: 0.1 to 19.0 GeV^2
B0 mass: 5230 to 5330 MeV
K* mass: 795.9 to 995.9 MeV
J/psi veto: 8.0 to 11.0 GeV^2
psi(2S) veto: 12.5 to 14.5 GeV^2
bins: 60
```

## Background family

Evaluate the full prespecified Chebyshev log-rate ladder from stage 33:

```text
degree = 2, 3, 4, 5, 6
```

No degree is selected as the primary result after looking at the output. The entire ladder is reported.

For degree d, the smooth Poisson model is

```text
log(lambda_i) = sum_{j=0}^d beta_j T_j(x_i)
```

where x is the fixed affine map of ln(q2) to [-1,1].

## Training phase estimate

Within a training run group, fit the smooth background and both fixed-frequency quadratures jointly:

```text
log(lambda_i) = background_d(ell_i)
              + a cos(k ell_i)
              + b sin(k ell_i)
```

with k fixed.

The training phase is

```text
phi_train = atan2(-b, a)
```

No frequency search is performed.

## Cross-run target test

In the other run group, refit the degree-d smooth background using that target group's data, but freeze the oscillation phase to phi_train. Fit only a nonnegative target amplitude A:

```text
log(lambda_i) = background_d,target(ell_i)
              + A cos(k ell_i + phi_train)
A >= 0
```

Compare against the nested target background-only model. Define

```text
q_target|train = 2 (logL_alt - logL_null), clipped at 0.
```

Repeat in the reverse direction.

The paired statistic for each degree is

```text
q_joint = q_B|A + q_A|B.
```

The two directional q values, training phases, target amplitudes, and phase difference between the independently fitted free-phase run-group solutions are all reported.

## Empirical paired null

For each Chebyshev degree independently:

1. Fit the background-only model to observed A and B.
2. Generate independent multinomial pseudo-histograms for A and B, conditional on each observed active total.
3. Refit the free-phase training model in each pseudo group at the same fixed k.
4. Use pseudo-A phase to test pseudo-B while refitting pseudo-B's smooth background and target amplitude.
5. Reverse the direction.
6. Record q_joint for the pseudo pair.

This reproduces the complete data-derived phase-prediction procedure under the smooth-background null and automatically accounts for dependence between the two cross-directions.

Default:

```text
1000 pseudo-pairs per Chebyshev degree
seed = 20260831
```

Report exceedance count and add-one p-value

```text
p = (r + 1) / (N + 1).
```

Zero exceedances are only a Monte Carlo resolution floor.

## Interpretation rule

The fixed-k component is considered **cross-run predictive within the tested background family** only if both directional target tests are positive and q_joint remains unusual under the paired refit null.

Special attention is given to degree 6 because stage 33 showed the largest attenuation there. A strong degree-6 cross-run result would weigh against the explanation that the stage-32/33 response is merely a low-order smooth-background artifact. A weak or null degree-6 result would preserve smooth-background misspecification as a viable explanation.

No degree is discarded because it gives an inconvenient result.

## Guardrails

- This is exploratory/post-unblinding.
- Frequency is fixed in every fit and pseudoexperiment.
- Target background coefficients are refit in every directional test.
- Pseudoexperiment backgrounds and training phases are refit in every null pair.
- Analytic chi-square/Chernoff values, if printed, are diagnostics only.
- Do not combine these p-values with CMS p-values or sigma values.
- This test does not model detector, trigger, reconstruction, acceptance, or Standard Model systematic uncertainty.

## Implementation

```text
34_cms_fixed_k_crossrun_background_test.py
```

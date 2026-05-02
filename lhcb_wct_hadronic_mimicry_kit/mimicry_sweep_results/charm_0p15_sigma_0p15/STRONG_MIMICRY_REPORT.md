# Strong Hadronic Mimicry Injection Test Report
Verdict: **PASS**

## Configuration
- Monte Carlo injections: `200`
- Target k: `11.7`
- k-window: `±1.0`
- k scan range: `2.0` to `25.0`
- Charm fake bins: `16`
- Injection sigma: `0.15`
- Charm scale: `0.15`

## Main single-dataset fit
- delta chi2 vs constant: `19.329846110128912`
- delta chi2 vs charm: `-3.897558116872043`
- best WCT k: `2.0`
- WCT AIC minus charm AIC: `7.897558116872043`
- WCT BIC minus charm BIC: `9.442735561351604`

## Monte Carlo false-positive rates
- `n_mc`: `200`
- `false_positive_rate_vs_const`: `0.97`
- `false_positive_rate_vs_charm`: `0.0`
- `false_positive_rate_vs_charm_and_k`: `0.0`
- `k_window_hit_rate`: `0.015`
- `wct_preferred_over_charm_by_aic_rate`: `0.05`
- `wct_preferred_over_charm_by_bic_rate`: `0.025`
- `target_delta`: `9.0`
- `target_k`: `11.7`
- `k_window`: `1.0`

## Pass criteria
- false_positive_rate_vs_charm <= `0.01`
- false_positive_rate_vs_charm_and_k <= `0.002`
- wct_preferred_over_charm_by_bic_rate <= `0.05`

## Interpretation
WCT did not repeatedly mistake charm-only Breit-Wigner tail pseudo-data for the target log-periodic signal. This reduces the charm-mimicry failure mode.

## Fit comparison table

| model      |     chi2 |   n_params |   dof |     aic |     bic |       par_c |       par_c0 |      par_c1 |   par_scale |      par_A |   par_k |   par_phi |    par_mu |   par_sigma_w |
|:-----------|---------:|-----------:|------:|--------:|--------:|------------:|-------------:|------------:|------------:|-----------:|--------:|----------:|----------:|--------------:|
| constant   | 28.6443  |          1 |    15 | 30.6443 | 31.4169 |   0.0336398 | nan          | nan         |  nan        | nan        |     nan | nan       | nan       |     nan       |
| charm_toy  |  5.4169  |          3 |    13 | 11.4169 | 13.7347 | nan         |  -0.00833069 |  -0.0289452 |    0.126416 | nan        |     nan | nan       | nan       |     nan       |
| wct_5param |  9.31446 |          5 |    11 | 19.3145 | 23.1774 | nan         | nan          | nan         |  nan        |  -0.283195 |       2 |   1.52589 |   2.05212 |       2.85069 |

# Strong Hadronic Mimicry Injection Test Report
Verdict: **PASS**

## Configuration
- Monte Carlo injections: `200`
- Target k: `11.7`
- k-window: `±1.0`
- k scan range: `2.0` to `25.0`
- Charm fake bins: `16`
- Injection sigma: `0.15`
- Charm scale: `0.5`

## Main single-dataset fit
- delta chi2 vs constant: `166.321111672745`
- delta chi2 vs charm: `-33.25504652176939`
- best WCT k: `2.0`
- WCT AIC minus charm AIC: `37.25504652176939`
- WCT BIC minus charm BIC: `38.80022396624895`

## Monte Carlo false-positive rates
- `n_mc`: `200`
- `false_positive_rate_vs_const`: `1.0`
- `false_positive_rate_vs_charm`: `0.0`
- `false_positive_rate_vs_charm_and_k`: `0.0`
- `k_window_hit_rate`: `0.0`
- `wct_preferred_over_charm_by_aic_rate`: `0.0`
- `wct_preferred_over_charm_by_bic_rate`: `0.0`
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

| model      |      chi2 |   n_params |   dof |      aic |      bic |       par_c |       par_c0 |      par_c1 |   par_scale |     par_A |   par_k |   par_phi |    par_mu |   par_sigma_w |
|:-----------|----------:|-----------:|------:|---------:|---------:|------------:|-------------:|------------:|------------:|----------:|--------:|----------:|----------:|--------------:|
| constant   | 204.991   |          1 |    15 | 206.991  | 207.763  |   0.0336398 | nan          | nan         |  nan        | nan       |     nan | nan       | nan       |     nan       |
| charm_toy  |   5.41436 |          3 |    13 |  11.4144 |  13.7321 | nan         |  -0.00746524 |  -0.0283483 |    0.477507 | nan       |     nan | nan       | nan       |     nan       |
| wct_5param |  38.6694  |          5 |    11 |  48.6694 |  52.5324 | nan         | nan          | nan         |  nan        |  -1.02104 |       2 |   1.50169 |   2.05212 |       1.15895 |

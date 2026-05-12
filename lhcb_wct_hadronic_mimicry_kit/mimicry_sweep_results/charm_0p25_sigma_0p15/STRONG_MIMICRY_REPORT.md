# Strong Hadronic Mimicry Injection Test Report
Verdict: **PASS**

## Configuration
- Monte Carlo injections: `200`
- Target k: `11.7`
- k-window: `±1.0`
- k scan range: `2.0` to `25.0`
- Charm fake bins: `16`
- Injection sigma: `0.15`
- Charm scale: `0.25`

## Main single-dataset fit
- delta chi2 vs constant: `46.47520181207609`
- delta chi2 vs charm: `-9.359784211696041`
- best WCT k: `2.0`
- WCT AIC minus charm AIC: `13.359784211696041`
- WCT BIC minus charm BIC: `14.904961656175606`

## Monte Carlo false-positive rates
- `n_mc`: `200`
- `false_positive_rate_vs_const`: `1.0`
- `false_positive_rate_vs_charm`: `0.0`
- `false_positive_rate_vs_charm_and_k`: `0.0`
- `k_window_hit_rate`: `0.0`
- `wct_preferred_over_charm_by_aic_rate`: `0.015`
- `wct_preferred_over_charm_by_bic_rate`: `0.005`
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
| constant   | 61.2512  |          1 |    15 | 63.2512 | 64.0237 |   0.0336398 | nan          | nan         |  nan        | nan        |     nan | nan       | nan       |     nan       |
| charm_toy  |  5.41617 |          3 |    13 | 11.4162 | 13.7339 | nan         |  -0.00808342 |  -0.0287746 |    0.226728 | nan        |     nan | nan       | nan       |     nan       |
| wct_5param | 14.776   |          5 |    11 | 24.776  | 28.6389 | nan         | nan          | nan         |  nan        |  -0.493654 |       2 |   1.51321 |   2.05212 |       1.50095 |

# Strong Hadronic Mimicry Injection Test Report
Verdict: **FAIL**

## Configuration
- Monte Carlo injections: `200`
- Target k: `11.7`
- k-window: `±1.0`
- k scan range: `2.0` to `25.0`
- Charm fake bins: `16`
- Injection sigma: `0.2`
- Charm scale: `0.15`

## Main single-dataset fit
- delta chi2 vs constant: `11.352318877957675`
- delta chi2 vs charm: `-2.346971751886235`
- best WCT k: `2.0`
- WCT AIC minus charm AIC: `6.346971751886235`
- WCT BIC minus charm BIC: `7.892149196365796`

## Monte Carlo false-positive rates
- `n_mc`: `200`
- `false_positive_rate_vs_const`: `0.77`
- `false_positive_rate_vs_charm`: `0.015`
- `false_positive_rate_vs_charm_and_k`: `0.0`
- `k_window_hit_rate`: `0.025`
- `wct_preferred_over_charm_by_aic_rate`: `0.105`
- `wct_preferred_over_charm_by_bic_rate`: `0.04`
- `target_delta`: `9.0`
- `target_k`: `11.7`
- `k_window`: `1.0`

## Pass criteria
- false_positive_rate_vs_charm <= `0.01`
- false_positive_rate_vs_charm_and_k <= `0.002`
- wct_preferred_over_charm_by_bic_rate <= `0.05`

## Interpretation
WCT produced too many charm-only false positives. The ansatz is too flexible or the null model is insufficient. Do not proceed to discovery claims until fixed.

## Fit comparison table

| model      |     chi2 |   n_params |   dof |     aic |     bic |       par_c |      par_c0 |      par_c1 |   par_scale |      par_A |   par_k |   par_phi |    par_mu |   par_sigma_w |
|:-----------|---------:|-----------:|------:|--------:|--------:|------------:|------------:|------------:|------------:|-----------:|--------:|----------:|----------:|--------------:|
| constant   | 19.1165  |          1 |    15 | 21.1165 | 21.8891 |   0.0400198 | nan         | nan         |    nan      | nan        |     nan | nan       | nan       |           nan |
| charm_toy  |  5.41717 |          3 |    13 | 11.4172 | 13.7349 | nan         |  -0.0112312 |  -0.0353455 |      0.1184 | nan        |     nan | nan       | nan       |           nan |
| wct_5param |  7.76414 |          5 |    11 | 17.7641 | 21.6271 | nan         | nan         | nan         |    nan      |  -0.290692 |       2 |   1.51516 |   2.05212 |             3 |

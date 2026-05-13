# Strong Hadronic Mimicry Injection Test Report
Verdict: **PASS**

## Configuration
- Monte Carlo injections: `200`
- Target k: `11.7`
- k-window: `±1.0`
- k scan range: `2.0` to `25.0`
- Charm fake bins: `16`
- Injection sigma: `0.2`
- Charm scale: `0.75`

## Main single-dataset fit
- delta chi2 vs constant: `204.57803795194513`
- delta chi2 vs charm: `-40.93366299352624`
- best WCT k: `2.0`
- WCT AIC minus charm AIC: `44.93366299352624`
- WCT BIC minus charm BIC: `46.478840438005804`

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

| model      |      chi2 |   n_params |   dof |      aic |      bic |       par_c |      par_c0 |      par_c1 |   par_scale |     par_A |   par_k |   par_phi |    par_mu |   par_sigma_w |
|:-----------|----------:|-----------:|------:|---------:|---------:|------------:|------------:|------------:|------------:|----------:|--------:|----------:|----------:|--------------:|
| constant   | 250.926   |          1 |    15 | 252.926  | 253.698  |   0.0400198 | nan         | nan         |  nan        | nan       |     nan | nan       | nan       |     nan       |
| charm_toy  |   5.41391 |          3 |    13 |  11.4139 |  13.7317 | nan         |  -0.0097476 |  -0.0343223 |    0.720268 | nan       |     nan | nan       | nan       |     nan       |
| wct_5param |  46.3476  |          5 |    11 |  56.3476 |  60.2105 | nan         | nan         | nan         |  nan        |  -1.54256 |       2 |   1.49791 |   2.05212 |       1.10151 |

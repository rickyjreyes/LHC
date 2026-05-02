# Strong Hadronic Mimicry Injection Test Report
Verdict: **PASS**

## Configuration
- Monte Carlo injections: `200`
- Target k: `11.7`
- k-window: `±1.0`
- k scan range: `2.0` to `25.0`
- Charm fake bins: `16`
- Injection sigma: `0.1`
- Charm scale: `0.25`

## Main single-dataset fit
- delta chi2 vs constant: `101.79957383132628`
- delta chi2 vs charm: `-20.242799909568127`
- best WCT k: `2.0`
- WCT AIC minus charm AIC: `24.242799909568127`
- WCT BIC minus charm BIC: `25.78797735404769`

## Monte Carlo false-positive rates
- `n_mc`: `200`
- `false_positive_rate_vs_const`: `1.0`
- `false_positive_rate_vs_charm`: `0.0`
- `false_positive_rate_vs_charm_and_k`: `0.0`
- `k_window_hit_rate`: `0.0`
- `wct_preferred_over_charm_by_aic_rate`: `0.005`
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

| model      |      chi2 |   n_params |   dof |      aic |     bic |       par_c |       par_c0 |      par_c1 |   par_scale |      par_A |   par_k |   par_phi |    par_mu |   par_sigma_w |
|:-----------|----------:|-----------:|------:|---------:|--------:|------------:|-------------:|------------:|------------:|-----------:|--------:|----------:|----------:|--------------:|
| constant   | 127.458   |          1 |    15 | 129.458  | 130.23  |   0.0272599 | nan          | nan         |  nan        | nan        |     nan | nan       | nan       |     nan       |
| charm_toy  |   5.41527 |          3 |    13 |  11.4153 |  13.733 | nan         |  -0.00518289 |  -0.0223743 |    0.234745 | nan        |     nan | nan       | nan       |     nan       |
| wct_5param |  25.6581  |          5 |    11 |  35.6581 |  39.521 | nan         | nan          | nan         |  nan        |  -0.501333 |       2 |   1.51111 |   2.05212 |       1.36348 |

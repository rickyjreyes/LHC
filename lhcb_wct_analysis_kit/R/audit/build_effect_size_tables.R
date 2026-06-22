# =============================================================================
# build_effect_size_tables.R   (section 9)
#
# Interpretable effect sizes for each headline single/two-mode/comb result:
# k, log-period, rho_q2, winding n, amplitude, phase, deviance improvement,
# null-standardised z (descriptive), percentile, and the conditional
# peak_to_trough_rate_ratio = exp(2A) for an isolated log-rate component.
#
# Does NOT use p->sigma conversion as the main effect size.
# =============================================================================

build_effect_size_tables <- function(table_dir, kit = AUDIT_KIT,
                                      s09 = NULL, s28 = NULL) {
  rows <- list()
  add <- function(...) rows[[length(rows)+1]] <<- data.frame(..., stringsAsFactors = FALSE)

  j09 <- read_json_safe(file.path(kit, "outputs_logcos_poisson_twomode_kde_polar/two_mode_summary.json"))
  if (!is.null(j09)) {
    nb <- if (!is.null(s09)) s09$null$null_best_deltaD_add else NULL
    for (tag in c("best_two", "reference_two")) {
      m <- j09[[tag]]
      k <- m$k2
      add(result = paste0("09d_", tag), stage = "09d", k = k,
          log_period = log_period(k), rho_q2 = rho_q2(k), winding_n = winding_n(k),
          amplitude = m$A2 %||% NA, phase = m$phi2 %||% NA,
          deviance_improvement = m$deltaD_add %||% NA,
          z_null = if (!is.null(nb)) z_null(m$deltaD_add, nb) else NA,
          null_percentile = if (!is.null(nb)) null_percentile(m$deltaD_add, nb) else NA,
          peak_to_trough_rate_ratio = peak_to_trough_ratio(m$A2 %||% NA),
          amplitude_bound_active = m$amplitude2_bound_active %||% NA)
    }
  }
  # comb headline (bw1, Q=2/3 and Q=4/9)
  cj <- read_json_safe(file.path(kit, "outputs_wct_koide_comb/koide_comb_summary.json"))
  if (!is.null(cj)) {
    krow <- cj$koide_Q_rows
    for (r in krow) if (isTRUE(r$KDE_BANDWIDTH_SCALE == 1.0)) {
      add(result = "13_comb_Q2over3_bw1", stage = "13", k = r$k0,
          log_period = log_period(r$k0), rho_q2 = rho_q2(r$k0), winding_n = winding_n(r$k0),
          amplitude = max(r$A_minus, r$A0, r$A_plus), phase = NA,
          deviance_improvement = r$deltaD, z_null = NA, null_percentile = NA,
          peak_to_trough_rate_ratio = peak_to_trough_ratio(max(r$A_minus, r$A0, r$A_plus)),
          amplitude_bound_active = r$any_A_bound_active)
    }
    brow <- cj$best_Q_by_bandwidth
    for (r in brow) if (isTRUE(r$KDE_BANDWIDTH_SCALE == 1.0)) {
      add(result = "13_best_Q4over9_bw1", stage = "13", k = r$k0,
          log_period = log_period(r$k0), rho_q2 = rho_q2(r$k0), winding_n = winding_n(r$k0),
          amplitude = max(r$A_minus, r$A0, r$A_plus), phase = NA,
          deviance_improvement = r$deltaD, z_null = NA, null_percentile = NA,
          peak_to_trough_rate_ratio = peak_to_trough_ratio(max(r$A_minus, r$A0, r$A_plus)),
          amplitude_bound_active = r$any_A_bound_active)
    }
  }
  # sideband control headline (non-survival)
  if (!is.null(s28)) {
    rr <- s28$regression
    add(result = "28_sideband_best_scan", stage = "28", k = rr$best_k,
        log_period = log_period(rr$best_k), rho_q2 = rho_q2(rr$best_k),
        winding_n = winding_n(rr$best_k), amplitude = NA, phase = NA,
        deviance_improvement = rr$best_dchi2, z_null = NA, null_percentile = NA,
        peak_to_trough_rate_ratio = NA, amplitude_bound_active = NA)
  }
  eff <- do.call(rbind, rows)
  eff$note <- "z_null is a descriptive standardised distance (null generally non-Gaussian); peak_to_trough applies to an isolated log-rate component"
  write_audit_csv(eff, file.path(table_dir, "effect_sizes.csv"))
  eff
}

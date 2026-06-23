# =============================================================================
# run_09d_sensitivity.R   (section 10)
#
# Stage-09d two-mode audit, replayable from committed scan + null tables:
#   outputs_logcos_poisson_twomode_kde_polar/two_mode_scan_mask.csv
#   outputs_logcos_poisson_twomode_kde_polar/two_mode_null_mask.csv
#   outputs_logcos_poisson_twomode_kde_polar/two_mode_summary.json
#
# Produces: profile-deviance curve for k2, best vs reference comparison,
# boundary activity, and the distinct fixed / local-scan / family p-values.
# Event-level full-pipeline null and event bootstrap are FULL-mode only and are
# flagged UNAVAILABLE_NO_EVENT_DATA in replay.
# =============================================================================

run_09d_sensitivity <- function(table_dir, figure_dir, kit = AUDIT_KIT,
                                 mode = "replay") {
  base <- file.path(kit, "outputs_logcos_poisson_twomode_kde_polar")
  scan <- read_csv_safe(file.path(base, "two_mode_scan_mask.csv"))
  null <- read_csv_safe(file.path(base, "two_mode_null_mask.csv"))
  summ <- read_json_safe(file.path(base, "two_mode_summary.json"))
  if (is.null(scan) || is.null(summ)) {
    warning("stage 09d artifacts missing; skipping")
    return(NULL)
  }

  k2 <- scan$k2
  dD <- scan$deltaD_add_exact
  a2bound <- as.logical(scan$amplitude2_bound_active)

  # --- best local scan + reference ---
  best_i <- which.max(dD)
  best_k2 <- k2[best_i]; best_dD <- dD[best_i]
  ref_k2 <- summ$reference_k2 %||% 19.5296
  ref_i <- which.min(abs(k2 - ref_k2))
  ref_dD_scan <- dD[ref_i]

  # null statistics
  null_best <- if (!is.null(null)) null$null_best_deltaD_add else numeric(0)
  null_ref  <- if (!is.null(null)) null$null_reference_deltaD_add else numeric(0)
  p_best <- audit_empirical_p(best_dD, null_best)
  p_ref_local <- audit_empirical_p(summ$reference_two$deltaD_add %||% ref_dD_scan, null_best)
  p_ref_fixed <- audit_empirical_p(summ$reference_two$deltaD_add %||% ref_dD_scan, null_ref)

  # boundary activity across the scan
  boundary_frac <- mean(a2bound, na.rm = TRUE)

  # peak-uncertainty table
  peak <- data.frame(
    quantity = c("best_local_scan", "prespecified_reference"),
    k2 = c(best_k2, ref_k2),
    deltaD_add = c(best_dD, summ$reference_two$deltaD_add %||% ref_dD_scan),
    A2 = c(summ$best_two$A2 %||% NA, summ$reference_two$A2 %||% NA),
    A2_bound_active = c(summ$best_two$amplitude2_bound_active %||% NA,
                        summ$reference_two$amplitude2_bound_active %||% NA),
    phase = c(summ$best_two$phi2 %||% NA, summ$reference_two$phi2 %||% NA),
    p_pointwise = c(NA, p_ref_fixed$p),
    p_local_scan_max = c(p_best$p, p_ref_local$p),
    log_period = log_period(c(best_k2, ref_k2)),
    rho_q2 = rho_q2(c(best_k2, ref_k2)),
    winding_n = winding_n(c(best_k2, ref_k2)),
    z_null = c(z_null(best_dD, null_best), z_null(summ$reference_two$deltaD_add %||% ref_dD_scan, null_best)),
    peak_to_trough_ratio = peak_to_trough_ratio(c(summ$best_two$A2 %||% NA, summ$reference_two$A2 %||% NA)),
    mc_resolution = c(p_best$resolution, p_ref_fixed$resolution),
    stringsAsFactors = FALSE)
  write_audit_csv(peak, file.path(table_dir, "stage09d_peak_uncertainty.csv"))

  # baseline sensitivity (replay: summarised from committed null tail + diag;
  # full-pipeline re-estimated KDE null is FULL-mode only)
  diag <- summ$diagnostics %||% list()
  baseline <- data.frame(
    null_level = c("parity_conditional", "full_pipeline_kde_reestimated"),
    description = c("baseline held fixed; refit nuisance; committed null",
                    "regenerate->rebin->retrain KDE->refit->rescan (event-level)"),
    available = c(TRUE, mode == "full" && have_event_data()),
    p_best = c(p_best$p, NA),
    p_reference = c(p_ref_fixed$p, NA),
    null_best_mean = c(mean(null_best), NA),
    null_best_p99 = c(as.numeric(stats::quantile(null_best, 0.99, names = FALSE)), NA),
    scan_A2_bound_fraction = c(boundary_frac, NA),
    note = c("replayable", if (mode == "full" && have_event_data()) "run in full mode"
             else "UNAVAILABLE_NO_EVENT_DATA"),
    stringsAsFactors = FALSE)
  write_audit_csv(baseline, file.path(table_dir, "stage09d_baseline_sensitivity.csv"))

  # boundary-selection frequency under the null (where does the null scan-max land?)
  null_peak_k2 <- if (!is.null(null)) null$null_best_k2 else numeric(0)
  list(
    scan = scan, null = null, summary = summ,
    best_k2 = best_k2, best_dD = best_dD, ref_k2 = ref_k2,
    p_best = p_best, p_ref_local = p_ref_local, p_ref_fixed = p_ref_fixed,
    boundary_frac = boundary_frac, peak = peak, baseline = baseline,
    null_peak_k2 = null_peak_k2,
    regression = list(
      D_base = summ$base$D_base, best_k2 = summ$best_two$k2,
      best_dD = summ$best_two$deltaD_add, best_A2 = summ$best_two$A2,
      ref_k2 = summ$reference_two$k2, ref_dD = summ$reference_two$deltaD_add))
}

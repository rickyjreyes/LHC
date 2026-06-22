# =============================================================================
# run_injection_recovery.R   (section 22)
#
# Injection-recovery. For the event-free WLS sideband pipeline we CAN inject a
# known log-cos mode into the signal counts and measure detection power and
# frequency/amplitude recovery in replay mode. For the KDE/Poisson 09d pipeline
# this is FULL-mode only.
#
# Also supports the signal-only / sideband-only / both / common-component
# injection design that tests whether sideband subtraction removes common
# structure and retains signal-specific structure.
# =============================================================================

#' Inject a log-cos modulation of amplitude A at frequency k into Poisson counts
#' with mean mu and return resampled counts.
.inject_counts <- function(mu, ell, A, k, phase = 0) {
  lam <- mu * (1 + A * cos(k * ell + phase))
  lam[lam < 0] <- 0
  rpois(length(lam), lam)
}

run_injection_recovery <- function(table_dir, figure_dir, kit = AUDIT_KIT,
                                    mode = "replay", injection_n = 200,
                                    seed = 161803, fast = FALSE) {
  if (fast) injection_n <- min(injection_n, 50)
  base <- file.path(kit, "outputs_sideband_subtracted")
  bins <- read_csv_safe(file.path(base, "sideband_subtracted_bins.csv"))
  summ <- read_json_safe(file.path(base, "sideband_subtracted_summary.json"))
  rows <- list()

  if (!is.null(bins) && !is.null(summ)) {
    ell <- as.numeric(bins$ell)
    k1 <- summ$scan_config$K1_FIXED %||% 7.61054
    kt <- summ$scan_config$k_targets
    mu_sig <- pmax(as.numeric(bins$N_signal), 0)
    mu_low <- pmax(as.numeric(bins$N_Blow), 0)
    mu_high <- pmax(as.numeric(bins$N_Bhigh), 0)
    k_inject <- c(reference = 19.5296, n15 = kt$n15 %||% 19.716488600662828,
                  n20 = kt$n20 %||% 26.28865146755044, between = 22.0, low_ctrl = 9.0)
    amps <- c(0.00, 0.02, 0.05, 0.10)
    # null reference: scan-max threshold from a no-injection MC
    audit_set_seed(seed)
    null_scanmax <- numeric(min(injection_n, 200))
    k_grid <- seq(summ$scan_config$K_SCAN_MIN %||% 6,
                  summ$scan_config$K_SCAN_MAX %||% 32, length.out = 121)
    for (o in seq_along(null_scanmax)) {
      s <- rpois(length(mu_sig), mu_sig); lo <- rpois(length(mu_low), mu_low)
      hi <- rpois(length(mu_high), mu_high); sc <- lo + hi
      a <- sum(s)/max(sum(sc),1); R <- s - a*sc; V <- s + a^2*sc; V[V<=0] <- 1; w <- 1/V
      cb <- .wls_chi2(R, w, .design_base(ell, k1))$chi2
      null_scanmax[o] <- max(vapply(k_grid, function(k2) cb - .wls_chi2(R,w,.design_alt(ell,k1,k2))$chi2, 0))
    }
    thr95 <- as.numeric(quantile(null_scanmax, 0.95, names = FALSE))

    for (design in c("signal_only", "sideband_only", "both", "common_component")) {
      for (knm in names(k_inject)) for (A in amps) {
        kk <- k_inject[[knm]]
        det <- 0; corr <- 0; khat <- numeric(0)
        for (o in seq_len(injection_n)) {
          inj_sig <- if (design %in% c("signal_only", "both")) A else 0
          inj_side <- if (design %in% c("sideband_only", "both")) A else 0
          inj_common <- if (design == "common_component") A else 0
          s <- .inject_counts(mu_sig, ell, inj_sig + inj_common, kk)
          lo <- .inject_counts(mu_low, ell, inj_side + inj_common, kk)
          hi <- .inject_counts(mu_high, ell, inj_side + inj_common, kk)
          sc <- lo + hi
          a <- sum(s)/max(sum(sc),1); R <- s - a*sc; V <- s + a^2*sc; V[V<=0] <- 1; w <- 1/V
          cb <- .wls_chi2(R, w, .design_base(ell, k1))$chi2
          dvals <- vapply(k_grid, function(k2) cb - .wls_chi2(R,w,.design_alt(ell,k1,k2))$chi2, 0)
          dm <- max(dvals); kbest <- k_grid[which.max(dvals)]
          if (dm > thr95) det <- det + 1
          if (dm > thr95 && abs(kbest - kk) / kk < 0.05) corr <- corr + 1
          khat <- c(khat, kbest)
        }
        rows[[length(rows)+1]] <- data.frame(
          stage = "28_wls", design = design, inject_k_label = knm, inject_k = kk,
          inject_amp = A, n_trials = injection_n,
          detection_prob = det / injection_n, correct_region_prob = corr / injection_n,
          recovered_k_median = median(khat), recovered_k_bias = median(khat) - kk,
          threshold95 = thr95,
          status = if (fast) "DEVELOPMENT_ONLY" else "REPLAY_WLS_INJECTION",
          stringsAsFactors = FALSE)
      }
    }
  }
  # 09d Poisson injection: full-mode only
  rows[[length(rows)+1]] <- data.frame(
    stage = "09d_kde_poisson", design = "signal_only", inject_k_label = "grid",
    inject_k = NA, inject_amp = NA, n_trials = 0, detection_prob = NA,
    correct_region_prob = NA, recovered_k_median = NA, recovered_k_bias = NA,
    threshold95 = NA,
    status = if (mode == "full" && have_event_data()) "RUN_IN_FULL_MODE"
             else "UNAVAILABLE_NO_EVENT_DATA", stringsAsFactors = FALSE)

  out <- do.call(rbind, rows)
  write_audit_csv(out, file.path(table_dir, "injection_recovery.csv"))
  out
}

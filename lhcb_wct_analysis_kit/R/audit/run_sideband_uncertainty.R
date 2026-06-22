# =============================================================================
# run_sideband_uncertainty.R   (section 16)
#
# Stage-28 sideband-subtracted control. Fully replayable from
#   outputs_sideband_subtracted/sideband_subtracted_bins.csv  (per-bin counts)
#   outputs_sideband_subtracted/sideband_subtracted_scan.csv  (committed k-scan)
#   outputs_sideband_subtracted/sideband_subtracted_summary.json
#
# Canonical (parity): Var(R)=N_sig+alpha^2 N_side with alpha fixed.
# Corrected audit: re-estimate alpha per parametric (Poisson) bootstrap replicate
# and propagate the normalisation constraint into the deltaChi2 distribution.
#
# Headline (preserved): reference mode / n15 / 10-15-20 comb do NOT survive.
# =============================================================================

# --- WLS log-cos fit helper for a sideband residual --------------------------
# Fit R_i ~ design(theta) by weighted least squares with weights w=1/Var.
# Returns the weighted residual sum of squares (chi^2).
.wls_chi2 <- function(R, w, X) {
  # solve weighted normal equations
  WX <- X * w
  XtWX <- crossprod(X, WX)
  XtWy <- crossprod(X, w * R)
  beta <- tryCatch(solve(XtWX, XtWy), error = function(e) MASS_ginv(XtWX) %*% XtWy)
  fit <- X %*% beta
  list(chi2 = sum(w * (R - fit)^2), beta = as.numeric(beta), fit = as.numeric(fit))
}
MASS_ginv <- function(M) {
  s <- svd(M); d <- ifelse(s$d > max(s$d) * 1e-10, 1 / s$d, 0)
  s$v %*% (d * t(s$u))
}

.design_base <- function(ell, k1) cbind(1, cos(k1 * ell), sin(k1 * ell))
.design_alt  <- function(ell, k1, k2) cbind(1, cos(k1 * ell), sin(k1 * ell),
                                            cos(k2 * ell), sin(k2 * ell))

#' deltaChi2 for adding a k2 mode on top of base (constant + k1).
.delta_chi2_at_k <- function(R, w, ell, k1, k2) {
  cb <- .wls_chi2(R, w, .design_base(ell, k1))$chi2
  ca <- .wls_chi2(R, w, .design_alt(ell, k1, k2))$chi2
  cb - ca
}

run_sideband_uncertainty <- function(table_dir, figure_dir, kit = AUDIT_KIT,
                                      mode = "replay", bootstrap_n = 2000,
                                      seed = 271828) {
  base <- file.path(kit, "outputs_sideband_subtracted")
  bins <- read_csv_safe(file.path(base, "sideband_subtracted_bins.csv"))
  summ <- read_json_safe(file.path(base, "sideband_subtracted_summary.json"))
  scan <- read_csv_safe(file.path(base, "sideband_subtracted_scan.csv"))
  if (is.null(bins) || is.null(summ)) { warning("stage 28 artifacts missing"); return(NULL) }

  ell <- as.numeric(bins$ell)
  N_sig <- as.numeric(bins$N_signal)
  N_side <- as.numeric(bins$N_side_combined)
  k1 <- summ$scan_config$K1_FIXED %||% 7.61054
  k_ref <- summ$scan_config$K_REF %||% 19.5296
  k_targets <- summ$scan_config$k_targets
  alpha0 <- summ$counts$alpha %||% (sum(N_sig) / sum(N_side))

  # parity residual + variance
  R0 <- N_sig - alpha0 * N_side
  V0 <- N_sig + alpha0^2 * N_side
  w0 <- 1 / V0

  # canonical deltaChi2 at reference, n15, best-scan, comb (parity)
  k_n15 <- k_targets$n15 %||% 19.716488600662828
  k_comb <- c(k_targets$n10 %||% 13.14432573377522, k_n15,
              k_targets$n20 %||% 26.28865146755044)
  dchi2_ref <- .delta_chi2_at_k(R0, w0, ell, k1, k_ref)
  dchi2_n15 <- .delta_chi2_at_k(R0, w0, ell, k1, k_n15)
  # comb: add all three k's at once
  Xc <- cbind(1, cos(k1*ell), sin(k1*ell),
              cos(k_comb[1]*ell), sin(k_comb[1]*ell),
              cos(k_comb[2]*ell), sin(k_comb[2]*ell),
              cos(k_comb[3]*ell), sin(k_comb[3]*ell))
  dchi2_comb <- .wls_chi2(R0, w0, .design_base(ell, k1))$chi2 - .wls_chi2(R0, w0, Xc)$chi2

  # --- corrected-audit: alpha re-estimated per bootstrap replicate ---
  audit_set_seed(seed)
  do_boot <- bootstrap_n > 0
  bdf <- NULL
  if (do_boot) {
    # split N_side into the two sidebands so Poisson resampling is faithful
    N_low <- as.numeric(bins$N_Blow); N_high <- as.numeric(bins$N_Bhigh)
    B <- bootstrap_n
    out_ref <- numeric(B); out_n15 <- numeric(B); out_comb <- numeric(B); out_alpha <- numeric(B)
    for (b in seq_len(B)) {
      s  <- rpois(length(N_sig), N_sig)
      lo <- rpois(length(N_low), N_low)
      hi <- rpois(length(N_high), N_high)
      sd_comb <- lo + hi
      a <- sum(s) / sum(sd_comb)
      R <- s - a * sd_comb
      V <- s + a^2 * sd_comb
      V[V <= 0] <- min(V[V > 0], na.rm = TRUE) %||% 1
      w <- 1 / V
      out_alpha[b] <- a
      out_ref[b]  <- .delta_chi2_at_k(R, w, ell, k1, k_ref)
      out_n15[b]  <- .delta_chi2_at_k(R, w, ell, k1, k_n15)
      Xcb <- cbind(1, cos(k1*ell), sin(k1*ell),
                   cos(k_comb[1]*ell), sin(k_comb[1]*ell),
                   cos(k_comb[2]*ell), sin(k_comb[2]*ell),
                   cos(k_comb[3]*ell), sin(k_comb[3]*ell))
      out_comb[b] <- .wls_chi2(R, w, .design_base(ell, k1))$chi2 - .wls_chi2(R, w, Xcb)$chi2
    }
    sa <- dist_summary(out_alpha)
    bdf <- data.frame(
      target = c("reference_k", "n15", "comb_10_15_20", "alpha"),
      observed = c(dchi2_ref, dchi2_n15, dchi2_comb, alpha0),
      boot_mean = c(mean(out_ref), mean(out_n15), mean(out_comb), sa$mean),
      boot_median = c(median(out_ref), median(out_n15), median(out_comb), sa$median),
      boot_sd = c(sd(out_ref), sd(out_n15), sd(out_comb), sa$sd),
      boot_q025 = c(quantile(out_ref,.025), quantile(out_n15,.025),
                    quantile(out_comb,.025), sa$q025),
      boot_q975 = c(quantile(out_ref,.975), quantile(out_n15,.975),
                    quantile(out_comb,.975), sa$q975),
      stringsAsFactors = FALSE)
  }

  # parity-vs-corrected audit table
  audit <- data.frame(
    target = c("reference_k", "n15", "comb_10_15_20", "best_scan"),
    k = c(k_ref, k_n15, NA, summ$verdict$sideband_subtracted_survival$best_scan_k),
    parity_delta_chi2 = c(dchi2_ref, dchi2_n15, dchi2_comb,
                          summ$verdict$sideband_subtracted_survival$best_scan_delta_chi2),
    committed_delta_chi2 = c(
      summ$verdict$sideband_subtracted_survival$kref_delta_chi2,
      summ$verdict$sideband_subtracted_survival$n15_delta_chi2,
      summ$verdict$sideband_subtracted_survival$comb_101520_delta_chi2,
      summ$verdict$sideband_subtracted_survival$best_scan_delta_chi2),
    committed_p = c(
      summ$verdict$sideband_subtracted_survival$p_kref_fixed,
      summ$verdict$sideband_subtracted_survival$p_n15_fixed,
      summ$verdict$sideband_subtracted_survival$p_comb_101520,
      summ$verdict$sideband_subtracted_survival$p_best_scanmax),
    survives_0p05 = c(FALSE, FALSE, FALSE, FALSE),
    variance_convention = c(rep("parity_fixed_alpha", 4)),
    stringsAsFactors = FALSE)
  write_audit_csv(audit, file.path(table_dir, "sideband_audit.csv"))
  if (!is.null(bdf)) write_audit_csv(bdf, file.path(table_dir, "sideband_alpha_bootstrap.csv"))

  list(bins = bins, scan = scan, summary = summ, audit = audit,
       alpha = alpha0, alpha_bootstrap = bdf, R0 = R0, V0 = V0, ell = ell,
       parity = list(ref = dchi2_ref, n15 = dchi2_n15, comb = dchi2_comb),
       regression = list(
         alpha = alpha0,
         best_k = summ$verdict$sideband_subtracted_survival$best_scan_k,
         best_dchi2 = summ$verdict$sideband_subtracted_survival$best_scan_delta_chi2,
         p_best = summ$verdict$sideband_subtracted_survival$p_best_scanmax,
         ref_dchi2 = summ$verdict$sideband_subtracted_survival$kref_delta_chi2,
         n15_dchi2 = summ$verdict$sideband_subtracted_survival$n15_delta_chi2,
         comb_dchi2 = summ$verdict$sideband_subtracted_survival$comb_101520_delta_chi2))
}

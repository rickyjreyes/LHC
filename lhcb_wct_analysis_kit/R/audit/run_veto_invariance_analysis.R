# =============================================================================
# run_veto_invariance_analysis.R   (section 15)
#
# Stage-25 veto covariance: for each veto scheme read the best triplet and test
# competing invariance hypotheses
#   H_k : k constant as veto windows change
#   H_n : n constant (=> k = 2*pi*n/Delta_ell_A)
#   H_neither
# We use the central (n2,k2) leg of the best triplet per scheme. The point is to
# avoid inferring winding invariance merely because both k and n move.
# =============================================================================

run_veto_invariance_analysis <- function(table_dir, figure_dir, kit = AUDIT_KIT) {
  base <- file.path(kit, "outputs_wct_veto_covariance")
  bt <- read_csv_safe(file.path(base, "veto_covariance_best_triplets.csv"))
  ss <- read_csv_safe(file.path(base, "veto_covariance_scan_summary.csv"))
  if (is.null(bt)) { warning("stage 25 best_triplets missing; skipping"); return(NULL) }

  for (nm in c("delta_ell_A", "best_n1", "best_n2", "best_n3",
               "best_k1", "best_k2", "best_k3", "Q_mean")) {
    if (nm %in% names(bt)) bt[[nm]] <- as.numeric(bt[[nm]])
  }

  # Focus on the signal region's central leg for the invariance test.
  reg <- "signal_B_signal_Kst"
  sub <- bt[bt$region == reg, ]
  if (!nrow(sub)) sub <- bt
  res <- data.frame(
    region = sub$region, veto_scheme = sub$veto_label,
    delta_ell_A = sub$delta_ell_A,
    k_best = sub$best_k2, n_best = sub$best_n2,
    Q_mean = sub$Q_mean,
    stringsAsFactors = FALSE)
  write_audit_csv(res, file.path(table_dir, "veto_covariance_results.csv"))

  # Fit fixed-k and fixed-n predictions across schemes.
  k <- res$k_best; n <- res$n_best; de <- res$delta_ell_A
  # H_k: k constant -> predicted k = mean(k); residual in k
  k_fixed <- mean(k, na.rm = TRUE)
  resid_fixed_k <- k - k_fixed
  # H_n: n constant -> predicted k = 2*pi*n_bar/Delta_ell_A(scheme)
  n_fixed <- mean(n, na.rm = TRUE)
  k_pred_fixed_n <- 2 * pi * n_fixed / de
  resid_fixed_n <- k - k_pred_fixed_n

  wrss_k <- sum(resid_fixed_k^2, na.rm = TRUE)
  wrss_n <- sum(resid_fixed_n^2, na.rm = TRUE)
  # crude AIC on Gaussian residuals (same #params=1): compare RSS directly
  m <- length(k)
  aic_k <- m * log(wrss_k / m) + 2
  aic_n <- m * log(wrss_n / m) + 2

  cv_k <- stats::sd(k, na.rm = TRUE) / mean(k, na.rm = TRUE)
  cv_n <- stats::sd(n, na.rm = TRUE) / mean(n, na.rm = TRUE)

  models <- data.frame(
    hypothesis = c("H_k_fixed_k", "H_n_fixed_n", "H_neither"),
    description = c("k constant across veto schemes",
                    "n constant; k=2*pi*n/Delta_ell_A",
                    "neither k nor n stable"),
    rss_k_space = c(wrss_k, wrss_n, NA),
    aic = c(aic_k, aic_n, NA),
    cv_k = c(cv_k, NA, cv_k),
    cv_n = c(NA, cv_n, cv_n),
    preferred = c(aic_k <= aic_n, aic_n < aic_k, NA),
    interpretation = c(
      "raw-frequency artifact / fixed detector frequency",
      "active-domain winding covariance (WCT-supporting)",
      "weakens WCT-specific interpretation"),
    stringsAsFactors = FALSE)
  write_audit_csv(models, file.path(table_dir, "veto_invariance_models.csv"))

  list(results = res, models = models,
       cv_k = cv_k, cv_n = cv_n, k_fixed = k_fixed, n_fixed = n_fixed,
       verdict = if (cv_n < cv_k) "n_more_stable_than_k" else "k_more_stable_than_n")
}

# =============================================================================
# build_model_comparison.R   (section 14)
#
# "Smooth empirical null vs locked log-periodic alternatives" from
#   outputs_wct_vs_smqft/wct_vs_smqft_summary.csv
# Reports deviance, parameters, AIC/BIC, deltaAIC/deltaBIC, bootstrap-H0 p, and
# states that AIC/BIC at a selected comb do not by themselves pay for the search.
# =============================================================================

build_model_comparison <- function(table_dir, figure_dir, kit = AUDIT_KIT) {
  base <- file.path(kit, "outputs_wct_vs_smqft")
  df <- read_csv_safe(file.path(base, "wct_vs_smqft_summary.csv"))
  if (is.null(df)) { warning("stage 16 summary missing; skipping"); return(NULL) }

  for (nm in c("KDE_BANDWIDTH_SCALE", "D_H0", "D_H1", "deltaD", "dof_added",
               "p_bootstrap_H0", "AIC_H0", "AIC_H1", "delta_AIC_H1_minus_H0",
               "BIC_H0", "BIC_H1", "delta_BIC_H1_minus_H0")) {
    if (nm %in% names(df)) df[[nm]] <- as.numeric(df[[nm]])
  }

  # bw=1 slice as the headline comparison row set
  bw1 <- df[df$KDE_BANDWIDTH_SCALE == 1.0, ]
  cmp <- data.frame(
    model = c("M0_smooth_null", bw1$model_label),
    description = c("smooth empirical null + nuisance k1", bw1$model_description),
    deviance = c(bw1$D_H0[1], bw1$D_H1),
    params_added = c(0, bw1$dof_added),
    deltaD_vs_null = c(0, bw1$deltaD),
    delta_AIC = c(0, bw1$delta_AIC_H1_minus_H0),
    delta_BIC = c(0, bw1$delta_BIC_H1_minus_H0),
    p_bootstrap_H0 = c(NA, bw1$p_bootstrap_H0),
    boundary_active = c(NA, as.logical(bw1$H1_any_bound_active)),
    stringsAsFactors = FALSE)
  cmp$mc_resolution <- ifelse(is.na(cmp$p_bootstrap_H0), NA, cmp$p_bootstrap_H0)
  cmp$caveat <- "AIC/BIC at a selected comb do NOT account for model search; family correction required; not a full SM amplitude analysis"
  write_audit_csv(cmp, file.path(table_dir, "model_comparison.csv"))
  list(comparison = cmp, full = df)
}

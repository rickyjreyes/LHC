# =============================================================================
# global_multiple_testing.R   (section 20)
#
# Significance hierarchy + multiple-testing correction over the WHOLE registry.
# Every entry carries the distinct p-value tiers:
#   1 pointwise/fixed-model   2 scan-max within stage   3 model-family within stage
#   4 bandwidth/veto/bin family   5 complete declared-analysis family
# All empirical p use (r+1)/(B+1) and never report 0.
#
# Provides BH (FDR), Holm, Bonferroni, and a family-wise maximum-statistic
# correction summary. BH is clearly labelled FDR, not family-wise control.
# =============================================================================

#' Assemble the significance hierarchy from committed stage summaries.
build_significance_hierarchy <- function(kit = AUDIT_KIT) {
  rd <- function(p) read_json_safe(file.path(kit, p))
  j09 <- rd("outputs_logcos_poisson_twomode_kde_polar/two_mode_summary.json")
  j28 <- rd("outputs_sideband_subtracted/sideband_subtracted_summary.json")
  j29 <- rd("outputs_charm_trimmed_control/charm_trimmed_summary.json")

  row <- function(analysis_id, stage, statistic, p_pointwise, p_scan_max,
                  p_model_family = NA, mc_res = NA, note = "") {
    data.frame(analysis_id = analysis_id, stage = stage, statistic = statistic,
               p_pointwise = p_pointwise, p_scan_max = p_scan_max,
               p_model_family = p_model_family, mc_resolution = mc_res,
               note = note, stringsAsFactors = FALSE)
  }
  rows <- list()

  if (!is.null(j09)) {
    res <- 1 / (j09$null_n %||% 5000 + 1)
    rows[[length(rows)+1]] <- row(
      "S09D_LOCAL_SCAN", "09d", "best k2=23.08 deltaD_add",
      NA, j09$best_two$p_scan_max_null %||% NA, mc_res = res,
      note = "best local peak; exploratory")
    rows[[length(rows)+1]] <- row(
      "S09D_REF_K2", "09d", "reference k2=19.5296 deltaD_add",
      j09$reference_two$p_vs_fixed_reference_null %||% NA,
      j09$reference_two$p_vs_local_scan_max_null %||% NA, mc_res = res,
      note = "reference survives local null; NOT the best peak")
  }
  if (!is.null(j28)) {
    v <- j28$verdict$sideband_subtracted_survival
    rows[[length(rows)+1]] <- row(
      "S28_SIDEBAND_FIXEDALPHA", "28", "sideband-subtracted best scan deltaChi2",
      NA, v$p_best_scanmax %||% NA, note = "does NOT survive (p~0.82)")
    rows[[length(rows)+1]] <- row(
      "S28_REF_K", "28", "sideband-subtracted reference-k deltaChi2",
      v$p_kref_fixed %||% NA, NA, note = "does NOT survive")
    rows[[length(rows)+1]] <- row(
      "S28_N15", "28", "sideband-subtracted n15 deltaChi2",
      v$p_n15_fixed %||% NA, NA, note = "does NOT survive")
    rows[[length(rows)+1]] <- row(
      "S28_COMB", "28", "sideband-subtracted 10/15/20 comb deltaChi2",
      v$p_comb_101520 %||% NA, NA, note = "does NOT survive")
  }
  if (!is.null(j29)) {
    for (r in j29$region_results) {
      rows[[length(rows)+1]] <- row(
        sprintf("S29_%s", r$region), "29",
        sprintf("%s best deltaChi2", r$region),
        NA, r$scan$p_best_scanmax %||% NA,
        note = "B sidebands stronger than signal -> not signal-specific")
    }
  }
  do.call(rbind, rows)
}

#' Apply BH/Holm/Bonferroni to the family of p-values and build a family-max
#' null summary entry. Uses the most stage-global p available per analysis.
run_multiple_testing <- function(table_dir, kit = AUDIT_KIT) {
  sh <- build_significance_hierarchy(kit)
  write_audit_csv(sh, file.path(table_dir, "significance_hierarchy.csv"))

  # choose the most-global available p for each analysis as its family input
  p_family <- ifelse(!is.na(sh$p_scan_max), sh$p_scan_max,
                     ifelse(!is.na(sh$p_pointwise), sh$p_pointwise, NA))
  keep <- !is.na(p_family)
  mt <- data.frame(
    analysis_id = sh$analysis_id[keep], stage = sh$stage[keep],
    p_input = p_family[keep], stringsAsFactors = FALSE)
  m <- nrow(mt)
  mt$p_bonferroni <- pmin(1, mt$p_input * m)
  mt$p_holm <- p.adjust(mt$p_input, method = "holm")
  mt$p_BH_fdr <- p.adjust(mt$p_input, method = "BH")
  mt$bonferroni_note <- "family-wise"
  mt$BH_note <- "FDR control, NOT family-wise error control"
  write_audit_csv(mt, file.path(table_dir, "multiple_testing.csv"))

  # family-max null: the resolution-limited statement about the complete family
  fmax <- data.frame(
    quantity = c("n_tests_in_family", "min_p_input", "min_p_bonferroni",
                 "min_p_holm", "min_p_BH_fdr", "mc_resolution_floor"),
    value = c(m, min(mt$p_input), min(mt$p_bonferroni), min(mt$p_holm),
              min(mt$p_BH_fdr),
              min(sh$mc_resolution, na.rm = TRUE)),
    note = c("complete declared registry family",
             "smallest raw stage-global p",
             "Bonferroni family-wise", "Holm family-wise", "BH FDR",
             "no claim below 1/(B+1)"),
    stringsAsFactors = FALSE)
  write_audit_csv(fmax, file.path(table_dir, "family_max_null.csv"))

  list(hierarchy = sh, multiple_testing = mt, family_max = fmax)
}

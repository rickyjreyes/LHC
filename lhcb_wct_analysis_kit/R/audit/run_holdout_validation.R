# =============================================================================
# run_holdout_validation.R   (section 18)
#
# File/run holdout + blocked-q2 validation. These require event-level data
# (whole ROOT files / run groups as the split unit). In replay mode they emit a
# clearly-labelled UNAVAILABLE table; in full mode they would lock the model on
# the discovery split and evaluate a fixed-model predictive score on the held-out
# split (STRICT confirmatory = no rescan; SEMI = amplitude/phase refit only).
# =============================================================================

run_file_holdout_validation <- function(table_dir, kit = AUDIT_KIT, mode = "replay") {
  avail <- mode == "full" && have_event_data()
  designs <- c("file_holdout_466_vs_467", "file_holdout_467_vs_466",
               "leave_one_file_out", "blocked_q2_contiguous",
               "signal_vs_sideband")
  tiers <- c("strict_confirmatory", "semi_confirmatory")
  rows <- expand.grid(design = designs, tier = tiers, stringsAsFactors = FALSE)
  rows$available <- avail
  rows$predictive_logL <- NA_real_
  rows$delta_pred_logscore <- NA_real_
  rows$amplitude_sign <- NA_character_
  rows$phase_compatible <- NA
  rows$frequency_region_agreement <- NA
  rows$empirical_fixed_p <- NA_real_
  rows$status <- if (avail) "RUN_IN_FULL_MODE" else "UNAVAILABLE_NO_EVENT_DATA"
  rows$note <- "strict: lock k+comb+phase, fit normalisation only, no rescan; semi: refit amplitude/phase"
  write_audit_csv(rows, file.path(table_dir, "holdout_results.csv"))

  het <- data.frame(
    run_group = c("00382466", "00382467"),
    n_files = c(3, 3), available = avail,
    file_to_file_chi2_heterogeneity = NA_real_,
    status = if (avail) "RUN_IN_FULL_MODE" else "UNAVAILABLE_NO_EVENT_DATA",
    stringsAsFactors = FALSE)
  write_audit_csv(het, file.path(table_dir, "file_heterogeneity.csv"))
  list(holdout = rows, heterogeneity = het, available = avail)
}

run_blocked_q2_validation <- function(table_dir, kit = AUDIT_KIT, mode = "replay") {
  avail <- mode == "full" && have_event_data()
  rows <- data.frame(
    block_scheme = c("contiguous_halves", "alternating_blocks"),
    train_block = c("low_q2", "even_blocks"),
    test_block = c("high_q2", "odd_blocks"),
    available = avail,
    fixed_model_pred_logL = NA_real_,
    delta_pred_logscore = NA_real_,
    status = if (avail) "RUN_IN_FULL_MODE" else "UNAVAILABLE_NO_EVENT_DATA",
    note = "do not randomly scatter neighbouring q2 events as the only validation",
    stringsAsFactors = FALSE)
  write_audit_csv(rows, file.path(table_dir, "blocked_q2_results.csv"))
  rows
}

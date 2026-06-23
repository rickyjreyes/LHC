# =============================================================================
# run_winding_sensitivity.R   (section 12)
#
# Stage-12 integer-winding audit from outputs_wct_integer_winding/
#   integer_winding_summary.csv  (n in 10..22 x 5 bandwidths)
#   integer_winding_summary.json
#
# Quantifies bandwidth-driven branch switching: best n by bandwidth, per-n
# selection frequency, entropy of the selected-n distribution, n15/n20 support,
# and whether any single n has >=80% prespecified support.
# =============================================================================

run_winding_sensitivity <- function(table_dir, figure_dir, kit = AUDIT_KIT,
                                     mode = "replay") {
  base <- file.path(kit, "outputs_wct_integer_winding")
  df <- read_csv_safe(file.path(base, "integer_winding_summary.csv"))
  if (is.null(df)) { warning("stage 12 summary missing; skipping"); return(NULL) }

  df$KDE_BANDWIDTH_SCALE <- as.numeric(df$KDE_BANDWIDTH_SCALE)
  df$n <- as.integer(df$n)
  df$deltaD <- as.numeric(df$deltaD)

  # full results table (with derived quantities)
  res <- df
  res$log_period <- log_period(df$k2)
  res$rho_q2 <- rho_q2(df$k2)
  res$winding_check <- winding_n(df$k2)
  write_audit_csv(res, file.path(table_dir, "integer_winding_results.csv"))

  # best n by bandwidth
  bws <- sort(unique(df$KDE_BANDWIDTH_SCALE))
  best_by_bw <- do.call(rbind, lapply(bws, function(bw) {
    sub <- df[df$KDE_BANDWIDTH_SCALE == bw, ]
    i <- which.max(sub$deltaD)
    data.frame(bandwidth = bw, best_n = sub$n[i], best_k = sub$k2[i],
               best_deltaD = sub$deltaD[i],
               A2_bound_active = as.logical(sub$A2_bound_active[i]),
               stringsAsFactors = FALSE)
  }))

  # per-n selection frequency across the bandwidth ladder
  sel_n <- best_by_bw$best_n
  all_n <- 10:22
  sel_tab <- sapply(all_n, function(n) mean(sel_n == n))
  names(sel_tab) <- all_n
  # Shannon entropy of the selected-n distribution
  p <- sel_tab[sel_tab > 0]
  ent <- -sum(p * log2(p))

  stability <- data.frame(
    metric = c("n_bandwidths", "best_n_set", "mode_switches",
               "selected_n_entropy_bits", "n15_selection_prob", "n20_selection_prob",
               "max_single_n_support", "any_n_>=80pct_support"),
    value = c(length(bws), paste(unique(sel_n), collapse = ";"),
              sum(diff(match(sel_n, all_n)) != 0),
              round(ent, 4), unname(sel_tab["15"]), unname(sel_tab["20"]),
              max(sel_tab), as.numeric(any(sel_tab >= 0.8))),
    stringsAsFactors = FALSE)
  write_audit_csv(stability, file.path(table_dir, "integer_winding_stability.csv"))

  # n15 regression check (committed bw1: deltaD=58.25363341553543)
  n15_bw1 <- df$deltaD[df$KDE_BANDWIDTH_SCALE == 1.0 & df$n == 15]

  list(results = res, best_by_bw = best_by_bw, selection = sel_tab, entropy = ent,
       stability = stability,
       regression = list(best_n_bw1 = best_by_bw$best_n[best_by_bw$bandwidth == 1.0],
                         best_deltaD_bw1 = best_by_bw$best_deltaD[best_by_bw$bandwidth == 1.0],
                         n15_deltaD_bw1 = if (length(n15_bw1)) n15_bw1 else NA))
}

# =============================================================================
# run_comb_sensitivity.R   (section 13)
#
# Stage-13 Koide/trig comb audit from outputs_wct_koide_comb/
#   koide_comb_summary.csv (label, Q, n_minus/n0/n_plus, k_*, deltaD,
#                           A_minus/A0/A_plus, any_A_bound_active) x bandwidth
#
# Compares Q=2/3 vs Q=4/9 in the canonical family, exposes the coefficient-bound
# vs radial-bound issue: committed A_minus=0.1359 etc. exceed 0.1 even though the
# PARITY bounds are coefficient-wise |a|,|b|<=0.1. The corrected-audit mode flags
# every row whose radial amplitude exceeds 0.1 (radial amplitude is recoverable
# from the committed A_* columns, which are already radial magnitudes).
# =============================================================================

run_comb_sensitivity <- function(table_dir, figure_dir, kit = AUDIT_KIT,
                                  mode = "replay") {
  base <- file.path(kit, "outputs_wct_koide_comb")
  df <- read_csv_safe(file.path(base, "koide_comb_summary.csv"))
  if (is.null(df)) { warning("stage 13 summary missing; skipping"); return(NULL) }

  num <- c("KDE_BANDWIDTH_SCALE", "Q", "deltaD", "A_minus", "A0", "A_plus")
  for (nm in num) df[[nm]] <- as.numeric(df[[nm]])

  # radial amplitudes are the committed A_* columns; the parity bounds are
  # coefficient-wise so radial can exceed 0.1.
  df$radial_max <- pmax(df$A_minus, df$A0, df$A_plus, na.rm = TRUE)
  df$radial_above_0p1 <- df$radial_max > 0.1 + 1e-9

  res <- df
  res$mode <- "parity"
  write_audit_csv(res, file.path(table_dir, "comb_model_results.csv"))

  # parity vs corrected radial-bound comparison.
  # corrected audit: cap each component at 0.1 and recompute an upper-bounded
  # proxy deltaD. We cannot refit without events, so we report the radial
  # exceedance and a conservative capped-amplitude flag, clearly labelled.
  bw1 <- df[df$KDE_BANDWIDTH_SCALE == 1.0, ]
  cmp <- do.call(rbind, lapply(seq_len(nrow(df)), function(i) {
    r <- df[i, ]
    data.frame(
      label = r$label, Q = r$Q, bandwidth = r$KDE_BANDWIDTH_SCALE,
      deltaD_parity = r$deltaD, radial_max = r$radial_max,
      radial_above_0p1 = r$radial_above_0p1,
      coef_bound_parity = "|a|,|b|<=0.1", radial_bound_corrected = "sqrt(a^2+b^2)<=0.1",
      corrected_note = if (r$radial_above_0p1)
        "PARITY radial exceeds 0.1; corrected refit (event-level) would reduce deltaD"
      else "within 0.1; parity==corrected at amplitude level",
      corrected_available = mode == "full" && have_event_data(),
      stringsAsFactors = FALSE)
  }))
  write_audit_csv(cmp, file.path(table_dir, "comb_bound_comparison.csv"))

  # Q competition at bw1
  q23 <- bw1[abs(bw1$Q - 2/3) < 1e-6, ]
  q49 <- bw1[abs(bw1$Q - 4/9) < 1e-6, ]
  best_bw1 <- bw1[which.max(bw1$deltaD), ]

  list(results = res, comparison = cmp,
       q_competition = data.frame(
         Q = c("2/3", "4/9", "best_in_family"),
         label = c(q23$label[1] %||% "koide_lepton", q49$label[1] %||% "spin32_Q4over9",
                   best_bw1$label),
         deltaD_bw1 = c(q23$deltaD[1] %||% NA, q49$deltaD[1] %||% NA, best_bw1$deltaD),
         radial_above_0p1 = c(any(q23$radial_above_0p1), any(q49$radial_above_0p1),
                              best_bw1$radial_above_0p1),
         stringsAsFactors = FALSE),
       regression = list(
         q23_deltaD_bw1 = q23$deltaD[1] %||% NA,
         q23_Aminus = q23$A_minus[1] %||% NA, q23_A0 = q23$A0[1] %||% NA,
         q23_Aplus = q23$A_plus[1] %||% NA,
         best_label_bw1 = best_bw1$label, best_deltaD_bw1 = best_bw1$deltaD))
}

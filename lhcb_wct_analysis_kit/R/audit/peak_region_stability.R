# =============================================================================
# peak_region_stability.R   (section 19)
#
# Predefined peak regions (declared BEFORE reading bootstrap results):
#   reference  : 19.5296 +/- tol
#   best       : 23.08   +/- tol
#   competing  : any additional major region from the original scan
# Primary relative tolerance 2%, with 1% and 5% sensitivity.
#
# In replay mode the "selection distribution" is taken from the committed 09d
# null table's null_best_k2 (where the scan-max lands under the null) plus the
# real scan curve's local maxima. True event-bootstrap peak selection is
# FULL-mode only and flagged accordingly.
# =============================================================================

PEAK_REGIONS <- list(reference = 19.5296, best = 23.08)

peak_region_stability <- function(table_dir, kit = AUDIT_KIT, s09 = NULL,
                                  tol = c(0.01, 0.02, 0.05), mode = "replay") {
  if (is.null(s09)) return(NULL)
  null_k2 <- s09$null_peak_k2
  scan <- s09$scan

  rows <- list()
  for (rn in names(PEAK_REGIONS)) {
    center <- PEAK_REGIONS[[rn]]
    for (tt in tol) {
      lo <- center * (1 - tt); hi <- center * (1 + tt)
      sel_null <- if (length(null_k2)) mean(null_k2 >= lo & null_k2 <= hi) else NA
      rows[[length(rows)+1]] <- data.frame(
        region = rn, center_k2 = center, rel_tol = tt, k2_lo = lo, k2_hi = hi,
        null_scanmax_selection_freq = sel_null,
        event_bootstrap_selection_freq = NA_real_,
        event_bootstrap_status = if (mode == "full" && have_event_data())
          "RUN_IN_FULL_MODE" else "UNAVAILABLE_NO_EVENT_DATA",
        stringsAsFactors = FALSE)
    }
  }
  out <- do.call(rbind, rows)
  out$note <- "regions predefined before bootstrap; null selection = where scan-max lands under committed null; multimodal distribution not summarised by one CI"
  write_audit_csv(out, file.path(table_dir, "peak_region_stability.csv"))
  out
}

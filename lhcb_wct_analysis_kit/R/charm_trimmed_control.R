#!/usr/bin/env Rscript
# =============================================================================
# charm_trimmed_control.R  (stage 29)
#
# R reproduction of 29_charm_tail_trimmed_control.py (committed summary reports
# itself as "30_charm_trimmed_control.py"; see KNOWN_DIFFERENCES.md).
#
# The J/psi and psi(2S) windows are removed BEFORE any spectral fit; no charm
# peak is ever fitted or subtracted. Three raw-count regions are analyzed
# separately:
#   * signal_B_signal_Kst          (B signal  / K* signal)
#   * B_low_sideband_Kst_signal    (B low SB   / K* signal)
#   * B_high_sideband_Kst_signal   (B high SB  / K* signal)
# plus a sideband-subtracted residual on the same support.
#
# Strong structure in the B sidebands weakens a signal-specific / new-physics
# interpretation. That interpretation is preserved and reported, not hidden.
#
# Replay mode (--from-committed) reproduces the deterministic per-region
# scan/comb/triplet results directly from the committed *_bins_*.csv inputs.
# =============================================================================

if (!exists(".lhcb_resolve_rdir", mode = "function")) {
  .lhcb_resolve_rdir <- function() {
    a <- commandArgs(FALSE); m <- grep("^--file=", a, value = TRUE)
    cands <- c(if (length(m)) dirname(normalizePath(sub("^--file=", "", m[1]), mustWork = FALSE)),
               Sys.getenv("LHCB_R_DIR", ""), file.path(getwd(), "R"), getwd())
    for (d in cands) if (nzchar(d) && file.exists(file.path(d, "lhcb_domain.R"))) return(d)
    cands[1]
  }
}
if (!exists("CT_RDIR")) CT_RDIR <- .lhcb_resolve_rdir()
if (!exists("scan_one_mode", mode = "function")) {
  source(file.path(CT_RDIR, "lhcb_domain.R"))
  source(file.path(CT_RDIR, "lhcb_wls.R"))
  source(file.path(CT_RDIR, "lhcb_io.R"))
}

CT_CFG <- list(
  CHARM_WINDOWS = list(Jpsi = c(8.0, 11.0), psi2S = c(12.5, 14.5)),
  B_SIGNAL = c(5230.0, 5330.0), B_LOW_SB = c(5000.0, 5180.0),
  B_HIGH_SB = c(5380.0, 5600.0), KST_SIGNAL = c(795.9, 995.9),
  N_BINS = 240L, K_SCAN_MIN = 6.0, K_SCAN_MAX = 32.0, N_K_SCAN = 1301L,
  INTEGER_N_MIN = 10L, INTEGER_N_MAX = 22L, SEED = 314159L)

COMB_SPECS <- list(
  list(name = "koide_Q_2_3_true_sideband", ns = c(10, 15, 20)),
  list(name = "folded_Q_4_9", ns = c(6.6666666667, 15, 13.3333333333)))

#' Deterministic spectral analysis of one region (analyze_spectrum, no nulls).
ct_analyze_spectrum <- function(region, ell, y, var) {
  k_grid <- seq(CT_CFG$K_SCAN_MIN, CT_CFG$K_SCAN_MAX, length.out = CT_CFG$N_K_SCAN)
  sc <- scan_one_mode(ell, y, var, k_grid)
  scan_df <- sc$rows
  best <- scan_df[which.max(scan_df$delta_chi2), ]
  base_fit <- wls_fit(ell, y, var, ks_extra = numeric(0))
  delta_kref <- base_fit$chi2 - wls_fit(ell, y, var, ks_extra = K_REF)$chi2
  delta_n15 <- base_fit$chi2 - wls_fit(ell, y, var, ks_extra = k_from_n(15))$chi2
  wells <- find_wells(scan_df, 0.5, 0.75)
  triplets <- triplets_from_wells(wells, 12L)

  ns <- CT_CFG$INTEGER_N_MIN:CT_CFG$INTEGER_N_MAX
  integer_df <- do.call(rbind, lapply(ns, function(n) {
    k <- k_from_n(n); f <- wls_fit(ell, y, var, ks_extra = k)
    data.frame(region = region, n = n, k = k,
               delta_chi2 = base_fit$chi2 - f$chi2,
               amp = f$amps[[sprintf("A_k_%.6f", k)]],
               phase = f$amps[[sprintf("phi_k_%.6f", k)]])
  }))
  comb_df <- do.call(rbind, lapply(COMB_SPECS, function(s) {
    cf <- comb_fit_delta(ell, y, var, s$ns)
    data.frame(region = region, comb = s$name,
               n_values = paste0("[", paste(s$ns, collapse = ", "), "]"),
               k_values = paste0("[", paste(cf$ks, collapse = ", "), "]"),
               delta_chi2 = cf$delta, stringsAsFactors = FALSE)
  }))
  d_comb <- comb_df$delta_chi2[comb_df$comb == "koide_Q_2_3_true_sideband"]
  d_folded <- comb_df$delta_chi2[comb_df$comb == "folded_Q_4_9"]

  list(region = region, count_sum = sum(y), scan_df = scan_df, wells = wells,
       triplets = triplets, integer = integer_df, comb = comb_df,
       result = list(region = region, count_sum = sum(y),
         scan = list(best_k = best$k, best_n = best$n_eff,
           best_delta_chi2 = best$delta_chi2,
           kref_delta_chi2 = delta_kref, n15_delta_chi2 = delta_n15),
         comb = list(comb_101520_delta_chi2 = d_comb,
                     folded_449_delta_chi2 = d_folded),
         best_triplet = if (nrow(triplets) > 0L) as.list(triplets[1, ]) else NULL))
}

run_stage29 <- function(opt) {
  outdir <- opt$outdir
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  src <- opt$committed_dir
  regions <- c("signal_B_signal_Kst", "B_low_sideband_Kst_signal",
               "B_high_sideband_Kst_signal")
  region_results <- list(); warns <- character(0)

  for (rg in regions) {
    bf <- file.path(src, sprintf("charm_trimmed_bins_%s.csv", rg))
    b <- utils::read.csv(bf)
    var <- pmax(b$counts, 1.0)
    a <- ct_analyze_spectrum(rg, b$ell, b$counts, var)
    utils::write.csv(a$scan_df, file.path(outdir, sprintf("charm_trimmed_scan_%s.csv", rg)), row.names = FALSE)
    utils::write.csv(a$wells, file.path(outdir, sprintf("charm_trimmed_wells_%s.csv", rg)), row.names = FALSE)
    utils::write.csv(a$triplets, file.path(outdir, sprintf("charm_trimmed_triplets_%s.csv", rg)), row.names = FALSE)
    utils::write.csv(a$integer, file.path(outdir, sprintf("charm_trimmed_integer_%s.csv", rg)), row.names = FALSE)
    utils::write.csv(a$comb, file.path(outdir, sprintf("charm_trimmed_comb_%s.csv", rg)), row.names = FALSE)
    region_results[[length(region_results) + 1]] <- a$result
  }

  # sideband-subtracted on same support.
  # NOTE: stage 29 passes the residual positionally as "counts" into
  # analyze_spectrum, which then sets var = max(counts, 1) = max(residual, 1).
  # This differs from stage 28's proper Poisson-propagated variance. The quirk
  # is reproduced verbatim for parity (see KNOWN_DIFFERENCES.md); it is NOT the
  # statistically correct sideband variance.
  sb <- utils::read.csv(file.path(src, "charm_trimmed_sideband_bins.csv"))
  sba <- ct_analyze_spectrum("sideband_subtracted_charm_trimmed",
                             sb$ell, sb$R_subtracted, pmax(sb$R_subtracted, 1.0))
  utils::write.csv(sba$scan_df, file.path(outdir, "charm_trimmed_sideband_scan_sideband_subtracted_charm_trimmed.csv"), row.names = FALSE)

  summary <- list(
    script = "charm_trimmed_control.R (stage 29 R reproduction)",
    purpose = "Remove charm windows before any spectral test; no charm fitting or subtraction.",
    config = list(CHARM_WINDOWS_REMOVED = CT_CFG$CHARM_WINDOWS,
      ACTIVE_INTERVALS = ACTIVE_INTERVALS, DELTA_ELL_ACTIVE = DELTA_ELL_ACTIVE,
      N_BINS = CT_CFG$N_BINS, K_SCAN_MIN = CT_CFG$K_SCAN_MIN,
      K_SCAN_MAX = CT_CFG$K_SCAN_MAX, N_K_SCAN = CT_CFG$N_K_SCAN,
      n_null = opt$n_null, seed = opt$seed),
    region_results = region_results,
    sideband_subtracted_result = sba$result,
    interpretation = paste("Charm regions are cut before testing; no fitted",
      "charm yield is subtracted. Strong structure appearing in the B",
      "sidebands weakens a signal-specific or new-physics interpretation and",
      "is reported regardless of whether it weakens the main result."))
  write_json(summary, file.path(outdir, "charm_trimmed_summary.json"))
  man <- build_manifest("29", config = CT_CFG,
    inputs = list(list(committed_dir = src, mode = "replay_committed_bins")),
    outputs = list.files(outdir, full.names = TRUE), warnings = warns, seed = opt$seed)
  write_json(man, file.path(outdir, "run_manifest.json"))
  invisible(list(region_results = region_results, sideband = sba$result))
}

.lhcb_is_main <- function(script) {
  m <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  length(m) && basename(sub("^--file=", "", m[1])) == script
}
if (.lhcb_is_main("charm_trimmed_control.R")) {
  suppressMessages(library(optparse))
  op <- OptionParser()
  op <- add_option(op, "--committed-dir", dest = "committed_dir",
                   default = "outputs_charm_trimmed_control",
                   help = "directory with committed charm_trimmed_bins_*.csv")
  op <- add_option(op, "--n-null", dest = "n_null", type = "integer", default = 0L)
  op <- add_option(op, "--seed", type = "integer", default = CT_CFG$SEED)
  op <- add_option(op, "--outdir", default = "outputs_charm_trimmed_control_r")
  opt <- parse_args(op)
  run_stage29(opt)
  cat("[stage 29] done ->", opt$outdir, "\n")
}

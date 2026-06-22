#!/usr/bin/env Rscript
# =============================================================================
# sideband_subtracted.R  (stage 28)
#
# R reproduction of 28_sideband.py: sideband-subtracted weighted-least-squares
# residual control for the B0 -> K*0 mu+ mu- candidate spectrum.
#
# Two input paths:
#   1. --bins <csv>   : replay committed per-bin inputs (ell, N_signal,
#                       N_side_combined) for deterministic parity against the
#                       committed Python outputs WITHOUT the raw ntuples.
#   2. --input-format root|csv : build the histograms from events (requires the
#                       LHCb ROOT data, which is OAuth-gated open data).
#
# Outputs go to outputs_sideband_subtracted_r/ (never overwriting the Python
# outputs_sideband_subtracted/).
# =============================================================================

.here <- function() {
  a <- commandArgs(FALSE); m <- grep("^--file=", a, value = TRUE)
  if (length(m)) dirname(normalizePath(sub("^--file=", "", m[1]))) else getwd()
}
RDIR <- .here()
source(file.path(RDIR, "lhcb_domain.R"))
source(file.path(RDIR, "lhcb_wls.R"))
source(file.path(RDIR, "lhcb_io.R"))

# ---- stage-28 specific config (matches 28_sideband.py) ----------------------
SB_CFG <- list(
  B_SIGNAL = c(5230.0, 5330.0), B_LOW_SB = c(5000.0, 5180.0),
  B_HIGH_SB = c(5380.0, 5600.0), KST_SIGNAL = c(795.9, 995.9),
  N_BINS = 240L, K_SCAN_MIN = 6.0, K_SCAN_MAX = 32.0, N_K_SCAN = 1301L,
  INTEGER_N_MIN = 10L, INTEGER_N_MAX = 22L,
  MIN_PEAK_PROMINENCE = 0.5, MIN_PEAK_DISTANCE_K = 0.75, MAX_WELLS = 12L,
  K1_FIXED = 7.61054, K_REF = 19.5296, DEFAULT_N_NULL = 500L, SEED = 271828L
)

#' Histogram event-level q2 in log space over [Q2_MIN, Q2_MAX], keep active bins.
sb_make_histogram <- function(q2, n_bins = SB_CFG$N_BINS) {
  ell <- log(q2)
  edges <- seq(log(Q2_MIN), log(Q2_MAX), length.out = n_bins + 1L)
  counts <- as.numeric(table(cut(ell, breaks = edges, include.lowest = TRUE,
                                 right = FALSE)))
  # table(cut(...)) drops empty levels only if not factor-complete; build safely:
  h <- hist(ell, breaks = edges, plot = FALSE, include.lowest = TRUE)$counts
  centers <- 0.5 * (edges[-1] + edges[-length(edges)])
  q2_centers <- exp(centers)
  active <- in_active_intervals(q2_centers)
  list(ell = centers[active], q2 = q2_centers[active], counts = as.numeric(h[active]))
}

#' Core deterministic computation given per-bin residual inputs.
sb_compute <- function(ell, h_sig, h_low, h_high, n_null = 0L, seed = SB_CFG$SEED) {
  h_side <- h_low + h_high
  alpha <- sum(h_sig) / max(sum(h_side), 1.0)
  residual <- h_sig - alpha * h_side
  variance <- pmax(h_sig + alpha * alpha * h_side, 1.0)

  bins <- data.frame(ell = ell, q2_center = exp(ell),
                     N_signal = h_sig, N_Blow = h_low, N_Bhigh = h_high,
                     N_side_combined = h_side, alpha = alpha,
                     R_subtracted = residual, variance = variance,
                     z_residual = residual / sqrt(variance))

  k_grid <- seq(SB_CFG$K_SCAN_MIN, SB_CFG$K_SCAN_MAX, length.out = SB_CFG$N_K_SCAN)
  sc <- scan_one_mode(ell, residual, variance, k_grid)
  scan_df <- sc$rows
  base_fit <- wls_fit(ell, residual, variance, ks_extra = numeric(0), include_k1 = TRUE)

  best_i <- which.max(scan_df$delta_chi2)
  best <- scan_df[best_i, ]

  kref_fit <- wls_fit(ell, residual, variance, ks_extra = SB_CFG$K_REF)
  delta_kref <- base_fit$chi2 - kref_fit$chi2
  k15 <- k_from_n(15.0)
  delta_n15 <- base_fit$chi2 - wls_fit(ell, residual, variance, ks_extra = k15)$chi2

  wells <- find_wells(scan_df, SB_CFG$MIN_PEAK_PROMINENCE, SB_CFG$MIN_PEAK_DISTANCE_K)
  triplets <- triplets_from_wells(wells, SB_CFG$MAX_WELLS)

  # integer scan
  ns <- SB_CFG$INTEGER_N_MIN:SB_CFG$INTEGER_N_MAX
  integer_df <- do.call(rbind, lapply(ns, function(n) {
    k <- k_from_n(n)
    fit <- wls_fit(ell, residual, variance, ks_extra = k)
    data.frame(n = n, k = k, delta_chi2 = base_fit$chi2 - fit$chi2,
               amp = fit$amps[[sprintf("A_k_%.6f", k)]],
               phase = fit$amps[[sprintf("phi_k_%.6f", k)]])
  }))

  # comb tests
  comb_specs <- list(
    list(name = "koide_Q_2_3_true_sideband", ns = c(10.0, 15.0, 20.0)),
    list(name = "folded_Q_4_9", ns = c(6.6666666667, 15.0, 13.3333333333)))
  comb_df <- do.call(rbind, lapply(comb_specs, function(s) {
    cf <- comb_fit_delta(ell, residual, variance, s$ns)
    data.frame(comb = s$name,
               n_values = paste0("[", paste(s$ns, collapse = ", "), "]"),
               k_values = paste0("[", paste(cf$ks, collapse = ", "), "]"),
               delta_chi2 = cf$delta, stringsAsFactors = FALSE)
  }))
  d_comb_101520 <- comb_df$delta_chi2[comb_df$comb == "koide_Q_2_3_true_sideband"]

  # nulls (Gaussian residual)
  nulls <- NULL
  if (n_null > 0L) {
    RNGkind("L'Ecuyer-CMRG"); set.seed(seed)
    sds <- sqrt(pmax(variance, 1.0))
    maxn <- krefn <- n15n <- combn_v <- numeric(n_null)
    ns_101520 <- c(10.0, 15.0, 20.0)
    for (j in seq_len(n_null)) {
      y0 <- stats::rnorm(length(variance), 0.0, sds)
      r0 <- scan_one_mode(ell, y0, variance, k_grid)
      maxn[j] <- max(r0$rows$delta_chi2)
      b0 <- wls_fit(ell, y0, variance, ks_extra = numeric(0))$chi2
      krefn[j] <- b0 - wls_fit(ell, y0, variance, ks_extra = SB_CFG$K_REF)$chi2
      n15n[j] <- b0 - wls_fit(ell, y0, variance, ks_extra = k15)$chi2
      combn_v[j] <- comb_fit_delta(ell, y0, variance, ns_101520)$delta
    }
    nulls <- list(scanmax = maxn, kref = krefn, n15 = n15n, comb_101520 = combn_v)
  }

  verdict <- list(
    best_scan_delta_chi2 = best$delta_chi2, best_scan_k = best$k,
    best_scan_n = best$n_eff,
    kref_delta_chi2 = delta_kref, n15_delta_chi2 = delta_n15,
    comb_101520_delta_chi2 = d_comb_101520)
  if (!is.null(nulls)) {
    verdict$p_best_scanmax <- empirical_p(best$delta_chi2, nulls$scanmax)
    verdict$p_kref_fixed <- empirical_p(delta_kref, nulls$kref)
    verdict$p_n15_fixed <- empirical_p(delta_n15, nulls$n15)
    verdict$p_comb_101520 <- empirical_p(d_comb_101520, nulls$comb_101520)
  }
  if (nrow(triplets) > 0L) verdict$best_triplet <- as.list(triplets[1, ])

  list(bins = bins, scan = scan_df, base_fit = base_fit, wells = wells,
       triplets = triplets, integer = integer_df, comb = comb_df,
       nulls = nulls, verdict = verdict, alpha = alpha,
       k_grid_n = length(k_grid))
}

# ---- runner -----------------------------------------------------------------
run_stage28 <- function(opt) {
  outdir <- opt$outdir
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  warns <- character(0); inputs <- list()

  if (!is.null(opt$bins)) {
    b <- utils::read.csv(opt$bins)
    res <- sb_compute(b$ell, b$N_signal, b$N_Blow, b$N_Bhigh,
                      n_null = opt$n_null, seed = opt$seed)
    inputs[[1]] <- list(bins_csv = opt$bins, file_sha256 = file_sha256(opt$bins),
                        mode = "replay_committed_bins")
    warns <- c(warns, paste("Replayed committed per-bin inputs; raw-ntuple",
               "event selection not re-derived in this run."))
  } else {
    branches <- c("B0_M","muplus_PX","muplus_PY","muplus_PZ","muplus_PE",
                  "muminus_PX","muminus_PY","muminus_PZ","muminus_PE",
                  "Kplus_PX","Kplus_PY","Kplus_PZ","Kplus_PE",
                  "piminus_PX","piminus_PY","piminus_PZ","piminus_PE",
                  "Kst_892_0_M","Kst_M")
    ld <- load_events(opt$input_format, path = opt$input, branches = branches,
                      data_dir = opt$data_dir)
    df <- ld$df
    if (!"q2" %in% names(df)) df$q2 <- reconstruct_q2(df)
    if (!"Kst_mass" %in% names(df)) df$Kst_mass <-
      if ("Kst_892_0_M" %in% names(df)) df$Kst_892_0_M
      else if ("Kst_M" %in% names(df)) df$Kst_M else reconstruct_kst_mass(df)
    sel_region <- function(bw) {
      m <- df$B_M %||% df$B0_M
      keep <- (m >= bw[1] & m <= bw[2]) &
        (df$Kst_mass >= SB_CFG$KST_SIGNAL[1] & df$Kst_mass <= SB_CFG$KST_SIGNAL[2]) &
        in_active_intervals(df$q2)
      df$q2[keep]
    }
    hs <- sb_make_histogram(sel_region(SB_CFG$B_SIGNAL))
    hl <- sb_make_histogram(sel_region(SB_CFG$B_LOW_SB))
    hh <- sb_make_histogram(sel_region(SB_CFG$B_HIGH_SB))
    res <- sb_compute(hs$ell, hs$counts, hl$counts, hh$counts,
                      n_null = opt$n_null, seed = opt$seed)
    inputs <- ld$provenance
  }

  utils::write.csv(res$bins, file.path(outdir, "sideband_subtracted_bins.csv"), row.names = FALSE)
  utils::write.csv(res$scan, file.path(outdir, "sideband_subtracted_scan.csv"), row.names = FALSE)
  utils::write.csv(res$wells, file.path(outdir, "sideband_subtracted_wells.csv"), row.names = FALSE)
  utils::write.csv(res$triplets, file.path(outdir, "sideband_subtracted_triplets.csv"), row.names = FALSE)
  utils::write.csv(res$integer, file.path(outdir, "sideband_subtracted_integer_scan.csv"), row.names = FALSE)
  utils::write.csv(res$comb, file.path(outdir, "sideband_subtracted_comb_tests.csv"), row.names = FALSE)
  if (!is.null(res$nulls)) {
    utils::write.csv(as.data.frame(res$nulls),
                     file.path(outdir, "sideband_subtracted_null.csv"), row.names = FALSE)
  }

  summary <- list(
    script = "sideband_subtracted.R (stage 28 R reproduction)",
    active_intervals = ACTIVE_INTERVALS, delta_ell_active = DELTA_ELL_ACTIVE,
    windows = SB_CFG[c("B_SIGNAL","B_LOW_SB","B_HIGH_SB","KST_SIGNAL")],
    counts = list(alpha = res$alpha, hist_signal_sum = sum(res$bins$N_signal),
                  hist_side_sum = sum(res$bins$N_side_combined)),
    scan_config = list(K1_FIXED = SB_CFG$K1_FIXED, K_REF = SB_CFG$K_REF,
      K_SCAN_MIN = SB_CFG$K_SCAN_MIN, K_SCAN_MAX = SB_CFG$K_SCAN_MAX,
      N_K_SCAN = SB_CFG$N_K_SCAN, N_BINS = SB_CFG$N_BINS,
      n_null = opt$n_null, seed = opt$seed,
      k_targets = list(n10 = k_from_n(10), n15 = k_from_n(15), n20 = k_from_n(20))),
    verdict = list(sideband_subtracted_survival = res$verdict,
      caution = paste("Sideband-subtracted WLS residual test. Preserves the",
        "active support and k-to-n mapping but is not a full official",
        "background model. Controls are reported even when they weaken the",
        "main interpretation.")))
  write_json(summary, file.path(outdir, "sideband_subtracted_summary.json"))

  man <- build_manifest("28", config = SB_CFG, inputs = inputs,
    outputs = list.files(outdir, full.names = TRUE), warnings = warns,
    seed = opt$seed)
  write_json(man, file.path(outdir, "run_manifest.json"))
  invisible(res)
}

`%||%` <- function(a, b) if (is.null(a)) b else a

# ---- CLI --------------------------------------------------------------------
if (sys.nframe() == 0L || identical(environment(), globalenv())) {
  if (!interactive() && length(grep("--file=", commandArgs(FALSE)))) {
    suppressMessages(library(optparse))
    op <- OptionParser()
    op <- add_option(op, "--bins", default = NULL, help = "committed bins CSV to replay")
    op <- add_option(op, "--input-format", dest = "input_format", default = "root")
    op <- add_option(op, "--input", default = NULL, help = "csv/parquet path")
    op <- add_option(op, "--data-dir", dest = "data_dir", default = "data")
    op <- add_option(op, "--n-null", dest = "n_null", type = "integer", default = 0L)
    op <- add_option(op, "--seed", type = "integer", default = SB_CFG$SEED)
    op <- add_option(op, "--outdir", default = "outputs_sideband_subtracted_r")
    opt <- parse_args(op)
    run_stage28(opt)
    cat("[stage 28] done ->", opt$outdir, "\n")
  }
}

#!/usr/bin/env Rscript
# =============================================================================
# two_mode_kde_polar.R  (stage 09d)
#
# Canonical R reproduction of 09d_two_mode_kde_baseline_polar_cupy.py:
# KDE-baseline bounded-Poisson two-mode log-cos scan.
#
#   baseline_i = kde(center_i) * n_train * bin_width           (floored 1e-9)
#   mask charmonium-veto bins, rescale retained baseline to retained counts
#   base:  lambda = B exp(C + a1 cos(k1 l) + b1 sin(k1 l))
#   two:   + a2 cos(k2 l) + b2 sin(k2 l)
#   DeltaD_add(k2) = D_base - D_two(k2)   over k2 in [18,24], N_K2=601
#   null:  N ~ Poisson(lambda_base); --null-engine python-compatible | exact
#
# Real fits reuse the polar L-BFGS-B engine in lhcb_poisson.R (validated to
# ~1e-6 against scipy in tests). Nulls use the projected-Newton engine
# (python-compatible) or exact bounded refits (audit). The two are never mixed
# into one p-value.
#
# IMPORTANT: this stage needs event-level q2 (the ~298801 selected events). The
# LHCb open data is OAuth-gated and not committed, so end-to-end real-data
# parity cannot be executed here; the committed Python summary lists the
# regression targets. Run with --input-format root once data/ is populated.
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
if (!exists("TM_RDIR")) TM_RDIR <- .lhcb_resolve_rdir()
if (!exists("poisson_deviance", mode = "function")) {
  for (f in c("lhcb_domain.R","lhcb_kde.R","lhcb_poisson.R","lhcb_io.R"))
    source(file.path(TM_RDIR, f))
}

TM_CFG <- list(
  Q2_BINS = 60L, K1_FIXED = 7.61054, REFERENCE_K2 = 19.5296, REFERENCE_TOL = 0.75,
  K2_MIN = 18.0, K2_MAX = 24.0, N_K2 = 601L, K_EDGE_TOL = 0.05,
  KDE_BW_SCALE = 1.50, A1_MAX = 0.10, A2_MAX = 0.10,
  NULL_N = 5000L, SEED = 12345L,
  # 09d uses PSI2S upper edge 14.5 for the active/veto definition.
  JPSI = c(8.0, 11.0), PSI2S = c(12.5, 14.5))

#' Build masked binned counts + KDE baseline exactly as 09d make_binned_counts.
tm_make_binned_counts <- function(q2_values, bw_scale = TM_CFG$KDE_BW_SCALE) {
  edges <- seq(Q2_MIN, Q2_MAX, length.out = TM_CFG$Q2_BINS + 1L)
  counts <- hist(q2_values, breaks = edges, plot = FALSE, include.lowest = TRUE)$counts
  centers <- 0.5 * (edges[-1] + edges[-length(edges)])
  bin_width <- edges[2] - edges[1]
  veto <- in_veto_q2(centers, TM_CFG$JPSI, TM_CFG$PSI2S)
  baseline <- kde_baseline(q2_values, centers, bin_width,
                           q2_min = Q2_MIN, q2_max = Q2_MAX, bw_scale = bw_scale,
                           veto_fn = function(q) in_veto_q2(q, TM_CFG$JPSI, TM_CFG$PSI2S))
  keep <- !veto
  N <- as.numeric(counts[keep]); B <- baseline[keep]
  q2 <- centers[keep]; ell <- log(q2)
  scale <- sum(N) / max(sum(B), 1e-12)
  B <- pmax(B * scale, 1e-9)
  list(centers = centers, counts_all = counts, baseline_all = baseline,
       veto_all = veto, keep = keep, q2 = q2, ell = ell, N = N, B = B)
}

#' Full stage-09d computation given event-level q2 values.
tm_run <- function(q2_values, null_n = 0L, seed = TM_CFG$SEED,
                   null_engine = "python-compatible") {
  data <- tm_make_binned_counts(q2_values)
  N <- data$N; B <- data$B; ell <- data$ell
  k1 <- TM_CFG$K1_FIXED

  base <- fit_base_bounded(N, B, ell, k1, TM_CFG$A1_MAX)
  D_base <- base$D_base
  k2_grid <- seq(TM_CFG$K2_MIN, TM_CFG$K2_MAX, length.out = TM_CFG$N_K2)

  scan <- vector("list", length(k2_grid))
  for (i in seq_along(k2_grid)) {
    r <- fit_two_bounded(N, B, ell, k1, k2_grid[i], TM_CFG$A1_MAX, TM_CFG$A2_MAX, base = base)
    scan[[i]] <- data.frame(k2 = k2_grid[i], deltaD_add_exact = D_base - r$D_two,
      D_base_exact = D_base, D_two_exact = r$D_two, C = r$C,
      a1 = r$a1, b1 = r$b1, A1 = r$A1, phi1 = r$phi1,
      a2 = r$a2, b2 = r$b2, A2 = r$A2, phi2 = r$phi2,
      success = r$success, n_iter = r$n_iter,
      amplitude1_bound_active = r$amplitude1_bound_active,
      amplitude2_bound_active = r$amplitude2_bound_active)
  }
  scan_df <- do.call(rbind, scan)
  best_i <- which.max(scan_df$deltaD_add_exact)
  best_k2 <- scan_df$k2[best_i]
  best_two <- fit_two_bounded(N, B, ell, k1, best_k2, base = base)
  ref_two <- fit_two_bounded(N, B, ell, k1, TM_CFG$REFERENCE_K2, base = base)
  delta_best <- D_base - best_two$D_two
  delta_ref <- D_base - ref_two$D_two

  # nulls
  null_best <- NULL
  if (null_n > 0L) {
    RNGkind("L'Ecuyer-CMRG"); set.seed(seed)
    lam_base <- base$lambda_base
    null_best <- numeric(null_n); null_ref <- numeric(null_n)
    ref_idx <- which.min(abs(k2_grid - TM_CFG$REFERENCE_K2))
    for (j in seq_len(null_n)) {
      Y <- rpois(length(N), lam_base)
      if (null_engine == "exact") {
        b0 <- fit_base_bounded(Y, B, ell, k1, TM_CFG$A1_MAX)
        dd <- vapply(k2_grid, function(k2)
          b0$D_base - fit_two_bounded(Y, B, ell, k1, k2, base = b0)$D_two, numeric(1))
      } else {
        dd <- pn_scan_two(Y, B, ell, k1, k2_grid, TM_CFG$A1_MAX, TM_CFG$A2_MAX)$delta
      }
      null_best[j] <- max(dd); null_ref[j] <- dd[ref_idx]
    }
  }

  p_best <- if (!is.null(null_best)) p_value(delta_best, null_best) else NA
  p_ref  <- if (!is.null(null_best)) p_value(delta_ref, null_best) else NA

  list(data = data, scan = scan_df, base = base, best_two = best_two,
       ref_two = ref_two, D_base = D_base, best_k2 = best_k2,
       delta_best = delta_best, delta_ref = delta_ref,
       p_best = p_best, p_ref = p_ref, null_best = null_best, null_ref = if(!is.null(null_best)) null_ref else NULL,
       n_fit_bins = length(N), n_events = length(q2_values), null_engine = null_engine)
}

tm_write_outputs <- function(res, outdir, seed, null_n) {
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  utils::write.csv(res$scan, file.path(outdir, "two_mode_scan_mask.csv"), row.names = FALSE)
  if (!is.null(res$null_best))
    utils::write.csv(data.frame(null_best_deltaD_add = res$null_best,
      null_reference_deltaD_add = res$null_ref),
      file.path(outdir, "two_mode_null_mask.csv"), row.names = FALSE)
  summary <- list(test = "two_mode_kde_baseline_polar_exact_local_scan", mode = "mask",
    k1_fixed = TM_CFG$K1_FIXED, reference_k2 = TM_CFG$REFERENCE_K2,
    k2_scan = c(TM_CFG$K2_MIN, TM_CFG$K2_MAX), n_k2 = TM_CFG$N_K2,
    A1_MAX = TM_CFG$A1_MAX, A2_MAX = TM_CFG$A2_MAX, null_n = null_n, seed = seed,
    null_engine = res$null_engine,
    data = list(q2_bins = TM_CFG$Q2_BINS, baseline_mode = "kde",
      kde_bw_method = "scott", kde_bw_scale = TM_CFG$KDE_BW_SCALE,
      vetoes = list(JPSI = TM_CFG$JPSI, PSI2S = TM_CFG$PSI2S),
      n_fit_bins = res$n_fit_bins,
      n_events_after_mass_cuts_before_charmonium_handling = res$n_events),
    base = list(k1 = TM_CFG$K1_FIXED, D_base = res$D_base, C = res$base$C,
      a1 = res$base$a1, b1 = res$base$b1, A1 = res$base$A1, phi1 = res$base$phi1,
      success = res$base$success, amplitude1_bound_active = res$base$amplitude1_bound_active),
    best_two = list(k2 = res$best_k2, deltaD_add = res$delta_best,
      A1 = res$best_two$A1, A2 = res$best_two$A2, p_scan_max_null = res$p_best),
    reference_two = list(k2 = TM_CFG$REFERENCE_K2, deltaD_add = res$delta_ref,
      A2 = res$ref_two$A2, p_vs_local_scan_max_null = res$p_ref),
    diagnostics = list(real_scan_min_deltaD = min(res$scan$deltaD_add_exact),
      real_scan_median_deltaD = stats::median(res$scan$deltaD_add_exact),
      real_scan_max_deltaD = max(res$scan$deltaD_add_exact),
      scan_success_fraction = mean(res$scan$success),
      scan_A2_bound_fraction = mean(res$scan$amplitude2_bound_active)))
  write_json(summary, file.path(outdir, "two_mode_summary.json"))
  man <- build_manifest("09d", config = TM_CFG, seed = seed,
    outputs = list.files(outdir, full.names = TRUE),
    extra = list(note = "KDE-baseline bounded-Poisson two-mode scan"))
  write_json(man, file.path(outdir, "run_manifest.json"))
}

.lhcb_is_main <- function(script) {
  m <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  length(m) && basename(sub("^--file=", "", m[1])) == script
}
if (.lhcb_is_main("two_mode_kde_polar.R")) {
  suppressMessages(library(optparse))
  op <- OptionParser()
  op <- add_option(op, "--input-format", dest = "input_format", default = "root")
  op <- add_option(op, "--input", default = NULL)
  op <- add_option(op, "--data-dir", dest = "data_dir", default = "data")
  op <- add_option(op, "--q2-csv", dest = "q2_csv", default = NULL,
                   help = "CSV with a single column of event-level q2 (GeV^2)")
  op <- add_option(op, "--n-null", dest = "n_null", type = "integer", default = 0L)
  op <- add_option(op, "--null-engine", dest = "null_engine", default = "python-compatible")
  op <- add_option(op, "--seed", type = "integer", default = TM_CFG$SEED)
  op <- add_option(op, "--outdir", default = "outputs_logcos_poisson_twomode_kde_polar_r")
  opt <- parse_args(op)

  if (!is.null(opt$q2_csv)) {
    q2 <- utils::read.csv(opt$q2_csv)[[1]]
  } else {
    branches <- c("B0_M","muplus_PX","muplus_PY","muplus_PZ","muplus_PE",
                  "muminus_PX","muminus_PY","muminus_PZ","muminus_PE",
                  "Kplus_PX","Kplus_PY","Kplus_PZ","Kplus_PE",
                  "piminus_PX","piminus_PY","piminus_PZ","piminus_PE","Kst_892_0_M","Kst_M")
    ld <- load_events(opt$input_format, path = opt$input, branches = branches,
                      data_dir = opt$data_dir)
    df <- ld$df
    df$q2 <- if ("q2" %in% names(df)) df$q2 else reconstruct_q2(df)
    kst <- if ("Kst_892_0_M" %in% names(df)) df$Kst_892_0_M else
           if ("Kst_M" %in% names(df)) df$Kst_M else reconstruct_kst_mass(df)
    bm <- if ("B0_M" %in% names(df)) df$B0_M else df$B_M
    sel <- is.finite(df$q2) & df$q2 >= Q2_MIN & df$q2 <= Q2_MAX &
      bm >= B0_M_MIN & bm <= B0_M_MAX & kst >= KST_M_MIN & kst <= KST_M_MAX
    q2 <- df$q2[sel]
  }
  cat(sprintf("[stage 09d] selected events = %d\n", length(q2)))
  res <- tm_run(q2, null_n = opt$n_null, seed = opt$seed, null_engine = opt$null_engine)
  tm_write_outputs(res, opt$outdir, opt$seed, opt$n_null)
  cat(sprintf("[stage 09d] D_base=%.6f best_k2=%.4f deltaD=%.6f -> %s\n",
              res$D_base, res$best_k2, res$delta_best, opt$outdir))
}

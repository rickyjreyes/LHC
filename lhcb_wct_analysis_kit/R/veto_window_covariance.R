#!/usr/bin/env Rscript
# =============================================================================
# veto_window_covariance.R  (stage 25)
#
# R reproduction of 25_veto_window_covariance_test.py: veto-window covariance /
# active-domain invariance test.
#
# For each veto scheme the retained active q2 intervals and the active-domain
# length Delta_ell_A change, which rescales the k <-> n map. The test scans the
# SAME raw k grid for every scheme, detects spectral wells, transforms each well
# to n-space using THAT scheme's active length, and compares stability in raw
# k-space versus transformed n-space.
#
# The verdict reports the measured k-space and n-space stability statistics
# directly; it does NOT collapse to a binary "supports WCT" label.
#
# Data-gated: needs event-level q2 (see two_mode_kde_polar.R header).
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
VC_RDIR <- .lhcb_resolve_rdir()
if (!exists("fit_coeffbound", mode = "function"))
  for (f in c("lhcb_domain.R","lhcb_kde.R","lhcb_poisson.R","lhcb_wls.R","lhcb_io.R"))
    source(file.path(VC_RDIR, f))

# Veto schemes: (jpsi window, psi2s window). Names match the Python stage.
# Windows match VETO_SCHEMES in 25_veto_window_covariance_test.py exactly.
VC_SCHEMES <- list(
  tight        = list(jpsi = c(8.5, 10.5), psi2s = c(12.8, 14.2)),
  baseline_wide= list(jpsi = c(8.0, 11.0), psi2s = c(12.5, 14.5)),
  wider        = list(jpsi = c(7.5, 11.5), psi2s = c(12.25, 14.75)),
  very_wide    = list(jpsi = c(7.0, 12.0), psi2s = c(12.0, 15.0)),
  shift_low    = list(jpsi = c(7.8, 10.8), psi2s = c(12.3, 14.3)),
  shift_high   = list(jpsi = c(8.2, 11.2), psi2s = c(12.7, 14.7)))

VC_CFG <- list(N_BINS = 240L, K1 = 7.61054, K_SCAN_MIN = 6.0, K_SCAN_MAX = 32.0,
  N_K_SCAN = 1301L, A_MAX = 0.10, SEED = 12345L)

#' Per-scheme active intervals, Delta_ell_A, and the k->n map.
vc_scheme_geometry <- function(scheme) {
  iv <- active_intervals_from_vetoes(scheme$jpsi, scheme$psi2s)
  d <- active_delta_ell(iv)
  list(intervals = iv, delta_ell = d)
}

#' Detect wells on the coefficient-bounded Poisson DeltaD scan for one scheme.
vc_scan_scheme <- function(q2_values, scheme, bw_scale = 1.0) {
  geo <- vc_scheme_geometry(scheme)
  edges <- seq(log(Q2_MIN), log(Q2_MAX), length.out = VC_CFG$N_BINS + 1L)
  h <- hist(log(q2_values), breaks = edges, plot = FALSE, include.lowest = TRUE)$counts
  centers <- 0.5 * (edges[-1] + edges[-length(edges)])
  active <- in_active_intervals(exp(centers), geo$intervals)
  ell <- centers[active]; counts <- as.numeric(h[active])
  B <- kde_baseline_from_hist(ell, counts, bw_scale)

  base <- fit_coeffbound(counts, B, comb_basis_matrix(ell, numeric(0), VC_CFG$K1), VC_CFG$A_MAX)
  k_grid <- seq(VC_CFG$K_SCAN_MIN, VC_CFG$K_SCAN_MAX, length.out = VC_CFG$N_K_SCAN)
  delta <- vapply(k_grid, function(k) {
    f <- fit_coeffbound(counts, B, comb_basis_matrix(ell, k, VC_CFG$K1), VC_CFG$A_MAX)
    base$dev - f$dev
  }, numeric(1))
  scan_df <- data.frame(k = k_grid, n_eff = n_from_k(k_grid, geo$delta_ell), delta_chi2 = delta)
  wells <- find_wells(scan_df, 0.5, 0.75)
  list(scheme_delta_ell = geo$delta_ell, intervals = geo$intervals,
       scan = scan_df, wells = wells)
}

vc_run <- function(q2_values, bw_scale = 1.0) {
  per <- lapply(names(VC_SCHEMES), function(nm) {
    r <- vc_scan_scheme(q2_values, VC_SCHEMES[[nm]], bw_scale); r$scheme <- nm; r
  })
  names(per) <- names(VC_SCHEMES)
  # top well per scheme in raw-k and n-space
  tops <- do.call(rbind, lapply(per, function(r) {
    if (nrow(r$wells) == 0L) return(NULL)
    w <- r$wells[1, ]
    data.frame(scheme = r$scheme, delta_ell = r$scheme_delta_ell,
               top_k = w$k, top_n = w$n_eff, top_delta = w$delta_chi2)
  }))
  stability <- data.frame(
    metric = c("k_space_sd", "n_space_sd", "k_space_mean", "n_space_mean"),
    value = c(stats::sd(tops$top_k), stats::sd(tops$top_n),
              mean(tops$top_k), mean(tops$top_n)))
  list(per_scheme = per, best_wells = tops, stability = stability)
}

vc_write <- function(res, outdir, seed) {
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  utils::write.csv(res$best_wells, file.path(outdir, "veto_covariance_best_triplets.csv"), row.names = FALSE)
  utils::write.csv(res$stability, file.path(outdir, "veto_covariance_stability.csv"), row.names = FALSE)
  allwells <- do.call(rbind, lapply(res$per_scheme, function(r)
    if (nrow(r$wells)) cbind(scheme = r$scheme, r$wells) else NULL))
  utils::write.csv(allwells, file.path(outdir, "veto_covariance_wells.csv"), row.names = FALSE)
  scan_summary <- do.call(rbind, lapply(res$per_scheme, function(r)
    data.frame(scheme = r$scheme, delta_ell = r$scheme_delta_ell,
               n_active_bins = nrow(r$scan), max_delta = max(r$scan$delta_chi2))))
  utils::write.csv(scan_summary, file.path(outdir, "veto_covariance_scan_summary.csv"), row.names = FALSE)
  write_json(list(stage = "25", schemes = names(VC_SCHEMES),
    delta_ell_by_scheme = setNames(lapply(res$per_scheme, function(r) r$scheme_delta_ell), names(res$per_scheme)),
    stability = res$stability,
    note = paste("Active intervals and Delta_ell_A are recomputed per veto",
      "scheme; the same raw k grid is scanned for all schemes. Stability is",
      "reported directly in k-space and n-space, not as a binary verdict.")),
    file.path(outdir, "veto_covariance_summary.json"))
  man <- build_manifest("25", config = VC_CFG, seed = seed,
    outputs = list.files(outdir, full.names = TRUE))
  write_json(man, file.path(outdir, "run_manifest.json"))
}

.lhcb_is_main <- function(script) {
  m <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  length(m) && basename(sub("^--file=", "", m[1])) == script
}
if (.lhcb_is_main("veto_window_covariance.R")) {
  suppressMessages(library(optparse))
  op <- OptionParser()
  op <- add_option(op, "--q2-csv", dest = "q2_csv", default = NULL)
  op <- add_option(op, "--bandwidth-scale", dest = "bw", type = "double", default = 1.0)
  op <- add_option(op, "--seed", type = "integer", default = VC_CFG$SEED)
  op <- add_option(op, "--outdir", default = "outputs_wct_veto_covariance_r")
  opt <- parse_args(op)
  if (is.null(opt$q2_csv)) stop("Provide --q2-csv (event-level q2).")
  q2 <- utils::read.csv(opt$q2_csv)[[1]]
  vc_write(vc_run(q2, opt$bw), opt$outdir, opt$seed)
  cat("[stage 25] done ->", opt$outdir, "\n")
}

#!/usr/bin/env Rscript
# =============================================================================
# integer_winding_scan.R  (stage 12)
#
# R reproduction of 12_wct_integer_winding_scan.py. Stage 12 reuses the exact
# 09d bounded-Poisson engine (60 linear-q2 bins, KDE baseline, polar L-BFGS-B)
# but fixes the high-k mode to the active-domain integer windings
#   k_n = 2*pi*n / Delta_ell_A,  n = 10..22
# across the KDE bandwidth ladder {0.50, 0.75, 1.00, 1.25, 1.50}.
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
IW_RDIR <- .lhcb_resolve_rdir()
if (!exists("fit_two_bounded", mode = "function"))
  for (f in c("lhcb_domain.R","lhcb_kde.R","lhcb_poisson.R","lhcb_io.R"))
    source(file.path(IW_RDIR, f))
if (!exists("tm_make_binned_counts", mode = "function"))
  source(file.path(IW_RDIR, "two_mode_kde_polar.R"))

IW_CFG <- list(N_MIN = 10L, N_MAX = 22L, K1 = 7.61054, REFERENCE_K2 = 19.5296,
  BW_SCALES = c(0.50, 0.75, 1.00, 1.25, 1.50), SEED = 12345L)

iw_run_one_bandwidth <- function(q2_values, scale, n_null = 0L, seed = IW_CFG$SEED) {
  data <- tm_make_binned_counts(q2_values, bw_scale = scale)
  N <- data$N; B <- data$B; ell <- data$ell; k1 <- IW_CFG$K1
  base <- fit_base_bounded(N, B, ell, k1)
  ns <- IW_CFG$N_MIN:IW_CFG$N_MAX
  ks <- k_from_n(ns)
  rows <- do.call(rbind, lapply(seq_along(ns), function(i) {
    r <- fit_two_bounded(N, B, ell, k1, ks[i], base = base)
    data.frame(KDE_BANDWIDTH_SCALE = scale, n = ns[i], k2 = ks[i],
      deltaD = base$D_base - r$D_two, A2 = r$A2,
      A2_bound_active = r$amplitude2_bound_active,
      abs_delta_k_from_reference = abs(ks[i] - IW_CFG$REFERENCE_K2))
  }))
  # discrete-grid null: max over integer windings
  if (n_null > 0L) {
    RNGkind("L'Ecuyer-CMRG"); set.seed(seed + as.integer(scale * 100))
    lam <- base$lambda_base
    nullmax <- numeric(n_null)
    for (j in seq_len(n_null)) {
      Y <- rpois(length(N), lam)
      dd <- pn_scan_two(Y, B, ell, k1, ks)$delta
      nullmax[j] <- max(dd)
    }
    rows$p_vs_integer_scanmax_null <- vapply(rows$deltaD,
      function(d) p_value(d, nullmax), numeric(1))
  }
  rows
}

iw_run <- function(q2_values, n_null = 0L, seed = IW_CFG$SEED) {
  do.call(rbind, lapply(IW_CFG$BW_SCALES, function(s)
    iw_run_one_bandwidth(q2_values, s, n_null, seed)))
}

iw_write <- function(summary_df, outdir, seed, n_null) {
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  utils::write.csv(summary_df, file.path(outdir, "integer_winding_summary.csv"), row.names = FALSE)
  best <- do.call(rbind, by(summary_df, summary_df$KDE_BANDWIDTH_SCALE,
    function(d) d[which.max(d$deltaD), ]))
  n15 <- summary_df[summary_df$n == 15, ]
  write_json(list(stage = "12", kde_bandwidth_scales = IW_CFG$BW_SCALES,
    best_n_by_bandwidth = best, n15_by_bandwidth = n15, n_null = n_null, seed = seed),
    file.path(outdir, "integer_winding_summary.json"))
  man <- build_manifest("12", config = IW_CFG, seed = seed,
    outputs = list.files(outdir, full.names = TRUE))
  write_json(man, file.path(outdir, "run_manifest.json"))
}

.lhcb_is_main <- function(script) {
  m <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  length(m) && basename(sub("^--file=", "", m[1])) == script
}
if (.lhcb_is_main("integer_winding_scan.R")) {
  suppressMessages(library(optparse))
  op <- OptionParser()
  op <- add_option(op, "--q2-csv", dest = "q2_csv", default = NULL)
  op <- add_option(op, "--input-format", dest = "input_format", default = "root")
  op <- add_option(op, "--data-dir", dest = "data_dir", default = "data")
  op <- add_option(op, "--n-null", dest = "n_null", type = "integer", default = 0L)
  op <- add_option(op, "--seed", type = "integer", default = IW_CFG$SEED)
  op <- add_option(op, "--outdir", default = "outputs_wct_integer_winding_r")
  opt <- parse_args(op)
  if (is.null(opt$q2_csv)) stop("Provide --q2-csv (event-level q2) or extend with ROOT intake.")
  q2 <- utils::read.csv(opt$q2_csv)[[1]]
  s <- iw_run(q2, n_null = opt$n_null, seed = opt$seed)
  iw_write(s, opt$outdir, opt$seed, opt$n_null)
  cat("[stage 12] done ->", opt$outdir, "\n")
}

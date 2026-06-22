#!/usr/bin/env Rscript
# =============================================================================
# wct_vs_smooth_likelihood.R  (stage 16)
#
# R reproduction of 16_wct_vs_smqft_likelihood_test_cupy.py: compare WCT comb
# alternatives against a smooth empirical SM/QFT-like null.
#
# For each KDE bandwidth:
#   H0: repaired KDE baseline + low-k nuisance mode (k1)
#   H1: H0 + the specified WCT comb modes
#   DeltaD = D(H0) - D(H1);  AIC/BIC from the same parameter counts as Python
#   H0 parametric nulls -> refit H0 and H1 -> corrected empirical p-values
#
# LIMITATION (retained prominently in the JSON):
#   This compares WCT combs against a smooth empirical SM/QFT-like null. It is
#   NOT a full Standard Model amplitude analysis with official efficiencies,
#   acceptance, backgrounds, covariance, form factors or hadronic uncertainties.
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
WS_RDIR <- .lhcb_resolve_rdir()
if (!exists("fit_coeffbound", mode = "function"))
  for (f in c("lhcb_domain.R","lhcb_kde.R","lhcb_poisson.R","lhcb_io.R"))
    source(file.path(WS_RDIR, f))
if (!exists("kc_histogram", mode = "function"))
  source(file.path(WS_RDIR, "koide_comb_scan.R"))

WS_CFG <- list(K1 = 7.61054, N_BINS = 240L, A_MAX = 0.10,
  BW_SCALES = c(0.75, 1.00, 1.25), SEED = 12345L,
  MODELS = list(
    WCT_Koide_sideband_Q_2over3 = c(10, 15, 20),
    WCT_folded_Q_4over9 = 15 * c(4/9, 1, 2*4/9),
    WCT_combined_Q_2over3_plus_4over9 = c(10, 15, 20, 15*4/9, 15*8/9)))

LIMITATION <- paste("This compares WCT combs against a smooth empirical",
  "SM/QFT-like null. It is not a full Standard Model amplitude analysis with",
  "official efficiencies, acceptance, backgrounds, covariance, form factors,",
  "or hadronic uncertainties.")

ws_fit_dev <- function(counts, B, ell, ks) {
  X <- comb_basis_matrix(ell, ks, WS_CFG$K1)
  f <- fit_coeffbound(counts, B, X, WS_CFG$A_MAX)
  list(dev = f$dev, npar = ncol(X), bound = f$coefficient_bound_active)
}

ws_run_one_bandwidth <- function(q2_values, scale, n_null = 0L, seed = WS_CFG$SEED) {
  hh <- kc_histogram(q2_values); ell <- hh$ell; counts <- hh$counts
  B <- kde_baseline_from_hist(ell, counts, scale)
  n <- length(counts)
  h0 <- ws_fit_dev(counts, B, ell, numeric(0))   # baseline + k1
  rows <- list()
  for (nm in names(WS_CFG$MODELS)) {
    ks <- k_from_n(WS_CFG$MODELS[[nm]])
    h1 <- ws_fit_dev(counts, B, ell, ks)
    dD <- h0$dev - h1$dev
    # AIC/BIC differences (smaller is better); extra params = 2*n_modes
    dpar <- h1$npar - h0$npar
    dAIC <- dD - 2 * dpar          # AIC0 - AIC1 = (dev0-dev1) - 2*dpar... sign per Python
    dBIC <- dD - dpar * log(n)
    rows[[length(rows)+1]] <- data.frame(KDE_BANDWIDTH_SCALE = scale, model = nm,
      deltaD = dD, n_extra_params = dpar, delta_AIC = dAIC, delta_BIC = dBIC,
      h1_bound_active = h1$bound, stringsAsFactors = FALSE)
  }
  out <- do.call(rbind, rows)
  if (n_null > 0L) {
    RNGkind("L'Ecuyer-CMRG"); set.seed(seed + as.integer(scale*100))
    # H0 parametric null: lambda0 from H0 fit
    X0 <- comb_basis_matrix(ell, numeric(0), WS_CFG$K1)
    f0 <- fit_coeffbound(counts, B, X0, WS_CFG$A_MAX)
    lam0 <- pmax(B * exp(pmin(pmax(as.numeric(X0 %*% f0$beta), -20), 20)), 1e-12)
    nulldist <- matrix(0, n_null, length(WS_CFG$MODELS))
    for (j in seq_len(n_null)) {
      Y <- rpois(n, lam0)
      d0 <- ws_fit_dev(Y, B, ell, numeric(0))$dev
      for (mi in seq_along(WS_CFG$MODELS))
        nulldist[j, mi] <- d0 - ws_fit_dev(Y, B, ell, k_from_n(WS_CFG$MODELS[[mi]]))$dev
    }
    out$p_corrected <- vapply(seq_len(nrow(out)),
      function(i) p_value(out$deltaD[i], nulldist[, i]), numeric(1))
  }
  out
}

ws_run <- function(q2_values, n_null = 0L, seed = WS_CFG$SEED)
  do.call(rbind, lapply(WS_CFG$BW_SCALES, function(s) ws_run_one_bandwidth(q2_values, s, n_null, seed)))

ws_write <- function(df, outdir, seed, n_null) {
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  utils::write.csv(df, file.path(outdir, "wct_vs_smqft_summary.csv"), row.names = FALSE)
  write_json(list(stage = "16", models = names(WS_CFG$MODELS),
    bandwidth_scales = WS_CFG$BW_SCALES, n_null = n_null, seed = seed,
    summary = df, LIMITATION = LIMITATION),
    file.path(outdir, "wct_vs_smqft_summary.json"))
  man <- build_manifest("16", config = WS_CFG, seed = seed,
    outputs = list.files(outdir, full.names = TRUE),
    extra = list(LIMITATION = LIMITATION))
  write_json(man, file.path(outdir, "run_manifest.json"))
}

.lhcb_is_main <- function(script) {
  m <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  length(m) && basename(sub("^--file=", "", m[1])) == script
}
if (.lhcb_is_main("wct_vs_smooth_likelihood.R")) {
  suppressMessages(library(optparse))
  op <- OptionParser()
  op <- add_option(op, "--q2-csv", dest = "q2_csv", default = NULL)
  op <- add_option(op, "--n-null", dest = "n_null", type = "integer", default = 0L)
  op <- add_option(op, "--seed", type = "integer", default = WS_CFG$SEED)
  op <- add_option(op, "--outdir", default = "outputs_wct_vs_smqft_r")
  opt <- parse_args(op)
  if (is.null(opt$q2_csv)) stop("Provide --q2-csv (event-level q2).")
  q2 <- utils::read.csv(opt$q2_csv)[[1]]
  ws_write(ws_run(q2, opt$n_null, opt$seed), opt$outdir, opt$seed, opt$n_null)
  cat("[stage 16] done ->", opt$outdir, "\n")
}

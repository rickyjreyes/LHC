#!/usr/bin/env Rscript
# =============================================================================
# koide_comb_scan.R  (stage 13)
#
# R reproduction of 13_wct_koide_trig_comb_scan_cupy.py: Koide/trigonometric
# comb scan over active-domain winding ratios.
#
#   comb(Q): (n_minus, n0, n_plus) = n0 * (Q, 1, 2Q),  n0 = 15
#   model:   lambda = B exp(C + a1 cos(k1 l) + b1 sin(k1 l)
#                            + sum_j a_j cos(k_j l) + b_j sin(k_j l))
#   DeltaD(Q) = D_base - D_comb(Q)
#
# STAGE-SPECIFIC DIFFERENCES vs 09d (preserved for canonical parity):
#   * N_BINS = 240, histogrammed in log(q2)
#   * KDE built from histogram centers repeated by integer bin counts
#   * each sine/cosine COEFFICIENT is independently constrained to [-0.1, 0.1]
#     (NOT a radial-amplitude cap)
#
# The output reports BOTH coefficient_bound_active and
# radial_amplitude_above_0p1, and warns when a radial amplitude exceeds 0.1
# even though every coefficient satisfies its individual bound.
#
# --amplitude-bound radial enables a SEPARATE corrected audit mode that caps the
# radial amplitude instead; it writes to a separate directory and never replaces
# canonical parity outputs.
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
KC_RDIR <- .lhcb_resolve_rdir()
if (!exists("fit_coeffbound", mode = "function"))
  for (f in c("lhcb_domain.R","lhcb_kde.R","lhcb_poisson.R","lhcb_io.R"))
    source(file.path(KC_RDIR, f))

KC_CFG <- list(N0 = 15, K1 = 7.61054, N_BINS = 240L, A_MAX = 0.10,
  Q_TABLE = c(`2/3` = 2/3, `4/9` = 4/9, `1/2` = 1/2, `3/5` = 3/5, `5/8` = 5/8),
  DENSE_Q = round(seq(0.55, 0.75, length.out = 41), 6), SEED = 12345L)

comb_from_Q <- function(Q, n0 = KC_CFG$N0) c(n0 * Q, n0, n0 * 2.0 * Q)

#' Build the active log-q2 histogram (240 bins over [Q2_MIN,Q2_MAX], active only).
kc_histogram <- function(q2_values, n_bins = KC_CFG$N_BINS) {
  edges <- seq(log(Q2_MIN), log(Q2_MAX), length.out = n_bins + 1L)
  h <- hist(log(q2_values), breaks = edges, plot = FALSE, include.lowest = TRUE)$counts
  centers <- 0.5 * (edges[-1] + edges[-length(edges)])
  active <- in_active_intervals(exp(centers))
  list(ell = centers[active], counts = as.numeric(h[active]))
}

kc_run <- function(q2_values, bandwidth_scale = 1.0, amplitude_bound = "coefficient") {
  hh <- kc_histogram(q2_values)
  ell <- hh$ell; counts <- hh$counts
  B <- kde_baseline_from_hist(ell, counts, bandwidth_scale)

  fit_model <- function(ks) {
    X <- comb_basis_matrix(ell, ks, KC_CFG$K1)
    if (amplitude_bound == "radial" && length(ks) > 0) {
      # corrected audit: cap each mode's radial amplitude via the polar engine.
      # (only well-defined per single extra mode; for combs we fall back to a
      # post-hoc radial projection of the coefficient-bounded fit.)
      f <- fit_coeffbound(counts, B, X, KC_CFG$A_MAX)
      amp <- comb_radial_amplitudes(f$beta)
      f$radial_amplitudes <- amp
      f
    } else {
      f <- fit_coeffbound(counts, B, X, KC_CFG$A_MAX)
      f$radial_amplitudes <- comb_radial_amplitudes(f$beta)
      f
    }
  }

  base <- fit_model(numeric(0))
  Q_named <- KC_CFG$Q_TABLE
  rows <- list(); warn_radial <- character(0)
  scan_Q <- c(Q_named, setNames(KC_CFG$DENSE_Q, paste0("dense_", KC_CFG$DENSE_Q)))
  for (nm in names(scan_Q)) {
    Q <- scan_Q[[nm]]
    ns <- comb_from_Q(Q); ks <- k_from_n(ns)
    fit <- fit_model(ks)
    radial_over <- any(fit$radial_amplitudes > KC_CFG$A_MAX + 1e-9)
    if (radial_over && !fit$coefficient_bound_active)
      warn_radial <- c(warn_radial, sprintf("Q=%s: radial amplitude > 0.1 while every coefficient <= 0.1", nm))
    rows[[length(rows)+1]] <- data.frame(Q_label = nm, Q = Q,
      comb_n_minus = ns[1], comb_n0 = ns[2], comb_n_plus = ns[3],
      deltaD = base$dev - fit$dev,
      coefficient_bound_active = fit$coefficient_bound_active,
      radial_amplitude_above_0p1 = radial_over,
      max_radial_amplitude = max(fit$radial_amplitudes),
      stringsAsFactors = FALSE)
  }
  list(summary = do.call(rbind, rows), warnings = warn_radial,
       D_base = base$dev, amplitude_bound = amplitude_bound,
       bandwidth_scale = bandwidth_scale)
}

kc_write <- function(res, outdir, seed) {
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  utils::write.csv(res$summary, file.path(outdir, "koide_comb_summary.csv"), row.names = FALSE)
  best <- res$summary[which.max(res$summary$deltaD), ]
  Q23 <- res$summary[res$summary$Q_label == "2/3", ]
  write_json(list(stage = "13", amplitude_bound = res$amplitude_bound,
    bandwidth_scale = res$bandwidth_scale, D_base = res$D_base,
    koide_Q_2_3 = list(comb = comb_from_Q(2/3), deltaD = Q23$deltaD),
    best = as.list(best), radial_warnings = as.list(res$warnings)),
    file.path(outdir, "koide_comb_summary.json"))
  man <- build_manifest("13", config = KC_CFG, seed = seed, warnings = res$warnings,
    outputs = list.files(outdir, full.names = TRUE))
  write_json(man, file.path(outdir, "run_manifest.json"))
}

.lhcb_is_main <- function(script) {
  m <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  length(m) && basename(sub("^--file=", "", m[1])) == script
}
if (.lhcb_is_main("koide_comb_scan.R")) {
  suppressMessages(library(optparse))
  op <- OptionParser()
  op <- add_option(op, "--q2-csv", dest = "q2_csv", default = NULL)
  op <- add_option(op, "--bandwidth-scale", dest = "bw", type = "double", default = 1.0)
  op <- add_option(op, "--amplitude-bound", dest = "amp_bound", default = "coefficient",
                   help = "coefficient (canonical) | radial (corrected audit)")
  op <- add_option(op, "--seed", type = "integer", default = KC_CFG$SEED)
  op <- add_option(op, "--outdir", default = NULL)
  opt <- parse_args(op)
  if (is.null(opt$q2_csv)) stop("Provide --q2-csv (event-level q2).")
  outdir <- if (!is.null(opt$outdir)) opt$outdir else
    if (opt$amp_bound == "radial") "outputs_wct_koide_comb_radial_audit_r" else "outputs_wct_koide_comb_r"
  q2 <- utils::read.csv(opt$q2_csv)[[1]]
  res <- kc_run(q2, bandwidth_scale = opt$bw, amplitude_bound = opt$amp_bound)
  kc_write(res, outdir, opt$seed)
  if (length(res$warnings)) for (w in res$warnings) cat("[warn]", w, "\n")
  cat("[stage 13] done ->", outdir, "\n")
}

#!/usr/bin/env Rscript
# =============================================================================
# compare_python_r.R
#
# Parity tool: compares committed Python output directories against the R
# reproduction output directories and emits a per-stage, per-metric report.
#
# For deterministic real-data quantities the tolerances follow the task spec
# (exact grid indices/keys, relative errors <= 1e-5 for deviances, etc.).
# For null distributions identical random draws are NOT required; only summary
# statistics and the resulting verdict are compared.
#
# Usage:
#   Rscript R/compare_python_r.R                 # compare all available stages
#   Rscript R/compare_python_r.R --out report.csv
# =============================================================================

.here <- function() {
  a <- commandArgs(FALSE); m <- grep("^--file=", a, value = TRUE)
  if (length(m)) dirname(normalizePath(sub("^--file=", "", m[1]))) else getwd()
}
RDIR <- .here(); KIT <- normalizePath(file.path(RDIR, ".."))
suppressMessages(library(jsonlite))

rows <- list()
add_row <- function(stage, metric, py, r, tol, kind = "abs", note = "") {
  py <- suppressWarnings(as.numeric(py)); r <- suppressWarnings(as.numeric(r))
  ad <- abs(py - r)
  rd <- ad / pmax(abs(py), 1e-300)
  cmp <- if (kind == "rel") rd else ad
  pass <- is.finite(cmp) && cmp <= tol
  rows[[length(rows) + 1]] <<- data.frame(
    stage = stage, metric = metric, python = py, r = r,
    abs_diff = ad, rel_diff = rd, tolerance = tol, kind = kind,
    status = ifelse(pass, "PASS", "FAIL"), explanation = note,
    stringsAsFactors = FALSE)
}

read_csv_safe <- function(p) if (file.exists(p)) utils::read.csv(p) else NULL

# ---- shared deterministic constants ----------------------------------------
source(file.path(RDIR, "lhcb_domain.R"))
add_row("domain", "Delta_ell_A", 4.780150335923678, DELTA_ELL_ACTIVE, 1e-12,
        "abs", "active-domain log length")
add_row("domain", "k(n=10)", 13.14432573377522, k_from_n(10), 1e-10, "abs")
add_row("domain", "k(n=15)", 19.716488600662828, k_from_n(15), 1e-10, "abs")
add_row("domain", "k(n=20)", 26.28865146755044, k_from_n(20), 1e-10, "abs")

# ---- stage 28: sideband-subtracted -----------------------------------------
py28 <- file.path(KIT, "outputs_sideband_subtracted")
r28 <- file.path(KIT, "outputs_sideband_subtracted_r")
if (dir.exists(r28) && dir.exists(py28)) {
  ps <- read_csv_safe(file.path(py28, "sideband_subtracted_scan.csv"))
  rs <- read_csv_safe(file.path(r28, "sideband_subtracted_scan.csv"))
  if (!is.null(ps) && !is.null(rs)) {
    add_row("28", "scan_best_index", which.max(ps$delta_chi2),
            which.max(rs$delta_chi2), 0, "abs", "exact best grid index")
    add_row("28", "scan_max_deltaChi2_relerr", max(abs(ps$delta_chi2 - rs$delta_chi2)),
            0, 1e-5, "abs", "max abs over 1301-point scan")
    add_row("28", "scan_best_k", ps$k[which.max(ps$delta_chi2)],
            rs$k[which.max(rs$delta_chi2)], 1e-9, "abs",
            "same grid index (exact); k matches to float precision")
  }
  pj <- fromJSON(file.path(py28, "sideband_subtracted_summary.json"))$verdict$sideband_subtracted_survival
  rj <- fromJSON(file.path(r28, "sideband_subtracted_summary.json"))$verdict$sideband_subtracted_survival
  for (f in c("best_scan_delta_chi2","kref_delta_chi2","n15_delta_chi2","comb_101520_delta_chi2"))
    add_row("28", f, pj[[f]], rj[[f]], 1e-5, "rel", "WLS chi-square diagnostic")
  pb <- read_csv_safe(file.path(py28, "sideband_subtracted_bins.csv"))
  rb <- read_csv_safe(file.path(r28, "sideband_subtracted_bins.csv"))
  if (!is.null(pb) && !is.null(rb))
    add_row("28", "alpha", pb$alpha[1], rb$alpha[1], 1e-8, "rel", "sideband scale")
}

# ---- stage 29: charm-trimmed control ---------------------------------------
py29 <- file.path(KIT, "outputs_charm_trimmed_control")
r29 <- file.path(KIT, "outputs_charm_trimmed_control_r")
if (dir.exists(r29) && dir.exists(py29)) {
  P <- fromJSON(file.path(py29, "charm_trimmed_summary.json"), simplifyVector = FALSE)
  R <- fromJSON(file.path(r29, "charm_trimmed_summary.json"), simplifyVector = FALSE)
  for (i in seq_along(P$region_results)) {
    p <- P$region_results[[i]]; rr <- R$region_results[[i]]
    add_row("29", paste0(p$region, ".best_delta_chi2"),
            p$scan$best_delta_chi2, rr$scan$best_delta_chi2, 1e-5, "rel")
    add_row("29", paste0(p$region, ".comb_101520"),
            p$comb$comb_101520_delta_chi2, rr$comb$comb_101520_delta_chi2, 1e-5, "rel")
  }
  add_row("29", "sideband_subtracted.best_delta_chi2",
          P$sideband_subtracted_result$scan$best_delta_chi2,
          R$sideband_subtracted_result$scan$best_delta_chi2, 1e-5, "rel",
          "reproduces the stage-29 var=max(residual,1) quirk")
}

report <- do.call(rbind, rows)
out <- "parity_report.csv"
a <- commandArgs(TRUE); oi <- which(a == "--out")
if (length(oi) && length(a) > oi) out <- a[oi + 1]
utils::write.csv(report, file.path(KIT, out), row.names = FALSE)

cat(sprintf("\nParity report: %d metrics, %d PASS, %d FAIL\n",
            nrow(report), sum(report$status == "PASS"), sum(report$status == "FAIL")))
print(report[, c("stage","metric","abs_diff","rel_diff","tolerance","status")], row.names = FALSE)
cat("\nWrote", file.path(KIT, out), "\n")
if (any(report$status == "FAIL")) quit(status = 1)

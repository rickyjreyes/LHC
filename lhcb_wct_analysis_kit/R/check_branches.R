#!/usr/bin/env Rscript
# =============================================================================
# check_branches.R  (stage 01)
#
# R reproduction of 01_check_branches.py: report readiness of the branches
# needed for the q2 scan (and, optionally, the angular pipeline). Writes a
# branch report and, crucially, never silently omits a required mass branch:
# a missing required branch fails unless --allow-missing is passed, in which
# case the omission is recorded in the report.
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
CB_RDIR <- .lhcb_resolve_rdir()
if (!exists("discover_root_files", mode = "function"))
  source(file.path(CB_RDIR, "lhcb_io.R"))

REQUIRED_FOR_Q2 <- c("muplus_PX","muplus_PY","muplus_PZ","muplus_PE",
                     "muminus_PX","muminus_PY","muminus_PZ","muminus_PE","B0_M")
REQUIRED_MASS <- c("B0_M")
KST_MASS_OR_VECTORS <- list(mass = c("Kst_892_0_M","Kst_M"),
  vectors = c("Kplus_PX","Kplus_PY","Kplus_PZ","Kplus_PE",
              "piminus_PX","piminus_PY","piminus_PZ","piminus_PE"))
REQUIRED_FOR_P5 <- c("cosThetaL","cosThetaK","phi")
Q2_DIRECT <- c("q2","Q2","q2_DTF","mumu_M2","Jpsi_M2","J_psi_1S_M2")

resolve_branch <- function(branches, name) {
  if (name %in% branches) return(name)
  lm <- setNames(branches, tolower(branches))
  if (tolower(name) %in% names(lm)) return(unname(lm[[tolower(name)]]))
  hits <- branches[grepl(tolower(name), tolower(branches), fixed = TRUE)]
  if (length(hits)) hits[1] else NA_character_
}

check_branches <- function(data_dir = "data", tree = "B0_KstMuMu/DecayTree",
                           allow_missing = FALSE, outdir = "outputs_r") {
  if (!requireNamespace("reticulate", quietly = TRUE))
    stop("check_branches needs reticulate + uproot.")
  files <- discover_root_files(data_dir)
  if (!length(files)) stop("No ROOT files under ", data_dir)
  uproot <- reticulate::import("uproot")
  f <- uproot$open(files[1])
  keys <- reticulate::py_to_r(f$keys(recursive = TRUE))
  tn <- if (tree %in% keys) tree else keys[grepl("DecayTree", keys)][1]
  branches <- reticulate::py_to_r(f[[tn]]$keys()); f$close()

  status <- function(names) setNames(lapply(names, resolve_branch, branches = branches), names)
  q2_direct <- Filter(Negate(is.na), lapply(Q2_DIRECT, resolve_branch, branches = branches))
  q2_vec <- status(REQUIRED_FOR_Q2)
  q2_ready <- all(!vapply(q2_vec, is.na, logical(1))) || length(q2_direct) > 0
  kst_mass <- Filter(Negate(is.na), lapply(KST_MASS_OR_VECTORS$mass, resolve_branch, branches = branches))
  kst_vec_ok <- all(!vapply(KST_MASS_OR_VECTORS$vectors, function(b) is.na(resolve_branch(branches, b)), logical(1)))
  kst_ready <- length(kst_mass) > 0 || kst_vec_ok
  b_mass <- resolve_branch(branches, "B0_M")

  missing_required <- character(0)
  if (is.na(b_mass)) missing_required <- c(missing_required, "B0_M")
  if (!kst_ready) missing_required <- c(missing_required, "Kst mass or K+/pi- four-vectors")

  if (length(missing_required) && !allow_missing) {
    stop("Missing required branch(es): ", paste(missing_required, collapse = ", "),
         ". Pass --allow-missing to record the omission and continue.")
  }

  report <- list(file = files[1], tree = tn, branch_count = length(branches),
    q2_ready = q2_ready, q2_direct_candidates = q2_direct,
    q2_reconstruction_branches = q2_vec, kst_mass_branches = kst_mass,
    kst_reconstruction_available = kst_vec_ok, p5_branches = status(REQUIRED_FOR_P5),
    missing_required = missing_required,
    allow_missing_used = (length(missing_required) > 0 && allow_missing))
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  write_json(report, file.path(outdir, "branch_report.json"))
  cat("q2 scan:", if (q2_ready) "READY" else "NOT READY", "\n")
  cat("K* mass:", if (kst_ready) "READY" else "NOT READY", "\n")
  invisible(report)
}

.lhcb_is_main <- function(script) {
  m <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  length(m) && basename(sub("^--file=", "", m[1])) == script
}
if (.lhcb_is_main("check_branches.R")) {
  suppressMessages(library(optparse))
  op <- OptionParser()
  op <- add_option(op, "--data-dir", dest = "data_dir", default = "data")
  op <- add_option(op, "--allow-missing", dest = "allow_missing", action = "store_true", default = FALSE)
  op <- add_option(op, "--outdir", default = "outputs_r")
  opt <- parse_args(op)
  check_branches(opt$data_dir, allow_missing = opt$allow_missing, outdir = opt$outdir)
}

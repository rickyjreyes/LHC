#!/usr/bin/env Rscript
# =============================================================================
# inspect_root.R  (stage 00)
#
# R reproduction of 00_inspect_root.py plus a deterministic export utility.
#
# Modes:
#   (default)  inspect the first discovered ROOT file: trees + branches.
#   --export   convert selected branches of all discovered ROOT files to a
#              single Parquet cache (option C in the task spec), so the rest of
#              the R pipeline can run without reticulate on subsequent calls.
#
# ROOT access uses reticulate + uproot strictly as an I/O adapter.
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
IR_RDIR <- .lhcb_resolve_rdir()
if (!exists("discover_root_files", mode = "function"))
  source(file.path(IR_RDIR, "lhcb_io.R"))

DEFAULT_EXPORT_BRANCHES <- c(
  "B0_M","muplus_PX","muplus_PY","muplus_PZ","muplus_PE",
  "muminus_PX","muminus_PY","muminus_PZ","muminus_PE",
  "Kplus_PX","Kplus_PY","Kplus_PZ","Kplus_PE",
  "piminus_PX","piminus_PY","piminus_PZ","piminus_PE",
  "Kst_892_0_M","Kst_M","q2","Q2")

inspect_first <- function(data_dir = "data", tree = "B0_KstMuMu/DecayTree") {
  if (!requireNamespace("reticulate", quietly = TRUE))
    stop("inspect needs reticulate + uproot.")
  files <- discover_root_files(data_dir)
  if (!length(files)) stop("No ROOT files under ", data_dir)
  uproot <- reticulate::import("uproot")
  f <- uproot$open(files[1])
  keys <- reticulate::py_to_r(f$keys(recursive = TRUE))
  cat("Inspecting:", files[1], "\nTop-level/tree keys:\n")
  for (k in keys) cat("  ", k, "\n")
  tn <- if (tree %in% keys) tree else keys[grepl("DecayTree", keys)][1]
  if (!is.na(tn)) {
    br <- reticulate::py_to_r(f[[tn]]$keys())
    cat(sprintf("\nFirst tree: %s  (%d branches)\n", tn, length(br)))
    cat(paste(" ", head(br, 60)), sep = "\n")
  }
  f$close()
  invisible(list(file = files[1], tree = tn, branches = br))
}

export_parquet <- function(data_dir = "data", out = "data_cache/events.parquet",
                           branches = DEFAULT_EXPORT_BRANCHES,
                           tree = "B0_KstMuMu/DecayTree") {
  if (!requireNamespace("arrow", quietly = TRUE)) stop("export needs the 'arrow' package.")
  files <- discover_root_files(data_dir)
  res <- read_root_uproot(files, branches, tree)
  dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
  arrow::write_parquet(res$df, out)
  prov_path <- sub("\\.parquet$", "_provenance.json", out)
  write_json(list(provenance = res$provenance, exported_branches = branches,
                  out = out, n_rows = nrow(res$df)), prov_path)
  cat(sprintf("Exported %d rows -> %s\nProvenance -> %s\n", nrow(res$df), out, prov_path))
  invisible(res)
}

.lhcb_is_main <- function(script) {
  m <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  length(m) && basename(sub("^--file=", "", m[1])) == script
}
if (.lhcb_is_main("inspect_root.R")) {
  suppressMessages(library(optparse))
  op <- OptionParser()
  op <- add_option(op, "--data-dir", dest = "data_dir", default = "data")
  op <- add_option(op, "--export", action = "store_true", default = FALSE)
  op <- add_option(op, "--out", default = "data_cache/events.parquet")
  opt <- parse_args(op)
  if (opt$export) export_parquet(opt$data_dir, opt$out) else inspect_first(opt$data_dir)
}

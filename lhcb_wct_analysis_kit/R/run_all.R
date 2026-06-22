#!/usr/bin/env Rscript
# =============================================================================
# run_all.R
#
# Orchestrator for the R yield-side reproduction. Mirrors run_all.py stage keys
# where practical and writes logs + a machine-readable run summary under
# outputs_run_all_r/.
#
# Stage keys: 00 01 09d 12 13 16 25 28 29
#   00,01    intake / branch readiness (need ROOT data)
#   09d      two-mode KDE bounded-Poisson scan (needs event q2)
#   12,13,16 winding / comb / smooth-null Poisson stages (need event q2)
#   25       veto-window covariance (needs event q2)
#   28,29    sideband-subtracted & charm-trimmed WLS controls
#            (replayable from committed per-bin inputs WITHOUT raw ntuples)
#
# Optional well-first / angular stages are never run by default.
#
# Usage:
#   Rscript R/run_all.R --dry-run
#   Rscript R/run_all.R --only 28,29 --continue-on-error
#   Rscript R/run_all.R --fast --controls
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
RA_RDIR <- .lhcb_resolve_rdir()
source(file.path(RA_RDIR, "lhcb_io.R"))

# stage key -> (script, default args builder, requires_event_data, output dir)
STAGES <- list(
  "00"  = list(script = "inspect_root.R",        needs_data = TRUE,  out = "outputs_r"),
  "01"  = list(script = "check_branches.R",       needs_data = TRUE,  out = "outputs_r"),
  "09d" = list(script = "two_mode_kde_polar.R",   needs_data = TRUE,  out = "outputs_logcos_poisson_twomode_kde_polar_r"),
  "12"  = list(script = "integer_winding_scan.R", needs_data = TRUE,  out = "outputs_wct_integer_winding_r"),
  "13"  = list(script = "koide_comb_scan.R",      needs_data = TRUE,  out = "outputs_wct_koide_comb_r"),
  "16"  = list(script = "wct_vs_smooth_likelihood.R", needs_data = TRUE, out = "outputs_wct_vs_smqft_r"),
  "25"  = list(script = "veto_window_covariance.R",   needs_data = TRUE, out = "outputs_wct_veto_covariance_r"),
  "28"  = list(script = "sideband_subtracted.R",  needs_data = FALSE, out = "outputs_sideband_subtracted_r"),
  "29"  = list(script = "charm_trimmed_control.R", needs_data = FALSE, out = "outputs_charm_trimmed_control_r"))

DEFAULT_ORDER <- c("00","01","09d","12","13","16","25","28","29")

parse_csv_arg <- function(x) if (is.null(x) || !nzchar(x)) character(0) else trimws(strsplit(x, ",")[[1]])

main <- function() {
  suppressMessages(library(optparse))
  op <- OptionParser()
  op <- add_option(op, "--dry-run", dest = "dry_run", action = "store_true", default = FALSE)
  op <- add_option(op, "--fast", action = "store_true", default = FALSE)
  op <- add_option(op, "--full", action = "store_true", default = FALSE)
  op <- add_option(op, "--controls", action = "store_true", default = FALSE)
  op <- add_option(op, "--only", default = NULL)
  op <- add_option(op, "--skip", default = NULL)
  op <- add_option(op, "--from", default = NULL)
  op <- add_option(op, "--continue-on-error", dest = "coe", action = "store_true", default = FALSE)
  op <- add_option(op, "--n-null", dest = "n_null", type = "integer", default = 0L)
  op <- add_option(op, "--seed", type = "integer", default = 12345L)
  op <- add_option(op, "--workers", type = "integer", default = 1L)
  op <- add_option(op, "--input-format", dest = "input_format", default = "root")
  op <- add_option(op, "--data-dir", dest = "data_dir", default = "data")
  op <- add_option(op, "--output-suffix", dest = "suffix", default = "")
  opt <- parse_args(op)

  order <- DEFAULT_ORDER
  only <- parse_csv_arg(opt$only); skip <- parse_csv_arg(opt$skip)
  if (length(only)) order <- order[order %in% only]
  if (length(skip)) order <- order[!order %in% skip]
  if (!is.null(opt$from)) { i <- match(opt$from, order); if (!is.na(i)) order <- order[i:length(order)] }
  if (opt$fast && !length(only)) opt$n_null <- min(opt$n_null, 20L)

  outdir <- "outputs_run_all_r"; dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  data_present <- length(discover_root_files(opt$data_dir)) > 0
  results <- list()

  cat(sprintf("Run plan: %s | n_null=%d seed=%d | data_present=%s\n",
              paste(order, collapse = ","), opt$n_null, opt$seed, data_present))

  for (key in order) {
    st <- STAGES[[key]]
    if (is.null(st)) next
    script <- file.path(RA_RDIR, st$script)
    out <- if (nzchar(opt$suffix)) paste0(st$out, opt$suffix) else st$out
    args <- c("--seed", opt$seed)
    replay <- FALSE
    if (key == "28") {
      committed <- file.path(dirname(RA_RDIR), "outputs_sideband_subtracted", "sideband_subtracted_bins.csv")
      if (!data_present && file.exists(committed)) { args <- c(args, "--bins", committed); replay <- TRUE }
      args <- c(args, "--n-null", opt$n_null, "--outdir", out)
    } else if (key == "29") {
      args <- c(args, "--outdir", out); replay <- !data_present
    } else {
      args <- c(args, "--data-dir", opt$data_dir, "--input-format", opt$input_format,
                "--n-null", opt$n_null, "--outdir", out)
    }

    skip_reason <- NULL
    if (st$needs_data && !data_present) skip_reason <- "requires ROOT event data (not present)"

    rec <- list(stage = key, script = st$script, output = out, replay = replay)
    if (opt$dry_run) {
      rec$status <- if (is.null(skip_reason)) "WOULD_RUN" else paste0("WOULD_SKIP: ", skip_reason)
      cat(sprintf("  [%s] %s -> %s\n", key, rec$status, out)); results[[key]] <- rec; next
    }
    if (!is.null(skip_reason)) {
      rec$status <- paste0("SKIPPED: ", skip_reason)
      cat(sprintf("  [%s] %s\n", key, rec$status)); results[[key]] <- rec; next
    }

    log_path <- file.path(outdir, sprintf("%s_%s.log", key, sub("\\.R$", "", st$script)))
    t0 <- Sys.time()
    rc <- system2("Rscript", c(script, args), stdout = log_path, stderr = log_path,
                  env = paste0("LHCB_R_DIR=", RA_RDIR))
    rec$runtime_sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    rec$exit_status <- rc
    rec$status <- if (rc == 0) "OK" else "FAILED"
    rec$log <- log_path
    cat(sprintf("  [%s] %s (%.1fs) log=%s\n", key, rec$status, rec$runtime_sec, log_path))
    results[[key]] <- rec
    if (rc != 0 && !opt$coe) { cat("Stopping (no --continue-on-error)\n"); break }
  }

  summary <- list(selected_stages = order, n_null = opt$n_null, seed = opt$seed,
    data_present = data_present, dry_run = opt$dry_run, results = unname(results),
    manifest = build_manifest("run_all", config = list(only = only, skip = skip),
                              seed = opt$seed))
  write_json(summary, file.path(outdir, "run_summary.json"))
  cat("Wrote", file.path(outdir, "run_summary.json"), "\n")
}

.lhcb_is_main <- function(script) {
  m <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  length(m) && basename(sub("^--file=", "", m[1])) == script
}
if (.lhcb_is_main("run_all.R")) main()

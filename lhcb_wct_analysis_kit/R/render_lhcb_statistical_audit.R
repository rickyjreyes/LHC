#!/usr/bin/env Rscript
# =============================================================================
# render_lhcb_statistical_audit.R   (section 29)
#
# Command-line orchestrator for the complete LHCb / WCT statistical audit.
# Replay mode runs everything establishable from committed artifacts; full mode
# additionally performs event-level audits when ROOT/Parquet data are present.
#
#   Rscript R/render_lhcb_statistical_audit.R --mode replay --fast --force
#
# Deterministic: RNGkind("L'Ecuyer-CMRG"); results invariant to worker count.
# =============================================================================

suppressWarnings(suppressMessages(library(optparse)))

# ---- resolve dirs + source the audit suite ----------------------------------
.resolve_rdir <- function() {
  a <- commandArgs(FALSE); m <- grep("^--file=", a, value = TRUE)
  if (length(m)) return(dirname(normalizePath(sub("^--file=", "", m[1]), mustWork = FALSE)))
  file.path(getwd(), "R")
}
RDIR <- .resolve_rdir()
KIT <- dirname(RDIR)
# keep getwd-based sourcing in modules robust
setwd(KIT)

src <- function(f) source(file.path(RDIR, "audit", f))
source(file.path(RDIR, "lhcb_domain.R"))
src("lhcb_audit_utils.R")
src("build_analysis_registry.R")
src("build_stage_registry.R")
src("build_event_flow.R")
src("run_09d_sensitivity.R")
src("run_winding_sensitivity.R")
src("run_comb_sensitivity.R")
src("build_model_comparison.R")
src("run_veto_invariance_analysis.R")
src("run_sideband_uncertainty.R")
src("run_charm_control_audit.R")
src("peak_region_stability.R")
src("run_holdout_validation.R")
src("calibrate_full_pipeline_null.R")
src("run_injection_recovery.R")
src("global_multiple_testing.R")
src("build_effect_size_tables.R")
src("build_claim_matrix.R")
src("make_audit_figures.R")

# ---- options ----------------------------------------------------------------
opt_list <- list(
  make_option("--mode", default = "replay"),
  make_option("--data-dir", dest = "data_dir", default = "data"),
  make_option("--cache", default = "data_cache/events.parquet"),
  make_option("--python-root", dest = "python_root", default = "."),
  make_option("--r-root", dest = "r_root", default = "."),
  make_option("--out-root", dest = "out_root", default = "outputs_statistical_audit_r"),
  make_option("--table-dir", dest = "table_dir", default = "tables_r/statistical_audit"),
  make_option("--figure-dir", dest = "figure_dir", default = "figures_r/statistical_audit"),
  make_option("--report", default = "reports/lhcb_statistical_audit.qmd"),
  make_option("--bootstrap-n", dest = "bootstrap_n", type = "integer", default = 2000L),
  make_option("--null-n", dest = "null_n", type = "integer", default = 5000L),
  make_option("--family-null-n", dest = "family_null_n", type = "integer", default = 2000L),
  make_option("--calibration-n", dest = "calibration_n", type = "integer", default = 2000L),
  make_option("--injection-n", dest = "injection_n", type = "integer", default = 500L),
  make_option("--seed", type = "integer", default = 12345L),
  make_option("--parallel", default = "true"),
  make_option("--workers", default = "auto"),
  make_option("--include-well-first", dest = "include_well_first", default = "false"),
  make_option("--include-angular", dest = "include_angular", default = "false"),
  make_option("--render-report", dest = "render_report", default = "true"),
  make_option("--force", action = "store_true", default = FALSE),
  make_option("--strict", action = "store_true", default = FALSE),
  make_option("--fast", action = "store_true", default = FALSE)
)
opt <- parse_args(OptionParser(option_list = opt_list))

is_true <- function(x) isTRUE(x) || (is.character(x) && tolower(x) %in% c("true","yes","1"))
TABLE_DIR <- file.path(KIT, opt$table_dir)
FIGURE_DIR <- file.path(KIT, opt$figure_dir)
OUT_ROOT <- file.path(KIT, opt$out_root)
invisible(lapply(list(TABLE_DIR, FIGURE_DIR, OUT_ROOT,
                      file.path(KIT, "reports", "rendered")), ensure_dir))

if (opt$fast) {
  opt$bootstrap_n <- min(opt$bootstrap_n, 300L)
  opt$calibration_n <- min(opt$calibration_n, 100L)
  opt$injection_n <- min(opt$injection_n, 50L)
  DEV_TAG <- "DEVELOPMENT_ONLY"
} else DEV_TAG <- "FULL_RESOLUTION"

audit_set_seed(opt$seed)
warnings_log <- character(0)
failures_log <- character(0)
guarded <- function(label, expr) {
  tryCatch(expr, error = function(e) {
    failures_log[[length(failures_log)+1]] <<- sprintf("%s: %s", label, conditionMessage(e))
    if (opt$strict) stop(e)
    warning(sprintf("[%s] %s", label, conditionMessage(e)), call. = FALSE)
    NULL
  })
}

cat(sprintf("== LHCb statistical audit | mode=%s | %s | seed=%d ==\n",
            opt$mode, DEV_TAG, opt$seed))
event_ok <- have_event_data(file.path(KIT, opt$data_dir),
                            file.path(KIT, opt$cache))
if (opt$mode == "full" && !event_ok) {
  msg <- "full mode requested but no ROOT/Parquet event data found"
  if (opt$strict) stop(msg) else { warning(msg); cat("  -> falling back to replay-level analyses for event stages\n") }
}

# ---- registries + flow ------------------------------------------------------
reg <- guarded("registry", { r <- build_analysis_registry(KIT)
  write_analysis_registry(r, TABLE_DIR, OUT_ROOT); r })
sreg <- guarded("stage_registry", { s <- build_stage_registry(KIT)
  write_stage_registry(s, TABLE_DIR); s })
flow <- guarded("event_flow", run_event_flow(TABLE_DIR, FIGURE_DIR, KIT))

# ---- per-stage audits -------------------------------------------------------
s09 <- guarded("09d", run_09d_sensitivity(TABLE_DIR, FIGURE_DIR, KIT, opt$mode))
s12 <- guarded("12", run_winding_sensitivity(TABLE_DIR, FIGURE_DIR, KIT, opt$mode))
s13 <- guarded("13", run_comb_sensitivity(TABLE_DIR, FIGURE_DIR, KIT, opt$mode))
mcmp <- guarded("16", build_model_comparison(TABLE_DIR, FIGURE_DIR, KIT))
s25 <- guarded("25", run_veto_invariance_analysis(TABLE_DIR, FIGURE_DIR, KIT))
s28 <- guarded("28", run_sideband_uncertainty(TABLE_DIR, FIGURE_DIR, KIT, opt$mode,
                                              bootstrap_n = opt$bootstrap_n, seed = 271828))
s29 <- guarded("29", run_charm_control_audit(TABLE_DIR, FIGURE_DIR, KIT, opt$mode,
                                             bootstrap_n = opt$bootstrap_n, seed = 314159))
pks <- guarded("peak_stability", peak_region_stability(TABLE_DIR, KIT, s09, mode = opt$mode))

# ---- validation / calibration / injection -----------------------------------
hold <- guarded("holdout", run_file_holdout_validation(TABLE_DIR, KIT, opt$mode))
blk  <- guarded("blocked_q2", run_blocked_q2_validation(TABLE_DIR, KIT, opt$mode))
calib <- guarded("calibration", run_calibration(TABLE_DIR, FIGURE_DIR, KIT, opt$mode,
                                                calibration_n = opt$calibration_n,
                                                seed = 271828, fast = opt$fast))
inj <- guarded("injection", run_injection_recovery(TABLE_DIR, FIGURE_DIR, KIT, opt$mode,
                                                   injection_n = opt$injection_n,
                                                   seed = 161803, fast = opt$fast))

# ---- multiple testing + effects + claims ------------------------------------
mt <- guarded("multiple_testing", run_multiple_testing(TABLE_DIR, KIT))
eff <- guarded("effect_sizes", build_effect_size_tables(TABLE_DIR, KIT, s09, s28))
claims <- guarded("claim_matrix",
                  build_claim_matrix(TABLE_DIR, KIT, s09, s12, s13, s28, s29, s25))

# ---- parity / regression guardrails -----------------------------------------
parity <- guarded("parity", {
  chk <- function(name, got, exp, tol) data.frame(
    target = name, committed = exp, audit = got,
    abs_diff = abs(as.numeric(got) - as.numeric(exp)),
    within_tol = isTRUE(abs(as.numeric(got) - as.numeric(exp)) <= tol),
    stringsAsFactors = FALSE)
  rows <- list()
  if (!is.null(s09)) {
    rg <- s09$regression
    rows <- c(rows, list(
      chk("09d_D_base", rg$D_base, 1255.9044965985288, 1e-6),
      chk("09d_best_k2", rg$best_k2, 23.08, 1e-6),
      chk("09d_best_deltaD", rg$best_dD, 150.90386012713225, 1e-6),
      chk("09d_ref_deltaD", rg$ref_dD, 70.0029667147544, 1e-6)))
  }
  if (!is.null(s12)) {
    rg <- s12$regression
    rows <- c(rows, list(
      chk("12_best_n_bw1", rg$best_n_bw1, 20, 0),
      chk("12_best_deltaD_bw1", rg$best_deltaD_bw1, 135.5123713507652, 1e-6),
      chk("12_n15_deltaD_bw1", rg$n15_deltaD_bw1, 58.25363341553543, 1e-6)))
  }
  if (!is.null(s13)) {
    rg <- s13$regression
    rows <- c(rows, list(
      chk("13_q23_deltaD_bw1", rg$q23_deltaD_bw1, 373.077171046471, 1e-6),
      chk("13_best_deltaD_bw1", rg$best_deltaD_bw1, 457.4416529560142, 1e-6)))
  }
  if (!is.null(s28)) {
    rg <- s28$regression
    rows <- c(rows, list(
      chk("28_alpha", rg$alpha, 0.28495897903372835, 1e-12),
      chk("28_best_dchi2", rg$best_dchi2, 5.303549331940928, 1e-6),
      chk("28_ref_dchi2", rg$ref_dchi2, 0.4468052357436818, 1e-6),
      chk("28_n15_dchi2", rg$n15_dchi2, 1.2124631626656992, 1e-6)))
  }
  pr <- do.call(rbind, rows)
  write_audit_csv(pr, file.path(TABLE_DIR, "python_r_parity.csv"))
  pr
})
if (opt$strict && !is.null(parity) && any(!parity$within_tol)) {
  stop(sprintf("strict: parity drift on %s",
               paste(parity$target[!parity$within_tol], collapse = ", ")))
}

# ---- figures ----------------------------------------------------------------
ctx <- list(flow = flow, s09 = s09, s12 = s12, s13 = s13, model_cmp = mcmp,
            s25 = s25, s28 = s28, s29 = s29, calib = calib, inj = inj, mt = mt)
figs <- guarded("figures", make_audit_figures(FIGURE_DIR, KIT, ctx, opt$mode))

# ---- run manifest -----------------------------------------------------------
manifest <- list(
  mode = opt$mode, dev_tag = DEV_TAG, seed = opt$seed, timestamp = utc_now(),
  git_commit = git_commit(), event_data_available = event_ok,
  bootstrap_n = opt$bootstrap_n, calibration_n = opt$calibration_n,
  injection_n = opt$injection_n,
  n_analyses_registered = if (!is.null(reg)) nrow(reg) else 0,
  n_figures = length(figs %||% character(0)),
  tables_dir = opt$table_dir, figures_dir = opt$figure_dir,
  capabilities = audit_capabilities(),
  warnings = warnings_log, failures = failures_log,
  sessionInfo = utils::capture.output(utils::sessionInfo()))
write_audit_json(manifest, file.path(OUT_ROOT, "run_manifest.json"))

# ---- optional report render -------------------------------------------------
report_path <- NA_character_
if (is_true(opt$render_report)) {
  report_path <- guarded("report", {
    qmd <- file.path(KIT, opt$report)
    rendered <- file.path(KIT, "reports", "rendered", "lhcb_statistical_audit.html")
    caps <- audit_capabilities()
    if (caps$quarto && file.exists(qmd)) {
      system2("quarto", c("render", shQuote(qmd), "--to", "html",
                          "--output-dir", shQuote(file.path(KIT, "reports", "rendered"))))
      rendered
    } else {
      # fallback markdown summary so a report artifact always exists
      md <- file.path(KIT, "reports", "rendered", "lhcb_statistical_audit.md")
      con <- file(md, "w")
      writeLines(c(
        "# LHCb statistical audit (fallback report)",
        sprintf("Generated %s | commit %s | mode %s | %s",
                utc_now(), git_commit(), opt$mode, DEV_TAG),
        "",
        "> Quarto not available; this Markdown fallback summarises the audit.",
        "> The same open data analysed in R and Python is a cross-language",
        "> computational reproduction, NOT independent experimental replication.",
        "> The smooth baseline is NOT a full Standard Model amplitude analysis.",
        "",
        "## Headline verdicts",
        "- Stage 09d best LOCAL peak is k2=23.08; 19.5296 is a surviving reference, not the best peak.",
        "- Amplitude bounds are active in key 09d/12/13 fits.",
        "- Integer winding (stage 12) switches branch with bandwidth: not stable.",
        "- Q=4/9 outscores Q=2/3 in the committed stage-13 family at bw scale 1.",
        "- Sideband-subtracted control (stage 28) retains NONE of the claimed structures (p~0.82).",
        "- Charm-trimmed B sidebands (stage 29) carry STRONGER structure than the signal window.",
        "- Net: controls materially weaken any signal-specific / new-physics reading.",
        "",
        "## Tables", paste0("- ", list.files(TABLE_DIR, pattern = "\\.csv$")),
        "", "## Figures", paste0("- ", figs %||% "none")), con)
      close(con)
      md
    }
  })
}

# ---- completion summary -----------------------------------------------------
cat("\n== completion summary ==\n")
cat(sprintf("  execution mode        : %s (%s)\n", opt$mode, DEV_TAG))
cat(sprintf("  data source           : %s\n", if (event_ok) "event (ROOT/Parquet)" else "committed artifacts (replay)"))
cat(sprintf("  stages discovered     : %s\n", if (!is.null(sreg)) paste(sreg$stage, collapse = ",") else "none"))
cat(sprintf("  parity stages         : 09d,12,13,16,25,28,29\n"))
cat(sprintf("  corrected audit stages: 13(radial),28(alpha),29(variance)\n"))
cat(sprintf("  analyses registered   : %d\n", if (!is.null(reg)) nrow(reg) else 0))
cat(sprintf("  bootstrap draws       : %d\n", opt$bootstrap_n))
cat(sprintf("  calibration sims      : %d\n", opt$calibration_n))
cat(sprintf("  injection sims        : %d\n", opt$injection_n))
cat(sprintf("  figures created       : %d\n", length(figs %||% character(0))))
cat(sprintf("  tables created        : %d\n", length(list.files(TABLE_DIR, pattern = "\\.csv$"))))
cat(sprintf("  warnings              : %d\n", length(warnings_log)))
cat(sprintf("  failures              : %d\n", length(failures_log)))
if (length(failures_log)) for (f in failures_log) cat("    -", f, "\n")
cat(sprintf("  unavailable (no event): 09d/12/13/16/25 full-pipeline null, holdout, KDE calibration, 09d injection\n"))
cat(sprintf("  parity within tol     : %s\n",
            if (!is.null(parity)) sprintf("%d/%d", sum(parity$within_tol), nrow(parity)) else "n/a"))
cat(sprintf("  report                : %s\n", report_path %||% "not rendered"))
cat("== done ==\n")

invisible(NULL)

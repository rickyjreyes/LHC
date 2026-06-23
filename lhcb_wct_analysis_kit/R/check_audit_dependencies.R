#!/usr/bin/env Rscript
# =============================================================================
# check_audit_dependencies.R   (section 6)
#
# Reports required + optional packages, prints a valid install.packages() line,
# returns nonzero when required packages are absent, and states whether ROOT,
# Parquet or replay mode is available.
# =============================================================================

REQUIRED <- c("jsonlite", "digest", "optparse", "dplyr", "tidyr", "purrr",
              "readr", "stringr", "tibble", "ggplot2", "scales")
OPTIONAL <- c("patchwork", "ggridges", "ggrepel", "viridis", "gt", "ragg",
              "svglite", "quarto", "boot", "broom", "rsample", "future",
              "future.apply", "progressr", "arrow", "reticulate", "testthat",
              "MASS")

main <- function() {
  ip <- rownames(installed.packages())
  req_missing <- setdiff(REQUIRED, ip)
  opt_missing <- setdiff(OPTIONAL, ip)

  cat("== LHCb statistical-audit dependency check ==\n\n")
  cat("Required packages:\n")
  for (p in REQUIRED) cat(sprintf("  [%s] %s\n", if (p %in% ip) "x" else " ", p))
  cat("\nOptional packages:\n")
  for (p in OPTIONAL) cat(sprintf("  [%s] %s\n", if (p %in% ip) "x" else " ", p))

  need <- c(req_missing, opt_missing)
  if (length(need)) {
    cat("\nInstall missing packages with:\n")
    cat(sprintf('  install.packages(c(%s))\n',
                paste(sprintf('"%s"', need), collapse = ", ")))
  } else cat("\nAll listed packages present.\n")

  # execution-mode availability
  kit <- getwd()
  data_dir <- file.path(kit, "data")
  cache <- file.path(kit, "data_cache", "events.parquet")
  has_root <- dir.exists(data_dir) && length(list.files(data_dir, pattern = "\\.root$")) > 0
  has_parquet <- file.exists(cache)
  has_arrow <- "arrow" %in% ip
  has_reticulate <- "reticulate" %in% ip

  cat("\nExecution modes:\n")
  cat(sprintf("  REPLAY mode      : AVAILABLE (committed artifacts)\n"))
  cat(sprintf("  FULL (ROOT)      : %s\n",
              if (has_root && (has_arrow || has_reticulate)) "AVAILABLE"
              else if (has_root) "ROOT present but need arrow or reticulate"
              else "UNAVAILABLE (no .root under data/)"))
  cat(sprintf("  FULL (Parquet)   : %s\n",
              if (has_parquet && has_arrow) "AVAILABLE"
              else if (has_parquet) "cache present but need arrow"
              else "UNAVAILABLE (no data_cache/events.parquet)"))

  if (length(req_missing)) {
    cat(sprintf("\nFAIL: %d required package(s) missing.\n", length(req_missing)))
    quit(status = 1L)
  }
  cat("\nOK: required packages satisfied.\n")
  quit(status = 0L)
}

if (sys.nframe() == 0L) main()

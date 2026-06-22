#!/usr/bin/env Rscript
# Standalone testthat runner (this kit is not an installed R package).
# Usage:  Rscript tests/testthat.R   (run from lhcb_wct_analysis_kit/)
library(testthat)
kit <- normalizePath(file.path(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), ".."), mustWork = FALSE)
if (is.na(kit) || !dir.exists(file.path(kit, "R"))) kit <- getwd()
Sys.setenv(LHCB_R_DIR = file.path(kit, "R"))
res <- test_dir(file.path(kit, "tests", "testthat"), reporter = "summary",
                stop_on_failure = FALSE)
df <- as.data.frame(res)
cat(sprintf("\nTOTAL pass=%d fail=%d warn=%d skip=%d\n",
            sum(df$passed), sum(df$failed), sum(df$warning), sum(df$skipped)))
if (sum(df$failed) > 0) quit(status = 1)

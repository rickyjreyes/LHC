#!/usr/bin/env Rscript
# Standalone runner for the statistical-audit testthat suite.
suppressWarnings(suppressMessages(library(testthat)))
here <- tryCatch(dirname(normalizePath(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))), error = function(e) getwd())
res <- test_dir(file.path(here, "testthat"), reporter = "summary", stop_on_failure = FALSE)
df <- as.data.frame(res)
fail <- sum(df$failed) + sum(df$error)
cat(sprintf("\nAUDIT TESTS: %d passed, %d failed, %d skipped\n",
            sum(df$passed), fail, sum(df$skipped)))
quit(status = if (fail > 0) 1L else 0L)

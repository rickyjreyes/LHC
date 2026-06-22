library(jsonlite)

CT_DIR <- file.path(KIT, "outputs_charm_trimmed_control")
have <- dir.exists(CT_DIR) &&
  file.exists(file.path(CT_DIR, "charm_trimmed_bins_signal_B_signal_Kst.csv"))

test_that("charm-trimmed raw-count region scan reproduces the committed result", {
  skip_if_not(have, "committed stage-29 outputs not present")
  source(file.path(RDIR, "charm_trimmed_control.R"), local = TRUE)
  b <- read.csv(file.path(CT_DIR, "charm_trimmed_bins_signal_B_signal_Kst.csv"))
  a <- ct_analyze_spectrum("signal_B_signal_Kst", b$ell, b$counts, pmax(b$counts, 1))
  py <- fromJSON(file.path(CT_DIR, "charm_trimmed_summary.json"),
                 simplifyVector = FALSE)$region_results[[1]]
  expect_equal(a$result$scan$best_delta_chi2, py$scan$best_delta_chi2, tolerance = 1e-5)
  expect_equal(a$result$scan$best_k, py$scan$best_k, tolerance = 1e-9)
  expect_equal(a$result$comb$comb_101520_delta_chi2,
               py$comb$comb_101520_delta_chi2, tolerance = 1e-5)
})

test_that("charm sideband-subtracted reproduces the var=max(residual,1) quirk", {
  skip_if_not(have, "committed stage-29 outputs not present")
  source(file.path(RDIR, "charm_trimmed_control.R"), local = TRUE)
  sb <- read.csv(file.path(CT_DIR, "charm_trimmed_sideband_bins.csv"))
  a <- ct_analyze_spectrum("sb", sb$ell, sb$R_subtracted, pmax(sb$R_subtracted, 1))
  py <- fromJSON(file.path(CT_DIR, "charm_trimmed_summary.json"),
                 simplifyVector = FALSE)$sideband_subtracted_result
  expect_equal(a$result$scan$best_delta_chi2, py$scan$best_delta_chi2, tolerance = 1e-5)
})

test_that("B-sideband regions show strong structure (control interpretation holds)", {
  skip_if_not(have, "committed stage-29 outputs not present")
  py <- fromJSON(file.path(CT_DIR, "charm_trimmed_summary.json"),
                 simplifyVector = FALSE)$region_results
  # The control is meaningful precisely because the sidebands are NOT flat.
  expect_gt(py[[2]]$scan$best_delta_chi2, 100)
  expect_gt(py[[3]]$scan$best_delta_chi2, 100)
})

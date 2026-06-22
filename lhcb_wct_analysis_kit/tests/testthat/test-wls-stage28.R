library(jsonlite)

BINS_PY <- file.path(KIT, "outputs_sideband_subtracted", "sideband_subtracted_bins.csv")
have_py <- file.exists(BINS_PY)

test_that("WLS residual and variance construction matches the committed bins", {
  skip_if_not(have_py, "committed Python stage-28 outputs not present")
  b <- read.csv(BINS_PY)
  alpha <- sum(b$N_signal) / max(sum(b$N_side_combined), 1.0)
  expect_equal(alpha, b$alpha[1], tolerance = 1e-12)
  R <- b$N_signal - alpha * b$N_side_combined
  v <- pmax(b$N_signal + alpha^2 * b$N_side_combined, 1.0)
  expect_equal(R, b$R_subtracted, tolerance = 1e-9)
  expect_equal(v, b$variance, tolerance = 1e-9)
})

test_that("histogram edge convention: log-spaced edges over [Q2_MIN,Q2_MAX]", {
  edges <- seq(log(Q2_MIN), log(Q2_MAX), length.out = 241)
  expect_equal(length(edges), 241L)
  expect_equal(edges[1], log(0.1)); expect_equal(edges[241], log(19.0))
})

test_that("continuous one-mode scan reproduces the committed Python scan", {
  skip_if_not(have_py, "committed outputs not present")
  b <- read.csv(BINS_PY)
  k_grid <- seq(6.0, 32.0, length.out = 1301)
  sc <- scan_one_mode(b$ell, b$R_subtracted, b$variance, k_grid)
  ps <- read.csv(file.path(KIT, "outputs_sideband_subtracted",
                           "sideband_subtracted_scan.csv"))
  expect_equal(sc$rows$k, ps$k, tolerance = 1e-12)
  expect_equal(sc$rows$delta_chi2, ps$delta_chi2, tolerance = 1e-9)
  expect_equal(which.max(sc$rows$delta_chi2), which.max(ps$delta_chi2))
})

test_that("scan maximum k and DeltaChi2 hit the regression target", {
  skip_if_not(have_py, "committed outputs not present")
  b <- read.csv(BINS_PY)
  k_grid <- seq(6.0, 32.0, length.out = 1301)
  sc <- scan_one_mode(b$ell, b$R_subtracted, b$variance, k_grid)
  best <- sc$rows[which.max(sc$rows$delta_chi2), ]
  expect_equal(best$k, 8.78, tolerance = 1e-9)
  expect_equal(best$delta_chi2, 5.303549331940928, tolerance = 1e-6)
})

test_that("peak/well detection reproduces the committed wells", {
  skip_if_not(have_py, "committed outputs not present")
  b <- read.csv(BINS_PY)
  k_grid <- seq(6.0, 32.0, length.out = 1301)
  sc <- scan_one_mode(b$ell, b$R_subtracted, b$variance, k_grid)
  wells <- find_wells(sc$rows, 0.5, 0.75)
  pw <- read.csv(file.path(KIT, "outputs_sideband_subtracted",
                           "sideband_subtracted_wells.csv"))
  expect_equal(nrow(wells), nrow(pw))
  expect_equal(wells$k, pw$k, tolerance = 1e-9)
  expect_equal(wells$delta_chi2, pw$delta_chi2, tolerance = 1e-9)
})

test_that("triplet geometry reproduces the committed best triplet", {
  skip_if_not(have_py, "committed outputs not present")
  b <- read.csv(BINS_PY)
  k_grid <- seq(6.0, 32.0, length.out = 1301)
  sc <- scan_one_mode(b$ell, b$R_subtracted, b$variance, k_grid)
  tri <- triplets_from_wells(find_wells(sc$rows, 0.5, 0.75), 12L)
  pt <- read.csv(file.path(KIT, "outputs_sideband_subtracted",
                           "sideband_subtracted_triplets.csv"))
  expect_equal(nrow(tri), nrow(pt))
  expect_equal(tri$koide_error[1], pt$koide_error[1], tolerance = 1e-9)
  expect_equal(tri$score[1], pt$score[1], tolerance = 1e-7)
})

test_that("locked (10,15,20) comb DeltaChi2 hits the regression target", {
  skip_if_not(have_py, "committed outputs not present")
  b <- read.csv(BINS_PY)
  d <- comb_fit_delta(b$ell, b$R_subtracted, b$variance, c(10,15,20))$delta
  expect_equal(d, 4.141641179068927, tolerance = 1e-6)
})

test_that("amplitude-cap distinction: coefficient vs radial bound", {
  # radial amplitude can exceed an individual coefficient bound of 0.1
  a <- 0.09; b <- 0.09
  radial <- sqrt(a^2 + b^2)
  expect_true(a <= 0.1 && b <= 0.1)
  expect_gt(radial, 0.1)
})

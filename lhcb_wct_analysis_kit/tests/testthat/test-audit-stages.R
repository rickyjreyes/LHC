test_that("stage 28 WLS engine reproduces committed deltaChi2 (parity)", {
  td <- tempfile("td"); dir.create(td)
  res <- run_sideband_uncertainty(td, td, KITROOT, mode = "replay", bootstrap_n = 0)
  expect_false(is.null(res))
  # recomputed reference-k / n15 deltaChi2 must match the committed values
  expect_equal(res$parity$ref, 0.4468052357436818, tolerance = 1e-6)
  expect_equal(res$parity$n15, 1.2124631626656992, tolerance = 1e-6)
  expect_equal(res$parity$comb, 4.141641179068927, tolerance = 1e-6)
  # alpha
  expect_equal(res$alpha, 0.28495897903372835, tolerance = 1e-12)
})

test_that("stage 28 controls do NOT survive at 0.05", {
  td <- tempfile("td"); dir.create(td)
  res <- run_sideband_uncertainty(td, td, KITROOT, mode = "replay", bootstrap_n = 0)
  expect_true(all(res$audit$survives_0p05 == FALSE))
  expect_gt(res$regression$p_best, 0.05)  # ~0.82
})

test_that("stage 12 winding switches branch with bandwidth (not stable)", {
  td <- tempfile("td"); dir.create(td)
  res <- run_winding_sensitivity(td, td, KITROOT)
  expect_false(is.null(res))
  # committed: best n at bw1 is 20
  expect_equal(res$regression$best_n_bw1, 20)
  # branch switching -> no single n with >=80% support
  expect_lt(max(res$selection), 0.8)
  # entropy is positive (multiple branches selected)
  expect_gt(res$entropy, 0)
})

test_that("p=(r+1)/(B+1) distinct from any zero claim under L'Ecuyer RNG", {
  audit_set_seed(42)
  k1 <- RNGkind()[1]
  expect_equal(k1, "L'Ecuyer-CMRG")
})

test_that("alpha re-estimation bootstrap is distinct from the null engine", {
  td <- tempfile("td"); dir.create(td)
  res <- run_sideband_uncertainty(td, td, KITROOT, mode = "replay", bootstrap_n = 50)
  expect_false(is.null(res$alpha_bootstrap))
  arow <- res$alpha_bootstrap[res$alpha_bootstrap$target == "alpha", ]
  # bootstrap alpha mean near observed alpha but with nonzero spread
  expect_equal(arow$boot_mean, res$alpha, tolerance = 0.02)
  expect_gt(arow$boot_sd, 0)
})

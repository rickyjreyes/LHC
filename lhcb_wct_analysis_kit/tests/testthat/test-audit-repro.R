test_that("same seed gives identical bootstrap output", {
  td1 <- tempfile("a"); dir.create(td1)
  td2 <- tempfile("b"); dir.create(td2)
  r1 <- run_sideband_uncertainty(td1, td1, KITROOT, bootstrap_n = 40, seed = 999)
  r2 <- run_sideband_uncertainty(td2, td2, KITROOT, bootstrap_n = 40, seed = 999)
  expect_equal(r1$alpha_bootstrap$boot_mean, r2$alpha_bootstrap$boot_mean)
  expect_equal(r1$alpha_bootstrap$boot_sd, r2$alpha_bootstrap$boot_sd)
})

test_that("different seeds give different draws", {
  td1 <- tempfile("c"); dir.create(td1)
  td2 <- tempfile("d"); dir.create(td2)
  r1 <- run_sideband_uncertainty(td1, td1, KITROOT, bootstrap_n = 40, seed = 1)
  r2 <- run_sideband_uncertainty(td2, td2, KITROOT, bootstrap_n = 40, seed = 2)
  expect_false(isTRUE(all.equal(r1$alpha_bootstrap$boot_sd, r2$alpha_bootstrap$boot_sd)))
})

test_that("zero exceedances never return p=0 and respect MC resolution", {
  p <- audit_empirical_p(Inf, runif(500))
  expect_equal(p$r, 0L)
  expect_equal(p$p, 1 / 501)
  expect_equal(p$resolution, 1 / 501)
})

test_that("registry declares both parity and corrected-audit modes per issue", {
  reg <- build_analysis_registry(KITROOT)
  # stage 13 must have a coefficient-bound parity AND a radial corrected entry
  s13 <- reg[reg$stage == "13", ]
  expect_true(any(grepl("coef", s13$amplitude_bound)))
  expect_true(any(grepl("radial", s13$amplitude_bound)))
  # stage 28 fixed-alpha parity AND re-estimated-alpha corrected
  s28 <- reg[reg$stage == "28", ]
  expect_true(any(s28$mode == "parity"))
  expect_true(any(s28$mode == "corrected_audit"))
})

test_that("winding selection probability is a valid distribution", {
  td <- tempfile("w"); dir.create(td)
  r <- run_winding_sensitivity(td, td, KITROOT)
  expect_true(all(r$selection >= 0 & r$selection <= 1))
  expect_equal(sum(r$selection), 1, tolerance = 1e-9)
})

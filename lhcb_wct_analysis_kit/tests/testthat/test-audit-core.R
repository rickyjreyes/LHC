test_that("domain maps: Delta_ell_A and n<->k round-trip", {
  expect_equal(DELTA_ELL_ACTIVE, 4.780150335923678, tolerance = 1e-12)
  expect_equal(n_from_k(k_from_n(15)), 15, tolerance = 1e-9)
  # 10/15/20 winding implied k values
  expect_equal(k_from_n(10), 13.14432573377522, tolerance = 1e-9)
  expect_equal(k_from_n(15), 19.716488600662828, tolerance = 1e-9)
  expect_equal(k_from_n(20), 26.28865146755044, tolerance = 1e-9)
})

test_that("Q=2/3 and Q=4/9 comb geometry", {
  n0 <- 15
  q23 <- c(n0 * 2/3, n0, n0 * 4/3)       # 10,15,20
  q49 <- c(n0 * 4/9, n0, n0 * 8/9)       # 6.667,15,13.333
  expect_equal(q23, c(10, 15, 20))
  expect_equal(q49[1], 6.666666666666666, tolerance = 1e-9)
  expect_equal(q49[3], 13.333333333333332, tolerance = 1e-9)
})

test_that("empirical p uses (r+1)/(B+1) and is never zero", {
  p0 <- audit_empirical_p(1e9, rnorm(100))    # zero exceedances
  expect_equal(p0$r, 0L)
  expect_equal(p0$p, 1 / 101)
  expect_gt(p0$p, 0)
  expect_equal(p0$resolution, 1 / 101)
  p1 <- audit_empirical_p(0, rnorm(99))       # ~half exceed
  expect_true(p1$p > 0 && p1$p <= 1)
})

test_that("derived quantities are consistent", {
  k <- 19.5296
  expect_equal(log_period(k), 2 * pi / k)
  expect_equal(rho_q2(k), exp(2 * pi / k))
  expect_equal(peak_to_trough_ratio(0.1), exp(0.2))
  expect_equal(winding_n(k_from_n(12)), 12, tolerance = 1e-9)
})

test_that("registry has unique analysis ids and required columns", {
  reg <- build_analysis_registry(KITROOT)
  expect_false(anyDuplicated(reg$analysis_id) > 0)
  expect_true(all(c("analysis_id", "stage", "mode", "null_model") %in% names(reg)))
  expect_true(any(reg$mode == "corrected_audit"))
  expect_true(any(reg$mode == "parity"))
})

test_that("event flow reconciles committed regression targets", {
  flow <- build_event_flow(KITROOT)
  rec <- reconcile_event_flow(flow)
  expect_true(rec$selected_298801)
  expect_true(rec$retained_bins_43)
  expect_true(rec$signal_active_15863)
  expect_true(rec$low_sb_29533)
  expect_true(rec$high_sb_26244)
  expect_true(rec$signal_hist_15630)
  expect_true(rec$side_hist_54850)
  expect_true(rec$alpha_ok)
})

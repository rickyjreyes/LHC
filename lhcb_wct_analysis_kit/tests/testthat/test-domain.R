test_that("active-domain length matches the reference constant", {
  expect_equal(DELTA_ELL_ACTIVE, 4.780150335923678, tolerance = 1e-12)
})

test_that("k<->n winding maps are exact inverses and hit reference frequencies", {
  expect_equal(k_from_n(10), 13.14432573377522, tolerance = 1e-10)
  expect_equal(k_from_n(15), 19.716488600662828, tolerance = 1e-10)
  expect_equal(k_from_n(20), 26.28865146755044, tolerance = 1e-10)
  expect_equal(n_from_k(k_from_n(13.37)), 13.37, tolerance = 1e-12)
})

test_that("active-interval masking is correct at the veto boundaries", {
  q2 <- c(0.05, 0.1, 5, 8.0, 9.0, 11.0, 12.0, 12.5, 13.5, 14.5, 18, 19, 20)
  m <- in_active_intervals(q2)
  expect_equal(m, c(FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE,
                    TRUE, TRUE, TRUE, FALSE))
})

test_that("veto mask flags charmonium windows", {
  expect_true(all(in_veto_q2(c(8.5, 10.9, 13.0, 14.4))))
  expect_false(any(in_veto_q2(c(7.9, 11.1, 12.4, 14.6))))
})

test_that("veto-scheme active-domain recalculation changes Delta_ell_A", {
  iv <- active_intervals_from_vetoes(c(8.0, 11.0), c(12.5, 15.0))
  expect_equal(length(iv), 3L)
  d <- active_delta_ell(iv)
  expect_true(is.finite(d) && d > 0)
  # tighter veto -> larger retained domain
  iv2 <- active_intervals_from_vetoes(c(8.5, 10.5), c(13.0, 14.0))
  expect_gt(active_delta_ell(iv2), d)
})

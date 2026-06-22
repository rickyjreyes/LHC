library(jsonlite)

test_that("Poisson deviance handles zero counts and matches the formula", {
  N <- c(0, 5, 10); lam <- c(2, 5, 8)
  expected <- 2 * sum(c(lam[1] - N[1],
                        lam[2] - N[2] + N[2]*log(N[2]/lam[2]),
                        lam[3] - N[3] + N[3]*log(N[3]/lam[3])))
  expect_equal(poisson_deviance(N, lam), expected, tolerance = 1e-12)
  expect_equal(poisson_deviance(c(3,4), c(3,4)), 0, tolerance = 1e-12)
})

test_that("polar coefficient conversion uses phi = atan2(-b,a) convention", {
  ab <- ab_from_polar(0.08, 0.5)
  expect_equal(unname(ab[1]), 0.08*cos(0.5), tolerance = 1e-14)
  expect_equal(unname(ab[2]), -0.08*sin(0.5), tolerance = 1e-14)
})

test_that("bounded base fit reproduces the Python L-BFGS-B reference", {
  ref <- fromJSON(file.path(FIX, "poisson_reference.json"))
  fit <- fit_base_bounded(ref$N, ref$B, ref$ell, ref$k1)
  expect_equal(fit$D_base, ref$base$D_base, tolerance = 1e-6)
  expect_equal(fit$A1, ref$base$A1, tolerance = 1e-5)
})

test_that("bounded two-mode fit reproduces the Python DeltaD at k2=20", {
  ref <- fromJSON(file.path(FIX, "poisson_reference.json"))
  base <- fit_base_bounded(ref$N, ref$B, ref$ell, ref$k1)
  two <- fit_two_bounded(ref$N, ref$B, ref$ell, ref$k1, 20.0, base = base)
  expect_equal(base$D_base - two$D_two, ref$two_k2_20$deltaD, tolerance = 1e-6)
})

test_that("multi-start optimizer beats a single naive start", {
  ref <- fromJSON(file.path(FIX, "poisson_reference.json"))
  N <- ref$N; B <- ref$B; ell <- ref$ell; k1 <- ref$k1
  single <- stats::optim(c(0,0,0), .polar_base_nll, method = "L-BFGS-B",
    lower = c(-0.2,0,-pi), upper = c(0.2,0.1,pi), N=N, B=B, ell=ell, k1=k1)$value
  best <- fit_base_bounded(N, B, ell, k1)
  best_nll <- .polar_base_nll(c(best$C, best$A1, best$phi1), N, B, ell, k1)
  expect_lte(best_nll, single + 1e-8)
})

test_that("amplitude-cap detection flags bound-active radius", {
  expect_true(abs(0.1 - A1_MAX) <= 1e-5)
  ref <- fromJSON(file.path(FIX, "poisson_reference.json"))
  base <- fit_base_bounded(ref$N, ref$B, ref$ell, ref$k1)
  two <- fit_two_bounded(ref$N, ref$B, ref$ell, ref$k1, 20.0, base = base)
  expect_type(two$amplitude2_bound_active, "logical")
})

test_that("projected-Newton null engine reproduces the Python scan", {
  ref <- fromJSON(file.path(FIX, "poisson_reference.json"))
  pnb <- pn_fit_base(ref$N, ref$B, ref$ell, ref$k1)
  expect_equal(pnb$D_base, ref$pn_base_Dbase, tolerance = 1e-9)
  sc <- pn_scan_two(ref$N, ref$B, ref$ell, ref$k1, ref$pn_scan$k2_grid)
  expect_equal(sc$best_idx, ref$pn_scan$best_idx + 1L)  # R is 1-based
  expect_equal(sc$best_delta, ref$pn_scan$best_delta, tolerance = 1e-9)
  expect_equal(sc$best_k2, ref$pn_scan$best_k2, tolerance = 1e-10)
})

test_that("deterministic null fixture matches across engines", {
  ref <- fromJSON(file.path(FIX, "poisson_reference.json"))
  nf <- ref$null_fixture
  sc <- pn_scan_two(nf$N, ref$B, ref$ell, ref$k1, ref$pn_scan$k2_grid)
  expect_equal(sc$best_delta, nf$best_delta, tolerance = 1e-9)
})

test_that("empirical p-value uses the (1+count)/(1+N) convention", {
  expect_equal(p_value(5, c(1,2,3,4)), (1+0)/(1+4))
  expect_equal(p_value(2, c(1,2,3,4)), (1+3)/(1+4))
})

test_that("deterministic reruns with the same seed are identical", {
  RNGkind("L'Ecuyer-CMRG"); set.seed(42); a <- rnorm(20)
  RNGkind("L'Ecuyer-CMRG"); set.seed(42); b <- rnorm(20)
  expect_identical(a, b)
})

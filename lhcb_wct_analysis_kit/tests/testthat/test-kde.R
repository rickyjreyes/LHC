library(jsonlite)

test_that("SciPy-compatible KDE matches gaussian_kde at multiple bandwidths", {
  x <- read.csv(file.path(FIX, "kde_sample.csv"))$x
  for (f in c("kde_reference_scale_1p0.json", "kde_reference_scale_1p5.json")) {
    ref <- fromJSON(file.path(FIX, f))
    kde <- gaussian_kde(x, "scott", ref$scale)
    expect_equal(kde$factor, ref$factor, tolerance = 1e-12)
    expect_equal(kde$covariance, ref$covariance, tolerance = 1e-10)
    d <- kde$evaluate(ref$points)
    expect_equal(d, as.numeric(ref$density), tolerance = 1e-12)
  }
})

test_that("Scott factor uses neff^(-1/5) for 1-D unweighted data", {
  x <- rnorm(100)
  kde <- gaussian_kde(x, "scott", 1.0)
  expect_equal(kde$scotts_factor, 100^(-1/5), tolerance = 1e-12)
})

test_that("KDE baseline normalization scales with bin width and n_train", {
  x <- read.csv(file.path(FIX, "kde_sample.csv"))$x
  centers <- seq(0.5, 18, length.out = 30)
  b1 <- kde_baseline(x, centers, bin_width = 0.3, bw_scale = 1.5)
  b2 <- kde_baseline(x, centers, bin_width = 0.6, bw_scale = 1.5)
  expect_equal(b2, 2 * b1, tolerance = 1e-9)
  expect_true(all(b1 >= 1e-9))
})

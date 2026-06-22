# =============================================================================
# lhcb_kde.R
#
# Explicit one-dimensional Gaussian KDE compatible with scipy.stats.gaussian_kde.
#
# R's stats::density() uses a different (binned, FFT-based) estimator with a
# different default bandwidth rule, so it cannot be substituted here without
# breaking numerical parity. This module reproduces the scipy estimator exactly:
#
#   * Scott bandwidth factor:    factor = neff^(-1/(d+4)),  d = 1
#   * neff (unweighted):         n
#   * data covariance:           unbiased sample variance (ddof = 1)
#   * bandwidth multiplier:      set_bandwidth(factor * scale)
#   * smoothing variance:        h2 = var(x, ddof=1) * (factor*scale)^2
#   * Gaussian kernel:           exp(-0.5 u^2 / h2) / sqrt(2*pi*h2)
#   * density at point x:        (1/n) * sum_i kernel(x - x_i)
#
# Parity is asserted directly against scipy in tests/testthat/test-kde.R using
# a committed fixture (R/fixtures/kde_reference.json).
# =============================================================================

#' Construct a scipy-compatible 1-D Gaussian KDE object.
#'
#' @param x numeric training sample.
#' @param bw_method currently only "scott" is supported (the value used by 09d).
#' @param bw_scale multiplicative bandwidth scale (KDE_BANDWIDTH_SCALE).
#' @return a list with the fitted bandwidth and an evaluate() closure.
gaussian_kde <- function(x, bw_method = "scott", bw_scale = 1.0) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  n <- length(x)
  if (n < 2) stop("gaussian_kde requires at least 2 finite points")

  d <- 1L
  neff <- n  # unweighted effective sample size

  if (!identical(bw_method, "scott")) {
    stop("Only bw_method='scott' is supported for parity.")
  }
  scotts_factor <- neff^(-1.0 / (d + 4))
  factor <- scotts_factor * bw_scale

  # Unbiased (ddof = 1) sample variance, matching np.cov(..., bias=False).
  data_var <- stats::var(x)            # R var() already uses (n-1)
  covariance <- data_var * factor^2
  norm_const <- sqrt(2.0 * pi * covariance)

  evaluate <- function(points) {
    points <- as.numeric(points)
    out <- numeric(length(points))
    inv2cov <- 1.0 / (2.0 * covariance)
    for (i in seq_along(points)) {
      u <- points[i] - x
      out[i] <- sum(exp(-(u * u) * inv2cov))
    }
    out / (n * norm_const)
  }

  list(
    n = n,
    factor = factor,
    scotts_factor = scotts_factor,
    bw_scale = bw_scale,
    covariance = covariance,
    bandwidth = sqrt(covariance),
    evaluate = evaluate
  )
}

#' Build a KDE-based expected-count baseline at histogram bin centers.
#'
#' Reproduces make_kde_baseline() in 09d:
#'   baseline_i = density(center_i) * n_train * bin_width, floored at 1e-9,
#' where the KDE is trained on q2 values inside [Q2_MIN, Q2_MAX] and outside
#' the widened charmonium vetoes.
#'
#' @param q2_values event-level q2 sample.
#' @param centers   linear-q2 histogram bin centers.
#' @param bin_width linear-q2 histogram bin width.
#' @param q2_min,q2_max training-domain bounds.
#' @param bw_scale  bandwidth scale.
kde_baseline <- function(q2_values, centers, bin_width,
                         q2_min = 0.1, q2_max = 19.0, bw_scale = 1.50,
                         veto_fn = NULL) {
  q2_values <- as.numeric(q2_values)
  keep <- is.finite(q2_values) & q2_values >= q2_min & q2_values <= q2_max
  if (!is.null(veto_fn)) keep <- keep & !veto_fn(q2_values)
  train <- q2_values[keep]
  if (length(train) < 100) {
    stop(sprintf("Too few events for KDE baseline after vetoes: %d", length(train)))
  }
  kde <- gaussian_kde(train, bw_method = "scott", bw_scale = bw_scale)
  dens <- kde$evaluate(centers)
  baseline <- dens * length(train) * bin_width
  pmax(baseline, 1e-9)
}

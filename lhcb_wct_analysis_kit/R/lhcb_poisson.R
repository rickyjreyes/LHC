# =============================================================================
# lhcb_poisson.R
#
# Bounded-Poisson log-cos fit engine shared by the two-mode KDE scan (09d),
# the integer-winding scan (12), the Koide comb scan (13) and the WCT-vs-smooth
# likelihood test (16).
#
# Reproduces the numerical core of 09d_two_mode_kde_baseline_polar_cupy.py:
#   * poisson_deviance:        D = 2 sum(lam - N + N log(N/lam)), zero-safe
#   * ab_from_polar:           a = r cos(phi),  b = -r sin(phi)
#   * fit_base_bounded:        polar L-BFGS-B base fit (multi-start)
#   * fit_two_bounded:         polar L-BFGS-B two-mode fit (multi-start)
#   * gpu_scan_two_batch_R:    python-compatible projected-Newton null engine
#                              (10 iters, ridge 1e-8, eta clip, radial project)
#
# The "exact" null engine simply refits each null with the bounded optimizer.
# =============================================================================

A1_MAX <- 0.10
A2_MAX <- 0.10
ETA_CLIP <- 0.2
IRLS_ITERS <- 10L
IRLS_RIDGE <- 1e-8

#' Poisson deviance with zero-count terms handled correctly.
poisson_deviance <- function(N, lam) {
  N <- as.numeric(N)
  lam <- pmax(as.numeric(lam), 1e-12)
  term <- lam - N
  nz <- N > 0
  term[nz] <- term[nz] + N[nz] * log(N[nz] / lam[nz])
  2.0 * sum(term)
}

#' Polar -> rectangular coefficient conversion (phi = atan2(-b, a) convention).
ab_from_polar <- function(r, phi) c(a = r * cos(phi), b = -r * sin(phi))

.poisson_nll_from_eta <- function(N, B, eta) {
  lam <- pmax(B * exp(eta), 1e-12)
  sum(lam - N * log(lam))
}

.polar_base_nll <- function(theta, N, B, ell, k1) {
  C <- theta[1]; r1 <- theta[2]; phi1 <- theta[3]
  ab <- ab_from_polar(r1, phi1)
  eta <- C + ab[1] * cos(k1 * ell) + ab[2] * sin(k1 * ell)
  .poisson_nll_from_eta(N, B, eta)
}

.polar_two_nll <- function(theta, N, B, ell, k1, k2) {
  C <- theta[1]; r1 <- theta[2]; phi1 <- theta[3]; r2 <- theta[4]; phi2 <- theta[5]
  ab1 <- ab_from_polar(r1, phi1); ab2 <- ab_from_polar(r2, phi2)
  eta <- C + ab1[1] * cos(k1 * ell) + ab1[2] * sin(k1 * ell) +
    ab2[1] * cos(k2 * ell) + ab2[2] * sin(k2 * ell)
  .poisson_nll_from_eta(N, B, eta)
}

.safe_log_scale <- function(N, B) {
  c0 <- log(max(sum(N), 1e-12) / max(sum(B), 1e-12))
  max(min(c0, ETA_CLIP), -ETA_CLIP)
}

#' Multi-start bounded L-BFGS-B, keeping the lowest objective.
.best_minimize <- function(fn, starts, lower, upper, ...) {
  best <- NULL
  for (x0 in starts) {
    res <- tryCatch(
      stats::optim(par = x0, fn = fn, method = "L-BFGS-B",
                   lower = lower, upper = upper,
                   control = list(maxit = 2000, factr = 1e1, pgtol = 1e-8),
                   ...),
      error = function(e) NULL)
    if (is.null(res)) next
    if (is.null(best) || res$value < best$value) best <- res
  }
  best
}

#' Polar bounded base fit (fit_base_cpu_bounded).
fit_base_bounded <- function(N, B, ell, k1, a1_max = A1_MAX) {
  N <- as.numeric(N); B <- pmax(as.numeric(B), 1e-12); ell <- as.numeric(ell)
  C0 <- .safe_log_scale(N, B)
  starts <- list()
  for (r in c(0.0, 0.5 * a1_max, a1_max)) {
    for (ph in c(0.0, 0.5 * pi, -0.5 * pi, pi)) {
      starts[[length(starts) + 1]] <- c(C0, r, ph)
    }
  }
  lower <- c(-ETA_CLIP, 0.0, -pi); upper <- c(ETA_CLIP, a1_max, pi)
  res <- .best_minimize(.polar_base_nll, starts, lower, upper,
                        N = N, B = B, ell = ell, k1 = k1)
  C <- res$par[1]; r1 <- res$par[2]; phi1 <- res$par[3]
  ab <- ab_from_polar(r1, phi1)
  eta <- C + ab[1] * cos(k1 * ell) + ab[2] * sin(k1 * ell)
  lam <- pmax(B * exp(eta), 1e-12)
  list(k1 = k1, D_base = poisson_deviance(N, lam),
       C = C, a1 = ab[1], b1 = ab[2], A1 = r1, phi1 = phi1,
       lambda_base = lam, success = (res$convergence == 0),
       n_iter = if (!is.null(res$counts)) unname(res$counts[1]) else -1L,
       optimizer = "polar_LBFGSB",
       amplitude1_bound_active = abs(r1 - a1_max) <= 1e-5)
}

#' Polar bounded two-mode fit (fit_two_cpu_bounded).
fit_two_bounded <- function(N, B, ell, k1, k2, a1_max = A1_MAX, a2_max = A2_MAX,
                            base = NULL) {
  N <- as.numeric(N); B <- pmax(as.numeric(B), 1e-12); ell <- as.numeric(ell)
  if (is.null(base)) base <- fit_base_bounded(N, B, ell, k1, a1_max)
  C0 <- max(min(base$C, ETA_CLIP), -ETA_CLIP)
  r10 <- max(min(base$A1, a1_max), 0.0)
  ph10 <- max(min(base$phi1, pi), -pi)

  starts <- list()
  for (r2 in c(0.0, 0.5 * a2_max, a2_max)) {
    for (ph2 in c(0.0, 0.5 * pi, -0.5 * pi, pi)) {
      starts[[length(starts) + 1]] <- c(C0, r10, ph10, r2, ph2)
    }
  }
  for (ph1 in c(0.0, 0.5 * pi, -0.5 * pi, pi)) {
    starts[[length(starts) + 1]] <- c(C0, a1_max, ph1, a2_max, 0.0)
    starts[[length(starts) + 1]] <- c(C0, a1_max, ph1, a2_max, pi)
  }
  lower <- c(-ETA_CLIP, 0.0, -pi, 0.0, -pi)
  upper <- c(ETA_CLIP, a1_max, pi, a2_max, pi)
  res <- .best_minimize(.polar_two_nll, starts, lower, upper,
                        N = N, B = B, ell = ell, k1 = k1, k2 = k2)
  C <- res$par[1]; r1 <- res$par[2]; phi1 <- res$par[3]
  r2 <- res$par[4]; phi2 <- res$par[5]
  ab1 <- ab_from_polar(r1, phi1); ab2 <- ab_from_polar(r2, phi2)
  eta <- C + ab1[1] * cos(k1 * ell) + ab1[2] * sin(k1 * ell) +
    ab2[1] * cos(k2 * ell) + ab2[2] * sin(k2 * ell)
  lam <- pmax(B * exp(eta), 1e-12)
  list(k1 = k1, k2 = k2, D_two = poisson_deviance(N, lam),
       C = C, a1 = ab1[1], b1 = ab1[2], A1 = r1, phi1 = phi1,
       a2 = ab2[1], b2 = ab2[2], A2 = r2, phi2 = phi2,
       lambda_two = lam, success = (res$convergence == 0),
       n_iter = if (!is.null(res$counts)) unname(res$counts[1]) else -1L,
       optimizer = "polar_LBFGSB",
       amplitude1_bound_active = abs(r1 - a1_max) <= 1e-5,
       amplitude2_bound_active = abs(r2 - a2_max) <= 1e-5)
}

# -----------------------------------------------------------------------------
# Coefficient-bounded Poisson fit (stage 13 canonical parity)
#
# lambda = B exp(X beta), intercept free, every cos/sin coefficient in
# [-A_MAX, A_MAX]. Single start (beta0 = 0), L-BFGS-B, eta clipped to [-20,20].
# This constrains each coefficient independently -- it is NOT a radial-amplitude
# cap, so sqrt(a^2+b^2) can exceed A_MAX even when |a|,|b| <= A_MAX.
# -----------------------------------------------------------------------------

#' Design matrix: intercept, fixed k1 cos/sin, then cos/sin for each comb k.
comb_basis_matrix <- function(ell, ks, k1 = 7.61054) {
  cols <- list(rep(1.0, length(ell)), cos(k1 * ell), sin(k1 * ell))
  for (k in ks) { cols[[length(cols)+1]] <- cos(k*ell); cols[[length(cols)+1]] <- sin(k*ell) }
  do.call(cbind, cols)
}

fit_coeffbound <- function(counts, baseline, X, a_max = 0.10) {
  y <- as.numeric(counts); B <- pmax(as.numeric(baseline), 1e-12)
  p <- ncol(X); beta0 <- rep(0.0, p)
  lower <- c(-Inf, rep(-a_max, p - 1)); upper <- c(Inf, rep(a_max, p - 1))
  nll <- function(beta) {
    eta <- pmin(pmax(as.numeric(X %*% beta), -20), 20)
    lam <- B * exp(eta)
    sum(lam - y * log(pmax(lam, 1e-12)))
  }
  res <- stats::optim(beta0, nll, method = "L-BFGS-B", lower = lower, upper = upper,
                      control = list(maxit = 2000, factr = 1e2, pgtol = 1e-8))
  beta <- res$par
  eta <- pmin(pmax(as.numeric(X %*% beta), -20), 20)
  lam <- B * exp(eta)
  list(success = (res$convergence == 0), dev = poisson_deviance(y, lam),
       nll = res$value, beta = beta,
       coefficient_bound_active = any(abs(beta[-1]) >= a_max - 1e-5))
}

#' Radial amplitudes from a fitted coefficient-bounded beta (pairs after intercept).
comb_radial_amplitudes <- function(beta) {
  pairs <- (length(beta) - 1L) / 2L
  vapply(seq_len(pairs), function(j) sqrt(beta[2*j]^2 + beta[2*j+1]^2), numeric(1))
}

#' SciPy-compatible KDE baseline built from histogram centers repeated by counts
#' (stage 13 kde_baseline): density normalized to total counts.
kde_baseline_from_hist <- function(ell_centers, counts, bandwidth_scale = 1.0) {
  repeated <- rep(ell_centers, pmax(as.integer(counts), 0L))
  if (length(repeated) < 100) stop("Too few repeated points for KDE baseline.")
  kde <- gaussian_kde(repeated, "scott", bandwidth_scale)
  dens <- pmax(kde$evaluate(ell_centers), 1e-12)
  pmax(dens / sum(dens) * sum(counts), 1e-9)
}

#' Empirical p-value (1 + #{null >= real}) / (1 + N).
p_value <- function(real, null_vals) {
  (1 + sum(null_vals >= real)) / (1 + length(null_vals))
}

gaussian_sigma_from_p <- function(p) {
  # two-sided-equivalent z from a one-tailed p, matching erfcinv form.
  if (p <= 0) return(Inf)
  stats::qnorm(1 - p)
}

# -----------------------------------------------------------------------------
# Python-compatible projected-Newton null engine
#
# Reproduces gpu_fit_base_batch / gpu_scan_two_batch from 09d on CPU:
#   * fixed IRLS_ITERS projected-Newton iterations
#   * ridge IRLS_RIDGE on the Hessian
#   * eta clipped to [-ETA_CLIP, ETA_CLIP]
#   * radial projection of mode-1 and mode-2 coefficients onto their caps
#   * full k2 scan for every null realization
# -----------------------------------------------------------------------------

.project_radial <- function(a, b, cap) {
  A <- sqrt(a * a + b * b)
  s <- pmin(1.0, cap / pmax(A, 1e-12))
  list(a = a * s, b = b * s)
}

#' Projected-Newton base fit for one null count vector.
pn_fit_base <- function(N, B, ell, k1, a1_max = A1_MAX) {
  X <- cbind(1, cos(k1 * ell), sin(k1 * ell))
  beta <- c(0, 0, 0)
  eye <- diag(3)
  for (it in seq_len(IRLS_ITERS)) {
    eta <- as.numeric(X %*% beta)
    eta <- pmin(pmax(eta, -ETA_CLIP), ETA_CLIP)
    mu <- B * exp(eta)
    resid <- mu - N
    grad <- as.numeric(t(X) %*% resid)
    H <- t(X) %*% (X * mu) + IRLS_RIDGE * eye
    step <- solve(H, grad)
    beta <- beta - step
    pr <- .project_radial(beta[2], beta[3], a1_max)
    beta[2] <- pr$a; beta[3] <- pr$b
  }
  eta <- pmin(pmax(as.numeric(X %*% beta), -ETA_CLIP), ETA_CLIP)
  lam <- B * exp(eta)
  list(D_base = poisson_deviance(N, lam), beta = beta)
}

#' Projected-Newton two-mode scan for one null count vector; returns the per-k2
#' DeltaD_add and the scan maximum.
pn_scan_two <- function(N, B, ell, k1, k2_grid, a1_max = A1_MAX, a2_max = A2_MAX) {
  base <- pn_fit_base(N, B, ell, k1, a1_max)
  D_base <- base$D_base
  c1 <- cos(k1 * ell); s1 <- sin(k1 * ell)
  delta <- numeric(length(k2_grid))
  eye <- diag(5)
  for (ki in seq_along(k2_grid)) {
    k2 <- k2_grid[ki]
    X <- cbind(1, c1, s1, cos(k2 * ell), sin(k2 * ell))
    beta <- rep(0, 5)
    for (it in seq_len(IRLS_ITERS)) {
      eta <- pmin(pmax(as.numeric(X %*% beta), -ETA_CLIP), ETA_CLIP)
      mu <- B * exp(eta)
      grad <- as.numeric(t(X) %*% (mu - N))
      H <- t(X) %*% (X * mu) + IRLS_RIDGE * eye
      beta <- beta - solve(H, grad)
      p1 <- .project_radial(beta[2], beta[3], a1_max); beta[2] <- p1$a; beta[3] <- p1$b
      p2 <- .project_radial(beta[4], beta[5], a2_max); beta[4] <- p2$a; beta[5] <- p2$b
    }
    eta <- pmin(pmax(as.numeric(X %*% beta), -ETA_CLIP), ETA_CLIP)
    lam <- B * exp(eta)
    delta[ki] <- D_base - poisson_deviance(N, lam)
  }
  best_idx <- which.max(delta)
  list(delta = delta, best_delta = delta[best_idx],
       best_k2 = k2_grid[best_idx], best_idx = best_idx)
}

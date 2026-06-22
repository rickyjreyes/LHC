# =============================================================================
# lhcb_wls.R
#
# Weighted-least-squares residual engine shared by the sideband-subtracted
# control (stage 28) and the charm-trimmed control (stage 29).
#
# Reproduces the deterministic numerical core of 28_sideband.py:
#   * wls_fit:        constant + fixed k1 cos/sin + extra k cos/sin via lstsq
#   * scan_one_mode:  continuous one-mode DeltaChi2 scan
#   * find_wells:     scipy.signal.find_peaks-compatible peak detection
#   * triplets:       Koide/integer triplet construction and scoring
#   * comb_fit_delta: locked multi-mode comb DeltaChi2
#
# All amplitude / phase / chi2 conventions match the Python script so that the
# committed outputs in outputs_sideband_subtracted/ can be reproduced exactly
# from the committed per-bin inputs (sideband_subtracted_bins.csv).
# =============================================================================

K1_FIXED <- 7.61054
K_REF <- 19.5296
KOIDE_Q <- 2.0 / 3.0

#' Weighted least-squares fit of a constant + fixed k1 mode (+ optional extra
#' modes) to residual y with per-bin variance var.
#'
#' Mirrors numpy.linalg.lstsq on the whitened design matrix.
wls_fit <- function(ell, y, var, ks_extra = numeric(0), include_k1 = TRUE,
                    k1 = K1_FIXED) {
  ell <- as.numeric(ell); y <- as.numeric(y)
  var <- pmax(as.numeric(var), 1.0)

  cols <- list(rep(1.0, length(ell)))
  if (include_k1) {
    cols[[length(cols) + 1]] <- cos(k1 * ell)
    cols[[length(cols) + 1]] <- sin(k1 * ell)
  }
  for (k in ks_extra) {
    cols[[length(cols) + 1]] <- cos(k * ell)
    cols[[length(cols) + 1]] <- sin(k * ell)
  }
  X <- do.call(cbind, cols)
  w <- 1.0 / sqrt(var)
  Xw <- X * w
  yw <- y * w

  # Least squares via QR (matches lstsq minimum-residual solution for full rank).
  beta <- qr.coef(qr(Xw), yw)
  beta[is.na(beta)] <- 0.0
  pred <- as.numeric(X %*% beta)
  chi2 <- sum((y - pred)^2 / var)

  amps <- list()
  idx <- 2L  # column 1 is the constant
  if (include_k1) {
    amps[["A_k1"]] <- sqrt(beta[idx]^2 + beta[idx + 1]^2)
    idx <- idx + 2L
  }
  for (k in ks_extra) {
    amp <- sqrt(beta[idx]^2 + beta[idx + 1]^2)
    phase <- atan2(-beta[idx + 1], beta[idx])
    amps[[sprintf("A_k_%.6f", k)]] <- amp
    amps[[sprintf("phi_k_%.6f", k)]] <- phase
    idx <- idx + 2L
  }

  list(chi2 = chi2, beta = beta, pred = pred, amps = amps,
       ndof = length(y) - length(beta))
}

#' Continuous one-mode DeltaChi2 scan over a k grid.
scan_one_mode <- function(ell, y, var, k_grid, k1 = K1_FIXED,
                          delta_ell = DELTA_ELL_ACTIVE) {
  base <- wls_fit(ell, y, var, ks_extra = numeric(0), include_k1 = TRUE, k1 = k1)
  chi2_base <- base$chi2
  n <- length(k_grid)
  out <- data.frame(k = numeric(n), n_eff = numeric(n),
                    delta_chi2 = numeric(n), amp = numeric(n), phase = numeric(n))
  for (i in seq_len(n)) {
    k <- k_grid[i]
    fit <- wls_fit(ell, y, var, ks_extra = k, include_k1 = TRUE, k1 = k1)
    out$k[i] <- k
    out$n_eff[i] <- n_from_k(k, delta_ell)
    out$delta_chi2[i] <- chi2_base - fit$chi2
    out$amp[i] <- fit$amps[[sprintf("A_k_%.6f", k)]]
    out$phase[i] <- fit$amps[[sprintf("phi_k_%.6f", k)]]
  }
  list(base = base, rows = out)
}

#' scipy.signal.find_peaks-compatible local maxima with prominence and minimum
#' inter-peak distance, restricted to the subset used by 28_sideband.py
#' (1-D real array, prominence threshold, distance in samples).
find_peaks_compat <- function(y, prominence = 0.5, distance = 1L) {
  n <- length(y)
  # --- local maxima (handle flat-topped plateaus like scipy) ---
  peaks <- integer(0)
  i <- 1L
  while (i < n) {
    if (y[i + 1L] > y[i]) {
      i <- i + 1L
    } else if (y[i + 1L] < y[i]) {
      # i is a candidate right edge of a (possibly flat) max
      if (i >= 2L && y[i - 1L] < y[i]) peaks <- c(peaks, i)
      i <- i + 1L
    } else {
      # flat region: find its extent
      i_ahead <- i + 1L
      while (i_ahead < n && y[i_ahead + 1L] == y[i]) i_ahead <- i_ahead + 1L
      if (i_ahead < n && y[i_ahead + 1L] < y[i] && i >= 2L && y[i - 1L] < y[i]) {
        peaks <- c(peaks, (i + i_ahead) %/% 2L)
      }
      i <- i_ahead + 1L
    }
  }
  if (length(peaks) == 0L) {
    return(list(peaks = integer(0), prominences = numeric(0)))
  }

  # --- prominences (scipy peak_prominences, wlen = full signal) ---
  prom <- numeric(length(peaks))
  for (pi in seq_along(peaks)) {
    p <- peaks[pi]
    h <- y[p]
    # left base
    li <- p - 1L; left_min <- h
    while (li >= 1L && y[li] < h) { if (y[li] < left_min) left_min <- y[li]; li <- li - 1L }
    # the boundary where signal first reaches >= h on the left
    left_base <- if (li >= 1L) min(y[(li + 1L):p]) else min(y[1L:p])
    # right base
    ri <- p + 1L; right_min <- h
    while (ri <= n && y[ri] < h) { if (y[ri] < right_min) right_min <- y[ri]; ri <- ri + 1L }
    right_base <- if (ri <= n) min(y[p:(ri - 1L)]) else min(y[p:n])
    prom[pi] <- h - max(left_base, right_base)
  }

  keep <- prom >= prominence
  peaks <- peaks[keep]; prom <- prom[keep]
  if (length(peaks) == 0L) return(list(peaks = integer(0), prominences = numeric(0)))

  # --- minimum distance filter (scipy: highest first, remove neighbours) ---
  if (distance > 1L && length(peaks) > 1L) {
    priority <- order(y[peaks])             # ascending priority value
    keep_mask <- rep(TRUE, length(peaks))
    # scipy iterates from highest priority to lowest
    for (j in rev(priority)) {
      if (!keep_mask[j]) next
      k <- j - 1L
      while (k >= 1L && (peaks[j] - peaks[k]) < distance) {
        keep_mask[k] <- FALSE; k <- k - 1L
      }
      k <- j + 1L
      while (k <= length(peaks) && (peaks[k] - peaks[j]) < distance) {
        keep_mask[k] <- FALSE; k <- k + 1L
      }
    }
    peaks <- peaks[keep_mask]; prom <- prom[keep_mask]
  }

  list(peaks = peaks, prominences = prom)
}

#' Well (peak in DeltaChi2) detection over a scan, mirroring find_wells().
find_wells <- function(scan_rows, min_prominence = 0.5, min_distance_k = 0.75) {
  if (nrow(scan_rows) == 0L) return(data.frame())
  y <- scan_rows$delta_chi2
  k_grid <- scan_rows$k
  dk <- stats::median(diff(k_grid))
  min_dist <- max(1L, as.integer(round(min_distance_k / dk)))

  pk <- find_peaks_compat(y, prominence = min_prominence, distance = min_dist)
  if (length(pk$peaks) == 0L) return(data.frame())

  ord <- order(y[pk$peaks], decreasing = TRUE)
  wells <- data.frame()
  rank <- 1L
  for (oi in ord) {
    pidx <- pk$peaks[oi]
    n_eff <- scan_rows$n_eff[pidx]
    nearest <- round(n_eff)
    wells <- rbind(wells, data.frame(
      well_rank = rank,
      peak_index = pidx - 1L,                 # 0-based to match pandas iloc index
      k = scan_rows$k[pidx],
      n_eff = n_eff,
      delta_chi2 = scan_rows$delta_chi2[pidx],
      prominence = pk$prominences[oi],
      nearest_integer_n = as.numeric(nearest),
      distance_to_integer = abs(n_eff - nearest),
      distance_to_n10 = abs(n_eff - 10.0),
      distance_to_n15 = abs(n_eff - 15.0),
      distance_to_n20 = abs(n_eff - 20.0)
    ))
    rank <- rank + 1L
  }
  wells
}

#' Triplet construction and scoring from detected wells (triplets_from_wells()).
triplets_from_wells <- function(wells, max_wells = 12L, koide_q = KOIDE_Q) {
  if (nrow(wells) < 3L) return(data.frame())
  cand <- wells[seq_len(min(max_wells, nrow(wells))), , drop = FALSE]
  cand <- cand[order(cand$n_eff), , drop = FALSE]
  combs <- utils::combn(nrow(cand), 3L)
  rows <- vector("list", ncol(combs))
  ri <- 0L
  for (c in seq_len(ncol(combs))) {
    i <- combs[1, c]; j <- combs[2, c]; m <- combs[3, c]
    n1 <- cand$n_eff[i]; n2 <- cand$n_eff[j]; n3 <- cand$n_eff[m]
    if (n2 <= 0) next
    q_low <- n1 / n2
    q_high <- n3 / (2.0 * n2)
    q_mean <- 0.5 * (q_low + q_high)
    koide_error <- sqrt((q_low - koide_q)^2 + (q_high - koide_q)^2)
    integer_error <- sqrt((n1 - 10.0)^2 + (n2 - 15.0)^2 + (n3 - 20.0)^2)
    mean_delta <- (cand$delta_chi2[i] + cand$delta_chi2[j] + cand$delta_chi2[m]) / 3.0
    score <- mean_delta / (1.0 + 25.0 * koide_error + 0.25 * integer_error)
    ri <- ri + 1L
    rows[[ri]] <- data.frame(
      k1 = cand$k[i], k2 = cand$k[j], k3 = cand$k[m],
      n1 = n1, n2 = n2, n3 = n3,
      delta1 = cand$delta_chi2[i], delta2 = cand$delta_chi2[j], delta3 = cand$delta_chi2[m],
      Q_low = q_low, Q_high = q_high, Q_mean = q_mean,
      koide_error = koide_error, integer_error_10_15_20 = integer_error, score = score
    )
  }
  if (ri == 0L) return(data.frame())
  out <- do.call(rbind, rows[seq_len(ri)])
  out <- out[order(out$koide_error, out$integer_error_10_15_20, -out$score), , drop = FALSE]
  rownames(out) <- NULL
  out
}

#' Locked multi-mode comb DeltaChi2 (comb_fit_delta()).
comb_fit_delta <- function(ell, y, var, ns, k1 = K1_FIXED,
                           delta_ell = DELTA_ELL_ACTIVE) {
  ks <- vapply(ns, function(n) k_from_n(n, delta_ell), numeric(1))
  base <- wls_fit(ell, y, var, ks_extra = numeric(0), include_k1 = TRUE, k1 = k1)
  fit <- wls_fit(ell, y, var, ks_extra = ks, include_k1 = TRUE, k1 = k1)
  list(delta = base$chi2 - fit$chi2, ks = ks, fit = fit)
}

#' Empirical p-value: (1 + #{null >= value}) / (1 + N).
empirical_p <- function(value, null_values) {
  null_values <- as.numeric(null_values)
  (1 + sum(null_values >= value)) / (length(null_values) + 1)
}

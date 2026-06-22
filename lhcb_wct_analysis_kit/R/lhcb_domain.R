# =============================================================================
# lhcb_domain.R
#
# Shared common physics definitions for the R reproduction of the LHCb / WCT
# yield-side analysis pipeline.
#
# This is the single source of truth for:
#   * q2 domain and mass windows
#   * widened charmonium vetoes and the retained active intervals
#   * the active-domain log length Delta_ell_A
#   * the active-domain integer-winding map n <-> k
#
# These definitions mirror the Python scripts (config.py, 28_sideband.py,
# 12_wct_integer_winding_scan.py, ...). Where individual Python stages
# intentionally use a different convention (for example the histogram bin
# count), that difference is applied inside the corresponding stage script,
# NOT here.
# =============================================================================

# ---- Primary q2 domain -----------------------------------------------------

Q2_MIN <- 0.1
Q2_MAX <- 19.0

# ---- Main mass windows (yield-side signal selection) -----------------------

B0_M_MIN <- 5230.0
B0_M_MAX <- 5330.0
KST_M_MIN <- 795.9
KST_M_MAX <- 995.9

# ---- Widened charmonium vetoes (resonance-tail stress test) ----------------
# NOTE: 09d uses PSI2S upper edge 14.5; the active intervals below are the
# canonical retained intervals shared by the winding/comb/sideband stages.

JPSI_VETO <- c(8.0, 11.0)
PSI2S_VETO <- c(12.5, 14.5)

# ---- Retained active intervals (log-domain support) ------------------------

ACTIVE_INTERVALS <- list(
  c(0.1, 8.0),
  c(11.0, 12.5),
  c(14.5, 19.0)
)

#' Active-domain log length: sum_i log(hi_i / lo_i)
#'
#' For the canonical ACTIVE_INTERVALS this equals 4.780150335923678.
active_delta_ell <- function(intervals = ACTIVE_INTERVALS) {
  s <- 0.0
  for (iv in intervals) s <- s + log(iv[2] / iv[1])
  s
}

DELTA_ELL_ACTIVE <- active_delta_ell(ACTIVE_INTERVALS)

# ---- Active-domain integer-winding map -------------------------------------

#' Map a raw angular frequency k to an active-domain winding number n.
n_from_k <- function(k, delta_ell = DELTA_ELL_ACTIVE) {
  k * delta_ell / (2.0 * pi)
}

#' Map an active-domain winding number n to a raw angular frequency k.
k_from_n <- function(n, delta_ell = DELTA_ELL_ACTIVE) {
  2.0 * pi * n / delta_ell
}

#' Boolean mask: TRUE where q2 falls inside any retained active interval.
in_active_intervals <- function(q2, intervals = ACTIVE_INTERVALS) {
  q2 <- as.numeric(q2)
  mask <- rep(FALSE, length(q2))
  for (iv in intervals) {
    mask <- mask | (q2 >= iv[1] & q2 <= iv[2])
  }
  mask
}

#' Boolean mask: TRUE where q2 falls inside the widened charmonium vetoes.
in_veto_q2 <- function(q2, jpsi = JPSI_VETO, psi2s = PSI2S_VETO) {
  q2 <- as.numeric(q2)
  (q2 >= jpsi[1] & q2 <= jpsi[2]) | (q2 >= psi2s[1] & q2 <= psi2s[2])
}

#' Recompute active intervals given explicit veto windows.
#'
#' Used by the veto-window covariance stage, where each veto scheme changes the
#' retained intervals and therefore Delta_ell_A and the k<->n map.
active_intervals_from_vetoes <- function(jpsi, psi2s,
                                         q2_min = Q2_MIN, q2_max = Q2_MAX) {
  # The active intervals are [q2_min, jpsi_lo], [jpsi_hi, psi2s_lo],
  # [psi2s_hi, q2_max] (only those with positive width are kept).
  edges <- list(
    c(q2_min, jpsi[1]),
    c(jpsi[2], psi2s[1]),
    c(psi2s[2], q2_max)
  )
  Filter(function(iv) iv[2] > iv[1], edges)
}

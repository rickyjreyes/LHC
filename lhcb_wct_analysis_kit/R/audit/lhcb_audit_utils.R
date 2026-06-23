# =============================================================================
# lhcb_audit_utils.R
#
# Shared utilities for the LHCb / WCT statistical-audit suite.
#
# Pure helpers only: empirical p-values, null-standardised effect sizes, derived
# log-periodic quantities, committed-artifact readers, optional-package guards,
# deterministic RNG, and a colourblind-safe plotting theme/palette.
#
# This module imports nothing scientific from the canonical stages and writes
# nothing. It is sourced by every R/audit/* module and by the orchestrator.
# =============================================================================

suppressWarnings(suppressMessages({
  library(jsonlite)
}))

# ---- Path resolution --------------------------------------------------------

if (!exists(".lhcb_resolve_rdir", mode = "function")) {
  .lhcb_resolve_rdir <- function() {
    a <- commandArgs(FALSE)
    m <- grep("^--file=", a, value = TRUE)
    cands <- c(
      if (length(m)) dirname(dirname(normalizePath(sub("^--file=", "", m[1]), mustWork = FALSE))),
      if (length(m)) dirname(normalizePath(sub("^--file=", "", m[1]), mustWork = FALSE)),
      Sys.getenv("LHCB_R_DIR", ""),
      file.path(getwd(), "R"),
      getwd()
    )
    for (d in cands) if (nzchar(d) && file.exists(file.path(d, "lhcb_domain.R"))) return(d)
    cands[1]
  }
}

# Resolve the kit root (parent of R/) and the R/ directory.
AUDIT_RDIR <- .lhcb_resolve_rdir()
AUDIT_KIT  <- dirname(AUDIT_RDIR)

# Make the canonical domain constants available (Q2_MIN, vetoes, Delta_ell_A, ...)
if (!exists("DELTA_ELL_ACTIVE")) {
  source(file.path(AUDIT_RDIR, "lhcb_domain.R"))
}

# ---- Optional-package guards ------------------------------------------------

#' Is an optional package installed? (does not load it)
have_pkg <- function(pkg) requireNamespace(pkg, quietly = TRUE)

#' Quietly load a package, returning TRUE/FALSE without erroring.
soft_require <- function(pkg) {
  if (!have_pkg(pkg)) return(FALSE)
  suppressWarnings(suppressMessages(requireNamespace(pkg, quietly = TRUE)))
}

#' Report availability of the major optional capabilities.
audit_capabilities <- function() {
  list(
    arrow      = have_pkg("arrow"),
    reticulate = have_pkg("reticulate"),
    gt         = have_pkg("gt"),
    patchwork  = have_pkg("patchwork"),
    viridis    = have_pkg("viridis"),
    ragg       = have_pkg("ragg"),
    svglite    = have_pkg("svglite"),
    quarto     = nzchar(Sys.which("quarto")) || have_pkg("quarto"),
    ggridges   = have_pkg("ggridges"),
    ggrepel    = have_pkg("ggrepel")
  )
}

# ---- Deterministic RNG ------------------------------------------------------

#' Set the parallel-safe deterministic RNG used throughout the audit.
#' Using L'Ecuyer-CMRG makes results invariant to worker count.
audit_set_seed <- function(seed = 12345L) {
  RNGkind("L'Ecuyer-CMRG")
  set.seed(as.integer(seed))
  invisible(seed)
}

# ---- Empirical p-values (the (r+1)/(B+1) rule) ------------------------------

#' Empirical right-tail p-value with the +1 correction so it is never zero.
#'
#' @param t_obs observed statistic (larger = more extreme)
#' @param t_null numeric vector of null statistics
#' @return list(p, r, B, resolution) where resolution = 1/(B+1)
audit_empirical_p <- function(t_obs, t_null) {
  t_null <- t_null[is.finite(t_null)]
  B <- length(t_null)
  if (B == 0L) return(list(p = NA_real_, r = NA_integer_, B = 0L, resolution = NA_real_))
  r <- sum(t_null >= t_obs)
  list(p = (r + 1) / (B + 1), r = as.integer(r), B = as.integer(B),
       resolution = 1 / (B + 1))
}

#' Null-standardised effect size (descriptive standardised distance).
#' Labelled descriptive because the null is generally non-Gaussian.
z_null <- function(t_obs, t_null) {
  t_null <- t_null[is.finite(t_null)]
  s <- stats::sd(t_null)
  if (!is.finite(s) || s == 0) return(NA_real_)
  (t_obs - mean(t_null)) / s
}

#' Percentile of the observed statistic within the null distribution.
null_percentile <- function(t_obs, t_null) {
  t_null <- t_null[is.finite(t_null)]
  if (!length(t_null)) return(NA_real_)
  100 * mean(t_null <= t_obs)
}

# ---- Derived log-periodic quantities ----------------------------------------

#' Log-period of an angular log-frequency k:  Delta_ell = 2*pi/k
log_period <- function(k) 2 * pi / k

#' q2 multiplicative scale ratio per cycle:  rho_q2 = exp(2*pi/k)
rho_q2 <- function(k) exp(2 * pi / k)

#' Active-domain winding of k (uses canonical Delta_ell_A by default).
winding_n <- function(k, delta_ell = DELTA_ELL_ACTIVE) k * delta_ell / (2 * pi)

#' Conditional peak-to-trough rate ratio of an *isolated* log-rate component
#' with amplitude A:  exp(2*A). NOT the ratio of the full multi-mode model.
peak_to_trough_ratio <- function(A) exp(2 * A)

#' Radial amplitude from cosine/sine coefficients.
radial_amp <- function(a, b) sqrt(a^2 + b^2)

#' Phase from cosine/sine coefficients (atan2(b, a)).
phase_from_coef <- function(a, b) atan2(b, a)

# ---- Committed-artifact readers --------------------------------------------

#' Absolute path inside the kit, normalising committed Windows separators.
kit_path <- function(...) {
  p <- file.path(AUDIT_KIT, ...)
  gsub("\\\\", "/", p)
}

#' Read a committed JSON summary, returning NULL (with a warning) if absent.
read_json_safe <- function(path) {
  if (!file.exists(path)) {
    warning(sprintf("audit: missing JSON artifact: %s", path), call. = FALSE)
    return(NULL)
  }
  jsonlite::fromJSON(path, simplifyVector = TRUE, simplifyDataFrame = FALSE)
}

#' Read a committed CSV, returning NULL (with a warning) if absent.
read_csv_safe <- function(path, ...) {
  if (!file.exists(path)) {
    warning(sprintf("audit: missing CSV artifact: %s", path), call. = FALSE)
    return(NULL)
  }
  utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE, ...)
}

#' Does the kit have raw ROOT event data available?
have_event_data <- function(data_dir = kit_path("data"),
                            cache = kit_path("data_cache", "events.parquet")) {
  if (file.exists(cache)) return(TRUE)
  if (dir.exists(data_dir) && length(list.files(data_dir, pattern = "\\.root$"))) return(TRUE)
  FALSE
}

# ---- Robust numeric coercion / summaries -----------------------------------

#' Five-number-plus distribution summary used by every bootstrap table.
dist_summary <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) {
    return(list(n = 0L, mean = NA, median = NA, sd = NA, iqr = NA,
                q025 = NA, q975 = NA, min = NA, max = NA))
  }
  qs <- stats::quantile(x, c(0.025, 0.25, 0.75, 0.975), names = FALSE, type = 7)
  list(n = length(x), mean = mean(x), median = stats::median(x), sd = stats::sd(x),
       iqr = qs[3] - qs[2], q025 = qs[1], q975 = qs[4], min = min(x), max = max(x))
}

# ---- Output helpers ---------------------------------------------------------

#' Ensure a directory exists; returns the path.
ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
  path
}

#' Write a data.frame to CSV with full numeric precision (15 sig digits).
write_audit_csv <- function(df, path) {
  ensure_dir(dirname(path))
  # full numeric precision in the machine-readable artifact
  old <- options(digits = 15); on.exit(options(old))
  utils::write.csv(df, path, row.names = FALSE)
  invisible(path)
}

#' Write a list/object to pretty JSON.
write_audit_json <- function(obj, path, digits = NA) {
  ensure_dir(dirname(path))
  jsonlite::write_json(obj, path, auto_unbox = TRUE, pretty = TRUE,
                       digits = if (is.na(digits)) NA else digits, null = "null")
  invisible(path)
}

# ---- Git / provenance -------------------------------------------------------

#' Current git commit (short) or NA.
git_commit <- function() {
  out <- tryCatch(
    suppressWarnings(system2("git", c("-C", AUDIT_KIT, "rev-parse", "--short", "HEAD"),
                             stdout = TRUE, stderr = FALSE)),
    error = function(e) NA_character_)
  if (length(out) && nzchar(out[1])) out[1] else NA_character_
}

#' ISO-8601 UTC timestamp.
utc_now <- function() format(Sys.time(), tz = "UTC", "%Y-%m-%dT%H:%M:%SZ")

# ---- Plotting: colourblind-safe palette + theme -----------------------------

# Stable colour mapping for the concepts that must remain visually distinct.
AUDIT_PALETTE <- c(
  signal          = "#0072B2",  # blue
  low_sideband    = "#E69F00",  # orange
  high_sideband   = "#D55E00",  # vermillion
  charm_veto      = "#999999",  # grey
  reference_k     = "#009E73",  # green
  best_k          = "#CC79A7",  # pink/magenta
  integer_winding = "#56B4E9",  # sky blue
  comb_2over3     = "#000000",  # black
  comb_4over9     = "#F0E442",  # yellow
  parity          = "#0072B2",
  corrected       = "#D55E00",
  null            = "#BBBBBB",
  supported       = "#009E73",
  weakened        = "#E69F00",
  failed          = "#D55E00",
  inconclusive    = "#999999"
)

#' A clean publication theme. Only depends on ggplot2.
audit_theme <- function(base_size = 11) {
  if (!have_pkg("ggplot2")) return(NULL)
  ggplot2::theme_minimal(base_size = base_size) +
    ggplot2::theme(
      plot.title    = ggplot2::element_text(face = "bold", size = base_size + 2),
      plot.subtitle = ggplot2::element_text(size = base_size - 1, colour = "grey30"),
      plot.caption  = ggplot2::element_text(size = base_size - 3, colour = "grey40", hjust = 0),
      panel.grid.minor = ggplot2::element_blank(),
      legend.position = "bottom",
      strip.text = ggplot2::element_text(face = "bold")
    )
}

#' Standard caption footer carrying provenance + interpretation boundary.
audit_caption <- function(stage, mode = "parity", region = NA,
                          extra = NULL) {
  bits <- c(
    sprintf("stage %s", stage),
    sprintf("mode: %s", mode),
    if (!is.na(region)) sprintf("region: %s", region),
    sprintf("commit %s", git_commit()),
    "cross-language reproduction; smooth null is not a full SM amplitude model"
  )
  if (!is.null(extra)) bits <- c(extra, bits)
  paste(bits, collapse = " | ")
}

#' Save a ggplot as 300-dpi PNG plus a vector companion (SVG if svglite,
#' otherwise PDF). Degrades gracefully; returns the PNG path.
save_audit_figure <- function(plot, path_png, width = 9, height = 6, dpi = 300) {
  if (!have_pkg("ggplot2")) {
    warning("ggplot2 unavailable; skipping figure ", basename(path_png), call. = FALSE)
    return(NA_character_)
  }
  ensure_dir(dirname(path_png))
  dev_png <- if (have_pkg("ragg")) ragg::agg_png else NULL
  ok <- tryCatch({
    if (!is.null(dev_png)) {
      ggplot2::ggsave(path_png, plot, width = width, height = height, dpi = dpi,
                      device = dev_png)
    } else {
      ggplot2::ggsave(path_png, plot, width = width, height = height, dpi = dpi)
    }
    TRUE
  }, error = function(e) { warning(conditionMessage(e), call. = FALSE); FALSE })
  # vector companion
  vec <- sub("\\.png$", if (have_pkg("svglite")) ".svg" else ".pdf", path_png)
  tryCatch({
    if (have_pkg("svglite")) {
      ggplot2::ggsave(vec, plot, width = width, height = height, device = svglite::svglite)
    } else {
      ggplot2::ggsave(vec, plot, width = width, height = height, device = "pdf")
    }
  }, error = function(e) invisible(NULL))
  if (ok) path_png else NA_character_
}

# ---- Misc -------------------------------------------------------------------

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0 || (length(a) == 1 && is.na(a))) b else a

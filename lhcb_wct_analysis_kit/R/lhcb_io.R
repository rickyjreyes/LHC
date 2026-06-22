# =============================================================================
# lhcb_io.R
#
# Shared event-intake layer for the R reproduction.
#
# Supports --input-format root | parquet | csv. All physics selections, q2
# reconstruction, fits, scans and verdicts happen in R; this layer only moves
# selected branches into a data.frame and records provenance.
#
# ROOT intake uses reticulate + Python uproot strictly as an I/O adapter
# (option B in the task spec). The original Python *analysis* scripts are never
# called from R. When reticulate/uproot are unavailable, root intake fails with
# a clear message pointing at the parquet/csv export path (option C).
# =============================================================================

suppressWarnings(suppressMessages({
  has_jsonlite <- requireNamespace("jsonlite", quietly = TRUE)
  has_digest <- requireNamespace("digest", quietly = TRUE)
}))

#' SHA-256 of a file (empty string if digest unavailable or file missing).
file_sha256 <- function(path) {
  if (!has_digest || !file.exists(path)) return("")
  digest::digest(file = path, algo = "sha256")
}

#' Invariant mass squared from a four-vector.
inv_mass2 <- function(px, py, pz, e) e^2 - px^2 - py^2 - pz^2

#' Reconstruct q2 (GeV^2) from muon four-vectors, matching the Python convention.
reconstruct_q2 <- function(df,
                           plus = c("muplus_PX","muplus_PY","muplus_PZ","muplus_PE"),
                           minus = c("muminus_PX","muminus_PY","muminus_PZ","muminus_PE")) {
  miss <- setdiff(c(plus, minus), names(df))
  if (length(miss)) stop(sprintf("Cannot reconstruct q2; missing: %s", paste(miss, collapse=", ")))
  E  <- df[[plus[4]]] + df[[minus[4]]]
  px <- df[[plus[1]]] + df[[minus[1]]]
  py <- df[[plus[2]]] + df[[minus[2]]]
  pz <- df[[plus[3]]] + df[[minus[3]]]
  q2_mev2 <- E*E - px*px - py*py - pz*pz
  q2_mev2 / 1e6
}

#' Reconstruct K* mass from K+ and pi- four-vectors (fallback when no mass branch).
reconstruct_kst_mass <- function(df,
                                 kp = c("Kplus_PX","Kplus_PY","Kplus_PZ","Kplus_PE"),
                                 pim = c("piminus_PX","piminus_PY","piminus_PZ","piminus_PE")) {
  miss <- setdiff(c(kp, pim), names(df))
  if (length(miss)) stop(sprintf("Cannot reconstruct K* mass; missing: %s", paste(miss, collapse=", ")))
  px <- df[[kp[1]]] + df[[pim[1]]]; py <- df[[kp[2]]] + df[[pim[2]]]
  pz <- df[[kp[3]]] + df[[pim[3]]]; e  <- df[[kp[4]]] + df[[pim[4]]]
  sqrt(pmax(inv_mass2(px, py, pz, e), 0.0))
}

#' Discover ROOT files for the default LHCb patterns.
discover_root_files <- function(data_dir = "data", pattern = NULL) {
  if (!is.null(pattern)) return(sort(Sys.glob(pattern)))
  pats <- file.path(data_dir, c("*.dvntuple.root", "*.root"))
  files <- unique(sort(unlist(lapply(pats, Sys.glob))))
  files
}

#' Read selected branches from ROOT files via reticulate + uproot.
#'
#' Returns a list(df, provenance). Each provenance entry records the source
#' file, tree, branch mappings, q2 source, row counts and file SHA-256.
read_root_uproot <- function(files, branches, tree_preferred = "B0_KstMuMu/DecayTree") {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("ROOT intake needs reticulate + Python uproot. Install reticulate and ",
         "uproot, or pre-export to parquet/csv (see R/inspect_root.R --export).")
  }
  uproot <- tryCatch(reticulate::import("uproot"), error = function(e)
    stop("Python 'uproot' not importable via reticulate: ", conditionMessage(e)))
  np <- reticulate::import("numpy")

  dfs <- list(); prov <- list()
  for (path in files) {
    f <- uproot$open(path)
    keys <- reticulate::py_to_r(f$keys(recursive = TRUE))
    tree_name <- if (tree_preferred %in% keys) tree_preferred else {
      dk <- keys[grepl("DecayTree", keys)]
      if (length(dk)) dk[1] else keys[1]
    }
    tree <- f[[tree_name]]
    avail <- reticulate::py_to_r(tree$keys())
    use <- intersect(branches, avail)
    arr <- tree$arrays(use, library = "np")
    cols <- lapply(use, function(b) as.numeric(reticulate::py_to_r(arr[[b]])))
    names(cols) <- use
    sub <- as.data.frame(cols)
    sub$source_file <- basename(path)
    dfs[[length(dfs) + 1]] <- sub
    prov[[length(prov) + 1]] <- list(file = path, tree = tree_name,
      branches = use, file_sha256 = file_sha256(path), n_rows = nrow(sub))
    f$close()
  }
  list(df = do.call(rbind, dfs), provenance = prov)
}

#' Format-dispatching loader. Returns list(df, provenance, format).
load_events <- function(input_format = c("csv","parquet","root"),
                        path = NULL, files = NULL, branches = NULL,
                        data_dir = "data", tree_preferred = "B0_KstMuMu/DecayTree") {
  input_format <- match.arg(input_format)
  if (input_format == "csv") {
    stopifnot(!is.null(path))
    df <- utils::read.csv(path)
    list(df = df, provenance = list(list(file = path, format = "csv",
         file_sha256 = file_sha256(path), n_rows = nrow(df))), format = "csv")
  } else if (input_format == "parquet") {
    if (!requireNamespace("arrow", quietly = TRUE))
      stop("parquet intake needs the 'arrow' package.")
    stopifnot(!is.null(path))
    df <- as.data.frame(arrow::read_parquet(path))
    list(df = df, provenance = list(list(file = path, format = "parquet",
         file_sha256 = file_sha256(path), n_rows = nrow(df))), format = "parquet")
  } else {
    if (is.null(files)) files <- discover_root_files(data_dir)
    if (!length(files)) stop("No ROOT files found under ", data_dir)
    res <- read_root_uproot(files, branches, tree_preferred)
    list(df = res$df, provenance = res$provenance, format = "root")
  }
}

#' Build a reproducibility manifest recorded in every stage output directory.
build_manifest <- function(stage, config = list(), inputs = list(),
                           outputs = character(0), warnings = character(0),
                           seed = NA, extra = list()) {
  git_commit <- tryCatch(trimws(system2("git", c("rev-parse","HEAD"),
                          stdout = TRUE, stderr = FALSE)), error = function(e) NA_character_)
  pkgs <- c("base","jsonlite","digest","optparse","data.table","arrow")
  pkg_versions <- setNames(lapply(pkgs, function(p)
    tryCatch(as.character(utils::packageVersion(p)), error = function(e) NA_character_)), pkgs)
  list(
    stage = stage,
    timestamp_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    r_version = R.version.string,
    platform = R.version$platform,
    os = Sys.info()[["sysname"]],
    git_commit = if (length(git_commit)) git_commit[1] else NA_character_,
    seed = seed,
    rng_kind = paste(RNGkind(), collapse = ","),
    package_versions = pkg_versions,
    config = config,
    inputs = inputs,
    outputs = as.list(outputs),
    warnings = as.list(warnings),
    extra = extra
  )
}

#' Write a list to pretty JSON (auto_unbox so scalars are not arrays).
write_json <- function(obj, path) {
  jsonlite::write_json(obj, path, auto_unbox = TRUE, pretty = TRUE, digits = 16, null = "null")
}

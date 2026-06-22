# Source the R implementation for the test session.
RDIR <- Sys.getenv("LHCB_R_DIR",
  normalizePath(file.path(dirname(dirname(getwd())), "R"), mustWork = FALSE))
if (!dir.exists(RDIR)) {
  # fall back: tests run from kit root
  cand <- file.path(getwd(), "R")
  if (dir.exists(cand)) RDIR <- cand
}
for (f in c("lhcb_domain.R","lhcb_kde.R","lhcb_wls.R","lhcb_poisson.R","lhcb_io.R"))
  source(file.path(RDIR, f))

FIX <- file.path(RDIR, "fixtures")
KIT <- normalizePath(file.path(RDIR, ".."), mustWork = FALSE)

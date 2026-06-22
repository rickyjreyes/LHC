# Source the audit suite for testing. Resolves the kit root by walking up from
# the current working directory until R/lhcb_domain.R is found.
KITROOT <- local({
  d <- getwd()
  while (!file.exists(file.path(d, "R", "lhcb_domain.R")) && dirname(d) != d) d <- dirname(d)
  d
})
setwd(KITROOT)
RDIR <- file.path(KITROOT, "R")
source(file.path(RDIR, "lhcb_domain.R"))
source(file.path(RDIR, "audit", "lhcb_audit_utils.R"))
source(file.path(RDIR, "audit", "build_analysis_registry.R"))
source(file.path(RDIR, "audit", "build_event_flow.R"))
source(file.path(RDIR, "audit", "run_sideband_uncertainty.R"))
source(file.path(RDIR, "audit", "run_winding_sensitivity.R"))

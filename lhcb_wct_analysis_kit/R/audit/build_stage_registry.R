# =============================================================================
# build_stage_registry.R
#
# One row per canonical stage (not per test): records the stage-level
# conventions that must remain distinct (section 3 of the spec), whether the
# stage is replayable from committed artifacts, and what data it requires.
# =============================================================================

build_stage_registry <- function(kit = AUDIT_KIT) {
  s <- function(stage, title, needs_event, replayable, hist_coord, n_bins,
                kde, amp_bound, null_model, output_dir, summary_json, notes) {
    data.frame(stage = stage, title = title, needs_event_data = needs_event,
               replayable_from_csv = replayable, hist_coord = hist_coord,
               n_bins = n_bins, kde_convention = kde, amplitude_bound = amp_bound,
               null_model = null_model, output_dir = output_dir,
               summary_json = summary_json, notes = notes,
               artifact_present = file.exists(file.path(kit, output_dir, summary_json)),
               stringsAsFactors = FALSE)
  }
  rbind(
    s("09d", "Two-mode KDE bounded-Poisson scan", TRUE, "summary_only",
      "linear_q2", 60, "event-level KDE bw1.50 outside widened vetoes",
      "radial A1,A2<=0.10", "local scan-max poisson 5000",
      "outputs_logcos_poisson_twomode_kde_polar", "two_mode_summary.json",
      "43 retained bins; best k2=23.08; ref k2=19.5296 survives local null"),
    s("12", "Integer active-domain winding", TRUE, "summary_only",
      "linear_q2", 60, "event-level KDE bw {0.5..1.5}", "radial A2<=0.10",
      "integer-scan-max poisson 5000", "outputs_wct_integer_winding",
      "integer_winding_summary.json", "branch switches n=10 (low bw) -> n=20 (high bw)"),
    s("13", "Koide/trig comb model family", TRUE, "summary_only",
      "log_q2", 240, "hist centers repeated by counts", "coef a,b in [-0.1,0.1]",
      "model-scan-max poisson 5000", "outputs_wct_koide_comb",
      "koide_comb_summary.json", "Q=4/9 outscores Q=2/3 at bw1; radial-bound audit separate"),
    s("16", "Smooth-null vs locked combs", TRUE, "summary_only",
      "log_q2", 240, "hist centers repeated by counts", "coef bounds",
      "smooth empirical bootstrap H0 5000", "outputs_wct_vs_smqft",
      "wct_vs_smqft_summary.json", "null is smooth empirical, NOT full SM"),
    s("25", "Veto covariance / active-domain invariance", TRUE, "summary_only",
      "linear_q2", NA, "event-level KDE", "A_MAX=0.05",
      "scan poisson k[6,36] 1501", "outputs_wct_veto_covariance",
      "veto_covariance_summary.json", "recompute Delta_ell_A per scheme; preferred over 26"),
    s("28", "Sideband-subtracted WLS control", FALSE, "full",
      "log_q2", 240, NA, "none (WLS)", "gaussian k[6,32] 1301; 1000 null",
      "outputs_sideband_subtracted", "sideband_subtracted_summary.json",
      "reference mode / n15 / comb do NOT survive -> weakens signal specificity"),
    s("29", "Charm-trimmed signal + sidebands", FALSE, "full",
      "log_q2", 240, NA, "none (WLS)", "gaussian per region; 500 null",
      "outputs_charm_trimmed_control", "charm_trimmed_summary.json",
      "B sidebands show stronger structure than signal -> NOT signal-specific")
  )
}

write_stage_registry <- function(sr, table_dir) {
  write_audit_csv(sr, file.path(table_dir, "stage_registry.csv"))
  invisible(sr)
}

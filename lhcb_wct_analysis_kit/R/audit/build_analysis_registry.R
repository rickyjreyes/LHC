# =============================================================================
# build_analysis_registry.R
#
# Builds the complete analysis registry: one row per declared statistical test,
# with a unique analysis_id. The multiple-testing correction operates on this
# registry (section 7 / 20 of the spec), NOT on the strongest result only.
#
# The registry is derived from the committed *_summary.json artifacts so that it
# stays faithful to what the canonical stages actually searched. Each row also
# records whether it is parity or corrected_audit, confirmatory or exploratory.
# =============================================================================

if (!exists("audit_empirical_p", mode = "function")) {
  source(file.path(getwd(), "R", "audit", "lhcb_audit_utils.R"))
}

# Registry column schema (kept stable for joins + tests).
.REGISTRY_COLS <- c(
  "analysis_id", "stage", "script", "implementation", "mode",
  "data_source", "root_group", "tree", "q2_source", "B_mass_branch",
  "Kst_mass_branch", "q2_range", "B_mass_window", "Kst_mass_window",
  "sideband_windows", "active_intervals", "veto_scheme", "delta_ell_A",
  "hist_coord", "n_bins", "kde_training", "kde_bandwidth", "nuisance_terms",
  "k1_fixed", "k2_tested", "k_scan_range", "n_scan_points", "integer_n_range",
  "Q_values", "amplitude_bound", "null_model", "null_count", "bootstrap_count",
  "seed", "confirmatory", "output_path", "timestamp", "git_commit"
)

.reg_row <- function(...) {
  args <- list(...)
  row <- as.list(setNames(rep(NA, length(.REGISTRY_COLS)), .REGISTRY_COLS))
  for (nm in names(args)) row[[nm]] <- args[[nm]]
  row$timestamp <- row$timestamp %||% utc_now()
  row$git_commit <- row$git_commit %||% git_commit()
  as.data.frame(row, stringsAsFactors = FALSE)
}

#' Build the analysis registry as a data.frame.
#'
#' @param kit path to the analysis kit root.
build_analysis_registry <- function(kit = AUDIT_KIT) {
  rows <- list()
  add <- function(r) rows[[length(rows) + 1L]] <<- r

  ai <- "[[0.1,8.0],[11.0,12.5],[14.5,19.0]]"
  de <- DELTA_ELL_ACTIVE

  # ---- Stage 09d: two-mode KDE bounded-Poisson, k2 scan + fixed reference ----
  add(.reg_row(
    analysis_id = "S09D_LOCAL_SCAN", stage = "09d",
    script = "09d_two_mode_kde_baseline_polar_cupy.py / two_mode_kde_polar.R",
    implementation = "python+R", mode = "parity", data_source = "event",
    root_group = "00382466+00382467", tree = "B0_KstMuMu/DecayTree",
    q2_source = "muon_four_vectors", B_mass_branch = "B0_M",
    Kst_mass_branch = "Kst_892_0_M", q2_range = "[0.1,19.0]",
    B_mass_window = "[5230,5330]", Kst_mass_window = "[795.9,995.9]",
    sideband_windows = NA, active_intervals = ai, veto_scheme = "JPSI[8,11];PSI2S[12.5,14.5]",
    delta_ell_A = de, hist_coord = "linear_q2", n_bins = 60,
    kde_training = "event_level_outside_widened_vetoes", kde_bandwidth = 1.50,
    nuisance_terms = "k1_fixed", k1_fixed = 7.61054, k2_tested = "scan[18,24]",
    k_scan_range = "[18,24]", n_scan_points = 601, integer_n_range = NA,
    Q_values = NA, amplitude_bound = "radial_A1A2<=0.10",
    null_model = "local_scan_max_poisson", null_count = 5000, bootstrap_count = NA,
    seed = 12345, confirmatory = "exploratory",
    output_path = "outputs_logcos_poisson_twomode_kde_polar"))
  add(.reg_row(
    analysis_id = "S09D_REF_K2", stage = "09d", script = "09d / two_mode_kde_polar.R",
    implementation = "python+R", mode = "parity", data_source = "event",
    root_group = "00382466+00382467", tree = "B0_KstMuMu/DecayTree",
    q2_source = "muon_four_vectors", B_mass_branch = "B0_M",
    Kst_mass_branch = "Kst_892_0_M", q2_range = "[0.1,19.0]",
    B_mass_window = "[5230,5330]", Kst_mass_window = "[795.9,995.9]",
    sideband_windows = NA, active_intervals = ai, veto_scheme = "JPSI[8,11];PSI2S[12.5,14.5]",
    delta_ell_A = de, hist_coord = "linear_q2", n_bins = 60,
    kde_training = "event_level_outside_widened_vetoes", kde_bandwidth = 1.50,
    nuisance_terms = "k1_fixed", k1_fixed = 7.61054, k2_tested = "19.5296",
    k_scan_range = "fixed", n_scan_points = 1, integer_n_range = NA, Q_values = NA,
    amplitude_bound = "radial_A1A2<=0.10", null_model = "fixed_reference_poisson",
    null_count = 5000, bootstrap_count = NA, seed = 12345,
    confirmatory = "confirmatory",
    output_path = "outputs_logcos_poisson_twomode_kde_polar"))

  # ---- Stage 12: integer winding, n x bandwidth ------------------------------
  for (bw in c(0.5, 0.75, 1.0, 1.25, 1.5)) {
    add(.reg_row(
      analysis_id = sprintf("S12_INTWIND_BW%0.2f", bw), stage = "12",
      script = "12_wct_integer_winding_scan.py / integer_winding_scan.R",
      implementation = "python+R", mode = "parity", data_source = "event",
      root_group = "00382466+00382467", tree = "B0_KstMuMu/DecayTree",
      q2_source = "muon_four_vectors", B_mass_branch = "B0_M",
      Kst_mass_branch = "Kst_892_0_M", q2_range = "[0.1,19.0]",
      B_mass_window = "[5230,5330]", Kst_mass_window = "[795.9,995.9]",
      sideband_windows = NA, active_intervals = ai, veto_scheme = "JPSI[8,11];PSI2S[12.5,14.5]",
      delta_ell_A = de, hist_coord = "linear_q2", n_bins = 60,
      kde_training = "event_level", kde_bandwidth = bw, nuisance_terms = "k1_fixed",
      k1_fixed = 7.61054, k2_tested = "integer_n", k_scan_range = "n[10,22]",
      n_scan_points = 13, integer_n_range = "[10,22]", Q_values = NA,
      amplitude_bound = "radial_A2<=0.10", null_model = "integer_scan_max_poisson",
      null_count = 5000, bootstrap_count = NA, seed = 12345,
      confirmatory = "exploratory", output_path = "outputs_wct_integer_winding"))
  }

  # ---- Stage 13: Koide/comb model family x bandwidth (parity + corrected) ----
  for (md in c("parity", "corrected_audit")) {
    for (bw in c(0.5, 0.75, 1.0, 1.25, 1.5)) {
      add(.reg_row(
        analysis_id = sprintf("S13_COMB_%s_BW%0.2f",
                              ifelse(md == "parity", "PAR", "RAD"), bw),
        stage = "13", script = "13_wct_koide_trig_comb_scan_cupy.py / koide_comb_scan.R",
        implementation = "python+R", mode = md, data_source = "event",
        root_group = "00382466+00382467", tree = "B0_KstMuMu/DecayTree",
        q2_source = "muon_four_vectors", B_mass_branch = "B0_M",
        Kst_mass_branch = "Kst_892_0_M", q2_range = "[0.1,19.0]",
        B_mass_window = "[5230,5330]", Kst_mass_window = "[795.9,995.9]",
        sideband_windows = NA, active_intervals = ai, veto_scheme = "JPSI[8,11];PSI2S[12.5,14.5]",
        delta_ell_A = de, hist_coord = "log_q2", n_bins = 240,
        kde_training = "hist_centers_repeated_by_counts", kde_bandwidth = bw,
        nuisance_terms = "k1_fixed", k1_fixed = 7.61054, k2_tested = "comb_models",
        k_scan_range = "model_family", n_scan_points = NA, integer_n_range = NA,
        Q_values = "{2/3,4/9,...}",
        amplitude_bound = if (md == "parity") "coef_ab_in[-0.1,0.1]" else "radial<=0.1",
        null_model = "model_scan_max_poisson", null_count = 5000,
        bootstrap_count = NA, seed = 12345, confirmatory = "exploratory",
        output_path = if (md == "parity") "outputs_wct_koide_comb"
                      else "outputs_statistical_audit_r/comb_radial_corrected"))
    }
  }

  # ---- Stage 16: smooth-null vs locked combs --------------------------------
  for (ml in c("WCT_Koide_sideband_Q_2over3", "WCT_folded_Q_4over9",
               "WCT_combined_Q_2over3_plus_4over9")) {
    add(.reg_row(
      analysis_id = sprintf("S16_%s", gsub("[^A-Za-z0-9]", "", ml)), stage = "16",
      script = "16_wct_vs_smqft_likelihood_test_cupy.py / wct_vs_smooth_likelihood.R",
      implementation = "python+R", mode = "parity", data_source = "event",
      root_group = "00382466+00382467", tree = "B0_KstMuMu/DecayTree",
      q2_source = "muon_four_vectors", B_mass_branch = "B0_M",
      Kst_mass_branch = "Kst_892_0_M", q2_range = "[0.1,19.0]",
      B_mass_window = "[5230,5330]", Kst_mass_window = "[795.9,995.9]",
      sideband_windows = NA, active_intervals = ai, veto_scheme = "JPSI[8,11];PSI2S[12.5,14.5]",
      delta_ell_A = de, hist_coord = "log_q2", n_bins = 240,
      kde_training = "hist_centers_repeated_by_counts", kde_bandwidth = "0.5..1.5",
      nuisance_terms = "k1_fixed", k1_fixed = 7.61054, k2_tested = ml,
      k_scan_range = "locked", n_scan_points = 1, integer_n_range = NA,
      Q_values = ml, amplitude_bound = "coef_bounds", null_model = "smooth_empirical_bootstrap_H0",
      null_count = 5000, bootstrap_count = NA, seed = 12345,
      confirmatory = "exploratory", output_path = "outputs_wct_vs_smqft"))
  }

  # ---- Stage 25: veto covariance per scheme ---------------------------------
  for (sch in c("tight", "baseline_wide", "wider", "very_wide", "shift_low", "shift_high")) {
    add(.reg_row(
      analysis_id = sprintf("S25_VETO_%s", toupper(sch)), stage = "25",
      script = "25_veto_window_covariance_test.py / veto_window_covariance.R",
      implementation = "python+R", mode = "parity", data_source = "event",
      root_group = "00382466+00382467", tree = "B0_KstMuMu/DecayTree",
      q2_source = "muon_four_vectors", B_mass_branch = "B0_M",
      Kst_mass_branch = "Kst_892_0_M", q2_range = "[0.1,19.0]",
      B_mass_window = "[5230,5330]", Kst_mass_window = "[795.9,995.9]",
      sideband_windows = NA, active_intervals = "per_scheme", veto_scheme = sch,
      delta_ell_A = NA, hist_coord = "linear_q2", n_bins = NA,
      kde_training = "event_level", kde_bandwidth = NA, nuisance_terms = "k1_fixed",
      k1_fixed = 7.61054, k2_tested = "scan[6,36]", k_scan_range = "[6,36]",
      n_scan_points = 1501, integer_n_range = NA, Q_values = NA,
      amplitude_bound = "A_MAX=0.05", null_model = "scan_poisson", null_count = NA,
      bootstrap_count = NA, seed = 12345, confirmatory = "exploratory",
      output_path = "outputs_wct_veto_covariance"))
  }

  # ---- Stage 28: sideband-subtracted control (parity + corrected alpha) ------
  for (md in c("parity", "corrected_audit")) {
    add(.reg_row(
      analysis_id = sprintf("S28_SIDEBAND_%s", ifelse(md == "parity", "FIXEDALPHA", "REESTALPHA")),
      stage = "28", script = "28_sideband.py / sideband_subtracted.R",
      implementation = "python+R", mode = md, data_source = "committed_per_bin_csv",
      root_group = "00382466+00382467", tree = "B0_KstMuMu/DecayTree",
      q2_source = "muon_four_vectors", B_mass_branch = "B0_M",
      Kst_mass_branch = "Kst_892_0_M", q2_range = "[0.1,19.0]",
      B_mass_window = "[5230,5330]", Kst_mass_window = "[795.9,995.9]",
      sideband_windows = "low[5000,5180];high[5380,5600]", active_intervals = ai,
      veto_scheme = "JPSI[8,11];PSI2S[12.5,14.5]", delta_ell_A = de,
      hist_coord = "log_q2", n_bins = 240, kde_training = NA, kde_bandwidth = NA,
      nuisance_terms = "constant+k1_fixed", k1_fixed = 7.61054,
      k2_tested = "scan[6,32]", k_scan_range = "[6,32]", n_scan_points = 1301,
      integer_n_range = "[10,22]", Q_values = "{2/3,4/9}",
      amplitude_bound = "none_WLS",
      null_model = if (md == "parity") "gaussian_fixed_alpha" else "bootstrap_reest_alpha",
      null_count = 1000, bootstrap_count = if (md == "parity") NA else 2000,
      seed = 271828, confirmatory = "confirmatory",
      output_path = if (md == "parity") "outputs_sideband_subtracted"
                    else "outputs_statistical_audit_r/sideband_corrected"))
  }

  # ---- Stage 29: charm-trimmed regions (parity + corrected variance) ---------
  for (reg in c("signal_B_signal_Kst", "B_low_sideband_Kst_signal",
                "B_high_sideband_Kst_signal", "sideband_subtracted_charm_trimmed")) {
    for (md in c("parity", "corrected_audit")) {
      add(.reg_row(
        analysis_id = sprintf("S29_%s_%s", toupper(gsub("[^A-Za-z0-9]", "", reg)),
                              ifelse(md == "parity", "PAR", "CORR")),
        stage = "29", script = "29_charm_tail_trimmed_control.py / charm_trimmed_control.R",
        implementation = "python+R", mode = md, data_source = "committed_per_bin_csv",
        root_group = "00382466+00382467", tree = "B0_KstMuMu/DecayTree",
        q2_source = "muon_four_vectors", B_mass_branch = "B0_M",
        Kst_mass_branch = "Kst_892_0_M", q2_range = "[0.1,19.0]",
        B_mass_window = "[5230,5330]", Kst_mass_window = "[795.9,995.9]",
        sideband_windows = "low[5000,5180];high[5380,5600]", active_intervals = ai,
        veto_scheme = "charm_removed_JPSI[8,11];PSI2S[12.5,14.5]", delta_ell_A = de,
        hist_coord = "log_q2", n_bins = 240, kde_training = NA, kde_bandwidth = NA,
        nuisance_terms = "constant+k1_fixed", k1_fixed = 7.61054,
        k2_tested = "scan[6,32]", k_scan_range = "[6,32]", n_scan_points = 1301,
        integer_n_range = "[10,22]", Q_values = "{2/3,4/9}",
        amplitude_bound = "none_WLS",
        null_model = if (md == "parity") "gaussian_parity_variance" else "gaussian_corrected_variance",
        null_count = 500, bootstrap_count = if (md == "parity") NA else 2000,
        seed = 314159, confirmatory = "confirmatory",
        output_path = if (md == "parity") "outputs_charm_trimmed_control"
                      else "outputs_statistical_audit_r/charm_corrected"))
    }
  }

  reg <- do.call(rbind, rows)
  rownames(reg) <- NULL
  # Guardrail: analysis_id must be unique.
  if (anyDuplicated(reg$analysis_id)) {
    dup <- reg$analysis_id[duplicated(reg$analysis_id)]
    stop(sprintf("Duplicate analysis_id in registry: %s", paste(unique(dup), collapse = ", ")))
  }
  reg
}

#' Write the registry to CSV (full precision) and JSON.
write_analysis_registry <- function(reg, table_dir, out_root) {
  write_audit_csv(reg, file.path(table_dir, "analysis_registry.csv"))
  write_audit_json(reg, file.path(out_root, "analysis_registry.json"))
  invisible(reg)
}

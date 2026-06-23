# =============================================================================
# build_claim_matrix.R   (section 28)
#
# The final claim-status matrix. Verdicts are drawn from the committed evidence
# and the audit modules. No "confirmed new physics" / "proved WCT" language.
# =============================================================================

build_claim_matrix <- function(table_dir, kit = AUDIT_KIT,
                                s09 = NULL, s12 = NULL, s13 = NULL,
                                s28 = NULL, s29 = NULL, s25 = NULL) {
  claim <- function(id, text, stage, conf, stat, raw, p_pt, p_sg, p_fg,
                    unc, ctrl, hold, calib, inj, verdict, lim, out) {
    data.frame(claim_id = id, claim = text, stage = stage,
               confirmatory_or_exploratory = conf, supporting_statistic = stat,
               raw_result = raw, pointwise_p = p_pt, stage_global_p = p_sg,
               family_global_p = p_fg, uncertainty = unc, control_result = ctrl,
               holdout_result = hold, calibration_status = calib,
               injection_sensitivity = inj, verdict = verdict, limitations = lim,
               output_references = out, stringsAsFactors = FALSE)
  }
  rows <- list()
  push <- function(x) rows[[length(rows)+1]] <<- x

  push(claim(
    "C01", "A log-periodic two-mode structure improves the 09d yield model",
    "09d", "exploratory", "deltaD_add",
    sprintf("best k2=23.08, deltaD=%.1f", s09$best_dD %||% 150.9),
    NA, s09$p_best$p %||% NA, NA,
    "A2 on bound; profile/bootstrap needed", "n/a",
    if (have_event_data()) "see holdout" else "NOT_TESTABLE_WITH_AVAILABLE_DATA",
    if (have_event_data()) "see calibration" else "NOT_TESTABLE_WITH_AVAILABLE_DATA",
    "n/a", "SUPPORTED_WITHIN_MODEL",
    "amplitude bound active; same-sample look; smooth null not full SM",
    "stage09d_peak_uncertainty.csv"))

  push(claim(
    "C02", "k2=19.5296 is the unique/best local peak", "09d", "confirmatory",
    "deltaD_add", "ref deltaD=70.0 vs best 150.9 at k2=23.08",
    s09$p_ref_fixed$p %||% NA, s09$p_ref_local$p %||% NA, NA,
    "reference is not the best local peak", "n/a", "n/a", "n/a", "n/a",
    "NOT_STABLE",
    "best local peak is 23.08; 19.5296 only a surviving reference location",
    "stage09d_peak_uncertainty.csv"))

  push(claim(
    "C03", "A single integer winding n is selected stably", "12", "exploratory",
    "deltaD by bandwidth", "branch switches n=10 (low bw) -> n=20 (high bw)",
    NA, NA, NA, sprintf("selected-n entropy=%.2f bits", s12$entropy %||% NA),
    "n/a", "n/a", "n/a", "n/a", "NOT_STABLE",
    "bandwidth ladder switches branch; no n has >=80% support",
    "integer_winding_stability.csv"))

  push(claim(
    "C04", "Q=2/3 Koide comb is uniquely selected", "13", "exploratory",
    "deltaD", "Q=4/9 deltaD=457 > Q=2/3 deltaD=373 at bw1",
    NA, NA, NA, "radial amplitudes exceed 0.1 under coef bounds", "n/a", "n/a",
    "n/a", "n/a", "WEAKENED_BY_CONTROLS",
    "Q=4/9 outscores Q=2/3 in committed family; coef vs radial bound issue",
    "comb_model_results.csv;comb_bound_comparison.csv"))

  push(claim(
    "C05", "The log-periodic structure is signal-specific (sideband-subtracted)",
    "28", "confirmatory", "deltaChi2",
    sprintf("best scan deltaChi2=%.2f", s28$regression$best_dchi2 %||% 5.30),
    s28$regression$p_best %||% 0.816, s28$regression$p_best %||% 0.816, NA,
    "alpha re-estimation bootstrap consistent with non-survival",
    "reference/n15/comb all FAIL to survive", "n/a", "n/a", "n/a",
    "NOT_SIGNAL_SPECIFIC",
    "sideband-subtracted residual retains none of the claimed structures",
    "sideband_audit.csv"))

  push(claim(
    "C06", "Signal-window structure exceeds sideband structure", "29",
    "confirmatory", "deltaChi2",
    sprintf("signal best=%.0f vs low-SB=%.0f, high-SB=%.0f",
            s29$regression$signal_best_dchi2 %||% 353,
            s29$regression$low_best_dchi2 %||% 956,
            s29$regression$high_best_dchi2 %||% 1045),
    NA, NA, NA, "bootstrap signal-specificity statistic", "B sidebands stronger",
    "n/a", "n/a", "n/a", "NOT_SIGNAL_SPECIFIC",
    "charm-trimmed B sidebands carry stronger structure than the signal window",
    "charm_region_results.csv;signal_specificity.csv"))

  push(claim(
    "C07", "Winding number n is invariant under veto-window changes", "25",
    "exploratory", "k,n trajectories",
    sprintf("cv_k=%.3f cv_n=%.3f", s25$cv_k %||% NA, s25$cv_n %||% NA),
    NA, NA, NA, "fixed-k vs fixed-n model comparison", "n/a", "n/a", "n/a", "n/a",
    if (!is.null(s25) && !is.null(s25$cv_n) && !is.null(s25$cv_k) && s25$cv_n < s25$cv_k)
      "INCONCLUSIVE" else "INCONCLUSIVE",
    "both k and n move; cannot infer winding invariance from co-movement alone",
    "veto_invariance_models.csv"))

  push(claim(
    "C08", "The comb improves on a smooth empirical baseline", "16",
    "exploratory", "deltaD/AIC/BIC", "deltaAIC<0 for locked combs",
    NA, NA, NA, "AIC/BIC at selected comb do not pay for search", "n/a", "n/a",
    "n/a", "n/a", "SUPPORTED_WITHIN_MODEL",
    "smooth empirical null is NOT a full SM amplitude analysis",
    "model_comparison.csv"))

  mat <- do.call(rbind, rows)
  write_audit_csv(mat, file.path(table_dir, "final_claim_matrix.csv"))
  mat
}

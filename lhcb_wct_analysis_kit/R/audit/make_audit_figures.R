# =============================================================================
# make_audit_figures.R   (sections 25-26)
#
# Publication figures via ggplot2 (colourblind-safe palette, consistent
# mappings). Each figure carries title/subtitle/caption with stage + parity tag.
# Full-mode-only figures are rendered as honest "unavailable in replay" panels
# so the declared filenames always exist.
# =============================================================================

.fig_unavailable <- function(title, stage, reason = "requires event-level data (full mode)") {
  if (!have_pkg("ggplot2")) return(NULL)
  ggplot2::ggplot() +
    ggplot2::annotate("text", x = 0, y = 0,
                      label = paste0(strwrap(reason, 40), collapse = "\n"),
                      size = 4, colour = "grey30") +
    ggplot2::lims(x = c(-1, 1), y = c(-1, 1)) +
    ggplot2::labs(title = title, subtitle = paste("stage", stage, "- not run in replay mode"),
                  caption = audit_caption(stage)) +
    audit_theme() +
    ggplot2::theme(axis.text = ggplot2::element_blank(),
                   axis.title = ggplot2::element_blank())
}

make_audit_figures <- function(figure_dir, kit = AUDIT_KIT, ctx = list(),
                               mode = "replay") {
  if (!have_pkg("ggplot2")) {
    warning("ggplot2 unavailable; no figures produced"); return(character(0))
  }
  library(ggplot2)
  ensure_dir(figure_dir)
  made <- character(0)
  P <- AUDIT_PALETTE
  fp <- function(n) file.path(figure_dir, n)
  save1 <- function(p, n, w = 9, h = 6) {
    r <- save_audit_figure(p, fp(n), width = w, height = h)
    if (!is.na(r)) made <<- c(made, n); r
  }

  # fig01 event flow
  if (!is.null(ctx$flow)) {
    fl <- ctx$flow$flow
    fl2 <- fl[fl$kind %in% c("sequential", "diagnostic", "diagnostic_bins"), ]
    fl2$step <- factor(fl2$step, levels = rev(fl2$step))
    p <- ggplot(fl2, aes(x = count, y = step, fill = kind)) +
      geom_col() +
      scale_x_continuous(labels = scales::comma) +
      scale_fill_manual(values = c(sequential = P[["signal"]],
                                   diagnostic = P[["low_sideband"]],
                                   diagnostic_bins = P[["reference_k"]])) +
      labs(title = "Event / dataset accounting", subtitle = "sequential vs diagnostic counts (overlaps not mutually exclusive)",
           x = "count", y = NULL, fill = NULL, caption = audit_caption("flow")) +
      audit_theme()
    save1(p, "fig01_event_flow.png")
  }

  # fig02 q2 selection windows
  {
    wins <- data.frame(
      label = c("active", "active", "active", "JPSI veto", "PSI2S veto"),
      xmin = c(0.1, 11.0, 14.5, 8.0, 12.5),
      xmax = c(8.0, 12.5, 19.0, 11.0, 14.5),
      kind = c("active", "active", "active", "charm_veto", "charm_veto"))
    p <- ggplot(wins) +
      geom_rect(aes(xmin = xmin, xmax = xmax, ymin = 0, ymax = 1, fill = kind),
                alpha = 0.7, colour = "grey40") +
      geom_vline(xintercept = 19.5296, colour = P[["reference_k"]], linetype = 2) +
      annotate("text", x = 19.5296, y = 1.05, label = "q2 domain edge", size = 3) +
      scale_fill_manual(values = c(active = P[["signal"]], charm_veto = P[["charm_veto"]])) +
      labs(title = "q2 selection windows and charmonium vetoes",
           subtitle = "active intervals [0.1,8.0] [11.0,12.5] [14.5,19.0]; widened vetoes shaded grey",
           x = expression(q^2~"[GeV"^2*"]"), y = NULL, fill = NULL,
           caption = audit_caption("selection")) +
      audit_theme() + theme(axis.text.y = element_blank())
    save1(p, "fig02_q2_selection_windows.png", h = 3.5)
  }

  # fig03 09d scan
  if (!is.null(ctx$s09)) {
    sc <- ctx$s09$scan
    p <- ggplot(sc, aes(k2, deltaD_add_exact)) +
      geom_line(colour = "grey40") +
      geom_vline(xintercept = 19.5296, colour = P[["reference_k"]], linetype = 2) +
      geom_vline(xintercept = ctx$s09$best_k2, colour = P[["best_k"]], linetype = 1) +
      annotate("text", x = 19.5296, y = max(sc$deltaD_add_exact), label = "reference 19.53",
               colour = P[["reference_k"]], hjust = 1.05, size = 3) +
      annotate("text", x = ctx$s09$best_k2, y = max(sc$deltaD_add_exact),
               label = sprintf("best %.2f", ctx$s09$best_k2), colour = P[["best_k"]],
               hjust = -0.05, size = 3) +
      labs(title = "Stage 09d local k2 scan (two-mode)",
           subtitle = "best local peak is 23.08; 19.5296 is a surviving reference, not the best peak",
           x = expression(k[2]), y = expression(Delta*D[add]),
           caption = audit_caption("09d")) +
      audit_theme()
    save1(p, "fig03_stage09d_scan.png")

    # fig06 boundary activity
    sc$bound <- as.logical(sc$amplitude2_bound_active)
    p6 <- ggplot(sc, aes(k2, as.numeric(bound))) +
      geom_step(colour = P[["best_k"]]) +
      labs(title = "Stage 09d A2 boundary activity across the scan",
           subtitle = sprintf("A2 hits its 0.10 bound on %.0f%% of scan points",
                              100 * ctx$s09$boundary_frac),
           x = expression(k[2]), y = "A2 bound active (1/0)",
           caption = audit_caption("09d")) + audit_theme()
    save1(p6, "fig06_stage09d_boundary_activity.png", h = 3.5)
  }

  # fig04 / fig05 require event-level fit residuals / event bootstrap
  save1(.fig_unavailable("Stage 09d fit & residuals", "09d"), "fig04_stage09d_fit_residuals.png", h = 5)
  save1(.fig_unavailable("Stage 09d bootstrap k2", "09d",
        "event-level bootstrap of k2; full mode only"), "fig05_stage09d_bootstrap_k2.png", h = 5)

  # fig07 integer winding heatmap
  if (!is.null(ctx$s12)) {
    res <- ctx$s12$results
    p <- ggplot(res, aes(factor(n), factor(KDE_BANDWIDTH_SCALE), fill = deltaD)) +
      geom_tile(colour = "white") +
      scale_fill_viridis_safe() +
      labs(title = "Stage 12 integer-winding deltaD heatmap",
           subtitle = "branch switches n=10 (low bw) to n=20 (high bw): model sensitivity",
           x = "winding n", y = "KDE bandwidth scale", fill = expression(Delta*D),
           caption = audit_caption("12")) + audit_theme()
    save1(p, "fig07_integer_winding_heatmap.png")

    sel <- data.frame(n = as.integer(names(ctx$s12$selection)),
                      prob = as.numeric(ctx$s12$selection))
    p8 <- ggplot(sel, aes(factor(n), prob)) +
      geom_col(fill = P[["integer_winding"]]) +
      geom_hline(yintercept = 0.8, linetype = 2, colour = P[["failed"]]) +
      labs(title = "Stage 12 selection probability by winding n",
           subtitle = "no single n reaches 80% support across the bandwidth ladder",
           x = "winding n", y = "selection frequency", caption = audit_caption("12")) +
      audit_theme()
    save1(p8, "fig08_integer_selection_probability.png", h = 5)
  }

  # fig09 comb model comparison; fig10 Q profile; fig11 bound audit
  if (!is.null(ctx$s13)) {
    res <- ctx$s13$results
    bw1 <- res[res$KDE_BANDWIDTH_SCALE == 1.0, ]
    bw1 <- bw1[order(-bw1$deltaD), ]
    bw1$label <- factor(bw1$label, levels = rev(bw1$label))
    p <- ggplot(bw1, aes(deltaD, label, fill = radial_above_0p1)) +
      geom_col() +
      scale_fill_manual(values = c(`FALSE` = P[["parity"]], `TRUE` = P[["corrected"]]),
                        labels = c("radial<=0.1", "radial>0.1 (bound issue)")) +
      labs(title = "Stage 13 comb model comparison (bw scale 1)",
           subtitle = "Q=4/9 outscores Q=2/3; coloured bars have radial amplitude > 0.1 under coef bounds",
           x = expression(Delta*D), y = NULL, fill = NULL, caption = audit_caption("13")) +
      audit_theme()
    save1(p, "fig09_comb_model_comparison.png", h = 7)

    p10 <- ggplot(res, aes(Q, deltaD, colour = factor(KDE_BANDWIDTH_SCALE))) +
      geom_point() + geom_line(aes(group = KDE_BANDWIDTH_SCALE)) +
      geom_vline(xintercept = 2/3, linetype = 2, colour = P[["comb_2over3"]]) +
      geom_vline(xintercept = 4/9, linetype = 3, colour = P[["high_sideband"]]) +
      labs(title = "Stage 13 deltaD vs Q profile", subtitle = "Q=2/3 (dashed) and Q=4/9 (dotted)",
           x = "Q", y = expression(Delta*D), colour = "bw scale", caption = audit_caption("13")) +
      audit_theme()
    save1(p10, "fig10_comb_q_profile.png")

    cmp <- ctx$s13$comparison
    p11 <- ggplot(cmp, aes(radial_max, deltaD_parity, colour = radial_above_0p1)) +
      geom_point(size = 2) +
      geom_vline(xintercept = 0.1, linetype = 2) +
      scale_colour_manual(values = c(`FALSE` = P[["parity"]], `TRUE` = P[["corrected"]])) +
      labs(title = "Stage 13 amplitude-bound audit",
           subtitle = "points right of 0.1 violate a radial amplitude cap (parity uses coef bounds)",
           x = "max radial amplitude", y = expression(Delta*D~"(parity)"),
           colour = "radial>0.1", caption = audit_caption("13", "parity_vs_corrected")) +
      audit_theme()
    save1(p11, "fig11_comb_bound_audit.png")
  }

  # fig12 model comparison
  if (!is.null(ctx$model_cmp)) {
    cm <- ctx$model_cmp$comparison
    cm$model <- factor(cm$model, levels = rev(cm$model))
    p <- ggplot(cm, aes(deltaD_vs_null, model)) +
      geom_col(fill = P[["signal"]]) +
      labs(title = "Stage 16 smooth-null vs locked combs",
           subtitle = "deltaD over smooth empirical null (NOT a full SM amplitude analysis)",
           x = expression(Delta*D~"vs smooth null"), y = NULL, caption = audit_caption("16")) +
      audit_theme()
    save1(p, "fig12_model_comparison.png", h = 4)
  }

  # fig13/14/15 veto trajectories
  if (!is.null(ctx$s25)) {
    res <- ctx$s25$results
    p13 <- ggplot(res, aes(reorder(veto_scheme, delta_ell_A), k_best, group = 1)) +
      geom_line(colour = P[["best_k"]]) + geom_point(colour = P[["best_k"]]) +
      labs(title = "Stage 25 k trajectory across veto schemes",
           x = "veto scheme (by Delta_ell_A)", y = "best k (central leg)",
           caption = audit_caption("25")) + audit_theme() +
      theme(axis.text.x = element_text(angle = 30, hjust = 1))
    save1(p13, "fig13_veto_k_trajectories.png", h = 5)
    p14 <- ggplot(res, aes(reorder(veto_scheme, delta_ell_A), n_best, group = 1)) +
      geom_line(colour = P[["integer_winding"]]) + geom_point(colour = P[["integer_winding"]]) +
      labs(title = "Stage 25 n trajectory across veto schemes",
           x = "veto scheme (by Delta_ell_A)", y = "best n (central leg)",
           caption = audit_caption("25")) + audit_theme() +
      theme(axis.text.x = element_text(angle = 30, hjust = 1))
    save1(p14, "fig14_veto_n_trajectories.png", h = 5)
    mdl <- ctx$s25$models
    p15 <- ggplot(mdl[!is.na(mdl$aic), ], aes(hypothesis, aic, fill = hypothesis)) +
      geom_col() +
      scale_fill_manual(values = c(H_k_fixed_k = P[["best_k"]], H_n_fixed_n = P[["integer_winding"]])) +
      labs(title = "Stage 25 fixed-k vs fixed-n", subtitle = "lower AIC = better; co-movement alone is not winding invariance",
           x = NULL, y = "AIC (k-space residuals)", fill = NULL, caption = audit_caption("25")) +
      audit_theme()
    save1(p15, "fig15_fixed_k_vs_fixed_n.png", h = 5)
  }

  # fig16/17/18 sideband
  if (!is.null(ctx$s28)) {
    bins <- ctx$s28$bins
    bins$q2_center <- as.numeric(bins$q2_center)
    sp <- data.frame(q2 = bins$q2_center,
                     signal = as.numeric(bins$N_signal),
                     scaled_side = ctx$s28$alpha * as.numeric(bins$N_side_combined),
                     residual = ctx$s28$R0)
    spl <- tidyr::pivot_longer(sp, -q2, names_to = "series", values_to = "value")
    p16 <- ggplot(spl, aes(q2, value, colour = series)) +
      geom_line() +
      scale_colour_manual(values = c(signal = P[["signal"]],
                                     scaled_side = P[["low_sideband"]],
                                     residual = P[["high_sideband"]])) +
      labs(title = "Stage 28 spectrum: signal, scaled sideband, residual",
           subtitle = sprintf("alpha = %.5f", ctx$s28$alpha),
           x = expression(q^2), y = "counts", colour = NULL, caption = audit_caption("28")) +
      audit_theme()
    save1(p16, "fig16_sideband_spectrum.png")

    sc <- ctx$s28$scan
    if (!is.null(sc)) {
      p17 <- ggplot(sc, aes(k, delta_chi2)) + geom_line(colour = "grey40") +
        geom_vline(xintercept = 19.5296, colour = P[["reference_k"]], linetype = 2) +
        labs(title = "Stage 28 sideband-subtracted k-scan",
             subtitle = "no surviving structure: best deltaChi2 ~ 5.3, p ~ 0.82",
             x = "k", y = expression(Delta*chi^2), caption = audit_caption("28")) +
        audit_theme()
      save1(p17, "fig17_sideband_scan.png")
    }
    if (!is.null(ctx$s28$alpha_bootstrap)) {
      ab <- ctx$s28$alpha_bootstrap
      arow <- ab[ab$target == "alpha", ]
      p18 <- ggplot(ab[ab$target != "alpha", ],
                    aes(target, observed)) +
        geom_col(fill = P[["signal"]], width = 0.5) +
        geom_errorbar(aes(ymin = boot_q025, ymax = boot_q975), width = 0.2,
                      colour = P[["corrected"]]) +
        labs(title = "Stage 28 alpha-uncertainty propagation",
             subtitle = sprintf("alpha re-estimated per bootstrap (mean %.4f); bands = 95%% bootstrap",
                                arow$boot_mean %||% NA),
             x = NULL, y = expression(Delta*chi^2), caption = audit_caption("28", "corrected_audit")) +
        audit_theme()
      save1(p18, "fig18_sideband_alpha_uncertainty.png", h = 5)
    } else {
      save1(.fig_unavailable("Stage 28 alpha uncertainty", "28",
            "bootstrap-n=0; rerun with --bootstrap-n>0"), "fig18_sideband_alpha_uncertainty.png", h = 5)
    }
  }

  # fig19/20/21 charm
  if (!is.null(ctx$s29)) {
    reg <- ctx$s29$regions
    reg2 <- reg[reg$region %in% c("signal_B_signal_Kst", "B_low_sideband_Kst_signal",
                                  "B_high_sideband_Kst_signal"), ]
    reg2$region <- factor(reg2$region,
                          levels = c("signal_B_signal_Kst", "B_low_sideband_Kst_signal",
                                     "B_high_sideband_Kst_signal"))
    p19 <- ggplot(reg2, aes(region, best_delta_chi2, fill = region)) +
      geom_col() +
      scale_fill_manual(values = c(signal_B_signal_Kst = P[["signal"]],
                                   B_low_sideband_Kst_signal = P[["low_sideband"]],
                                   B_high_sideband_Kst_signal = P[["high_sideband"]])) +
      labs(title = "Stage 29 charm-trimmed best deltaChi2 by region",
           subtitle = "B sidebands carry STRONGER structure than the signal window",
           x = NULL, y = expression(best~Delta*chi^2), fill = NULL, caption = audit_caption("29")) +
      audit_theme() + theme(axis.text.x = element_text(angle = 20, hjust = 1))
    save1(p19, "fig19_charm_region_scans.png", h = 5)

    # fig20 signal vs sideband effects (raw + per-event)
    reg2$per_kevent <- reg2$best_delta_chi2_per_kevent
    pe <- tidyr::pivot_longer(reg2[, c("region", "best_delta_chi2", "per_kevent")],
                              -region, names_to = "metric", values_to = "value")
    p20 <- ggplot(pe, aes(region, value, fill = region)) +
      geom_col() + facet_wrap(~metric, scales = "free_y") +
      scale_fill_manual(values = c(signal_B_signal_Kst = P[["signal"]],
                                   B_low_sideband_Kst_signal = P[["low_sideband"]],
                                   B_high_sideband_Kst_signal = P[["high_sideband"]])) +
      labs(title = "Stage 29 signal vs sideband effect (raw and per-1000-events)",
           x = NULL, y = "deltaChi2", fill = NULL, caption = audit_caption("29")) +
      audit_theme() + theme(axis.text.x = element_blank())
    save1(p20, "fig20_signal_sideband_effects.png", h = 5)

    if (!is.null(ctx$s29$variance)) {
      vc <- ctx$s29$variance
      p21 <- ggplot(vc, aes(region, parity_delta_chi2_kref, fill = region)) +
        geom_col() +
        scale_fill_manual(values = c(signal_B_signal_Kst = P[["signal"]],
                                     B_low_sideband_Kst_signal = P[["low_sideband"]],
                                     B_high_sideband_Kst_signal = P[["high_sideband"]])) +
        labs(title = "Stage 29 variance audit (deltaChi2 at reference k)",
             subtitle = "parity vs corrected variance kept separate; sideband structure remains visible",
             x = NULL, y = expression(Delta*chi^2~"at k_ref"), fill = NULL,
             caption = audit_caption("29", "parity_vs_corrected")) +
        audit_theme() + theme(axis.text.x = element_text(angle = 20, hjust = 1))
      save1(p21, "fig21_charm_variance_audit.png", h = 5)
    }
  }

  # fig22/23 holdout / blocked: full-mode only
  save1(.fig_unavailable("File/run holdout validation", "holdout"), "fig22_file_holdout.png", h = 5)
  save1(.fig_unavailable("Blocked-q2 validation", "blocked"), "fig23_blocked_validation.png", h = 5)

  # fig24 calibration
  if (!is.null(ctx$calib) && !is.null(ctx$calib$calibration)) {
    cal <- ctx$calib$calibration
    cal2 <- cal[!is.na(cal$realised_rate), ]
    if (nrow(cal2)) {
      p24 <- ggplot(cal2, aes(nominal_alpha, realised_rate)) +
        geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "grey50") +
        geom_point(colour = P[["signal"]], size = 3) +
        geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.005, colour = P[["signal"]]) +
        labs(title = "Synthetic-null calibration (stage 28 WLS)",
             subtitle = "realised vs nominal false-positive rate; dashed = ideal",
             x = "nominal alpha", y = "realised rate", caption = audit_caption("28")) +
        audit_theme()
      save1(p24, "fig24_null_calibration.png", h = 5)
    } else save1(.fig_unavailable("Null calibration", "calib"), "fig24_null_calibration.png", h = 5)
  } else save1(.fig_unavailable("Null calibration", "calib"), "fig24_null_calibration.png", h = 5)

  # fig25/26/27 injection
  if (!is.null(ctx$inj) && nrow(ctx$inj[ctx$inj$n_trials > 0, ])) {
    ij <- ctx$inj[ctx$inj$n_trials > 0, ]
    ij_so <- ij[ij$design == "signal_only", ]
    p25 <- ggplot(ij_so, aes(inject_amp, detection_prob, colour = inject_k_label)) +
      geom_line() + geom_point() +
      labs(title = "Injection power (stage 28 WLS, signal-only)",
           subtitle = "detection probability vs injected amplitude",
           x = "injected amplitude", y = "detection probability", colour = "inject k",
           caption = audit_caption("28")) + audit_theme()
    save1(p25, "fig25_injection_power.png")
    p26 <- ggplot(ij_so, aes(inject_amp, recovered_k_bias, colour = inject_k_label)) +
      geom_line() + geom_point() + geom_hline(yintercept = 0, linetype = 2) +
      labs(title = "Frequency recovery bias (stage 28 WLS)",
           x = "injected amplitude", y = "recovered k - injected k", colour = "inject k",
           caption = audit_caption("28")) + audit_theme()
    save1(p26, "fig26_frequency_recovery.png")
    ij_d <- ij[ij$inject_k_label == "reference", ]
    p27 <- ggplot(ij_d, aes(inject_amp, detection_prob, colour = design)) +
      geom_line() + geom_point() +
      labs(title = "Signal vs background injection (stage 28)",
           subtitle = "does sideband subtraction remove common structure and keep signal-specific?",
           x = "injected amplitude", y = "detection probability", colour = "design",
           caption = audit_caption("28")) + audit_theme()
    save1(p27, "fig27_signal_background_injection.png")
  } else {
    save1(.fig_unavailable("Injection power", "28", "injection-n=0"), "fig25_injection_power.png")
    save1(.fig_unavailable("Frequency recovery", "28", "injection-n=0"), "fig26_frequency_recovery.png")
    save1(.fig_unavailable("Signal/background injection", "28", "injection-n=0"), "fig27_signal_background_injection.png")
  }

  # fig28 significance hierarchy
  if (!is.null(ctx$mt)) {
    sh <- ctx$mt$hierarchy
    shl <- tidyr::pivot_longer(sh[, c("analysis_id", "p_pointwise", "p_scan_max")],
                              -analysis_id, names_to = "tier", values_to = "p")
    shl <- shl[!is.na(shl$p), ]
    p28 <- ggplot(shl, aes(p, analysis_id, colour = tier)) +
      geom_point(size = 2) +
      geom_vline(xintercept = 0.05, linetype = 2, colour = P[["failed"]]) +
      scale_x_log10() +
      labs(title = "Significance hierarchy (pointwise vs stage-global)",
           subtitle = "controls (28/29) shown with equal prominence; dashed = 0.05",
           x = "empirical p (log)", y = NULL, colour = NULL, caption = audit_caption("registry")) +
      audit_theme()
    save1(p28, "fig28_significance_hierarchy.png", h = 6)
  }

  # fig29 python/r parity
  par_csv <- file.path(kit, "parity_report.csv")
  if (file.exists(par_csv)) {
    pr <- read_csv_safe(par_csv)
    save1(.fig_unavailable("Python/R parity", "parity",
          paste0("parity_report.csv present with ", nrow(pr), " rows; see python_r_parity.csv")),
          "fig29_python_r_parity.png", h = 5)
  } else save1(.fig_unavailable("Python/R parity", "parity"), "fig29_python_r_parity.png", h = 5)

  # fig30 evidence dashboard (faceted multipanel without patchwork)
  if (!is.null(ctx$s29) && !is.null(ctx$s28)) {
    dash <- data.frame(
      panel = c("28 sideband best", "28 ref-k", "29 signal", "29 low SB", "29 high SB"),
      deltaChi2 = c(ctx$s28$regression$best_dchi2 %||% NA,
                    ctx$s28$regression$ref_dchi2 %||% NA,
                    ctx$s29$regression$signal_best_dchi2 %||% NA,
                    ctx$s29$regression$low_best_dchi2 %||% NA,
                    ctx$s29$regression$high_best_dchi2 %||% NA),
      group = c("control_nonsurvival", "control_nonsurvival",
                "signal", "sideband", "sideband"))
    p30 <- ggplot(dash, aes(reorder(panel, deltaChi2), deltaChi2, fill = group)) +
      geom_col() + coord_flip() +
      scale_fill_manual(values = c(control_nonsurvival = P[["null"]],
                                   signal = P[["signal"]], sideband = P[["high_sideband"]])) +
      labs(title = "Evidence dashboard",
           subtitle = "controls and contradictory results shown with equal prominence",
           x = NULL, y = expression(Delta*chi^2), fill = NULL,
           caption = audit_caption("dashboard")) +
      audit_theme()
    save1(p30, "fig30_evidence_dashboard.png", h = 6)
  }

  made
}

# viridis if available, else a manual sequential blue scale
scale_fill_viridis_safe <- function(...) {
  if (have_pkg("viridis")) ggplot2::scale_fill_viridis_c(...)
  else ggplot2::scale_fill_gradient(low = "#deebf7", high = "#08306b", ...)
}

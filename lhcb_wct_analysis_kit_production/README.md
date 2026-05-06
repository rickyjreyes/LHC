# LHCb WCT Analysis Kit (LFT + FFT edition)

Target channel:
    B0 -> (K*(892)0 -> K+ pi-) mu+ mu-

Two parallel spectral tests run on the same selected events:

  - LFT (log-Fourier transform): bin uniformly in ell = ln(q^2 / Q2_REF)
    and FFT the residual. PRIMARY WCT TEST. The signal it looks for is:
        R(q^2) = A * cos(k_ell * ln(q^2 / Q2_REF) + phi)
    with k_ell ~ 12 (band: 8 <= k_ell <= 20).

  - FFT (linear-q^2): bin uniformly in q^2 and FFT the residual.
    DIAGNOSTIC / ARTIFACT TEST. A peak appearing in BOTH spectra
    indicates a binning, cut, or instrumental structure rather than
    log-periodic physics.

The cut sweep (05_cut_sweep.py) runs both tests across four selection
modes (raw_q2 / loose / medium / tight) so peak stability can be assessed.


## Install

    pip install -r requirements.txt


## Layout

    data/                    # main signal channel ROOT files
    data_signal/             # (optional) signal sample for control comparison
    data_control/            # (optional) B0 -> J/psi K* control sample
    outputs/                 # all results land here


## Run

    python run_all.py


## Pipeline order

    00_inspect_root.py            # dump tree / branch structure
    01_check_branches.py          # verify required branches present
    02_lft_wct_scan.py            # PRIMARY: log-Fourier on ell
    03_fft_control_scan.py        # ARTIFACT TEST: linear-q^2 FFT
    04_bootstrap_lft.py           # event resampling + null shuffle on LFT
    05_cut_sweep.py               # both tests across raw_q2/loose/medium/tight
    06_angle_branch_report.py     # P5 readiness diagnostic
    07_control_compare.py         # data_signal/ vs data_control/ (soft)
    08_p5_placeholder.py          # disabled until angles present (soft)


## Main outputs

LFT (primary):
    lft_residuals.csv             ell-binned counts, baseline, residual
    lft_power.csv                 k_ell, FFT amplitude
    lft_spectrum.png, lft_residual.png, lft_power.png
    lft_summary.json              peaks, cutflow, gates passed

FFT (artifact diagnostic):
    fft_residuals.csv             q^2-binned counts, baseline, residual
    fft_power.csv                 k_q2 (units 1/GeV^2), FFT amplitude
    fft_spectrum.png, fft_residual.png, fft_power.png
    fft_summary.json              peak, gates, interpretation note

Bootstrap:
    bootstrap_peaks.csv           legacy band-restricted argmax
    bootstrap_peaks_global.csv    honest global-search peaks
    null_bootstrap.csv            residual-shuffle null trials
    bootstrap_summary.json        all metrics + null FP rates

Cut sweep:
    cut_sweep.csv                 one row per mode
    cut_sweep.json                structured peak metrics + interpretation

Other:
    branch_report.json
    angle_branch_candidates.json
    control_compare.json          (when data_signal/ and data_control/ present)


## Detection criteria

A WCT-consistent observation requires ALL of:

  1. selected_events >= MIN_EVENTS_FOR_LFT (default 100) in tight mode.
     Below this, every result is labeled diagnostic_only and no claim
     can be made.

  2. LFT shows a significant peak in [WCT_K_MIN, WCT_K_MAX] passing
     the local-max + (amp - median) / robust_sigma >= SNR_MIN gate.

  3. FFT control scan does NOT show a co-occurring significant peak
     in the same selection mode. Co-occurrence in linear-q^2 FFT
     indicates the structure is not log-periodic.

  4. Cut sweep: LFT peak persists across at least two of the four
     selection modes (raw_q2/loose/medium/tight) and is NOT
     accompanied by FFT peaks in those same modes.

  5. Event-resampling bootstrap: a high fraction of bootstraps both
     pass the significance gate AND land in the WCT band. The
     bootstrap-vs-null ratio should be much larger than 1.

  6. Null residual-shuffle bootstrap: low false-positive rate at the
     same significance gate, calibrating what "high" means in (5).

  7. Control channel B0 -> J/psi K* does NOT have a matching peak
     within DELTA_K_MATCH of the signal peak. A matching control peak
     indicates detector / selection structure, not new physics.

P5 angular analysis remains disabled (P5_READY = False) until cosThetaL,
cosThetaK, phi are present as branches or reconstructed from four-vectors.


## Result labels (use these verbatim)

    diagnostic only                        N < 100, no physics claim
    no WCT-band detection                  LFT shows nothing
    possible binning/cut artifact          LFT peak + FFT co-occurs in same modes
    fragile artifact                       LFT peak in only one cut mode
    WCT-band candidate                     LFT peak across cuts, FFT clean,
                                           bootstrap > null, control clean
    evidence candidate                     all of the above + repeated on
                                           independent dataset


## Configuration knobs (config.py)

    Q2_REF              reference scale for ell = ln(q^2 / Q2_REF), default 1 GeV^2
    KST_MODE            "tight" (792-992) or "loose" (700-1100)
    VETO_MODE           "none" | "mask" | "segment_lft"
    BASELINE_MODE       "savgol" | "floor" | "kde"  (default "kde")
    SNR_MIN             default 3.0, gate on (amp_peak - median) / robust_sigma
    PROMINENCE_MIN      default 1.0, in units of robust_sigma
    MIN_EVENTS_FOR_LFT  default 100, below this results labeled diagnostic
    WCT_K_MIN, WCT_K_MAX  log-frequency target band, default [8, 20]
    WCT_K_TARGET        central WCT prediction, default 12.0 (informational)
    N_LFT_BINS          ell-bin count, default 60
    N_FFT_BINS          linear-q^2 bin count, default 60 (matches LFT)
    NULL_BOOTSTRAP_N    residual-shuffle trials for FP calibration
    BOOTSTRAP_N         event-resampling trials


## What the kit will NOT do

  - Reconstruct angular observables from four-vectors (P5 stays disabled).
  - Apply trigger / PID / vertex quality cuts (these depend on the
    specific dataset and must be added before this kit if needed).
  - Replace a publication-grade unbinned likelihood fit.

# Welcome to the LHCb / WCT Analysis Discussions

This forum is for technical discussion of the open-data `B0 → K*0 μ+μ−` analysis pipeline, including event reconstruction, logarithmic-domain residuals, active-domain winding, comb geometry, sidebands, charm controls, veto stability, null models, and physical interpretation.

The objective is to determine which results reproduce, which survive appropriate controls, and which disappear under alternate selections or multiplicity correction.

## Good contributions include

- exact pipeline reproductions;
- independent implementations;
- failed or incomplete reproductions;
- alternate smooth baselines and residual models;
- sideband, charm-tail, veto-covariance, blind-control, and holdout analyses;
- injected-signal recovery and false-positive benchmarks;
- cross-period, cross-channel, or cross-detector tests;
- preregistered confirmatory analyses;
- identification of reconstruction, selection, fitting, or trial-factor errors.

## Reporting standard

Identify the source release, input files and checksums, tree and branches, decay channel, cuts, veto windows, q² interval, active-domain definition, sidebands, and retained event counts. Include the repository commit, exact command, environment, seeds, CPU or GPU path, baseline, residual family, scan ranges, tested model variants, null-generation method, multiplicity correction, confidence intervals, controls, and machine-readable outputs.

Separate three claims:

1. a computational result reproduced;
2. a residual feature survived statistical and robustness controls;
3. the feature supports a particular physical interpretation.

These are not equivalent. A selected-spectrum residual does not by itself establish WCT, new particle physics, or a failure of the Standard Model.

Exploratory results are welcome when labeled honestly. Null results, unstable fits, failed controls, and negative replications are valuable and should remain public.

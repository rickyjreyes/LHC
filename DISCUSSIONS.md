# GitHub Discussions Guide

Use GitHub Discussions for questions, preregistered analysis proposals, exact reproductions, failed reproductions, independent implementations, null-model audits, control-channel results, and interpretation of the LHCb open-data analyses.

## Recommended categories

- **Announcements**: Frozen datasets, releases, canonical pipeline changes, preregistrations, and maintainer notices.
- **General**: Broad discussion about the analysis program and repository organization.
- **Q&A**: Focused questions tied to an exact stage, script, dataset, selection, model, or output.
- **Ideas**: Proposed analyses with a frozen selection, full multiplicity accounting, controls, and an advance decision rule.
- **Show and tell**: Reproductions, failed reproductions, independent implementations, robustness studies, blind controls, and confirmatory results.
- **Polls**: Community priorities only. Polls are not statistical evidence.

## Required evidence separation

Every substantive post should distinguish among:

1. **Data provenance and reconstruction**: source release, files, trees, branches, event cuts, vetoes, q² reconstruction, and retained counts.
2. **Statistical result**: baseline, residual family, scan range, test statistic, nuisance treatment, null generation, and uncertainty.
3. **Multiplicity and robustness**: all tested windows, cuts, modes, baselines, channels, stages, and controls.
4. **Physical interpretation**: what the result implies about the selected spectrum, and what additional evidence would be required for a WCT or beyond-Standard-Model conclusion.

A reproducible residual pattern in one selected candidate spectrum is not by itself evidence for a new physical law. Exploratory analyses should not be relabeled as confirmatory after the target result is known.

## Minimum standard for quantitative claims

Include the repository commit, source release and checksums, input files, tree and branch map, decay channel, cuts, veto windows, active-domain definition, retained event counts, exact command, environment, random seeds, CPU or GPU path, baseline, residual model, scan grid, all model and selection variants, null count, corrected and uncorrected significance, confidence intervals, sideband and charm controls, veto-covariance tests, blind controls, machine-readable outputs, and known failures.

Use **PASS**, **FAIL**, or **INCOMPLETE** only when the acceptance gate was specified before evaluating the target result. Preserve null results and unstable fits. Move bounded software tasks to issues while keeping statistical design and physical interpretation in Discussions.

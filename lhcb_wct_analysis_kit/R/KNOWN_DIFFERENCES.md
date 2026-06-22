# Known differences and stage-specific conventions

These are intentional, documented differences. None of them is a silent
"improvement"; each reproduces the corresponding Python stage exactly. Where a
correction exists it is a separate, explicitly named audit mode.

## Cross-language numerical

* **RNG**: numpy PCG64 vs R L'Ecuyer-CMRG — null draws differ; only summary
  statistics and verdicts are compared (see METHODOLOGY_PARITY.md).
* **Optimizer**: scipy `L-BFGS-B` vs R `optim(method="L-BFGS-B")`. Both wrap the
  same Fortran code; objective and predictions agree to <=1e-6 (rel). Raw polar
  coefficients can differ because phase-equivalent parameterizations exist, so
  parity compares fitted predictions / deviances, not raw `phi`.
* **Grid generation**: numpy `linspace` vs R `seq` agree to ~1e-14; the best
  grid **index** is compared exactly and the best **k** to float precision.

## Stage-specific baseline / binning conventions

| Stage | Histogram | Bins | KDE training | Veto upper (psi2S) |
|---|---|---|---|---|
| 09d | linear q2 | 60 | event-level q2 outside vetoes | 14.5 |
| 12  | linear q2 | 60 | event-level q2 (reuses 09d) | 14.5 |
| 13  | log q2 | 240 | hist centers repeated by counts | active intervals only |
| 16  | log q2 | 240 | hist centers repeated by counts | active intervals only |
| 25  | log q2 | 240 | hist centers repeated by counts | per-scheme |
| 28  | log q2 | 240 | n/a (WLS on residual) | active intervals only |
| 29  | log q2 | 240 | n/a (WLS on counts) | active intervals only |

These are not bugs; the Python stages genuinely use these different conventions
and they are preserved for parity.

## Stage 13: coefficient bound vs radial bound (CANONICAL)

Stage 13 constrains **each** sine and cosine coefficient independently to
`[-0.1, 0.1]`. This differs from the radial-amplitude cap `sqrt(a^2+b^2) <= 0.1`
used by 09d/12. The canonical R output preserves the coefficient-wise bound and
reports **both** `coefficient_bound_active` and `radial_amplitude_above_0p1`,
warning when a radial amplitude exceeds 0.1 even though every coefficient is in
bounds. `--amplitude-bound radial` is a **separate corrected audit mode** writing
to `outputs_wct_koide_comb_radial_audit_r/`; it never replaces parity outputs.

## Stage 29: sideband-subtracted variance quirk (PRESERVED)

`29_charm_tail_trimmed_control.py` passes the residual positionally as `counts`
into `analyze_spectrum`, which then sets `var = max(counts, 1) = max(residual,
1)`. This is **not** the statistically correct sideband variance
(`N_sig + alpha^2 N_side`, as stage 28 uses). The R port reproduces the quirk
verbatim (charm sideband best DeltaChi2 = 1100.847328) for parity, with an
explicit code comment. A corrected variant would be a separate audit mode.

## Stage naming

The committed `charm_trimmed_summary.json` reports `script:
"30_charm_trimmed_control.py"` while the actual repository file is
`29_charm_tail_trimmed_control.py`. The R reproduction targets the actual file
(`R/charm_trimmed_control.R`, stage key `29`) and its committed outputs.

## Data-gated stages

09d, 12, 13, 16 and 25 require the event-level ntuples (OAuth-gated, not
present). They are implemented and smoke-tested on synthetic q2 (pipeline
wiring, bin counts, bound detection, per-scheme Delta_ell_A) but their real-data
regression numbers cannot be executed/verified in this environment. Their fit
engines ARE verified against fixtures (see METHODOLOGY_PARITY.md).

## Plots

The Python stages emit PNGs via matplotlib. The R reproduction focuses on the
CSV/JSON numerical artifacts (the parity-relevant outputs). PNG generation is
optional and can be added with ggplot2 without affecting parity.

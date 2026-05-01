# WCT Falsification Run — Verdict

**Channel:** B0 → (K*(892)0 → K+ π−) μ+ μ−
**Test:** Log-Fourier transform of the q^2 spectrum residual, scanning for a
log-periodic mode R(ℓ) = A cos(k_ℓ ℓ + φ) with ℓ = ln(q^2/Q2_REF) and
k_ℓ in the WCT target band [8, 20] (central prediction k_ℓ ≈ 12).

**Data:** Open-data test ntuples `data/job0.root`, `data/job1.root`
(LHCb `B0_KstMuMu/DecayTree`, 1573 raw candidates total).

## Cutflow (tight selection)

| step              |    n |
|-------------------|-----:|
| raw               | 1573 |
| finite q^2        | 1573 |
| q^2 ∈ (0.1, 19)   | 1504 |
| B0_M ∈ (5100,5600)|   42 |
| Kst_M ∈ (792,992) |   27 |
| J/ψ veto          |    7 |
| ψ(2S) veto        |    5 |
| **final**         |  **5** |

The kit's `MIN_EVENTS_FOR_LFT` threshold is 100. The strict B0 → K*0 μ+ μ−
selection is therefore **diagnostic only** — no WCT claim either way is
possible from this test sample.

## Cut-sweep (raw_q2 / loose / medium / tight)

| mode    | n_sel | LFT global k_ℓ | LFT SNR | in WCT band? | linear-q^2 FFT peak |
|---------|------:|---------------:|--------:|:------------:|:-------------------:|
| raw_q2  |  1504 |          15.57 |    3.32 |     yes      |  none               |
| loose   |    42 |              — |       — |       —      |   —                 |
| medium  |    27 |              — |       — |       —      |   —                 |
| tight   |     5 |              — |       — |       —      |   —                 |

`raw_q2` has no B0 mass cut, no K*(892) cut, and no charmonium veto.
It is combinatorial-dominated and **not** the B0 → K*0 μ+ μ− physics
channel — but it is the only mode with enough stats to run an LFT in
this test sample.

## raw_q2 stability tests (200 trials each)

Event-resampling bootstrap (recompute LFT on each draw, take global peak):

| metric                                     | value          |
|--------------------------------------------|----------------|
| fraction passing significance gate         | 35.5%          |
| fraction in [8, 20] band overall           | 28.5%          |
| fraction in band given passed              | 80.3%          |
| k_peak mean ± std                          | 19.63 ± 6.70   |
| k_peak 5–95% range                         | [15.57, 33.53] |

Residual-shuffle null (same FFT pipeline on shuffled residuals):

| metric                                     | value |
|--------------------------------------------|------:|
| FP rate (significance)                     | 10.5% |
| FP rate (in band)                          |  3.5% |

The bootstrap-in-band rate (28.5%) is ~8× the null in-band FP rate
(3.5%), so the elevated high-k power is *not* purely a fluctuation of
the noise floor. **However**, the bootstrap k_ℓ position has σ ≈ 6.7
and spans 15.6–33.5 across resamples — there is no specific
log-frequency being reproducibly preferred. The nominal peak at
k_ℓ = 15.57 sits at the 5th percentile of the bootstrap distribution.

## Detection criteria (from README)

A WCT-consistent observation requires **all** of:

1. selected_events ≥ 100 in **tight** mode — **FAIL** (n = 5)
2. LFT peak in [8, 20] passing SNR_MIN ≥ 3 — only in raw_q2
3. linear-q^2 FFT clean in same mode — pass (none)
4. LFT peak persists across ≥ 2 cut modes — **FAIL** (only raw_q2)
5. Bootstrap stability much greater than null in-band rate — partial
   (rate ratio ≈ 8, but k position σ = 6.7 is not stable)
6. Null shuffle low in-band FP rate — pass (3.5%)
7. Control channel B0 → J/ψK* clean — not run (no control sample)

## Verdict (use README labels)

**`diagnostic only`** for the physics channel (n_tight = 5 ≪ 100), and
**`fragile artifact`** for the only populated mode (raw_q2 has a single
in-band peak with σ_k ≈ 6.7, the peak does not persist into stricter
selections, and stricter selections do not have enough events to
reproduce or refute it).

This run **does not confirm WCT** in B0 → K*0 μ+ μ−. It also **does
not falsify WCT**: the proper-selection sample (n = 5) is far too
small. The bottleneck is statistics, not method.

## What is needed for a real test

* The full LHCb open-data production for `B0_KstMuMu_Run2`
  (estimated ~34 GB; the current test sample is ~3 MB).
* On the full sample, run `python run_all.py` and `python
  09_bootstrap_raw_q2.py` (after re-pointing it at the proper-mode
  selection). Then check criteria 1–7 in README.md.
* If the full sample also lacks the angular branches `cosThetaL`,
  `cosThetaK`, `phi`, reconstruct them from four-vectors before
  running any P5′ test.

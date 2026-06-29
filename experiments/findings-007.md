# Findings 007 — Muon × induction-head emergence: earlier, zero-variance (third phenomenon), and no L² law for anyone

**Direction:** `directions/007-muon-induction-emergence.md` · **Novelty:** confirmatory
PROCEED 2026-06-10 night rescan (2511.01033 withdrawn from ICLR26 — its L² law is
unreviewed; 2511.16893 fixes batch×context with zero optimizer ablation; we fix B
throughout per that warning).
**Data:** `results/induction_emergence/` — 90 runs: {muon, adamw, sgdm} ×
L{64, 128, 256} × **10 seeds**, online copy/ICL stream, 427k-param causal
transformer, eval_every=100. **Plus** `results/induction_fine/` — 25 runs re-measuring
the floor-censored cells at eval_every=10 (muon × all L, adamw × L{64,128}, 5 seeds).
**Measurement discipline note:** in the main grid, 5/9 cells sat exactly at the
eval-interval floor (emergence "100 ± 0") — the same pitfall caught in 001 and 003.
All conclusions below use floor-resolved numbers (fine arm where applicable,
main grid where emergence ≫ floor).

## Resolved emergence table (steps; mean ± seed std)

| L | muon | adamw | sgdm |
|---|---|---|---|
| 64 | **30 ± 0** | 48 ± 4 | 290 ± 30 |
| 128 | **40 ± 0** | 90 ± 0 | 480 ± 75 |
| 256 | **70 ± 0** | 260 ± 49 | 480 ± 40 |

(All cells reach final ICL score ≈ 0.91–0.95; no failure confound.)

## P1 — Timing: **HOLDS at every L, and the gap widens with L**

Muon emerges 1.6× (L64), 2.3× (L128), 3.7× (L256) earlier than AdamW, and
4–10× earlier than SGDM. The optimizer-geometry acceleration extrapolates from
grokking (001/002) to an online ICL phenomenon with no memorization phase.

## P2 — Variance signature: **HOLDS — third phenomenon class**

Muon's emergence step is **seed-exact at every L: 15/15 fine-arm seeds land on
30/40/70 with zero spread** (and 30/30 main-grid seeds at the floor were already
spread-free). AdamW shows ±4 to ±49; SGDM ±30 to ±75. The zero-variance
signature now recurs across grokking (001, mod-add), S5 grokking (002, per-seed
exact), and online ICL emergence (007) — it is a property of the optimizer, not
of the grokking setup. (Caveat: "zero" is bounded by the 10-step eval grid.)

## P3 — L² exponent invariance: **both halves fail — no L², and no invariance**

Local (per-interval) scaling exponents log₂(T_{2L}/T_L):

| interval | muon | adamw | sgdm |
|---|---|---|---|
| 64→128 | 0.42 | 0.91 | 0.73 |
| 128→256 | 0.81 | 1.53 | 0.00 |

Nobody is near 2 anywhere in this regime; the exponent is **optimizer-dependent**
(muon shallowest, adamw steepest, sgdm saturating) and **L-dependent** (upward
curvature for muon/adamw). 2511.01033's L² claim does not describe this regime
under any optimizer, and the direction's "stronger positive" branch lands: the
context-length scaling of induction emergence is **not optimizer-invariant**.
Caveats: three L values give two intervals only; muon's small values (30/40/70)
are partially grid-quantized (10-step resolution), so its exponents carry ±~0.2
quantization slack.

## P4 — LR confound: **RULED OUT (completed 2026-06-11, `results/induction_p4/`)**

AdamW at 3× lr (3e-3) on L256: emergence **8770 / 6250 / failed** (one seed never
emerges, final ICL 0.007) versus 260 ± 49 at its standard lr — raising AdamW's
effective step makes emergence 24–34× *later* and unstable, the opposite of
Muon's 70 ± 0. Third reproduction of the 001-protocol result: Muon's
earlier/zero-variance emergence is not an effective-step-size artifact.

## Limitations

- Single architecture/width, single task stream family, B fixed (by design, per
  2511.16893's confound warning); L spans only 4×.
- Emergence threshold and slope/width metrics inherit the eval grid; the
  transition-width fields in the main grid are floor-degenerate and were not used.
- SGDM's flat 128→256 interval (480→480) may reflect its own ceiling effects;
  its emergence sits well below the step budget, but we did not probe L=512.
- P4 arm is 3 seeds at one elevated lr (3e-3); a downward lr arm was not run
  (001's pattern suggests lower lr only slows AdamW further).

## Figures

`results/figures-007/`: `fig_emergence_scaling.png`, `fig_variance_signature.png`,
`fig_sharpness.png`, `induction_verdicts.json` (main grid; fine-arm numbers in
`results/induction_fine/` summaries — table above is the merged authoritative set).

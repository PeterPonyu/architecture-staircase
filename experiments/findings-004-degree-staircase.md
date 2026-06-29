# Findings 004 — Does Muon erase the by-degree learning staircase? **No.**

**Direction:** `directions/004-muon-degree-staircase.md` · **Novelty record:** `.omc/research/novelty-004.md`
**Positioning:** head-on collision test between 2410.19637 (transformers learn
low-degree interactions first; Adam-family, natural language) and 2603.00742
(Spectral GD / idealized Muon provably removes sequential simplicity bias; deep
linear networks). 2606.08388 explicitly lists this as uncovered.
**Data:** `results/degree_staircase/` — 45 runs: {muon, adamw, sgdm} × {staircase,
mixed, pure3} × 5 seeds; GrokTransformer-skeleton with scalar readout (397,440
params), L=16 Boolean tokens, Walsh-character targets (nested S₁⊂S₂⊂S₃⊂S₄,
degrees 1–4), online fresh-batch training, 4,000 steps, eval every 50.
T_k = first step |corr_k| ≥ ½·its own final value; SI = geometric mean of adjacent
gap ratios T_{k+1}/T_k (1 = simultaneous; ≫1 = staircase); span = max T_k / min T_k.

## TL;DR — the law-boundary statement

**The by-degree staircase survives spectrum-flattened updates.** On softmax
transformers with controlled Fourier-degree targets, Muon's staircase index is
statistically identical to AdamW's (SI 3.26 ± 0.85 vs 3.28 ± 1.72) — the
deep-linear prediction that orthogonalized updates learn modes simultaneously
does **not** transfer from singular-value modes to function-space degree. What
Muon does deliver is a large, zero-variance **generic** speedup (pure degree-3:
T₃ = 100 ± 0 vs AdamW 660 ± 460, ≈6.6×) that rescales every step of the ladder
without reordering it. The sequential simplicity bias is a property of the
data/architecture, not of the optimizer's update spectrum.

## P1 — Staircase reproduction under Adam-family: **HOLDS**

AdamW on the nested-staircase task: degree-1 correlation saturates within ~50
steps, degree-2 at ~250–500, degree-3 ramps to ~1500–2500, degree-4 still
climbing at budget end. SI = 3.28 ± 1.72, span ratio ≈ 39×. SGDM: SI 3.74 ± 2.20.
The controlled-spectrum instrument reproduces the 2410.19637 phenomenon cleanly
(`fig_rho_curves.png`).

## P2 — Muon erases the staircase: **REFUTED (the headline negative)**

Muon SI = **3.26 ± 0.85** vs AdamW **3.28 ± 1.72** — indistinguishable means;
Muon's ρₖ(t) panel is qualitatively identical to AdamW's (same order, same
shape), only time-compressed. Span ratio 25.8 vs 39.3 (mild compression, driven
by the generic speedup of the slowest component, not by reordering). This is the
pre-registered negative — though stated honestly it is **underpowered** (n=5,
AdamW SD 1.72, per-seed SI range 1.0–5.6 ⇒ MDE ≈3 SI units; "no flattening
detected", no equivalence test yet): **2603.00742's simultaneous-learning law is
scoped to (deep) linear networks / singular-value complexity and shows no
detectable extension to Boolean-Fourier degree on softmax transformers.** The
staircase is data/architecture-driven (directionally; ≥15 seeds + a TOST
equivalence test to upgrade to a clean equivalence).

## P4 — Generic-speedup decomposition: **clean separation**

Pure degree-3 (no ladder available):
- **Muon: T₃ = 100 on all 5 seeds (zero variance), fit 1.00.**
- AdamW: T₃ = 300–1550 (mean 660), fit 1.00.
- ⇒ Muon's generic speedup ≈ 6.6× on the hard component, with 001's zero-variance
  signature recurring in a non-grokking setting.
Since SI is built from gap *ratios* (scale-free), a uniform speedup cancels —
and indeed SI stays at AdamW's value. Muon is "faster at everything, sequential
all the same."

## Bonus finding — the ladder is load-bearing for weak optimizers (Abbe staircase effect, observed)

**SGDM cannot learn pure degree-3 at all (fit ≈ 0.00, 5/5 seeds) — but on the
nested ladder it learns all four degrees (fit 0.73–0.88).** The low-degree
components bootstrap the high-degree ones, exactly as staircase theory
(2108.10573 / leap complexity) predicts; we observe it directly with per-degree
probes. Echoes 002's finding that bare SGD-momentum fails on hard structures:
Muon/AdamW can take the no-ladder route; SGDM needs the ladder.

## P5 — Variance signature: **partial support**

Muon halves the cross-seed SI spread (0.85 vs 1.72) and is zero-variance on
pure-3 T₃; but its T_k variances on the ladder are not uniformly smaller.
Severable, as pre-registered.

## P3 — LR confound: **null hardened (completed 2026-06-11)**

P3 was designed to defend a *positive* compression claim; with P2 refuted, the
symmetric control is whether Muon's lr was mistuned to *hide* its bias-removal.
Hardening arm (`results/degree_staircase_p3/`, muon_lr ∈ {0.01, 0.04} × 3 seeds):

| muon lr | 0.01 | 0.02 (main) | 0.04 |
|---|---|---|---|
| SI | 2.36 (2.24–2.47) | 3.26 ± 0.85 | 3.00 (1.41–4.58) |

**The staircase survives at every tested Muon lr** — SI stays in AdamW's band
(3.28) across a 4× lr range, never approaching 1 (simultaneous). The P2 null is
not an lr artifact. Instrument sensitivity is independently evidenced by P1
(staircase visible) and P4 (6.6× speedup visible).

## Limitations

- Degree-4 is only partially learned within the 4,000-step budget (final |corr|
  ≈ 0.2 for all optimizers); T₄ and hence the top gap ratio are
  threshold-relative on a low plateau. SI computed over defined degrees only.
- T_k uses each degree's **own final value** as reference (robust to partial
  saturation but not comparable to absolute-threshold definitions).
- Online fresh-batch training (scaffold's choice; the direction doc's full-batch
  primary arm was not what shipped) — finite-sample staircase artifacts are
  thereby excluded by construction, but the full-batch replication arm is unrun.
- "mixed" profile is an exact duplicate of "staircase" in the shipped scaffold
  (equal weights, same sets) — verified identical per-seed and used only as a
  determinism check; the reweighted-mixture condition was never actually run.
- Single geometry (L=16, D=4, nested sets), single width; degree>4 untested.


## Figures

`results/figures-004/`: `fig_rho_curves.png` (P1/P2), `fig_si_summary.png`
(SI/span by optimizer), `fig_pure3.png` (P4), `staircase_verdicts.json` (all
numbers incl. per-seed T_k).

## TOST equivalence (2026-06-15, n=15 — closes paper-C §3.1)

The P2 "Muon does not flatten the staircase" claim is upgraded from failed-to-reject to
a formal EQUIVALENCE. Densified the staircase-profile cell to n=15 per optimizer
(`staircase_seeds/run_staircase_seeds.py`) and ran a pre-registered Welch TOST
(`staircase_seeds/tost_si.py`, `results/figures-004/tost_si.json`) on the staircase
index SI, equivalence margin Δ=1.5 SI units (the competing simultaneous-learning /
spectral-flattening account 2603.00742 predicts SI≈1, ~2 units below the observed ≈3):

| pair | mean SI (n=15) | diff | 90% CI | margin | p_TOST | verdict |
|---|---|---|---|---|---|---|
| muon vs adamw | 3.53 vs 2.98 | +0.55 | [−0.36, +1.46] | ±1.5 | 0.044 | **EQUIVALENT** |
| muon vs sgdm | 3.53 vs 3.20 | +0.33 | [−0.65, +1.31] | ±1.5 | 0.026 | EQUIVALENT |
| adamw vs sgdm | 2.98 vs 3.20 | −0.22 | [−1.16, +0.72] | ±1.5 | 0.014 | EQUIVALENT |

The absolute-SI TOST (pre-registered primary) concludes equivalence for all pairs. The
stricter multiplicative test (log-SI, factor-1.5 margin) is inconclusive (muon ~24%
higher point estimate, CI up to factor ~1.7), so the claim is "equivalent on the
pre-registered absolute-SI margin", with the log-scale test reported as inconclusive —
not overclaimed as factor-level invariance.

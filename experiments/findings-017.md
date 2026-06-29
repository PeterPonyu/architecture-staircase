# Findings 017 — The degree staircase is owned by DEPTH and ATTENTION, not width or MLP

**Direction:** `directions/017-arch-staircase-anatomy.md` (bank A1 8.7, rank #1).
**Tests findings-004's OWN attribution:** 004 called the surviving degree
staircase "architecture-driven" without dissecting which component. This
anatomizes it on a controlled Boolean (Walsh) degree-staircase target across
width {64,128,256,512}, depth {1,2,4}, and attn/mlp freeze arms — AdamW fixed
throughout (the 004↔007 optimizer interlock), 43 runs (d64 column to 8 seeds
per the P4 statistics-discipline rule).
**Data:** `results/arch_staircase/` (43 runs) → `results/figures-017/`.

## Metric robustness (load-bearing, decided before reading verdicts)

The runner's `staircase_span_ratio` is **ill-defined** whenever a degree's
half-time is 0 (learned at the first eval): the min→0 collapses the ratio to its
max, which is exactly why span_ratio showed std>mean across seeds in the raw
grid. This analysis headlines two robust quantities instead: the **staircase
rank index** = mean over seeds of Spearman(degree, half-time) (+1 = clean
ascending staircase, 0 = degrees learned together, <0 = inverted), and the
**per-degree median half-time**. Freeze arms read within-arm only.

## Headline

The degree staircase has a specific architectural address: **depth ≥ 2 and the
attention sub-block.** A single layer produces no staircase (degrees learned
together, weak fit); two layers produce a clean one; width is irrelevant across
an 8× range; and freezing attention *inverts* the staircase while freezing the
MLP preserves it. This sharpens findings-004's "architecture-driven" claim from
a black box to "**depth-gated, attention-owned**".

## P1 — Depth rises, width flat: **CONFIRMED (depth as a threshold, not a monotone gain)**

Width axis (L2, AdamW), rank index across d_model:

| d_model | rank index | fit_corr | deg-4 median half-time | n |
|---|---|---|---|---|
| 64 | +0.53 ± 0.57 | 0.93 | 1225 | 8 |
| 128 | +0.80 ± 0.22 | 0.91 | 1050 | 5 |
| 256 | +1.00 ± 0.00 | 0.93 | 900 | 5 |
| 512 | +0.60 ± 0.46 | 0.86 | 1100 | 5 |

No monotone trend with width — the staircase is present and positive at every
width (rank ~0.5–1.0), fit ~0.9 throughout. **Width does not own the staircase**
(consistent with 2502.07553's width-sufficiency boundary: width buys fit
capacity, not the ordering).

Depth axis (d128, AdamW):

| n_layers | rank index | fit_corr | deg-4 median half-time |
|---|---|---|---|
| 1 | **+0.12 ± 0.47** | **0.71** | **50** (deg-3/4 corr ≈0 — *failed to learn*, floor not ordering) |
| 2 | +0.80 ± 0.22 | 0.91 | 1050 (late = clean staircase) |
| 4 | +0.56 ± 0.43 | 0.87 | 150 |

The L1→L2 jump is the signal: **a single layer fails to learn the high degrees**
(deg-3 corr 0.16, deg-4 corr 0.01; the deg-4 half-time of 50 is the failed-to-learn
floor, NOT "learned at the first eval"; fit weak 0.71), while two layers learn them
into a clean ascending staircase (rank +0.80, deg-4 corr 0.225 delayed to ~1050). The effect is a **depth threshold (≥2), not a monotone depth gain** —
L4's rank (+0.56) does not exceed L2's, so the honest statement is "staircase
requires depth ≥ 2", a partial confirmation that revises rather than rubber-
stamps the naive "more depth → more staircase" reading.

## P2 — Component ownership: **attention carries high-degree acquisition (freezing attention kills deg-4)**

Freeze arms at d128/L2 (within-arm; the only legal read per the doc):

| frozen | fit_corr | rank index | per-degree median half-times {1,2,3,4} |
|---|---|---|---|
| none | 0.91 | +0.80 | {50, 300, 1300, 1050} — clean staircase (deg-4 corr 0.225) |
| **attn** | 0.86 | **−0.32** | {100, 550, 1450, **0**} — deg-4 *fails to learn* (corr 0.005); half=0 is the failed-to-learn floor, not "immediate" |
| mlp | 0.98 | +0.60 | {50, 150, 500, 850} — high degree preserved (deg-4 corr 0.30); fit preserved (mean 0.92 ≈ none, 0.98 is the median) |

**Robustness-corrected reading (red-team audit 2026-06-13):** the rank index is
seed-noisy (std 0.2–0.5; none itself only 2/5 seeds strictly ascending), so the
"+0.80→−0.32 inversion" is reported as a noisy corroborator, NOT the primary
evidence. **SEED-DENSIFIED to n=15 (C-MAJOR-4, 2026-06-15) — the overall-fit gap WASHED OUT;
component-ownership survives via deg-4.** (i) **Overall fit_corr is NOT the
signal:** at n=15, attn-freeze fit 0.841±0.076 vs none 0.885±0.080 — gap +0.044,
Welch t=1.49, 95% bootstrap CI [−0.014, +0.102] (INCLUDES 0). The n=5 "fit 0.80 vs
0.92, p≈0.027" was a small-sample fluke (exactly the red-team's prediction);
**the overall-fit framing is DROPPED.** (ii) **The robust signal is high-degree
acquisition:** the attn-frozen model fails deg-4 — deg-4 final corr **+0.035** vs
none **+0.214** / mlp **+0.309**; none-vs-attn gap +0.180, Welch t=2.98, 95% CI
[+0.061, +0.292] (EXCLUDES 0), LOO |t| 2.67–4.16 (all >2); **14/15 attn seeds fail
deg-4 (corr<0.1)** vs 6/15 none, 5/15 mlp. The deg-4 half-time reads 0 because
deg-4 is never learned (failed-to-learn artifact, NOT "no longer waits its turn");
MLP-freeze preserves/enhances deg-4 (0.309). So **attention — not the MLP — carries
acquisition of the highest degree** (a component-necessity statement on deg-4
specifically, ROBUST at n=15; the overall-fit gap and the rank-index inversion are
washed-out / noisy and demoted to corroborators only). This is the
component-ownership statement 004 left open, at its true (deg-4-specific) strength.

## Tier-2 (C-KILL-1, 2026-06-15) — optimizer axis: staircase geometry is optimizer-invariant, but component-ownership is AdamW-specific

**Why:** 017 (like 004) fixed AdamW per the 004↔007 interlock; the red-team
(C-KILL-1) noted the "geometry ⊥ timing" dissociation was therefore measured under
one optimizer. This arm re-runs the depth + freeze grid under **Muon and SGDM**
(`results/arch_staircase_optaxis/`, d128, 5 seeds/cell; muon_lr=0.01 validated by a
matched-fit mini-sweep), analyzed by `analyze_optaxis.py` per the red-team-corrected
spec (scale-free **degree-acquisition ORDER** rank + **pointwise deg_corr at matched
fit**, never absolute spread; freeze **deltas**; SGDM **fit-gate ≥0.85**).

**(1) Staircase ORDER + difficulty geometry: OPTIMIZER-INVARIANT — geometry ⊥ timing
CONFIRMED, now a 2-leg result (AdamW + Muon).**

| cell | adamw rank | muon rank | sgdm rank |
|---|---|---|---|
| d128 L1 (control) | +0.12 | +0.16 | −0.24 (fit<0.85, excluded) |
| d128 L2 (staircase) | +0.67 | +0.52 | +0.32 |
| d128 L4 | +0.56 | +0.96 | +0.72 |

The depth-threshold itself is optimizer-invariant (L1 flat, L2/L4 positive, all
three optimizers). **Matched-fit pointwise (the clean dissociation):** at AdamW's
median final fit (0.90) the per-degree corr vector is essentially identical across
optimizers — deg1–3 ≈ 0.55 for all, deg-4 lags for all (adamw +0.13 / muon +0.21 /
sgdm +0.02) — **while the step to reach that fit differs ~4.5×** (muon 850 vs adamw
3850 vs sgdm 4000). Same geometry, different time.

**(2) Component ownership (P2, "attention owns deg-4"): AdamW-SPECIFIC — does NOT
survive the optimizer axis.** Freeze deltas on deg-4 final corr (attn-minus-none /
mlp-minus-none, the only optimizer-comparable quantity since "Muon" acts on a
different param set per freeze arm):
- **AdamW:** Δattn = **−0.17** (attn-freeze kills deg-4: 0.17→−0.00), Δmlp = **+0.29**
  (mlp-freeze preserves/raises) — the n=15 P2 result above.
- **Muon:** Δattn = **+0.005**, Δmlp = **−0.013** — Muon reaches deg-4 ≈ 0.50 and fit
  1.00 in *every* freeze arm (none/attn/mlp); freezing attention does NOT block deg-4.
- **SGDM:** deg-4 floored ≈ 0 in all arms (uninformative — SGDM fits degrees 1–3 only).

Reading: under AdamW attention is *necessary* for deg-4; under Muon's orthogonalized
updates the MLP-only (or attn-only) sub-network is already sufficient, so the
attention bottleneck dissolves. **P2's "attention owns high-degree acquisition" is a
statement about the AdamW optimization regime, not an optimizer-invariant
architectural fact.**

**Net:** C-KILL-1 (017 leg) is PARTIALLY closed — the staircase order/difficulty
geometry ⊥ timing is upgraded 1-leg → 2-leg (AdamW + Muon; SGDM corroborates the
order where it fits); the component-ownership sub-claim must be scoped to AdamW.
Verdict: `results/figures-017/optaxis_verdict.json`.

## Limitations

- Seed noise is real: rank-index std 0.22–0.57; half-times are quantized to the
  50-step eval grid. The width-flatness and the L1-vs-L2 / attn-vs-mlp contrasts
  survive the noise; the L2-vs-L4 ordering does not (reported as a threshold).
- `staircase_span_ratio` abandoned as ill-defined (see metric note); a re-run
  logging a 0-safe span (or finer eval grid) would tighten the rank estimates.
- One target family (Walsh degree staircase), one optimizer (AdamW, by the
  004↔007 interlock), one task scale. n_heads sub-arm (P3) not in this 43-run grid.
- Freeze = `requires_grad=False` on attn{qkv,proj} / mlp{fc1,fc2} + optimizer
  filter; partial-freeze / lr-adapted arms (the doc's contingency if a frozen
  arm simply fails to train) were not needed — both frozen arms still fit (≥0.86).

## Figures / data

`results/figures-017/`: `fig_arch.png` (P1 width | P1 depth | P2 freeze),
`arch_verdicts.json` (per-cell robust metrics + the three axes).

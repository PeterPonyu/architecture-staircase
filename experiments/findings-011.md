# Findings 011 — On a learnable non-Fourier emergence task (induction/copy), data-presentation ORDER does not compress the emergence delay

**Direction:** `directions/011-ordering-emergence-delay.md` (bank K4 7.95;
"can a data curriculum / presentation order replace the TARGET staircase" — the
data-side counterpart of findings-004's reward discovery).
**Status: adjudicated on the COPY (induction) family — the suite's one cleanly
learnable non-Fourier task. The Walsh and mod-add families did not yield an
adjudicable readout (testbed boundaries, documented below).**
**Data:** `results/curriculum_order/` — 120 runs: {walsh, modadd, copy} ×
{iid, easy_to_hard, hard_first, structured} × {adamw, sgdm} × 5 seeds, GrokTransformer
/ SeqTransformer, eval_every=50.

## Headline

On the induction/copy family — the only task in the suite that is learnable AND
non-Fourier AND shows a measurable emergence delay — **the data-presentation
ordering does not compress the emergence delay**. Across all four ordering
policies the induction circuit emerges at 1750–1950 steps, and the iid
(no-curriculum) baseline is the *fastest* (mean 1780) while the easy→hard
curriculum is mildly but **borderline-significantly slower** (mean 1900; Welch
t≈2.25, p≈0.05, n=5) — so the honest statement is *no ordering helps and easy→hard
mildly hurts*, NOT a flat zero-effect null. The emergence delay is set by the task,
not improved by the order in which examples are shown: ordering does not substitute
for target structure.

## The copy/induction readout (the adjudicable arm)

Emergence step (median of 5 seeds) by ordering × optimizer:

| ordering | AdamW | SGDM |
|---|---|---|
| iid (baseline) | **1750** | 1850 |
| easy_to_hard | 1950 | 1850 |
| hard_first | 1850 | 1850 |
| structured | 1850 | 1850 |

All within ~200 steps (≈ 4 eval intervals). The ordering effect is
null-to-mildly-**adverse**: no policy beats iid, and easy→hard is significantly
(borderline, p≈0.05 at n=5) *worse* than iid — so this is "no ordering accelerates
emergence; the one structured curriculum mildly delays it", a power-bounded
near-null (n=5, MDE ≈60–90 steps), not a proven zero effect. Per the preregistered
framing this is still the clean-negative direction — on a learnable emergence task,
**presentation order is not a positive lever on the emergence delay**; "curriculum
can replace the target staircase" is refuted for this task family. (≥10 seeds would
firm the null vs the mild-adverse effect.)

## Why Walsh and mod-add did not adjudicate (testbed boundaries)

- **Walsh (the originally-intended PRIMARY family): unlearnable in this
  pipeline.** The eval target was a *pure degree-3 Walsh character* (a single
  3-parity), which is uncorrelated with the curriculum's lower-degree ramp, so
  the scaffold cannot transfer and direct fit stays ≈0 (fit_corr 0.04; confirmed
  by 011's own code note "SGDM pure deg-3 fit 0.00 vs staircase 0.73-0.88"). A
  redesign to the learnable degree-STAIRCASE target (the 004/017-proven target,
  017 fit 0.91 trained directly) was implemented (v1 staircase-terminal, v2
  modadd-pattern direct-staircase) but **still reached only fit 0.13** in this
  pipeline — a deeper 011-vs-017 data-router/batch difference remains unlocated.
  The Walsh family is therefore set aside as a testbed boundary; the staircase
  curriculum question is best pursued in 004/017's proven pipeline, not here.
- **mod-add: floor-censored.** Emergence at step 50 (the first/second eval) for
  every ordering — the task is too easy at this config for an ordering effect to
  register (the 003 lesson: the primary-axis regime must make the target
  measurable). A harder mod-add (larger p / smaller train_frac) or eval_every=10
  would be needed to de-floor.

## Reconciliation with the main axis

The copy-family negative is consistent with 004's finding that the degree
staircase is *architecture/target-driven* (017: depth + attention own it): if the
staircase lives in the target/architecture, then merely reordering the data
should not move the emergence timing — which is what the copy arm shows. Ordering
is not a substitute for target structure.

## Limitations

- The adjudicable conclusion rests on ONE learnable family (copy/induction); the
  intended primary (Walsh) and the mod-add control did not yield clean readouts
  (boundaries documented, not hidden).
- eval_every=50 bounds the emergence-step resolution to ±50; the ~200-step
  spread across orderings is within a few eval intervals.
- Two optimizers (AdamW, SGDM); no Muon arm (out of scope per the single-optimizer
  K4 framing).

## Figures / data

`results/curriculum_order/` (120 runs) + `_pilot8k/_pilot16k/` (Walsh budget
refutation) + `_scpilot/_scpilot2/` (Walsh staircase-redesign attempts, fit 0.13).
`results/figures-011/fig_curriculum.png` — copy emergence step by ordering ×
optimizer (the null effect: all bars 1750–1950, iid baseline fastest).

# Findings 010 — The induction emergence IS ~3-dimensional, but NOT along the theory's three directions

**Direction:** `directions/010-induction-subspace.md` · **Discipline:** AdamW fixed
throughout (optimizer axis belongs to 007 per the F2↔K3 interlock).
**Data:** `results/induction_subspace/` — 45 runs: L{64, 96, 128, 192, 256} × 9 seeds,
repeat/ICL stream, online fresh-batch, AdamW lr 1e-3, batch 64. Probes: projection of
emergence-window cumulative Δθ onto the idealized theory's 3 ansatz directions
(`final_captured_total`) and PCA effective dimension at 90% variance
(`final_pca_dim`). **Probe validity:** `subspace.py` self-test passes both controls —
planted-direction Δ captures 1.0000; random Δ captures 0.0061; 3-planted-axes PCA
returns exactly 3 (noise control: 31).

## The dissociation (the headline)

| L | ansatz-captured fraction (mean / max of 9) | PCA dim (90% var) |
|---|---|---|
| 64 | 0.0008 / 0.0027 | 3.7 ± 0.7 |
| 96 | 0.0003 / 0.0009 | **3.0 ± 0.0 (9/9 exactly 3)** |
| 128 | 0.0004 / 0.0008 | 3.1 ± 0.3 |
| 192 | 0.0011 / 0.0051 | 2.8 ± 0.4 |
| 256 | 0.0009 / 0.0019 | 3.9 ± 0.3 |

## P3 — Effective dimension: **CONFIRMED, strikingly**

The emergence-window weight movement is **single-digit-dimensional at every L**
(all 45 values in {2,3,4,5}, grand mean 3.3, with L=96 hitting exactly 3 on all
nine seeds) and approximately L-invariant across a 4× context range. The
"emergence happens in a ~3-D subspace" *dimensionality* claim survives standard
architecture + realistic-ish online training.

## P1 — Idealized ansatz directions: **NOT DETECTED — below the random baseline (mapping-validity caveat)**

The theory's three analytical directions capture **≤0.5% of update variance in
every one of 45 runs** (mean ~0.001) — *less than a random direction triple
captures by chance* (0.0061, self-test). The emergence-window updates are
essentially orthogonal to the ansatz. Combined with P3: **the idealized theory
gets the dimension count right, while its directions are not detected** on a
standard architecture — the 3-D structure is real, but the ansatz basis captures
≈0 variance. **Caveat (red-team 2026-06-14):** "not detected" is only "directions
wrong" IF the disentangled→standard mapping is faithful; the planted-direction
self-test (=1.0) checks the projector arithmetic, not the mapping's transport into
this architecture's basis — if the mapping is mis-specified, capture≈0 is the
trivial expected outcome. The defensible statement: the idealization preserves a
coarse invariant (rank) while the geometry (basis) is not recovered here
(mapping-validity unconfirmed).

## P2 — Realism collapse/survival: **recast question answered — the refutation is regime-robust** (arm verdict: Appendix)

The scaffold trains online fresh-batch only (no finite-dataset mode), so the
designed finite-data contrast was not run. The live question after P1's failure
was the reverse: *is batch 64 / lr 1e-3 simply not idealized enough for the
ansatz to appear?* The idealization-hardening arm (batch 256 + lr 3e-4, plus
matched batch-64 cells, L{64,128,256} × 3 seeds, eval_every=10, 18 runs on the
4080, `results/induction_subspace_ideal/`) answers **no**: captured fraction
stays ≈0 under harder idealization (17/18 runs ≤ 0.006 ≈ the random baseline;
single outlier 0.019). P1's refutation is not an artifact of insufficient
idealization.

## P4 — L² across the idealization axis: **REFUTED in floor-free data — emergence step scales ≈ linearly in L, both regimes** (Appendix)

The arm's eval_every=10 cells de-floor the emergence step (main grid was
floor-censored at L ≤ 128). Pairwise log₂ exponents: b64/lr1e-3 0.95 and 1.24;
b256/lr3e-4 0.83 and 1.27 — ≈ L^1.0±0.25 throughout, nobody near L². Computed
at fixed AdamW per the interlock; converges with 007's optimizer-side double
negation from an orthogonal design.

## Limitations

- One task family (repeat/ICL stream), one width/depth, AdamW only (by design).
- Ansatz directions implemented per the e-print's construction for our
  architecture mapping; a hostile reading is that the mapping itself is the
  failure point — the self-test rules out mechanical bugs but not construction
  mismatch. The construction is documented in `subspace.py` for audit.
- PCA dim measured on the emergence *window*; window boundaries inherit the
  eval grid (and the floor at L ≤ 128 — though P3's dimension is insensitive to
  the exact window per the L192/L256 cells, which are floor-free).
- P2's original finite-data arm requires a trainer extension (not yet built).

## Appendix (2026-06-11) — idealization-hardening arm verdict

**Data:** `results/induction_subspace_ideal/` — 18 runs, two regimes ×
L{64,128,256} × 3 seeds, eval_every=10. Run interrupted by a 4080 reboot and
resumed from 7/18 (launcher: `experiments/induction_subspace/launch_ideal_arm.sh`;
the partial L256 cell was cleanly overwritten).

| regime | L | captured (mean / max of 3) | PCA dim | emergence step (mean) |
|---|---|---|---|---|
| b256 lr3e-4 | 64 | 0.0001 / 0.0002 | 4.0 | 90 |
| b256 lr3e-4 | 128 | 0.0071 / **0.0186** | 2.3 | 160 |
| b256 lr3e-4 | 256 | 0.0003 / 0.0005 | 3.0 | 387 |
| b64 lr1e-3 | 64 | 0.0022 / 0.0056 | 4.0 | 47 |
| b64 lr1e-3 | 128 | 0.0002 / 0.0004 | 3.7 | 90 |
| b64 lr1e-3 | 256 | 0.0006 / 0.0010 | 4.0 | 213 |

- **P2:** moving 4× along the batch axis and 3.3× down the lr axis (toward the
  theory's gradient-flow/population limit) leaves ansatz capture at ≈0 — only
  1/18 runs (b256 L128 s0, 0.0186) exceeds the 0.0061 random baseline, still
  40× below the theory's ≥80% prediction. Refutation is regime-robust.
- **P3 reinforcement:** PCA dim stays in {2,3,4} in all 18 arm runs — the
  low-dimensionality invariant holds in both regimes.
- **P4:** floor-free emergence steps give ≈ linear-in-L scaling in both regimes
  (exponents 0.83–1.27); no cell pair approaches quadratic.
- Descriptive note (no claim): at matched L, the harder-idealized regime emerges
  ~1.8× later in steps — consistent with its 3.3× smaller lr; per-sample
  efficiency not analyzed (B×C scaling belongs to 2511.16893's axis).

## Tier-2 (C-KILL-1, 2026-06-15) — optimizer axis: the capture-null is optimizer-invariant; no positive subspace dissociation (weakest leg)

**Why:** 010 fixed AdamW (the F2↔K3 interlock); the red-team (C-KILL-1) needs the
subspace geometry remeasured under other optimizers to claim "geometry ⊥ timing".
Re-ran the probes under **Muon & SGDM** (`results/induction_subspace_optaxis/`,
repeat + markov × L{64,128} × 5 seeds) against the closed AdamW baseline; matched at
AdamW median final ICL via `analyze_optaxis.py`. **Scope limit (stated):** the closed
AdamW baseline covers `repeat` only — `markov` has no AdamW anchor.

- **Capture-null is optimizer-invariant.** `final_captured_total` ≈ 0 (median
  0.000–0.001) for AdamW, Muon, AND SGDM on every task/length — so P1's "NOT
  DETECTED" is not an AdamW artifact; the ansatz directions are unrecovered under any
  optimizer.
- **No positive geometry⊥timing dissociation is establishable here.** (i) `repeat`
  is too easy: ICL saturates (~0.92) at the *first* eval (step 100), so at the point
  of solution pca_dim = 1 for all optimizers (trivially invariant); the final pca_dim
  (adamw 3–4 / muon 5–6 / sgdm 3) is post-solution representational drift — not
  emergence geometry — and is mildly optimizer-dependent (Muon diffuses to higher
  dim). (ii) `markov` shows genuine emergence but has no AdamW anchor; Muon reaches
  higher ICL (0.97–0.98) at pca≈5 while SGDM plateaus (0.84–0.90) at pca≈2, so at
  matched ICL the two optimizers DIFFER in pca_dim — dimensionality is
  optimizer-dependent under genuine emergence, with no AdamW reference to adjudicate.
- **Net:** the 010 leg confirms the *null* (capture) generalizes across optimizers
  but provides no positive subspace dissociation; pca_dim is optimizer-dependent.
  010 stays the weakest C-KILL-1 leg (AdamW-scoped for any positive structure).
  Verdict: `results/figures-010/optaxis_verdict.json`.

## Figures

`results/figures-010/`: `fig_subspace.png`, `subspace_verdicts.json` (now
includes `ideal_arm`); `optaxis_verdict.json` (Tier-2 optimizer axis).

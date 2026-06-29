# Direction 010 — Does induction-head emergence live in a 3-D subspace?

Tests, on **REAL** training, the idealized claim of **arXiv:2511.01033** ("On the
Emergence of Induction Heads for In-Context Learning", Musat et al., ETH Zürich;
ICLR-2026 withdrawn). The paper proves — under *gradient flow + population MSE
loss + zero init + orthonormal inputs* in a *disentangled attention-only*
transformer — that the weight trajectory stays in a **19-dimensional** subspace,
and finds empirically that only **3** pseudo-parameters (α₃, β₂, γ₃) account for
induction-head emergence, with emergence time `t_ICL = Θ(N²)` (quadratic in
context length N).

We ask whether that 3-D structure survives **non-idealized** training: minibatch,
**softmax** attention, **cross-entropy** loss, finite data, **FIXED AdamW**. At
every eval we project the Q/K/OV weight **update** onto 3 canonical
induction-circuit directions, measure the **captured-variance fraction** through
the emergence window, and track the **PCA effective dimension** of the
cumulative-Δθ trajectory.

**Interlock with direction 007.** The optimizer is FIXED to AdamW. This study
varies the TASK and probes the SUBSPACE; it does **not** compare optimizers
(007's job). Fixing AdamW isolates the subspace question from optimizer geometry.

## Reuse (induction_emergence is NOT modified)
Imported WHOLESALE from `../induction_emergence` (which itself imports the
grokking infra by file-spec, so `SeqTransformer → GrokTransformer` resolves from
that dir):
- `data.py` — `InductionSpec`, `sample_batch`, `batch_seed`
- `probes.py` — `position_accuracies`, `icl_score`, `prefix_match_score`, `detect_emergence`
- `model.py` — `SeqTransformer` (2 layers, 4 heads, d=128, full-sequence logits)

Import discipline: **LOCAL FIRST**, `induction_emergence` appended. Our
load-bearing NEW module `subspace.py` is uniquely named, so no collision occurs.

## The 3 canonical directions (`subspace.py`)
Built from the paper's standard-transformer **progress measures** (its Eqs.
20–22), as unit-normalized **gradients of each scalar measure** w.r.t. the
relevant weight block — a weight update that raises that measure moves along it:

| Dir | Measure | Construction | Mechanism |
|-----|---------|--------------|-----------|
| **D1** | α̃₃ = Σ (W_K¹ p_{i-1})ᵀ(W_Q¹ p_i) | layer-1 QK, **positional** outer products, subdiagonal | attend position t → t−1 (prev-token) |
| **D2** | β̃₂ = Σ (W_K² W_O¹ W_V¹ t_i)ᵀ(W_Q² t_i) | layer-2 QK, **token** outer products routed via L1 OV | match the preceding-token |
| **D3** | γ̃₃ = tr(W_o W_O² W_V² T) | layer-2 OV, **embedding–unembedding** alignment | copy attended token to its logit |

Projection API: `captured_fraction(delta, directions) = ‖proj‖² / ‖delta‖²`
(per-direction + total over the orthonormalized span). Effective dimension:
`pca_effective_dim(list_of_Δθ, var_threshold=0.9)`.

### EXTRACTION-TODO caveat (read before any findings claim)
These are the **generic canonical** constructions, derived from the paper's
progress-measure DEFINITIONS and mechanistic description — **not** a verbatim
transcription of the paper's idealized **merged-QK 3-parameter ansatz** (its Eqs.
5–7), which is stated for a *different architecture* (disentangled/concatenated
residual stream, no MLP, no LayerNorm). The disentangled-→-standard change of
variables (§4.2 "interpretable transformation", Fig. 7) is **not** re-derived
here. The basis is exposed as a **pluggable** interface
(`build_canonical_directions`); swap it to test the exact merged-QK ansatz. The
full extracted construction (architecture, theorems, emergence order
T_γ<T_β<T_α, t_ICL=Θ(N²)) is recorded in `subspace.EXTRACTION_TODO`. **Verify the
paper construction against the PDF before any findings claim that depends on the
exact ansatz subspace** (vs the progress-measure directions used here, which are
the paper's own real-weight bridge).

## Self-tests
```
python subspace.py            # planted-direction exactness + PCA dim recovery (PASS)
```
Certifies: a Δ exactly along D1 → `captured=1.0` for D1, ~0 for D2/D3, total 1.0;
a random Δ → small captured fraction; 3 planted axes + noise → PCA dim = 3 (and
isotropic noise → many components).

## Smoke (no files written, <60 s)
```
python train_subspace.py --smoke      # labeled smoke lines (incl. subspace probe)
python run_subspace.py --smoke        # delegates to the trainer smoke
```
Prints exactly:
```
SMOKE DATASET SHAPE: ...
SMOKE PARAM COUNT: 427264
SMOKE FORWARD LOSS: <float>
SMOKE OPTIMIZER STEP: OK            # AdamW
SMOKE SUBSPACE PROBE: captured=<f> pca_dim=<n>   # real update delta; captured ~0 untrained
```
(≤1 step, no jsonl, exit 0; the projection machinery runs end-to-end on one real
AdamW update delta — captured is expected ~0 on an untrained step.)

## Dry run (prints planned cells, launches nothing)
```
python run_subspace.py --dry-run                # 108 cells (2 task × 6 L × 9 seeds)
python run_subspace.py --dry-run --vocab-arm    # 108 + 54 vocab-robustness = 162
```

## Real grid (run when ready — do NOT launch yet; GPU is busy)
```
python run_subspace.py              # FIXED AdamW; 108 cells
python run_subspace.py --vocab-arm  # + vocab-robustness arm (54 cells)
```
- **task** ∈ {repeat, markov} — repeated-segment vs Markov-bigram induction (same
  induction signal, different surface statistics).
- **seq_len** ∈ {64, 96, 128, 192, 256, 384} — 6 lengths for the `t_ICL ~ N²` fit.
- **seed** ∈ 0..8.

Per-eval jsonl (real runs only): loss, repeat/first acc, `icl_score`,
prefix-match per head, `captured_{D1,D2,D3,total}`, `pca_dim`,
`pca_participation`. Results land in
`../../experiments/results/induction_subspace/`. Resume-aware: a cell whose
`.jsonl` already ends with a `_summary` line is skipped.

## Reference
Full research write-up: `directions/010-induction-subspace.md` (planned).
Paper under test: arXiv:2511.01033.

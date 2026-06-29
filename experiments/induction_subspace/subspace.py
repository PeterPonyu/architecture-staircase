"""Direction 010 — the 3-D induction-circuit subspace probe (load-bearing module).

THE CLAIM UNDER TEST (arXiv:2511.01033, ICLR-2026 withdrawn). In an *idealized*
setting — a disentangled 2-layer attention-only transformer, ZERO init,
POPULATION MSE loss, GRADIENT FLOW, orthonormal inputs — the authors prove the
weight trajectory is confined to a 19-dimensional subspace, and empirically that
only **3** pseudo-parameters (their α₃, β₂, γ₃) account for induction-head
emergence. This module asks the empirical question for **REAL** training
(minibatch, softmax attention, cross-entropy, finite data, AdamW): do the
per-eval Q/K/OV weight *updates* concentrate along the same 3 canonical
directions, and is the cumulative-Δθ trajectory effectively low-dimensional?

THE THREE CANONICAL DIRECTIONS (mapping paper → SeqTransformer weights)
----------------------------------------------------------------------
The paper's three emergent quantities are *progress measures* — scalar bilinear
forms on the weights (their Eqs. 20–22 for the standard transformer):

  α̃₃ = Σ_i (W_K¹ p_{i-1})ᵀ (W_Q¹ p_i)          # L1 prev-token attention (positions)
  β̃₂ = Σ_i (W_K² W_O¹ W_V¹ t_i)ᵀ (W_Q² t_i)     # L2 match attention (tokens via L1 OV)
  γ̃₃ = tr(W_o W_O² W_V² T)                       # L2 OV copy (emb–unembed alignment)

Mechanistically (paper §3.3 / §4.3): α₃ makes each position attend to the
PREVIOUS position (subdiagonal of the L1 key-query matrix on positional
embeddings); β₂ makes the query token attend to the position whose preceding
token MATCHES the current token (L2 key-query on token embeddings, the key side
routed through the L1 OV circuit); γ₃ COPIES the attended token's embedding to
its own output logit direction (L2 OV composed with the unembedding).

A "direction in weight space" for each measure is the **gradient of that scalar
measure w.r.t. the relevant weight block**, normalized to unit Frobenius norm. A
weight update that increases progress-measure α (resp. β, γ) moves along
∂α̃₃/∂W (resp. ∂β̃₂/∂W, ∂γ̃₃/∂W). Concretely these gradients are exactly the
outer-product / aligned-matrix constructions the task description calls for:

  D1 (W_Q¹, W_K¹ space): built from POSITIONAL-embedding outer products with a
     position→position-1 (subdiagonal) shift  — "attend t→t-1".
  D2 (W_Q², W_K² space): built from TOKEN-embedding outer products, the key side
     routed through the learned L1 OV map (W_O¹ W_V¹) — "match preceding token".
  D3 (W_V², W_O² / unembed space): built from TOKEN-embedding ⊗ unembedding
     alignment — "copy attended token to its logit".

Projection API:  captured_fraction(delta, directions) = ||proj||² / ||delta||²,
where the projection is onto the orthonormalized span of the supplied direction
tensors (each tensor lives in the SAME weight-block space as the corresponding
slice of `delta`). Per-direction fractions use each (unit) direction alone.

HONEST CAVEAT — EXTRACTION-TODO (read before any findings claim)
----------------------------------------------------------------
These are the GENERIC canonical constructions of the three induction-circuit
directions, derived from the paper's progress-measure DEFINITIONS (Eqs. 20–22)
and its mechanistic description. They are NOT a verbatim transcription of the
paper's idealized 3-parameter ANSATZ basis, which is stated for a *disentangled,
merged-QK, attention-only* architecture (their Eqs. 5–7, W^{(1)},W^{(2)},W^{(3)};
α,β,γ index blocks of those merged matrices). Our SeqTransformer is a *standard*
pre-norm transformer with fused `qkv`, separate `proj`, an MLP, and LayerNorm —
so an exact-basis match is impossible without re-deriving the change of variables
the paper applies (their "interpretable transformation", §4.2, Fig. 7). The basis
here is therefore exposed as a PLUGGABLE interface (`build_canonical_directions`
returns a dict you can swap) and the constructions follow the standard-transformer
progress measures, which the paper itself uses (Eqs. 20–22) as the empirical
bridge to real weights. BEFORE writing findings: verify the construction against
the PDF (see EXTRACTION_TODO at the bottom of this file).

WebFetch on arxiv.org/abs/2511.01033 + the PDF succeeded; the extracted
construction (Eqs. 20–22, emergence order T_γ<T_β<T_α, t_ICL=Θ(N²)) is recorded
in EXTRACTION_TODO and drove the constructions above.

Self-tests (`python subspace.py`)
---------------------------------
  * a synthetic Δ placed EXACTLY along D1 → captured_fraction≈1.0 for D1, ≈0 for
    D2/D3, and ≈1.0 for the full 3-direction span;
  * a random Δ → small captured fraction (≈ k/dim for k directions);
  * PCA effective dim of (3 planted orthogonal directions + small noise) ≈ 3.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch

# Import discipline (LOCAL FIRST, induction_emergence APPENDED). Our uniquely-named
# `subspace.py` resolves from THIS dir; `model`/`data`/`probes` are reused WHOLESALE
# from ../induction_emergence (which itself imports the grokking infra by file-spec,
# so the chain SeqTransformer → GrokTransformer resolves from there). We append (not
# prepend) induction_emergence so any local module would still win a name collision.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
_IE_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "induction_emergence"))
if _IE_DIR not in sys.path:
    sys.path.append(_IE_DIR)


# ---------------------------------------------------------------------------
# Weight extraction helpers — pull the Q/K/V/O sub-blocks out of the fused qkv.
# ---------------------------------------------------------------------------
# SeqTransformer block uses a single fused `qkv = Linear(d, 3d)` whose weight is
# [3d, d] stacked as [Wq; Wk; Wv]; the output projection is `proj = Linear(d,d)`,
# whose weight [d, d] plays the role of W_O (per-head output mixed back to model).
def qkv_slices(d_model: int):
    """Row-index slices (q, k, v) into a fused qkv weight of shape [3*d, d]."""
    return (slice(0, d_model),
            slice(d_model, 2 * d_model),
            slice(2 * d_model, 3 * d_model))


def extract_qkvo(model) -> dict:
    """Return per-layer {Wq, Wk, Wv, Wo} weight matrices (each [d, d]) as views.

    Wq/Wk/Wv are the row-slices of the fused `blocks[l].qkv.weight`; Wo is
    `blocks[l].proj.weight`. Layer index l in {0,1} == paper layers {1,2}.
    """
    out = {}
    d = model.blocks[0].qkv.weight.shape[1]
    sq, sk, sv = qkv_slices(d)
    for l, blk in enumerate(model.blocks):
        W = blk.qkv.weight                       # [3d, d]
        out[l] = {
            "Wq": W[sq], "Wk": W[sk], "Wv": W[sv],   # each [d, d]
            "Wo": blk.proj.weight,                    # [d, d]
        }
    return out


# ---------------------------------------------------------------------------
# Canonical direction constructions (the 3 induction-circuit directions).
# ---------------------------------------------------------------------------
# Each direction is returned as a dict mapping a PARAMETER KEY -> tensor in that
# parameter's space, so a single direction can span several weight blocks (e.g.
# D1 lives in BOTH W_Q¹ and W_K¹). Keys follow `extract_qkvo` layer indexing:
#   "qkv.{l}.q" / ".k" / ".v"  -> row-slice of blocks[l].qkv.weight
#   "proj.{l}"                  -> blocks[l].proj.weight   (W_O)
#   "unembed"                   -> model.unembed.weight
# A direction tensor has the SAME shape as the slice it indexes.

def _unit(d: dict) -> dict:
    """Normalize a multi-block direction dict to unit total Frobenius norm."""
    nrm = torch.sqrt(sum((t * t).sum() for t in d.values())).clamp_min(1e-12)
    return {k: v / nrm for k, v in d.items()}


@torch.no_grad()
def build_canonical_directions(model) -> dict:
    """Build the 3 canonical induction-circuit directions on `model`'s weights.

    Returns {"D1": dir_dict, "D2": dir_dict, "D3": dir_dict}, each `dir_dict`
    mapping parameter-key -> unit-normalized direction tensor (multi-block).
    PLUGGABLE: swap this function to test a different ansatz basis.

    Constructions (progress-measure gradients; see module docstring & paper Eqs):
      D1 — L1 prev-token QK: gradient of α̃₃=Σ(W_K¹ p_{i-1})ᵀ(W_Q¹ p_i) wrt W_Q¹,W_K¹.
           ∂α̃₃/∂W_Q¹ = Σ_i (W_K¹ p_{i-1}) p_iᵀ ;  ∂α̃₃/∂W_K¹ = Σ_i (W_Q¹ p_i) p_{i-1}ᵀ.
           At the canonical (identity-on-positions) circuit these reduce to the
           POSITIONAL outer-product sums P_shift below — "attend position t→t-1".
      D2 — L2 match QK: gradient of β̃₂=Σ(W_K² M t_i)ᵀ(W_Q² t_i) wrt W_Q²,W_K²,
           with M = W_O¹ W_V¹ the learned L1 OV map (routes the key side).
           ∂β̃₂/∂W_Q² = Σ_i (W_K² M t_i) t_iᵀ ;  ∂β̃₂/∂W_K² = Σ_i (W_Q² t_i)(M t_i)ᵀ.
           Built from TOKEN-embedding outer products routed through M.
      D3 — L2 OV copy: gradient of γ̃₃=tr(W_o W_O² W_V² T) wrt W_V²,W_O².
           ∂γ̃₃/∂W_V² = (W_o W_O²)ᵀ Tᵀ-shaped ;  built from EMBEDDING–UNEMBEDDING
           alignment (T ⊗ W_o) — "copy attended token to its own logit".
    """
    P = model.pos_emb.weight        # [seq_len, d] positional embeddings
    Tt = model.tok_emb.weight       # [vocab, d]   token embeddings
    Wu = model.unembed.weight       # [vocab, d]   unembedding
    qkvo = extract_qkvo(model)
    d = P.shape[1]

    # --- D1: layer-1 prev-token QK direction (positional, subdiagonal) ---------
    # Position pairs (i, i-1): keys read p_{i-1}, queries read p_i.
    Pq = P[1:]                       # p_i      for i=1..L-1   [L-1, d]
    Pk = P[:-1]                      # p_{i-1}              [L-1, d]
    Wq1, Wk1 = qkvo[0]["Wq"], qkvo[0]["Wk"]
    # gradient pieces (above); reduce over position pairs -> [d, d] each
    gWq1 = (Wk1 @ Pk.T) @ Pq         # Σ_i (W_K¹ p_{i-1}) p_iᵀ      [d, d]
    gWk1 = (Wq1 @ Pq.T) @ Pk         # Σ_i (W_Q¹ p_i) p_{i-1}ᵀ      [d, d]
    D1 = _unit({"qkv.0.q": gWq1, "qkv.0.k": gWk1})

    # --- D2: layer-2 match QK direction (tokens routed via L1 OV) --------------
    M = qkvo[0]["Wo"] @ qkvo[0]["Wv"]            # W_O¹ W_V¹   [d, d] L1 OV map
    Wq2, Wk2 = qkvo[1]["Wq"], qkvo[1]["Wk"]
    MT = (M @ Tt.T)                              # M t_i for all tokens  [d, vocab]
    gWq2 = (Wk2 @ MT) @ Tt                       # Σ_i (W_K² M t_i) t_iᵀ  [d, d]
    gWk2 = (Wq2 @ Tt.T) @ MT.T                   # Σ_i (W_Q² t_i)(M t_i)ᵀ [d, d]
    D2 = _unit({"qkv.1.q": gWq2, "qkv.1.k": gWk2})

    # --- D3: layer-2 OV copy direction (embedding–unembedding alignment) -------
    Wv2, Wo2 = qkvo[1]["Wv"], qkvo[1]["Wo"]
    # γ̃₃ = tr(W_o W_O² W_V² T); gradients wrt W_V² and W_O² (T·W_o coupling):
    #   ∂/∂W_V² = (W_o W_O²)ᵀ Tᵀ summed -> (W_o W_O²)ᵀ @ Tᵀ.T-shaped  [d, d]
    A = Wu @ Wo2                                 # W_o W_O²    [vocab, d]
    gWv2 = A.T @ Tt                              # (W_o W_O²)ᵀ T   [d, d]
    #   ∂/∂W_O² = W_oᵀ (T W_V²)ᵀ-style coupling -> W_oᵀ @ (W_V² Tᵀ).T  [d, d]
    gWo2 = Wu.T @ (Wv2 @ Tt.T).T                 # [d, d]
    D3 = _unit({"qkv.1.v": gWv2, "proj.1": gWo2})

    return {"D1": D1, "D2": D2, "D3": D3}


# ---------------------------------------------------------------------------
# Projection API.
# ---------------------------------------------------------------------------
def _flatten_aligned(delta: dict, direction: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten `direction` and the matching keys of `delta` into aligned vectors.

    Only keys present in `direction` are used from `delta`; missing delta keys
    contribute zero (the direction simply finds no update mass there).
    """
    dvec, gvec = [], []
    for k, gt in direction.items():
        gvec.append(gt.reshape(-1))
        dt = delta.get(k)
        dvec.append(dt.reshape(-1) if dt is not None else torch.zeros_like(gt).reshape(-1))
    return torch.cat(dvec), torch.cat(gvec)


@torch.no_grad()
def captured_fraction(delta: dict, directions) -> dict:
    """Fraction of an update's energy captured by the canonical direction(s).

    Args:
      delta      : {param_key: Δweight tensor} — the weight update to analyze.
      directions : a single direction dict, OR a mapping {name: direction dict}
                   (e.g. the output of build_canonical_directions).

    Returns dict:
      per_direction : {name: ||proj_onto_that_unit_dir||² / ||delta_restricted||²}
                      where delta is restricted to the blocks that direction spans.
      total         : ||proj_onto_orthonormal_span(all dirs)||² / ||delta_all||²,
                      computed over the UNION of blocks the directions touch.
    Definition: captured_fraction = ||proj||²/||Δ||²  (the requested formula).

    Blocks are aligned by shared key. A direction key absent from `delta` means
    that update block is zero in the projection (the direction finds no mass).
    """
    if all(isinstance(v, torch.Tensor) for v in directions.values()):
        directions = {"D": directions}              # a bare single direction dict

    # Per-direction fraction: project delta (restricted to the direction's blocks)
    # onto the single unit direction.
    per = {}
    union_keys: list[str] = []
    for name, dvec in directions.items():
        d_flat, g_flat = _flatten_aligned(delta, dvec)
        g_unit = g_flat / g_flat.norm().clamp_min(1e-12)
        denom = (d_flat @ d_flat).clamp_min(1e-24)
        per[name] = float((d_flat @ g_unit) ** 2 / denom)
        for k in dvec:
            if k not in union_keys:
                union_keys.append(k)

    # Total: project delta (restricted to the UNION of direction blocks) onto the
    # orthonormal span of all directions. Each direction key has a known reference
    # tensor, so a missing delta block is a genuine zero vector of that exact
    # length, device, and dtype.
    ref = {}
    for dvec in directions.values():
        for k, t in dvec.items():
            ref[k] = t

    def over_union(src: dict) -> torch.Tensor:
        parts = []
        for k in union_keys:
            t = src.get(k)
            parts.append(t.reshape(-1) if t is not None
                         else torch.zeros(ref[k].numel(), device=ref[k].device,
                                          dtype=ref[k].dtype))
        return torch.cat(parts)

    x = over_union(delta)
    cols = [over_union(dvec) for dvec in directions.values()]
    Dmat = torch.stack(cols, dim=1)                 # [dim, n_dirs]
    Q, _ = torch.linalg.qr(Dmat, mode="reduced")    # orthonormal span [dim, r]
    coords = Q.T @ x
    per["total"] = float((coords @ coords) / (x @ x).clamp_min(1e-24))
    return per


@torch.no_grad()
def snapshot_weights(model) -> dict:
    """Clone the Q/K/V/O + unembed weights into a flat {param_key: tensor} dict.

    Keys match the direction-dict convention so deltas align with directions:
      "qkv.{l}.q/.k/.v", "proj.{l}", "unembed".
    """
    snap = {}
    d = model.blocks[0].qkv.weight.shape[1]
    sq, sk, sv = qkv_slices(d)
    for l, blk in enumerate(model.blocks):
        W = blk.qkv.weight
        snap[f"qkv.{l}.q"] = W[sq].detach().clone()
        snap[f"qkv.{l}.k"] = W[sk].detach().clone()
        snap[f"qkv.{l}.v"] = W[sv].detach().clone()
        snap[f"proj.{l}"] = blk.proj.weight.detach().clone()
    snap["unembed"] = model.unembed.weight.detach().clone()
    return snap


def weight_delta(prev: dict, curr: dict) -> dict:
    """Per-key Δweight = curr - prev (only keys present in both)."""
    return {k: curr[k] - prev[k] for k in curr if k in prev}


def flatten_delta(delta: dict) -> torch.Tensor:
    """Concatenate a Δweight dict into a single 1-D vector (sorted-key order)."""
    return torch.cat([delta[k].reshape(-1) for k in sorted(delta)])


# ---------------------------------------------------------------------------
# PCA effective dimension of a set of cumulative-Δθ vectors.
# ---------------------------------------------------------------------------
@torch.no_grad()
def pca_effective_dim(vectors, var_threshold: float = 0.9) -> dict:
    """Effective dimensionality of a list of Δθ vectors via PCA.

    Args:
      vectors       : list of 1-D tensors (equal length), e.g. cumulative Δθ at
                      successive evals (each row = one observation).
      var_threshold : cumulative explained-variance level (default 0.9).

    Returns dict:
      n_components  : smallest #PCs whose cumulative explained variance >= thresh.
      participation : participation ratio (Σλ)²/Σλ² — a continuous effective dim.
      explained     : list of per-PC explained-variance ratios (descending).
    Mean-centered across observations; uses SVD on the [n_obs, dim] matrix.
    """
    if len(vectors) == 0:
        return {"n_components": 0, "participation": 0.0, "explained": []}
    M = torch.stack([v.reshape(-1).float() for v in vectors], dim=0)  # [n, dim]
    M = M - M.mean(dim=0, keepdim=True)
    n = M.shape[0]
    if n < 2:
        return {"n_components": 1, "participation": 1.0, "explained": [1.0]}
    # singular values of centered data -> variances λ_k ∝ s_k².
    s = torch.linalg.svdvals(M)                      # [min(n,dim)]
    var = (s * s)
    total = var.sum().clamp_min(1e-24)
    ratios = (var / total)
    csum = torch.cumsum(ratios, dim=0)
    n_comp = int((csum < var_threshold).sum().item()) + 1
    n_comp = min(n_comp, ratios.numel())
    participation = float((var.sum() ** 2) / (var * var).sum().clamp_min(1e-24))
    return {
        "n_components": n_comp,
        "participation": participation,
        "explained": [float(x) for x in ratios],
    }


# ---------------------------------------------------------------------------
# EXTRACTION_TODO — paper construction extracted from arXiv:2511.01033 (v2).
# ---------------------------------------------------------------------------
EXTRACTION_TODO = """
arXiv:2511.01033 "On the Emergence of Induction Heads for In-Context Learning"
(Musat, Pimentel, Noci, Stolfo, Sachan, Hofmann; ETH Zürich; v2, 8 Jan 2026).
PDF fetched + read (pages 1–8). Key construction extracted:

 * Idealized model (§3): disentangled 2-layer ATTENTION-ONLY transformer, merged
   key-query (W^{(1)}∈R^{2D×2D}, W^{(2)}∈R^{4D×4D}) and OV/projection W^{(3)},
   residual stream built by CONCATENATION not addition (Eqs. 5–7); zero init
   (Asm. 1), POPULATION MSE loss (Asm. 2), gradient flow unit-lr (Asm. 5),
   ORTHONORMAL/isotropic inputs (Asm. 3,7), query-last (Asm. 8).
 * Theorem 1: weights stay in a 19-D subspace, indexed by α∈R³, β∈R^12, γ∈R⁴,
   with explicit block structure W^{(1)}=diag(α₁I, α₂I+α₃M), etc. (M = the fixed
   rotation Eq. 4 that pairs items with their labels).
 * §3.3: only **α₃, β₂, γ₃** grow large (Fig. 3) and together implement the
   induction head — α₃: L1 attend-to-previous; β₂: L2 match-on-token; γ₃: L2 copy.
   Emergence is SELF-CONTAINED in the 3-D subspace (Fig. 5, Asm. 4).
 * Theorem 2 / Cor. 1: emergence order T_γ < T_β < T_α; T_α,T_β=Θ(N²), T_γ=Θ(N);
   total t_ICL = Θ(N²) — QUADRATIC in context length N.
 * §4 (STANDARD transformer bridge — what we actually instrument): progress
   measures (Eqs. 20–22), which our D1/D2/D3 implement as weight-space gradients:
     α̃₃ = Σ_i (W_K¹ p_{i-1})ᵀ (W_Q¹ p_i)
     β̃₂ = Σ_i (W_K² W_O¹ W_V¹ t_i)ᵀ (W_Q² t_i)
     γ̃₃ = tr(W_o W_O² W_V² T)
   They report the SAME emergence order in the standard model (Fig. 8), but note
   cross-entropy (vs MSE) inflates γ₃,β₂ magnitudes and delays loss drop.

VERIFY-BEFORE-FINDINGS: our directions are gradients of the §4 progress measures
(Eqs. 20–22), the paper's own real-weight bridge — NOT the idealized merged-QK
ansatz basis (Eqs. 5–7), which assumes a different architecture (concatenated
residual, no MLP, no LayerNorm). The disentangled-→-standard change of variables
(§4.2 "interpretable transformation", Fig. 7) is NOT re-derived here. If a
findings claim depends on the EXACT ansatz subspace (not just the progress-measure
directions), re-derive that transformation against the PDF first. Pluggable:
replace `build_canonical_directions` to test the merged-QK basis.
"""


# ---------------------------------------------------------------------------
# Self-test.  `python subspace.py`
# ---------------------------------------------------------------------------
def _self_test() -> int:
    torch.manual_seed(0)
    ok = True

    # Build a tiny real model and its 3 canonical directions.
    from model import SeqTransformer
    model = SeqTransformer(vocab_size=16, seq_len=12, d_model=16, n_heads=2,
                           n_layers=2, mlp_ratio=2, init_scale=1.0)
    dirs = build_canonical_directions(model)
    assert set(dirs) == {"D1", "D2", "D3"}

    # (1) Δ EXACTLY along D1 -> captured≈1 for D1, ≈0 for D2/D3, total≈1.
    delta_d1 = {k: v.clone() for k, v in dirs["D1"].items()}
    cf = captured_fraction(delta_d1, dirs)
    print("SELF-TEST planted-D1 captured fractions:")
    print(f"  D1={cf['D1']:.4f} (expect ~1)  D2={cf['D2']:.4f} (expect ~0)  "
          f"D3={cf['D3']:.4f} (expect ~0)  total={cf['total']:.4f} (expect ~1)")
    if not (cf["D1"] > 0.999 and cf["D2"] < 0.05 and cf["D3"] < 0.05
            and cf["total"] > 0.999):
        ok = False
        print("  FAIL: planted-D1 exactness violated")

    # (2) random Δ -> small captured fraction (≈ 3 dirs / total dimension).
    rand_delta = {k: torch.randn_like(v) for k, v in snapshot_weights(model).items()}
    cfr = captured_fraction(rand_delta, dirs)
    # total dimension spanned by the 3 directions:
    span_dim = sum(v.numel() for d in dirs.values() for v in d.values())
    # (distinct blocks only — D2/D3 may share none; rough upper bound for message)
    print(f"  random-Δ total captured = {cfr['total']:.4f} "
          f"(expect small, << 1)")
    if cfr["total"] > 0.25:
        ok = False
        print("  FAIL: random delta captured too much energy")

    # (3) PCA: 3 planted orthogonal directions (comparable strength) + tiny noise
    #     -> all 3 needed to clear the 90% variance threshold, eff dim ≈ 3.
    dim = 200
    torch.manual_seed(1)
    basis = torch.linalg.qr(torch.randn(dim, 3))[0]    # [dim, 3] orthonormal
    obs = []
    for _ in range(40):
        coeffs = torch.randn(3) * torch.tensor([1.2, 1.0, 0.9])  # 3 comparable axes
        v = basis @ coeffs + 0.01 * torch.randn(dim)             # tiny noise
        obs.append(v)
    pca = pca_effective_dim(obs, var_threshold=0.9)
    print("SELF-TEST PCA effective dim (3 planted axes + noise):")
    print(f"  n_components(90% var) = {pca['n_components']} (expect 3)  "
          f"participation = {pca['participation']:.2f} (expect ~3)")
    if pca["n_components"] != 3:
        ok = False
        print(f"  FAIL: PCA n_components {pca['n_components']} != 3")
    if not (2.0 <= pca["participation"] <= 4.0):
        ok = False
        print(f"  FAIL: participation ratio {pca['participation']:.2f} not ~3")

    # (4) full-rank noise -> PCA effective dim large (sanity: not collapsed to 3).
    noise_obs = [torch.randn(dim) for _ in range(40)]
    pca_n = pca_effective_dim(noise_obs, var_threshold=0.9)
    print(f"  full-noise n_components(90% var) = {pca_n['n_components']} "
          f"(expect >> 3)")
    if pca_n["n_components"] <= 5:
        ok = False
        print("  FAIL: isotropic noise should need many components")

    print("SUBSPACE SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())

# Mathematical backing for Paper C — feasibility audit (2026-06-21)

**Scope.** Paper C = "Degree-staircase emergence geometry is architecture-shaped while
optimizers shift timing." This report assesses which of C's load-bearing EMPIRICAL claims
admit a MATHEMATICAL argument (expressivity / approximation theory / learning dynamics)
that would corroborate them and raise the work above pure phenomenology for an SCI venue
(Neurocomputing / Physica D, which value an analytic backbone).

**Sources read for exact claims/quantities.**
- `papers/C/main.tex` (manuscript)
- `experiments/findings-004-degree-staircase.md`, `findings-007.md`, `findings-010.md`, `findings-017.md`
- Architecture/target ground truth from code:
  - `experiments/grokking/model.py:15-77` — `GrokTransformer`: pre-norm `Block` = full
    (causal) softmax self-attention (`qkv`/`proj`, no bias) + 2-layer GELU MLP
    (`fc1`/`fc2`, `mlp_ratio=4`), residual; `n_layers=2`, `d_model=128`, `n_heads=4`.
    `forward` reads out at the FINAL position only.
  - `experiments/degree_staircase/data.py:7-44,77-89` — input is L=16 Boolean (±1) tokens
    + trailing `EQ` marker (`seq_len=L+1=17`, `vocab_size=3`); target
    `g(x)=sum_{k=1..4} chi_{S_k}(x)` with disjoint position sets `|S_k|=k` (a clean
    nested "merged-staircase / leap" construction); scalar model function
    `f(x)=logit[1]-logit[0]` at EQ, regressed onto `g` with MSE (`probes.py:33-66`).
  - `experiments/arch_staircase/train_arch.py:35-54` — freeze arms = `requires_grad=False`
    on `{qkv,proj}` (attn) or `{fc1,fc2}` (mlp), depth/width grids.

A blanket honesty note up front: **all named theorems below are recalled from my training
knowledge and MUST be checked against the actual papers before any are cited in C.** I flag
the highest-risk recollections inline. The toy setup matches some theorem hypotheses well
(Boolean cube, orthonormal Walsh basis, online/fresh-batch SGD-family training) and others
poorly (finite L vs asymptotic d, softmax attention + LayerNorm vs the clean architectures
in expressivity proofs). I am explicit where an argument would only *rationalize* rather
than *prove*.

---

## Per-claim table

| # | Claim (quoted, with empirical quantity) | Proposed mathematical argument | Feasibility | Worth including? | Risk |
|---|---|---|---|---|---|
| C1 | "the staircase requires depth >= 2 ... a single layer **fails to learn the high degrees** (degree-3/4 final correlation ~0, fit 0.71), while two layers learn them (fit 0.91)" (main.tex:155-158; findings-017 P1) | **EXPRESSIVITY / interaction-order bound.** Show a 1-layer model in this exact readout (attention-pool/mean → linear, or a single attention layer with no second mixing stage) can represent only bounded-interaction-order functions of the input bits, so it cannot represent `chi_{S_4}` (a degree-4 multilinear monomial = product of 4 distinct bits) without exponential width, whereas 2 layers compose pairwise products up to degree 4. Connects to parity/multiplication lower bounds and the "composition doubles achievable degree" intuition. | **Plausible-with-work** (clean version is Hard) | **Yes — flagship.** This is the single argument that converts an empirical depth threshold into a *necessity*, exactly the upgrade an SCI venue rewards. | **MEDIUM–HIGH.** The honest obstacle: a softmax-attention layer is NOT degree-limited the way a linear/mean-pool layer is (softmax is non-polynomial; one attention layer can already compute pairwise products via QK, and an MLP is a universal approximator at finite L). So a *clean* "1 layer literally cannot represent degree 4" theorem is likely FALSE as stated for this architecture. The defensible version is an *efficient-representation / approximation-with-bounded-width* statement, or a learnability (not representability) argument. Risk of hand-waving if the representability vs learnability distinction is blurred. See sketch §A. |
| C2 | "the per-degree learning staircase is statistically equivalent across optimizers" — SI 3.53 (Muon) vs 2.98 (AdamW), TOST p=0.044; order is low→high degree (findings-004 P1/P2; main.tex:103-125) | **GRADIENT-CORRELATION / spectral-priority argument.** At small init the gradient signal on the degree-k Walsh component is proportional to its current correlation with the residual; because the Walsh characters are ORTHONORMAL on the cube, the population gradient toward `chi_{S_k}` is initially ~0 for high k and only becomes nonzero once lower-degree structure seeds it (leap/staircase mechanism). The ORDER (low→high) is a property of the target's hierarchical structure + the orthonormal basis, NOT of the optimizer's preconditioner; an optimizer rescales step sizes (timing) but a diagonal/orthogonalizing preconditioner does not create degree-k correlation where the population gradient is zero. This *predicts* SI invariance up to a time-rescaling. | **Plausible-with-work** | **Yes — second flagship.** Directly rationalizes the headline equivalence and the "timing not order" dissociation; ties to the very theory C is arguing against (idealized-Muon flattening). | **MEDIUM.** The leap/staircase gradient argument is real and well-matched to the disjoint-monomial target (this IS the construction those theorems use). The risk is the *optimizer-invariance* leap: Muon orthogonalizes the per-matrix update, which is NOT a pure per-coordinate time rescaling, so "order is preserved because preconditioning can't manufacture zero-gradient signal" needs care — orthogonalization can change effective per-mode rates non-uniformly. Defensible as a *mechanism sketch / heuristic prediction*, risky as a theorem. See sketch §B. |
| C3 | "attention, not the MLP, carries acquisition of the high-degree component" — deg-4 corr +0.035 (attn-frozen) vs +0.309 (mlp-frozen); **AdamW-specific** (Muon Δ_attn=+0.005) (main.tex:151-188; findings-017 P2 + Tier-2) | **EXPRESSIVITY of the frozen sub-network.** With attention frozen at (near-random) init, the attention block is a fixed (roughly position-mixing-free / content-agnostic) map; the trainable MLP acts position-wise and cannot by itself form the cross-position product `prod_{i in S_4} x_i` (multiplicative interaction ACROSS sequence positions requires the token-mixing that attention provides). So degree-4 (a 4-way cross-position product) is unreachable with attention frozen — a representational bottleneck. | **Plausible-with-work**, but **scope-limited** | **Partial / cautious.** Worth a short paragraph as the *mechanistic reason* attention owns cross-position products — BUT the empirical result is explicitly AdamW-specific (Muon bypasses it), and a pure-expressivity theorem would predict the bottleneck under EVERY optimizer. The math therefore over-predicts. | **HIGH (as a clean theorem).** An expressivity argument cannot be the explanation, because Muon (same architecture, same frozen attention) DOES reach degree 4 — so it is NOT a representability wall; it is an optimization/conditioning effect. Citing expressivity here would be *wrong* (contradicted by the Muon arm). The only honest math is a *partial/conditioning* statement: frozen random attention still provides enough fixed cross-position mixing to be representationally sufficient, and whether the MLP can exploit it is optimizer-dependent. See sketch §C. |
| C4 | "induction emerges in a ~3-dimensional update subspace ... the idealized theory's directions are not detected (captured variance below the random baseline in 62/63 runs)" (main.tex:257-270; findings-010 P1/P3) | **(a)** For the ~3-D effective dimension: an argument that the induction/copy solution is parameterized by O(1) functionally-relevant directions (a low-rank update implementing a QK "match-previous-token" + OV "copy" circuit), so the emergence-window Δθ concentrates in a few directions. **(b)** For the null: an argument that the published ansatz is derived in a *disentangled/idealized* coordinate frame whose linear transport into this architecture's (LayerNorm'd, residual, learned-embedding) basis is not norm-preserving / not even well-defined, so capture≈0 is the *expected* outcome of a frame mismatch — i.e. the null licenses "directions not recovered here", nothing stronger. | **(a) Plausible-with-work; (b) Tractable-now as a caveat, not a theorem** | **Caveat only.** The paper ALREADY states this correctly ("directions not detected", mapping-validity caveat). A formal argument for (a) is a nice-to-have; (b) is a *negative* result and math can only explain why a null is uninformative, not corroborate a positive claim. | **MEDIUM.** (a) is defensible but generic (low-rank circuit ⇒ few directions); the "exactly 3" is not something math will reproduce. (b) is the honest point but it *weakens* rather than strengthens — formalizing it mainly documents that the leg licenses little. Do NOT dress the null as theory-backed. See sketch §D. |
| C5 | "changing the optimizer advances the emergence TIME (1.6–3.7x) ... while the emergence geometry is unchanged" (main.tex:293-305; findings-007 P1) | **Time-rescaling corollary of C2.** If order/geometry is a target+architecture property (C2) and the optimizer acts (to first order) as a per-mode learning-rate map, then the emergence step is a monotone reparameterization of a shared underlying trajectory ⇒ "same geometry, rescaled time." | **Plausible-with-work** (rides on C2) | **Yes, as a one-line corollary** of the C2 mechanism, not a standalone theorem. | **MEDIUM.** Same caveat as C2 (orthogonalization ≠ pure rescaling). The *measured* 1.6–3.7× ratio is not predicted by any clean theorem; present as consistency, not derivation. |

Legend: Tractable-now = a correct, citable argument can be written this week with existing
theory; Plausible-with-work = a defensible argument exists but needs careful statement /
possibly a small additional experiment to match hypotheses; Hard/Open = a clean theorem is
not currently available.

---

## Concrete sketch §A — the depth/expressivity case (claim C1), the flagship

**What theorem shape would corroborate C1, and what it would need.**

*Target.* `chi_{S_k}(x) = prod_{i in S_k} x_i`, `x_i in {-1,+1}`, `|S_k|=k`, degrees k=1..4
over L=16 positions, disjoint sets. This is exactly the merged-staircase / "leap" target of
the Abbe-line theory and a generalized parity (degree-k parity = product of k bits).

*Desired statement (efficient-representation form, the defensible one):*
> **Proposition (shape).** Let F_1 be the function class realized by the 1-layer model in
> this readout (one pre-norm block + final-position scalar head) with width d, and F_2 the
> 2-layer class. There exist constants such that representing `chi_{S_4}` to L2 error < eps
> on the uniform cube requires width d = Omega(phi(eps)) in F_1 but only d = O(poly) in F_2,
> where the separation phi grows as the interaction order exceeds what one mixing stage can
> compose.

*Why depth helps, concretely (the mechanism a referee will accept):* a degree-k monomial is
a k-WISE MULTIPLICATIVE interaction across positions. A single attention layer composes
pairwise (token-token) interactions via the QK bilinear form, and the subsequent MLP can
square/threshold those; one full block can thus comfortably build degree-2 and, with width,
low-degree terms. To assemble a degree-4 product over 4 distinct positions, the cleanest
route is "compose two degree-2 products" — i.e. a second mixing stage that multiplies the
outputs of the first. This is the standard depth-vs-degree composition intuition
(achievable polynomial degree roughly doubles per multiplicative layer). The empirical
fingerprint matches: 1 layer learns degrees 1-2 and fails 3-4; 2 layers reach 4.

**What it would NEED (and the honest obstacles):**
1. The model uses **softmax** attention + **LayerNorm** + **GELU**, all non-polynomial and
   (formally) universal approximators at finite L. A literal "1 layer CANNOT represent
   degree 4" theorem is therefore most likely **false** — finite-width universal
   approximation will represent any function on the 2^16 cube given enough width. So the
   correct claim is a **width/efficiency separation** (1 layer needs blow-up; 2 layers are
   efficient), or a **learnability under gradient flow** separation, NOT a representability
   impossibility. The paper's own language ("capacity threshold", fit 0.71 vs 0.91) is
   compatible with the efficiency framing.
2. To make it rigorous you would either (a) cite an existing depth-separation result for
   transformers/threshold-circuits on parities and *adapt* the architecture mapping, or
   (b) prove a clean lemma for a SIMPLIFIED 1-layer model (e.g. linear attention or
   mean-pool + MLP), state it for that surrogate, and present the softmax result as the
   empirical confirmation. Option (b) is the realistic path: prove the separation for a
   linear-attention / single-mixing surrogate, then argue softmax inherits the *learning*
   difficulty empirically.
3. **Relevant named results to verify before citing:**
   - **Sanford, Hsu, Telgarsky** (2023-24) — transformer representational
     separations / "one self-attention layer" limits and depth/width tradeoffs for
     sequence functions; the closest existing depth-separation machinery. *Verify the exact
     class and whether it covers cross-position products.*
   - **TC^0 / threshold-circuit depth lower bounds for PARITY** (classical Hastad/parity ∉
     AC^0; bounded-depth transformers ⊆ TC^0 type results, e.g. Merrill-Sabharwal). Parity
     of k bits is the canonical hardness object and `chi_{S_k}` IS a (signed) parity. *This
     is the strongest formal anchor: degree-L parity is provably hard for bounded-depth
     threshold circuits; our degrees are only up to 4, so the finite-degree version is
     weaker and may not bite at k=4 — verify whether the bound is asymptotic in k.*
   - **Telgarsky depth-separation** (general deep-vs-shallow) for the "composition doubles
     degree" backbone.
   *Caution:* most of these are asymptotic in input dimension; at L=16, k=4 the constants
   may not produce a real separation, which is precisely why 2 layers *do* eventually fail
   too at larger scale (paper's own §3.6 scale audit shows 2-layer recovery is
   seed/optimizer-sensitive at L=24). That scale-fragility is actually *consistent* with a
   constants-matter efficiency separation rather than a hard impossibility — a point worth
   making honestly.

**Verdict on §A:** include as the paper's *one* analytic contribution, framed as an
**efficiency / interaction-order argument** (degree-k = k-wise cross-position product;
depth composes products; one mixing stage is efficient only to low degree), explicitly
NOT as a representability impossibility. Pair it with a citation to a verified
depth/parity separation and present 2-layer scale-fragility as consistent with
"constants-limited efficiency," not contradiction.

---

## Concrete sketch §B — gradient-correlation ordering (claim C2)

**Mechanism.** Train `f_theta` by MSE onto `g = sum_k chi_{S_k}`. Decompose the residual in
the Walsh basis. Because `{chi_S}` is ORTHONORMAL under the uniform cube measure
(`probes.py:5-8` literally relies on this), the population gradient of the loss w.r.t. a
parameter that controls the degree-k component is proportional to the current Walsh
coefficient deficit on `S_k` weighted by how much of `chi_{S_k}` the current `f` already
expresses. At small init, `f ≈ 0`, so the *direct* driving signal for every degree is
present in magnitude 1 — BUT a model that builds high degree by COMPOSING low-degree
features has, at init, ~0 representational overlap with `chi_{S_4}`, so the *reachable*
gradient on the degree-4 direction is second-order until degree-2/3 features exist. This is
the merged-staircase / leap mechanism: low degrees seed high degrees, enforcing a low→high
ORDER independent of step-size schedule.

**Why optimizer-invariant (the C2 punchline, and its risk).** A diagonal preconditioner
(Adam) or a sign/normalization map rescales each coordinate's step but cannot create
gradient signal along a direction where the *reachable* gradient is ~0; it can only change
HOW FAST an already-driven mode moves. Hence: order set by target structure (invariant),
rate set by optimizer (timing). **Risk:** Muon orthogonalizes the *matrix-valued* update
(`U V^T` from the SVD), which is a non-diagonal, non-commuting transform — it can in
principle re-weight modes non-uniformly and is exactly what the idealized-Muon flattening
account 2603.00742 claims removes the bias. So the clean "preconditioning can't manufacture
zero-gradient signal" lemma is true for *diagonal* preconditioners but is NOT obviously true
for orthogonalization. The defensible framing: **the staircase order is a fixed point of the
target's orthonormal-basis geometry; spectral flattening is proven only for deep-LINEAR
singular modes and does not transfer to function-space Walsh degree** (this is literally
finding-004's law-boundary statement, and the empirical TOST is the evidence that the
transfer fails). So the math here EXPLAINS WHY the flattening account is scoped to linear
nets — a *boundary* argument — rather than proving Muon preserves order from scratch.

**Named results to verify:** Abbe-Boix-Adsera-Misiakiewicz **"merged staircase property"**
(2021-22) and **"leap complexity"** (2023) for the low→high SGD ordering on Boolean
functions; the deep-linear simultaneous-learning / spectral result (paper's
`dragutinovic2026usemuon` / 2603.00742) for the *scope* of the flattening claim;
distributional-simplicity-bias (`rende2024distributional` / 2410.19637) for the empirical
low-degree-first phenomenon. **Verify** that the leap/staircase theorems are stated for
SGD/GD and what they say (if anything) about preconditioned/orthogonalized updates — my
recollection is they do NOT cover Muon, which is precisely why C's experiment is novel.

**Verdict on §B:** include as a *mechanism paragraph* that (i) explains the low→high order
via orthonormal-basis gradient seeding (solid, cite leap/staircase) and (ii) frames
optimizer-invariance as a SCOPE statement about where the flattening theorem applies
(solid, and matches the paper's existing positioning). Do NOT claim a theorem that Muon
preserves the staircase.

---

## Concrete sketch §C — attention owns cross-position products (claim C3)

The tempting expressivity argument ("MLP is position-wise, so cross-position degree-4
products need attention's token mixing") is **mechanistically appealing but empirically
contradicted** within the paper itself: under **Muon**, attention-frozen still reaches
degree 4 (Δ_attn = +0.005; findings-017 Tier-2; main.tex:177-181). A representability wall
would hold for every optimizer. Therefore:
- The only correct math is a **conditional / partial** statement: frozen *random* attention
  already supplies a fixed (data-independent) cross-position mixing channel that is
  representationally *sufficient* for the MLP-on-top to fit degree 4 in principle; whether
  gradient descent actually *finds* that solution is optimizer-dependent (AdamW's diagonal
  conditioning gets stuck, Muon's orthogonalized update does not). This is an
  optimization/loss-landscape claim, not an expressivity claim.
- **Do not** present C3 as expressivity-backed. The honest message (already in the paper) is
  "AdamW-scoped component-ownership," and the math that *fits* is conditioning/trainability,
  for which no clean theorem exists at this scale (Hard/Open).

**Verdict on §C:** keep as empirical + a one-sentence mechanistic hypothesis explicitly
labeled "optimization, not representation." Adding expressivity math here would be a
*wrong* corroboration and a referee with the Muon table in hand would catch it.

---

## Concrete sketch §D — induction subspace null (claim C4)

- **Dimension (~3):** defensible low-rank-circuit argument — the copy/induction solution is
  an O(1)-parameter circuit (a previous-token QK match + an OV copy), so the
  functionally-relevant update lives in a few directions; PCA dim ~3 is consistent. This is
  generic (any low-rank target ⇒ few directions) and will not reproduce "exactly 3," so cite
  it as *consistency*, not derivation.
- **Null (ansatz directions not captured):** math can only show the null is *uninformative*
  unless the disentangled→standard mapping is norm-preserving. Formalizing the frame
  mismatch (LayerNorm rescaling + learned-embedding basis ⇒ the ansatz transport is not an
  isometry ⇒ projected variance ≈ 0 is the expected outcome of a basis mismatch) is the
  honest content — and it WEAKENS the leg, exactly as the paper already says
  (main.tex:262-270). 

**Verdict on §D:** no positive math. At most a footnote: "the ~3-D dimensionality is
consistent with an O(1)-parameter copy circuit; the capture null is not interpretable as
'directions wrong' absent a validated isometric transport of the ansatz." This is already
the paper's stance; formalizing it adds rigor to a *caveat*, not support to a *claim*.

---

## Bottom line / prioritized recommendations

1. **Highest value — write §A (depth = interaction-order efficiency).** *Effort: medium
   (a careful prose argument + 1 verified citation; a real lemma on a linear-attention
   surrogate is medium-high). Impact: high* — turns C's strongest empirical result (depth
   ≥ 2 owns high degree) into an approximation-theoretic *necessity-flavored* statement,
   which is exactly the analytic backbone Neurocomputing/Physica D reward. Frame as
   efficiency/interaction-order, NOT representability impossibility; anchor on a verified
   transformer depth-separation and/or TC^0-parity result; present 2-layer scale-fragility
   (§3.6) as consistent with a constants-limited separation.
2. **Second value — write §B (gradient-correlation ordering + flattening scope).** *Effort:
   medium. Impact: medium-high* — explains low→high order via orthonormal Walsh-basis
   gradient seeding (cite merged-staircase/leap) and reframes optimizer-invariance as a
   SCOPE boundary on the deep-linear flattening theorem (already the paper's positioning).
   This directly rationalizes the headline TOST equivalence (C2) and gives the C5 timing
   corollary for free.
3. **Do NOT add expressivity math to C3 (attention ownership).** It is contradicted by the
   paper's own Muon arm; the correct (and already-stated) framing is AdamW-scoped
   optimization, for which no clean theorem exists. Adding math here would be a *wrong*
   corroboration.
4. **C4 subspace: formalize only the caveat, claim nothing.** Math here documents that the
   null is uninformative (frame mismatch), which strengthens honesty, not the result.

**Trade-off summary.**

| Option | Pros | Cons |
|---|---|---|
| Add §A + §B as a short "Analysis" section | Gives SCI venue the analytic backbone; converts depth + order results into principled statements; rides theory the paper already cites | Real risk of over-claiming if representability vs efficiency / diagonal vs orthogonalized preconditioning distinctions are blurred; needs citation verification; a *clean* lemma is medium-high effort |
| Keep paper purely empirical (status quo) | Zero risk of a wrong theorem; honest hedging already strong | Reads as phenomenology; weaker fit to Physica D / Neurocomputing expectations; leaves the depth result as a black-box observation |
| Add a single surrogate lemma (linear-attention depth separation) + empirical-confirms-it framing | Most rigorous; a genuine theorem the architecture *approximates* | Highest effort; surrogate-vs-real gap must be argued; may invite "why not prove it for the actual softmax model" |

**Honesty flags (do not skip).**
- Every named theorem (Sanford-Hsu-Telgarsky, Merrill-Sabharwal TC^0, Abbe et al.
  merged-staircase/leap, Telgarsky depth separation) is from my training memory and MUST be
  read and verified before citation; in particular verify (i) whether any covers
  cross-position products / parity at FINITE small degree (k=4), and (ii) whether the
  staircase/leap theorems say anything about preconditioned or orthogonalized updates.
- The cleanest expressivity statements are asymptotic in input dimension; at L=16, k=4 the
  separation may be constants-only — which is *consistent* with the paper's observed
  2-layer scale-fragility but means "necessity" must be softened to "efficiency."
- C3 is the trap: an expressivity argument predicts an optimizer-invariant bottleneck and is
  falsified by the Muon arm in the paper's own Table 1. Resist adding math there.

## References (file:line)
- `papers/C/main.tex:103-125` — C2 staircase-index TOST equivalence claim + quantities.
- `papers/C/main.tex:151-188` — C1 depth threshold + C3 attention-ownership (AdamW-scoped) claims.
- `papers/C/main.tex:257-270` — C4 induction subspace dimension + ansatz null + mapping caveat.
- `papers/C/main.tex:293-305` — C5 optimizer-moves-timing + geometry-invariant claim.
- `papers/C/main.tex:316-336,451-468` — scale audit (2-layer recovery seed/optimizer-sensitive) relevant to C1 efficiency-vs-impossibility framing.
- `experiments/grokking/model.py:15-77` — exact architecture (softmax attn + GELU MLP + LN, depth 2, final-position readout) the expressivity argument must target.
- `experiments/degree_staircase/data.py:17-44,77-89` — exact target g = sum chi_{S_k}, disjoint degrees 1..4, ±1 cube, scalar logit-difference readout.
- `experiments/degree_staircase/probes.py:5-8,48-66` — orthonormality of Walsh basis (the load-bearing fact for the §B gradient-correlation argument).
- `experiments/arch_staircase/train_arch.py:35-54` — freeze-arm definition (attn={qkv,proj}, mlp={fc1,fc2}) grounding C3.
- `experiments/findings-004-degree-staircase.md:112-131` — n=15 TOST numbers for C2.
- `experiments/findings-017.md:93-138` — Tier-2 optimizer axis: C3 is AdamW-specific (the fact that kills the C3 expressivity route).
- `experiments/findings-010.md:107-132` — C4 capture-null optimizer-invariant; pca_dim optimizer-dependent.

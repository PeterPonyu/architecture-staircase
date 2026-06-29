"""Direction 004 — controlled boolean staircase dataset.

We build target functions over boolean (+/-1) token sequences whose Walsh /
monomial-degree decomposition is KNOWN by construction, so a per-degree probe
(probes.py) can read off how much of each degree the model has learned.

Encoding for the transformer
----------------------------
Each example is a length L+1 token sequence:
    [b_0, b_1, ..., b_{L-1}, EQ]
where each b_i in {0, 1} is the token id for a boolean variable whose +/-1
value is x_i = 2*b_i - 1 (so token 0 -> -1, token 1 -> +1), and EQ is a
trailing "=" marker token. Vocabulary is therefore {0, 1, EQ} -> vocab_size=3.
The model (GrokTransformer) predicts at the final (EQ) position only, exactly
like the grokking modular task. seq_len = L + 1.

Target function (staircase of monomials)
-----------------------------------------
We fix D disjoint "stages". Stage k (k = 1..D) owns a position subset S_k with
|S_k| = k drawn from a seeded permutation of the L positions. The k-th
monomial is the product of the +/-1 variables on S_k:
    chi_{S_k}(x) = prod_{i in S_k} x_i        (a degree-k Walsh character)
The staircase regression target is the equal-weight sum
    g(x) = sum_{k=1..D} chi_{S_k}(x)
This is the "msp"-style nested/incrementing-degree construction: g contains one
clean monomial of every degree 1..D. (profile="staircase".)

Other profiles
---------------
- profile="mixed": same monomial set, equal-weight sum (alias of staircase here;
  kept as a named knob so the grid can sweep weighting later).
- profile="pure": a single monomial of degree `pure_degree` only,
  g(x) = chi_{S}(x) with |S| = pure_degree. Used by the probe self-test
  (known function -> probe must return ~1.0 at that degree, ~0.0 elsewhere).

The classification/regression head choice
------------------------------------------
We pick REGRESSION (predict the scalar g(x)) as the simplest faithful choice
for the F1 design: the F1 probe is "model logit vs each Fourier feature
correlation", which is most directly a regression of a scalar model output onto
the +/-1 monomials. The GrokTransformer is reused UNMODIFIED (vocab_size=3
classification head with 2 active token logits); the scalar model function is
the logit difference  f(x) = logit[token=1] - logit[token=0]  (see probes.py).
Training regresses f(x) onto g(x) with MSE. No architecture change is needed.

Everything is deterministic given `seed` and produced in memory.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

# Token vocabulary: 0 -> x=-1, 1 -> x=+1, 2 -> EQ marker.
EQ_TOKEN = 2
VOCAB_SIZE = 3


@dataclass
class StaircaseSpec:
    """Fully describes a boolean staircase task (deterministic given seed)."""
    L: int = 16                 # number of boolean variables (positions)
    D: int = 4                  # highest degree / number of stages
    profile: str = "staircase"  # "staircase" | "mixed" | "pure"
    pure_degree: int = 2        # only used when profile == "pure"
    seed: int = 0

    @property
    def seq_len(self) -> int:
        return self.L + 1  # +1 for the trailing EQ marker

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE


def monomial_sets(spec: StaircaseSpec) -> dict[int, list[int]]:
    """Return {degree k -> sorted position subset S_k}, deterministic per seed.

    Stages are carved from a seeded permutation of the L positions so that the
    subsets are disjoint (a clean, non-overlapping staircase). Requires
    L >= 1+2+...+D = D(D+1)/2 for the disjoint construction.
    """
    if spec.profile == "pure":
        need = spec.pure_degree
        g = torch.Generator().manual_seed(spec.seed)
        perm = torch.randperm(spec.L, generator=g).tolist()
        assert need <= spec.L, f"pure_degree {need} exceeds L={spec.L}"
        return {need: sorted(perm[:need])}

    total = spec.D * (spec.D + 1) // 2
    assert spec.L >= total, (
        f"L={spec.L} too small for disjoint stages up to degree D={spec.D} "
        f"(need >= {total})"
    )
    g = torch.Generator().manual_seed(spec.seed)
    perm = torch.randperm(spec.L, generator=g).tolist()
    sets: dict[int, list[int]] = {}
    cursor = 0
    for k in range(1, spec.D + 1):
        sets[k] = sorted(perm[cursor:cursor + k])
        cursor += k
    return sets


def _target_from_signs(x: torch.Tensor, sets: dict[int, list[int]]) -> torch.Tensor:
    """g(x) = sum_k prod_{i in S_k} x_i over +/-1 inputs x: [N, L] -> [N]."""
    y = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for _, S in sets.items():
        mono = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        for i in S:
            mono = mono * x[:, i]
        y = y + mono
    return y


def sample_batch(spec: StaircaseSpec, n: int, seed: int, device: str = "cpu"):
    """Sample n boolean examples and their regression targets.

    Returns (X, y, signs):
      X      : LongTensor [n, L+1]  token ids ([b_0..b_{L-1}, EQ])
      y      : FloatTensor [n]      regression target g(x)
      signs  : FloatTensor [n, L]   the +/-1 variable values (for the probe)
    Deterministic given (spec, seed). In-memory, fresh-batch friendly.
    """
    g = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (n, spec.L), generator=g)  # {0,1}
    signs = (2 * bits - 1).float().to(device)             # {-1,+1}
    sets = monomial_sets(spec)
    y = _target_from_signs(signs, sets)
    eq = torch.full((n, 1), EQ_TOKEN, dtype=torch.long)
    X = torch.cat([bits, eq], dim=1).to(device)           # [n, L+1] token ids
    return X, y, signs


def enumerate_dataset(spec: StaircaseSpec, max_n: int = 8192, seed: int = 0,
                      device: str = "cpu"):
    """Full boolean cube if 2**L <= max_n, else a large deterministic sample.

    Used by the probe for accurate Walsh-coefficient (correlation) estimation.
    Returns the same (X, y, signs) triple as sample_batch.
    """
    if 2 ** spec.L <= max_n:
        n = 2 ** spec.L
        ar = torch.arange(n)
        # bit j of each integer -> position j
        bits = ((ar[:, None] >> torch.arange(spec.L)[None, :]) & 1).long()
        signs = (2 * bits - 1).float().to(device)
        sets = monomial_sets(spec)
        y = _target_from_signs(signs, sets)
        eq = torch.full((n, 1), EQ_TOKEN, dtype=torch.long)
        X = torch.cat([bits, eq], dim=1).to(device)
        return X, y, signs
    return sample_batch(spec, max_n, seed=seed, device=device)

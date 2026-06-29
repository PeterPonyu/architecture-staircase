"""Direction 004 — per-degree decomposition probe.

Given a scalar model function f(x) over boolean (+/-1) inputs and the KNOWN
monomial sets {k -> S_k}, we estimate how much of each degree-k Walsh character
the model has learned. Because the monomials are an orthonormal basis on the
uniform boolean cube (E[chi_S chi_T] = delta_{S,T}), the population correlation
of f with chi_{S_k} is just a normalized Walsh coefficient — cheap to estimate
by Monte-Carlo / full-cube averaging on the known sets only.

degree_correlations(f_vals, signs, sets)
    -> {k: corr(f, chi_{S_k})}   (Pearson correlation, in [-1, 1])

staircase_index(history)
    -> spread of per-degree HALF-LEARNING times (the step at which each
       degree's correlation first crosses half of its own final value). A large
       spread = a pronounced sequential "staircase"; a small spread = degrees
       learned together (the Muon-erases-the-staircase prediction).

model_scalar(model, X)
    f(x) = logit[token=1] - logit[token=0]  at the final position. This reuses
    GrokTransformer's vocab=3 head unmodified as a scalar regression output.

Run `python probes.py` for a self-test on a known pure-degree function.
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def model_scalar(model, X: torch.Tensor) -> torch.Tensor:
    """Scalar model function f(x) = logit[1] - logit[0] at the EQ position."""
    logits = model(X)              # [N, vocab]
    return (logits[:, 1] - logits[:, 0]).float()


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom < 1e-12:
        return 0.0
    return float(torch.dot(a, b) / denom)


def chi(signs: torch.Tensor, S) -> torch.Tensor:
    """Walsh character chi_S(x) = prod_{i in S} x_i over +/-1 signs [N, L]."""
    out = torch.ones(signs.shape[0], device=signs.device, dtype=signs.dtype)
    for i in S:
        out = out * signs[:, i]
    return out


def degree_correlations(f_vals: torch.Tensor, signs: torch.Tensor,
                        sets: dict) -> dict:
    """Pearson correlation of model output f with each known monomial chi_{S_k}.

    f_vals : [N] scalar model outputs
    signs  : [N, L] the +/-1 variable values
    sets   : {degree k -> position subset S_k}
    Returns {k: correlation in [-1, 1]}.
    """
    f = f_vals.float()
    return {k: _pearson(f, chi(signs.float(), S)) for k, S in sorted(sets.items())}


def half_learning_times(history: list, degrees: list) -> dict:
    """First step at which |corr_k| reaches half its OWN final |value|.

    history: list of dicts each containing "step" and "deg_corr" = {k: corr}.
    Returns {k: step or None}. None means the degree never crossed its own
    half-final level (e.g. censored / never learned).
    """
    if not history:
        return {k: None for k in degrees}
    finals = {k: abs(history[-1]["deg_corr"].get(str(k),
                     history[-1]["deg_corr"].get(k, 0.0)))
              for k in degrees}
    out: dict = {}
    for k in degrees:
        target = 0.5 * finals[k]
        hit = None
        for rec in history:
            dc = rec["deg_corr"]
            val = abs(dc.get(str(k), dc.get(k, 0.0)))
            if target > 1e-6 and val >= target:
                hit = rec["step"]
                break
        out[k] = hit
    return out


def staircase_index(history: list, degrees: list) -> dict:
    """Quantify the by-degree sequential staircase.

    Returns a dict with:
      half_times    : {k: step or None}
      spread        : (max - min) of defined half-times  (steps); the headline
                      staircase-ness number. Larger => more sequential.
      span_ratio    : max/min of defined half-times (scale-free); 1.0 => no
                      staircase, large => strong staircase.
      n_learned     : how many degrees ever crossed their half-final level.
    A large spread/span_ratio = pronounced AdamW/SGDM-style staircase; values
    pressed toward 0 / 1.0 = the Muon-erases-the-staircase signature.
    """
    ht = half_learning_times(history, degrees)
    defined = [v for v in ht.values() if v is not None]
    if len(defined) >= 2:
        spread = float(max(defined) - min(defined))
        lo = max(min(defined), 1)
        span_ratio = float(max(defined) / lo)
    else:
        spread = 0.0
        span_ratio = 1.0
    return {
        "half_times": ht,
        "spread": spread,
        "span_ratio": span_ratio,
        "n_learned": len(defined),
    }


# ---------------------------------------------------------------------------
# Self-test: on a KNOWN function (pure degree-d monomial used as the "model
# output"), the probe must return ~1.0 at degree d and ~0.0 at every other
# degree. Run with `python probes.py`.
# ---------------------------------------------------------------------------
def _self_test() -> int:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import StaircaseSpec, enumerate_dataset, monomial_sets

    # A staircase spec gives us monomials of degrees 1..D to probe against.
    # Use L=12 so the FULL boolean cube (2**12 = 4096) is enumerated -> the
    # Walsh basis is exactly orthonormal and off-diagonal correlations are
    # exactly 0 (no Monte-Carlo finite-sample noise).
    spec = StaircaseSpec(L=12, D=4, profile="staircase", seed=0)
    sets = monomial_sets(spec)
    _, _, signs = enumerate_dataset(spec, max_n=8192, seed=0)
    assert signs.shape[0] == 2 ** spec.L, "self-test expects full-cube enumeration"

    # "Model output" := the pure degree-2 monomial chi_{S_2} (a KNOWN function).
    target_deg = 2
    f_known = chi(signs.float(), sets[target_deg])

    corrs = degree_correlations(f_known, signs, sets)
    print("SELF-TEST per-degree correlations (model output = pure degree-2):")
    for k, c in corrs.items():
        print(f"  degree {k}: corr = {c:+.4f}")

    ok = True
    for k, c in corrs.items():
        if k == target_deg:
            if abs(c - 1.0) > 1e-6:
                ok = False
                print(f"  FAIL: degree {k} corr {c:.6f} not ~1.0")
        else:
            if abs(c) > 1e-6:
                ok = False
                print(f"  FAIL: degree {k} corr {c:.6f} not ~0.0")

    # Bonus: an equal-weight sum of all monomials should correlate ~1/sqrt(D)
    # with each degree (orthonormal basis), confirming separability of stages.
    f_sum = sum(chi(signs.float(), S) for S in sets.values())
    corrs_sum = degree_correlations(f_sum, signs, sets)
    expected = 1.0 / math.sqrt(len(sets))
    print(f"SELF-TEST staircase target corr (expect ~{expected:.4f} each):")
    for k, c in corrs_sum.items():
        print(f"  degree {k}: corr = {c:+.4f}")
        if abs(abs(c) - expected) > 0.02:
            ok = False
            print(f"  FAIL: degree {k} corr {c:.4f} not ~{expected:.4f}")

    print("PROBE SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())

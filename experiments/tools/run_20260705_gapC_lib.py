"""Shared machinery for the paper-C pre-submission gap battery (2026-07-05).

This module is NON-INVASIVE: it imports the closed staircase harnesses
(degree_staircase/train_staircase.py, arch_staircase/train_arch.py,
arch_staircase/run_arch_optaxis.py) and NEVER modifies them. It supplies two
capabilities the harnesses do not expose, both needed by the C gap experiments:

  1. custom_freeze_run() -- a field-for-field mirror of train_arch.run that
     freezes an EXPLICIT set of parameter-name marks (not just the built-in
     none/attn/mlp arms). Used for C#2 (BOTH-frozen floor control + per-layer
     knockouts). Reuses train_arch primitives (build_model / sample_batch /
     evaluate / staircase_index) so the logged schema is byte-for-byte the same
     as train_arch.run -> the analyzers and other papers' tooling still apply.
     The optimizer over the surviving trainable params is the muon/sgdm/adamw
     hybrid from run_arch_optaxis.build_opt_trainable_any (imported, not copied).

  2. target_override() -- a context manager that monkeypatches the data module's
     monomial_sets + _target_from_signs (and train_staircase's imported
     monomial_sets name) so a run can use a CUSTOM degree set and/or a CUSTOM
     per-degree coefficient weighting a_k. The degree_staircase target is an
     EQUAL-weight sum of one monomial per degree; there is no numeric weighting
     knob in data.py (only the profile="mixed" alias), so C#1 (target-family
     generality) adds it here by monkeypatch. The per-degree Walsh probe is
     scale/weight invariant, so all logged metrics remain comparable.

Every run writes one .jsonl ending in a {"_summary": ...} marker line
(resume marker; the analyzer refuses marker-less/incomplete files per
LESSONS-AND-ERRATA sec 6c).
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F

# --- import the closed harnesses (path discipline mirrors run_arch_optaxis) ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP_DIR = os.path.dirname(_THIS_DIR)
_ARCH_DIR = os.path.join(_EXP_DIR, "arch_staircase")
_DS_DIR = os.path.join(_EXP_DIR, "degree_staircase")
for _d in (_ARCH_DIR, _DS_DIR, _EXP_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import train_arch as TA            # noqa: E402  (017 arch trainer; read-only)
TS = TA.TS                         # closed 004 staircase trainer (train_staircase)
import data as DATA                # noqa: E402  (degree_staircase data module)
import probes as PROBES            # noqa: E402  (degree_staircase probe module)
# importing run_arch_optaxis installs the muon/sgdm/adamw builder onto TA and
# gives us the trainable-only optimizer factory used for the freeze arms.
import run_arch_optaxis as ROA     # noqa: E402


RESULTS_ROOT = os.path.join(_EXP_DIR, "results", "ieee_gap_20260705", "C")


# ---------------------------------------------------------------------------
# resume marker helpers (LESSONS-AND-ERRATA sec 6c: existence != completeness)
# ---------------------------------------------------------------------------
def already_done(path: str) -> bool:
    """True iff the file exists AND its last non-empty line has a _summary."""
    if not os.path.exists(path):
        return False
    last = ""
    with open(path) as fh:
        for line in fh:
            if line.strip():
                last = line
    try:
        return "_summary" in json.loads(last)
    except (json.JSONDecodeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# 1. custom-freeze runner (C#2) -- explicit param-mark freeze, arch schema
# ---------------------------------------------------------------------------
def _apply_marks(model, marks) -> tuple[int, int]:
    """requires_grad_(False) on every param whose dotted name contains a mark."""
    n_frozen = 0
    for name, p in model.named_parameters():
        if any(m in "." + name + "." for m in marks):
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_trainable, n_frozen


def custom_freeze_run(cfg, marks, arm_name: str, out_path: str | None = None):
    """Mirror of train_arch.run with an EXPLICIT freeze mark-set.

    cfg      : TA.ArchConfig (carries optimizer / d_model / n_layers / D / seed)
    marks    : tuple of dotted param-name substrings to freeze (may be empty)
    arm_name : label stored in _meta/_summary (e.g. 'both', 'l0attn')
    Logged fields are identical to train_arch.run plus 'freeze_marks'/'arm'.
    """
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    spec = TS.make_spec(cfg)

    model = TS.build_model(cfg, spec, device)
    n_trainable, n_frozen = _apply_marks(model, marks)
    assert n_trainable > 0, f"arm {arm_name}: freeze left no trainable params"
    optimizers = ROA.build_opt_trainable_any(model, cfg)
    degrees = sorted(TS.monomial_sets(spec).keys())

    history: list = []
    t0 = time.time()
    f = open(out_path, "w") if out_path else None
    if f:
        f.write(json.dumps({"_meta": {**asdict(cfg), "arm": arm_name,
                                      "freeze_marks": list(marks),
                                      "n_trainable": n_trainable,
                                      "n_frozen": n_frozen}}) + "\n")

    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0:
            mse, fit_corr, deg_corr = TS.evaluate(model, spec, cfg, device)
            rec = {"step": step, "mse": mse, "fit_corr": fit_corr,
                   "deg_corr": {str(k): v for k, v in deg_corr.items()}}
            si = TS.staircase_index(history + [rec], degrees)
            rec["staircase_spread"] = si["spread"]
            rec["staircase_span_ratio"] = si["span_ratio"]
            history.append(rec)
            if f:
                f.write(json.dumps(rec) + "\n")
                f.flush()

        model.train()
        Xb, yb, _ = TS.sample_batch(spec, cfg.batch_size,
                                    seed=step * 100003 + cfg.seed, device=device)
        loss = F.mse_loss(TS.model_scalar_train(model, Xb), yb)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()

    elapsed = time.time() - t0
    final_si = TS.staircase_index(history, degrees)
    summary = {
        **asdict(cfg), "arm": arm_name, "freeze_marks": list(marks),
        "final_mse": history[-1]["mse"],
        "final_fit_corr": history[-1]["fit_corr"],
        "final_deg_corr": history[-1]["deg_corr"],
        "staircase_spread": final_si["spread"],
        "staircase_span_ratio": final_si["span_ratio"],
        "half_times": {str(k): v for k, v in final_si["half_times"].items()},
        "n_learned": final_si["n_learned"],
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_trainable": n_trainable, "n_frozen": n_frozen,
        "elapsed_sec": elapsed, "stopped_step": history[-1]["step"],
    }
    if f:
        f.write(json.dumps({"_summary": summary}) + "\n")
        f.close()
    return summary, history


# per-layer / both freeze mark-sets (model uses blocks.<i>.{qkv,proj,fc1,fc2})
def freeze_marks(arm: str) -> tuple:
    table = {
        "none": (),
        "attn": TA.ATTN_MARKS,
        "mlp": TA.MLP_MARKS,
        "both": TA.ATTN_MARKS + TA.MLP_MARKS,
        "l0attn": (".blocks.0.qkv.", ".blocks.0.proj."),
        "l1attn": (".blocks.1.qkv.", ".blocks.1.proj."),
        "l0mlp": (".blocks.0.fc1.", ".blocks.0.fc2."),
        "l1mlp": (".blocks.1.fc1.", ".blocks.1.fc2."),
    }
    return table[arm]


# ---------------------------------------------------------------------------
# 2. target-family override (C#1) -- custom degree set + per-degree weights
# ---------------------------------------------------------------------------
def build_custom_sets(degrees, L: int, seed: int) -> dict:
    """Disjoint position subsets sized by each requested degree (seeded)."""
    if sum(degrees) > L:
        raise ValueError(f"degrees {degrees} need {sum(degrees)} positions > L={L}")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(L, generator=g).tolist()
    sets, cur = {}, 0
    for k in degrees:
        sets[k] = sorted(perm[cur:cur + k])
        cur += k
    return sets


@contextlib.contextmanager
def target_override(degrees, weight_fn):
    """Monkeypatch data + train_staircase so runs use custom degrees/weights.

    degrees   : iterable of degree ints (e.g. [1,3,4] for a gapped staircase)
    weight_fn : callable k -> a_k coefficient (e.g. lambda k: 2.0 ** -k)
    Patches: data.monomial_sets, data._target_from_signs, TS.monomial_sets.
    The Walsh probe (degree_correlations) is weight-invariant, so all metrics
    stay comparable. Restores originals on exit (defensive; each cell is a fresh
    process anyway).
    """
    deg_list = list(degrees)
    orig_ms_data = DATA.monomial_sets
    orig_tfs = DATA._target_from_signs
    orig_ms_ts = TS.monomial_sets

    def patched_monomial_sets(spec):
        return build_custom_sets(deg_list, spec.L, spec.seed)

    def patched_target(x, sets):
        y = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for k, S in sets.items():
            mono = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
            for i in S:
                mono = mono * x[:, i]
            y = y + float(weight_fn(k)) * mono
        return y

    DATA.monomial_sets = patched_monomial_sets
    DATA._target_from_signs = patched_target
    TS.monomial_sets = patched_monomial_sets
    try:
        yield
    finally:
        DATA.monomial_sets = orig_ms_data
        DATA._target_from_signs = orig_tfs
        TS.monomial_sets = orig_ms_ts


# named C#1 target families: (degree set, per-degree weight fn)
TARGET_FAMILIES = {
    # geometric decay a_k = 2^-k over the standard 1..4 staircase
    "geom": ([1, 2, 3, 4], lambda k: 2.0 ** (-k)),
    # gapped staircase, degrees {1,3,4}, equal weights (degree 2 absent)
    "gapped": ([1, 3, 4], lambda k: 1.0),
    # inverse weighting a_k = 2^k (high degrees dominate the target)
    "invw": ([1, 2, 3, 4], lambda k: 2.0 ** k),
}


def staircase_run(cfg, family: str, out_path: str | None = None):
    """train_staircase.run under a C#1 target family (custom degrees/weights)."""
    degrees, wfn = TARGET_FAMILIES[family]
    with target_override(degrees, wfn):
        summary, history = TS.run(cfg, out_path=out_path)
    # tag the family into the summary line for the analyzer (append, non-clobber)
    if out_path is not None:
        summary["target_family"] = family
        summary["target_degrees"] = degrees
    return summary, history

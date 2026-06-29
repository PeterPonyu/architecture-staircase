"""Direction 010 — induction-subspace trainer (online fresh-batch, FIXED AdamW).

One run = one (task-variant, seq_len, seed) configuration. Trains the induction
task with REAL minibatch / softmax-attention / cross-entropy / AdamW dynamics
and, at every eval, instruments the per-eval Q/K/OV weight UPDATE against the 3
canonical induction-circuit directions (subspace.py): how much of the update's
energy is captured per direction and in total, plus the PCA effective dimension
of the cumulative-Δθ trajectory — the empirical test of arXiv:2511.01033's
idealized "3-D subspace" claim under real (non-idealized) training.

INTERLOCK DISCIPLINE with direction 007: the optimizer is FIXED to AdamW. This
study varies the TASK and probes the SUBSPACE; it does NOT compare optimizers
(that is 007's job). Keeping AdamW fixed isolates the subspace question from the
optimizer-geometry question.

Reuses induction_emergence WHOLESALE by import (those files are NOT modified):
  data.py   — InductionSpec, sample_batch, batch_seed (+ markov variant local)
  probes.py — position_accuracies, icl_score, prefix_match_score, detect_emergence
  model.py  — SeqTransformer (→ grokking GrokTransformer, resolved from its dir)
The NEW machinery (3 canonical directions, projection, PCA) lives in subspace.py.

Per-eval instrumentation
------------------------
  * snapshot Q/K/V/O + unembed weights; Δ = current − last-snapshot;
  * captured_fraction(Δ, {D1,D2,D3}) -> per-direction + total fractions;
  * append cumulative Δθ (vs init) -> PCA effective dim over the trajectory;
  * icl_score, prefix_match (final layer), emergence detection on the ICL curve.

Flags
-----
--smoke : print the labeled smoke lines, run <=1 step, write NO files, exit 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict

# Import discipline: LOCAL FIRST (subspace.py), induction_emergence APPENDED
# (model/data/probes reused wholesale; chain SeqTransformer→GrokTransformer
# resolves inside induction_emergence/model.py).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
_IE_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "induction_emergence"))
if _IE_DIR not in sys.path:
    sys.path.append(_IE_DIR)

import torch
import torch.nn.functional as F

from data import InductionSpec, sample_batch, batch_seed              # noqa: E402
from model import SeqTransformer                                       # noqa: E402
from probes import (                                                   # noqa: E402
    position_accuracies, icl_score, prefix_match_score, detect_emergence,
)
from subspace import (                                                 # noqa: E402
    build_canonical_directions, captured_fraction, snapshot_weights,
    weight_delta, flatten_delta, pca_effective_dim,
)


@dataclass
class Config:
    # task / data
    task: str = "repeat"         # "repeat" (repeated-segment) | "markov"
    vocab_size: int = 64         # V
    seq_len: int = 128           # L
    period: int = 0              # repeated-segment length; 0 => auto (L // 4)
    batch_size: int = 64         # fresh batch per step (online)
    eval_batch: int = 256        # held-out eval batch (fixed stream)
    n_eval_batches: int = 1      # eval batches averaged per eval
    # model (grokking spec: 2 layers, 4 heads, d=128)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4
    init_scale: float = 1.0
    # optimization — FIXED AdamW (interlock with direction 007; no opt comparison)
    lr: float = 1e-3
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.98
    steps: int = 10000
    eval_every: int = 100
    emergence_threshold: float = 0.5
    pca_var_threshold: float = 0.9   # explained-variance level for effective dim
    seed: int = 0
    device: str = "cuda"


def make_spec(cfg: Config) -> InductionSpec:
    return InductionSpec(vocab_size=cfg.vocab_size, seq_len=cfg.seq_len,
                         period=cfg.period)


def build_model(cfg: Config, spec: InductionSpec, device: str) -> SeqTransformer:
    return SeqTransformer(
        vocab_size=spec.vocab_size, seq_len=spec.seq_len, d_model=cfg.d_model,
        n_heads=cfg.n_heads, n_layers=cfg.n_layers, mlp_ratio=cfg.mlp_ratio,
        init_scale=cfg.init_scale,
    ).to(device)


def build_optimizer(model, cfg: Config):
    """FIXED AdamW over ALL parameters (interlock with 007: no optimizer arm)."""
    return torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay)


# ---------------------------------------------------------------------------
# Task variants. "repeat" reuses induction_emergence.sample_batch unchanged.
# "markov" is a local variant: a periodic backbone perturbed so the induction
# rule still holds on repeats but the surface statistics differ (varies the task
# without touching the reused data.py).
# ---------------------------------------------------------------------------
def sample_task(cfg: Config, spec: InductionSpec, n: int, seed: int, device: str):
    if cfg.task == "repeat":
        return sample_batch(spec, n, seed=seed, device=device)
    elif cfg.task == "markov":
        return _sample_markov(spec, n, seed=seed, device=device)
    raise ValueError(f"unknown task {cfg.task!r}")


def _sample_markov(spec: InductionSpec, n: int, seed: int, device: str):
    """Markov-bigram induction variant: each token's successor is a FIXED random
    function of the token (a per-sequence bigram map), so the induction rule
    "copy the token that followed the previous occurrence of x[t]" is still exact
    on repeat positions, but the sequence is a bigram walk rather than a tiled
    segment — a genuinely different surface task with the same induction signal.

    Returns (X, Y, repeat_mask, target_mask) with the SAME contract as
    data.sample_batch. Deterministic given seed.
    """
    g = torch.Generator().manual_seed(int(seed) & 0x7FFF_FFFF)
    L, V = spec.seq_len, spec.vocab_size
    # per-sequence bigram successor map: succ[b, v] = next token after value v.
    succ = torch.randint(0, V, (n, V), generator=g)
    start = torch.randint(0, V, (n,), generator=g)
    X = torch.empty(n, L, dtype=torch.long)
    X[:, 0] = start
    for t in range(1, L):
        X[:, t] = torch.gather(succ, 1, X[:, t - 1:t]).squeeze(1)
    Y = torch.empty_like(X)
    Y[:, :-1] = X[:, 1:]
    Y[:, -1] = X[:, -1]
    target_mask = torch.ones(n, L, dtype=torch.bool)
    target_mask[:, -1] = False
    # repeat = token value seen earlier (induction-predictable under the bigram map)
    eq = X[:, :, None] == X[:, None, :]
    earlier = torch.tril(torch.ones(L, L, dtype=torch.bool), diagonal=-1)
    repeat_mask = (eq & earlier[None]).any(dim=2) & target_mask
    return (X.to(device), Y.to(device),
            repeat_mask.to(device), target_mask.to(device))


def make_eval_batches(cfg: Config, spec: InductionSpec, device: str):
    """Fixed held-out eval stream (constant across steps/runs of same cfg)."""
    batches = []
    for j in range(cfg.n_eval_batches):
        bs = batch_seed(50_000_000 + cfg.seed, j)
        batches.append(sample_task(cfg, spec, cfg.eval_batch, seed=bs, device=device))
    return batches


@torch.no_grad()
def evaluate(model, eval_batches, layer: int = -1) -> dict:
    """Average per-role acc/loss + ICL score + per-head prefix-match."""
    keys = ["loss", "repeat_acc", "first_acc", "repeat_loss", "first_loss"]
    agg: dict[str, float] = {k: 0.0 for k in keys}
    pm_sum = torch.zeros(0)
    for batch in eval_batches:
        pa = position_accuracies(model, batch)
        for k in keys:
            agg[k] += pa[k]
        ph = torch.tensor(prefix_match_score(model, batch, layer=layer)["per_head"])
        pm_sum = ph if pm_sum.numel() == 0 else pm_sum + ph
    nb = max(1, len(eval_batches))
    for k in keys:
        agg[k] /= nb
    per_head = [float(x / nb) for x in pm_sum.tolist()]
    out: dict = dict(agg)
    out["icl_score"] = agg["repeat_acc"] - agg["first_acc"]
    out["prefix_match_per_head"] = per_head
    out["prefix_match_max"] = max(per_head) if per_head else 0.0
    return out


def run(cfg: Config, out_path: str | None = None):
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    spec = make_spec(cfg)

    model = build_model(cfg, spec, device)
    optimizer = build_optimizer(model, cfg)
    eval_batches = make_eval_batches(cfg, spec, device)

    # canonical directions are rebuilt at each eval from the CURRENT weights
    # (they depend on the learned embeddings / OV map, per the progress measures).
    init_snap = snapshot_weights(model)
    last_snap = init_snap
    cum_deltas: list = []                          # cumulative-Δθ vectors (vs init)

    history: list = []
    t0 = time.time()
    f = open(out_path, "w") if out_path else None
    if f:
        f.write(json.dumps({"_meta": asdict(cfg)}) + "\n")

    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0:
            model.eval()
            ev = evaluate(model, eval_batches)

            curr_snap = snapshot_weights(model)
            directions = build_canonical_directions(model)
            step_delta = weight_delta(last_snap, curr_snap)    # update since last eval
            cf = captured_fraction(step_delta, directions)
            cum_delta = weight_delta(init_snap, curr_snap)     # cumulative vs init
            cum_deltas.append(flatten_delta(cum_delta))
            pca = pca_effective_dim(cum_deltas, var_threshold=cfg.pca_var_threshold)
            last_snap = curr_snap

            rec = {
                "step": step, **ev,
                "captured_D1": cf["D1"], "captured_D2": cf["D2"],
                "captured_D3": cf["D3"], "captured_total": cf["total"],
                "pca_dim": pca["n_components"],
                "pca_participation": pca["participation"],
            }
            history.append(rec)
            if f:
                f.write(json.dumps(rec) + "\n")
                f.flush()

        # online fresh-batch causal-LM step (FIXED AdamW)
        model.train()
        Xb, Yb, _, tmask = sample_task(
            cfg, spec, cfg.batch_size, seed=batch_seed(cfg.seed, step), device=device)
        logits = model(Xb)
        V = logits.shape[-1]
        ce = F.cross_entropy(logits.reshape(-1, V), Yb.reshape(-1),
                             reduction="none").reshape(Yb.shape)
        loss = (ce * tmask).sum() / tmask.sum().clamp(min=1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    elapsed = time.time() - t0
    steps_seq = [r["step"] for r in history]
    icl_seq = [r["icl_score"] for r in history]
    em = detect_emergence(steps_seq, icl_seq, threshold=cfg.emergence_threshold)
    summary = {
        **asdict(cfg),
        "final_icl_score": history[-1]["icl_score"],
        "final_repeat_acc": history[-1]["repeat_acc"],
        "final_prefix_match_max": history[-1]["prefix_match_max"],
        "final_captured_total": history[-1]["captured_total"],
        "final_pca_dim": history[-1]["pca_dim"],
        "emergence_step": em["emergence_step"],
        "emergence_max_slope": em["max_slope"],
        "emergence_transition_width": em["transition_width"],
        "n_params": sum(p.numel() for p in model.parameters()),
        "elapsed_sec": elapsed,
        "stopped_step": history[-1]["step"],
    }
    if f:
        f.write(json.dumps({"_summary": summary}) + "\n")
        f.close()
    return summary, history


# ---------------------------------------------------------------------------
# Smoke: labeled lines, <=1 training step, NO files, exit 0.
# ---------------------------------------------------------------------------
def run_smoke():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cfg = Config(task="repeat", vocab_size=64, seq_len=128, batch_size=64, seed=0)
    spec = make_spec(cfg)

    # 1. one online batch shape
    X, Y, repeat_mask, target_mask = sample_task(cfg, spec, cfg.batch_size,
                                                 seed=0, device=device)
    print(f"SMOKE DATASET SHAPE: X={tuple(X.shape)}, Y={tuple(Y.shape)}, "
          f"repeat_mask={tuple(repeat_mask.shape)}")

    # 2. param count
    model = build_model(cfg, spec, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SMOKE PARAM COUNT: {n_params}")

    # 3. forward + loss (full-sequence causal LM)
    logits = model(X)
    V = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, V), Y.reshape(-1),
                         reduction="none").reshape(Y.shape)
    loss = (ce * target_mask).sum() / target_mask.sum().clamp(min=1)
    print(f"SMOKE FORWARD LOSS: {loss.item():.6f}")

    # 4. one AdamW step, snapshotting weights around it for the subspace probe
    optimizer = build_optimizer(model, cfg)
    pre = snapshot_weights(model)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print("SMOKE OPTIMIZER STEP: OK")

    # 5. bonus: subspace projection machinery end-to-end on this REAL update delta
    #    (untrained step -> captured expected small; pca on a 2-point trajectory).
    directions = build_canonical_directions(model)
    delta = weight_delta(pre, snapshot_weights(model))
    cf = captured_fraction(delta, directions)
    delta_flat = flatten_delta(delta)
    pca = pca_effective_dim([torch.zeros_like(delta_flat), delta_flat],
                            var_threshold=cfg.pca_var_threshold)
    print(f"SMOKE SUBSPACE PROBE: captured={cf['total']:.4f} "
          f"pca_dim={pca['n_components']}")


def parse_args() -> tuple[Config, bool]:
    ap = argparse.ArgumentParser(description="induction-subspace trainer (FIXED AdamW)")
    ap.add_argument("--smoke", action="store_true",
                    help="Run smoke checks and exit (no files written)")
    defaults = asdict(Config())
    for k, v in defaults.items():
        ap.add_argument(f"--{k}", type=type(v) if v is not None else str, default=v)
    a = vars(ap.parse_args())
    smoke = a.pop("smoke")
    cfg = Config(**a)
    return cfg, smoke


if __name__ == "__main__":
    cfg, smoke = parse_args()
    if smoke:
        run_smoke()
        sys.exit(0)
    summary, _ = run(cfg, out_path=None)
    print(json.dumps(summary, indent=2))

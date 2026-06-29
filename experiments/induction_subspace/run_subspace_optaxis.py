"""Direction 010 TIER-2 — the OPTIMIZER axis on the induction subspace.

WHY THIS BREAKS THE 007 INTERLOCK ON PURPOSE.
Same C-paper red-team motivation as arch_staircase/run_arch_optaxis.py: the
"geometry ⊥ optimizer" thesis needs the ~3-D-dimensionality and the
captured-fraction (010) measured under MORE than AdamW. This runner re-runs the
subspace probe under Muon and SGDM so we can test whether (a) the PCA effective
dim stays ~3 and (b) the idealized-ansatz capture stays below the random baseline
REGARDLESS of optimizer (the C claim) — or whether the optimizer moves the
geometry (which would weaken "geometry ⊥ timing").

Non-invasive: train_subspace.py and the closed infra are NOT modified. We
monkeypatch ONLY train_subspace.build_optimizer with a muon/sgdm hybrid builder.
train_subspace.run() treats the optimizer as a SINGLE object (one .zero_grad /
.step), so the Muon hybrid (Muon on 2-D hidden matrices + AdamW on the rest) is
wrapped in a tiny MultiOpt that forwards both calls.

MATCHED-PERFORMANCE NOTE: as in the 017 Tier-2 runner, Muon trains faster — the
per-eval trajectory (captured_total / pca_dim / icl_score) is logged, so the
geometry must be compared at MATCHED icl_score, not just at the final step.

  python run_subspace_optaxis.py --smoke         # CPU self-test, no writes
  python run_subspace_optaxis.py --dry-run       # print planned cells
  python run_subspace_optaxis.py [--optimizers muon,sgdm] [--seeds 5]
                                 [--seq-lens 64,128,256] [--num-shards N --shard-id I]

Output: ../../experiments/results/induction_subspace_optaxis/<name>_<opt>.jsonl
Resume-aware (skips cells whose jsonl ends with a _summary line).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
_EXP = os.path.dirname(_THIS_DIR)
if _EXP not in sys.path:
    sys.path.append(_EXP)

# Import the 010 trainer FIRST so its own sys.path discipline resolves `data`/
# `model` to induction_emergence (NOT grokking, which also ships a data.py). Only
# AFTER that do we add grokking for `muon` — by then induction_emergence.data is
# cached in sys.modules, so grokking's data.py cannot shadow it.
import train_subspace as TSUB  # noqa: E402  (010 trainer; we patch its optimizer)
from runner_utils import (  # noqa: E402
    add_shard_args, shard_cells, validate_shard_args)

_GROKKING = os.path.abspath(os.path.join(_THIS_DIR, "..", "grokking"))
if _GROKKING not in sys.path:
    sys.path.append(_GROKKING)
from muon import Muon, split_params_for_muon  # noqa: E402  (grokking infra)

OUT = os.path.join(_THIS_DIR, "..", "..", "experiments", "results",
                   "induction_subspace_optaxis")
STEPS = 15000
EVAL_EVERY = 100
TASKS = ["repeat", "markov"]
MUON_LR = 0.02


class MultiOpt:
    """Forwards zero_grad/step to several optimizers (Muon-hybrid as one object).

    train_subspace.run() uses a single optimizer handle; this lets the
    Muon(2-D) + AdamW(rest) pair drop in unchanged.
    """

    def __init__(self, opts):
        self.opts = opts

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()


def make_builder(opt, muon_lr):
    """Return a build_optimizer(model, cfg) replacement for the chosen optimizer."""
    def _build(model, cfg):
        muon_p, adamw_p = split_params_for_muon(model)
        opts = []
        if opt == "muon":
            if muon_p:
                opts.append(Muon(muon_p, lr=muon_lr, momentum=0.95,
                                 nesterov=True, ns_steps=5,
                                 weight_decay=cfg.weight_decay))
        elif opt == "sgdm":
            if muon_p:
                opts.append(torch.optim.SGD(muon_p, lr=muon_lr, momentum=0.95,
                                            nesterov=True,
                                            weight_decay=cfg.weight_decay))
        elif opt == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                     betas=(cfg.beta1, cfg.beta2),
                                     weight_decay=cfg.weight_decay)
        else:
            raise ValueError(opt)
        if adamw_p:
            opts.append(torch.optim.AdamW(adamw_p, lr=cfg.lr,
                                          betas=(cfg.beta1, cfg.beta2),
                                          weight_decay=cfg.weight_decay))
        return MultiOpt(opts)
    return _build


def make_cells(optimizers, seq_lens, n_seeds):
    cells = []
    for opt in optimizers:
        for task in TASKS:
            for L in seq_lens:
                for s in range(n_seeds):
                    cells.append((opt, task, L, s))
    return cells


def already_done(path):
    if not os.path.exists(path):
        return False
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        if size == 0:
            return False
        fh.seek(max(0, size - 4096))
        tail = fh.read().decode("utf-8", errors="replace")
    return '"_summary"' in tail


def run_smoke():
    TSUB.build_optimizer = make_builder("muon", MUON_LR)
    cfg = TSUB.Config(task="repeat", vocab_size=64, seq_len=64, batch_size=64,
                      steps=40, eval_every=20, device="cpu", seed=0)
    summary, _ = TSUB.run(cfg, out_path=None)
    assert all(k in summary for k in
               ("final_pca_dim", "final_captured_total", "final_icl_score")), \
        "summary missing subspace geometry fields"
    # sgdm path builds too
    TSUB.build_optimizer = make_builder("sgdm", MUON_LR)
    cfg2 = TSUB.Config(task="markov", vocab_size=64, seq_len=64, batch_size=64,
                       steps=20, eval_every=20, device="cpu", seed=1)
    TSUB.run(cfg2, out_path=None)
    print("RUN_SUBSPACE_OPTAXIS SMOKE PASS: muon+sgdm MultiOpt drive "
          "train_subspace.run; PCA-dim + captured-fraction geometry logged; "
          "zero writes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--optimizers", default="muon,sgdm")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seq-lens", default="64,128,256")
    ap.add_argument("--muon-lr", type=float, default=MUON_LR)
    add_shard_args(ap)
    args = ap.parse_args()
    if args.smoke:
        run_smoke()
        return
    optimizers = [o.strip() for o in args.optimizers.split(",") if o.strip()]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    cells = make_cells(optimizers, seq_lens, args.seeds)
    validate_shard_args(args)
    cells = shard_cells(cells, args.num_shards, args.shard_id)
    if args.dry_run:
        for opt, task, L, s in cells:
            print(f"{task}_L{L}_s{s}_{opt}")
        print(f"{len(cells)} cells (this shard); optimizers={optimizers} "
              f"seq_lens={seq_lens} seeds={args.seeds}")
        return
    os.makedirs(OUT, exist_ok=True)
    for i, (opt, task, L, s) in enumerate(cells):
        name = f"{task}_L{L}_s{s}_{opt}"
        path = os.path.join(OUT, name + ".jsonl")
        if already_done(path):
            print(f"[{i + 1}/{len(cells)}] skip {name}", flush=True)
            continue
        TSUB.build_optimizer = make_builder(opt, args.muon_lr)
        cfg = TSUB.Config(task=task, seq_len=L, seed=s,
                          steps=STEPS, eval_every=EVAL_EVERY)
        t0 = time.time()
        summary, _ = TSUB.run(cfg, out_path=path)
        print(f"[{i + 1}/{len(cells)}] {name}: "
              f"icl={summary['final_icl_score']:.3f} "
              f"emerge={summary['emergence_step']} "
              f"captured={summary['final_captured_total']:.4f} "
              f"pca_dim={summary['final_pca_dim']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    print("[induction_subspace_optaxis] DONE", flush=True)


if __name__ == "__main__":
    main()

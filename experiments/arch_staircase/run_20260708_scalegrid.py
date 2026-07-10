"""Direction 017/004 follow-up — does the staircase geometry SCALE?

004/017/optaxis all run at the single (L16, d128, l2) operating point (plus the
017 depth/freeze arms at d128). This runner asks whether the depth-gated,
attention-owned staircase (017) and its AdamW-vs-Muon behavior (optaxis) survive
moving to larger input dimension, larger models, and deeper stacks — i.e. is the
017/optaxis picture a scale-artifact of the small operating point, or does it hold
on a real grid.

Two arms, one runner:

  scalegrid — L in {16,24,32} x d_model in {256,512} x n_layers in {2,4} x
              optimizer in {adamw,muon} x seeds (5 @ d256, 3 @ d512, cost-scaled).
              steps=4000 for L in {16,24}; steps=8000 for L=32 (017/013 note the
              staircase's high-degree rungs arrive later as L grows, so the L=32
              cells get double the steps to give deg-4 a fair chance to land).
              D=4 staircase profile (the 004/017 main setting), batch_size=4096 —
              everything else is an ArchConfig default (freeze=none included).

  freeze8    — the 017 freeze arm at FIXED (L24, d256, l2) x freeze in
              {none,attn,mlp} x optimizer in {adamw,muon} x 8 seeds, steps=4000.
              This is the freeze x optimizer cross at higher seed count than the
              017/optaxis 5-seed pass, to firm up the attn-ownership dissociation
              (C-MAJOR-4 asked for more seeds; this gives 8 at the d256 point).

Non-invasive: train_arch.py and the closed 004 infra are NOT modified. Reuses the
optaxis runner's muon/sgdm/adamw-over-trainable-params builder (validated
muon_lr=0.02 convention; see run_arch_optaxis.py) by importing it directly — same
monkeypatch of train_arch.build_optimizer_trainable, not re-derived here.

  python run_20260708_scalegrid.py --smoke      # CPU self-test, no writes
  python run_20260708_scalegrid.py --dry-run    # print planned cells
  python run_20260708_scalegrid.py [--arms scalegrid,freeze8] [--optimizers adamw,muon]
                                   [--num-shards N --shard-id I]

Output: ../../experiments/results/arch_staircase_scale/<name>.jsonl
Resume-aware (skips cells whose jsonl ends with a _summary line).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.append(os.path.abspath(os.path.join(_THIS_DIR, "..")))

import train_arch as TA  # noqa: E402  (017 trainer; ArchConfig + run())
import run_arch_optaxis as ROA  # noqa: E402  (reuse its trainable-params muon/adamw builder)
from runner_utils import (  # noqa: E402
    add_shard_args, shard_cells, shard_suffix, validate_shard_args)

RESULTS_DIR = os.path.join(_THIS_DIR, "..", "results", "arch_staircase_scale")

# Reuse the optaxis hybrid optimizer builder (muon/sgdm/adamw over TRAINABLE
# params only; honors freeze arms via split_params_for_muon). Same monkeypatch
# target train_arch.build_optimizer_trainable, applied explicitly here too so
# this runner does not silently depend on run_arch_optaxis's own module-level
# side effect.
TA.build_optimizer_trainable = ROA.build_opt_trainable_any


# ---------------------------------------------------------------------------
# cell construction
# ---------------------------------------------------------------------------

def make_scalegrid_cells(Ls, dmodels, layers_list, optimizers,
                          seeds_d256, seeds_d512, muon_lr):
    """L x d_model x n_layers x optimizer x seeds (seed count cost-scaled by d)."""
    cells = []
    for L in Ls:
        steps = 8000 if L == 32 else 4000
        for d in dmodels:
            n_seeds = seeds_d256 if d == 256 else seeds_d512
            for nl in layers_list:
                for opt in optimizers:
                    for s in range(n_seeds):
                        cfg = TA.ArchConfig(
                            L=L, D=4, profile="staircase", batch_size=4096,
                            d_model=d, n_layers=nl, freeze="none",
                            optimizer=opt, muon_lr=muon_lr, steps=steps,
                            seed=s)
                        name = f"scalegrid_L{L}_d{d}_l{nl}_fnone_{opt}_s{s}"
                        cells.append((name, cfg))
    return cells


def make_freeze8_cells(optimizers, freezes, n_seeds, muon_lr):
    """Fixed (L24, d256, l2) x freeze x optimizer x 8 seeds, steps=4000."""
    cells = []
    for fz in freezes:
        for opt in optimizers:
            for s in range(n_seeds):
                cfg = TA.ArchConfig(
                    L=24, D=4, profile="staircase", batch_size=4096,
                    d_model=256, n_layers=2, freeze=fz,
                    optimizer=opt, muon_lr=muon_lr, steps=4000,
                    seed=s)
                name = f"freeze8_L24_d256_l2_f{fz}_{opt}_s{s}"
                cells.append((name, cfg))
    return cells


def cell_path(name):
    return os.path.join(RESULTS_DIR, f"{name}.jsonl")


def cell_done(path):
    if not os.path.exists(path):
        return False
    last = ""
    with open(path) as f:
        for line in f:
            if line.strip():
                last = line
    try:
        return "_summary" in json.loads(last)
    except json.JSONDecodeError:
        return False


# ---------------------------------------------------------------------------
# smoke — tiny CPU cells, zero writes
# ---------------------------------------------------------------------------

def run_smoke():
    """CPU: one scalegrid-shaped cell + one freeze8-shaped cell, both optimizers."""
    for opt in ("adamw", "muon"):
        # scalegrid-shaped: tiny L/d/steps, freeze=none
        cfg = TA.ArchConfig(L=10, D=3, profile="staircase", batch_size=256,
                            eval_n=512, d_model=32, n_heads=2, n_layers=2,
                            freeze="none", optimizer=opt, muon_lr=0.02,
                            steps=40, eval_every=20, device="cpu")
        summary, _ = TA.run(cfg, out_path=None)
        assert all(k in summary for k in
                   ("staircase_spread", "half_times", "final_fit_corr", "n_learned")), \
            f"{opt} scalegrid-shaped cell: summary schema incomplete"

        # freeze8-shaped: tiny L/d/steps, freeze=attn (exercises the same
        # trainable-params builder the freeze8 arm depends on)
        cfg_f = TA.ArchConfig(L=10, D=3, profile="staircase", batch_size=256,
                              eval_n=512, d_model=32, n_heads=2, n_layers=2,
                              freeze="attn", optimizer=opt, muon_lr=0.02,
                              steps=40, eval_every=20, device="cpu")
        summary_f, _ = TA.run(cfg_f, out_path=None)
        assert summary_f["n_frozen"] > 0 and summary_f["n_trainable"] > 0, \
            f"{opt} freeze8-shaped cell: freeze arm did not partition params"

    print("RUN_20260708_SCALEGRID SMOKE PASS: scalegrid- and freeze8-shaped "
          "cells run under adamw+muon, 004 SI schema present, freeze arm "
          "partitions trainable params; zero writes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--arms", default="scalegrid,freeze8",
                    help="comma list from {scalegrid,freeze8}")
    ap.add_argument("--Ls", default="16,24,32", help="scalegrid L values")
    ap.add_argument("--d-models", default="256,512", help="scalegrid d_model values")
    ap.add_argument("--layers", default="2,4", help="scalegrid n_layers values")
    ap.add_argument("--optimizers", default="adamw,muon",
                    help="comma list from {adamw,muon,sgdm}, both arms")
    ap.add_argument("--seeds-d256", type=int, default=5,
                    help="scalegrid seeds at d_model=256")
    ap.add_argument("--seeds-d512", type=int, default=3,
                    help="scalegrid seeds at d_model=512")
    ap.add_argument("--freezes", default="none,attn,mlp", help="freeze8 freeze arms")
    ap.add_argument("--freeze8-seeds", type=int, default=8, help="freeze8 seeds")
    ap.add_argument("--muon-lr", type=float, default=0.02,
                    help="Muon hidden-matrix lr (validated 017/optaxis convention)")
    add_shard_args(ap)
    args = ap.parse_args()
    if args.smoke:
        run_smoke()
        return

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    Ls = [int(x) for x in args.Ls.split(",") if x.strip()]
    dmodels = [int(x) for x in args.d_models.split(",") if x.strip()]
    layers_list = [int(x) for x in args.layers.split(",") if x.strip()]
    optimizers = [o.strip() for o in args.optimizers.split(",") if o.strip()]
    freezes = [f.strip() for f in args.freezes.split(",") if f.strip()]

    cells = []
    counts = {}
    if "scalegrid" in arms:
        sg = make_scalegrid_cells(Ls, dmodels, layers_list, optimizers,
                                  args.seeds_d256, args.seeds_d512, args.muon_lr)
        counts["scalegrid"] = len(sg)
        cells += sg
    if "freeze8" in arms:
        fz = make_freeze8_cells(optimizers, freezes, args.freeze8_seeds, args.muon_lr)
        counts["freeze8"] = len(fz)
        cells += fz

    validate_shard_args(args)
    total = len(cells)
    cells = shard_cells(cells, args.num_shards, args.shard_id)

    if args.dry_run:
        for name, _ in cells:
            print(name)
        suffix = shard_suffix(args.num_shards, args.shard_id, total, len(cells))
        print(f"{len(cells)} cells (this shard); arms={arms} counts={counts}{suffix}")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for i, (name, cfg) in enumerate(cells):
        path = cell_path(name)
        if cell_done(path):
            print(f"[{i + 1}/{len(cells)}] skip {name}", flush=True)
            continue
        summary, _ = TA.run(cfg, out_path=path)
        print(f"[{i + 1}/{len(cells)}] {name}: fit={summary['final_fit_corr']:.3f} "
              f"n_learned={summary['n_learned']} "
              f"spread={summary['staircase_spread']}", flush=True)
    print("[arch_staircase_scale] DONE", flush=True)


if __name__ == "__main__":
    main()

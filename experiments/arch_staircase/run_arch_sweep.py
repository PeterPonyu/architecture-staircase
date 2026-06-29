"""Direction 017 grid runner — architecture anatomy of the degree staircase.

  python run_arch_sweep.py --smoke      # freeze-instrument self-test, CPU, no writes
  python run_arch_sweep.py --dry-run    # print planned cells
  python run_arch_sweep.py [--heads] [--num-shards N --shard-id I]   # formal (executor)

Grid (direction doc "Conditions", AdamW fixed):
  width  : d_model {64, 128, 256, 512} x 2L, seeds (d64 -> 8, else 5)
  depth  : n_layers {1, 2, 4} x d128 (d128/2L shared with width arm), 5 seeds
  freeze : {attn, mlp} x d128/2L, 5 seeds
  --heads (P3 sub-arm, optional): n_heads {2, 8} x d128/2L, 3 seeds
Resume-aware: skips cells whose jsonl ends with a _summary line.
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

from train_arch import ArchConfig, RESULTS_DIR, run, run_smoke  # noqa: E402
from runner_utils import (  # noqa: E402
    add_shard_args, shard_cells, validate_shard_args)


def make_cells(include_heads=False):
    cells, seen = [], set()

    def add(**kw):
        cfg = ArchConfig(**kw)
        if cfg.name() not in seen:
            seen.add(cfg.name())
            cells.append(cfg)

    for d in (64, 128, 256, 512):                      # width arm
        n_seeds = 8 if d == 64 else 5                  # vetoer fix: d64 -> 8
        for s in range(n_seeds):
            add(d_model=d, seed=s)
    for nl in (1, 2, 4):                               # depth arm
        for s in range(5):
            add(n_layers=nl, seed=s)
    for fz in ("attn", "mlp"):                         # freeze arm
        for s in range(5):
            add(freeze=fz, seed=s)
    if include_heads:                                  # P3 sub-arm
        for nh in (2, 8):
            for s in range(3):
                add(n_heads=nh, seed=s)
    return cells


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--heads", action="store_true",
                    help="include the P3 n_heads sub-arm")
    add_shard_args(ap)
    args = ap.parse_args()
    if args.smoke:
        run_smoke()
        return
    cells = make_cells(include_heads=args.heads)
    validate_shard_args(args)
    cells = shard_cells(cells, args.num_shards, args.shard_id)
    if args.dry_run:
        for c in cells:
            print(c.name())
        print(f"{len(cells)} cells (this shard)")
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for i, cfg in enumerate(cells):
        path = os.path.join(RESULTS_DIR, cfg.name() + ".jsonl")
        if cell_done(path):
            print(f"[{i + 1}/{len(cells)}] skip {cfg.name()}")
            continue
        summary, _ = run(cfg, out_path=path)
        print(f"[{i + 1}/{len(cells)}] {cfg.name()}: "
              f"n_learned={summary['n_learned']} "
              f"spread={summary['staircase_spread']}")
    print("[arch_staircase] DONE")


if __name__ == "__main__":
    main()

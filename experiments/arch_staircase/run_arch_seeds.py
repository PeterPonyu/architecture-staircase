"""Direction 017 TIER-2 — seed densification of the freeze arms (AdamW).

The C-paper red-team (2026-06-14) showed the none-vs-attn staircase-fit gap is
p≈0.027 at n=5 and FRAGILE (leave-one-out swings |t| to 2.1-3.0). To firm the
component-ownership claim we extend the freeze arms (none/attn/mlp, d128/L2) from
5 to 15 seeds. This is the SAME trainer and the SAME AdamW as 017 (no optimizer
axis here — that is run_arch_optaxis.py), and it writes to the SAME result dir
with the SAME naming, so the extra seeds MERGE with the existing s0-s4 cells and
analyze_arch.py picks them up automatically.

  python run_arch_seeds.py --smoke                 # delegates to train_arch smoke
  python run_arch_seeds.py --dry-run               # print planned cells
  python run_arch_seeds.py [--start-seed 5] [--n-seeds 10]
                           [--freezes none,attn,mlp] [--num-shards N --shard-id I]

Output: ../../experiments/results/arch_staircase/<name>.jsonl  (merges with 017)
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

from train_arch import ArchConfig, RESULTS_DIR, run, run_smoke  # noqa: E402
from runner_utils import (  # noqa: E402
    add_shard_args, shard_cells, validate_shard_args)


def make_cells(freezes, start_seed, n_seeds):
    """Freeze arms at d128/L2 (017's component-ownership cells), extra seeds."""
    cells = []
    for fz in freezes:
        for s in range(start_seed, start_seed + n_seeds):
            cells.append(ArchConfig(freeze=fz, seed=s))   # d128/L2/adamw default
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
    ap.add_argument("--start-seed", type=int, default=5)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--freezes", default="none,attn,mlp")
    add_shard_args(ap)
    args = ap.parse_args()
    if args.smoke:
        run_smoke()
        return
    freezes = [f.strip() for f in args.freezes.split(",") if f.strip()]
    cells = make_cells(freezes, args.start_seed, args.n_seeds)
    validate_shard_args(args)
    cells = shard_cells(cells, args.num_shards, args.shard_id)
    if args.dry_run:
        for c in cells:
            print(c.name())
        print(f"{len(cells)} cells (this shard); freezes={freezes} "
              f"seeds={args.start_seed}..{args.start_seed + args.n_seeds - 1}")
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for i, cfg in enumerate(cells):
        path = os.path.join(RESULTS_DIR, cfg.name() + ".jsonl")
        if cell_done(path):
            print(f"[{i + 1}/{len(cells)}] skip {cfg.name()}", flush=True)
            continue
        summary, _ = run(cfg, out_path=path)
        print(f"[{i + 1}/{len(cells)}] {cfg.name()}: "
              f"fit={summary['final_fit_corr']:.3f} "
              f"spread={summary['staircase_spread']}", flush=True)
    print("[arch_staircase seeds] DONE", flush=True)


if __name__ == "__main__":
    main()

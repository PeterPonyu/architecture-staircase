"""Direction 004 — degree-staircase grid runner (NOT executed yet).

Grid (F1 design, ~45 cells):
    optimizer ∈ {muon, adamw, sgdm}
    profile   ∈ {staircase, mixed, pure3}      (the degree-profile settings)
    seed      ∈ 0..4
3 x 3 x 5 = 45 cells. Each cell trains online (fresh-batch) on a boolean
staircase task and logs per-degree Walsh correlations + the staircase index.

profiles:
    staircase : g = sum_{k=1..D} chi_{S_k}  (one monomial per degree 1..D)
    mixed     : equal-weight sum (same set; named knob for later reweighting)
    pure3     : a single pure degree-3 monomial (profile="pure", pure_degree=3)

Output: ../../experiments/results/degree_staircase/<name>.jsonl
Resume-aware: skips a cell whose jsonl already ends with a _summary line.

Flags
-----
--smoke   : delegate to train_staircase smoke (no files, <60s) and exit 0.
--dry-run : print planned cells and exit 0 (launches NOTHING).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_EXPERIMENTS_DIR = os.path.dirname(_THIS_DIR)
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.append(_EXPERIMENTS_DIR)

from train_staircase import Config, run, run_smoke  # noqa: E402
from runner_utils import (  # noqa: E402
    add_shard_args,
    shard_cells,
    shard_suffix,
    validate_shard_args,
)


OPTIMIZERS = ["muon", "adamw", "sgdm"]
# (profile name -> overrides applied on top of base Config)
PROFILES = {
    "staircase": dict(profile="staircase", D=4, L=16),
    "mixed":     dict(profile="mixed",     D=4, L=16),
    "pure3":     dict(profile="pure",      pure_degree=3, D=4, L=16),
}
SEEDS = list(range(5))            # 0-4
STEPS = 4000
EVAL_EVERY = 50

OUT = os.path.join(_THIS_DIR, "..", "..", "experiments", "results", "degree_staircase")


def _build_cells():
    return [
        (opt, pname, seed)
        for opt in OPTIMIZERS
        for pname in PROFILES
        for seed in SEEDS
    ]


def already_done(path: str) -> bool:
    """True iff the jsonl exists and ends with a _summary line."""
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


def main():
    ap = argparse.ArgumentParser(description="degree-staircase grid runner")
    ap.add_argument("--smoke", action="store_true",
                    help="Run smoke checks and exit (no files written)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned cells and exit (no training)")
    add_shard_args(ap)
    args = ap.parse_args()
    validate_shard_args(args)

    if args.smoke:
        run_smoke()
        sys.exit(0)

    all_cells = _build_cells()
    cells = shard_cells(all_cells, args.num_shards, args.shard_id)

    if args.dry_run:
        print(f"[degree_staircase] dry-run: {len(cells)} cells planned"
              + shard_suffix(args.num_shards, args.shard_id,
                             len(all_cells), len(cells)))
        for i, (opt, pname, seed) in enumerate(cells):
            name = f"{opt}_{pname}_s{seed}"
            print(f"  [{i+1:02d}/{len(cells)}] {name}  steps={STEPS}")
        sys.exit(0)

    # --- real training path (only when neither flag set) ---
    os.makedirs(OUT, exist_ok=True)
    print(f"[degree_staircase] {len(cells)} cells -> {OUT}"
          + shard_suffix(args.num_shards, args.shard_id,
                         len(all_cells), len(cells)),
          flush=True)

    for i, (opt, pname, seed) in enumerate(cells):
        name = f"{opt}_{pname}_s{seed}"
        path = os.path.join(OUT, name + ".jsonl")
        if already_done(path):
            print(f"[{i+1}/{len(cells)}] skip {name}", flush=True)
            continue
        cfg = Config(
            optimizer=opt,
            seed=seed,
            steps=STEPS,
            eval_every=EVAL_EVERY,
            **PROFILES[pname],
        )
        t0 = time.time()
        s, _ = run(cfg, out_path=path)
        print(
            f"[{i+1}/{len(cells)}] {name}: "
            f"fit={s['final_fit_corr']:.3f} spread={s['staircase_spread']:.0f} "
            f"n_learned={s['n_learned']} ({time.time()-t0:.0f}s)",
            flush=True,
        )

    print("[degree_staircase] DONE", flush=True)


if __name__ == "__main__":
    main()

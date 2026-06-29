"""Direction 010 — induction-subspace grid runner (NOT executed yet).

Grid design (test the 3-D subspace claim across tasks, context lengths, seeds;
FIXED AdamW throughout — interlock with direction 007, NO optimizer arm):

    task    ∈ {repeat, markov}                 2 task variants
    seq_len ∈ {64, 96, 128, 192, 256, 384}     6 context lengths (for the L²-fit)
    seed    ∈ 0..8                              9 seeds (cross-seed variance)
    2 × 6 × 9 = 108 cells.

  --vocab-arm adds a third task surface by varying the vocabulary at the middle
  context lengths, so a subspace finding is not a single-vocab artifact:
      task=repeat, vocab ∈ {32, 128}, seq_len ∈ {96, 128, 192}, seed 0..8
      2 vocab × 3 L × 9 seeds = 54 cells  ->  grand total 162 cells.

Why these knobs. The paper predicts emergence time t_ICL = Θ(N²) (quadratic in
context length); spanning 6 seq_lens lets a real-training run be fit against that
law. The two task variants + vocab arm probe whether the "captured-fraction" and
"PCA effective dim ≈ 3" signals are task-robust or repeated-segment-specific.

Output: ../../experiments/results/induction_subspace/<name>.jsonl
Resume-aware: skips a cell whose jsonl already ends with a _summary line.

Flags
-----
--smoke    : delegate to train_subspace smoke (no files, <60s) and exit 0.
--dry-run  : print planned cells and exit 0 (launches NOTHING).
--vocab-arm: include the vocab-robustness arm in the (dry-run or real) plan.
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

from train_subspace import Config, run, run_smoke  # noqa: E402
from runner_utils import (  # noqa: E402
    add_shard_args,
    shard_cells,
    shard_suffix,
    validate_shard_args,
)


TASKS = ["repeat", "markov"]
SEQ_LENS = [64, 96, 128, 192, 256, 384]   # 6 lengths for the t_ICL ~ N² fit
SEEDS = list(range(9))                     # 0-8
STEPS = 15000
EVAL_EVERY = 100

# vocab-robustness arm.
VOCAB_ARM_VOCABS = [32, 128]
VOCAB_ARM_SEQ_LENS = [96, 128, 192]
VOCAB_ARM_SEEDS = list(range(9))

OUT = os.path.join(_THIS_DIR, "..", "..", "experiments", "results",
                   "induction_subspace")


def _main_cells():
    """Main grid cells: (name, overrides dict)."""
    cells = []
    for task in TASKS:
        for L in SEQ_LENS:
            for seed in SEEDS:
                name = f"{task}_L{L}_s{seed}"
                cells.append((name, dict(task=task, seq_len=L, seed=seed)))
    return cells


def _vocab_arm_cells():
    """Vocab-robustness arm cells (flagged): (name, overrides dict)."""
    cells = []
    for vocab in VOCAB_ARM_VOCABS:
        for L in VOCAB_ARM_SEQ_LENS:
            for seed in VOCAB_ARM_SEEDS:
                name = f"vocabarm_V{vocab}_L{L}_s{seed}"
                cells.append((name, dict(task="repeat", vocab_size=vocab,
                                         seq_len=L, seed=seed)))
    return cells


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
    ap = argparse.ArgumentParser(description="induction-subspace grid runner")
    ap.add_argument("--smoke", action="store_true",
                    help="Run smoke checks and exit (no files written)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned cells and exit (no training)")
    ap.add_argument("--vocab-arm", action="store_true",
                    help="Include the vocab-robustness arm in the plan")
    add_shard_args(ap)
    args = ap.parse_args()
    validate_shard_args(args)

    if args.smoke:
        run_smoke()
        sys.exit(0)

    all_cells = _main_cells()
    if args.vocab_arm:
        all_cells = all_cells + _vocab_arm_cells()
    cells = shard_cells(all_cells, args.num_shards, args.shard_id)

    if args.dry_run:
        n_main = len(_main_cells())
        n_vocab = len(_vocab_arm_cells())
        print(f"[induction_subspace] dry-run: {len(cells)} cells planned "
              f"(main={n_main}" + (f" + vocab_arm={n_vocab}" if args.vocab_arm else "")
              + ")"
              + shard_suffix(args.num_shards, args.shard_id,
                             len(all_cells), len(cells)))
        print(f"  main grid: {len(TASKS)} task x {len(SEQ_LENS)} seq_len "
              f"x {len(SEEDS)} seeds = {n_main}  [FIXED AdamW]")
        if args.vocab_arm:
            print(f"  vocab arm: {len(VOCAB_ARM_VOCABS)} vocab x "
                  f"{len(VOCAB_ARM_SEQ_LENS)} seq_len x {len(VOCAB_ARM_SEEDS)} "
                  f"seeds = {n_vocab}")
        for i, (name, ov) in enumerate(cells):
            print(f"  [{i+1:03d}/{len(cells)}] {name}  steps={STEPS}  {ov}")
        sys.exit(0)

    # --- real training path (only when neither smoke nor dry-run set) ----------
    os.makedirs(OUT, exist_ok=True)
    print(f"[induction_subspace] {len(cells)} cells -> {OUT}"
          + shard_suffix(args.num_shards, args.shard_id,
                         len(all_cells), len(cells)),
          flush=True)

    for i, (name, ov) in enumerate(cells):
        path = os.path.join(OUT, name + ".jsonl")
        if already_done(path):
            print(f"[{i+1}/{len(cells)}] skip {name}", flush=True)
            continue
        cfg = Config(steps=STEPS, eval_every=EVAL_EVERY, **ov)
        t0 = time.time()
        s, _ = run(cfg, out_path=path)
        print(
            f"[{i+1}/{len(cells)}] {name}: "
            f"icl={s['final_icl_score']:.3f} "
            f"emerge={s['emergence_step']} "
            f"captured={s['final_captured_total']:.3f} "
            f"pca_dim={s['final_pca_dim']} "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )

    print("[induction_subspace] DONE", flush=True)


if __name__ == "__main__":
    main()

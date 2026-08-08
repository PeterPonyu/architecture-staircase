#!/usr/bin/env python3
"""G002 C-paper Muon lr/freeze robustness cell runner.

Writes only the second-round ultragoal namespace:
experiments/results/arch_staircase_optaxis_lrsweep/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCH = ROOT / 'experiments' / 'arch_staircase'
OUT = ROOT / 'experiments' / 'results' / 'arch_staircase_optaxis_lrsweep'
if str(ARCH) in sys.path:
    sys.path.remove(str(ARCH))
sys.path.insert(0, str(ARCH))

import run_arch_optaxis as RAO  # noqa: F401,E402 installs Muon/SGDM trainable-param patch
import train_arch as TA  # noqa: E402


def lr_slug(x: float) -> str:
    return (f"{x:g}").replace('.', 'p').replace('-', 'm')


def cell_done(path: Path) -> bool:
    if not path.exists():
        return False
    last = ''
    with path.open() as f:
        for line in f:
            if line.strip():
                last = line
    if not last:
        return False
    try:
        return '_summary' in json.loads(last)
    except json.JSONDecodeError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', choices=['none', 'attn'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--muon-lr', type=float, required=True)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--eval-every', type=int, default=50)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"muon_l2_{args.freeze}_lr{lr_slug(args.muon_lr)}_s{args.seed}.jsonl"
    if cell_done(path):
        print(json.dumps({'status': 'skip', 'path': str(path), 'freeze': args.freeze,
                          'seed': args.seed, 'muon_lr': args.muon_lr}), flush=True)
        return 0
    if path.exists():
        path.unlink()

    cfg = TA.ArchConfig(optimizer='muon', freeze=args.freeze, n_layers=2,
                        d_model=128, seed=args.seed, muon_lr=args.muon_lr,
                        steps=args.steps, eval_every=args.eval_every)
    summary, _ = TA.run(cfg, out_path=str(path))
    print(json.dumps({
        'status': 'done',
        'path': str(path),
        'freeze': args.freeze,
        'seed': args.seed,
        'muon_lr': args.muon_lr,
        'final_fit_corr': summary['final_fit_corr'],
        'final_deg4': summary['final_deg_corr'].get('4'),
        'n_learned': summary['n_learned'],
        'elapsed_sec': summary['elapsed_sec'],
    }), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

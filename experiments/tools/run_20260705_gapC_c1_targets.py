#!/usr/bin/env python3
"""C#1 -- target-family generality (TIER-1, paper-C gap battery).

Tests whether the low->high degree acquisition ORDER (the "staircase") is
intrinsic to feature learning or an artifact of the equal-weight nested target.
Three target families (implemented by monkeypatch in run_20260705_gapC_lib; the
degree_staircase harness has no numeric coefficient knob and is NOT modified):

  geom   : geometric decay a_k = 2^-k over degrees 1..4  (low degrees weighted UP)
  gapped : degrees {1,3,4}, equal weights (degree 2 removed -> is the order robust
           to a missing rung?)
  invw   : inverse weighting a_k = 2^k over degrees 1..4  (HIGH degrees dominate;
           if low degrees are STILL learned first -> order is intrinsic, not
           target-driven; if the order flips -> order is set by the target)

Grid: 3 families x {adamw, muon, sgdm} x n=15. L=16, d_model=128, n_layers=2,
4000 steps. Metrics per run: staircase rank index + spread (SI) + per-degree
half-times (all in the standard train_staircase schema).

Cells are structured pilot-first: run seeds 0-4 for a family x optimizer, then
seeds 5-14. One .jsonl + _summary per (family, opt, seed), resume-aware.
  python run_20260705_gapC_c1_targets.py --family geom --opt adamw --seeds 0-4
Output: results/ieee_gap_20260705/C/c1_targets/<family>_<opt>_s<seed>.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_20260705_gapC_lib as L  # noqa: E402

FAMILIES = ["geom", "gapped", "invw"]
OPTIMIZERS = ["adamw", "muon", "sgdm"]
OUT_DIR = os.path.join(L.RESULTS_ROOT, "c1_targets")
STEPS = 4000
EVAL_EVERY = 50


def parse_seeds(spec: str):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x != ""]


def run_smoke():
    for fam in FAMILIES:
        cfg = L.TS.Config(d_model=64, n_layers=2, L=16, D=4, optimizer="adamw",
                          seed=0, steps=120, eval_every=60, device="cuda")
        s, _ = L.staircase_run(cfg, fam, out_path=None)
        exp = L.TARGET_FAMILIES[fam][0]
        got = [int(k) for k in s["final_deg_corr"].keys()]
        assert got == exp, f"{fam}: degrees {got} != {exp}"
    print("C1 SMOKE PASS: geom/gapped/invw target families produce the right "
          "degree sets under the monkeypatched staircase target")


def main():
    ap = argparse.ArgumentParser(description="C#1 target-family generality")
    ap.add_argument("--family", choices=FAMILIES)
    ap.add_argument("--opt", choices=OPTIMIZERS)
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return
    if not (args.family and args.opt):
        ap.error("--family and --opt required (unless --smoke)")

    seeds = parse_seeds(args.seeds)
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, seed in enumerate(seeds):
        name = f"{args.family}_{args.opt}_s{seed}"
        path = os.path.join(OUT_DIR, name + ".jsonl")
        if args.dry_run:
            print(name)
            continue
        if L.already_done(path):
            print(f"[c1 {i+1}/{len(seeds)}] skip {name}", flush=True)
            continue
        cfg = L.TS.Config(d_model=128, n_layers=2, L=16, D=4, optimizer=args.opt,
                          seed=seed, steps=args.steps, eval_every=EVAL_EVERY,
                          device="cuda")
        t0 = time.time()
        s, _ = L.staircase_run(cfg, args.family, out_path=path)
        print(f"[c1 {i+1}/{len(seeds)}] {name}: fit={s['final_fit_corr']:.3f} "
              f"spread={s['staircase_spread']:.0f} n_learned={s['n_learned']} "
              f"({time.time()-t0:.0f}s)", flush=True)
    print("[c1_targets] DONE", flush=True)


if __name__ == "__main__":
    main()

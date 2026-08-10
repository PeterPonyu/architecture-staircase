#!/usr/bin/env python3
"""C#5 -- frontier-degree freeze grid (TIER-2, paper-C gap battery).

At D_degree=4 the finding is "attention owns the frontier degree (deg-4)".
This asks whether that is really about the FRONTIER degree or specifically deg-4:
move the frontier up to degree 5 (D_degree=5 staircase; L=16 supports it since
1+2+3+4+5=15) and run the freeze grid. Prediction/upgrade to test:
  - attention gates deg-5 (the NEW frontier) -> attn-frozen kills deg-5, and
  - deg-4 becomes MLP-reachable (no longer the frontier) -> attn-frozen preserves
    deg-4.
That would upgrade the claim from "attention owns deg-4" to "attention owns the
frontier degree".

Grid: freeze {none, attn, mlp} x AdamW x n=15 (primary);
      + Muon x n=15 as a STRETCH block appended at the manifest end.
Setting: d_model=128, n_layers=2, L=16, D_degree=5, 4000 steps.
One .jsonl + _summary per (arm, opt, seed), resume-aware.
  python run_20260705_gapC_c5_frontier.py --arm attn --opt adamw --seeds 0-4
Output: results/ieee_gap_20260705/C/c5_frontier/<arm>_<opt>_s<seed>.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_20260705_gapC_lib as L  # noqa: E402

ARMS = ["none", "attn", "mlp"]
OPTIMIZERS = ["adamw", "muon"]
OUT_DIR = os.path.join(L.RESULTS_ROOT, "c5_frontier")
DEG = 5
STEPS = 4000
EVAL_EVERY = 50


def parse_seeds(spec: str):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x != ""]


def run_smoke():
    cfg = L.TA.ArchConfig(d_model=64, n_layers=2, D=DEG, L=16, optimizer="adamw",
                          seed=0, steps=120, eval_every=60, device="cuda")
    for arm in ("none", "attn"):
        s, _ = L.custom_freeze_run(cfg, L.freeze_marks(arm), arm, out_path=None)
        assert "5" in s["final_deg_corr"] and "4" in s["final_deg_corr"]
    print("C5 SMOKE PASS: D_degree=5 staircase trains, deg-4 + deg-5 logged, "
          "none/attn freeze honored")


def main():
    ap = argparse.ArgumentParser(description="C#5 frontier-degree freeze grid")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--opt", choices=OPTIMIZERS, default="adamw")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return
    if not args.arm:
        ap.error("--arm required (unless --smoke)")

    seeds = parse_seeds(args.seeds)
    os.makedirs(OUT_DIR, exist_ok=True)
    marks = L.freeze_marks(args.arm)
    for i, seed in enumerate(seeds):
        name = f"{args.arm}_{args.opt}_s{seed}"
        path = os.path.join(OUT_DIR, name + ".jsonl")
        if args.dry_run:
            print(name)
            continue
        if L.already_done(path):
            print(f"[c5 {i+1}/{len(seeds)}] skip {name}", flush=True)
            continue
        cfg = L.TA.ArchConfig(d_model=128, n_layers=2, D=DEG, L=16,
                              optimizer=args.opt, seed=seed, steps=args.steps,
                              eval_every=EVAL_EVERY, device="cuda")
        t0 = time.time()
        s, _ = L.custom_freeze_run(cfg, marks, args.arm, out_path=path)
        dc = s["final_deg_corr"]
        print(f"[c5 {i+1}/{len(seeds)}] {name}: fit={s['final_fit_corr']:.3f} "
              f"deg4={dc.get('4'):} deg5={dc.get('5'):} "
              f"({time.time()-t0:.0f}s)", flush=True)
    print("[c5_frontier] DONE", flush=True)


if __name__ == "__main__":
    main()

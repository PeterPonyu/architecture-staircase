#!/usr/bin/env python3
"""C#2 -- Muon-side ownership localization (TIER-1, paper-C gap battery).

Context: under Muon, deg-4 is acquired in ALL single-freeze arms
(none / attn-frozen / mlp-frozen ~= 0.49-0.50) -> redundancy; the manuscripts
say "either sub-network suffices". This sharpens that claim:

  (a) both-frozen FLOOR CONTROL: attention AND MLP frozen (only embed / readout /
      layernorm trainable). If deg-4 STILL appears -> the freeze methodology is
      leaky (the readout alone can express it); if deg-4 DIES -> the redundancy
      is a real two-path property of the hidden network.
  (b) per-layer knockouts: layer-0-attn / layer-1-attn / layer-0-mlp / layer-1-mlp,
      each frozen alone. Localizes which layer/component carries deg-4 under Muon.

Setting: d_model=128, n_layers=2 (D), L=16, D_degree=4, Muon, 4000 steps, n=15.
The single-freeze none/attn/mlp Muon arms already exist in
results/arch_staircase_optaxis (used as reference by the analyzer); this runner
adds only the new arms.

Cells (one shell command each), resume-aware, one .jsonl + _summary per seed:
  python run_20260705_gapC_c2_ownership.py --arm both   --seeds 0-4
  python run_20260705_gapC_c2_ownership.py --arm l0attn --seeds 5-14
  ...
Output: results/ieee_gap_20260705/C/c2_ownership/<arm>_muon_s<seed>.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_20260705_gapC_lib as L  # noqa: E402

ARMS = ["both", "l0attn", "l1attn", "l0mlp", "l1mlp"]
OUT_DIR = os.path.join(L.RESULTS_ROOT, "c2_ownership")
OPT = "muon"
STEPS = 4000
EVAL_EVERY = 50


def parse_seeds(spec: str):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x != ""]


def run_smoke():
    cfg = L.TA.ArchConfig(d_model=64, n_layers=2, D=4, L=16, optimizer=OPT,
                          seed=0, steps=120, eval_every=60, device="cuda")
    for arm in ("both", "l0attn"):
        s, _ = L.custom_freeze_run(cfg, L.freeze_marks(arm), arm, out_path=None)
        assert s["n_frozen"] > 0 and s["n_trainable"] > 0
        assert "4" in s["final_deg_corr"]
    print("C2 SMOKE PASS: both + l0attn arms train, freeze honored, deg-4 logged")


def main():
    ap = argparse.ArgumentParser(description="C#2 Muon ownership localization")
    ap.add_argument("--arm", choices=ARMS, help="freeze arm")
    ap.add_argument("--seeds", default="0-4", help="e.g. 0-4 or 5-14 or 0,1,2")
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
        name = f"{args.arm}_{OPT}_s{seed}"
        path = os.path.join(OUT_DIR, name + ".jsonl")
        if args.dry_run:
            print(name)
            continue
        if L.already_done(path):
            print(f"[c2 {i+1}/{len(seeds)}] skip {name}", flush=True)
            continue
        cfg = L.TA.ArchConfig(d_model=128, n_layers=2, D=4, L=16, optimizer=OPT,
                              seed=seed, steps=args.steps, eval_every=EVAL_EVERY,
                              device="cuda")
        t0 = time.time()
        s, _ = L.custom_freeze_run(cfg, marks, args.arm, out_path=path)
        print(f"[c2 {i+1}/{len(seeds)}] {name}: fit={s['final_fit_corr']:.3f} "
              f"deg4={s['final_deg_corr'].get('4'):} n_fr={s['n_frozen']} "
              f"({time.time()-t0:.0f}s)", flush=True)
    print("[c2_ownership] DONE", flush=True)


if __name__ == "__main__":
    main()

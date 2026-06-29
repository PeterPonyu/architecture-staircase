"""C scale-robustness — does the depth-threshold + staircase survive larger scale?

Re-runs the degree-staircase at LARGER scale (L=24 bits vs 16, d_model=256 vs
128) across depth in {1,2} and optimizer in {adamw,muon}, to check that (i) a
single layer still fails the high degrees while two layers learn them
(depth-threshold), and (ii) the per-degree staircase is not flattened by Muon.

Writes results/arch_staircase_scaleup/{opt}_L{L}_l{layers}_s{seed}.jsonl
Usage: python run_staircase_scaleup.py [--smoke]
"""
from __future__ import annotations
import argparse, os, sys, time

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)
from train_staircase import Config, run  # noqa: E402

OUT = os.path.abspath(os.path.join(_THIS, "..", "results", "arch_staircase_scaleup"))
OPTS = ["adamw", "muon"]
LAYERS = [1, 2]
SEEDS = [0, 1, 2]
L_BITS = 24


def cfg_for(opt, layers, seed, steps):
    return Config(profile="staircase", D=4, L=L_BITS, d_model=256, n_heads=4,
                  n_layers=layers, optimizer=opt, seed=seed, steps=steps,
                  device="cuda")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    a = ap.parse_args()
    if a.smoke:
        s, _ = run(cfg_for("adamw", 2, 0, 200), out_path=None)
        print(f"SMOKE C-scaleup OK L=24 d=256: fit={s.get('final_fit_corr'):.3f} "
              f"deg4={s.get('final_deg_corr',{}).get('4')}")
        return
    os.makedirs(OUT, exist_ok=True)
    cells = [(o, l, s) for o in OPTS for l in LAYERS for s in a.seeds]
    print(f"C scale-up: {len(cells)} cells (L=24,d=256) -> {OUT}", flush=True)
    for i, (o, l, s) in enumerate(cells):
        out = os.path.join(OUT, f"{o}_L{L_BITS}_l{l}_s{s}.jsonl")
        t = time.time()
        summ, _ = run(cfg_for(o, l, s, a.steps), out_path=out)
        dc = summ.get("final_deg_corr", {})
        print(f"[{i+1}/{len(cells)}] {o} L{l} s{s}: fit={summ.get('final_fit_corr'):.3f} "
              f"deg=[{dc.get('1')},{dc.get('2')},{dc.get('3')},{dc.get('4')}] "
              f"({time.time()-t:.0f}s)", flush=True)
    print("DONE C scale-up")


if __name__ == "__main__":
    main()

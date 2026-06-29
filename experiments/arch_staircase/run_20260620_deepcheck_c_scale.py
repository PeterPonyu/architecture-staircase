#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "degree_staircase"
sys.path.insert(0, str(EXP))
sys.path.append(str(ROOT / "experiments"))
from train_staircase import Config, run  # noqa:E402

OUT = ROOT / "experiments/results/ultragoal_20260620_deepcheck/c_scale_extension"


def archive_incomplete(p: Path, reason: str) -> None:
    target = p.with_name(f"{p.name}.{reason}.{int(time.time())}")
    p.rename(target)
    print(f"archived incomplete evidence {p} -> {target}", flush=True)


def done(p: Path):
    if not p.exists():
        return None
    last = ""
    for line in p.read_text().splitlines():
        if line.strip():
            last = line
    if not last:
        archive_incomplete(p, "empty")
        return None
    try:
        obj = json.loads(last)
    except json.JSONDecodeError:
        archive_incomplete(p, "corrupt")
        return None
    summary = obj.get("_summary")
    if summary is None:
        archive_incomplete(p, "partial")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--opts", nargs="+", default=["adamw", "muon"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    cells = [
        (o, layer_count, s) for o in a.opts for layer_count in a.layers for s in a.seeds
    ]
    for i, (opt, layers, seed) in enumerate(cells, 1):
        p = OUT / f"{opt}_L24_l{layers}_s{seed}_steps{a.steps}.jsonl"
        s = done(p)
        if s is not None:
            print(f"[{i}/{len(cells)}] skip {p.name}", flush=True)
            summaries.append(s)
            continue
        cfg = Config(
            profile="staircase",
            D=4,
            L=24,
            d_model=256,
            n_heads=4,
            n_layers=layers,
            optimizer=opt,
            seed=seed,
            steps=a.steps,
            device="cuda",
        )
        print(f"[{i}/{len(cells)}] run {p.name}", flush=True)
        t = time.time()
        s, _ = run(cfg, out_path=str(p))
        s["elapsed_wall_sec"] = time.time() - t
        summaries.append(s)
        print(
            json.dumps(
                {
                    "opt": opt,
                    "layers": layers,
                    "seed": seed,
                    "fit": s.get("final_fit_corr"),
                    "deg4": s.get("final_deg_corr", {}).get("4"),
                }
            ),
            flush=True,
        )
    (OUT / "summaries.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=True) + "\n"
    )


if __name__ == "__main__":
    main()

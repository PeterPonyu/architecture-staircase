"""Direction 010 analysis — P1/P3 verdicts for the induction 3-D subspace test.

Reads results/induction_subspace/*.jsonl (5 L × 9 seeds, AdamW fixed) and, if
present, results/induction_subspace_ideal/*.jsonl (idealization-hardening arm).
Writes to results/figures-010/:
  subspace_verdicts.json
  fig_subspace.png   captured fraction (P1) and PCA dim (P3) vs L, seed scatter
"""
from __future__ import annotations
from pathlib import Path

import glob
import json
import os
from collections import defaultdict

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figstyle
figstyle.apply()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "experiments", "results")
FIG = os.path.join(RES, "figures-010")


def load_dir(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        last = None
        with open(p) as f:
            for line in f:
                last = line
        try:
            s = json.loads(last).get("_summary") if last else None
        except Exception:
            s = None
        if s:
            out.append(s)
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    main_runs = load_dir(os.path.join(RES, "induction_subspace"))
    ideal_runs = load_dir(os.path.join(RES, "induction_subspace_ideal"))
    print(f"main: {len(main_runs)} runs; ideal arm: {len(ideal_runs)} runs")

    cells = defaultdict(list)
    for s in main_runs:
        cells[s["seq_len"]].append(s)

    table = {"main": {}, "ideal_arm": {}}
    for L, ss in sorted(cells.items()):
        cap = [s["final_captured_total"] for s in ss]
        dim = [s["final_pca_dim"] for s in ss]
        table["main"][f"L{L}"] = {
            "n": len(ss),
            "captured_mean": float(np.mean(cap)), "captured_max": float(np.max(cap)),
            "pca_dim_mean": float(np.mean(dim)), "pca_dim_values": sorted(set(dim)),
        }
    for s in ideal_runs:
        key = f"b{s['batch_size']}_lr{s['lr']}_L{s['seq_len']}"
        table["ideal_arm"].setdefault(key, []).append(
            {"captured": s["final_captured_total"], "pca_dim": s["final_pca_dim"],
             "emergence": s["emergence_step"]})

    with open(os.path.join(FIG, "subspace_verdicts.json"), "w") as f:
        json.dump(table, f, indent=2)
    print("wrote subspace_verdicts.json")
    for L, row in table["main"].items():
        print(f"  {L}: captured mean {row['captured_mean']:.4f} (max {row['captured_max']:.4f}) "
              f"pca_dim {row['pca_dim_mean']:.1f} values {row['pca_dim_values']}")

    fig, axes = plt.subplots(1, 2, figsize=(figstyle.WIDTH_IN["col2_full"], 3.0))
    Ls = sorted(cells)
    cb_blue = figstyle.CB["blue"]
    for L in Ls:
        ss = cells[L]
        axes[0].scatter([L] * len(ss), [s["final_captured_total"] for s in ss],
                        color=cb_blue, alpha=0.55, s=24)
        axes[1].scatter([L] * len(ss), [s["final_pca_dim"] for s in ss],
                        color=cb_blue, alpha=0.55, s=24)
    # Left: positive log band only (all captures > 0; range ~1.5e-5..5e-3) so the
    # random baseline (6.1e-3) and the points sit in one readable decade band.
    axes[0].axhline(0.0061, color=figstyle.CB["vermillion"], ls="--", lw=1.2,
                    label="Random baseline (0.0061)")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-5, 1e-2)
    axes[0].set_ylabel("Ansatz-captured variance fraction")
    axes[0].set_title("(a) Ansatz-direction capture", loc="left", fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[1].axhline(3, color=figstyle.CB["vermillion"], ls="--", lw=1.2,
                    label="Claimed dim = 3")
    axes[1].set_ylabel("PCA components for 90% variance")
    axes[1].set_title("(b) Effective update dimension", loc="left", fontweight="bold")
    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.set_xlabel("Context length L")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_subspace.png"))
    plt.close(fig)
    print("wrote fig_subspace.png")


if __name__ == "__main__":
    main()

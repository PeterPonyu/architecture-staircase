"""Direction 017 analysis — A1 architecture anatomy of the degree staircase.

Tests findings-004's OWN "architecture-driven" attribution: does the per-degree
learning-time staircase rise with DEPTH and stay flat across WIDTH (P1), and
which component (attn vs mlp) owns the high-degree timing (P2)?

ROBUSTNESS NOTE (load-bearing): the runner's `staircase_span_ratio` is
ill-defined whenever any degree's half-time is 0 (learned at the first eval):
min→0 makes the ratio degenerate (it collapses to max), which is why span_ratio
shows std>mean across seeds. This analyzer therefore does NOT headline span_ratio.
It uses two robust quantities instead:
  - per-degree MEDIAN half-time across seeds (the staircase shape itself);
  - staircase rank index = mean over seeds of Spearman(degree, half_time)
    (+1 = clean ascending staircase, 0 = degrees learned together, <0 inverted);
  - final_fit_corr median (did the model actually fit the target).
P2 freeze arms are read WITHIN-arm only (per direction doc discipline).

Writes results/figures-017/: arch_verdicts.json, fig_arch.png.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import sys as _sys
_sys.path.insert(0, "/home/zeyufu/Desktop/dl-research/experiments")
import figstyle
figstyle.apply()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_THIS, "..", "..", "experiments", "results", "arch_staircase")
FIG = os.path.join(_THIS, "..", "..", "experiments", "results", "figures-017")

DEGREES = [1, 2, 3, 4]


def load():
    cells = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(RES, "*.jsonl"))):
        with open(path) as f:
            last = None
            for l in f:
                if l.strip():
                    last = l
        try:
            s = json.loads(last)["_summary"]
        except (KeyError, json.JSONDecodeError, TypeError):
            continue
        key = f"d{s['d_model']}_l{s['n_layers']}_h{s['n_heads']}_{s['freeze']}"
        cells[key].append(s)
    return cells


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def cell_metrics(runs):
    # per-degree half-times across seeds
    htimes = {d: [] for d in DEGREES}
    ranks, fits = [], []
    for s in runs:
        ht = s.get("half_times", {})
        vec = [ht.get(str(d), None) for d in DEGREES]
        for d, v in zip(DEGREES, vec):
            if v is not None:
                htimes[d].append(v)
        if all(v is not None for v in vec):
            ranks.append(spearman(DEGREES, vec))
        fits.append(s.get("final_fit_corr"))
    fits = [f for f in fits if f is not None]
    return {
        "n_seeds": len(runs),
        "half_median": {d: (float(np.median(htimes[d])) if htimes[d] else None)
                        for d in DEGREES},
        "half_iqr": {d: (float(np.subtract(*np.percentile(htimes[d], [75, 25])))
                         if len(htimes[d]) >= 2 else None) for d in DEGREES},
        "staircase_rank_index": float(np.mean(ranks)) if ranks else None,
        "staircase_rank_std": float(np.std(ranks)) if ranks else None,
        "fit_corr_median": float(np.median(fits)) if fits else None,
        "n_monotone": int(sum(1 for r in ranks if r >= 0.999)),
    }


def main():
    os.makedirs(FIG, exist_ok=True)
    cells = load()
    print(f"loaded {sum(len(v) for v in cells.values())} runs / {len(cells)} cells")
    M = {k: cell_metrics(v) for k, v in cells.items()}

    def get(key):
        return M.get(key, {})

    # ---- P1 width axis (L2, none): is the staircase flat across d_model? ----
    width_axis = {}
    for d in (64, 128, 256, 512):
        m = get(f"d{d}_l2_h4_none")
        if m:
            width_axis[d] = {"rank_index": m["staircase_rank_index"],
                             "rank_std": m["staircase_rank_std"],
                             "fit": m["fit_corr_median"],
                             "half_deg4": m["half_median"][4],
                             "n": m["n_seeds"]}

    # ---- P1 depth axis (d128, none): does it rise with n_layers? ----
    depth_axis = {}
    for L in (1, 2, 4):
        m = get(f"d128_l{L}_h4_none")
        if m:
            depth_axis[L] = {"rank_index": m["staircase_rank_index"],
                             "rank_std": m["staircase_rank_std"],
                             "fit": m["fit_corr_median"],
                             "half_deg4": m["half_median"][4],
                             "n": m["n_seeds"]}

    # ---- P2 freeze arms (d128_l2): within-arm component ownership ----
    freeze_arms = {}
    for fr in ("none", "attn", "mlp"):
        m = get(f"d128_l2_h4_{fr}")
        if m:
            freeze_arms[fr] = {"fit_median": m["fit_corr_median"],
                               "rank_index": m["staircase_rank_index"],
                               "half_median": m["half_median"],
                               "n": m["n_seeds"]}

    verdicts = {
        "robustness_note": "span_ratio ill-defined under half_time=0; rank index "
                           "+ per-degree medians used instead",
        "p1_width_axis": width_axis,
        "p1_depth_axis": depth_axis,
        "p2_freeze_arms": freeze_arms,
        "all_cells": M,
    }
    out = os.path.join(FIG, "arch_verdicts.json")
    with open(out, "w") as f:
        json.dump(verdicts, f, indent=1, default=str)
    print(f"wrote {out}")

    # ---- figure (figure*, two-column span ~7in) ----
    FREEZE_LABEL = {"none": "None\n(unfrozen)", "attn": "Attn\nfrozen",
                    "mlp": "MLP\nfrozen"}
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.WIDTH_IN["col2_full"], 2.7))
    ax = axes[0]
    ds = sorted(width_axis); ax.errorbar(
        ds, [width_axis[d]["rank_index"] for d in ds],
        yerr=[width_axis[d]["rank_std"] for d in ds], marker="o", capsize=4,
        color=figstyle.CB["blue"])
    ax.set_xscale("log"); ax.set_xlabel(r"$d_{\mathrm{model}}$ (2 layers)")
    ax.set_ylabel("Staircase rank index"); ax.set_ylim(-1.05, 1.05)
    ax.set_title("(a) Width axis", loc="left", fontweight="bold")

    ax = axes[1]
    Ls = sorted(depth_axis); ax.errorbar(
        Ls, [depth_axis[L]["rank_index"] for L in Ls],
        yerr=[depth_axis[L]["rank_std"] for L in Ls], marker="s",
        color=figstyle.CB["orange"], capsize=4)
    ax.set_xlabel("Number of layers (d=128)")
    ax.set_xticks(sorted(depth_axis))
    ax.set_ylabel("Staircase rank index")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("(b) Depth axis", loc="left", fontweight="bold")

    ax = axes[2]
    frs = [f for f in ("none", "attn", "mlp") if f in freeze_arms]
    x = range(len(frs))
    ax.bar([i - 0.2 for i in x], [freeze_arms[f]["fit_median"] for f in frs],
           width=0.38, label="Fit corr.", color=figstyle.CB["green"])
    ax.bar([i + 0.2 for i in x], [freeze_arms[f]["rank_index"] for f in frs],
           width=0.38, label="Rank index", color=figstyle.CB["vermillion"])
    ax.set_xticks(list(x)); ax.set_xticklabels([FREEZE_LABEL[f] for f in frs])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("(c) Freeze arms", loc="left", fontweight="bold")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fp = os.path.join(FIG, "fig_arch.png"); fig.savefig(fp)
    print(f"wrote {fp}")

    # ---- console ----
    print("\n=== P1 width axis (L2, none) — rank index | fit | deg4 half ===")
    for d in sorted(width_axis):
        w = width_axis[d]
        print(f"  d{d:<4d} rank={w['rank_index']:+.2f}±{w['rank_std']:.2f} "
              f"fit={w['fit']:.2f} deg4_half={w['half_deg4']} (n={w['n']})")
    print("=== P1 depth axis (d128, none) ===")
    for L in sorted(depth_axis):
        a = depth_axis[L]
        print(f"  L{L} rank={a['rank_index']:+.2f}±{a['rank_std']:.2f} "
              f"fit={a['fit']:.2f} deg4_half={a['half_deg4']} (n={a['n']})")
    print("=== P2 freeze arms (d128_l2, within-arm) ===")
    for fr in frs:
        a = freeze_arms[fr]
        print(f"  {fr:5s} fit={a['fit_median']:.2f} rank={a['rank_index']:+.2f} "
              f"half={a['half_median']} (n={a['n']})")


if __name__ == "__main__":
    main()

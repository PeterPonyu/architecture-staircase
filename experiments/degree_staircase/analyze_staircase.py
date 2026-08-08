"""Direction 004 analysis — P1/P2/P4(/P5) verdicts for the degree staircase.

Reads results/degree_staircase/*.jsonl. Writes to results/figures-004/:
  staircase_verdicts.json   per-cell T_k, gap ratios, SI, span_ratio (+ aggregates)
  fig_rho_curves.png        per-degree correlation trajectories, per optimizer
  fig_si_summary.png        SI / span_ratio by optimizer with seed scatter
  fig_pure3.png             pure degree-3 learning curves (P4 generic-speedup arm)

Definitions:
  T_k  : first step |corr_k| >= 0.5 * |final corr_k|  (probes.half_learning_times)
  SI   : geometric mean of adjacent gap ratios T_{k+1}/T_k over defined, ordered
         degrees (direction doc's staircase index; 1 = simultaneous, >>1 = staircase)
  span_ratio : max(T_k)/min(T_k) (scale-free; uniform speedups cancel — this is what
         makes P2's compression claim immune to P4's generic-speedup confound)
Also verifies the mixed==staircase duplication (scaffold placeholder) and treats
mixed as a determinism check, NOT an independent condition.
"""
from __future__ import annotations
from pathlib import Path

import glob
import json
import os
from collections import defaultdict

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments'))
import figstyle  # noqa: E402
figstyle.apply()

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "experiments", "results")
DIR = os.path.join(RES, "degree_staircase")
FIG = os.path.join(RES, "figures-004")
COLORS = {"adamw": figstyle.OPT["adamw"], "muon": figstyle.OPT["muon"],
          "sgdm": figstyle.OPT["sgdm"]}
OPT_LABEL = {"adamw": "AdamW", "muon": "Muon", "sgdm": "SGDM"}
DEG_LS = {1: "-", 2: "--", 3: "-.", 4: ":"}


def load(path):
    meta = summ = None
    hist = []
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            if "_meta" in o:
                meta = o["_meta"]
            elif "_summary" in o:
                summ = o["_summary"]
            else:
                hist.append(o)
    return meta, summ, hist


def corr_at(rec, k):
    dc = rec["deg_corr"]
    return abs(dc.get(str(k), dc.get(k, 0.0)))


def half_times(hist, degrees):
    finals = {k: corr_at(hist[-1], k) for k in degrees}
    out = {}
    for k in degrees:
        tgt = 0.5 * finals[k]
        out[k] = next((r["step"] for r in hist if tgt > 1e-6 and corr_at(r, k) >= tgt),
                      None)
    return out, finals


def si_of(ht):
    """Geometric mean of adjacent gap ratios over consecutively defined degrees."""
    ks = sorted(k for k, v in ht.items() if v is not None and v > 0)
    ratios = [ht[b] / ht[a] for a, b in zip(ks, ks[1:]) if ht[a] > 0]
    if not ratios:
        return None
    return float(np.exp(np.mean(np.log(ratios)))), ratios


def main():
    os.makedirs(FIG, exist_ok=True)
    runs = defaultdict(dict)   # (opt, profile) -> seed -> (summ, hist)
    for p in sorted(glob.glob(os.path.join(DIR, "*.jsonl"))):
        meta, summ, hist = load(p)
        if not (meta and summ and hist):
            continue
        runs[(meta["optimizer"], meta["profile"])][meta["seed"]] = (summ, hist)

    # ---- mixed == staircase placeholder verification ----
    dup_ok = True
    for opt in ["muon", "adamw", "sgdm"]:
        for seed in range(5):
            a = runs.get((opt, "staircase"), {}).get(seed)
            b = runs.get((opt, "mixed"), {}).get(seed)
            if a and b:
                same = (a[0].get("staircase_spread") == b[0].get("staircase_spread")
                        and a[0].get("final_fit_corr") == b[0].get("final_fit_corr"))
                dup_ok &= same
    print(f"mixed==staircase placeholder identity: {dup_ok} "
          f"(treated as determinism check, excluded from claims)")

    # ---- per-cell verdict table (staircase profile only) ----
    table = {}
    DEGREES = [1, 2, 3, 4]
    for opt in ["muon", "adamw", "sgdm"]:
        sis, spans, hts_all, finals_all = [], [], [], []
        for seed, (summ, hist) in sorted(runs.get((opt, "staircase"), {}).items()):
            ht, finals = half_times(hist, DEGREES)
            r = si_of(ht)
            hts_all.append(ht)
            finals_all.append(finals)
            if r:
                sis.append(r[0])
            defined = [v for v in ht.values() if v]
            if len(defined) >= 2 and min(defined) > 0:
                spans.append(max(defined) / min(defined))
        table[opt] = {
            "SI_mean": float(np.mean(sis)) if sis else None,
            "SI_std": float(np.std(sis)) if sis else None,
            "SI_per_seed": sis,
            "span_ratio_mean": float(np.mean(spans)) if spans else None,
            "span_ratio_per_seed": spans,
            "half_times_per_seed": [{str(k): v for k, v in h.items()} for h in hts_all],
            "final_corrs_mean": {str(k): float(np.mean([f[k] for f in finals_all]))
                                 for k in DEGREES} if finals_all else {},
        }
        print(f"{opt:6s} SI {table[opt]['SI_mean']} ± {table[opt]['SI_std']}  "
              f"span_ratio {table[opt]['span_ratio_mean']}")

    # ---- pure3 arm (P4) ----
    pure = {}
    for opt in ["muon", "adamw", "sgdm"]:
        t3s, fits = [], []
        for seed, (summ, hist) in sorted(runs.get((opt, "pure"), {}).items()):
            fits.append(summ.get("final_fit_corr"))
            ht, _ = half_times(hist, [3])
            t3s.append(ht.get(3))
        pure[opt] = {"final_fit_per_seed": fits,
                     "T3_half_per_seed": t3s}
        print(f"pure3 {opt:6s} fits {[('%.2f' % f) if f is not None else 'NA' for f in fits]} T3 {t3s}")
    table["_pure3"] = pure
    table["_meta"] = {"mixed_placeholder_identical": bool(dup_ok)}

    with open(os.path.join(FIG, "staircase_verdicts.json"), "w") as f:
        json.dump(table, f, indent=2)
    print("wrote staircase_verdicts.json")

    # ---- rho_k(t) curves (staircase profile, seed-mean) ----
    # Publication layout: avoid the old 15.5in-wide dashboard row, which was
    # compressed to \linewidth in the PDF and made labels too small.  Render at
    # final paper width with a stacked 3-panel layout.
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.2), sharey=True)
    for ax, opt in zip(axes, ["adamw", "sgdm", "muon"]):
        cell = runs.get((opt, "staircase"), {})
        if not cell:
            continue
        steps = [r["step"] for r in next(iter(cell.values()))[1]]
        for k in DEGREES:
            mat = []
            for seed, (summ, hist) in sorted(cell.items()):
                mat.append([corr_at(r, k) for r in hist])
            n = min(map(len, mat))
            arr = np.array([m[:n] for m in mat])
            ax.plot(steps[:n], arr.mean(0), DEG_LS[k], color=COLORS[opt],
                    label=f"deg {k}", alpha=0.9 - 0.12 * k)
        ax.set_title(f"{opt} (staircase task)")
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes:
        ax.set_ylabel("|corr|")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_rho_curves.png"), dpi=130)
    plt.close(fig)
    print("wrote fig_rho_curves.png")

    # ---- SI summary (single panel; span_ratio dropped — ill-defined when any
    #      half-time is 0, see findings-017) ----
    fig, ax = plt.subplots(figsize=(figstyle.WIDTH_IN["col2_full"], 3.2))
    for i, opt in enumerate(["adamw", "sgdm", "muon"]):
        vals = table[opt]["SI_per_seed"]
        jit = np.random.default_rng(7 + i).uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jit, vals, color=COLORS[opt],
                   s=42, alpha=0.8, zorder=3)
        if vals:
            ax.hlines(np.mean(vals), i - 0.28, i + 0.28, color=COLORS[opt],
                      lw=2.6, zorder=4)
    ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7,
               label="SI = 1 (simultaneous)")
    ax.set_xticks(range(3))
    ax.set_xticklabels([OPT_LABEL[o] for o in ["adamw", "sgdm", "muon"]])
    ax.set_ylabel("Staircase index SI")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_si_summary.png"))
    plt.close(fig)
    print("wrote fig_si_summary.png")

    # ---- pure3 curves ----
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for opt in ["adamw", "sgdm", "muon"]:
        cell = runs.get((opt, "pure"), {})
        for j, (seed, (summ, hist)) in enumerate(sorted(cell.items())):
            steps = [r["step"] for r in hist]
            ax.plot(steps, [corr_at(r, 3) for r in hist], color=COLORS[opt],
                    alpha=0.5, label=opt if j == 0 else None)
    ax.set_xlabel("step")
    ax.set_ylabel("|corr with χ_S3|")
    ax.set_title("P4 arm: pure degree-3 (no ladder) — generic speed & ladder need")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_pure3.png"), dpi=130)
    plt.close(fig)
    print("wrote fig_pure3.png")


if __name__ == "__main__":
    main()

"""C-T0-3 — bimodality-appropriate summaries + cadence-consistent timing numbers.

Three parts, all recomputed from raw jsonls (CPU-only):

(1) Scale-grid fit distribution (96 scalegrid runs): solved fraction
    (fit >= 0.999), plateau band (0.85-0.90), median/IQR of fit overall,
    at L=24, and per optimizer at L=24 — companions for the mean+-SD numbers
    in paper C Sec 3.5 (published: 29 solved / 65 plateau / 2 between; mean
    fit at L24 dips to 0.859, mean rank to 0.263).

(2) Freeze-grid mean-vs-median reconciliation for Table 3 (tab:ownershipopt)
    vs the Sec 3.3 text: AdamW n=15 (results/arch_staircase d128_l2 arms),
    Muon/SGDM n=5 (results/arch_staircase_optaxis). The table reports medians
    (0.170 / -0.000 / 0.461 for AdamW), the text reports means
    (0.214 / 0.035 / 0.309) — both are checked here, with solved fractions
    (deg-4 corr >= 0.3) and IQRs, since the deg-4 distribution is bimodal.

(3) Timing-table cadence audit for Table 7 (tab:timing): recompute emergence
    steps (first eval with icl_score >= 0.5) on the coarse 100-step main grid
    (n=10, all three optimizers) and the fine 10-step grid (n=5; Muon at all
    L, AdamW at L in {64,128}), so every Table 7 cell can be footnoted with
    its cadence and the matched-cadence (coarse-grid) ratios can be quoted.

Writes t03_bimodal_cadence.json next to this file.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_THIS, "..", "..", "results")


def load_summaries(folder, pattern="*.jsonl"):
    out = []
    for path in sorted(glob.glob(os.path.join(folder, pattern))):
        summ, hist = None, []
        with open(path) as f:
            for l in f:
                if not l.strip():
                    continue
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                if "_summary" in r:
                    summ = r["_summary"]
                elif "_meta" not in r:
                    hist.append(r)
        if summ is not None:
            out.append((os.path.basename(path), summ, hist))
    return out


def med_iqr(v):
    v = np.asarray(v, float)
    q1, q3 = np.percentile(v, [25, 75])
    return {"n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "median": float(np.median(v)), "q1": float(q1), "q3": float(q3),
            "iqr": float(q3 - q1)}


def main():
    report = {}

    # ---- (1) scale-grid fit distribution -------------------------------------
    runs = load_summaries(os.path.join(RES, "arch_staircase_scale"),
                          "scalegrid_*.jsonl")
    fits = np.array([s["final_fit_corr"] for _, s, _ in runs])
    solved = fits >= 0.999
    plateau = (fits >= 0.85) & (fits <= 0.90)
    between = ~solved & ~plateau
    L = np.array([int(n.split("_")[1][1:]) for n, _, _ in runs])
    opt = np.array([n.split("_")[5] for n, _, _ in runs])
    part1 = {
        "n_runs": len(runs),
        "n_solved_ge_0.999": int(solved.sum()),
        "n_plateau_0.85_0.90": int(plateau.sum()),
        "n_between": int(between.sum()),
        "fit_overall": med_iqr(fits),
        "solved_fraction_overall": float(solved.mean()),
        "by_L": {}, "L24_by_opt": {}}
    for l in (16, 24, 32):
        m = L == l
        part1["by_L"][l] = {**med_iqr(fits[m]),
                            "solved_fraction": float(solved[m].mean()),
                            "n_solved": int(solved[m].sum())}
    for o in ("adamw", "muon"):
        m = (L == 24) & (opt == o)
        part1["L24_by_opt"][o] = {**med_iqr(fits[m]),
                                  "solved_fraction": float(solved[m].mean())}
    # published "29 solved / 65 plateau" audit: those counts only reproduce over
    # scalegrid + the 48-run freeze8 audit combined (144 runs), NOT the 96
    # scale-grid runs the paper attributes them to; and the remainder is 50, not 2.
    fr8 = load_summaries(os.path.join(RES, "arch_staircase_scale"),
                         "freeze8_*.jsonl")
    fits144 = np.concatenate([fits, [s["final_fit_corr"] for _, s, _ in fr8]])
    part1["bands_144_with_freeze8"] = {
        "n": len(fits144),
        "n_solved_ge_0.999": int((fits144 >= 0.999).sum()),
        "n_plateau_0.85_0.90": int(((fits144 >= 0.85) & (fits144 <= 0.90)).sum()),
        "n_other": int(len(fits144) - (fits144 >= 0.999).sum()
                       - ((fits144 >= 0.85) & (fits144 <= 0.90)).sum())}
    part1["bands_96_scalegrid_only"] = {
        "n_solved_ge_0.999": int(solved.sum()),
        "n_plateau_0.85_0.90": int(plateau.sum()),
        "n_above_plateau_below_solved": int(((fits > 0.90) & (fits < 0.999)).sum()),
        "n_below_0.85": int((fits < 0.85).sum())}
    # published "mean fit dips to 0.859 at L24" = mean over the 8 L24 cells of the
    # per-cell MEDIAN fit (verified estimator)
    cells96 = defaultdict(list)
    for n, s, _ in runs:
        cells96["_".join(n.split("_")[:6])].append(s["final_fit_corr"])
    l24_cell_medians = [float(np.median(v)) for k, v in cells96.items()
                        if "_L24_" in k]
    part1["L24_mean_of_cell_median_fit"] = float(np.mean(l24_cell_medians))
    report["scalegrid_fit"] = part1

    # ---- (2) freeze grid: means vs medians vs solved fractions ---------------
    def deg4(s):
        dc = s["final_deg_corr"]
        return float(dc.get("4", dc.get(4)))

    part2 = {}
    # AdamW n=15 arms (arch_staircase)
    adamw = load_summaries(os.path.join(RES, "arch_staircase"), "d128_l2_h4_*.jsonl")
    arms = defaultdict(list)
    for name, s, _ in adamw:
        arms[s["freeze"]].append(s)
    for fr in ("none", "attn", "mlp"):
        d4 = [deg4(s) for s in arms[fr]]
        ft = [s["final_fit_corr"] for s in arms[fr]]
        part2[f"adamw_{fr}"] = {
            "deg4": med_iqr(d4),
            "deg4_learned_ge_0.3": int(sum(1 for v in d4 if v >= 0.3)),
            "deg4_fail_lt_0.05": int(sum(1 for v in d4 if v < 0.05)),
            "fit": med_iqr(ft)}
    # Muon/SGDM n=5 arms (arch_staircase_optaxis)
    optax = load_summaries(os.path.join(RES, "arch_staircase_optaxis"),
                           "d128_l2_h4_*.jsonl")
    arms2 = defaultdict(list)
    for name, s, _ in optax:
        arms2[(s["optimizer"], s["freeze"])].append(s)
    for (o, fr), ss in sorted(arms2.items()):
        d4 = [deg4(s) for s in ss]
        ft = [s["final_fit_corr"] for s in ss]
        part2[f"{o}_{fr}"] = {
            "deg4": med_iqr(d4),
            "deg4_learned_ge_0.3": int(sum(1 for v in d4 if v >= 0.3)),
            "fit": med_iqr(ft)}
    report["freeze_grid"] = part2

    # ---- (3) timing cadence audit --------------------------------------------
    def emergence(hist, thr=0.5):
        return next((r["step"] for r in hist if r.get("icl_score", -9) >= thr),
                    None)

    part3 = {"main_grid_100step": {}, "fine_grid_10step": {},
             "matched_cadence_ratios": {}}
    main_grid = load_summaries(os.path.join(RES, "induction_emergence"))
    cells = defaultdict(list)
    for name, s, hist in main_grid:
        e = emergence(hist)
        cells[(s["optimizer"], s["seq_len"])].append(
            (e, s.get("emergence_step")))
    for (o, l), v in sorted(cells.items()):
        es = [x[0] for x in v if x[0] is not None]
        part3["main_grid_100step"][f"{o}_L{l}"] = {
            **med_iqr(es), "n_no_emergence": sum(1 for x in v if x[0] is None),
            "summary_field_agrees": all(
                x[0] == x[1] for x in v if x[0] is not None and x[1] is not None)}
    fine = load_summaries(os.path.join(RES, "induction_fine"))
    fcells = defaultdict(list)
    for name, s, hist in fine:
        fcells[(s["optimizer"], s["seq_len"])].append(emergence(hist))
    for (o, l), v in sorted(fcells.items()):
        es = [x for x in v if x is not None]
        part3["fine_grid_10step"][f"{o}_L{l}"] = med_iqr(es)
    # matched-cadence (coarse) ratios vs Muon
    for l in (64, 128, 256):
        mu = np.mean([x[0] for x in cells[("muon", l)]])
        for o in ("adamw", "sgdm"):
            oo = np.mean([x[0] for x in cells[(o, l)]])
            part3["matched_cadence_ratios"][f"{o}_over_muon_L{l}"] = float(oo / mu)
    report["timing"] = part3

    with open(os.path.join(_THIS, "t03_bimodal_cadence.json"), "w") as f:
        json.dump(report, f, indent=1)

    # ---- console --------------------------------------------------------------
    p = part1
    print(f"(1) scalegrid: n={p['n_runs']} solved={p['n_solved_ge_0.999']} "
          f"plateau={p['n_plateau_0.85_0.90']} between={p['n_between']}")
    print(f"    fit overall: median={p['fit_overall']['median']:.3f} "
          f"IQR=[{p['fit_overall']['q1']:.3f},{p['fit_overall']['q3']:.3f}] "
          f"solved-frac={p['solved_fraction_overall']:.3f}")
    for l in (16, 24, 32):
        b = p["by_L"][l]
        print(f"    L{l}: mean={b['mean']:.3f} median={b['median']:.3f} "
              f"IQR=[{b['q1']:.3f},{b['q3']:.3f}] solved={b['n_solved']}/{b['n']}")
    for o, b in p["L24_by_opt"].items():
        print(f"    L24 {o}: mean={b['mean']:.3f} median={b['median']:.3f} "
              f"solved-frac={b['solved_fraction']:.3f}")
    print(f"    bands over 144 (scalegrid+freeze8): {p['bands_144_with_freeze8']}")
    print(f"    bands over 96 (scalegrid only): {p['bands_96_scalegrid_only']}")
    print(f"    L24 mean of per-cell median fit: "
          f"{p['L24_mean_of_cell_median_fit']:.4f} (published 0.859)")
    print("(2) freeze arms (deg4 mean | median | IQR | learned>=0.3):")
    for k, v in part2.items():
        d = v["deg4"]
        print(f"    {k:12s} mean={d['mean']:+.3f} med={d['median']:+.3f} "
              f"IQR=[{d['q1']:+.3f},{d['q3']:+.3f}] "
              f"learned={v['deg4_learned_ge_0.3']}/{d['n']} "
              f"fit mean={v['fit']['mean']:.3f} med={v['fit']['median']:.3f}")
    print("(3) timing:")
    for k, v in part3["main_grid_100step"].items():
        print(f"    main  {k:12s} mean={v['mean']:.0f} sd={v['sd']:.0f} "
              f"median={v['median']:.0f} n={v['n']} "
              f"summary_agrees={v['summary_field_agrees']}")
    for k, v in part3["fine_grid_10step"].items():
        print(f"    fine  {k:12s} mean={v['mean']:.0f} sd={v['sd']:.0f} n={v['n']}")
    for k, v in part3["matched_cadence_ratios"].items():
        print(f"    ratio {k}: {v:.2f}x")
    print("wrote t03_bimodal_cadence.json")


if __name__ == "__main__":
    main()

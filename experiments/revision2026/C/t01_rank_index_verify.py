"""C-T0-1 — verify the written rank-index definition reproduces published values.

The Methods definition being added to the manuscript source states:
  - per-degree half-time t_k: first evaluation step at which |rho_k(t)| >= 0.5*|rho_k(T)|,
    provided 0.5*|rho_k(T)| > 1e-6; otherwise undefined (degree never learned).
  - per-seed rank r = Spearman rank correlation between (1,2,3,4) and (t_1..t_4),
    computed only when all four half-times are defined; ties get distinct ordinal
    ranks in ascending-degree order (stable sort on length-4 vectors).
  - cell rank index = mean of r over included seeds.

This script recomputes, from raw jsonls, with EXACTLY that definition:
  (a) the 24-configuration scale grid: positive count, min, max
      (published: 24/24 positive, range +0.035 to +1.000; Sec. 3.5)
  (b) freeze8 L=24 d=256 Muon: attn-frozen rank (+1.000, 8/8) and unfrozen (+0.109)
  (c) d128 AdamW width/depth/freeze arms (analyze_arch.py populations)
plus a midrank (average-rank) tie-handling sensitivity on the same cells.

CPU-only; reads experiments/results/{arch_staircase,arch_staircase_scale}.
Writes t01_rank_index_verify.json next to this file.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_THIS, "..", "..", "results")
DEGREES = [1, 2, 3, 4]


def load_runs(folder):
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
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
                elif "_meta" in r:
                    continue
                else:
                    hist.append(r)
        if summ is not None:
            out.append((os.path.basename(path), summ, hist))
    return out


def half_times_from_hist(hist, degrees=DEGREES):
    """EXACT probes.half_learning_times semantics (recomputed from trajectory)."""
    if not hist:
        return {k: None for k in degrees}
    def corr_at(rec, k):
        dc = rec["deg_corr"]
        return abs(dc.get(str(k), dc.get(k, 0.0)))
    finals = {k: corr_at(hist[-1], k) for k in degrees}
    out = {}
    for k in degrees:
        tgt = 0.5 * finals[k]
        out[k] = next((r["step"] for r in hist if tgt > 1e-6 and corr_at(r, k) >= tgt),
                      None)
    return out


def spearman_ordinal(x, y):
    """Analyzer implementation: double-argsort ordinal ranks (ties -> sort order)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def _midranks(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v), float)
    i = 0
    sv = v[order]
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)  # average of positions i..j
        i = j + 1
    return ranks


def spearman_midrank(x, y):
    rx, ry = _midranks(x), _midranks(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def cell_rank(runs, use_summary_half_times=True, tie="ordinal"):
    """Rank index over runs; returns (mean, n_included, n_total, n_tied_seeds)."""
    f = spearman_ordinal if tie == "ordinal" else spearman_midrank
    ranks, n_tied = [], 0
    for _, summ, hist in runs:
        if use_summary_half_times and "half_times" in summ:
            ht = {int(k): v for k, v in summ["half_times"].items()}
        else:
            ht = half_times_from_hist(hist)
        vec = [ht.get(d) for d in DEGREES]
        if any(v is None for v in vec):
            continue
        if len(set(vec)) < len(vec):
            n_tied += 1
        ranks.append(f(DEGREES, vec))
    return ((float(np.mean(ranks)) if ranks else None), len(ranks), len(runs), n_tied)


def group_by(runs, keyfun):
    g = defaultdict(list)
    for r in runs:
        g[keyfun(r)].append(r)
    return g


def main():
    report = {}

    # ---- (a) 24-configuration scale grid -------------------------------------
    scale = load_runs(os.path.join(RES, "arch_staircase_scale"))
    sgrid = [r for r in scale if r[0].startswith("scalegrid_")]
    cells = group_by(sgrid, lambda r: "_".join(r[0].split("_")[:6]))
    grid = {}
    for cell, runs in sorted(cells.items()):
        m_ord, n_inc, n_tot, n_tie = cell_rank(runs, tie="ordinal")
        m_mid, _, _, _ = cell_rank(runs, tie="midrank")
        grid[cell] = {"rank_ordinal": m_ord, "rank_midrank": m_mid,
                      "n_included": n_inc, "n_runs": n_tot, "n_tied_seeds": n_tie}
    vals = [v["rank_ordinal"] for v in grid.values() if v["rank_ordinal"] is not None]
    mids = [v["rank_midrank"] for v in grid.values() if v["rank_midrank"] is not None]
    deltas = [abs(v["rank_ordinal"] - v["rank_midrank"]) for v in grid.values()
              if v["rank_ordinal"] is not None]
    report["scalegrid"] = {
        "n_configs": len(vals),
        "n_positive_ordinal": int(sum(1 for v in vals if v > 0)),
        "min_ordinal": float(min(vals)), "max_ordinal": float(max(vals)),
        "n_positive_midrank": int(sum(1 for v in mids if v > 0)),
        "min_midrank": float(min(mids)), "max_midrank": float(max(mids)),
        "max_abs_tie_delta": float(max(deltas)),
        "mean_abs_tie_delta": float(np.mean(deltas)),
        "published": "24/24 positive, range +0.035 to +1.000",
        "cells": grid,
    }

    # ---- (b) freeze8 Muon arms at L=24, d=256 ---------------------------------
    fr8 = [r for r in scale if r[0].startswith("freeze8_")]
    fcells = group_by(fr8, lambda r: "_".join(r[0].split("_")[:6]))
    fout = {}
    for cell, runs in sorted(fcells.items()):
        m_ord, n_inc, n_tot, n_tie = cell_rank(runs, tie="ordinal")
        m_mid, _, _, _ = cell_rank(runs, tie="midrank")
        fout[cell] = {"rank_ordinal": m_ord, "rank_midrank": m_mid,
                      "n_included": n_inc, "n_runs": n_tot, "n_tied_seeds": n_tie}
    report["freeze8"] = {
        "published": {"muon_attn": "+1.000 (8/8)", "muon_none": "+0.109",
                      "muon_mlp": "+0.435"},
        "cells": fout,
    }

    # ---- (c) d128 AdamW arch arms (analyze_arch populations) ------------------
    arch = load_runs(os.path.join(RES, "arch_staircase"))
    acells = group_by(arch, lambda r: (
        f"d{r[1]['d_model']}_l{r[1]['n_layers']}_{r[1]['freeze']}"))
    aout = {}
    for cell, runs in sorted(acells.items()):
        m_ord, n_inc, n_tot, n_tie = cell_rank(runs, tie="ordinal")
        m_mid, _, _, _ = cell_rank(runs, tie="midrank")
        aout[cell] = {"rank_ordinal": m_ord, "rank_midrank": m_mid,
                      "n_included": n_inc, "n_runs": n_tot, "n_tied_seeds": n_tie}
    report["arch_d128_adamw"] = aout

    # cross-check: summary half_times vs trajectory-recomputed half_times agree
    mismatch = 0
    for _, summ, hist in arch:
        if "half_times" not in summ or not hist:
            continue
        a = {int(k): v for k, v in summ["half_times"].items()}
        b = half_times_from_hist(hist)
        if a != b:
            mismatch += 1
    report["summary_vs_trajectory_half_time_mismatches"] = mismatch

    out = os.path.join(_THIS, "t01_rank_index_verify.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)

    print("== (a) scale grid ==")
    s = report["scalegrid"]
    print(f"  configs={s['n_configs']} positive={s['n_positive_ordinal']} "
          f"range=[{s['min_ordinal']:+.3f},{s['max_ordinal']:+.3f}]  "
          f"(published: {s['published']})")
    print(f"  midrank ties: positive={s['n_positive_midrank']} "
          f"range=[{s['min_midrank']:+.3f},{s['max_midrank']:+.3f}]  "
          f"max|delta|={s['max_abs_tie_delta']:.3f} mean={s['mean_abs_tie_delta']:.3f}")
    print("== (b) freeze8 ==")
    for c, v in fout.items():
        print(f"  {c}: ordinal={v['rank_ordinal']:+.3f} midrank={v['rank_midrank']:+.3f} "
              f"n={v['n_included']}/{v['n_runs']} tied_seeds={v['n_tied_seeds']}")
    print("== (c) d128 AdamW arms ==")
    for c, v in aout.items():
        r = v['rank_ordinal']
        print(f"  {c}: ordinal={'None' if r is None else format(r, '+.3f')} "
              f"n={v['n_included']}/{v['n_runs']} tied_seeds={v['n_tied_seeds']}")
    print(f"summary-vs-trajectory half-time mismatches: {mismatch}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

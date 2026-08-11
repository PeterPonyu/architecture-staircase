#!/usr/bin/env python3
"""Analyzer for the pre-submission gap battery (2026-07-05).

Reads the three C arms and emits one verdict JSON (+ console table). REFUSES
marker-less / incomplete run files: a file counts only if its last non-empty
line carries a "_summary" (LESSONS-AND-ERRATA sec 6c -- existence != complete).

  C#2 (c2_ownership) : per-arm final deg-4 corr (median over seeds) + overall
                       fit; BOTH-FROZEN FLOOR VERDICT (deg-4 dies -> redundancy
                       is a real two-path property; deg-4 survives -> the freeze
                       methodology is leaky). Per-layer knockout table.
                       Single-freeze Muon none/attn/mlp reference is pulled from
                       results/arch_staircase_optaxis if present (context only).
  C#1 (c1_targets)   : rank index (Spearman degree vs half-time) + spread (SI)
                       per (family, optimizer) + "order invariant?" flag
                       (ascending order preserved under inverse weighting).
  C#5 (c5_frontier)  : frontier table -- per (arm, opt) median deg-4 & deg-5
                       corr; does attn gate deg-5 while deg-4 stays MLP-reachable.

  python analyze_gapC_20260705.py            # writes verdict + prints table
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_THIS)
C_ROOT = os.path.join(_EXP, "results", "ieee_gap_20260705", "C")
OPTAXIS_REF = os.path.join(_EXP, "results", "arch_staircase_optaxis")
OUT = os.path.join(C_ROOT, "gapC_verdict.json")

# a deg-corr this small at the frontier counts as "not acquired"
ACQUIRE_THRESH = 0.15
FIT_GATE = 0.85  # SGDM/failed-fit runs have no geometry to interpret


def _load_summary(path):
    """Return the _summary dict iff the file's last non-empty line has one."""
    last = ""
    with open(path) as fh:
        for line in fh:
            if line.strip():
                last = line
    try:
        rec = json.loads(last)
    except (json.JSONDecodeError, ValueError):
        return None
    return rec.get("_summary")


def _load_all(folder):
    """{basename -> summary} for every COMPLETE file in folder."""
    out, incomplete = {}, []
    for path in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        s = _load_summary(path)
        if s is None:
            incomplete.append(os.path.basename(path))
        else:
            out[os.path.basename(path)] = s
    return out, incomplete


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def _median(xs):
    xs = [v for v in xs if v is not None]
    return float(np.median(xs)) if xs else None


# ---------------------------------------------------------------------------
# C#2 -- ownership localization
# ---------------------------------------------------------------------------
def analyze_c2():
    folder = os.path.join(C_ROOT, "c2_ownership")
    summ, incomplete = _load_all(folder)
    by_arm = defaultdict(list)
    for name, s in summ.items():
        by_arm[s.get("arm", name.split("_")[0])].append(s)

    arms = {}
    for arm, runs in sorted(by_arm.items()):
        deg4 = _median([r["final_deg_corr"].get("4") for r in runs])
        fit = _median([r.get("final_fit_corr") for r in runs])
        arms[arm] = {"n": len(runs), "deg4_med": deg4, "fit_med": fit,
                     "deg4_all": [r["final_deg_corr"].get("4") for r in runs]}

    # reference single-freeze Muon arms (context only)
    ref = {}
    if os.path.isdir(OPTAXIS_REF):
        rsumm, _ = _load_all(OPTAXIS_REF)
        rby = defaultdict(list)
        for name, s in rsumm.items():
            if s.get("optimizer") == "muon" and s.get("d_model") == 128 \
                    and s.get("n_layers") == 2:
                rby[s.get("freeze")].append(s)
        for fz, runs in rby.items():
            ref[fz] = {"n": len(runs),
                       "deg4_med": _median([r["final_deg_corr"].get("4") for r in runs])}

    both = arms.get("both", {})
    both_deg4 = both.get("deg4_med")
    if both_deg4 is None:
        verdict = "PENDING (no complete both-frozen runs)"
    elif abs(both_deg4) < ACQUIRE_THRESH:
        verdict = (f"TWO-PATH REAL: both-frozen deg-4={both_deg4:+.3f} dies "
                   f"(<{ACQUIRE_THRESH}) -> redundancy is a genuine hidden-network "
                   f"property, not a leaky readout")
    else:
        verdict = (f"METHODOLOGY LEAKY: both-frozen deg-4={both_deg4:+.3f} survives "
                   f"(>={ACQUIRE_THRESH}) -> embed/readout alone expresses deg-4; the "
                   f"single-freeze redundancy result is not load-bearing")

    return {"arms": arms, "single_freeze_muon_ref": ref,
            "both_frozen_verdict": verdict, "acquire_thresh": ACQUIRE_THRESH,
            "incomplete_files": incomplete}


# ---------------------------------------------------------------------------
# C#1 -- target-family generality
# ---------------------------------------------------------------------------
def analyze_c1():
    folder = os.path.join(C_ROOT, "c1_targets")
    summ, incomplete = _load_all(folder)
    # key by (family, optimizer); family/opt parsed from filename <fam>_<opt>_s<seed>
    by = defaultdict(list)
    for name, s in summ.items():
        parts = name[:-6].split("_")  # strip .jsonl
        fam, opt = parts[0], parts[1]
        by[(fam, opt)].append(s)

    cells = {}
    for (fam, opt), runs in sorted(by.items()):
        degrees = sorted(int(k) for k in runs[0]["final_deg_corr"].keys())
        ranks, spreads, fits = [], [], []
        half = {d: [] for d in degrees}
        for r in runs:
            ht = r.get("half_times", {})
            vec = [ht.get(str(d)) for d in degrees]
            for d, v in zip(degrees, vec):
                if v is not None:
                    half[d].append(v)
            if all(v is not None for v in vec) and r.get("n_learned", 0) >= 2:
                ranks.append(_spearman(degrees, vec))
            spreads.append(r.get("staircase_spread"))
            fits.append(r.get("final_fit_corr"))
        cells[f"{fam}|{opt}"] = {
            "family": fam, "optimizer": opt, "degrees": degrees, "n": len(runs),
            "rank_index": (float(np.mean(ranks)) if ranks else None),
            "rank_std": (float(np.std(ranks)) if ranks else None),
            "n_rankable": len(ranks),
            "spread_med": _median(spreads), "fit_med": _median(fits),
            "half_med": {d: _median(half[d]) for d in degrees},
        }

    # order-invariance: under inverse weighting (invw), is the ascending
    # low->high order still present (rank_index > 0) per optimizer?
    order_flags = {}
    for opt in ("adamw", "muon", "sgdm"):
        inv = cells.get(f"invw|{opt}")
        base = cells.get(f"geom|{opt}")
        ri_inv = inv["rank_index"] if inv else None
        ri_base = base["rank_index"] if base else None
        gated = inv and (inv["fit_med"] or 0) < FIT_GATE
        if ri_inv is None:
            flag = "PENDING"
        elif gated:
            flag = f"FIT-GATED (invw fit={inv['fit_med']:.2f}<{FIT_GATE}; order n/a)"
        elif ri_inv > 0.25:
            flag = (f"ORDER INTRINSIC: invw rank={ri_inv:+.2f} still ascending "
                    f"despite high-degree-dominant target")
        elif ri_inv < -0.25:
            flag = f"ORDER TARGET-DRIVEN: invw rank={ri_inv:+.2f} flips to descending"
        else:
            flag = f"ORDER WASHED OUT: invw rank={ri_inv:+.2f} (no clear order)"
        order_flags[opt] = {"invw_rank_index": ri_inv, "geom_rank_index": ri_base,
                            "invw_fit_med": (inv["fit_med"] if inv else None),
                            "verdict": flag}

    return {"cells": cells, "order_invariance": order_flags,
            "fit_gate": FIT_GATE, "incomplete_files": incomplete}


# ---------------------------------------------------------------------------
# C#5 -- frontier-degree freeze grid
# ---------------------------------------------------------------------------
def analyze_c5():
    folder = os.path.join(C_ROOT, "c5_frontier")
    summ, incomplete = _load_all(folder)
    by = defaultdict(list)
    for name, s in summ.items():
        parts = name[:-6].split("_")  # <arm>_<opt>_s<seed>
        arm, opt = parts[0], parts[1]
        by[(arm, opt)].append(s)

    table = {}
    for (arm, opt), runs in sorted(by.items()):
        table[f"{arm}|{opt}"] = {
            "arm": arm, "optimizer": opt, "n": len(runs),
            "deg4_med": _median([r["final_deg_corr"].get("4") for r in runs]),
            "deg5_med": _median([r["final_deg_corr"].get("5") for r in runs]),
            "fit_med": _median([r.get("final_fit_corr") for r in runs]),
        }

    # frontier verdict per optimizer: attn gates deg-5, deg-4 stays MLP-reachable?
    verdicts = {}
    for opt in ("adamw", "muon"):
        none_c = table.get(f"none|{opt}")
        attn_c = table.get(f"attn|{opt}")
        mlp_c = table.get(f"mlp|{opt}")
        if not (none_c and attn_c):
            verdicts[opt] = "PENDING"
            continue
        d5_none, d5_attn = none_c["deg5_med"], attn_c["deg5_med"]
        d4_none, d4_attn = none_c["deg4_med"], attn_c["deg4_med"]
        gates5 = (d5_none is not None and d5_attn is not None
                  and d5_none >= ACQUIRE_THRESH and d5_attn < ACQUIRE_THRESH)
        keeps4 = (d4_attn is not None and d4_attn >= ACQUIRE_THRESH)
        if gates5 and keeps4:
            v = ("FRONTIER OWNERSHIP UPGRADE: attn-freeze kills deg-5 "
                 f"({d5_none:+.2f}->{d5_attn:+.2f}) but preserves deg-4 "
                 f"({d4_attn:+.2f}) -> attention owns the FRONTIER degree")
        elif d5_none is not None and d5_none < ACQUIRE_THRESH:
            v = (f"DEG-5 NOT ACQUIRED even unfrozen (none deg5={d5_none:+.2f}) -> "
                 "frontier freeze test inconclusive (target too hard at 4000 steps)")
        else:
            v = (f"NO CLEAN FRONTIER GATING: deg5 none={d5_none} attn={d5_attn}, "
                 f"deg4 attn={d4_attn} (see table)")
        verdicts[opt] = {"deg5_none": d5_none, "deg5_attn": d5_attn,
                         "deg4_none": d4_none, "deg4_attn": d4_attn,
                         "deg4_mlp": (mlp_c["deg4_med"] if mlp_c else None),
                         "deg5_mlp": (mlp_c["deg5_med"] if mlp_c else None),
                         "verdict": v}

    return {"table": table, "frontier_verdict": verdicts,
            "acquire_thresh": ACQUIRE_THRESH, "incomplete_files": incomplete}


def main():
    verdict = {"c2_ownership": analyze_c2(),
               "c1_targets": analyze_c1(),
               "c5_frontier": analyze_c5()}
    os.makedirs(C_ROOT, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(verdict, f, indent=1, default=str)

    print(f"wrote {OUT}\n")
    # ---- C#2 ----
    c2 = verdict["c2_ownership"]
    print("== C#2 MUON OWNERSHIP (deg-4 corr per arm; both = floor control) ==")
    for arm, r in c2["arms"].items():
        d4 = r["deg4_med"]
        print(f"   {arm:8s} deg4={d4:+.3f} fit={r['fit_med']:.2f} n={r['n']}"
              if d4 is not None else f"   {arm:8s} (no data)")
    if c2["single_freeze_muon_ref"]:
        print("   [ref single-freeze muon]:",
              {k: round(v['deg4_med'], 3) if v['deg4_med'] is not None else None
               for k, v in c2["single_freeze_muon_ref"].items()})
    print("   VERDICT:", c2["both_frozen_verdict"])
    if c2["incomplete_files"]:
        print("   INCOMPLETE (ignored):", c2["incomplete_files"])

    # ---- C#1 ----
    print("\n== C#1 TARGET-FAMILY (rank index; +1 ascending staircase) ==")
    for key, r in verdict["c1_targets"]["cells"].items():
        ri = r["rank_index"]
        ris = f"{ri:+.2f}±{r['rank_std']:.2f}" if ri is not None else "n/a"
        print(f"   {key:14s} degrees={r['degrees']} rank={ris} "
              f"spread={r['spread_med']} fit={r['fit_med']:.2f} n={r['n']}")
    print("   ORDER INVARIANCE (under inverse weighting):")
    for opt, r in verdict["c1_targets"]["order_invariance"].items():
        print(f"     {opt:5s}: {r['verdict']}")
    if verdict["c1_targets"]["incomplete_files"]:
        print("   INCOMPLETE (ignored):", verdict["c1_targets"]["incomplete_files"])

    # ---- C#5 ----
    print("\n== C#5 FRONTIER (deg-4 / deg-5 corr per arm) ==")
    for key, r in verdict["c5_frontier"]["table"].items():
        print(f"   {key:12s} deg4={r['deg4_med']} deg5={r['deg5_med']} "
              f"fit={r['fit_med']:.2f} n={r['n']}")
    for opt, r in verdict["c5_frontier"]["frontier_verdict"].items():
        v = r["verdict"] if isinstance(r, dict) else r
        print(f"   {opt} VERDICT: {v}")
    if verdict["c5_frontier"]["incomplete_files"]:
        print("   INCOMPLETE (ignored):", verdict["c5_frontier"]["incomplete_files"])


if __name__ == "__main__":
    main()

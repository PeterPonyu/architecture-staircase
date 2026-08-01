"""C revision — Fisher exact tests on degree-4 acquisition/fail fractions for the
primary freeze contrasts, to accompany the Welch tests on bimodal per-seed
degree-4 correlations (referee suggestion, 2026-07 revision round).

Contrasts (all counts recomputed from raw jsonl summaries, never recalled):
  (a) AdamW, d=128, L=16, n=15/arm (results/arch_staircase d128_l2_h4_*):
      none vs attn, none vs mlp — fail criterion deg4 < 0.1 (the criterion the
      Sec 3.3 sentence and fig:case caption quote: 14/15 vs 6/15 vs 5/15) and
      acquisition criterion deg4 >= 0.3 (tab:ownershipopt fractions 6/1/9 of 15).
  (b) AdamW, L=24, d=256, n=8/arm (freeze8): none vs attn (4/8 vs 8/8 fail).
  (c) Muon, L=24, d=256, n=8/arm (freeze8): none vs attn (7/8 vs 0/8 fail).
  (d) Muon/SGDM d=128 optaxis arms (n=5) for completeness.

Writes t06_fisher_freeze.json next to this file.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

from scipy.stats import fisher_exact

_THIS = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_THIS, "..", "..", "results")


def summaries(folder, pattern):
    out = []
    for path in sorted(glob.glob(os.path.join(RES, folder, pattern))):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "_summary" in r:
                    out.append((os.path.basename(path), r["_summary"]))
    return out


def deg4(s):
    dc = s["final_deg_corr"]
    return float(dc.get("4", dc.get(4)))


def counts(vals):
    return {
        "n": len(vals),
        "fail_lt_0.1": sum(1 for v in vals if v < 0.1),
        "acq_ge_0.3": sum(1 for v in vals if v >= 0.3),
        "values": [round(v, 4) for v in sorted(vals)],
    }


def fisher(c1, c2, key):
    """2x2 Fisher exact on criterion `key` between two count dicts."""
    a, b = c1[key], c1["n"] - c1[key]
    c, d = c2[key], c2["n"] - c2[key]
    for alt in ("two-sided",):
        odds, p = fisher_exact([[a, b], [c, d]], alternative=alt)
    return {"table": [[a, b], [c, d]], "p_two_sided": float(p)}


def main():
    report = {}

    # (a) AdamW d=128 n=15
    arms = defaultdict(list)
    for name, s in summaries("arch_staircase", "d128_l2_h4_*.jsonl"):
        arms[s["freeze"]].append(deg4(s))
    a = {fr: counts(v) for fr, v in arms.items()}
    report["adamw_d128"] = {
        "counts": a,
        "fisher_fail_none_vs_attn": fisher(a["none"], a["attn"], "fail_lt_0.1"),
        "fisher_fail_none_vs_mlp": fisher(a["none"], a["mlp"], "fail_lt_0.1"),
        "fisher_acq_none_vs_attn": fisher(a["none"], a["attn"], "acq_ge_0.3"),
        "fisher_acq_none_vs_mlp": fisher(a["none"], a["mlp"], "acq_ge_0.3"),
    }

    # (b)+(c) freeze8 L=24 d=256 n=8, both optimizers
    cells = defaultdict(list)
    for name, s in summaries("arch_staircase_scale", "freeze8_*.jsonl"):
        cells[(s["optimizer"], s["freeze"])].append(deg4(s))
    for opt in ("adamw", "muon"):
        c = {fr: counts(cells[(opt, fr)]) for fr in ("none", "attn", "mlp")}
        report[f"{opt}_L24_d256"] = {
            "counts": c,
            "fisher_fail_none_vs_attn": fisher(c["none"], c["attn"], "fail_lt_0.1"),
            "fisher_fail_none_vs_mlp": fisher(c["none"], c["mlp"], "fail_lt_0.1"),
        }

    # (d) Muon/SGDM d=128 optaxis n=5
    cells = defaultdict(list)
    for name, s in summaries("arch_staircase_optaxis", "d128_l2_h4_*.jsonl"):
        cells[(s["optimizer"], s["freeze"])].append(deg4(s))
    for opt in ("muon", "sgdm"):
        have = {fr: counts(cells[(opt, fr)])
                for fr in ("none", "attn", "mlp") if cells[(opt, fr)]}
        rep = {"counts": have}
        if "none" in have and "attn" in have:
            rep["fisher_acq_none_vs_attn"] = fisher(have["none"], have["attn"],
                                                    "acq_ge_0.3")
        report[f"{opt}_d128_optaxis"] = rep

    out = os.path.join(_THIS, "t06_fisher_freeze.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

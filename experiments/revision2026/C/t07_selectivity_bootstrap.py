"""C revision — percentile bootstrap CIs on the trained-model ablation
selectivity scalar (referee suggestion, 2026-07 revision round).

Selectivity of an ablation condition (paper Sec `ablation`, tab:ablation):
    selectivity = Delta_4 - mean(Delta_{1..3}),
where Delta_k = median-over-seeds deg_corr_k(unablated) - median deg_corr_k(cond)
at the final checkpoint (step 4000). Point estimates in print: layer-1 MLP
+0.369 (AdamW) / +0.344 (Muon); layer-1 attn +0.279 / +0.204.

Bootstrap: resample the 8 seeds with replacement (per optimizer), recompute the
median-based selectivity for attn_l1 and mlp_l1 from the SAME resampled seed
set (paired), 20,000 resamples, percentile 95% CI; also the paired difference
selectivity(mlp_l1) - selectivity(attn_l1).

Data: experiments/revision2026/cg3-C/*.ablate.jsonl (8 seeds x {adamw, muon}).
Writes t07_selectivity_bootstrap.json next to this file.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
CG3 = os.path.join(_THIS, "..", "cg3-C")
DEGREES = ["1", "2", "3", "4"]
CONDS = ["none", "attn_l1", "mlp_l1"]
B = 20000
SEED = 20260717


def load_final(path):
    recs = {}
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            if "condition" not in o:
                continue
            c = o["condition"]
            if c not in recs or o["step"] > recs[c]["step"]:
                recs[c] = o
    return recs


def selectivity(dc_none, dc_cond):
    """dc_*: (n_seeds, 4) arrays; median-based selectivity scalar."""
    med_n = np.median(dc_none, axis=0)
    med_c = np.median(dc_cond, axis=0)
    delta = med_n - med_c
    return float(delta[3] - delta[:3].mean())


def main():
    rng = np.random.default_rng(SEED)
    report = {}
    for opt in ("adamw", "muon"):
        files = sorted(glob.glob(os.path.join(CG3, f"*_{opt}.ablate.jsonl")))
        finals = [load_final(p) for p in files]
        n = len(finals)
        dc = {c: np.array([[f[c]["deg_corr"][d] for d in DEGREES] for f in finals])
              for c in CONDS}
        point = {c: selectivity(dc["none"], dc[c]) for c in ("attn_l1", "mlp_l1")}
        point["mlp_minus_attn"] = point["mlp_l1"] - point["attn_l1"]

        boots = {k: np.empty(B) for k in ("attn_l1", "mlp_l1", "mlp_minus_attn")}
        for b in range(B):
            idx = rng.integers(0, n, n)
            s_attn = selectivity(dc["none"][idx], dc["attn_l1"][idx])
            s_mlp = selectivity(dc["none"][idx], dc["mlp_l1"][idx])
            boots["attn_l1"][b] = s_attn
            boots["mlp_l1"][b] = s_mlp
            boots["mlp_minus_attn"][b] = s_mlp - s_attn

        rep = {"n_seeds": n, "n_resamples": B}
        for k in boots:
            lo, hi = np.percentile(boots[k], [2.5, 97.5])
            rep[k] = {"point": round(point[k], 4),
                      "ci95": [round(float(lo), 4), round(float(hi), 4)],
                      "excludes_zero": bool(lo > 0 or hi < 0)}
        report[opt] = rep

    out = os.path.join(_THIS, "t07_selectivity_bootstrap.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

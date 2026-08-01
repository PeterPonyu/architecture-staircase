"""C-G3 verification — recompute the activation-patching verdict from raw jsonls.

Reads the 16 BOX-4 ablation logs (experiments/revision2026/cg3-C/*.ablate.jsonl:
8 seeds x {adamw, muon}, 7 conditions x 12 checkpoints each) and recomputes,
independently of cg3_verdict.json:

  - per condition, at the FINAL checkpoint (step 4000): median-over-seeds
    deg_corr vector and median fit;
  - deltas vs the unablated model: delta_d4 = med(none)_4 - med(cond)_4,
    delta_d123 = mean over degrees 1-3 of the median drops;
  - selectivity = delta_d4 - delta_d123 (positive = deg-4-selective damage);
  - the structural-degeneracy check on full-attn and attn-l0 ablation
    (fit and all deg_corr identically ~0: the token embedding enters the
    residual stream only through layer-0 attention in this architecture).

Cross-checks every recomputed number against cg3_verdict.json (tolerance 1e-6)
and prints a paper-facing summary. Writes t05_cg3_verify.json.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
CG3 = os.path.join(_THIS, "..", "cg3-C")
DEGREES = ["1", "2", "3", "4"]
CONDS = ["none", "attn", "mlp", "attn_l0", "attn_l1", "mlp_l0", "mlp_l1"]


def load_final(path):
    """condition -> final-step record."""
    recs = {}
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            if "condition" not in o:  # _meta / _summary lines
                continue
            c = o["condition"]
            if c not in recs or o["step"] > recs[c]["step"]:
                recs[c] = o
    return recs


def main():
    per_opt = {}
    for opt in ("adamw", "muon"):
        files = sorted(glob.glob(os.path.join(CG3, f"*_{opt}.ablate.jsonl")))
        finals = [load_final(p) for p in files]
        conds = {}
        for c in CONDS:
            dc = np.array([[f[c]["deg_corr"][d] for d in DEGREES] for f in finals])
            fit = np.array([f[c]["fit_corr"] for f in finals])
            conds[c] = {"deg_corr_med": [float(np.median(dc[:, i])) for i in range(4)],
                        "fit_med": float(np.median(fit)), "n": len(finals)}
        deltas = {}
        none = conds["none"]["deg_corr_med"]
        for c in CONDS[1:]:
            m = conds[c]["deg_corr_med"]
            d4 = none[3] - m[3]
            d123 = float(np.mean([none[i] - m[i] for i in range(3)]))
            deltas[c] = {"delta_d4": d4, "delta_d123": d123,
                         "selectivity": d4 - d123,
                         "fit_med": conds[c]["fit_med"]}
        per_opt[opt] = {"n_files": len(files), "conditions": conds,
                        "deltas": deltas}

    # degeneracy check: full-attn and attn-l0 ablation
    degen = {}
    for opt in ("adamw", "muon"):
        files = sorted(glob.glob(os.path.join(CG3, f"*_{opt}.ablate.jsonl")))
        attn_all_zero = True
        l0_max_absfit = 0.0
        for p in files:
            with open(p) as f:
                for line in f:
                    o = json.loads(line)
                    if "condition" not in o:
                        continue
                    if o["condition"] == "attn":
                        if abs(o["fit_corr"]) > 1e-9 or any(
                                abs(v) > 1e-9 for v in o["deg_corr"].values()):
                            attn_all_zero = False
                    if o["condition"] == "attn_l0" and o["step"] == 4000:
                        l0_max_absfit = max(l0_max_absfit, abs(o["fit_corr"]))
        degen[opt] = {"full_attn_identically_zero_all_ckpts": attn_all_zero,
                      "attn_l0_final_max_abs_fit": l0_max_absfit}

    # cross-check vs the BOX-4 verdict json
    with open(os.path.join(CG3, "cg3_verdict.json")) as f:
        v = json.load(f)
    mism = []
    for opt in ("adamw", "muon"):
        for c in CONDS:
            a = per_opt[opt]["conditions"][c]
            b = v["per_optimizer"][opt]["conditions"][c]
            if any(abs(x - y) > 1e-6 for x, y in zip(a["deg_corr_med"],
                                                     b["deg_corr_med"])) \
               or abs(a["fit_med"] - b["fit_med"]) > 1e-6 or a["n"] != b["n"]:
                mism.append(f"{opt}/{c}/conditions")
        for c in CONDS[1:]:
            a = per_opt[opt]["deltas"][c]
            b = v["per_optimizer"][opt]["deltas"][c]
            for k in ("delta_d4", "delta_d123", "selectivity"):
                if abs(a[k] - b[k]) > 1e-6:
                    mism.append(f"{opt}/{c}/{k}")

    out = {"per_optimizer": per_opt, "degeneracy": degen,
           "verdict_json_mismatches": mism}
    with open(os.path.join(_THIS, "t05_cg3_verify.json"), "w") as f:
        json.dump(out, f, indent=1)

    print(f"verdict-json cross-check mismatches: {len(mism)} {mism}")
    for opt in ("adamw", "muon"):
        print(f"\n== {opt} (n={per_opt[opt]['n_files']}) ==")
        for c in CONDS:
            m = per_opt[opt]["conditions"][c]
            dc = " ".join(f"{x:+.3f}" for x in m["deg_corr_med"])
            print(f"  {c:8s} fit={m['fit_med']:+.3f}  deg_corr_med=[{dc}]")
        for c in CONDS[1:]:
            d = per_opt[opt]["deltas"][c]
            print(f"  Δ {c:8s} d4={d['delta_d4']:+.3f} d123={d['delta_d123']:+.3f} "
                  f"sel={d['selectivity']:+.3f}")
    print(f"\ndegeneracy: {json.dumps(degen, indent=1)}")


if __name__ == "__main__":
    main()

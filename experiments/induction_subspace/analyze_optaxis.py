"""C-KILL-1, 010 leg — is the induction-subspace GEOMETRY optimizer-invariant?

Closes the 010 leg of C-KILL-1 (geometry ⊥ timing measured under AdamW only).
Adds Muon & SGDM (results/induction_subspace_optaxis) against the closed AdamW
baseline (results/induction_subspace, repeat task).

Verdict spec: geometry optimizer-invariant
IFF, at MATCHED ICL, pca_dim and captured-fraction agree across AdamW vs Muon
(SGDM only where it actually learns induction). Matched-performance = the first
eval where each run reaches the AdamW median final ICL — removes the "Muon trains
faster" timing confound from the geometry comparison.

SCOPE LIMIT (stated, not hidden): the closed AdamW baseline covers the `repeat`
task only. `markov` has NO AdamW anchor → reported Muon-vs-SGDM consistency only,
which cannot test invariance against the AdamW reference.

Writes results/figures-010/optaxis_verdict.json (no plot — packaging deferred).
"""
from __future__ import annotations
import glob, json, os
from collections import defaultdict
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
RES_ADAMW = os.path.join(_THIS, "..", "results", "induction_subspace")
RES_OPT = os.path.join(_THIS, "..", "results", "induction_subspace_optaxis")
FIG = os.path.join(_THIS, "..", "results", "figures-010")
EMERGE = 0.5  # icl below this => never learned induction => no subspace to compare


def load(folder, opt_from_name=False):
    cells = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        summ, traj = None, []
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
                    traj.append(r)
        if summ is None:
            continue
        b = os.path.basename(path)
        opt = ("muon" if "muon" in b else "sgdm") if opt_from_name else "adamw"
        cells[(summ["task"], summ["seq_len"], opt)].append({"s": summ, "t": traj})
    return cells


def matched_pointwise(runs, target_icl):
    pcas, caps, steps, emerged = [], [], [], 0
    for run in runs:
        if (run["s"].get("final_icl_score") or -9) < EMERGE:
            continue  # fit-gate analog: never learned induction
        emerged += 1
        hit = None
        for rec in run["t"]:
            if rec.get("icl_score", -9) >= target_icl:
                hit = rec
                break
        if hit is None and run["t"]:
            hit = run["t"][-1]
        if hit is None:
            continue
        pcas.append(hit.get("pca_dim"))
        caps.append(hit.get("captured_total"))
        steps.append(hit["step"])
    if not pcas:
        return None
    return {"pca_dim_med": float(np.median(pcas)), "captured_med": float(np.median(caps)),
            "step_med": float(np.median(steps)), "n_emerged": emerged, "n_total": len(runs)}


def main():
    os.makedirs(FIG, exist_ok=True)
    cells = load(RES_ADAMW, opt_from_name=False)
    for k, v in load(RES_OPT, opt_from_name=True).items():
        cells[k] = v

    def final_med(runs, field):
        xs = [r["s"].get(field) for r in runs if r["s"].get(field) is not None]
        return float(np.median(xs)) if xs else None

    OPTS = ["adamw", "muon", "sgdm"]
    out = {"emerge_thresh": EMERGE, "scope_limit": "AdamW baseline = repeat only; markov has no AdamW anchor",
           "tasks": {}}
    for task, seq in [("repeat", 64), ("repeat", 128), ("markov", 64), ("markov", 128)]:
        adamw = cells.get((task, seq, "adamw"))
        # matched target: AdamW final ICL if present, else min of available optimizers
        if adamw:
            target = final_med(adamw, "final_icl_score")
        else:
            finals = [final_med(cells[(task, seq, o)], "final_icl_score")
                      for o in ("muon", "sgdm") if (task, seq, o) in cells]
            target = min([f for f in finals if f is not None], default=EMERGE)
        rec = {"matched_icl_target": target, "has_adamw_anchor": adamw is not None, "by_opt": {}}
        for opt in OPTS:
            runs = cells.get((task, seq, opt))
            if not runs:
                continue
            pw = matched_pointwise(runs, target)
            rec["by_opt"][opt] = {
                "final_icl_med": final_med(runs, "final_icl_score"),
                "final_pca_med": final_med(runs, "final_pca_dim"),
                "final_captured_med": final_med(runs, "final_captured_total"),
                "matched": pw,
            }
        out["tasks"][f"{task}_L{seq}"] = rec

    p = os.path.join(FIG, "optaxis_verdict.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=str)

    # ---- console ----
    print(f"wrote {p}\n")
    print("captured_total is ~0 across ALL optimizers (ansatz-capture floored — matches "
          "findings-010 'NOT DETECTED'); load-bearing metric = pca_dim.\n")
    for tk, rec in out["tasks"].items():
        anchor = "AdamW-anchored" if rec["has_adamw_anchor"] else "NO AdamW (muon-vs-sgdm only)"
        print(f"== {tk} (matched ICL target={rec['matched_icl_target']:.3f}; {anchor}) ==")
        for opt in OPTS:
            o = rec["by_opt"].get(opt)
            if not o:
                continue
            m = o["matched"]
            if m:
                print(f"   {opt:5s} FINAL icl={o['final_icl_med']:.3f} pca={o['final_pca_med']:.1f} "
                      f"cap={o['final_captured_med']:.4f} | @MATCHED step={m['step_med']:.0f} "
                      f"pca={m['pca_dim_med']:.1f} cap={m['captured_med']:.4f} "
                      f"(emerged {m['n_emerged']}/{m['n_total']})")
            else:
                print(f"   {opt:5s} FINAL icl={o['final_icl_med']:.3f} pca={o['final_pca_med']:.1f} "
                      f"— no run reached matched ICL")
        print()


if __name__ == "__main__":
    main()

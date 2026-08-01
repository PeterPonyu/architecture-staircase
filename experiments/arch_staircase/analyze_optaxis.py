"""C-KILL-1, 017 leg — is the degree-staircase GEOMETRY optimizer-invariant?

The internal red-team audit corrected the verdict spec; this analyzer follows it
verbatim:

  * Geometry ⊥ timing  ⇔  at MATCHED performance the degree-acquisition ORDER
    (Spearman rank of degree vs half-time) and the POINTWISE deg_corr agree across
    AdamW (results/arch_staircase) vs Muon vs SGDM (results/arch_staircase_optaxis),
    while ABSOLUTE half-times differ (Muon faster).
  * DROP staircase_spread (absolute steps; Muon compresses mechanically).
  * Freeze arms: compare DELTAS (attn-minus-none, mlp-minus-none), never absolute
    geometry — "Muon" acts on a different param set per freeze arm.
  * SGDM fit-gate: exclude any cell with median final_fit_corr < 0.85 (a run that
    did not learn the function has no geometry to be invariant). SGDM is a
    fit-floor control → this closes the leg AdamW+Muon (2 legs), not 1→3.
  * Exclude nl=1 from the staircase verdict (no staircase by 017's own finding);
    keep nl=1 as a control.

Writes results/figures-017/optaxis_verdict.json (no plot — packaging deferred).
"""
from __future__ import annotations
import glob, json, os
from collections import defaultdict
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
RES_ADAMW = os.path.join(_THIS, "..", "results", "arch_staircase")
RES_OPT = os.path.join(_THIS, "..", "results", "arch_staircase_optaxis")
FIG = os.path.join(_THIS, "..", "results", "figures-017")
DEGREES = [1, 2, 3, 4]
FIT_GATE = 0.85


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def load(folder):
    """returns dict[(cell,opt)] -> list of {summary, traj}"""
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
        cell = f"d{summ['d_model']}_l{summ['n_layers']}_{summ['freeze']}"
        opt = summ.get("optimizer", "adamw")
        cells[(cell, opt)].append({"s": summ, "t": traj})
    return cells


def matched_pointwise(runs, target_fit):
    """median deg_corr vector + step at the first eval each run reaches target_fit."""
    vecs, steps = [], []
    for run in runs:
        hit = None
        for rec in run["t"]:
            if rec.get("fit_corr", -9) >= target_fit:
                hit = rec
                break
        if hit is None and run["t"]:
            hit = run["t"][-1]  # never reached target -> use final (flagged by step==last)
        if hit is None:
            continue
        dc = hit.get("deg_corr", {})
        vecs.append([dc.get(str(d), np.nan) for d in DEGREES])
        steps.append(hit["step"])
    if not vecs:
        return None
    arr = np.array(vecs, float)
    return {
        "deg_corr_med": [float(np.nanmedian(arr[:, i])) for i in range(4)],
        "step_med": float(np.median(steps)),
        "n": len(vecs),
    }


def final_degcorr(runs):
    vecs = []
    for run in runs:
        if not run["t"]:
            continue
        dc = run["t"][-1].get("deg_corr", {})
        vecs.append([dc.get(str(d), np.nan) for d in DEGREES])
    if not vecs:
        return [None] * 4
    arr = np.array(vecs, float)
    return [float(np.nanmedian(arr[:, i])) for i in range(4)]


def cell_metrics(runs):
    ranks, fits = [], []
    htimes = {d: [] for d in DEGREES}
    for run in runs:
        s = run["s"]
        ht = s.get("half_times", {})
        vec = [ht.get(str(d)) for d in DEGREES]
        for d, v in zip(DEGREES, vec):
            if v is not None:
                htimes[d].append(v)
        if all(v is not None for v in vec) and s.get("n_learned", 0) >= 2:
            ranks.append(spearman(DEGREES, vec))
        if s.get("final_fit_corr") is not None:
            fits.append(s["final_fit_corr"])
    return {
        "n": len(runs),
        "fit_med": float(np.median(fits)) if fits else None,
        "rank_index": float(np.mean(ranks)) if ranks else None,
        "rank_std": float(np.std(ranks)) if ranks else None,
        "half_med": {d: (float(np.median(htimes[d])) if htimes[d] else None) for d in DEGREES},
        "final_deg_corr": final_degcorr(runs),
    }


def main():
    os.makedirs(FIG, exist_ok=True)
    cells = load(RES_ADAMW)
    for k, v in load(RES_OPT).items():
        cells[k] = v
    M = {f"{c}|{o}": cell_metrics(r) for (c, o), r in cells.items()}

    OPTS = ["adamw", "muon", "sgdm"]

    def g(cell, opt):
        return M.get(f"{cell}|{opt}")

    # ---- staircase ORDER across optimizers (exclude nl=1) ----
    order = {}
    for cell in ("d128_l2_none", "d128_l4_none", "d128_l1_none"):  # l1 = control
        order[cell] = {}
        for opt in OPTS:
            m = g(cell, opt)
            if m:
                gated = (opt == "sgdm" and (m["fit_med"] or 0) < FIT_GATE)
                order[cell][opt] = {"rank_index": m["rank_index"], "rank_std": m["rank_std"],
                                    "fit_med": m["fit_med"], "n": m["n"],
                                    "fit_gated_out": gated, "deg4_half_med": m["half_med"][4]}

    # ---- matched-fit pointwise deg_corr on the main staircase cell ----
    base = g("d128_l2_none", "adamw")
    target = base["fit_med"] if base else 0.85
    pointwise = {"target_fit": target, "note": "deg_corr at first eval reaching AdamW median final fit"}
    for opt in OPTS:
        runs = cells.get(("d128_l2_none", opt))
        if runs:
            pw = matched_pointwise(runs, target)
            if pw:
                gated = (opt == "sgdm" and ((g("d128_l2_none", opt) or {}).get("fit_med") or 0) < FIT_GATE)
                pw["fit_gated_out"] = gated
                pointwise[opt] = pw

    # ---- freeze DELTAS (d128_l2): deg-4 ownership, attn-none & mlp-none ----
    freeze = {}
    for opt in OPTS:
        none_m, attn_m, mlp_m = g("d128_l2_none", opt), g("d128_l2_attn", opt), g("d128_l2_mlp", opt)
        if not none_m:
            continue
        d4 = lambda m: (m["final_deg_corr"][3] if m and m["final_deg_corr"][3] is not None else None)
        rec = {"none_deg4": d4(none_m), "attn_deg4": d4(attn_m), "mlp_deg4": d4(mlp_m),
               "none_fit": none_m["fit_med"],
               "attn_fit": attn_m["fit_med"] if attn_m else None,
               "mlp_fit": mlp_m["fit_med"] if mlp_m else None}
        if rec["attn_deg4"] is not None and rec["none_deg4"] is not None:
            rec["delta_attn_minus_none"] = rec["attn_deg4"] - rec["none_deg4"]
        if rec["mlp_deg4"] is not None and rec["none_deg4"] is not None:
            rec["delta_mlp_minus_none"] = rec["mlp_deg4"] - rec["none_deg4"]
        rec["fit_gated_out"] = (opt == "sgdm" and (none_m["fit_med"] or 0) < FIT_GATE)
        freeze[opt] = rec

    verdict = {"fit_gate": FIT_GATE, "staircase_order": order,
               "matched_pointwise_deg_corr": pointwise, "freeze_deltas_deg4": freeze}
    out = os.path.join(FIG, "optaxis_verdict.json")
    with open(out, "w") as f:
        json.dump(verdict, f, indent=1, default=str)

    # ---- console ----
    print(f"wrote {out}\n")
    print("== STAIRCASE ORDER (rank index; +1 ascending staircase). nl=1 = control ==")
    for cell in ("d128_l1_none", "d128_l2_none", "d128_l4_none"):
        print(f" {cell}:")
        for opt in OPTS:
            r = order[cell].get(opt)
            if r:
                flag = " [SGDM fit<0.85 EXCLUDED]" if r["fit_gated_out"] else ""
                print(f"   {opt:5s} rank={r['rank_index']:+.2f}±{r['rank_std']:.2f} "
                      f"fit={r['fit_med']:.2f} deg4_half(med)={r['deg4_half_med']} n={r['n']}{flag}")
    print("\n== MATCHED-FIT POINTWISE deg_corr (target fit = AdamW %.2f) ==" % target)
    for opt in OPTS:
        pw = pointwise.get(opt)
        if pw:
            flag = " [GATED]" if pw.get("fit_gated_out") else ""
            dc = ", ".join(f"d{d}={v:+.2f}" for d, v in zip(DEGREES, pw["deg_corr_med"]))
            print(f"   {opt:5s} step@match(med)={pw['step_med']:.0f}  [{dc}]  n={pw['n']}{flag}")
    print("\n== FREEZE DELTAS — deg-4 final corr (attn should KILL deg4, mlp PRESERVE) ==")
    for opt in OPTS:
        fr = freeze.get(opt)
        if fr:
            flag = " [SGDM GATED]" if fr["fit_gated_out"] else ""
            print(f"   {opt:5s} none_deg4={fr['none_deg4']} attn_deg4={fr['attn_deg4']} "
                  f"mlp_deg4={fr['mlp_deg4']}")
            print(f"         Δattn={fr.get('delta_attn_minus_none')}  "
                  f"Δmlp={fr.get('delta_mlp_minus_none')}  "
                  f"(fits none/attn/mlp = {fr['none_fit']:.2f}/"
                  f"{(fr['attn_fit'] or float('nan')):.2f}/{(fr['mlp_fit'] or float('nan')):.2f}){flag}")


if __name__ == "__main__":
    main()

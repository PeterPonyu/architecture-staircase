"""C-T0-2 — TOST multiplicity correction + SESOI sensitivity for the manuscript Sec 3.1.

Recomputes the three pairwise Welch TOSTs on the per-seed staircase index
(SI = geometric mean of adjacent half-time ratios; EXACT 004 estimator, n=15
per optimizer) from the raw jsonls, then:
  1. applies Bonferroni (alpha' = 0.05/3) and Holm corrections across the three
     pairwise tests (family = the 3 pairwise equivalence claims);
  2. reports the TOST CI at both the nominal level (90%) and the
     Bonferroni-corrected level (100*(1-2*alpha')% = 96.67%);
  3. runs a SESOI sensitivity sweep over margins {1.0, 1.5, 2.0} SI units
     (and the log-scale factor margins {1.25, 1.5, 2.0} as secondary).

Published values being checked (the manuscript, Table 2 / Sec 3.1):
  SI mean+-SD: AdamW 2.98+-1.39, SGDM 3.20+-1.61, Muon 3.53+-1.54
  muon-adamw: diff +0.55, 90% CI [-0.36,+1.46], p=0.044
  muon-sgdm : diff +0.33, 90% CI [-0.65,+1.31], p=0.026
  adamw-sgdm: diff -0.22, 90% CI [-1.15,+0.72], p=0.014

CPU-only. Writes t02_tost_multiplicity.json next to this file.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy import stats

_THIS = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(_THIS, "..", "..", "results", "degree_staircase")
DEGREES = [1, 2, 3, 4]
ALPHA = 0.05
K = 3  # number of pairwise tests in the family


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
        out[k] = next((r["step"] for r in hist
                       if tgt > 1e-6 and corr_at(r, k) >= tgt), None)
    return out


def si_of(ht):
    ks = sorted(k for k, v in ht.items() if v is not None and v > 0)
    ratios = [ht[b] / ht[a] for a, b in zip(ks, ks[1:]) if ht[a] > 0]
    if not ratios:
        return None
    return float(np.exp(np.mean(np.log(ratios))))


def si_per_seed(opt):
    out = []
    for p in sorted(glob.glob(os.path.join(DIR, f"{opt}_staircase_s*.jsonl"))):
        meta, summ, hist = load(p)
        if not (meta and summ and hist):
            continue
        v = si_of(half_times(hist, DEGREES))
        if v is not None:
            out.append((meta["seed"], v))
    return out


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return d, se, df


def tost(a, b, delta, alpha):
    """Welch TOST; returns p_TOST and the 100(1-2*alpha)% CI decision."""
    d, se, df = welch(a, b)
    p_lower = stats.t.sf((d + delta) / se, df)   # H0: d <= -delta
    p_upper = stats.t.cdf((d - delta) / se, df)  # H0: d >= +delta
    p = max(float(p_lower), float(p_upper))
    crit = stats.t.ppf(1 - alpha, df)
    ci = (float(d - crit * se), float(d + crit * se))
    return {"diff": float(d), "se": float(se), "df": float(df),
            "ci_level": 1 - 2 * alpha, "ci": ci, "p_tost": p,
            "equivalent": bool(ci[0] > -delta and ci[1] < delta)}


def main():
    si = {opt: si_per_seed(opt) for opt in ("muon", "adamw", "sgdm")}
    vals = {o: [v for _, v in si[o]] for o in si}
    print("SI per optimizer:")
    for o in ("adamw", "sgdm", "muon"):
        v = np.array(vals[o])
        print(f"  {o:6s} n={len(v):2d} mean={v.mean():.3f} sd={v.std(ddof=1):.3f} "
              f"median={np.median(v):.3f}")

    pairs = [("muon", "adamw"), ("muon", "sgdm"), ("adamw", "sgdm")]
    out = {"si_summary": {o: {"n": len(vals[o]),
                              "mean": float(np.mean(vals[o])),
                              "sd": float(np.std(vals[o], ddof=1)),
                              "median": float(np.median(vals[o]))} for o in vals},
           "family_size": K, "alpha_nominal": ALPHA,
           "alpha_bonferroni": ALPHA / K,
           "margins_linear": [1.0, 1.5, 2.0],
           "margins_log_factor": [1.25, 1.5, 2.0],
           "tests": {}}

    # --- main margin 1.5, nominal + Bonferroni CIs, linear + log ---
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        rec = {}
        rec["linear_margin1.5_nominal"] = tost(vals[a], vals[b], 1.5, ALPHA)
        rec["linear_margin1.5_bonferroni"] = tost(vals[a], vals[b], 1.5, ALPHA / K)
        rec["log_margin1.5_nominal"] = tost(np.log(vals[a]), np.log(vals[b]),
                                            np.log(1.5), ALPHA)
        rec["log_margin1.5_bonferroni"] = tost(np.log(vals[a]), np.log(vals[b]),
                                               np.log(1.5), ALPHA / K)
        out["tests"][key] = rec

    # --- Holm on the linear margin-1.5 p-values ---
    ps = {f"{a}_vs_{b}": out["tests"][f"{a}_vs_{b}"]["linear_margin1.5_nominal"]["p_tost"]
          for a, b in pairs}
    order = sorted(ps, key=ps.get)
    holm = {}
    rejected_so_far = True
    for i, k in enumerate(order):
        thresh = ALPHA / (K - i)
        rej = rejected_so_far and (ps[k] <= thresh)
        holm[k] = {"p": ps[k], "holm_threshold": thresh, "equivalent_holm": rej}
        rejected_so_far = rej
    bonf = {k: {"p": ps[k], "threshold": ALPHA / K,
                "equivalent_bonferroni": ps[k] <= ALPHA / K} for k in ps}
    out["holm"] = holm
    out["bonferroni"] = bonf

    # --- SESOI sensitivity table (nominal alpha AND Bonferroni alpha) ---
    sens = {}
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        sens[key] = {}
        for delta in (1.0, 1.5, 2.0):
            nom = tost(vals[a], vals[b], delta, ALPHA)
            cor = tost(vals[a], vals[b], delta, ALPHA / K)
            sens[key][f"margin_{delta}"] = {
                "p_tost": nom["p_tost"],
                "equivalent_nominal_0.05": nom["equivalent"],
                "equivalent_bonferroni_0.0167": cor["equivalent"],
                "ci90": nom["ci"], "ci96.7": cor["ci"]}
        for fac in (1.25, 1.5, 2.0):
            nom = tost(np.log(vals[a]), np.log(vals[b]), np.log(fac), ALPHA)
            sens[key][f"log_factor_{fac}"] = {
                "p_tost": nom["p_tost"],
                "equivalent_nominal_0.05": nom["equivalent"]}
    out["sensitivity"] = sens

    with open(os.path.join(_THIS, "t02_tost_multiplicity.json"), "w") as f:
        json.dump(out, f, indent=1)

    print("\n== margin 1.5 SI, per pair ==")
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        n = out["tests"][key]["linear_margin1.5_nominal"]
        c = out["tests"][key]["linear_margin1.5_bonferroni"]
        print(f"  {key:15s} diff={n['diff']:+.3f} 90%CI=[{n['ci'][0]:+.3f},{n['ci'][1]:+.3f}] "
              f"p={n['p_tost']:.4f} | 96.7%CI=[{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] "
              f"equiv@Bonf={c['equivalent']}")
    print("\n== Holm ==")
    for k in order:
        h = holm[k]
        print(f"  {k:15s} p={h['p']:.4f} vs {h['holm_threshold']:.4f} "
              f"-> {'EQUIV' if h['equivalent_holm'] else 'NOT equiv (inconclusive)'}")
    print("\n== SESOI sensitivity (p_TOST; nominal / Bonferroni verdicts) ==")
    for key in sens:
        row = []
        for delta in (1.0, 1.5, 2.0):
            s = sens[key][f"margin_{delta}"]
            row.append(f"D={delta}: p={s['p_tost']:.4f} "
                       f"{'N+' if s['equivalent_nominal_0.05'] else 'N-'}"
                       f"{'B+' if s['equivalent_bonferroni_0.0167'] else 'B-'}")
        print(f"  {key:15s} " + "  ".join(row))
    print("\nwrote t02_tost_multiplicity.json")


if __name__ == "__main__":
    main()

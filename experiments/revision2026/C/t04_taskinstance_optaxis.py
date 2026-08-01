"""C1 task-instance generality — matched-fit analyze_optaxis adjudication (CPU).

Applies paper C's own ordering conventions (Methods, "Ordering estimators and
fit gate"; code truth analyze_optaxis.py / t01_rank_index_verify.py /
t02_tost_multiplicity.py) to the 30 P1-C1 runs (3 fresh monomial draws
t2101/t2102/t2103 x {adamw, muon} x seeds 100-104) and to the 16 P1-C2
fresh-seed baseline runs (original published target, seeds 100-107):

  - half-time t_k: first eval step with |rho_k(t)| >= 0.5|rho_k(T)|, defined
    only when 0.5|rho_k(T)| > 1e-6 (EXACT probes.half_learning_times
    semantics, recomputed from trajectory and cross-checked vs _summary);
  - per-seed rank index: Spearman((1..4), (t_1..t_4)), seed included only if
    all four half-times defined; ordinal ties (d=128 body convention) with a
    midrank sensitivity column;
  - SI: per-seed geometric mean of consecutive-defined half-time ratios
    (EXACT t02/004 estimator);
  - fit gate: cell median final_fit_corr >= 0.85 (FIT_GATE of
    analyze_optaxis.py); cells below the gate are flagged, and a seed-level
    sensitivity (rank over seeds with final fit >= 0.85) is reported;
  - matched-fit pointwise deg_corr: per draw, median deg_corr vector at the
    first eval reaching that draw's AdamW median final fit (EXACT
    matched_pointwise of analyze_optaxis.py).

Verification block first: the same pipeline must reproduce the published d128
values (rank d128_l2_none AdamW +0.667 n=15 / Muon +0.52 n=5, fit_med
0.898/1.000; SI means AdamW 2.98 / SGDM 3.20 / Muon 3.53, n=15).

Writes t04_taskinstance_optaxis.json next to this file.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(_THIS, "..", "..")
C1 = os.path.join(ROOT, "revision2026", "pilot-CE1", "results", "p1_c1")
C2 = os.path.join(ROOT, "revision2026", "pilot-CE1", "results", "p1_c2")
ARCH = os.path.join(ROOT, "results", "arch_staircase")
OPTAX = os.path.join(ROOT, "results", "arch_staircase_optaxis")
STAIR = os.path.join(ROOT, "results", "degree_staircase")
DEGREES = [1, 2, 3, 4]
FIT_GATE = 0.85


def load_runs(pattern):
    out = []
    for path in sorted(glob.glob(pattern)):
        meta = summ = None
        hist = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "_meta" in r:
                    meta = r["_meta"]
                elif "_summary" in r:
                    summ = r["_summary"]
                else:
                    hist.append(r)
        if summ is not None:
            out.append({"file": os.path.basename(path), "meta": meta,
                        "s": summ, "t": hist})
    return out


def corr_at(rec, k):
    dc = rec["deg_corr"]
    return abs(dc.get(str(k), dc.get(k, 0.0)))


def half_times(hist, degrees=DEGREES):
    """EXACT probes.half_learning_times semantics from the trajectory."""
    if not hist:
        return {k: None for k in degrees}
    finals = {k: corr_at(hist[-1], k) for k in degrees}
    out = {}
    for k in degrees:
        tgt = 0.5 * finals[k]
        out[k] = next((r["step"] for r in hist
                       if tgt > 1e-6 and corr_at(r, k) >= tgt), None)
    return out


def spearman_ordinal(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
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
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman_midrank(x, y):
    rx, ry = _midranks(x), _midranks(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def si_of(ht):
    """EXACT t02/004 estimator: geo-mean of consecutive-defined ratios (>0)."""
    ks = sorted(k for k, v in ht.items() if v is not None and v > 0)
    ratios = [ht[b] / ht[a] for a, b in zip(ks, ks[1:]) if ht[a] > 0]
    if not ratios:
        return None
    return float(np.exp(np.mean(np.log(ratios))))


def matched_pointwise(runs, target_fit):
    """EXACT analyze_optaxis.matched_pointwise: deg_corr at first eval
    reaching target_fit (final eval if never reached)."""
    vecs, steps, n_never = [], [], 0
    for run in runs:
        hit = None
        for rec in run["t"]:
            if rec.get("fit_corr", -9) >= target_fit:
                hit = rec
                break
        if hit is None and run["t"]:
            hit = run["t"][-1]
            n_never += 1
        if hit is None:
            continue
        dc = hit.get("deg_corr", {})
        vecs.append([dc.get(str(d), np.nan) for d in DEGREES])
        steps.append(hit["step"])
    if not vecs:
        return None
    arr = np.array(vecs, float)
    return {"deg_corr_med": [float(np.nanmedian(arr[:, i])) for i in range(4)],
            "step_med": float(np.median(steps)), "n": len(vecs),
            "n_never_reached": n_never}


def cell_metrics(runs):
    """Per-cell: fit gate, rank index (ordinal + midrank), SI, per-seed detail."""
    per_seed = []
    ht_mismatch = 0
    for run in runs:
        s, hist = run["s"], run["t"]
        ht = half_times(hist)
        if "half_times" in s:
            sht = {int(k): v for k, v in s["half_times"].items()}
            # summary encodes half-times directly; 0 is a real step value
            if sht != {k: (v if v is not None else None) for k, v in ht.items()} \
               and any(sht.get(k) != ht[k] for k in DEGREES):
                ht_mismatch += 1
        vec = [ht[d] for d in DEGREES]
        fit = s.get("final_fit_corr")
        fd4 = s.get("final_deg_corr", {}).get("4")
        rec = {"seed": s.get("seed"), "final_fit": fit,
               "final_deg4": fd4,
               "deg4_acquired": (fd4 is not None and abs(fd4) >= 0.3),
               "half_times": {d: ht[d] for d in DEGREES},
               "all_defined": all(v is not None for v in vec),
               "si": si_of(ht)}
        if rec["all_defined"]:
            rec["rank_ordinal"] = spearman_ordinal(DEGREES, vec)
            rec["rank_midrank"] = spearman_midrank(DEGREES, vec)
        per_seed.append(rec)

    fits = [r["final_fit"] for r in per_seed if r["final_fit"] is not None]
    fit_med = float(np.median(fits)) if fits else None
    inc = [r for r in per_seed if r["all_defined"]]
    inc_fit = [r for r in inc if (r["final_fit"] or -9) >= FIT_GATE]
    sis = [r["si"] for r in per_seed if r["si"] is not None]

    def mean_of(key, rows):
        v = [r[key] for r in rows if key in r]
        return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (None, None, 0)

    m_ord = mean_of("rank_ordinal", inc)
    m_mid = mean_of("rank_midrank", inc)
    m_ord_f = mean_of("rank_ordinal", inc_fit)
    inc_acq = [r for r in inc if r["deg4_acquired"]]
    m_ord_a = mean_of("rank_ordinal", inc_acq)
    return {
        "n": len(runs),
        "fit_med": fit_med,
        "fit_per_seed": sorted(round(f, 3) for f in fits),
        "clears_fit_gate": (fit_med is not None and fit_med >= FIT_GATE),
        "rank_index": m_ord[0], "rank_std": m_ord[1], "n_rank": m_ord[2],
        "rank_index_midrank": m_mid[0],
        "rank_index_fitgated_seeds": m_ord_f[0],
        "n_rank_fitgated_seeds": m_ord_f[2],
        "n_deg4_acquired": len(inc_acq),
        "rank_index_deg4acq_seeds": m_ord_a[0],
        "si_mean": float(np.mean(sis)) if sis else None,
        "si_sd": float(np.std(sis, ddof=1)) if len(sis) > 1 else None,
        "si_median": float(np.median(sis)) if sis else None,
        "n_si": len(sis),
        "half_med": {d: (float(np.median([r["half_times"][d] for r in per_seed
                                          if r["half_times"][d] is not None]))
                         if any(r["half_times"][d] is not None for r in per_seed)
                         else None) for d in DEGREES},
        "summary_vs_traj_halftime_mismatches": ht_mismatch,
        "per_seed": per_seed,
    }


def main():
    report = {"fit_gate": FIT_GATE}

    # ---------------- verification: published d128 values ----------------
    ver = {}
    arch = load_runs(os.path.join(ARCH, "d128_l2_h4_*.jsonl"))
    optax = load_runs(os.path.join(OPTAX, "*.jsonl"))
    pools = defaultdict(list)
    for r in arch:
        s = r["s"]
        if s["d_model"] == 128 and s["n_layers"] == 2 and s["freeze"] == "none":
            pools[("d128_l2_none", s.get("optimizer", "adamw"))].append(r)
    for r in optax:
        s = r["s"]
        if s["d_model"] == 128 and s["n_layers"] == 2 and s["freeze"] == "none":
            pools[("d128_l2_none", s.get("optimizer", "adamw"))].append(r)
    for opt, pub_rank, pub_n, pub_fit in [("adamw", 0.667, 15, 0.898),
                                          ("muon", 0.520, 5, 1.000)]:
        m = cell_metrics(pools[("d128_l2_none", opt)])
        ver[f"d128_l2_none_{opt}"] = {
            "rank_index": m["rank_index"], "n_rank": m["n_rank"],
            "fit_med": m["fit_med"],
            "published": {"rank": pub_rank, "n": pub_n, "fit_med": pub_fit},
            "match": (m["rank_index"] is not None
                      and abs(m["rank_index"] - pub_rank) < 5e-3)}
    si_pub = {"adamw": 2.98, "sgdm": 3.20, "muon": 3.53}
    for opt in ("adamw", "sgdm", "muon"):
        runs = load_runs(os.path.join(STAIR, f"{opt}_staircase_s*.jsonl"))
        sis = [si_of(half_times(r["t"])) for r in runs]
        sis = [v for v in sis if v is not None]
        ver[f"SI_{opt}"] = {"n": len(sis), "mean": float(np.mean(sis)),
                            "sd": float(np.std(sis, ddof=1)),
                            "published_mean": si_pub[opt],
                            "match": abs(float(np.mean(sis)) - si_pub[opt]) < 5e-3}
    report["verification"] = ver

    # ---------------- C1: 3 fresh draws x 2 optimizers ----------------
    c1 = load_runs(os.path.join(C1, "*.jsonl"))
    draws = defaultdict(list)
    for r in c1:
        draws[(r["s"]["target_seed"], r["s"].get("optimizer", "adamw"))].append(r)
    c1_out = {}
    for (ts, opt), runs in sorted(draws.items()):
        c1_out[f"t{ts}_{opt}"] = cell_metrics(runs)
    # matched-fit pointwise per draw (target = draw's AdamW median final fit)
    mp = {}
    for ts in sorted({k[0] for k in draws}):
        base = c1_out[f"t{ts}_adamw"]["fit_med"]
        mp[f"t{ts}"] = {"target_fit": base}
        for opt in ("adamw", "muon"):
            r = matched_pointwise(draws[(ts, opt)], base)
            if r:
                mp[f"t{ts}"][opt] = r
    report["c1_cells"] = c1_out
    report["c1_matched_pointwise"] = mp

    # ---------------- C2: fresh-seed baseline (original target) ----------------
    c2 = load_runs(os.path.join(C2, "*.jsonl"))
    base = defaultdict(list)
    for r in c2:
        base[r["s"].get("optimizer", "adamw")].append(r)
    report["c2_baseline"] = {opt: cell_metrics(runs) for opt, runs in base.items()}

    out = os.path.join(_THIS, "t04_taskinstance_optaxis.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1, default=str)

    # ---------------- console ----------------
    print("== VERIFICATION (published d128 through this pipeline) ==")
    for k, v in ver.items():
        print(f"  {k}: {v}")
    print("\n== C1 draws (fit gate %.2f) ==" % FIT_GATE)
    for key, m in c1_out.items():
        r = m["rank_index"]
        rf = m["rank_index_fitgated_seeds"]
        print(f"  {key}: fit_med={m['fit_med']:.3f} gate={'PASS' if m['clears_fit_gate'] else 'FAIL'} "
              f"rank={'None' if r is None else format(r, '+.3f')}"
              f"±{m['rank_std'] if m['rank_std'] is not None else 0:.2f} (n={m['n_rank']}/{m['n']}) "
              f"rank@fit-seeds={'None' if rf is None else format(rf, '+.3f')} "
              f"(n={m['n_rank_fitgated_seeds']}) "
              f"d4acq={m['n_deg4_acquired']}/{m['n']} "
              f"rank@acq={'None' if m['rank_index_deg4acq_seeds'] is None else format(m['rank_index_deg4acq_seeds'], '+.3f')} "
              f"SI med={m['si_median'] if m['si_median'] else float('nan'):.2f} "
              f"mean={m['si_mean'] if m['si_mean'] else float('nan'):.2f} (n={m['n_si']})")
        print(f"     fits={m['fit_per_seed']}  half_med={m['half_med']}")
    print("\n== C1 matched-fit pointwise (per-draw AdamW-median target) ==")
    for ts, rec in mp.items():
        print(f"  {ts} target={rec['target_fit']:.3f}")
        for opt in ("adamw", "muon"):
            if opt in rec:
                r = rec[opt]
                dc = ", ".join(f"d{d}={v:+.2f}" for d, v in zip(DEGREES, r["deg_corr_med"]))
                print(f"    {opt:5s} step@match(med)={r['step_med']:.0f} [{dc}] "
                      f"n={r['n']} never_reached={r['n_never_reached']}")
    print("\n== C2 fresh-seed baseline (original target, n=8) ==")
    for opt, m in report["c2_baseline"].items():
        print(f"  {opt}: fit_med={m['fit_med']:.3f} rank={m['rank_index']:+.3f} "
              f"(n={m['n_rank']}/{m['n']}) SI mean={m['si_mean']:.2f} "
              f"med={m['si_median']:.2f} half_med={m['half_med']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

"""Inference layer for the staircase index (SI), Paper C Section 3.1.

The published analyzer (analyze_staircase.py) computes SI per run and reports
mean/SD plus pairwise TOST equivalence.  This script adds the inferential tests
that the equivalence framing leaves on the table, and discloses the structure of
the SI estimator itself:

  1. regression check: recompute the published per-optimizer SI mean/SD by
     importing half_times/si_of from analyze_staircase (no reimplementation);
  2. one-sample tests of SI against the refuted prediction SI = 1, both
     parametric (t) and nonparametric (Wilcoxon signed-rank), since SI is a
     geometric mean of ratios and may be right-skewed;
  3. the m-distribution: how many adjacent-degree ratios entered each run's
     geometric mean, and how many runs silently drop degree 1 because its
     half-threshold is crossed at step 0;
  4. the minimum TOST margin the data support, uncorrected and under
     Bonferroni/Holm, so the pre-specified SESOI can be checked against it;
  5. a complete-case (m = 3) TOST sensitivity analysis reported together with
     the differential missingness that induces it.

Writes results/figures-004/si_inference.json and prints a readable table.
CPU only.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Reuse the published SI machinery verbatim rather than reimplementing it.
# analyze_staircase imports figstyle/matplotlib at module level; that is fine on
# CPU (Agg backend) and keeps the SI definition single-sourced.
from analyze_staircase import load, half_times, si_of  # noqa: E402

RES = os.path.join(HERE, "..", "results")
DIR = os.path.join(RES, "degree_staircase")
OUT = os.path.join(RES, "figures-004")
DEGREES = [1, 2, 3, 4]
OPTS = ["adamw", "muon", "sgdm"]
OPT_LABEL = {"adamw": "AdamW", "muon": "Muon", "sgdm": "SGDM"}
ALPHA = 0.05
N_PAIRS = 3

# Published values in papers/C/main.tex Section 3.1 / Table 1, for the
# regression check.  (mean, sd, median); the table's SD is the sample SD
# (ddof=1), not analyze_staircase.py's np.std default.
PUBLISHED_SI = {"adamw": (2.98, 1.39, 2.80), "muon": (3.53, 1.54, 3.18),
                "sgdm": (3.20, 1.61, 3.42)}
UNLEARNED = 0.10   # final |corr| below this = the degree was never acquired
PUBLISHED_TOST = {"muon_vs_adamw": {"diff": 0.55, "p": 0.044, "ci": (-0.36, 1.46)},
                  "adamw_vs_sgdm": {"p": 0.014},
                  "muon_vs_sgdm": {"p": 0.026}}


# --------------------------------------------------------------------------- #
# data                                                                          #
# --------------------------------------------------------------------------- #
def collect():
    """Per optimizer, a list of per-run records for the staircase profile."""
    runs = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(DIR, "*.jsonl"))):
        meta, summ, hist = load(p)
        if not (meta and summ and hist):
            continue
        if meta["profile"] != "staircase":
            continue
        runs[meta["optimizer"]][meta["seed"]] = (summ, hist, os.path.basename(p))

    out = {}
    for opt in OPTS:
        recs = []
        for seed, (summ, hist, fname) in sorted(runs.get(opt, {}).items()):
            ht, finals = half_times(hist, DEGREES)
            r = si_of(ht)
            si, ratios = (r if r else (None, []))
            # degrees actually entering the geometric mean
            entering = sorted(k for k, v in ht.items() if v is not None and v > 0)
            recs.append({
                "file": fname,
                "seed": seed,
                "SI": si,
                "m": len(ratios),
                "ratios": ratios,
                "degrees_entering": entering,
                "half_times": {str(k): ht[k] for k in DEGREES},
                "step0_corr": {str(k): float(abs(hist[0]["deg_corr"][str(k)])) for k in DEGREES},
                "final_corr": {str(k): float(finals[k]) for k in DEGREES},
                "deg1_at_step0": ht.get(1) == 0,
                "deg4_at_step0": ht.get(4) == 0,
                "deg4_undefined": ht.get(4) is None,
                "deg4_final_corr": float(finals[4]),
                "deg4_unlearned": bool(finals[4] < UNLEARNED),
                # a degree can be dropped (T_k = 0) OR enter with a meaningless
                # half-time because the degree was never acquired
                "unlearned_degrees_entering": [k for k in entering if finals[k] < UNLEARNED],
            })
        out[opt] = recs
    return out


# --------------------------------------------------------------------------- #
# inference helpers                                                             #
# --------------------------------------------------------------------------- #
def welch(a, b):
    """Welch difference (a - b), pooled-free SE and Satterthwaite df."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    diff = a.mean() - b.mean()
    se = np.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return float(diff), float(se), float(df)


def tost(a, b, margin, alpha=ALPHA):
    """Two one-sided tests for equivalence of means within +-margin (Welch)."""
    diff, se, df = welch(a, b)
    t_lo = (diff + margin) / se          # H0: diff <= -margin
    t_hi = (diff - margin) / se          # H0: diff >= +margin
    p_lo = float(stats.t.sf(t_lo, df))
    p_hi = float(stats.t.cdf(t_hi, df))
    p = max(p_lo, p_hi)
    tcrit = float(stats.t.ppf(1 - alpha, df))
    ci = (diff - tcrit * se, diff + tcrit * se)
    return {"diff": diff, "se": se, "df": df, "p_lower": p_lo, "p_upper": p_hi,
            "p_tost": p, "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
            "alpha": alpha, "margin": margin, "equivalent": bool(p < alpha)}


def min_margin(a, b, alpha=ALPHA):
    """Smallest margin at which TOST rejects at level alpha: |diff| + t_{1-a,df}*se."""
    diff, se, df = welch(a, b)
    return float(abs(diff) + stats.t.ppf(1 - alpha, df) * se)


def one_sample_vs(vals, mu0=1.0):
    v = np.asarray(vals, float)
    n = len(v)
    t, p = stats.ttest_1samp(v, mu0)
    se = v.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    w = stats.wilcoxon(v - mu0, alternative="two-sided", zero_method="wilcox")
    # one-sided (SI > 1) versions, which is the directional scientific claim
    t_1s, p_1s = stats.ttest_1samp(v, mu0, alternative="greater")
    w_1s = stats.wilcoxon(v - mu0, alternative="greater", zero_method="wilcox")
    return {
        "n": int(n), "mean": float(v.mean()), "sd_ddof1": float(v.std(ddof=1)),
        "sd_ddof0": float(v.std()), "median": float(np.median(v)),
        "min": float(v.min()), "max": float(v.max()),
        "skew": float(stats.skew(v)),
        "shapiro_p": float(stats.shapiro(v).pvalue),
        "t": float(t), "df": int(n - 1), "p_two_sided": float(p),
        "t_one_sided": float(t_1s), "p_one_sided": float(p_1s),
        "ci95_lo": float(v.mean() - tcrit * se), "ci95_hi": float(v.mean() + tcrit * se),
        "wilcoxon_W": float(w.statistic), "wilcoxon_p_two_sided": float(w.pvalue),
        "wilcoxon_p_one_sided": float(w_1s.pvalue),
        "n_below_mu0": int((v < mu0).sum()),
    }


def holm(pvals):
    """Holm-Bonferroni: returns per-test adjusted threshold and reject flags."""
    order = np.argsort(pvals)
    k = len(pvals)
    thresh = np.empty(k)
    reject = np.zeros(k, bool)
    still = True
    for rank, idx in enumerate(order):
        thresh[idx] = ALPHA / (k - rank)
        if still and pvals[idx] < thresh[idx]:
            reject[idx] = True
        else:
            still = False
    return thresh, reject


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT, exist_ok=True)
    data = collect()
    report = {"definition": {
        "T_k": "first step at which |corr with degree-k Walsh basis| >= 0.5 x its final value",
        "SI": "geometric mean of adjacent half-time ratios over degrees with T_k defined and > 0",
        "m": "number of ratios entering the geometric mean = (#degrees with T_k defined and > 0) - 1",
    }}

    # ---------------- 1. regression check ---------------- #
    si = {opt: [r["SI"] for r in data[opt] if r["SI"] is not None] for opt in OPTS}
    reg = {}
    print("=" * 78)
    print("1. REGRESSION CHECK vs published SI values")
    print("=" * 78)
    for opt in OPTS:
        v = np.asarray(si[opt], float)
        rec = {"n": len(v), "mean": float(v.mean()), "sd_ddof0": float(v.std()),
               "sd_ddof1": float(v.std(ddof=1)), "median": float(np.median(v)),
               "per_seed": [float(x) for x in v]}
        if opt in PUBLISHED_SI:
            pm, ps, pmed = PUBLISHED_SI[opt]
            rec["published_mean"], rec["published_sd"], rec["published_median"] = pm, ps, pmed
            # the paper's SD is the sample SD (ddof=1)
            rec["matches_published"] = bool(round(rec["mean"], 2) == pm
                                            and round(rec["sd_ddof1"], 2) == ps
                                            and round(rec["median"], 2) == pmed)
        reg[opt] = rec
        tag = ""
        if opt in PUBLISHED_SI:
            tag = f"  published {PUBLISHED_SI[opt][0]:.2f} +/- {PUBLISHED_SI[opt][1]:.2f} " \
                  f"med {PUBLISHED_SI[opt][2]:.2f}  MATCH={rec['matches_published']}"
        print(f"  {OPT_LABEL[opt]:6s} n={rec['n']:2d}  SI = {rec['mean']:.4f} "
              f"+/- {rec['sd_ddof0']:.4f} (ddof=0) / {rec['sd_ddof1']:.4f} (ddof=1)  "
              f"median {rec['median']:.3f}{tag}")
    report["regression_check"] = reg

    # published pairwise TOST at margin 1.5
    print("\n  Published pairwise TOST at margin 1.5 (reproduction):")
    pairs = [("muon", "adamw"), ("adamw", "sgdm"), ("muon", "sgdm")]
    tost15 = {}
    for a, b in pairs:
        r = tost(si[a], si[b], 1.5)
        key = f"{a}_vs_{b}"
        tost15[key] = r
        pub = PUBLISHED_TOST.get(key, {})
        note = ""
        if "p" in pub:
            note = f"  published p={pub['p']:.3f} MATCH={abs(r['p_tost'] - pub['p']) < 0.0015}"
        print(f"    {OPT_LABEL[a]:5s} - {OPT_LABEL[b]:5s}: diff {r['diff']:+.4f}  "
              f"90% CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  df {r['df']:.1f}  "
              f"p_TOST {r['p_tost']:.4f}{note}")
    report["tost_margin_1.5"] = tost15

    # ---------------- 2. one-sample tests vs SI = 1 ---------------- #
    print("\n" + "=" * 78)
    print("2. ONE-SAMPLE TESTS AGAINST THE REFUTED PREDICTION SI = 1")
    print("=" * 78)
    one = {}
    for opt in OPTS:
        r = one_sample_vs(si[opt], 1.0)
        one[opt] = r
        print(f"  {OPT_LABEL[opt]:6s} mean {r['mean']:.3f} (95% CI [{r['ci95_lo']:.3f}, "
              f"{r['ci95_hi']:.3f}])  t({r['df']}) = {r['t']:.3f}  p = {r['p_two_sided']:.3e}")
        print(f"         Wilcoxon W = {r['wilcoxon_W']:.1f}  p = {r['wilcoxon_p_two_sided']:.3e}"
              f"   ({r['n_below_mu0']}/{r['n']} runs below 1)")
        print(f"         skew {r['skew']:+.3f}  Shapiro-Wilk p = {r['shapiro_p']:.4f}  "
              f"one-sided t p = {r['p_one_sided']:.3e}, Wilcoxon p = {r['wilcoxon_p_one_sided']:.3e}")
    # Bonferroni/Holm over the three one-sample tests
    pv = [one[o]["p_two_sided"] for o in OPTS]
    th, rej = holm(pv)
    for o, t_, r_ in zip(OPTS, th, rej):
        one[o]["holm_threshold"] = float(t_)
        one[o]["holm_reject"] = bool(r_)
        one[o]["bonferroni_reject"] = bool(one[o]["p_two_sided"] < ALPHA / N_PAIRS)
    print(f"  Holm over the 3 one-sample tests: all reject = {bool(rej.all())} "
          f"(largest p = {max(pv):.3e} vs threshold {ALPHA/1:.4f})")
    report["one_sample_vs_1"] = one

    # ---------------- 3. m-distribution ---------------- #
    print("\n" + "=" * 78)
    print("3. m-DISTRIBUTION AND DEGREE-1-AT-STEP-0 DROPPING")
    print("=" * 78)
    mdist = {}
    tot = Counter()
    for opt in OPTS:
        recs = data[opt]
        c = Counter(r["m"] for r in recs)
        degsets = Counter(tuple(r["degrees_entering"]) for r in recs)
        drop_by_deg = {}
        for k in DEGREES:
            n_zero = sum(1 for r in recs if r["half_times"][str(k)] == 0)
            n_none = sum(1 for r in recs if r["half_times"][str(k)] is None)
            drop_by_deg[str(k)] = {"half_time_zero": int(n_zero), "undefined": int(n_none)}
            tot[f"deg{k}_zero"] += n_zero
            tot[f"deg{k}_none"] += n_none
        n_unl_entering = sum(1 for r in recs if r["unlearned_degrees_entering"])
        mdist[opt] = {
            "m_counts": {str(k): int(v) for k, v in sorted(c.items())},
            "drop_by_degree": drop_by_deg,
            "n_deg1_half_time_zero": drop_by_deg["1"]["half_time_zero"],
            "n_deg4_half_time_zero": drop_by_deg["4"]["half_time_zero"],
            "n_deg4_unlearned_final_corr_lt_%.2f" % UNLEARNED:
                int(sum(r["deg4_unlearned"] for r in recs)),
            "n_runs_with_unlearned_degree_entering_SI": int(n_unl_entering),
            "deg4_final_corr_median": float(np.median([r["deg4_final_corr"] for r in recs])),
            "degree_sets": {str(list(k)): int(v) for k, v in sorted(degsets.items())},
        }
        print(f"  {OPT_LABEL[opt]:6s} m-counts {dict(sorted(c.items()))}")
        for k in DEGREES:
            d = drop_by_deg[str(k)]
            print(f"           deg {k}: half-time = 0 in {d['half_time_zero']}/{len(recs)} runs, "
                  f"undefined in {d['undefined']}/{len(recs)}")
        print(f"           deg-4 never acquired (final |corr| < {UNLEARNED}): "
              f"{mdist[opt]['n_deg4_unlearned_final_corr_lt_%.2f' % UNLEARNED]}/{len(recs)}"
              f"   (median final deg-4 |corr| {mdist[opt]['deg4_final_corr_median']:.3f})")
        print(f"           runs where an unacquired degree still enters SI: {n_unl_entering}/{len(recs)}")
        for k, v in sorted(degsets.items()):
            print(f"           degrees entering {list(k)}: {v} runs")
    mdist["_totals_over_45"] = {k: int(v) for k, v in sorted(tot.items())}
    print(f"  TOTAL over 45 runs: degree 1 dropped (T=0) in {tot['deg1_zero']}, "
          f"degree 4 dropped (T=0) in {tot['deg4_zero']}; "
          f"no half-time is ever undefined ({sum(v for k, v in tot.items() if k.endswith('none'))} None).")
    report["m_distribution"] = mdist

    # ---------------- 4. minimum supportable margin ---------------- #
    print("\n" + "=" * 78)
    print("4. MINIMUM TOST MARGIN SUPPORTED BY THE DATA")
    print("=" * 78)
    mm = {}
    for a, b in pairs:
        unc = min_margin(si[a], si[b], ALPHA)
        bonf = min_margin(si[a], si[b], ALPHA / N_PAIRS)
        mm[f"{a}_vs_{b}"] = {"uncorrected": unc, "bonferroni": bonf}
        print(f"  {OPT_LABEL[a]:5s} - {OPT_LABEL[b]:5s}: min margin "
              f"uncorrected {unc:.4f}   Bonferroni (alpha/3) {bonf:.4f}")
    fam_unc = max(v["uncorrected"] for v in mm.values())
    fam_bonf = max(v["bonferroni"] for v in mm.values())
    mm["_family_uncorrected"] = fam_unc
    mm["_family_bonferroni"] = fam_bonf
    print(f"  Family-wide (all three pairs equivalent): uncorrected {fam_unc:.4f}, "
          f"Bonferroni {fam_bonf:.4f}")
    print(f"  Pre-specified SESOI = 1.5  -> headroom {1.5 - fam_unc:+.4f} uncorrected, "
          f"{1.5 - fam_bonf:+.4f} Bonferroni")
    # margin sweep for the paper's sensitivity table
    sweep = {}
    for margin in (1.0, 1.5, 2.0):
        sweep[str(margin)] = {f"{a}_vs_{b}": tost(si[a], si[b], margin)["p_tost"]
                              for a, b in pairs}
    mm["_sweep"] = sweep
    print("  SESOI sweep p_TOST:")
    for margin, row in sweep.items():
        print("    margin %-4s %s" % (margin, "  ".join(f"{k}={v:.4f}" for k, v in row.items())))
    report["min_margin"] = mm

    # ---------------- 5. complete-case (m = 3) sensitivity ---------------- #
    print("\n" + "=" * 78)
    print("5. COMPLETE-CASE (m = 3) SENSITIVITY + DIFFERENTIAL MISSINGNESS")
    print("=" * 78)
    cc_si = {opt: [r["SI"] for r in data[opt] if r["m"] == 3] for opt in OPTS}
    cc = {"n_per_optimizer": {o: len(cc_si[o]) for o in OPTS},
          "si_mean": {o: float(np.mean(cc_si[o])) for o in OPTS},
          "si_sd_ddof0": {o: float(np.std(cc_si[o])) for o in OPTS}}
    for o in OPTS:
        print(f"  {OPT_LABEL[o]:6s} complete-case n={len(cc_si[o]):2d}  "
              f"SI = {np.mean(cc_si[o]):.4f} +/- {np.std(cc_si[o]):.4f}")
    cc["one_sample_vs_1"] = {o: one_sample_vs(cc_si[o], 1.0) for o in OPTS}
    for o in OPTS:
        r = cc["one_sample_vs_1"][o]
        print(f"  {OPT_LABEL[o]:6s} complete-case vs SI = 1: t({r['df']}) = {r['t']:.3f}, "
              f"p = {r['p_two_sided']:.3e}; Wilcoxon p = {r['wilcoxon_p_two_sided']:.3e}")
    cc_tost = {}
    for a, b in pairs:
        r = tost(cc_si[a], cc_si[b], 1.5)
        cc_tost[f"{a}_vs_{b}"] = r
        print(f"  {OPT_LABEL[a]:5s} - {OPT_LABEL[b]:5s}: diff {r['diff']:+.4f}  "
              f"90% CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  df {r['df']:.1f}  "
              f"p_TOST {r['p_tost']:.4f}")
    pv_cc = [cc_tost[f"{a}_vs_{b}"]["p_tost"] for a, b in pairs]
    th_cc, rej_cc = holm(pv_cc)
    for (a, b), t_, r_ in zip(pairs, th_cc, rej_cc):
        cc_tost[f"{a}_vs_{b}"]["holm_threshold"] = float(t_)
        cc_tost[f"{a}_vs_{b}"]["holm_reject"] = bool(r_)
        print(f"    Holm {OPT_LABEL[a]:5s}-{OPT_LABEL[b]:5s}: p={cc_tost[f'{a}_vs_{b}']['p_tost']:.4f} "
              f"vs threshold {t_:.4f} -> reject={bool(r_)}")
    cc["tost_margin_1.5"] = cc_tost
    # also the full-sample Holm for comparison
    pv_full = [tost15[f"{a}_vs_{b}"]["p_tost"] for a, b in pairs]
    th_f, rej_f = holm(pv_full)
    full_holm = {f"{a}_vs_{b}": {"p": float(p), "threshold": float(t_), "reject": bool(r_)}
                 for (a, b), p, t_, r_ in zip(pairs, pv_full, th_f, rej_f)}
    report["full_sample_holm_margin_1.5"] = full_holm
    print("  Full-sample Holm at margin 1.5:")
    for k, v in full_holm.items():
        print(f"    {k}: p={v['p']:.4f} vs threshold {v['threshold']:.4f} -> reject={v['reject']}")

    # differential missingness: two distinct quantities, kept separate.
    #   (i) the mechanism that removes a run from the complete case: T_4 = 0
    #   (ii) the underlying degree-4 acquisition gap: final |corr| < UNLEARNED
    miss = {}
    for o in OPTS:
        recs = data[o]
        miss[o] = {
            "n": len(recs),
            "deg4_dropped_T0": int(sum(r["deg4_at_step0"] for r in recs)),
            "deg1_dropped_T0": int(sum(r["deg1_at_step0"] for r in recs)),
            "deg4_unlearned": int(sum(r["deg4_unlearned"] for r in recs)),
            "deg4_final_corr_median": float(np.median([r["deg4_final_corr"] for r in recs])),
            "complete_case_seeds": [r["seed"] for r in recs if r["m"] == 3],
        }

    def fx(key):
        a, m_ = miss["adamw"][key], miss["muon"][key]
        r = stats.fisher_exact([[a, 15 - a], [m_, 15 - m_]])
        return {"adamw": a, "muon": m_, "odds_ratio": float(r.statistic), "p": float(r.pvalue)}

    miss["_fisher_deg4_dropped"] = fx("deg4_dropped_T0")
    miss["_fisher_deg4_unlearned"] = fx("deg4_unlearned")
    print("  Differential missingness:")
    print("    (i) degree-4 removed from the SI product (half-time = 0):")
    for o in OPTS:
        print(f"        {OPT_LABEL[o]:6s} {miss[o]['deg4_dropped_T0']}/15   "
              f"degree-1 removed {miss[o]['deg1_dropped_T0']}/15")
    f1 = miss["_fisher_deg4_dropped"]
    print(f"        Fisher AdamW vs Muon: {f1['adamw']} vs {f1['muon']}, p = {f1['p']:.4f}")
    print(f"    (ii) degree 4 never acquired (final |corr| < {UNLEARNED}):")
    for o in OPTS:
        print(f"        {OPT_LABEL[o]:6s} {miss[o]['deg4_unlearned']}/15  "
              f"(median final deg-4 |corr| {miss[o]['deg4_final_corr_median']:.3f})")
    f2 = miss["_fisher_deg4_unlearned"]
    print(f"        Fisher AdamW vs Muon: {f2['adamw']} vs {f2['muon']}, p = {f2['p']:.4f}")
    # seed overlap of the complete case (initialization is seed-shared across optimizers)
    sets = {o: set(miss[o]["complete_case_seeds"]) for o in OPTS}
    miss["_complete_case_overlap"] = {
        "adamw_seeds": sorted(sets["adamw"]), "muon_seeds": sorted(sets["muon"]),
        "sgdm_seeds": sorted(sets["sgdm"]),
        "adamw_muon_shared": sorted(sets["adamw"] & sets["muon"]),
        "all_three_shared": sorted(sets["adamw"] & sets["muon"] & sets["sgdm"]),
    }
    print(f"    complete-case seeds: AdamW {sorted(sets['adamw'])}")
    print(f"                         Muon  {sorted(sets['muon'])}")
    print(f"                         SGDM  {sorted(sets['sgdm'])}")
    print(f"    shared AdamW/Muon: {sorted(sets['adamw'] & sets['muon'])}, "
          f"all three: {sorted(sets['adamw'] & sets['muon'] & sets['sgdm'])}")
    cc["differential_missingness"] = miss
    report["complete_case"] = cc

    # ---------------- 6. acquired-degrees-only SI ---------------- #
    # The T_k > 0 filter removes a degree only when the step-0 correlation already
    # exceeds half the final value.  A degree that is never acquired can still slip
    # through with a noise-driven half-time.  Recompute SI over degrees that were
    # actually acquired (final |corr| >= UNLEARNED) as a definition-robustness check.
    print("\n" + "=" * 78)
    print(f"6. ACQUIRED-DEGREES-ONLY SI (drop degrees with final |corr| < {UNLEARNED})")
    print("=" * 78)
    acq = {}
    for opt in OPTS:
        vals, ms = [], []
        for r in data[opt]:
            ks = [k for k in r["degrees_entering"] if r["final_corr"][str(k)] >= UNLEARNED]
            ht = r["half_times"]
            ratios = [ht[str(b)] / ht[str(a)] for a, b in zip(ks, ks[1:])]
            if ratios:
                vals.append(float(np.exp(np.mean(np.log(ratios)))))
                ms.append(len(ratios))
        one_a = one_sample_vs(vals, 1.0)
        acq[opt] = {"n": len(vals), "mean": float(np.mean(vals)),
                    "sd_ddof1": float(np.std(vals, ddof=1)),
                    "median": float(np.median(vals)),
                    "m_counts": {str(k): int(v) for k, v in sorted(Counter(ms).items())},
                    "one_sample": one_a, "per_seed": [float(x) for x in vals]}
        print(f"  {OPT_LABEL[opt]:6s} n={len(vals):2d}  SI = {np.mean(vals):.4f} +/- "
              f"{np.std(vals, ddof=1):.4f}  median {np.median(vals):.3f}  "
              f"m-counts {dict(sorted(Counter(ms).items()))}")
        print(f"         vs 1: t({one_a['df']}) = {one_a['t']:.3f}, p = {one_a['p_two_sided']:.3e}; "
              f"Wilcoxon p = {one_a['wilcoxon_p_two_sided']:.3e}")
    acq_tost = {}
    for a, b in pairs:
        r = tost(acq[a]["per_seed"], acq[b]["per_seed"], 1.5)
        acq_tost[f"{a}_vs_{b}"] = r
        print(f"  TOST {OPT_LABEL[a]:5s} - {OPT_LABEL[b]:5s}: diff {r['diff']:+.4f}  "
              f"90% CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  p_TOST {r['p_tost']:.4f}")
    acq["_tost_margin_1.5"] = acq_tost
    acq["_min_margin_family_uncorrected"] = float(max(
        min_margin(acq[a]["per_seed"], acq[b]["per_seed"], ALPHA) for a, b in pairs))
    print(f"  Family-wide minimum margin (uncorrected): "
          f"{acq['_min_margin_family_uncorrected']:.4f}")
    report["acquired_only"] = acq

    # per-run dump for auditability
    report["per_run"] = {o: data[o] for o in OPTS}

    path = os.path.join(OUT, "si_inference.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

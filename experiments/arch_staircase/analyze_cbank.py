#!/usr/bin/env python3
"""C-BANK verdict -- ablation selectivity at n=15, with an n=8 regression gate.

The scalar is paper C's (Sec `ablation`, tab:ablation):

    selectivity(cond) = Delta_4 - mean(Delta_{1..3}),
    Delta_k = median_seeds deg_corr_k(none) - median_seeds deg_corr_k(cond)

at the final checkpoint, with a paired percentile bootstrap over seeds. The
convention -- median-based deltas, the SAME resampled seed index reused for
attn_l1 and mlp_l1 so their difference is paired, B=20000, seed 20260717 --
is imported live from revision2026/C/t07_selectivity_bootstrap.py rather than
restated, so the n=15 numbers are directly comparable to the published n=8
ones.

`--regression` (also run inside the runner's --smoke) recomputes the published
figures from the archived n=8 data in revision2026/cg3-C/ and checks them
against the values in print: adamw mlp_l1 +0.369, muon mlp_l1 +0.344, muon
mlp_minus_attn +0.141. If that directory is ever absent the check reports
itself as unavailable rather than passing vacuously.

Usage:
  python analyze_cbank.py --regression        # n=8 gate only
  python analyze_cbank.py                     # gate, then the n=15 verdict
  python analyze_cbank.py --pool-archived     # new seeds + archived n=8
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from experiments.gpu2026_contracts import atomic_write_json, sha256_file, write_analysis_envelope
from experiments.tools.verify_gpu2026_retrieval import check_arm

_THIS = Path(__file__).resolve().parent
ROOT = _THIS.parents[1]
ARM = ROOT / "experiments" / "revision2026" / "gpu2026" / "cbank"
ABLATE_DIR = ARM / "ablate"
ARCHIVED = ROOT / "experiments" / "revision2026" / "cg3-C"

_T07_SRC = ROOT / "experiments" / "revision2026" / "C" / "t07_selectivity_bootstrap.py"
_spec = importlib.util.spec_from_file_location("t07_selectivity_bootstrap",
                                               _T07_SRC)
T07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T07)  # type: ignore[union-attr]

DEGREES = T07.DEGREES
CONDS = T07.CONDS
B = T07.B
SEED = T07.SEED

PUBLISHED = {("adamw", "mlp_l1"): 0.369,
             ("muon", "mlp_l1"): 0.344,
             ("muon", "mlp_minus_attn"): 0.141}


def bootstrap_dir(directories, opt, b=B, seed=SEED):
    """Paired seed bootstrap of the selectivity scalars for one optimizer.

    directories: list of dirs to pool; files are matched as *_<opt>.ablate.jsonl.
    """
    files = []
    for d in directories:
        files.extend(sorted(glob.glob(os.path.join(str(d),
                                                   f"*_{opt}.ablate.jsonl"))))
    if not files:
        return None
    finals = [T07.load_final(p) for p in files]
    n = len(finals)
    dc = {c: np.array([[f[c]["deg_corr"][d] for d in DEGREES] for f in finals])
          for c in CONDS}
    point = {c: T07.selectivity(dc["none"], dc[c])
             for c in ("attn_l1", "mlp_l1")}
    point["mlp_minus_attn"] = point["mlp_l1"] - point["attn_l1"]

    rng = np.random.default_rng(seed)
    boots = {k: np.empty(b) for k in point}
    for i in range(b):
        idx = rng.integers(0, n, n)
        s_attn = T07.selectivity(dc["none"][idx], dc["attn_l1"][idx])
        s_mlp = T07.selectivity(dc["none"][idx], dc["mlp_l1"][idx])
        boots["attn_l1"][i] = s_attn
        boots["mlp_l1"][i] = s_mlp
        boots["mlp_minus_attn"][i] = s_mlp - s_attn

    rep = {"n_seeds": n, "n_resamples": b,
           "seeds": sorted(f["none"].get("seed", -1) for f in finals)
           if all("seed" in f["none"] for f in finals) else None,
           "files": [os.path.basename(p) for p in files]}
    for k, arr in boots.items():
        lo, hi = np.percentile(arr, [2.5, 97.5])
        rep[k] = {"point": round(float(point[k]), 4),
                  "ci95": [round(float(lo), 4), round(float(hi), 4)],
                  "excludes_zero": bool(lo > 0 or hi < 0)}
    return rep


def regression_check(tol=0.002, b=2000):
    """Recompute the published n=8 numbers from the archived data.

    A small resample count is enough: the POINT estimates are deterministic
    given the data, and those are what is checked. Returns
    {"ok", "available", "checked"}.
    """
    if not ARCHIVED.is_dir() or not glob.glob(str(ARCHIVED / "*.ablate.jsonl")):
        return {"ok": False, "available": False, "checked": {},
                "note": f"archived n=8 ablation data absent at {ARCHIVED}; "
                        f"the published-number regression check CANNOT be "
                        f"performed and is not being claimed as passed"}
    checked, ok = {}, True
    for (opt, key), published in PUBLISHED.items():
        rep = bootstrap_dir([ARCHIVED], opt, b=b)
        got = rep[key]["point"]
        match = abs(got - published) <= tol
        ok = ok and match and rep["n_seeds"] == 8
        checked[f"{opt}.{key}"] = {"point": got, "published": published,
                                   "delta": round(got - published, 5),
                                   "n_seeds": rep["n_seeds"], "match": match}
    return {"ok": ok, "available": True, "checked": checked, "tol": tol}


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_regression_check():
    try:
        return regression_check()
    except (OSError, ValueError, KeyError, TypeError, IndexError,
            json.JSONDecodeError) as exc:
        print(f"invalid C-BANK regression input: {exc}")
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regression", action="store_true",
                    help="only run the n=8 published-number gate")
    ap.add_argument("--pool-archived", action="store_true",
                    help="pool the new ablations with the archived n=8 set")
    ap.add_argument("--dir", default=str(ABLATE_DIR))
    args = ap.parse_args(argv)

    if args.regression:
        reg = _run_regression_check()
        if reg is None:
            return 1
        if not reg["available"]:
            print("REGRESSION CHECK UNAVAILABLE: " + reg["note"])
        else:
            print(f"n=8 regression gate (tol {reg['tol']}):")
            for k, v in reg["checked"].items():
                print(f"  {k:<26} {v['point']:+.4f} vs published "
                      f"{v['published']:+.3f}  delta {v['delta']:+.5f}  "
                      f"n={v['n_seeds']}  {'PASS' if v['match'] else 'FAIL'}")
            print(f"  gate: {'PASS' if reg['ok'] else 'FAIL'}")
        return 0 if reg.get("available") is True and reg.get("ok") is True else 1

    ablate_dir = Path(args.dir)
    arm_dir = ablate_dir.resolve().parent
    manifest = arm_dir / "MANIFEST.json"
    canonical_manifest = arm_dir / "checkpoint_manifest.json"
    fidelity = _load_json(arm_dir / "cbank_fidelity.json")
    check = check_arm(str(arm_dir.parent), "cbank")
    required_predicates = {
        "inputs_complete": check["ok"],
        "fidelity_pass": fidelity.get("headline") == "PASS",
        "fidelity_manifest_matches": (
            canonical_manifest.is_file()
            and fidelity.get("checkpoint_manifest_sha256") == sha256_file(canonical_manifest)
        ),
        "thirty_ablation_files": sum(
            1 for p in ablate_dir.glob("*.ablate.jsonl")
        ) == 30,
    }
    if not manifest.is_file():
        print(f"missing mandatory manifest: {manifest}")
        return 1
    if not all(required_predicates.values()):
        print(f"invalid C-BANK retrieval inputs: {required_predicates}")
        return 1

    reg = _run_regression_check()
    if reg is None:
        return 1
    required_predicates.update({
        "regression_available": reg.get("available") is True,
        "regression_ok": reg.get("ok") is True,
    })
    if not reg["available"]:
        print("REGRESSION CHECK UNAVAILABLE: " + reg["note"])
    else:
        print(f"n=8 regression gate (tol {reg['tol']}):")
        for k, v in reg["checked"].items():
            print(f"  {k:<26} {v['point']:+.4f} vs published "
                  f"{v['published']:+.3f}  delta {v['delta']:+.5f}  "
                  f"n={v['n_seeds']}  {'PASS' if v['match'] else 'FAIL'}")
        print(f"  gate: {'PASS' if reg['ok'] else 'FAIL'}")
    if not all(required_predicates.values()):
        print(f"invalid C-BANK regression inputs: {required_predicates}")
        return 1

    dirs = [ablate_dir]
    if args.pool_archived:
        dirs.append(ARCHIVED)
    report = {"regression": reg, "pooled_with_archived": args.pool_archived,
              "dirs": [str(d) for d in dirs]}
    for opt in ("adamw", "muon"):
        try:
            rep = bootstrap_dir(dirs, opt)
        except (OSError, ValueError, KeyError, TypeError, IndexError,
                json.JSONDecodeError) as exc:
            print(f"invalid C-BANK analyzer input for {opt}: {exc}")
            return 1
        if rep is None:
            print(f"\nno {opt} ablation outputs under {args.dir}")
            continue
        report[opt] = rep
        print(f"\n{opt}  n_seeds={rep['n_seeds']}")
        for key in ("attn_l1", "mlp_l1", "mlp_minus_attn"):
            v = rep[key]
            print(f"  {key:<16} {v['point']:+.4f}  CI95 "
                  f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  "
                  f"excludes_zero={v['excludes_zero']}")

    expected_seeds = list(range(100, 115))
    adamw = report.get("adamw", {})
    muon = report.get("muon", {})
    required_predicates.update({
        "adamw_n15": adamw.get("n_seeds") == 15,
        "muon_n15": muon.get("n_seeds") == 15,
        "adamw_exact_seeds": adamw.get("seeds") == expected_seeds,
        "muon_exact_seeds": muon.get("seeds") == expected_seeds,
    })
    if not all(required_predicates.values()):
        print(f"invalid C-BANK analyzed inputs: {required_predicates}")
        return 1

    out = arm_dir / "cbank_verdict.json"
    atomic_write_json(out, report)
    write_analysis_envelope(
        arm_dir / "cbank_analysis.json", arm="cbank", manifest_path=manifest,
        required_predicates=required_predicates, scientific_outcome=report,
    )
    print(f"\nwrote {out}")
    print(f"wrote {arm_dir / 'cbank_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""C-BANK -- extend the trained-model mean-ablation bank from n=8 to n=15 seeds.

WHY. Paper C's abstract states the degree-4-selective carrier is the layer-1
MLP under BOTH optimizers, but the seed-level bootstrap that backs it
(revision2026/C/t07_selectivity_bootstrap.json, 8 seeds) does not put either
absolute selectivity off zero:

  adamw mlp_l1  +0.369  CI95 [-0.045, +0.415]   excludes_zero false
  muon  mlp_l1  +0.344  CI95 [-0.039, +0.525]   excludes_zero false
  muon  mlp_minus_attn  +0.141  CI95 [+0.058, +0.224]   excludes_zero TRUE

Only the Muon paired contrast survives. Seven more seeds per optimizer is the
cheapest route to resolving the absolute selectivities.

WHY THIS REGENERATES THE BANK. The previous round's 192 checkpoints
(16 runs x 12) were never retrieved from the box -- only the trajectories
(revision2026/pilot-CE1/results/p1_c2/), the checkpoint manifest, and the
ablation outputs (revision2026/cg3-C/*.ablate.jsonl) are on local disk. The
.pt files are gone, so the ablation cannot simply be extended; the bank is
rebuilt first. Phases are separately invokable so the bank can be produced on
one GPU while another does unrelated work.

RECOVERED FROM THE PREVIOUS ROUND (nothing here was newly chosen):
  * 12 checkpoint positions {0,50,100,150,200,300,450,700,1050,1650,2600,4000}
    -- from revision2026/pilot-CE1/ckpt_manifest.jsonl (192 rows, 12 distinct
    steps) and the P1-C2 block of RUNBOOK-P1-CE1.md.
  * 7 ablation conditions none|attn|mlp|attn_l0|attn_l1|mlp_l0|mlp_l1 --
    imported live from stier2026/run_cg3_ablate.py rather than restated, so
    the protocol cannot drift.
  * muon_lr = 0.01 (NOT ArchConfig's 0.02 default) -- from the `_meta` of the
    archived muon trajectories and run_cg3_ablate.py's header.

SEEDS. 100..114. The 100..107 block deliberately REPEATS the archived seeds
instead of being fresh: sample_batch and enumerate_dataset both draw from
explicit torch.Generator objects (degree_staircase/data.py), and checkpoint
saving consumes no RNG, so those eight runs reproduce the archived
trajectories and give a live end-to-end check that the rebuilt bank is the
same object as the old one. 108..114 is the genuinely fresh extension, unused
by any prior arm (archived arch_staircase used 0..14 adamw / 0..4 muon,
optaxis 0..4, lrsweep 5..7, pilot 100..107). Pass `--seeds 108-114` to build
only the extension and pool with the archived ablation outputs -- cheaper, but
it rests on the two halves having been produced by identical pipelines, which
the full default rebuild does not have to assume.

MUON AND THE FREEZE ARMS. analyze_optaxis.py warns that "Muon acts on a
different param set per freeze arm", so freeze arms may only be compared as
deltas. That hazard does not arise here: every C-BANK run is freeze=none, and
component ownership is read POST HOC by mean-ablating a fully trained model,
never by training with a component frozen. The optimizer therefore acts on the
identical parameter set in every cell of this arm. The muon-capable optimizer
builder is reused from run_arch_optaxis.py (importing it installs the
monkeypatch that replaces train_arch's AdamW-only assert) rather than
reimplemented.

Phases:
  --phase bank    train 15 seeds x {adamw, muon}, saving the 12 checkpoints
  --phase ablate  7 mean-ablation conditions at every checkpoint + sanity gate
  --phase both    bank then ablate (default)

  --smoke     CPU, minutes: both phases end-to-end on a tiny model with 2
              checkpoints, plus the n=8 bootstrap regression check
  --dry-run   cell list, run count, GPU-hours, checkpoint disk usage
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parents[1]


def _early_gpu_pin(argv):
    """CUDA_VISIBLE_DEVICES must be set before torch is imported."""
    for i, a in enumerate(argv):
        if a == "--gpu" and i + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = argv[i + 1]
        elif a.startswith("--gpu="):
            os.environ["CUDA_VISIBLE_DEVICES"] = a.split("=", 1)[1]


_early_gpu_pin(sys.argv)

if str(_THIS_DIR) in sys.path:
    sys.path.remove(str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR))
sys.path.append(str(ROOT / "experiments"))

import torch  # noqa: E402

import train_arch as TA  # noqa: E402  (017 trainer; not modified)
import run_arch_optaxis as OPTAXIS  # noqa: E402  (installs muon/sgdm builder)
from runner_utils import (  # noqa: E402
    add_shard_args, shard_cells, validate_shard_args)

# stier2026/run_cg3_ablate.py resolves its imports off DLR_CODE; point it at
# this checkout and load it by path so the ablation protocol (conditions,
# mean-ablation hooks, per-condition readout) is the previous round's code
# verbatim rather than a restatement of it.
os.environ.setdefault("DLR_CODE", str(ROOT / "experiments"))
os.environ.setdefault("DLR_CG3", str(ROOT / "experiments" / "revision2026"
                                     / "gpu2026" / "cbank"))
_CG3_SRC = ROOT / "experiments" / "stier2026" / "run_cg3_ablate.py"
_spec = importlib.util.spec_from_file_location("run_cg3_ablate", _CG3_SRC)
CG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CG)  # type: ignore[union-attr]

ARM = ROOT / "experiments" / "revision2026" / "gpu2026" / "cbank"
BANK_DIR = ARM / "bank"
CKPT_ROOT = ARM / "ckpts"
ABLATE_DIR = ARM / "ablate"
ARCHIVED_ABLATE = ROOT / "experiments" / "revision2026" / "cg3-C"

CKPT_STEPS = (0, 50, 100, 150, 200, 300, 450, 700, 1050, 1650, 2600, 4000)
SEEDS = tuple(range(100, 115))
OPTIMIZERS = ("adamw", "muon")
MUON_LR = 0.01
CONDITIONS = CG.CONDITIONS

# Measured on the previous round's box over the identical cell
# (pilot-CE1/results/p1_c2, 16 runs): median 318.6 s, max 339.1 s.
SEC_PER_BANK_RUN = 319.0
# Measured from ckpt_manifest.jsonl: every checkpoint is 1,597,594 bytes.
BYTES_PER_CKPT = 1_597_594
# Archived ablation cost, 12 ckpts x 7 conditions (cg3-C `_summary`): ~1.3 s.
SEC_PER_ABLATE_RUN = 2.0


def make_cells(seeds=SEEDS, optimizers=OPTIMIZERS):
    return [TA.ArchConfig(d_model=128, n_layers=2, n_heads=4, freeze="none",
                          optimizer=opt, seed=s, muon_lr=MUON_LR)
            for opt in optimizers for s in seeds]


def run_name(cfg):
    return f"{cfg.name()}_{cfg.optimizer}"


def bank_path(cfg):
    return BANK_DIR / f"{run_name(cfg)}.jsonl"


def ckpt_dir(cfg):
    return CKPT_ROOT / run_name(cfg)


def ablate_path(cfg):
    return ABLATE_DIR / f"{run_name(cfg)}.ablate.jsonl"


def _tail_has(path: Path, key: str):
    if not path.exists():
        return False
    last = ""
    with open(path) as f:
        for line in f:
            if line.strip():
                last = line
    try:
        return key in json.loads(last)
    except json.JSONDecodeError:
        return False


def bank_done(cfg, ckpt_steps=CKPT_STEPS):
    """Complete iff the trajectory closed with a `_summary` AND every
    checkpoint the ablate phase will ask for is on disk."""
    if not _tail_has(bank_path(cfg), "_summary"):
        return False
    d = ckpt_dir(cfg)
    return all((d / f"step{s:05d}.pt").exists() for s in ckpt_steps)


def ablate_done(cfg):
    return _tail_has(ablate_path(cfg), "_summary")


# ------------------------------------------------------------------ phase 1

def run_with_ckpts(cfg, out_path, ckpt_steps, manifest_path, device=None):
    """train_arch.run mirrored field-for-field, plus checkpoint saving.

    train_arch.run has no checkpoint hook and train_arch.py is not to be
    modified, so the loop is restated here. Every RNG draw is identical to
    TA.run's: torch.manual_seed once, then sample_batch/enumerate_dataset
    which take explicit generator seeds, so saving state_dicts cannot perturb
    the trajectory. The checkpoint is taken immediately after the same-step
    eval, i.e. the model has had exactly `step` updates.
    """
    import torch.nn.functional as F

    torch.manual_seed(cfg.seed)
    device = device or (cfg.device if torch.cuda.is_available() else "cpu")
    spec = TA.TS.make_spec(cfg)

    model = TA.TS.build_model(cfg, spec, device)
    n_trainable, n_frozen = TA.freeze_component(model, cfg.freeze)
    optimizers = TA.build_optimizer_trainable(model, cfg)
    degrees = sorted(TA.TS.monomial_sets(spec).keys())

    name = run_name(cfg)
    cdir = Path(manifest_path).parent / name
    cdir.mkdir(parents=True, exist_ok=True)

    history: list = []
    t0 = time.time()
    f = open(out_path, "w") if out_path else None
    if f:
        f.write(json.dumps({"_meta": {
            **asdict(cfg), "n_trainable": n_trainable, "n_frozen": n_frozen,
            "arm": "C-BANK", "ckpt_steps": list(ckpt_steps),
            "ckpt_dir": str(cdir)}}) + "\n")

    saved = []
    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0:
            mse, fit_corr, deg_corr = TA.TS.evaluate(model, spec, cfg, device)
            rec = {"step": step, "mse": mse, "fit_corr": fit_corr,
                   "deg_corr": {str(k): v for k, v in deg_corr.items()}}
            si = TA.TS.staircase_index(history + [rec], degrees)
            rec["staircase_spread"] = si["spread"]
            rec["staircase_span_ratio"] = si["span_ratio"]
            history.append(rec)
            if f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            if step in ckpt_steps:
                cpath = cdir / f"step{step:05d}.pt"
                torch.save({"run": name, "step": step,
                            "optimizer": cfg.optimizer, "seed": cfg.seed,
                            "model_state_dict": model.state_dict()}, cpath)
                saved.append({"run": name, "optimizer": cfg.optimizer,
                              "seed": cfg.seed, "step": step,
                              "path": str(cpath),
                              "bytes": cpath.stat().st_size})

        model.train()
        Xb, yb, _ = TA.TS.sample_batch(spec, cfg.batch_size,
                                       seed=step * 100003 + cfg.seed,
                                       device=device)
        loss = F.mse_loss(TA.TS.model_scalar_train(model, Xb), yb)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()

    missing = [s for s in ckpt_steps if s % cfg.eval_every != 0]
    assert not missing, f"checkpoint steps off the eval grid: {missing}"

    with open(manifest_path, "a") as mf:
        for row in saved:
            mf.write(json.dumps(row) + "\n")

    final_si = TA.TS.staircase_index(history, degrees)
    summary = {
        **asdict(cfg),
        "final_mse": history[-1]["mse"],
        "final_fit_corr": history[-1]["fit_corr"],
        "final_deg_corr": history[-1]["deg_corr"],
        "staircase_spread": final_si["spread"],
        "staircase_span_ratio": final_si["span_ratio"],
        "half_times": {str(k): v for k, v in final_si["half_times"].items()},
        "n_learned": final_si["n_learned"],
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_trainable": n_trainable, "n_frozen": n_frozen,
        "elapsed_sec": time.time() - t0,
        "stopped_step": history[-1]["step"],
        "n_ckpts": len(saved),
    }
    if f:
        f.write(json.dumps({"_summary": summary}) + "\n")
        f.close()
    return summary, history


# ------------------------------------------------------------------ phase 2

def ablate_run(cfg, bank_jsonl, out_path, ckpt_steps, device):
    """The previous round's protocol, driven over this arm's paths.

    Conditions, the mean-ablation hooks and the per-condition readout all come
    from run_cg3_ablate; only the file layout differs. The sanity gate is kept:
    the none-condition per-degree correlations must reproduce the trajectory
    logged during training to 1e-4.
    """
    meta, traj = CG.load_run_jsonl(str(bank_jsonl))
    assert meta is not None, f"no _meta in {bank_jsonl}"
    spec, model = CG.build_model_from_meta(meta, device)
    sets = TA.TS.monomial_sets(spec)
    Xe, ye, se = TA.TS.enumerate_dataset(spec, max_n=meta["eval_n"],
                                         seed=10_000 + meta["seed"],
                                         device=device)
    name = run_name(cfg)
    cdir = ckpt_dir(cfg)
    t0, max_dev = time.time(), 0.0
    with open(out_path, "w") as f:
        f.write(json.dumps({"_meta": {
            **meta, "arm": "C-BANK", "conditions": CONDITIONS,
            "ablation": "per-position eval-set mean of Block.proj / "
                        "Block.fc2 output, recomputed per checkpoint",
            "eval_seed": 10_000 + meta["seed"],
            "n_ckpts": len(ckpt_steps)}}) + "\n")
        for step in ckpt_steps:
            sd = torch.load(cdir / f"step{step:05d}.pt", map_location=device,
                            weights_only=True)
            if "model_state_dict" in sd:
                assert sd["step"] == step and sd["run"] == name
                sd = sd["model_state_dict"]
            model.load_state_dict(sd)
            model.eval()
            means = CG.record_means(model, Xe)
            for cond in CONDITIONS:
                res = CG.eval_condition(model, Xe, ye, se, sets, cond, means)
                f.write(json.dumps({"step": step, "condition": cond,
                                    **res}) + "\n")
                if cond == "none" and step in traj:
                    logged = traj[step]["deg_corr"]
                    dev = max(abs(res["deg_corr"][k] - logged[k])
                              for k in logged)
                    max_dev = max(max_dev, dev)
                    assert dev <= 1e-4, \
                        f"SANITY FAIL {name} step {step}: dev={dev}"
            f.flush()
        f.write(json.dumps({"_summary": {
            "run": name, "optimizer": meta["optimizer"], "seed": meta["seed"],
            "n_ckpts": len(ckpt_steps), "n_conditions": len(CONDITIONS),
            "sanity_max_dev_vs_logged": max_dev,
            "elapsed_sec": round(time.time() - t0, 1)}}) + "\n")
    return max_dev


# ------------------------------------------------------------------ manifest

def write_manifest(out_dir: Path, seeds=SEEDS, optimizers=OPTIMIZERS):
    """The FULL planned grid, never the restricted selection: --seeds/--only
    and concurrent shards must not shrink the round's record of itself."""
    cells = make_cells(seeds, optimizers)
    n_ck = len(CKPT_STEPS)
    manifest = {
        "arm": "C-BANK",
        "purpose": "extend the trained-model mean-ablation bank from n=8 to "
                   "n=15 seeds so the absolute mlp_l1 selectivities can be "
                   "resolved off zero",
        "cell": "d128_l2_h4_none (L=16, D=4, staircase, 4000 steps)",
        "seeds": list(seeds), "optimizers": list(optimizers),
        "muon_lr": MUON_LR,
        "ckpt_steps": list(CKPT_STEPS),
        "ckpt_steps_source": "revision2026/pilot-CE1/ckpt_manifest.jsonl "
                             "(192 rows, 12 distinct steps) + P1-C2 runbook",
        "conditions": list(CONDITIONS),
        "conditions_source": "stier2026/run_cg3_ablate.py CONDITIONS "
                             "(imported live)",
        "n_runs": len(cells),
        "n_checkpoints": len(cells) * n_ck,
        "est_bank_gpu_hours": round(len(cells) * SEC_PER_BANK_RUN / 3600, 2),
        "est_ablate_gpu_hours": round(len(cells) * SEC_PER_ABLATE_RUN / 3600,
                                      3),
        "est_ckpt_bytes": len(cells) * n_ck * BYTES_PER_CKPT,
        "est_ckpt_gb": round(len(cells) * n_ck * BYTES_PER_CKPT / 1e9, 3),
        "seed_block_rationale":
            "100-107 repeats the archived pilot seeds on purpose (identical "
            "trajectories -> live rebuild check); 108-114 is the fresh "
            "extension, unused by arch_staircase (0-14), optaxis (0-4), "
            "lrsweep (5-7) or the pilot (100-107)",
        "runs": [{"name": run_name(c), "optimizer": c.optimizer,
                  "seed": c.seed, "n_ckpts": n_ck} for c in cells],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1)
                                           + "\n")
    return manifest


# ------------------------------------------------------------------ smoke

def run_smoke():
    """Both phases end-to-end on CPU with a tiny model, plus the archived-data
    regression check. Writes only into a temp dir."""
    import shutil
    import tempfile

    # Muon's Newton-Schulz runs in bfloat16 (grokking/muon.py, closed infra);
    # on CPU that kernel is thread-thrash bound (measured 700 ms/call at 24
    # threads vs 0.9 ms at 1). Smoke-only -- GPU runs never reach here.
    torch.set_num_threads(1)

    sys.path.insert(0, str(_THIS_DIR))
    import analyze_cbank as AN

    exercised = []
    tmp = Path(tempfile.mkdtemp(prefix="cbank_smoke_"))
    try:
        man = write_manifest(tmp, seeds=SEEDS, optimizers=OPTIMIZERS)
        assert man["n_runs"] == 30 and man["n_checkpoints"] == 360, man
        exercised.append(f"MANIFEST.json ({man['n_runs']} runs, "
                         f"{man['n_checkpoints']} ckpts, "
                         f"{man['est_ckpt_gb']} GB, "
                         f"{man['est_bank_gpu_hours']} GPU-h)")

        tiny_steps = (0, 40)
        manifest_path = tmp / "manifest.jsonl"
        for opt in OPTIMIZERS:
            cfg = TA.ArchConfig(L=10, D=3, batch_size=256, eval_n=512,
                                d_model=32, n_heads=2, n_layers=2, steps=40,
                                eval_every=20, device="cpu", freeze="none",
                                optimizer=opt, seed=100, muon_lr=MUON_LR)
            bank = tmp / f"{run_name(cfg)}.jsonl"
            summary, _ = run_with_ckpts(cfg, bank, tiny_steps, manifest_path,
                                        device="cpu")
            assert summary["n_ckpts"] == len(tiny_steps)
            assert all(k in summary for k in
                       ("staircase_spread", "half_times", "n_learned",
                        "final_fit_corr"))
            for s in tiny_steps:
                assert (tmp / run_name(cfg) / f"step{s:05d}.pt").exists()

            # the restated loop must BE train_arch.run plus saving: seeds
            # 100-107 reproducing the archived trajectories depends on it
            ref_summary, ref_hist = TA.run(cfg, out_path=None)
            got_hist = [json.loads(l) for l in open(bank) if l.strip()]
            got_hist = [r for r in got_hist if "step" in r]
            assert len(ref_hist) == len(got_hist), \
                (len(ref_hist), len(got_hist))
            dev = max(
                max(abs(a["mse"] - b["mse"]), abs(a["fit_corr"] - b["fit_corr"]),
                    max(abs(a["deg_corr"][k] - b["deg_corr"][k])
                        for k in a["deg_corr"]))
                for a, b in zip(ref_hist, got_hist))
            assert dev == 0.0, \
                f"{opt}: checkpoint saving perturbed the trajectory ({dev})"
            exercised.append(f"{opt}: restated loop == train_arch.run "
                             f"bit-for-bit over {len(ref_hist)} evals")

            abl = tmp / f"{run_name(cfg)}.ablate.jsonl"
            dev = ablate_run_smoke(cfg, bank, abl, tiny_steps, tmp)
            recs = [json.loads(l) for l in open(abl) if l.strip()]
            conds = {r["condition"] for r in recs if "condition" in r}
            assert conds == set(CONDITIONS), conds
            exercised.append(f"{opt}: bank({len(tiny_steps)} ckpts + "
                             f"manifest) -> ablate({len(CONDITIONS)} conds, "
                             f"sanity dev {dev:.1e})")

        rows = [json.loads(l) for l in open(manifest_path) if l.strip()]
        assert len(rows) == len(OPTIMIZERS) * len(tiny_steps), rows
        assert all({"run", "optimizer", "seed", "step", "path", "bytes"}
                   <= set(r) for r in rows)
        exercised.append(f"ckpt manifest schema ({len(rows)} rows)")

        cfg0 = TA.ArchConfig(d_model=32, n_heads=2, n_layers=2, seed=100,
                             optimizer="adamw")
        assert not bank_done(cfg0), "resume gate must reject a missing bank"
        exercised.append("resume gates (bank needs _summary + all ckpts, "
                         "ablate needs _summary)")

        reg = AN.regression_check()
        exercised.append(
            "n=8 regression vs published: " + ", ".join(
                f"{k} {v['point']:+.3f} (pub {v['published']:+.3f})"
                for k, v in reg["checked"].items()))
        assert reg["ok"], reg
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("SMOKE PASS: " + "; ".join(exercised) + "; zero writes outside tmp")


def ablate_run_smoke(cfg, bank, abl, ckpt_steps, tmp):
    """ablate_run with ckpt_dir redirected into the smoke's temp tree."""
    global CKPT_ROOT
    keep = CKPT_ROOT
    CKPT_ROOT = tmp
    try:
        return ablate_run(cfg, bank, abl, ckpt_steps, "cpu")
    finally:
        CKPT_ROOT = keep


# ------------------------------------------------------------------ main

def _parse_seeds(raw):
    if raw is None:
        return SEEDS
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("bank", "ablate", "both"),
                    default="both")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None,
                    help="substring filter on the run name")
    ap.add_argument("--gpu", default=None,
                    help="CUDA device id (sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--seeds", default=None,
                    help="comma list / ranges, e.g. 108-114 (default 100-114)")
    ap.add_argument("--optimizers", default=None,
                    help="comma list from {adamw,muon}")
    add_shard_args(ap)
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return

    seeds = _parse_seeds(args.seeds)
    optimizers = (OPTIMIZERS if args.optimizers is None else
                  tuple(o.strip() for o in args.optimizers.split(",")
                        if o.strip()))
    cells = make_cells(seeds, optimizers)
    if args.only:
        cells = [c for c in cells if args.only in run_name(c)]
    validate_shard_args(args)
    cells = shard_cells(cells, args.num_shards, args.shard_id)

    if args.dry_run:
        for c in cells:
            print(f"{run_name(c):<34} muon_lr={c.muon_lr} "
                  f"ckpts={len(CKPT_STEPS)} ~{SEC_PER_BANK_RUN / 60:.1f} min")
        n_ck = len(cells) * len(CKPT_STEPS)
        print(f"\n{len(cells)} runs (this selection) | "
              f"bank {len(cells) * SEC_PER_BANK_RUN / 3600:.2f} GPU-hours + "
              f"ablate {len(cells) * SEC_PER_ABLATE_RUN / 3600:.3f} GPU-hours")
        print(f"checkpoints: {n_ck} files x {BYTES_PER_CKPT / 1e6:.2f} MB = "
              f"{n_ck * BYTES_PER_CKPT / 1e9:.2f} GB "
              f"(previous round: 192 x 1.60 MB = 0.31 GB)")
        print(f"ckpt steps {list(CKPT_STEPS)}")
        print(f"conditions {list(CONDITIONS)}")
        return

    write_manifest(ARM)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.phase in ("bank", "both"):
        BANK_DIR.mkdir(parents=True, exist_ok=True)
        CKPT_ROOT.mkdir(parents=True, exist_ok=True)
        manifest_path = CKPT_ROOT / "manifest.jsonl"
        for i, cfg in enumerate(cells, 1):
            if bank_done(cfg):
                print(f"[bank {i}/{len(cells)}] skip {run_name(cfg)}",
                      flush=True)
                continue
            summary, _ = run_with_ckpts(cfg, bank_path(cfg), CKPT_STEPS,
                                        manifest_path, device=device)
            print(f"[bank {i}/{len(cells)}] {run_name(cfg)}: "
                  f"fit={summary['final_fit_corr']:.3f} "
                  f"deg4={summary['final_deg_corr']['4']:.3f} "
                  f"ckpts={summary['n_ckpts']} "
                  f"({summary['elapsed_sec'] / 60:.1f} min)", flush=True)

    if args.phase in ("ablate", "both"):
        ABLATE_DIR.mkdir(parents=True, exist_ok=True)
        for i, cfg in enumerate(cells, 1):
            if ablate_done(cfg):
                print(f"[ablate {i}/{len(cells)}] skip {run_name(cfg)}",
                      flush=True)
                continue
            if not bank_done(cfg):
                print(f"[ablate {i}/{len(cells)}] SKIP {run_name(cfg)}: "
                      f"bank incomplete (run --phase bank first)", flush=True)
                continue
            dev = ablate_run(cfg, bank_path(cfg), ablate_path(cfg),
                             CKPT_STEPS, device)
            print(f"[ablate {i}/{len(cells)}] {run_name(cfg)}: "
                  f"sanity_max_dev={dev:.2e}", flush=True)

    print(f"[cbank] DONE phase={args.phase}", flush=True)


if __name__ == "__main__":
    main()

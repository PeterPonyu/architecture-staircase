"""Direction 017 TIER-2 — the OPTIMIZER axis on the staircase anatomy.

WHY THIS BREAKS THE 004↔007 INTERLOCK ON PURPOSE.
The C-paper red-team (2026-06-14) showed the "emergence geometry ⊥ optimizer
timing" thesis is measured DIRECTLY only for the staircase index (004,
multi-optimizer); the depth/attention ownership (017) and the ~3-D subspace (010)
were run under AdamW ONLY, so their optimizer-invariance is asserted-by-design,
not shown. This runner closes the 017 leg: it re-runs the depth + freeze arms
under Muon and SGDM, so we can test whether the depth-threshold and
attention-owns-the-high-degree results are optimizer-invariant (the C claim) or
move with the optimizer (which would kill "geometry ⊥ timing").

Non-invasive: train_arch.py and the closed 004 infra are NOT modified. We
monkeypatch ONLY train_arch.build_optimizer_trainable (whose sole job was the
AdamW-fix assert) with a muon/sgdm/adamw hybrid builder over TRAINABLE params.
`split_params_for_muon` already skips requires_grad=False, so the freeze arms are
respected automatically (frozen 2-D matrices never reach Muon).

MATCHED-PERFORMANCE NOTE: Muon trains faster, so a fixed-step comparison is not
apples-to-apples for the geometry readout. The full per-eval trajectory is logged
(staircase_spread / half_times / deg_corr at every eval_every), so analyze_*.py
must compare the geometry at MATCHED fit_corr (e.g. the eval where each run first
reaches AdamW's final fit_corr), NOT just at the final step. This runner only
produces the trajectories; the matched-performance read is an analysis step.

  python run_arch_optaxis.py --smoke           # CPU self-test, no writes
  python run_arch_optaxis.py --dry-run         # print planned cells
  python run_arch_optaxis.py [--optimizers muon,sgdm] [--seeds 5]
                             [--num-shards N --shard-id I]

Output: ../../experiments/results/arch_staircase_optaxis/<name>_<opt>.jsonl
Resume-aware (skips cells whose jsonl ends with a _summary line).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.append(os.path.abspath(os.path.join(_THIS_DIR, "..")))

import train_arch as TA  # noqa: E402  (017 trainer; we monkeypatch its optimizer)
from runner_utils import (  # noqa: E402
    add_shard_args, shard_cells, validate_shard_args)

RESULTS_DIR = os.path.join(_THIS_DIR, "..", "results", "arch_staircase_optaxis")


# --- the hybrid optimizer over TRAINABLE params (Muon/SGD on 2-D hidden mats) ---
def build_opt_trainable_any(model, cfg):
    """muon/sgdm/adamw over trainable params only.

    Replaces train_arch.build_optimizer_trainable. split_params_for_muon already
    excludes requires_grad=False params, so the freeze arms are honored: e.g.
    freeze='attn' leaves only the MLP 2-D matrices in the Muon group.
    """
    if cfg.optimizer == "adamw":
        params = [p for p in model.parameters() if p.requires_grad]
        assert params, "freeze arm left no trainable parameters"
        return [torch.optim.AdamW(params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                                  weight_decay=cfg.weight_decay)]
    muon_p, adamw_p = TA.TS.split_params_for_muon(model)  # skips frozen
    assert muon_p or adamw_p, "freeze arm left no trainable parameters"
    opts = []
    if cfg.optimizer == "muon":
        if muon_p:
            opts.append(TA.TS.Muon(muon_p, lr=cfg.muon_lr, momentum=0.95,
                                   nesterov=True, ns_steps=5,
                                   weight_decay=cfg.weight_decay))
    elif cfg.optimizer == "sgdm":
        if muon_p:
            opts.append(torch.optim.SGD(muon_p, lr=cfg.muon_lr, momentum=0.95,
                                        nesterov=True,
                                        weight_decay=cfg.weight_decay))
    else:
        raise ValueError(cfg.optimizer)
    if adamw_p:
        opts.append(torch.optim.AdamW(adamw_p, lr=cfg.lr,
                                      betas=(cfg.beta1, cfg.beta2),
                                      weight_decay=cfg.weight_decay))
    return opts


# install the patch (module-level so dry-run/real both use it)
TA.build_optimizer_trainable = build_opt_trainable_any


def make_cells(optimizers, n_seeds):
    """The geometry legs that were AdamW-only in 017 — now under each optimizer.

    depth arm: n_layers {1,2,4} x d128 ; freeze arm: {none,attn,mlp} x d128/L2.
    (none/L2 is shared between the two arms; dedup by name+opt.)
    """
    cells, seen = [], set()

    def add(opt, **kw):
        cfg = TA.ArchConfig(optimizer=opt, **kw)
        key = (cfg.name(), opt)
        if key not in seen:
            seen.add(key)
            cells.append(cfg)

    for opt in optimizers:
        for nl in (1, 2, 4):                       # depth arm
            for s in range(n_seeds):
                add(opt, n_layers=nl, seed=s)
        for fz in ("none", "attn", "mlp"):         # freeze arm (d128/L2)
            for s in range(n_seeds):
                add(opt, freeze=fz, seed=s)
    return cells


def cell_path(cfg):
    return os.path.join(RESULTS_DIR, f"{cfg.name()}_{cfg.optimizer}.jsonl")


def cell_done(path):
    if not os.path.exists(path):
        return False
    last = ""
    with open(path) as f:
        for line in f:
            if line.strip():
                last = line
    try:
        return "_summary" in json.loads(last)
    except json.JSONDecodeError:
        return False


def run_smoke():
    """CPU: muon/sgdm hybrid honors freeze arms + produces the 004 schema."""
    import torch.nn.functional as F
    TA.build_optimizer_trainable = build_opt_trainable_any
    # 1. muon none-arm tiny run produces the SI summary schema
    cfg = TA.ArchConfig(L=10, D=3, batch_size=256, eval_n=512, d_model=32,
                        n_heads=2, n_layers=2, steps=40, eval_every=20,
                        device="cpu", optimizer="muon")
    summary, _ = TA.run(cfg, out_path=None)
    assert all(k in summary for k in
               ("staircase_spread", "half_times", "final_fit_corr", "n_learned"))
    # 2. muon & sgdm attn-freeze: frozen attn bit-identical, live mlp moves,
    #    and the Muon/SGD group excludes the frozen 2-D matrices.
    for opt in ("muon", "sgdm"):
        cfg_f = TA.ArchConfig(L=10, D=3, batch_size=256, eval_n=512, d_model=32,
                              n_heads=2, n_layers=2, steps=10, eval_every=20,
                              device="cpu", optimizer=opt, freeze="attn")
        torch.manual_seed(0)
        spec = TA.TS.make_spec(cfg_f)
        model = TA.TS.build_model(cfg_f, spec, "cpu")
        n_tr, n_fr = TA.freeze_component(model, "attn")
        assert n_fr > 0 and n_tr > 0
        snap_frozen = TA._component_snapshot(model, TA.ATTN_MARKS)
        snap_live = TA._component_snapshot(model, TA.MLP_MARKS)
        opts = build_opt_trainable_any(model, cfg_f)
        n_opt = sum(p.numel() for o in opts for g in o.param_groups
                    for p in g["params"])
        assert n_opt == n_tr, f"{opt}: optimizer params {n_opt} != trainable {n_tr}"
        for step in range(8):
            Xb, yb, _ = TA.TS.sample_batch(spec, 256, seed=step, device="cpu")
            loss = F.mse_loss(TA.TS.model_scalar_train(model, Xb), yb)
            for o in opts:
                o.zero_grad(set_to_none=True)
            loss.backward()
            for o in opts:
                o.step()
        for n, p in model.named_parameters():
            if n in snap_frozen:
                assert torch.equal(p, snap_frozen[n]), \
                    f"{opt}: FROZEN attn param moved: {n}"
        moved = any(not torch.equal(p, snap_live[n])
                    for n, p in model.named_parameters() if n in snap_live)
        assert moved, f"{opt}: live MLP did not train"
    print("RUN_ARCH_OPTAXIS SMOKE PASS: muon+sgdm hybrid honor freeze arms "
          "(frozen attn bit-identical, live mlp trains, opt excludes frozen); "
          "004 SI schema mirrored; zero writes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--optimizers", default="muon,sgdm",
                    help="comma list from {muon,sgdm,adamw}")
    ap.add_argument("--seeds", type=int, default=5, help="seeds per cell (0..n-1)")
    ap.add_argument("--muon-lr", type=float, default=0.02,
                    help="Muon/SGD hidden-matrix lr (VALIDATE first; red-team MAJOR-4)")
    add_shard_args(ap)
    args = ap.parse_args()
    if args.smoke:
        run_smoke()
        return
    optimizers = [o.strip() for o in args.optimizers.split(",") if o.strip()]
    cells = make_cells(optimizers, args.seeds)
    for c in cells:
        c.muon_lr = args.muon_lr
    validate_shard_args(args)
    cells = shard_cells(cells, args.num_shards, args.shard_id)
    if args.dry_run:
        for c in cells:
            print(f"{c.name()}_{c.optimizer}")
        print(f"{len(cells)} cells (this shard); optimizers={optimizers} "
              f"seeds={args.seeds}")
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for i, cfg in enumerate(cells):
        path = cell_path(cfg)
        if cell_done(path):
            print(f"[{i + 1}/{len(cells)}] skip {cfg.name()}_{cfg.optimizer}",
                  flush=True)
            continue
        summary, _ = TA.run(cfg, out_path=path)
        print(f"[{i + 1}/{len(cells)}] {cfg.name()}_{cfg.optimizer}: "
              f"fit={summary['final_fit_corr']:.3f} "
              f"n_learned={summary['n_learned']} "
              f"spread={summary['staircase_spread']}", flush=True)
    print("[arch_staircase_optaxis] DONE", flush=True)


if __name__ == "__main__":
    main()

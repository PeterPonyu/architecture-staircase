"""Direction 017 trainer — architecture-anatomy sweep over the 004 staircase.

Reuses degree_staircase end-to-end (data/probes/build_model/evaluate via the
root-README sys.path discipline; HARD RULE: degree_staircase/ and grokking/
are closed 004 infra — never modified). This file carries its own train loop
for exactly one reason: the freeze arms must intervene between model build
and optimizer build, and the optimizer must contain trainable params only.
Logged fields MIRROR train_staircase.run field-for-field (SI comparability
is kill criterion 3 of the direction doc).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
_DS_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "degree_staircase"))
if _DS_DIR not in sys.path:
    sys.path.append(_DS_DIR)

import train_staircase as TS  # noqa: E402  (closed 004 infra, read-only)

RESULTS_DIR = os.path.join(_THIS_DIR, "..", "results", "arch_staircase")

ATTN_MARKS = (".qkv.", ".proj.")
MLP_MARKS = (".fc1.", ".fc2.")


@dataclass
class ArchConfig:
    # task / data (004 main setting)
    L: int = 16
    D: int = 4
    profile: str = "staircase"
    pure_degree: int = 2
    batch_size: int = 4096
    eval_n: int = 8192
    # model — the swept factors
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4
    init_scale: float = 1.0
    freeze: str = "none"          # "none" | "attn" | "mlp"
    # optimization — FIXED AdamW (optimizer axis belongs to 004, interlock)
    optimizer: str = "adamw"
    lr: float = 1e-3
    muon_lr: float = 0.02         # kept for Config-shape parity; unused
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.98
    steps: int = 4000
    eval_every: int = 50
    seed: int = 0
    device: str = "cuda"

    def name(self) -> str:
        return (f"d{self.d_model}_l{self.n_layers}_h{self.n_heads}_"
                f"{self.freeze}_s{self.seed}")


def freeze_component(model, which: str):
    """Set requires_grad=False on the chosen component; return counts."""
    marks = {"none": (), "attn": ATTN_MARKS, "mlp": MLP_MARKS}[which]
    n_frozen = 0
    for name, p in model.named_parameters():
        if any(m in "." + name + "." for m in marks):
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_trainable, n_frozen


def build_optimizer_trainable(model, cfg: ArchConfig):
    """AdamW over TRAINABLE params only (frozen params excluded)."""
    assert cfg.optimizer == "adamw", \
        "017 fixes AdamW (optimizer axis belongs to 004; see interlock)"
    params = [p for p in model.parameters() if p.requires_grad]
    assert params, "freeze arm left no trainable parameters"
    return [torch.optim.AdamW(params, lr=cfg.lr,
                              betas=(cfg.beta1, cfg.beta2),
                              weight_decay=cfg.weight_decay)]


def run(cfg: ArchConfig, out_path: str | None = None):
    """Field-for-field mirror of train_staircase.run + freeze instrumentation."""
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    spec = TS.make_spec(cfg)

    model = TS.build_model(cfg, spec, device)
    n_trainable, n_frozen = freeze_component(model, cfg.freeze)
    optimizers = build_optimizer_trainable(model, cfg)
    degrees = sorted(TS.monomial_sets(spec).keys())

    history: list = []
    t0 = time.time()
    f = open(out_path, "w") if out_path else None
    if f:
        f.write(json.dumps({"_meta": {**asdict(cfg),
                                      "n_trainable": n_trainable,
                                      "n_frozen": n_frozen}}) + "\n")

    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0:
            mse, fit_corr, deg_corr = TS.evaluate(model, spec, cfg, device)
            rec = {"step": step, "mse": mse, "fit_corr": fit_corr,
                   "deg_corr": {str(k): v for k, v in deg_corr.items()}}
            si = TS.staircase_index(history + [rec], degrees)
            rec["staircase_spread"] = si["spread"]
            rec["staircase_span_ratio"] = si["span_ratio"]
            history.append(rec)
            if f:
                f.write(json.dumps(rec) + "\n")
                f.flush()

        model.train()
        Xb, yb, _ = TS.sample_batch(spec, cfg.batch_size,
                                    seed=step * 100003 + cfg.seed,
                                    device=device)
        f_pred = TS.model_scalar_train(model, Xb)
        loss = F.mse_loss(f_pred, yb)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()

    elapsed = time.time() - t0
    final_si = TS.staircase_index(history, degrees)
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
        "n_trainable": n_trainable,
        "n_frozen": n_frozen,
        "elapsed_sec": elapsed,
        "stopped_step": history[-1]["step"],
    }
    if f:
        f.write(json.dumps({"_summary": summary}) + "\n")
        f.close()
    return summary, history


# ---------------------------------------------------------------------------
# Smoke — freeze-instrument self-test, CPU, no files.
# ---------------------------------------------------------------------------

def _tiny_cfg(**kw):
    base = dict(L=10, D=3, batch_size=256, eval_n=512, d_model=32, n_heads=2,
                n_layers=2, steps=40, eval_every=20, device="cpu")
    base.update(kw)
    return ArchConfig(**base)


def _component_snapshot(model, marks):
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if any(m in "." + n + "." for m in marks)}


def run_smoke():
    # 1. full arm mirrors the 004 trainer's field schema (SI comparability)
    cfg = _tiny_cfg()
    summary, history = run(cfg, out_path=None)
    ts_summary, _ = TS.run(TS.Config(
        L=10, D=3, batch_size=256, eval_n=512, d_model=32, n_heads=2,
        n_layers=2, steps=40, eval_every=20, device="cpu"), out_path=None)
    missing = set(ts_summary.keys()) - set(summary.keys()) - {"muon_lr"}
    assert not missing, f"summary fields missing vs 004 schema: {missing}"
    assert all(k in summary for k in
               ("staircase_spread", "half_times", "n_learned"))

    # 2/3. freeze arms: frozen component bit-identical after training,
    #      the other component moves, optimizer counts are consistent
    for which, frozen_marks, live_marks in (
            ("attn", ATTN_MARKS, MLP_MARKS), ("mlp", MLP_MARKS, ATTN_MARKS)):
        cfg_f = _tiny_cfg(freeze=which)
        torch.manual_seed(cfg_f.seed)
        device = "cpu"
        spec = TS.make_spec(cfg_f)
        model = TS.build_model(cfg_f, spec, device)
        n_tr, n_fr = freeze_component(model, which)
        assert n_fr > 0 and n_tr > 0
        snap_frozen = _component_snapshot(model, frozen_marks)
        snap_live = _component_snapshot(model, live_marks)
        opts = build_optimizer_trainable(model, cfg_f)
        n_opt = sum(p.numel() for g in opts[0].param_groups
                    for p in g["params"])
        assert n_opt == n_tr, "optimizer params != trainable params"
        for step in range(10):
            Xb, yb, _ = TS.sample_batch(spec, 256, seed=step, device=device)
            loss = F.mse_loss(TS.model_scalar_train(model, Xb), yb)
            opts[0].zero_grad(set_to_none=True)
            loss.backward()
            opts[0].step()
        for n, p in model.named_parameters():
            if n in snap_frozen:
                assert torch.equal(p, snap_frozen[n]), \
                    f"FROZEN PARAM MOVED: {n} ({which} arm)"
        moved = any(not torch.equal(p, snap_live[n])
                    for n, p in model.named_parameters() if n in snap_live)
        assert moved, f"live component did not train ({which} arm)"

    print(f"SMOKE PASS: schema mirrors 004 ({len(summary)} fields, SI suite "
          f"present); attn/mlp freeze arms verified bit-identical-frozen + "
          f"live-component-trains + optimizer excludes frozen; zero writes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        run_smoke()
    else:
        print("use run_arch_sweep.py for the grid; --smoke for the self-test")

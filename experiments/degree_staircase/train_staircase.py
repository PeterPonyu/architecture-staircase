"""Direction 004 — staircase trainer (fresh-batch online regression).

One run = one (optimizer, profile, L, D, seed) configuration. Adapts the
grokking trainer (Config dataclass + dynamic argparse + the muon/adamw/sgdm
hybrid split) but for the F1 design:
  - REGRESSION (MSE) of the scalar model function f(x) = logit[1]-logit[0]
    onto the boolean staircase target g(x);
  - ONLINE fresh-batch training (a new batch sampled every step) to avoid the
    grokking memorization phase entirely — we study feature-learning ORDER, not
    train/test generalization delay;
  - per-eval per-degree Walsh correlations + the staircase index.

GrokTransformer is reused by import from the sibling grokking package, with
vocab_size / seq_len coming from data.py (StaircaseSpec).

Flags
-----
--smoke : print the five labeled smoke lines, run <=1 step, write NO files, exit 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict

# --- import the grokking infra without modifying it (spec-mandated pattern) ---
# This dir MUST win for the names that collide with grokking (e.g. `data`), so
# put it at the front of sys.path and the grokking dir at the BACK (we only pull
# `model` / `muon` from there, which do not collide).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
_GROKKING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "grokking"))
if _GROKKING_DIR not in sys.path:
    sys.path.append(_GROKKING_DIR)

import torch
import torch.nn.functional as F

from model import GrokTransformer            # noqa: E402  (grokking infra)
from muon import Muon, split_params_for_muon  # noqa: E402  (grokking infra)

from data import StaircaseSpec, sample_batch, enumerate_dataset, monomial_sets  # noqa: E402
from probes import model_scalar, degree_correlations, staircase_index           # noqa: E402


@dataclass
class Config:
    # task / data
    L: int = 16                  # boolean variables
    D: int = 4                   # highest degree (stages 1..D)
    profile: str = "staircase"   # "staircase" | "mixed" | "pure"
    pure_degree: int = 2         # only for profile == "pure"
    batch_size: int = 4096       # fresh batch per step (online)
    eval_n: int = 8192           # samples for the per-degree probe
    # model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4
    init_scale: float = 1.0
    # optimization
    optimizer: str = "adamw"     # "adamw" | "muon" | "sgdm"
    lr: float = 1e-3             # AdamW lr (and AdamW side of hybrids)
    muon_lr: float = 0.02        # Muon/SGDM lr for hidden matrices
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.98
    steps: int = 4000
    eval_every: int = 50
    seed: int = 0
    device: str = "cuda"


def make_spec(cfg: Config) -> StaircaseSpec:
    return StaircaseSpec(L=cfg.L, D=cfg.D, profile=cfg.profile,
                         pure_degree=cfg.pure_degree, seed=cfg.seed)


def build_model(cfg: Config, spec: StaircaseSpec, device: str) -> GrokTransformer:
    return GrokTransformer(
        vocab_size=spec.vocab_size,
        seq_len=spec.seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        mlp_ratio=cfg.mlp_ratio,
        init_scale=cfg.init_scale,
    ).to(device)


def build_optimizer(model, cfg: Config):
    """Same muon/adamw/sgdm hybrid split as grokking's train.py."""
    if cfg.optimizer == "adamw":
        return [torch.optim.AdamW(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)]
    elif cfg.optimizer == "muon":
        muon_p, adamw_p = split_params_for_muon(model)
        opt_muon = Muon(muon_p, lr=cfg.muon_lr, momentum=0.95, nesterov=True,
                        ns_steps=5, weight_decay=cfg.weight_decay)
        opt_adamw = torch.optim.AdamW(
            adamw_p, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay)
        return [opt_muon, opt_adamw]
    elif cfg.optimizer == "sgdm":
        sgd_p, adamw_p = split_params_for_muon(model)
        opt_sgd = torch.optim.SGD(sgd_p, lr=cfg.muon_lr, momentum=0.95,
                                  nesterov=True, weight_decay=cfg.weight_decay)
        opt_adamw = torch.optim.AdamW(
            adamw_p, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay)
        return [opt_sgd, opt_adamw]
    else:
        raise ValueError(cfg.optimizer)


@torch.no_grad()
def evaluate(model, spec: StaircaseSpec, cfg: Config, device: str):
    """Held-out MSE/correlation + per-degree Walsh correlations on a fixed eval set."""
    Xe, ye, se = enumerate_dataset(spec, max_n=cfg.eval_n, seed=10_000 + cfg.seed,
                                   device=device)
    f = model_scalar(model, Xe)
    mse = F.mse_loss(f, ye).item()
    # overall fit correlation between model scalar and target
    fc = f - f.mean()
    yc = ye - ye.mean()
    denom = fc.norm() * yc.norm()
    fit_corr = float(torch.dot(fc, yc) / denom) if denom > 1e-12 else 0.0
    sets = monomial_sets(spec)
    deg_corr = degree_correlations(f, se, sets)
    return mse, fit_corr, deg_corr


def run(cfg: Config, out_path: str | None = None):
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    spec = make_spec(cfg)

    model = build_model(cfg, spec, device)
    optimizers = build_optimizer(model, cfg)
    degrees = sorted(monomial_sets(spec).keys())

    history: list = []
    t0 = time.time()

    f = open(out_path, "w") if out_path else None
    if f:
        f.write(json.dumps({"_meta": asdict(cfg)}) + "\n")

    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0:
            mse, fit_corr, deg_corr = evaluate(model, spec, cfg, device)
            rec = {
                "step": step,
                "mse": mse,
                "fit_corr": fit_corr,
                "deg_corr": {str(k): v for k, v in deg_corr.items()},
            }
            si = staircase_index(history + [rec], degrees)
            rec["staircase_spread"] = si["spread"]
            rec["staircase_span_ratio"] = si["span_ratio"]
            history.append(rec)
            if f:
                f.write(json.dumps(rec) + "\n")
                f.flush()

        # online fresh-batch regression step
        model.train()
        Xb, yb, _ = sample_batch(spec, cfg.batch_size, seed=step * 100003 + cfg.seed,
                                 device=device)
        f_pred = model_scalar_train(model, Xb)
        loss = F.mse_loss(f_pred, yb)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()

    elapsed = time.time() - t0
    final_si = staircase_index(history, degrees)
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
        "elapsed_sec": elapsed,
        "stopped_step": history[-1]["step"],
    }
    if f:
        f.write(json.dumps({"_summary": summary}) + "\n")
        f.close()
    return summary, history


def model_scalar_train(model, X):
    """Training-time scalar output (grad-enabled): logit[1]-logit[0] at EQ."""
    logits = model(X)
    return logits[:, 1] - logits[:, 0]


# ---------------------------------------------------------------------------
# Smoke: five labeled lines, <=1 training step, NO files, exit 0.
# ---------------------------------------------------------------------------
def run_smoke():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cfg = Config(L=16, D=4, profile="staircase", optimizer="muon",
                 batch_size=512, seed=0)
    spec = make_spec(cfg)

    # 1. dataset shape
    X, y, signs = sample_batch(spec, cfg.batch_size, seed=0, device=device)
    print(f"SMOKE DATASET SHAPE: X={tuple(X.shape)}, y={tuple(y.shape)}")

    # 2. param count
    model = build_model(cfg, spec, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SMOKE PARAM COUNT: {n_params}")

    # 3. forward + loss
    f_pred = model_scalar_train(model, X)
    loss = F.mse_loss(f_pred, y)
    print(f"SMOKE FORWARD LOSS: {loss.item():.6f}")

    # 4. one Muon-hybrid optimizer step
    optimizers = build_optimizer(model, cfg)
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    loss.backward()
    for opt in optimizers:
        opt.step()
    print("SMOKE OPTIMIZER STEP: OK")

    # 5. bonus: per-degree probe on the (untrained) model end-to-end
    sets = monomial_sets(spec)
    with torch.no_grad():
        f_eval = model_scalar(model, X)
    deg_corr = degree_correlations(f_eval, signs, sets)
    vec = ", ".join(f"d{k}={deg_corr[k]:+.3f}" for k in sorted(deg_corr))
    print(f"SMOKE DEGREE PROBE: [{vec}]")


def parse_args() -> tuple[Config, bool]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Run smoke checks and exit (no files written)")
    defaults = asdict(Config())
    for k, v in defaults.items():
        ap.add_argument(f"--{k}", type=type(v) if v is not None else str, default=v)
    a = vars(ap.parse_args())
    smoke = a.pop("smoke")
    cfg = Config(**a)
    return cfg, smoke


if __name__ == "__main__":
    cfg, smoke = parse_args()
    if smoke:
        run_smoke()
        sys.exit(0)
    summary, _ = run(cfg, out_path=None)
    print(json.dumps(summary, indent=2))

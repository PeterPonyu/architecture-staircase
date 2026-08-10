#!/bin/bash
# Direction 010 — idealization-hardening arm launcher (portable).
# Grid: (b256, lr3e-4) + matched (b64, lr1e-3), L{64,128,256} x 3 seeds = 18 runs,
# eval_every=10 (floor-free emergence). Resume-safe: skips jsonl ending in _summary.
#
# Paths are relative to this repository (no author-machine home paths).
# Override interpreter with: PYTHON=/path/to/python ./launch_ideal_arm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SUB="$ROOT/experiments/induction_subspace"
OUT="$ROOT/experiments/results/induction_subspace_ideal"
PYTHON="${PYTHON:-python3}"
RUNPY="${TMPDIR:-/tmp}/run010_arm_$$.py"

mkdir -p "$OUT"

cat > "$RUNPY" <<PYEOF
import os, sys
sys.path.insert(0, r"${SUB}")
from train_subspace import Config, run
out = r"${OUT}"
os.makedirs(out, exist_ok=True)
# idealization-hardening: larger batch + smaller lr (closer to population gradient flow)
# plus matched batch64 cells, all at eval_every=10 (floor-free emergence)
cells = [(256, 0.0003), (64, 0.001)]
for bs, lr in cells:
    for L in [64, 128, 256]:
        for seed in range(3):
            name = f'b{bs}_lr{lr}_L{L}_s{seed}'
            path = os.path.join(out, name + '.jsonl')
            if os.path.exists(path) and '_summary' in open(path).read()[-2000:]:
                print('skip', name, flush=True); continue
            cfg = Config(task='repeat', seq_len=L, batch_size=bs, lr=lr,
                         seed=seed, eval_every=10)
            s, _ = run(cfg, out_path=path)
            print(f"{name}: captured={s['final_captured_total']:.4f} "
                  f"pca_dim={s['final_pca_dim']} emerge={s['emergence_step']}", flush=True)
print('### 010 IDEAL ARM DONE ###', flush=True)
PYEOF

if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s exp010ideal \
    "OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 ${PYTHON} ${RUNPY} 2>&1 | tee ${TMPDIR:-/tmp}/exp010ideal.log"
  sleep 2
  tmux list-sessions || true
else
  echo "tmux not found; running in foreground with ${PYTHON}"
  OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 "$PYTHON" "$RUNPY"
fi

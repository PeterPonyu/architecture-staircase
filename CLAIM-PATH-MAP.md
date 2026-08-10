# Claim → path map (Paper C / architecture-staircase)

Use this map to recompute headline numbers from the deposit without guessing paths.

| Claim | Path(s) |
|---|---|
| SI TOST / SI=1 (n=15 × 3 optimizers) | `experiments/results/degree_staircase/` + `revision2026/C/t02_*` + `degree_staircase/analyze_si_inference.py` |
| AdamW freeze n=15; Muon/SGDM ownership | `experiments/results/arch_staircase/` + `arch_staircase_optaxis/` |
| Layer-resolved Muon localize + floor | `experiments/results/ieee_gap_20260705/C/c2_ownership/` + `gapC_verdict.json` |
| Target-family robustness | `experiments/results/ieee_gap_20260705/C/c1_targets/` |
| Degree-5 frontier freeze note | `experiments/results/ieee_gap_20260705/C/c5_frontier/` |
| Scale grid 24-config + L24 freeze8 | `experiments/results/arch_staircase_scale/` |
| Mean-ablation / selectivity bootstrap | `revision2026/cg3-C/` + `revision2026/C/t05_*` / `t07_*` |
| Timing ratios (1.6–12×) | `experiments/results/induction_emergence/` + `induction_fine/` |
| Induction subspace PCA null | `experiments/results/induction_subspace/` (+ ideal/optaxis variants if present) |
| Figure scripts (C only) | `papers/figs/make_C_*.R`, `make_gap20260705_C_figs_r.R`, `fig_pipeline.R` |

**Synthetic only.** No third-party datasets. Checkpoints not shipped; ablation from `*.ablate.jsonl`.

**Code deps for retrain (smoke):** `experiments/grokking/{model,muon}.py`, staircase/arch train scripts under `experiments/`.

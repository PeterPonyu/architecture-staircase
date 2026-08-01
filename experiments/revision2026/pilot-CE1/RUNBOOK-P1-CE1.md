# Phase-1 PILOT runbook — Papers C + E1 (BOX-2, 1x RTX 4090 D) — 2026-07-16

88 runs, one sequential queue, order **C2 -> C1 -> E1a -> E1b** (checkpoint bank first).
Source plan: `revision-plan-2026-0716.md` (P1-C1/C2/E1a/E1b); evidence notes
`revision-evidence-2026-07/{C,E1}.md`. Hard rules honored: only these four
blocks; new disk << 8G (ckpts ~0.31G measured-size-checked at launch, logs ~50MB);
nohup queue + PROGRESS.json heartbeat + 2h deadman tarball; archived configs extracted
verbatim from laptop results `_meta` headers; new runs differ ONLY in the stated factor
+ fresh seeds. The FineWeb/code/Pythia downloads under `/root/autodl-tmp/dlr/data` are
NOT touched.

## Archived baseline configs (extracted from `_meta`, verified 07-16)

C freeze-headline BASELINE cell (`results/arch_staircase/d128_l2_h4_none_s*.jsonl` and
`results/arch_staircase_optaxis/d128_l2_h4_none_s*_muon.jsonl`):

```
L=16 D=4 profile=staircase pure_degree=2 batch_size=4096 eval_n=8192
d_model=128 n_heads=4 n_layers=2 mlp_ratio=4 init_scale=1.0 freeze=none
steps=4000 eval_every=50 weight_decay=0.0 beta1=0.9 beta2=0.98
AdamW arm: optimizer=adamw lr=0.001
Muon arm:  optimizer=muon  muon_lr=0.01 lr=0.001   # archived optaxis used 0.01,
                                                   # NOT the ArchConfig default 0.02
n_trainable=397440
```

E1 main-grid cell (`results/repeated_data/*.jsonl`, e.g. small_low_U2M_E10_s0):

```
seq_len=128 batch_size=32 generator=markov vocab=256 mlp_ratio=4 init_scale=1.0
n_canary=8 canary_repeats=4 optimizer=adamw lr=0.0003 weight_decay=0.01
beta1=0.9 beta2=0.98 eval_every=1(epoch) val_batches=4 total_budget=20000000
capacity in {small(2.53M) med(5.56M) large(9.93M)}; U*E = 20M
```

Generator description (ACTUAL code path, `repeated_data/data.py` — per the E1
evidence-note flag, NOT the paper's old Dirichlet description): branching-masked
Gaussian logits + temperature softmax over V=256; presets
low = order-2 (hashed 2-token context), branching=2, temperature=0.4;
med = order-1, branching=8, temperature=1.0;
high = order-1, branching=64, temperature=2.0.
Training pool = contiguous 128-token windows of one sampled stream, reshuffled
per epoch; canaries: 8 x 128 tokens from the same law on a dedicated seed, each
spliced 4x at uniform-random offsets.

## Block P1-C2 (16 runs) — checkpoint donors, runs FIRST

Baseline cell above, NO freezing, {adamw, muon} x seeds **100..107** (fresh; archived
used 0..14 adamw / 0..4 muon). Identical training trajectory to the archived config
(checkpoint saving consumes no RNG); only additions are `pilot_block`, `ckpt_steps`,
`ckpt_dir` keys in `_meta`. Checkpoints: model `state_dict` (fp32, ~1.6MB each) at 12
log-spaced eval-grid steps {0,50,100,150,200,300,450,700,1050,1650,2600,4000}, saved
right after the same-step eval (state = model after exactly `step` updates). Bank:
`/root/autodl-tmp/dlr/pilot/ckpts/<run>/step*.pt` + append-only
`ckpts/manifest.jsonl` (run, opt, seed, step, path, bytes). Budget check:
1.6MB x 12 x 16 ~ 0.31G << 2-3G cap — no thinning needed.

## Block P1-C1 (30 runs) — task-instance generality

DESIGN NOTE (discovered from code, documented here): in the archived harness the
monomial target draw is coupled to the training seed (`make_spec` sets
`StaircaseSpec.seed = cfg.seed`), so archived seeds conflate init/batch stream with
target instance. C1 DECOUPLES them via a `target_seed` field (new runner monkeypatches
`make_spec` only; closed 004 infra untouched): 3 fresh documented draws x {adamw,muon}
x 5 fresh training seeds **100..104**. Everything else identical to the baseline cell.

Fresh monomial draws (L=16, D=4, disjoint staircase subsets; verified identical
laptop-vs-box):

```
target_seed 2101: {1:[7],  2:[1,4],  3:[3,5,9],   4:[2,8,14,15]}
target_seed 2102: {1:[10], 2:[1,8],  3:[6,12,13], 4:[0,2,5,7]}
target_seed 2103: {1:[10], 2:[7,12], 3:[3,4,15],  4:[8,9,13,14]}
(archived reference, seed 0: {1:[12], 2:[9,10], 3:[6,8,11], 4:[2,5,13,14]})
```

## Block P1-E1a (30 runs) — repeated-data fine grid {6,8}

The 5 main-grid cells WITHOUT Sec-P4 fine-grid audits. Paper E1 Sec P4 fine grid
(n in {6,8,12,16}) covers med/med, large/high, small/high, large/low
(`results/repeated_data_finegrid/`); complement = **small/low, small/med, med/low,
med/high, large/med** (matches the plan's "5 cells"). Caveat documented: small/low +
small/med already have 10-seed {6,8} rungs from the boundary seed audit
(`repeated_data_ultragoal_seed_audit`); the pilot's fresh-seed points there are
independent replication, while med/low, med/high, large/med get their FIRST {6,8}
coverage (their R_free=4 verdicts currently rest on the coarse 4-vs-10 rungs).

Cells: 5 cells x {(U=3,333,333, E=6), (U=2,500,000, E=8)} x seeds **20,21,22**
(fresh; archived main grid used 0-2/0-4, audit used 3-14). B=20M held. Generator
presets exactly as archived (U tags: U3p33333M / U2p5M, matching finegrid naming).

## Block P1-E1b (12 runs) — boundary bootstrap

Cell choice (justification): the two R_free=10 boundary cells (small/low, small/med)
are ALREADY at 15 seeds at n=10 (seed audit; paper App. bootstrap) — re-running them
buys nothing. Among the remaining 3-seed cells whose verdict is called at the n=10
rung, **med/low** is the tightest call in both conventions: smallest n=10 excess
margin (0.0776 vs the 0.05 threshold; next: large/low 0.0879, small/high 0.0900) AND
the evidence note's flagged 0.0016-nat relative-convention margin at n=4 (eps_rel
0.0108 vs excess(4) 0.0092 — the fragile "7/9 verdicts unchanged" leg). So E1b =
**med/low, U=2,000,000, E=10 (U2M_E10) x 12 fresh seeds 20..31** -> n=15 total
with the 3 archived seeds.

## Outputs (box)

```
/root/autodl-tmp/dlr/pilot/
  results/p1_c2/d128_l2_h4_none_s{100..107}_{adamw,muon}.jsonl
  results/p1_c1/d128_l2_h4_none_t{2101,2102,2103}_s{100..104}_{adamw,muon}.jsonl
  results/p1_e1a/{cell}_U{tag}_E{6,8}_s{20,21,22}.jsonl
  results/p1_e1b/med_low_U2M_E10_s{20..31}.jsonl
  ckpts/<c2-run>/step*.pt + ckpts/manifest.jsonl
  PROGRESS.json (heartbeat, rewritten after every run)
  pilot_logs_*.tgz (2h deadman, keeps last 3, excludes .pt)
```

All jsonls are append-only with `_meta` first-line headers in the archived house
format (C2/C1 via the mirrored train loop / TA.run; E1 via train_repeat.run verbatim).
Smoke gate before launch: one short GPU run per harness (C adamw+muon incl. one ckpt
save; E1 small/low tiny budget), asserting finite losses + parseable jsonl.
Retrieval: jsonls + ckpt manifest (NOT the .pt files) ->
`~/Desktop/dl-research/experiments/revision2026/pilot-CE1/`.

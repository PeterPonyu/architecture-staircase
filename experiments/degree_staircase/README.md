# Direction 004 — Muon × by-degree learning staircase

Tests whether Muon's orthogonalized (spectrum-flattening) updates erase the
by-degree sequential learning staircase on softmax transformers. Controlled
synthetic boolean (+/-1) data with a known Walsh/monomial-degree decomposition
(staircase polynomial: one clean monomial per degree 1..D); a per-degree
correlation probe tracks how much of each degree the model has learned through
training, compared across optimizer families {muon, adamw, sgdm}. Online
fresh-batch regression — no grokking memorization phase. Reuses the grokking
infra by import (GrokTransformer, Muon, split_params_for_muon).

## Smoke check (no files written, <60 s)
```
python train_staircase.py --smoke      # five labeled smoke lines
python probes.py                       # probe self-test (PASS on known function)
```

## Dry run (prints planned cells, launches nothing)
```
python run_staircase.py --dry-run
```

## Real grid (run when ready — do NOT launch yet)
```
python run_staircase.py    # muon/adamw/sgdm × {staircase,mixed,pure3} × seeds 0-4 = 45 cells
```
Results land in `experiments/results/degree_staircase/`.

## Reference
See `directions/004-muon-degree-staircase.md` for the full research write-up
(authoritative design: `.omc/research/novelty-004.md`, F1 winner).

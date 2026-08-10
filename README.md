<!-- Paper C claim map: see CLAIM-PATH-MAP.md. All paper-C tasks are synthetic (Boolean staircase + induction streams); no MNIST/WikiText. -->

# Architecture-shaped degree-staircase emergence — code & data

Reproducibility archive: **experiment code and per-run result logs only**.
Manuscript and write-up/derivation documents are intentionally **not** included.

## Contents
- `experiments/<study>/` — runner / analysis code per sub-experiment.
- `experiments/results/` — per-run logs (JSON/JSONL) behind every reported number.

## Reproducing
The committed per-run logs are the recorded outputs. To re-run a study from
scratch (GPU recommended): `python experiments/<study>/run_*.py`. Runs are seeded
(seed lists appear in result-log filenames). Dependencies: Python 3.11+, PyTorch,
numpy. All inputs are synthetic and fully specified in the code, except large
standard datasets (MNIST / WikiText) which are not bundled.

## Scale-hardening additions (v1.3, 2026-07)
- `experiments/arch_staircase/run_20260708_scalegrid.py` + `experiments/results/arch_staircase_scale/`
  (144 runs): the 24-configuration scale grid L {16,24,32} x d {256,512} x depth {2,4} x
  {adamw,muon} (scalegrid, 96) and the 8-seed freeze-ownership audit at L=24/d=256 (freeze8, 48).

## v1.5.1 (2026-08-10) — reviewer-safety scrub
- Vendored `experiments/figstyle.py` so figure analyzers import without a private monorepo.
- Scrubbed remaining absolute/home paths (`analyze_induction.py`, `launch_ideal_arm.sh`).
- PeerJ AI-in-code disclosure pack lives with the submission materials
  (`papers/peerj-C/ai_code_disclosure/`), not in this code-only archive.

## License
Code: MIT (`LICENSE`). Result logs: CC BY 4.0. See `CITATION.cff`.

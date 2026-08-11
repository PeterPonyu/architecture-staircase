<!-- Claim map: see CLAIM-PATH-MAP.md. All tasks in this archive are synthetic (Boolean staircase + induction streams); no MNIST/WikiText. -->

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

## Which archive to use for review
Use **tag `v1.5.4` or later**, or Zenodo concept DOI
[`10.5281/zenodo.21020348`](https://doi.org/10.5281/zenodo.21020348)
(always resolves to the latest published version).

**Do not use tag `v1.5.3` or earlier** for review: the v1.5.3 Zenodo tarball shipped a
nested `.omc` process cache (GitHub tag was clean); older tags also retain series-label /
process artifacts fixed in v1.5.3+. Git history retains older blobs; that is expected for
open source and is not rewritten. Prefer the latest Zenodo version DOI on the concept page.

## v1.5.4 (2026-08-11) — clean Zenodo rebuild (no nested `.omc`)
- Pack via `git archive` + hard excludes; fail-closed verification before upload.
- Extends `.gitignore` for process/IDE caches. No scientific log changes.
- Supersedes the dirty v1.5.3 Zenodo file for review.

## v1.5.3 (2026-08-11) — series-label / process-artifact scrub
- Removed internal manuscript-series codenames from README / claim map / source comments.
- Redacted multi-paper process status JSON and pilot runbook sibling-series language.
- PeerJ AI-in-code disclosure pack lives with the journal submission materials, not in
  this code-only archive. The PeerJ **BEFORE** zip uses `/REDACTED/...` path placeholders
  as intentional AI-edit contrast — use **AFTER** / this archive for release code.

## v1.5.2 (2026-08-10) — residual infra-identity redaction
- Redacted lab hostname / GPU inventory strings from redteam status JSON and pilot runbook.
- Superseded for review by v1.5.4.

## v1.5.1 (2026-08-10) — reviewer-safety scrub
- Vendored `experiments/figstyle.py` so figure analyzers import without a private monorepo.
- Scrubbed remaining absolute/home paths (`analyze_induction.py`, `launch_ideal_arm.sh`).
- Historical tags (e.g. v1.5.0) and git history retain pre-scrub paths; prefer latest tag.

## License
Code: MIT (`LICENSE`). Result logs: CC BY 4.0. See `CITATION.cff`.

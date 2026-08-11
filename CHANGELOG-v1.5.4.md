# v1.5.4 (2026-08-11) — Zenodo pack hygiene

## Why
v1.5.3’s published tarball accidentally included a nested `.omc/state/` process
cache under `experiments/induction_emergence/` (local absolute paths). GitHub
tag `v1.5.3` itself was clean; the dirty artifact came from a workdir pack.
Publish a clean rebuild and pin reviewers to this version.

## Changes
- Add `pack_zenodo_tarball.sh` (prefer `git archive` + fail-closed verification).
- Extend `.gitignore` for process/IDE caches (`.omc`, `.cursor`, `.claude`, …).
- CITATION.cff / README / `.zenodo.json` → 1.5.4.
- No scientific result-log changes.

## Reviewer pin
Use **v1.5.4+** only. Do not use the v1.5.3 Zenodo file for review (dirty pack).
Concept DOI unchanged: 10.5281/zenodo.21020348.

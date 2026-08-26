# v1.5.4 (2026-08-11) — Zenodo pack hygiene

## Why
v1.5.3's published tarball accidentally included a nested `.omc/state/` process
cache under `experiments/induction_emergence/` (local absolute paths). GitHub
tag `v1.5.3` itself was clean. This version is a rebuild without that cache.

## Changes
- Add `pack_zenodo_tarball.sh` (prefer `git archive` + fail-closed verification).
- Extend `.gitignore` for process/IDE caches (`.omc`, `.cursor`, `.claude`, …).
- CITATION.cff / README / `.zenodo.json` → 1.5.4.
- No scientific result-log changes.

Concept DOI unchanged: 10.5281/zenodo.21020348.

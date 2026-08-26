# v1.5.4 (2026-08-11) — Zenodo pack hygiene

## Why
This version is the experiment code and recorded result logs packed from a
clean `git archive` (concept DOI 10.5281/zenodo.21020348; version DOI
10.5281/zenodo.21882597). GitHub tag `v1.5.3` itself was clean; the published
v1.5.3 tarball shipped a nested `.omc/state/` process cache under
`experiments/induction_emergence/` (local absolute paths).

## Changes
- Add `pack_zenodo_tarball.sh` (prefer `git archive` + fail-closed verification).
- Extend `.gitignore` for process/IDE caches (`.omc`, `.cursor`, `.claude`, …).
- CITATION.cff / README / `.zenodo.json` → 1.5.4.
- No scientific result-log changes.

Concept DOI unchanged: 10.5281/zenodo.21020348.

# v1.5.3 (2026-08-11) — local prep (not yet published)

## Why
Remove internal manuscript-series labels and multi-paper process artifacts from the
public reproducibility archive used by journal reviewers (Zenodo concept DOI
10.5281/zenodo.21020348).

## Changes
- README / CLAIM-PATH-MAP: removed internal series codenames and private packaging paths.
- Source comments: neutralized manuscript-series / private-tree path references
  (including gap-battery tool module docstrings).
- `.zenodo.json`: title/description aligned to the manuscript objective (no local
  series codenames); keywords refreshed for the two-probe study.
- `experiments/results/figures-redteam/*.json` multi-paper process status files
  replaced with redacted stubs (scientific per-run logs untouched).
- `experiments/revision2026/pilot-CE1/RUNBOOK-P1-CE1.md`: sibling-series prose scrubbed.
- CITATION.cff version → 1.5.3.

## Deferred / MEDIUM
- Historical path stems (`make_C_*.R`, `papers/figs/C_*.png`, `revision2026/C/`)
  remain as filesystem identifiers; renaming would break figure pipelines.
- Older Zenodo versions (≤v1.5.2) cannot be rewritten; pin reviewers to v1.5.3+.

## Publish steps (user must run)
See the PeerJ submit-runbook file `PUBLISH-v1.5.3-STEPS.md` (kept outside this
code-only archive so local ops paths are not deposited).

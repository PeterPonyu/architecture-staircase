# Two-probe architecture staircase

**Live door:** https://peterponyu.github.io/architecture-staircase/

The door is the entrance. It loads freeze and ablation probes on the
architecture-shaped degree staircase: Probes, Staircase, Scale, Subspace,
Ledger, and Rebuild. Train the pathway, deposit the computation.

Repository: https://github.com/PeterPonyu/architecture-staircase

## Archive
- `experiments/<study>/` — runner / analysis code per sub-experiment.
- `experiments/results/` — per-run logs (JSON/JSONL) behind every reported number.
- `portal/` — live-door source (excluded from Zenodo `git archive` packs).

## Reproducing
The committed per-run logs are the recorded outputs. To re-run a study from
scratch (GPU recommended): `python experiments/<study>/run_*.py`. Runs are seeded
(seed lists appear in result-log filenames). Dependencies: Python 3.11+, PyTorch,
numpy. All inputs are synthetic and fully specified in the code, except large
standard datasets which are not bundled.

## Scale-hardening additions (v1.3, 2026-07)
- `experiments/arch_staircase/run_20260708_scalegrid.py` + `experiments/results/arch_staircase_scale/`
  (144 runs): the 24-configuration scale grid L {16,24,32} x d {256,512} x depth {2,4} x
  {adamw,muon} (scalegrid, 96) and the 8-seed freeze-ownership audit at L=24/d=256 (freeze8, 48).

## Which archive to use
Use **tag `v1.5.4` or later**, or Zenodo concept DOI
[`10.5281/zenodo.21020348`](https://doi.org/10.5281/zenodo.21020348)
(always resolves to the latest published version). Version pin:
[`10.5281/zenodo.21882597`](https://doi.org/10.5281/zenodo.21882597).

**Do not use tag `v1.5.3` or earlier**: the v1.5.3 Zenodo tarball shipped a
nested process cache (GitHub tag was clean). Prefer the latest Zenodo version
DOI on the concept page.

## v1.5.4 (2026-08-11) — clean Zenodo rebuild
- Pack via `git archive` + hard excludes; fail-closed verification before upload.
- Extends `.gitignore` for process/IDE caches. No scientific log changes.

## License
Code: MIT (`LICENSE`). Result logs: CC BY 4.0. See `CITATION.cff`.

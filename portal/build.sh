#!/usr/bin/env bash
# Validate FIGURE-INDEX, Next.js static export, copy to _site/. No LaTeX.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
import json
from pathlib import Path
import jsonschema

root = Path(".")
schema = json.loads((root / "papers/FIGURE-INDEX.schema.json").read_text(encoding="utf-8"))
index = json.loads((root / "papers/FIGURE-INDEX.json").read_text(encoding="utf-8"))
jsonschema.validate(instance=index, schema=schema)
pdfs = sorted(p for p in (root / "papers").rglob("*.pdf") if p.is_file())
if pdfs:
    raise SystemExit(f"refuse: PDFs under papers/: {pdfs}")
print("INDEX valid; no papers/**/*.pdf in the working tree")
PY

tracked="$(git ls-files 'papers/**/*.pdf' || true)"
if [[ -n "${tracked}" ]]; then
  echo "F4: committed PDFs forbidden:" >&2
  printf '%s\n' "${tracked}" >&2
  exit 1
fi

mkdir -p portal/public/data
cp papers/FIGURE-INDEX.json portal/public/data/figures.json
python3 portal/scripts/build_science.py

if [[ ! -d portal/node_modules ]]; then
  (cd portal && npm ci)
else
  (cd portal && npm ci --prefer-offline --no-audit --no-fund)
fi
(cd portal && npm run build)

rm -rf _site
mkdir -p _site
cp -a portal/out/. _site/
mkdir -p _site/data
cp papers/FIGURE-INDEX.json _site/data/figures.json

# F9: the artifact INDEX keeps papers/-relative figs/ paths, so the figs data
# tiers it names must resolve under _site/data/figs/.
mkdir -p _site/data/figs
cp -a papers/figs/summaries _site/data/figs/summaries
if [[ -d papers/figs/previews ]]; then
  cp -a papers/figs/previews _site/data/figs/previews
fi

if [[ -e _site/experiments || -e _site/.omc ]]; then
  echo "I4: experiments or .omc leaked into _site" >&2
  exit 1
fi

python3 - <<'PY'
from pathlib import Path
root = Path("_site")
html_files = list(root.rglob("*.html"))
if not html_files:
    raise SystemExit("export proof failed: no HTML in _site/")
blob = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in html_files)
if "architecture-staircase" not in blob:
    raise SystemExit("export proof failed: basePath prefix missing from HTML")
# Patterns split so this scanner file is not a leak source.
needles = [
    "Attention must be " + "trained",
    "Uninformative for " + "this probe",
    "+" + "0.37",
    "+" + "0.141",
    "Figure" + "1.pdf",
    "Figure" + "16.pdf",
]
hits = [s for s in needles if s in blob]
if hits:
    raise SystemExit(f"leak in export HTML: {hits}")
chrome = [
    "doc" + "uments",
    "pap" + "ers",
    "jour" + "nals",
    "manu" + "scripts",
    "sub" + "missions",
    "Peer" + "J",
    "main" + ".tex",
    "FIGURE" + "-INDEX",
    "PIPE" + "LINE",
    "ware" + "house",
]
chrome_hits = [s for s in chrome if s.lower() in blob.lower()]
if chrome_hits:
    raise SystemExit(f"chrome in export HTML: {chrome_hits}")
print(f"export proof: {len(html_files)} html files; basePath present; leak scan clean")
PY

echo "built _site/ from Next.js export + FIGURE-INDEX"

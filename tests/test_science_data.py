"""Door science view: captions/questions only, no warehouse chrome."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from conftest import PORTAL_DIR, REPO_ROOT

SCIENCE_JSON = PORTAL_DIR / "public" / "science.json"
BUILD_SCIENCE = PORTAL_DIR / "scripts" / "build_science.py"

PATH_LEAK = re.compile(
    r"(papers/|figs/|experiments/|portal/|_site/|"
    r"\bC_[A-Za-z0-9_]+|"
    r"Figure(?:1[0-6]|[1-9])\.pdf|"
    r"main\.tex|FIGURE-INDEX|PIPELINE|"
    r"\.(tex|json|jsonl|R|md|ya?ml)\b)",
    re.I,
)
PUB_LEAK = re.compile(
    r"peerj|arxiv|manuscript|journal|warehouse|submission|preprint",
    re.I,
)
LETTER_CODE = re.compile(r"\bC_[A-Za-z0-9_]+|\bLab book C\b|\bPaper C\b")


def test_science_builder_emits_clean_view() -> None:
    subprocess.check_call(["python3", str(BUILD_SCIENCE)], cwd=REPO_ROOT)
    assert SCIENCE_JSON.is_file()
    data = json.loads(SCIENCE_JSON.read_text(encoding="utf-8"))
    blob = SCIENCE_JSON.read_text(encoding="utf-8")
    assert not PATH_LEAK.search(blob), blob[:400]
    assert not PUB_LEAK.search(blob)
    assert not LETTER_CODE.search(blob)
    assert len(data["probes"]["freeze"]) >= 2
    assert len(data["probes"]["ablation"]) >= 1
    assert len(data["staircase"]) >= 3
    assert len(data["scale"]) >= 3
    assert len(data["subspace"]) >= 1
    assert len(data["ledger"]) == 16
    assert len(data["rebuild"]) == 16
    freeze_q = sum(len(e["questions"]) for e in data["probes"]["freeze"])
    stair_q = sum(len(e["questions"]) for e in data["staircase"])
    assert freeze_q >= 4
    assert stair_q >= 4


def test_pages_render_science_not_filenames() -> None:
    for rel, needle in (
        ("app/page.tsx", "science.probes"),
        ("app/staircase/page.tsx", "science.staircase"),
        ("app/scale/page.tsx", "science.scale"),
        ("app/subspace/page.tsx", "science.subspace"),
        ("app/ledger/page.tsx", "science.ledger"),
        ("app/reproduce/page.tsx", "science.rebuild"),
    ):
        text = (PORTAL_DIR / rel).read_text(encoding="utf-8")
        assert needle in text
        assert "papers/" not in text
        assert "C_" not in text
        assert "generator" not in text


def test_readme_is_live_door_entrance() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    lines = [ln for ln in readme.splitlines() if ln.strip()]
    head = "\n".join(lines[:8])
    assert "https://peterponyu.github.io/architecture-staircase/" in head
    assert re.search(r"\bstars?\b", readme, re.I) is None
    for tok in (
        "PeerJ",
        "manuscript",
        "journal",
        "arXiv",
        "papers/C/",
        "FIGURE-INDEX",
        "main.tex",
    ):
        assert tok not in readme, tok

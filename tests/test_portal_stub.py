"""Portal stub contract (I1, U3–U6). Not a full lab-notebook UI."""

from __future__ import annotations

import re
from pathlib import Path

from conftest import (
    CONCEPT_DOI,
    GITHUB_URL,
    PORTAL_DIR,
    PORTAL_INDEX,
    REPO_ROOT,
    VERSION_DOI,
)

EMOJI = re.compile(r"[\U0001F300-\U0001FAFF]")
PEERJ_PDF = re.compile(r"Figure(?:1[0-6]|[1-9])\.pdf")
NAV_SUBSET = ("Probes", "Staircase", "Subspace")


def _portal_blob() -> str:
    assert PORTAL_DIR.is_dir(), f"portal/ stub missing at {PORTAL_DIR}"
    chunks: list[str] = []
    for path in sorted(PORTAL_DIR.rglob("*")):
        if path.suffix.lower() in {".html", ".css", ".js", ".sh"}:
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def test_portal_index_stub_exists() -> None:
    assert PORTAL_INDEX.is_file(), (
        "minimal portal stub required at portal/index.html (not a full lab-notebook UI)"
    )


def test_portal_uses_notebook_landmark_not_sibling_shells() -> None:
    html = PORTAL_INDEX.read_text(encoding="utf-8")
    lowered = html.lower()
    assert "notebook" in lowered or "lab-book" in lowered or 'class="lab-book"' in lowered
    assert "instrument-chrome" not in lowered
    assert "field-guide" not in lowered
    assert "three-pane" not in lowered
    assert 'class="atlas"' not in lowered
    assert 'class="console"' not in lowered


def test_portal_nav_subset_present() -> None:
    html = PORTAL_INDEX.read_text(encoding="utf-8")
    missing = [label for label in NAV_SUBSET if label not in html]
    assert not missing, f"P-C nav subset missing: {missing}"


def test_portal_footer_lists_doi_github_license() -> None:
    html = PORTAL_INDEX.read_text(encoding="utf-8")
    assert CONCEPT_DOI in html
    assert GITHUB_URL in html
    assert "MIT" in html
    assert "CC BY" in html or "CC-BY" in html
    assert "21882597" in html or VERSION_DOI in html


def test_portal_consumes_figure_index() -> None:
    blob = _portal_blob()
    assert (
        "FIGURE-INDEX.json" in blob
        or "data/figures.json" in blob
        or "figures.json" in blob
    )
    assert not PEERJ_PDF.search(blob), "U6: must not hardcode PeerJ FigureN.pdf paths"


def test_portal_has_no_emoji() -> None:
    assert not EMOJI.search(_portal_blob()), "U3: portal stub must not contain emoji"


def test_portal_has_no_journal_pdfs() -> None:
    forbidden = {"main.pdf", "manuscript.pdf"}
    found = [p for p in PORTAL_DIR.rglob("*") if p.name.lower() in forbidden]
    found += list(PORTAL_DIR.rglob("Figure*.pdf"))
    assert not found, f"U5: portal must not host journal PDFs: {found}"


def test_portal_assets_are_relative() -> None:
    html = PORTAL_INDEX.read_text(encoding="utf-8")
    assert 'href="/' not in html and 'src="/' not in html, (
        "project Pages lives under /architecture-staircase/; use relative asset URLs"
    )


def test_build_script_validates_then_copies() -> None:
    build = PORTAL_DIR / "build.sh"
    assert build.is_file(), "I1: portal/build.sh must exist"
    text = build.read_text(encoding="utf-8")
    assert "latexmk" not in text.lower()
    assert "_site" in text
    assert "FIGURE-INDEX" in text or "figures.json" in text
    assert "experiments" not in text.split("rsync")[-1] if "rsync" in text else True

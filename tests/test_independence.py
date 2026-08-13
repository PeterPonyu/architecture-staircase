"""Independence / preprint stance checks that apply to this warehouse."""

from __future__ import annotations

from conftest import PORTAL_DIR, REPO_ROOT


def test_no_suite_chrome_in_portal() -> None:
    blob = ""
    for path in PORTAL_DIR.rglob("*.tsx"):
        if "node_modules" in path.parts or ".next" in path.parts:
            continue
        blob += path.read_text(encoding="utf-8").lower()
    assert "five-paper" not in blob
    assert "muon-norm-cap-grokking" not in blob
    assert "grokking-clock" not in blob
    assert "free-repetition-band" not in blob
    assert "calibration-traps" not in blob


def test_no_compiled_manuscript_pdf_in_portal_or_site() -> None:
    roots = [PORTAL_DIR, REPO_ROOT / "_site"]
    found = []
    for root in roots:
        if not root.exists():
            continue
        for name in ("main.pdf", "manuscript.pdf"):
            found.extend(
                p
                for p in root.rglob(name)
                if "node_modules" not in p.parts and ".next" not in p.parts
            )
    assert not found


def test_readme_cites_github() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://github.com/PeterPonyu/architecture-staircase" in readme
    assert "papers/C/main.tex" in readme

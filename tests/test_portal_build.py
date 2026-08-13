"""I1/I4/U5: copy+validate portal build; no freeze PDFs; relative URLs only."""

from __future__ import annotations

import re
import subprocess

from conftest import PORTAL_DIR, REPO_ROOT

BUILD = PORTAL_DIR / "build.sh"
SITE = REPO_ROOT / "_site"


def test_portal_build_script_exists() -> None:
    assert BUILD.is_file()
    text = BUILD.read_text(encoding="utf-8")
    assert "FIGURE-INDEX" in text or "jsonschema" in text
    assert "latexmk" not in text.lower()


def test_build_script_copies_portal_without_experiments() -> None:
    proc = subprocess.run(
        ["bash", str(BUILD)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert SITE.is_dir()
    assert (SITE / "index.html").is_file()
    assert not (SITE / "experiments").exists()
    assert not (SITE / ".omc").exists()
    data = SITE / "data"
    assert (data / "FIGURE-INDEX.json").is_file() or (data / "figures.json").is_file()
    assert not list(SITE.rglob("main.pdf"))
    assert not list(SITE.rglob("manuscript.pdf"))
    assert not list(SITE.rglob("Figure*.pdf"))


def test_portal_assets_are_relative_not_user_site_absolute() -> None:
    abs_hits: list[str] = []
    for path in PORTAL_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".html", ".css", ".js"}:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"""(?:href|src)\s*=\s*["']/(?!/)""", text):
            abs_hits.append(str(path.relative_to(REPO_ROOT)))
        if re.search(r"url\(/", text):
            abs_hits.append(str(path.relative_to(REPO_ROOT)))
    assert not abs_hits, f"U7: root-absolute assets break project Pages: {abs_hits}"

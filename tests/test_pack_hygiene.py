"""Zenodo pack + site artifact hygiene (I4, I5, architect portal exclude)."""

from __future__ import annotations

from conftest import REPO_ROOT


def test_pack_script_excludes_github_portal_and_site() -> None:
    script = (REPO_ROOT / "pack_zenodo_tarball.sh").read_text(encoding="utf-8")
    assert ".github" in script
    assert "portal" in script, "next git-archive deposit must not ship portal/"
    assert "_site" in script, "next git-archive deposit must not ship _site/"


def test_gitignore_excludes_site_output() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "_site/" in gitignore


def test_site_artifact_would_not_include_experiments() -> None:
    build = REPO_ROOT / "portal" / "build.sh"
    assert build.is_file()
    text = build.read_text(encoding="utf-8")
    assert "experiments/" not in text
    assert "cp -a experiments" not in text
    assert "rsync" not in text or "experiments" not in text

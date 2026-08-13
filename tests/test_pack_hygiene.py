"""Zenodo pack + site artifact hygiene (I4, I5b)."""

from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

from conftest import REPO_ROOT


def test_pack_script_excludes_github_portal_and_site() -> None:
    script = (REPO_ROOT / "pack_zenodo_tarball.sh").read_text(encoding="utf-8")
    assert ".github" in script
    assert "portal" in script
    assert "_site" in script


def test_gitattributes_export_ignore_website_trees() -> None:
    attrs = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "portal/" in attrs and "export-ignore" in attrs
    assert "_site/" in attrs
    assert ".github/" in attrs


def test_gitignore_excludes_site_output() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "_site/" in gitignore


def test_site_artifact_would_not_include_experiments() -> None:
    text = (REPO_ROOT / "portal" / "build.sh").read_text(encoding="utf-8")
    assert "cp -a experiments" not in text
    assert "latexmk" not in text.lower()


def test_workdir_pack_omits_portal_and_site(tmp_path: Path) -> None:
    out = tmp_path / "architecture-staircase-workdir.tar.gz"
    subprocess.run(
        [str(REPO_ROOT / "pack_zenodo_tarball.sh"), "--from-workdir", "--out", str(out)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert out.is_file()
    members = tarfile.open(out).getnames()
    leaked = [
        name
        for name in members
        if "/portal/" in f"/{name}/"
        or name.endswith("/portal")
        or "/_site/" in f"/{name}/"
        or "/.github/" in f"/{name}/"
    ]
    assert not leaked, f"I5b: website trees leaked into workdir pack: {leaked[:20]}"

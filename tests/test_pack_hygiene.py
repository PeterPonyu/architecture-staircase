"""Zenodo pack + site artifact hygiene (I4, I5b, architect portal exclude)."""

from __future__ import annotations

import io
import subprocess
import tarfile

from conftest import REPO_ROOT


def test_pack_script_excludes_github_portal_and_site() -> None:
    script = (REPO_ROOT / "pack_zenodo_tarball.sh").read_text(encoding="utf-8")
    assert ".github" in script
    assert "portal" in script, "next git-archive deposit must not ship portal/"
    assert "_site" in script, "next git-archive deposit must not ship _site/"


def test_gitattributes_export_ignores_website_trees() -> None:
    text = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "portal/ export-ignore" in text
    assert "_site/ export-ignore" in text
    assert ".github/ export-ignore" in text


def test_git_archive_omits_portal_github_and_site() -> None:
    proc = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    with tarfile.open(fileobj=io.BytesIO(proc.stdout), mode="r:") as archive:
        names = archive.getnames()
    leaked = [
        name
        for name in names
        if any(part in {"portal", "_site", ".github"} for part in name.split("/"))
    ]
    assert not leaked, f"I5b: git archive must omit website/Actions trees: {leaked[:20]}"


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

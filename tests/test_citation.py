"""Citation / license contract (test-spec C1–C4, G4) for architecture-staircase."""

from __future__ import annotations

import json

import yaml

from conftest import CONCEPT_DOI, REPO_ROOT, VERSION_DOI


def test_citation_cff_parses() -> None:
    raw = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    assert isinstance(data, dict), "C1: CITATION.cff must parse to a mapping"
    assert isinstance(data.get("title"), str) and data["title"].strip(), (
        "C2: CFF title must be a single YAML string"
    )
    assert "\n" not in data["title"], "C2: CFF title must be one string, not concatenated titles"


def test_citation_cff_uses_concept_doi() -> None:
    data = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert data["doi"] == CONCEPT_DOI, f"C3: doi: must be concept {CONCEPT_DOI}"
    identifiers = data.get("identifiers") or []
    values = {item.get("value") for item in identifiers if isinstance(item, dict)}
    assert VERSION_DOI in values, (
        f"C3: version DOI {VERSION_DOI} must live under identifiers, not doi:"
    )


def test_citation_cff_is_not_five_paper_bundle() -> None:
    data = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    blob = json_blob(data)
    assert "muon-norm-cap" not in blob
    assert "free-repetition-band" not in blob
    assert "calibration-traps" not in blob
    assert "grokking-clock" not in blob


def json_blob(data: object) -> str:
    return json.dumps(data).lower()


def test_dual_license_notice_preserved() -> None:
    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "Creative Commons Attribution 4.0" in license_text or "CC BY 4.0" in license_text


def test_readme_cites_github_warehouse_url() -> None:
    from conftest import GITHUB_URL

    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert GITHUB_URL in text, "README must cite https://github.com/PeterPonyu/architecture-staircase"

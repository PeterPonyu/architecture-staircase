"""Citation / license contract (C1–C4, G4)."""

from __future__ import annotations

import json

import yaml

from conftest import CONCEPT_DOI, REPO_ROOT, VERSION_DOI


def test_citation_cff_parses() -> None:
    raw = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    assert isinstance(data, dict)
    assert isinstance(data.get("title"), str) and data["title"].strip()
    assert "\n" not in data["title"]


def test_citation_cff_uses_concept_doi() -> None:
    data = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert data["doi"] == CONCEPT_DOI
    identifiers = data.get("identifiers") or []
    values = {item.get("value") for item in identifiers if isinstance(item, dict)}
    assert VERSION_DOI in values


def test_citation_cff_is_not_five_paper_bundle() -> None:
    data = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    blob = json.dumps(data).lower()
    assert "muon-norm-cap" not in blob
    assert "free-repetition-band" not in blob
    assert "calibration-traps" not in blob
    assert "grokking-clock" not in blob


def test_dual_license_notice_preserved() -> None:
    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "Creative Commons Attribution 4.0" in license_text or "CC BY 4.0" in license_text

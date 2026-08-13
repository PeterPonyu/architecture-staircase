"""Warehouse-root fixtures for Paper C contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = REPO_ROOT / "papers" / "FIGURE-INDEX.json"
SCHEMA_PATH = REPO_ROOT / "papers" / "figure-index.schema.json"
PORTAL_DIR = REPO_ROOT / "portal"
PORTAL_INDEX = PORTAL_DIR / "index.html"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

PEERJ_FIGURE_MAP = {
    "Figure1.pdf": "C_landscape",
    "Figure2.pdf": "C_ladder_necessity",
    "Figure3.pdf": "C_targetfamily",
    "Figure4.pdf": "C_arch_depth",
    "Figure5.pdf": "C_ownership_optaxis",
    "Figure6.pdf": "C_ownership_lr_sensitivity",
    "Figure7.pdf": "C_scheme",
    "Figure8.pdf": "C_staircase_si",
    "Figure9.pdf": "C_staircase_rho",
    "Figure10.pdf": "C_pure3_speedup",
    "Figure11.pdf": "C_ownership_localize",
    "Figure12.pdf": "C_case",
    "Figure13.pdf": "C_timing",
    "Figure14.pdf": "C_scale",
    "Figure15.pdf": "C_dissoc",
    "Figure16.pdf": "C_subspace",
}

CONCEPT_DOI = "10.5281/zenodo.21020348"
VERSION_DOI = "10.5281/zenodo.21882597"
GITHUB_REPO = "PeterPonyu/architecture-staircase"
GITHUB_URL = f"https://github.com/{GITHUB_REPO}"


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def index_path() -> Path:
    return INDEX_PATH


@pytest.fixture
def schema_path() -> Path:
    return SCHEMA_PATH


@pytest.fixture
def portal_dir() -> Path:
    return PORTAL_DIR


@pytest.fixture
def portal_index() -> Path:
    return PORTAL_INDEX


def load_yaml(path: Path) -> object:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def workflow_path(name: str) -> Path:
    return WORKFLOWS_DIR / name

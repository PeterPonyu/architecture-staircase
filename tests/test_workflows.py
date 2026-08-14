"""GitHub Actions contract: CI + Pages on main (I2, I3)."""

from __future__ import annotations

from pathlib import Path

import yaml

from conftest import REPO_ROOT, WORKFLOWS_DIR


def _load(name: str) -> dict:
    path = WORKFLOWS_DIR / name
    assert path.is_file(), f"missing .github/workflows/{name}"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_required_ci_workflow_exists() -> None:
    data = _load("ci.yml")
    jobs = data.get("jobs") or {}
    assert "contract" in jobs
    blob = yaml.safe_dump(data)
    assert "pytest" in blob
    assert "latexmk" not in blob.lower()
    assert "ci/comprehensive" not in blob


def test_pages_workflow_permissions_and_environment() -> None:
    data = _load("pages.yml")
    perms = data.get("permissions") or {}
    assert perms.get("pages") == "write"
    env_names = []
    for job in (data.get("jobs") or {}).values():
        env = job.get("environment")
        if isinstance(env, str):
            env_names.append(env)
        elif isinstance(env, dict):
            env_names.append(env.get("name"))
    assert "github-pages" in env_names


def test_pages_triggers_are_main_only_with_workflow_dispatch() -> None:
    data = _load("pages.yml")
    on = data.get("on") or data.get(True)
    assert isinstance(on, dict)
    assert "workflow_dispatch" in on
    push = on.get("push") or {}
    branches = push.get("branches") or []
    assert branches == ["main"], f"I3: branches must be [main], got {branches}"
    paths = push.get("paths") or []
    joined = " ".join(paths)
    assert "portal/**" in joined
    assert "papers/FIGURE-INDEX.json" in joined
    assert "papers/figs/summaries/**" in joined
    assert "papers/figs/previews/**" in joined
    assert ".github/workflows/pages.yml" in joined
    assert "ci/comprehensive" not in yaml.safe_dump(data)


def test_no_visual_pixel_workflow() -> None:
    extras = list(WORKFLOWS_DIR.glob("visual-*.yml"))
    assert not extras, f"G7: extra visual workflows are out of PRD: {extras}"


def test_no_workflow_runs_latexmk() -> None:
    workflows = list(WORKFLOWS_DIR.glob("*.yml"))
    assert workflows
    for path in workflows:
        text = path.read_text(encoding="utf-8").lower()
        assert "latexmk" not in text
        assert "pdflatex" not in text
        assert "lualatex" not in text

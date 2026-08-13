"""GitHub Actions contract: required CI green; Pages/visual-pixel gated."""

from __future__ import annotations

from pathlib import Path

import yaml

from conftest import REPO_ROOT, workflow_path

CI_WORKFLOW = "ci.yml"
PAGES_WORKFLOW = "pages.yml"
VISUAL_WORKFLOW = "visual-pixel.yml"


def _load_workflow(name: str) -> dict:
    path = workflow_path(name)
    assert path.is_file(), f"I2: missing .github/workflows/{name}"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{name} must be a mapping"
    return data


def _job_if(job: dict) -> object:
    return job.get("if")


def _is_gated_off(value: object) -> bool:
    if value is False:
        return True
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"false", "${{ false }}"}:
            return True
        if "visual_reference_approved" in lowered and "== 'true'" in lowered:
            return True
        if "vars.pages_deploy_enabled" in lowered:
            return True
    return False


def test_required_ci_workflow_exists() -> None:
    data = _load_workflow(CI_WORKFLOW)
    jobs = data.get("jobs") or {}
    assert "contract" in jobs, "required CI must define a contract job"


def test_ci_runs_comprehensive_pytest() -> None:
    data = _load_workflow(CI_WORKFLOW)
    contract = data["jobs"]["contract"]
    steps = contract.get("steps") or []
    blob = "\n".join(str(step) for step in steps)
    assert "pytest" in blob, "contract job must run pytest on tests/"
    assert "latexmk" not in blob.lower()


def test_ci_does_not_require_visual_pixel_job() -> None:
    data = _load_workflow(CI_WORKFLOW)
    jobs = data["jobs"]
    assert "visual-pixel" not in jobs
    blob = yaml.safe_dump(data)
    assert "visual-pixel.yml" not in blob
    for job in jobs.values():
        needs = job.get("needs") or []
        if isinstance(needs, str):
            needs = [needs]
        assert "visual-pixel" not in needs


def test_pages_workflow_declares_permissions_and_environment() -> None:
    data = _load_workflow(PAGES_WORKFLOW)
    perms = data.get("permissions") or {}
    assert perms.get("pages") == "write", "I2: pages.yml needs permissions.pages: write"
    jobs = data.get("jobs") or {}
    env_names = []
    for job in jobs.values():
        env = job.get("environment")
        if isinstance(env, str):
            env_names.append(env)
        elif isinstance(env, dict):
            env_names.append(env.get("name"))
    assert "github-pages" in env_names, "I2: pages.yml must declare environment github-pages"


def test_pages_push_paths_are_portal_and_index_only() -> None:
    data = _load_workflow(PAGES_WORKFLOW)
    on = data.get("on") or data.get(True)
    assert isinstance(on, dict), "I3: pages.yml must use a mapping on: trigger"
    assert "workflow_dispatch" in on, "I3: first enable must not depend on a portal-only push"
    push = on.get("push") or {}
    paths = push.get("paths") or []
    assert "portal/**" in paths, "I3: path filters must include portal/**"
    assert "papers/FIGURE-INDEX.json" in paths
    assert "papers/figs/summaries/**" in paths
    assert "papers/figs/previews/**" in paths, (
        "I3: previews live under papers/figs/previews/, not papers/previews/"
    )
    assert ".github/workflows/pages.yml" in paths, (
        "I3: path filters must include the workflow file itself"
    )
    assert "experiments/**" not in paths
    assert not any(p.startswith("experiments") for p in paths)


def test_pages_deploy_job_is_gated_off() -> None:
    data = _load_workflow(PAGES_WORKFLOW)
    jobs = data["jobs"]
    assert "deploy" in jobs, "pages.yml must keep a deploy job so the sketch is complete"
    assert _is_gated_off(_job_if(jobs["deploy"])), (
        "Pages must not deploy until explicitly enabled; deploy.if must be false / gated"
    )
    blob = yaml.safe_dump(data)
    assert "latexmk" not in blob.lower()


def test_pages_does_not_enable_pages_via_api() -> None:
    text = workflow_path(PAGES_WORKFLOW).read_text(encoding="utf-8")
    assert "repos/PeterPonyu/architecture-staircase/pages" not in text
    assert "gh api" not in text.lower() or "pages" not in text.split("gh api")[-1][:80]


def test_visual_pixel_workflow_is_optional_and_gated() -> None:
    data = _load_workflow(VISUAL_WORKFLOW)
    jobs = data.get("jobs") or {}
    assert jobs, "visual-pixel.yml must exist as an optional workflow"
    for name, job in jobs.items():
        raw_if = str(job.get("if", "")).lower().replace(" ", "")
        assert raw_if in {"false", "${{false}}"}, (
            f"visual-pixel job {name!r} must be gated with if: false until "
            "reference.png is user-approved (must not fail required CI)"
        )
    on = data.get("on") or data.get(True)
    assert "workflow_dispatch" in on or (
        isinstance(on, dict) and "workflow_dispatch" in on
    )


def test_no_workflow_runs_latexmk() -> None:
    workflows = list((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
    assert workflows, "expected workflow files"
    for path in workflows:
        text = path.read_text(encoding="utf-8").lower()
        assert "latexmk" not in text, f"{path.name} must not compile LaTeX"
        assert "pdflatex" not in text
        assert "lualatex" not in text

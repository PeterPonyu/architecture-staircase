"""Figure-pointer contract (test-spec F1–F8) for architecture-staircase."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import jsonschema

from conftest import (
    INDEX_PATH,
    PEERJ_FIGURE_MAP,
    REPO_ROOT,
    SCHEMA_PATH,
)

SCHEMATIC_IDS = {"C_landscape", "C_scheme"}
PEERJ_INCLUDE = re.compile(r"\\includegraphics\{[^}]*Figure(?:1[0-6]|[1-9])\.pdf")


def _index() -> dict:
    assert INDEX_PATH.is_file(), f"F1: missing {INDEX_PATH}"
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def test_figure_index_exists() -> None:
    assert INDEX_PATH.is_file(), (
        "F1: papers/FIGURE-INDEX.json is the portal contract and must exist"
    )


def test_figure_index_schema_file_exists() -> None:
    assert SCHEMA_PATH.is_file(), f"F1: schema missing at {SCHEMA_PATH}"


def test_figure_index_validates_against_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=_index(), schema=schema)


def test_index_identifies_paper_c_warehouse() -> None:
    data = _index()
    assert data["paper_id"] == "C"
    assert data["github"] == "PeterPonyu/architecture-staircase"
    assert data["zenodo_concept_doi"] == "10.5281/zenodo.21020348"


def test_index_ids_are_warehouse_or_documented_schematics() -> None:
    data = _index()
    figs_dir = REPO_ROOT / "papers" / "figs"
    for fig in data["figures"]:
        fig_id = fig["id"]
        generator = fig.get("generator") or ""
        if fig_id in SCHEMATIC_IDS:
            assert generator, f"F2: schematic {fig_id} needs a generator pointer"
            continue
        gen_path = REPO_ROOT / generator if generator and not generator.startswith("schematic:") else None
        summary = fig.get("summary")
        if gen_path is not None:
            assert gen_path.is_file(), f"F2: generator missing for {fig_id}: {gen_path}"
        elif summary:
            assert (REPO_ROOT / summary).is_file(), f"F2: unknown id {fig_id}"
        else:
            raise AssertionError(
                f"F2: {fig_id} is not a documented schematic and has no generator under {figs_dir}"
            )


def test_index_includes_peerj_venue_names_as_metadata_only() -> None:
    data = _index()
    by_id = {fig["id"]: fig for fig in data["figures"]}
    for venue_name, warehouse_id in PEERJ_FIGURE_MAP.items():
        assert warehouse_id in by_id, f"F3: INDEX missing warehouse id {warehouse_id} ({venue_name})"
        assert by_id[warehouse_id].get("venue_flat_name") == venue_name, (
            f"F3: {warehouse_id} must record venue_flat_name={venue_name} as metadata only"
        )


def test_no_peerj_figure_pdfs_are_committed() -> None:
    proc = subprocess.run(
        ["git", "ls-files", "papers/**/*.pdf", "paper/**/*.pdf", "*.pdf"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [line for line in proc.stdout.splitlines() if line.strip()]
    assert not tracked, f"F4: committed PDFs are forbidden on the warehouse: {tracked}"
    stray = list((REPO_ROOT / "papers").rglob("Figure*.pdf"))
    assert not stray, f"F3/F4: must not commit PeerJ FigureN.pdf: {stray}"


def test_existing_tex_does_not_include_previews() -> None:
    hits: list[str] = []
    for path in (REPO_ROOT / "papers").rglob("*.tex"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "previews/" in text:
            hits.append(str(path.relative_to(REPO_ROOT)))
    assert not hits, f"F5: tex must not include previews/: {hits}"


def test_existing_tex_does_not_include_peerj_figure_pdfs() -> None:
    hits: list[str] = []
    for path in (REPO_ROOT / "papers").rglob("*.tex"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if PEERJ_INCLUDE.search(line) or (
                "includegraphics" in line and "Figure" in line and ".pdf" in line
            ):
                hits.append(f"{path.name}: {line.strip()}")
    assert not hits, f"F6: tex must not include PeerJ FigureN.pdf: {hits}"


def test_summaries_exist_when_index_declares_them() -> None:
    data = _index()
    missing: list[str] = []
    for fig in data["figures"]:
        summary = fig.get("summary")
        if not summary:
            continue
        path = REPO_ROOT / summary
        if not path.is_file():
            missing.append(f"{fig['id']}: {summary}")
    assert not missing, f"F7: declared summaries missing: {missing}"


def test_gitignore_excludes_compiled_figure_tiers() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "papers/figs/tex/" in gitignore or "figs/tex/" in gitignore, (
        "F8: .gitignore must exclude compiled papers/figs/tex/"
    )
    assert "papers/figs/vec/" in gitignore or "figs/vec/" in gitignore, (
        "F8: .gitignore must exclude compiled papers/figs/vec/"
    )


def test_v154_generator_paths_still_exist() -> None:
    figs = REPO_ROOT / "papers" / "figs"
    required = [
        figs / "fig_pipeline.R",
        figs / "figpreamble.tex",
        figs / "make_C_figs_r.R",
        figs / "make_C_new_figs_r.R",
        figs / "make_gap20260705_C_figs_r.R",
        figs / "make_landscape_r.R",
    ]
    missing = [str(p.relative_to(REPO_ROOT)) for p in required if not p.is_file()]
    assert not missing, f"I5: v1.5.4 generator paths missing: {missing}"

"""Figure-pointer contract (test-spec F1–F9, F6b) for architecture-staircase."""

from __future__ import annotations

import json
import re
import subprocess

import jsonschema

from conftest import (
    INDEX_PATH,
    PEERJ_FIGURE_MAP,
    POINTER_TEX,
    REPO_ROOT,
    SCHEMA_PATH,
    YAML_MISSING_BODY,
)

SCHEMATIC_IDS = {"C_landscape", "C_scheme"}
PEERJ_INCLUDE = re.compile(r"\\includegraphics\{[^}]*Figure(?:1[0-6]|[1-9])\.pdf")
PATH_FIELDS = ("generator", "summary", "preview_svg", "tex_build", "vec_build")


def _index() -> dict:
    assert INDEX_PATH.is_file(), f"F1: missing {INDEX_PATH}"
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def test_figure_index_exists() -> None:
    assert INDEX_PATH.is_file(), "F1: papers/FIGURE-INDEX.json must exist"


def test_figure_index_schema_file_exists() -> None:
    assert SCHEMA_PATH.is_file(), f"F1: schema missing at {SCHEMA_PATH}"
    assert SCHEMA_PATH.name == "FIGURE-INDEX.schema.json"


def test_schema_paper_id_is_enum_not_const() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    paper_id = schema["properties"]["paper_id"]
    assert paper_id.get("const") is None, "F1: shared schema must not const paper_id"
    assert set(paper_id.get("enum") or []) == {"A", "B", "C", "E1", "E2"}


def test_figure_index_validates_against_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=_index(), schema=schema)


def test_index_identifies_paper_c_warehouse() -> None:
    data = _index()
    assert data["paper_id"] == "C"
    assert data["github"] == "PeterPonyu/architecture-staircase"
    assert data["zenodo_concept_doi"] == "10.5281/zenodo.21020348"
    assert str(data["pipeline"]).startswith("figs/")


def test_index_ids_match_yaml_schematic_or_map_exceptions() -> None:
    data = _index()
    for fig in data["figures"]:
        fig_id = fig["id"]
        generator = fig.get("generator") or ""
        assert generator.startswith("figs/"), f"F9: generator must start with figs/: {fig_id}"
        if fig_id in SCHEMATIC_IDS:
            assert generator, f"F2: schematic {fig_id} needs a generator pointer"
            continue
        if fig_id in YAML_MISSING_BODY:
            assert generator == "figs/make_C_figs_r.R", (
                f"F2(c): {fig_id} must use figs/make_C_figs_r.R"
            )
            continue
        gen_path = REPO_ROOT / "papers" / generator
        assert gen_path.is_file(), f"F2: generator missing for {fig_id}: {gen_path}"


def test_yaml_missing_body_figures_have_null_summary() -> None:
    by_id = {fig["id"]: fig for fig in _index()["figures"]}
    for fig_id in YAML_MISSING_BODY:
        assert fig_id in by_id, f"F2(c): INDEX must admit {fig_id}"
        assert by_id[fig_id].get("summary") is None, f"F2(c): {fig_id} summary must be null"


def test_c_scheme_tex_build_is_figs_path_not_figtikz_token() -> None:
    by_id = {fig["id"]: fig for fig in _index()["figures"]}
    scheme = by_id["C_scheme"]
    assert scheme["tex_build"] == "figs/tex/C_scheme.tex"
    blob = json.dumps(scheme)
    assert r"\figtikz" not in blob


def test_index_includes_all_sixteen_peerj_map_names() -> None:
    by_id = {fig["id"]: fig for fig in _index()["figures"]}
    assert len(PEERJ_FIGURE_MAP) == 16
    for venue_name, warehouse_id in PEERJ_FIGURE_MAP.items():
        assert warehouse_id in by_id, f"F3: INDEX missing {warehouse_id} ({venue_name})"
        assert by_id[warehouse_id].get("venue_flat_name") == venue_name


def test_no_peerj_figure_pdfs_are_committed() -> None:
    proc = subprocess.run(
        ["git", "ls-files", "papers/**/*.pdf"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [line for line in proc.stdout.splitlines() if line.strip()]
    assert not tracked, f"F4: committed PDFs forbidden: {tracked}"


def test_pointer_tex_does_not_include_previews() -> None:
    text = POINTER_TEX.read_text(encoding="utf-8")
    assert "previews/" not in text


def test_pointer_tex_does_not_include_peerj_figure_pdfs() -> None:
    text = POINTER_TEX.read_text(encoding="utf-8")
    hits = [line.strip() for line in text.splitlines() if PEERJ_INCLUDE.search(line)]
    assert not hits, f"F6: PeerJ FigureN.pdf includes: {hits}"


def test_canonical_includes_survive() -> None:
    text = POINTER_TEX.read_text(encoding="utf-8")
    assert r"\input{../figs/figpreamble.tex}" in text
    assert r"\graphicspath{{./}}" not in text
    assert r"\includegraphics[width=\linewidth]{C_arch_depth.pdf}" in text


def test_pointer_tex_is_full_manuscript_not_stub() -> None:
    assert POINTER_TEX.is_file(), "F6b: papers/C/main.tex must exist"
    lines = POINTER_TEX.read_text(encoding="utf-8").splitlines()
    assert 2133 <= len(lines) <= 2357, (
        f"F6b: expected ~2245 lines after scar strip, got {len(lines)}"
    )
    blob = POINTER_TEX.read_text(encoding="utf-8")
    assert "Two-probe contract" not in blob
    assert "pointer-only GitHub SSOT" not in blob


def test_summaries_exist_when_index_declares_them() -> None:
    missing: list[str] = []
    for fig in _index()["figures"]:
        summary = fig.get("summary")
        if not summary:
            continue
        path = REPO_ROOT / "papers" / summary
        if not path.is_file():
            missing.append(f"{fig['id']}: {summary}")
    assert not missing, f"F7: declared summaries missing: {missing}"


def test_gitignore_excludes_compiled_figure_tiers() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "papers/figs/tex/" in gitignore or "figs/tex/" in gitignore
    assert "papers/figs/vec/" in gitignore or "figs/vec/" in gitignore


def test_index_path_grammar_is_papers_relative() -> None:
    mixed: list[str] = []
    for fig in _index()["figures"]:
        for field in PATH_FIELDS:
            value = fig.get(field)
            if value is None:
                continue
            if not str(value).startswith("figs/"):
                mixed.append(f"{fig['id']}.{field}={value}")
    assert not mixed, f"F9: mixed or non-figs/ paths: {mixed}"


def test_v154_generator_paths_still_exist() -> None:
    figs = REPO_ROOT / "papers" / "figs"
    required = [
        figs / "fig_pipeline.R",
        figs / "figpreamble.tex",
        figs / "make_C_figs_r.R",
        figs / "make_C_new_figs_r.R",
        figs / "make_gap20260705_C_figs_r.R",
        figs / "make_landscape_r.R",
        figs / "PIPELINE.md",
    ]
    missing = [str(p.relative_to(REPO_ROOT)) for p in required if not p.is_file()]
    assert not missing, f"I5: v1.5.4 generator paths missing: {missing}"

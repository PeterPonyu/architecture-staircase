"""F5–F6 / I5: warehouse pointer tex at papers/C/main.tex (not a journal-flat export)."""

from __future__ import annotations

import re

from conftest import REPO_ROOT

MAIN_TEX = REPO_ROOT / "papers" / "C" / "main.tex"
PREAMBLE = REPO_ROOT / "papers" / "figs" / "figpreamble.tex"
PEERJ_PDF = re.compile(r"Figure(?:1[0-6]|[1-9])\.pdf")
VENUE_FLAT_PDF = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*C_[^}]*\.pdf")


def test_pointer_main_tex_exists() -> None:
    assert MAIN_TEX.is_file(), (
        "I5: papers/C/main.tex must exist so \\input{../figs/figpreamble.tex} stays byte-stable"
    )


def test_main_tex_inputs_unmodified_figpreamble() -> None:
    text = MAIN_TEX.read_text(encoding="utf-8")
    assert r"\input{../figs/figpreamble.tex}" in text, (
        "F6: warehouse pointer must keep \\input{../figs/figpreamble.tex}"
    )


def test_main_tex_does_not_use_venue_flatten_graphicspath() -> None:
    text = MAIN_TEX.read_text(encoding="utf-8")
    assert r"\graphicspath{{./}}" not in text, (
        "F6: \\graphicspath{{./}} is the venue flattener; keep the figpreamble dialect"
    )


def test_figpreamble_defines_figtikz_and_vec_graphicspath() -> None:
    text = PREAMBLE.read_text(encoding="utf-8")
    assert r"\graphicspath{{../figs/vec/}{../figs/}}" in text
    assert r"\newcommand{\figtikz}" in text
    assert r"\input{../figs/tex/#2.tex}" in text


def test_main_tex_uses_figtikz_pointer_includes() -> None:
    text = MAIN_TEX.read_text(encoding="utf-8")
    assert r"\figtikz" in text


def test_main_tex_does_not_include_previews() -> None:
    text = MAIN_TEX.read_text(encoding="utf-8")
    assert "previews/" not in text


def test_main_tex_does_not_include_peerj_or_committed_pdfs() -> None:
    text = MAIN_TEX.read_text(encoding="utf-8")
    assert PEERJ_PDF.search(text) is None, "F6: must not include PeerJ FigureN.pdf"
    assert VENUE_FLAT_PDF.search(text) is None, (
        "F6: must not \\includegraphics committed C_*.pdf beside the tex"
    )
    assert "Figure1.pdf" not in text


def test_papers_tree_is_not_renamed_to_paper() -> None:
    assert not (REPO_ROOT / "paper").exists(), (
        "I5: C keeps papers/figs/ at v1.5.4 paths; do not rename to paper/"
    )
    assert (REPO_ROOT / "papers" / "figs" / "figpreamble.tex").is_file()

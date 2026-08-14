"""Portal must not dump paper numbers, captions, or PeerJ PDF text."""

from __future__ import annotations

import re
from pathlib import Path

from conftest import PORTAL_DIR

SKIP_PARTS = {"node_modules", ".next", "out", "public"}
SOURCE_SUFFIXES = {".tsx", ".ts", ".css", ".js", ".mjs", ".md"}

LEAK_PATTERNS = (
    re.compile(r"Attention must be trained"),
    re.compile(r"Uninformative for this probe"),
    re.compile(r"fit-gated"),
    re.compile(r"8/8 seeds"),
    re.compile(r"\+0\.37"),
    re.compile(r"\+0\.141"),
    re.compile(r"\+0\.090"),
    re.compile(r"Walsh-correlation"),
    re.compile(r"Figure(?:1[0-6]|[1-9])\.pdf"),
    re.compile(r"TOST equivalence"),
    re.compile(r"Holm p"),
    re.compile(r"Δ=1\.5"),
    re.compile(r"±2\.0"),
    re.compile(r"AdamW"),
    re.compile(r"\bMuon\b"),
    re.compile(r"\bSGDM\b"),
    re.compile(r"d=128"),
    re.compile(r"n=8"),
)


def _portal_source() -> str:
    chunks: list[str] = []
    for path in sorted(PORTAL_DIR.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.suffix.lower() in SOURCE_SUFFIXES or path.name == "build.sh":
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def test_portal_source_has_no_result_leak() -> None:
    blob = _portal_source()
    hits = [pat.pattern for pat in LEAK_PATTERNS if pat.search(blob)]
    assert not hits, f"paper/post leak in portal source: {hits}"


def test_ledger_does_not_render_captions_or_venue_pdf_names() -> None:
    ledger = (PORTAL_DIR / "app" / "ledger" / "page.tsx").read_text(encoding="utf-8")
    assert "caption" not in ledger.lower()
    assert "venue_flat_name" not in ledger
    assert "Figure" not in ledger

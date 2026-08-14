"""P-C Next.js lab-notebook portal (layout, fonts, nav, anti-stub)."""

from __future__ import annotations

import re
from pathlib import Path

from conftest import CONCEPT_DOI, GITHUB_URL, PORTAL_DIR, VERSION_DOI

REQUIRED_NAV = (
    "Probes",
    "Staircase",
    "Scale",
    "Subspace",
    "Ledger",
    "Rebuild",
)
SOURCE_SUFFIXES = {".tsx", ".ts", ".css", ".js", ".mjs", ".json"}
SKIP_PARTS = {"node_modules", ".next", "out", "public"}

LITERATA = re.compile(r"literata", re.I)
STIX_TWO = re.compile(r"stix[_\s-]*two", re.I)
FORBIDDEN_TYPE = (
    re.compile(r"ibm\s*plex", re.I),
    re.compile(r"source\s*serif\s*4", re.I),
    re.compile(r"source\s*sans\s*3", re.I),
    re.compile(r"fraunces", re.I),
    re.compile(r"atkinson\s*hyperlegible", re.I),
    re.compile(r"jetbrains\s*mono", re.I),
    re.compile(r"newsreader", re.I),
)
FORBIDDEN_LAYOUT = (
    re.compile(r"instrument[-_\s]?chrome|\.instrument\b|status[-_\s]?strip", re.I),
    re.compile(r"\batlas\b|regime[-_\s]?map|isobar", re.I),
    re.compile(r"field[-_\s]?guide|epoch[-_\s]?stratum|stratum[-_\s]?hero", re.I),
    re.compile(r"\bconsole\b|nomogram|three[-_\s]?pane", re.I),
)
STUB = re.compile(
    r"CI stub|instrument stub|Two-probe contract stub|"
    r"waits on a user-approved reference\.png|\bstub\b",
    re.I,
)
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF]")
PEERJ_FIGURE_PDF = re.compile(r"Figure(?:1[0-6]|[1-9])\.pdf")


def _source_files() -> list[Path]:
    files: list[Path] = []
    for path in PORTAL_DIR.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.suffix.lower() in SOURCE_SUFFIXES or path.name in {
            "next.config.ts",
            "next.config.js",
            "next.config.mjs",
        }:
            files.append(path)
    return files


def _blob() -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in _source_files())


def test_next_app_page_exists() -> None:
    assert (PORTAL_DIR / "app" / "page.tsx").is_file()
    assert (PORTAL_DIR / "next.config.ts").is_file()


def test_next_static_export_and_base_path() -> None:
    text = (PORTAL_DIR / "next.config.ts").read_text(encoding="utf-8")
    assert 'output: "export"' in text or "output: 'export'" in text
    assert "architecture-staircase" in text
    assert "basePath" in text
    assert "export" in text


def test_layout_is_split_gutter_notebook() -> None:
    probes = (PORTAL_DIR / "app" / "page.tsx").read_text(encoding="utf-8").lower()
    chrome = (PORTAL_DIR / "components" / "Notebook.tsx").read_text(encoding="utf-8").lower()
    blob = probes + chrome + (PORTAL_DIR / "app" / "globals.css").read_text(encoding="utf-8").lower()
    assert "notebook" in blob
    assert "freeze" in probes and "training necessity" in probes
    assert "ablation" in probes and "trained computation" in probes
    assert "gutter" in probes or "spine" in blob


def test_type_is_literata_and_stix_two() -> None:
    blob = _blob()
    assert LITERATA.search(blob)
    assert STIX_TWO.search(blob)


def test_nav_full_spine() -> None:
    spine = (PORTAL_DIR / "components" / "Spine.tsx").read_text(encoding="utf-8")
    missing = [label for label in REQUIRED_NAV if label not in spine]
    assert not missing, f"spine nav missing {missing}"


def test_forbids_sibling_portal_skins() -> None:
    blob = _blob()
    hits = [pat.pattern for pat in FORBIDDEN_TYPE if pat.search(blob)]
    hits += [pat.pattern for pat in FORBIDDEN_LAYOUT if pat.search(blob)]
    assert not hits, f"P-C must not share sibling skins: {hits}"


def test_anti_stub() -> None:
    assert not STUB.search(_blob())


def test_footer_has_doi_github_license() -> None:
    html = (PORTAL_DIR / "components" / "Notebook.tsx").read_text(encoding="utf-8")
    assert CONCEPT_DOI in html
    assert GITHUB_URL in html
    assert "MIT" in html
    assert "CC BY" in html or "CC-BY" in html
    assert "21882597" in html or VERSION_DOI in html


DOOR_CHROME = re.compile(
    r"documents?|papers?|journals?|manuscripts?|submissions?|"
    r"peerj|main\.tex|figure-index|pipeline|warehouse",
    re.I,
)


def _door_copy() -> str:
    chunks: list[str] = []
    for rel in (
        "app/layout.tsx",
        "app/page.tsx",
        "app/staircase/page.tsx",
        "app/scale/page.tsx",
        "app/subspace/page.tsx",
        "app/ledger/page.tsx",
        "app/reproduce/page.tsx",
        "components/Notebook.tsx",
        "components/Spine.tsx",
        "components/Entries.tsx",
        "lib/science.ts",
    ):
        chunks.append((PORTAL_DIR / rel).read_text(encoding="utf-8"))
    return "\n".join(chunks)


def test_door_copy_has_no_chrome() -> None:
    hits = DOOR_CHROME.findall(_door_copy())
    assert not hits, f"science-door copy still names chrome: {hits}"


def test_consumes_science_view_not_peerj_pdfs() -> None:
    blob = _blob()
    science_ts = (PORTAL_DIR / "lib" / "science.ts").read_text(encoding="utf-8")
    assert "science.json" in science_ts
    for rel in (
        "app/page.tsx",
        "app/staircase/page.tsx",
        "app/scale/page.tsx",
        "app/subspace/page.tsx",
        "app/ledger/page.tsx",
        "app/reproduce/page.tsx",
    ):
        text = (PORTAL_DIR / rel).read_text(encoding="utf-8")
        assert "science" in text
        assert "C_" not in text
        assert "figures.json" not in text
    assert not PEERJ_FIGURE_PDF.search(blob)


def test_no_live_journal_pdf_in_portal() -> None:
    forbidden = {"main.pdf", "manuscript.pdf"}
    found = [
        p
        for p in PORTAL_DIR.rglob("*")
        if p.is_file()
        and p.name.lower() in forbidden
        and "node_modules" not in p.parts
    ]
    found += [
        p
        for p in PORTAL_DIR.rglob("Figure*.pdf")
        if "node_modules" not in p.parts
    ]
    assert not found


def test_no_emoji_slop() -> None:
    assert not EMOJI.search(_blob())


def test_si_tost_gutter_is_structural() -> None:
    probes = (PORTAL_DIR / "app" / "page.tsx").read_text(encoding="utf-8")
    assert "SI" in probes
    assert "TOST" in probes
    assert "Holm" in probes


def test_assets_use_repo_base_path_not_user_site() -> None:
    config = (PORTAL_DIR / "next.config.ts").read_text(encoding="utf-8")
    assert 'basePath: `/${repo}`' in config or "/architecture-staircase" in config
    blob = _blob()
    assert "/assets" not in blob.replace("assets/", "")
    assert "peterponyu.github.io/" not in blob.lower() or "architecture-staircase" in blob


def test_build_script_exports_next() -> None:
    text = (PORTAL_DIR / "build.sh").read_text(encoding="utf-8")
    assert "latexmk" not in text.lower()
    assert "_site" in text
    assert "FIGURE-INDEX" in text
    assert "npm run build" in text
    assert "output" in (PORTAL_DIR / "next.config.ts").read_text(encoding="utf-8")
    assert "cp -a experiments" not in text

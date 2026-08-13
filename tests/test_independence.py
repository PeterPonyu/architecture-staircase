"""Independence / freeze hygiene for this warehouse (Z3, Z5)."""

from __future__ import annotations

from conftest import PORTAL_DIR, REPO_ROOT


def test_portal_does_not_present_a_five_paper_suite() -> None:
    html = (PORTAL_DIR / "index.html").read_text(encoding="utf-8")
    for foreign in (
        "muon-norm-cap-grokking",
        "grokking-clock",
        "free-repetition-band",
        "calibration-traps",
    ):
        assert foreign not in html, f"Z3: portal must not chrome sibling remote {foreign}"


def test_readme_does_not_link_a_shared_five_paper_site() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "paper-a" not in text
    assert "peterponyu.github.io/muon-norm-cap-grokking" not in text
    assert "peterponyu.github.io/grokking-clock" not in text

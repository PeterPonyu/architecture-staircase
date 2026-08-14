#!/usr/bin/env python3
"""Build a door-safe science view from FIGURE-INDEX + summaries.

The live door may render titles, panel names, and HOLD questions.
It must never receive warehouse ids, generators, sources, venue names,
or file paths.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "papers" / "FIGURE-INDEX.json"
SUMMARIES = ROOT / "papers" / "figs" / "summaries"
OUT_PATH = ROOT / "portal" / "public" / "science.json"

PATH_LEAK = re.compile(
    r"(papers/|figs/|experiments/|portal/|_site/|"
    r"\bC_[A-Za-z0-9_]+|"
    r"Figure(?:1[0-6]|[1-9])\.pdf|"
    r"main\.tex|FIGURE-INDEX|PIPELINE|"
    r"\.(tex|json|jsonl|R|md|ya?ml)\b)",
    re.I,
)
PUB_LEAK = re.compile(
    r"peerj|arxiv|manuscript|journal|warehouse|submission|preprint",
    re.I,
)

CAPTION_REWRITE = {
    "C_case": "Per-degree trajectories for a single seed.",
    "C_scheme": "Study design for the two-probe staircase.",
    "C_ownership_lr_sensitivity": "Learning-rate sweep for the attention-freeze ownership probe.",
}

LEAF_IDS = {
    "probes_freeze": ("C_ladder_necessity", "C_ownership_localize"),
    "probes_ablation": ("C_ownership_optaxis",),
    "staircase": (
        "C_landscape",
        "C_targetfamily",
        "C_staircase_si",
        "C_staircase_rho",
    ),
    "scale": ("C_arch_depth", "C_scale", "C_pure3_speedup", "C_timing"),
    "subspace": ("C_subspace",),
}

# Extra HOLD questions moved off their home object so Probes/Subspace/Scale
# each get the two-probe split without dumping the whole dissociation object.
DISSOC_EXTRA = {
    "probes_freeze": ("b",),
    "probes_ablation": ("a",),
    "scale": ("c",),
    "subspace": ("d",),
}


def clean_text(text: str) -> str:
    text = " ".join(text.split())
    text = text.replace("Walsh-correlation ", "")
    text = text.replace("Walsh-correlation", "degree")
    if PATH_LEAK.search(text) or PUB_LEAK.search(text):
        raise SystemExit(f"refuse: science text still leaks chrome: {text!r}")
    return text


def load_summary(fig_id: str, summary_rel: str | None) -> dict | None:
    if not summary_rel:
        return None
    name = Path(summary_rel).name
    path = SUMMARIES / name
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("artifact") == "C_smoke":
        return None
    return data


def questions_from(summary: dict | None, only: tuple[str, ...] | None = None) -> list[str]:
    if not summary:
        return []
    out: list[str] = []
    for panel in summary.get("panels") or []:
        if only and panel.get("id") not in only:
            continue
        question = panel.get("question")
        if isinstance(question, str) and question.strip():
            out.append(clean_text(question))
    return out


def object_view(fig: dict) -> dict:
    fig_id = fig["id"]
    title = CAPTION_REWRITE.get(fig_id, fig.get("caption") or fig_id)
    title = clean_text(title)
    panels = [clean_text(p) for p in (fig.get("caption_panels") or []) if p]
    summary = load_summary(fig_id, fig.get("summary"))
    questions = questions_from(summary)
    return {
        "title": title,
        "panels": panels,
        "questions": questions,
    }


def leaf_entries(index_by_id: dict[str, dict], ids: tuple[str, ...]) -> list[dict]:
    entries = []
    for fig_id in ids:
        view = object_view(index_by_id[fig_id])
        entries.append(
            {
                "title": view["title"],
                "panels": view["panels"],
                "questions": view["questions"],
            }
        )
    return entries


def main() -> None:
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    figures = index["figures"]
    index_by_id = {fig["id"]: fig for fig in figures}
    dissoc = load_summary("C_dissoc", "figs/summaries/C_dissoc.json")

    science = {
        "probes": {
            "freeze": leaf_entries(index_by_id, LEAF_IDS["probes_freeze"]),
            "ablation": leaf_entries(index_by_id, LEAF_IDS["probes_ablation"]),
        },
        "staircase": leaf_entries(index_by_id, LEAF_IDS["staircase"]),
        "scale": leaf_entries(index_by_id, LEAF_IDS["scale"]),
        "subspace": leaf_entries(index_by_id, LEAF_IDS["subspace"]),
        "ledger": [object_view(fig) for fig in figures],
        "rebuild": [{"title": object_view(fig)["title"]} for fig in figures],
    }

    for leaf, panel_ids in DISSOC_EXTRA.items():
        extra_q = questions_from(dissoc, panel_ids)
        if not extra_q:
            continue
        extra = {
            "title": "Two-probe ownership plus geometry and timing",
            "panels": [],
            "questions": extra_q,
        }
        if leaf == "probes_freeze":
            science["probes"]["freeze"].append(extra)
        elif leaf == "probes_ablation":
            science["probes"]["ablation"].append(extra)
        else:
            science[leaf].append(extra)

    blob = json.dumps(science, indent=2, ensure_ascii=True) + "\n"
    if PATH_LEAK.search(blob) or PUB_LEAK.search(blob):
        raise SystemExit("refuse: generated science.json still leaks chrome")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(blob, encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(ROOT)} ({len(science['ledger'])} objects)")


if __name__ == "__main__":
    main()

"""Shared publication figure style for all five papers (A/B/C/E1/E2).

Import at the top of every generator:

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # to experiments/
    import figstyle
    figstyle.apply()
    ...
    ax.plot(..., color=figstyle.OPT["muon"])

Design rules (publication-ready, double-column Neurocomputing / Physica D and
single-column TMLR):

1. COLOURS — Okabe-Ito colourblind-safe palette. Optimizers use a FIXED mapping
   across every figure: AdamW=blue, Muon=vermillion, SGDM=grey.
2. FONTS — base 10.5pt. To avoid LaTeX down-scaling shrinking text below ~8pt,
   render each figure at its FINAL PRINTED WIDTH (see WIDTH_IN) so the
   \\includegraphics scale factor is ~1.0. Do NOT draw a figure at 14in wide and
   then place it at width=0.3\\linewidth.
3. NO in-figure suptitles/titles that duplicate the caption. Keep only short
   structural panel tags "(a)", "(b)". Never bake section numbers, file paths,
   or internal direction/probe IDs (P1/P2, 012, 019) into a PNG.
4. Optimizer labels are Title-case in any visible text: AdamW / Muon / SGDM.
"""
from __future__ import annotations
import matplotlib as mpl

# ---- Okabe-Ito colourblind-safe palette ----
CB = {
    "blue":       "#0072B2",
    "orange":     "#E69F00",
    "vermillion": "#D55E00",
    "green":      "#009E73",
    "skyblue":    "#56B4E9",
    "yellow":     "#F0E442",
    "purple":     "#CC79A7",
    "grey":       "#999999",
    "black":      "#000000",
}

# ---- canonical, fixed optimizer colours (use everywhere) ----
OPT = {
    "adamw": CB["blue"], "AdamW": CB["blue"],
    "muon":  CB["vermillion"], "Muon": CB["vermillion"],
    "sgdm":  CB["grey"], "SGDM": CB["grey"],
}
OPT_LABEL = {"adamw": "AdamW", "muon": "Muon", "sgdm": "SGDM"}

# ---- final printed widths (inches) to render at, so scale factor ~= 1 ----
# Neurocomputing/Physica D: single column ~3.35in, full (figure*) ~7.0in.
# TMLR: single column ~6.0in.
WIDTH_IN = {
    "col2_single": 3.35,   # one column in a 2-column journal
    "col2_full":   7.0,    # figure* spanning both columns
    "tmlr":        6.0,    # TMLR full text width (\linewidth)
    "tmlr_0.7":    4.2,
}

# sequential colourmap that is colourblind-safe (replace RdYlGn / jet)
SEQ_CMAP = "cividis"
DIVERGING_CMAP = "coolwarm"  # if a true diverging map is needed


def apply():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.0,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.6,
        "lines.markersize": 5.0,
    })


def opt_color(name: str) -> str:
    """Colour for an optimizer name (case-insensitive); grey fallback."""
    return OPT.get(name, OPT.get(name.lower(), CB["grey"]))

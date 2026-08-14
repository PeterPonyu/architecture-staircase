# Figure pipeline (Paper C)

R → TikZ / vector-PDF → LuaLaTeX. Warehouse manuscript `papers/C/main.tex`
loads `\input{../figs/figpreamble.tex}` and `\figtikz{Name}` (heatmap tier:
preamble-routed `\includegraphics{Name.pdf}` into untracked `figs/vec/`).

## Emitters in this warehouse

- `fig_pipeline.R` — shared emitter (`emit_vector`)
- `figpreamble.tex` — `\figtikz` + `\graphicspath{{../figs/vec/}{../figs/}}`
- `make_C_figs_r.R` — body figures including the four YAML-absent MAP names
  (`C_case`, `C_scale`, `C_subspace`, `C_ownership_lr_sensitivity`)
- `make_C_new_figs_r.R`
- `make_gap20260705_C_figs_r.R`
- `make_landscape_r.R` — `C_landscape`
- `C_scheme` is a TikZ schematic; INDEX `tex_build` is `figs/tex/C_scheme.tex`

Compiled `figs/tex/` and `figs/vec/` are gitignored. Do not commit PeerJ
`FigureN.pdf`. Portal reads `papers/FIGURE-INDEX.json` (JSON-first).

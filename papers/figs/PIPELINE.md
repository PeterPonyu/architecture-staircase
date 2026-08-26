# Figure pipeline

R → TikZ / vector-PDF → LuaLaTeX.

Shared pieces in this directory:

- `fig_pipeline.R` — shared emitter (`emit_vector`)
- `figpreamble.tex` — `\figtikz` plus `\graphicspath` into untracked `figs/vec/`

Compiled `figs/tex/` and `figs/vec/` are gitignored. The portal reads
`papers/FIGURE-INDEX.json`.

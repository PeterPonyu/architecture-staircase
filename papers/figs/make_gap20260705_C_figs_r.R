#!/usr/bin/env Rscript
# make_gap20260705_C_figs_r.R — two manuscript gap-battery figures (2026-07-06).
# Data source: experiments/results/ieee_gap_20260705/C/gapC_verdict.json (read here,
# never hardcoded). Emits canonical PNG (papers/figs/<name>.png), SVG
# (papers/figs/evidence_r/<name>.svg), and vector tikz (papers/figs/tex/<name>.tex)
# via save_both()/emit_vector(), matching the make_C_figs_r.R conventions exactly so
# the IEEE-variant maker can capture the objects (save_both -> .capture rewrite).
#
#   C_targetfamily        — G1 target-family rank-index strip (3 families x 3 opt)
#   C_ownership_localize   — Muon ownership localization bar (both/l0attn/l0mlp/l1attn/l1mlp)
#
# Run from repo root:  Rscript papers/figs/make_gap20260705_C_figs_r.R

ver <- paste(R.version$major, sub('\\..*', '', R.version$minor), sep = '.')
userlib <- file.path(Sys.getenv('HOME'), 'R', 'x86_64-pc-linux-gnu-library', ver)
.libPaths(c(userlib, .libPaths()))

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(patchwork)
  library(ragg)
  library(svglite)
  library(cowplot)
})

root    <- normalizePath(file.path(getwd()))
fig_dir <- file.path(root, 'papers', 'figs')
evd_dir <- file.path(root, 'papers', 'figs', 'evidence_r')
res_dir <- file.path(root, 'experiments', 'results')
dir.create(evd_dir, showWarnings = FALSE, recursive = TRUE)
source(file.path(fig_dir, 'fig_pipeline.R'))  # emit_vector(): tikz/.tex + cairo_pdf/.pdf
source(file.path(fig_dir, 'C_panel_contract.R'))

# ── Colour palette (colourblind-safe, matches make_C_figs_r.R / figstyle.py) ──
CB <- list(
  blue      = '#0072B2',
  orange    = '#E69F00',
  green     = '#009E73',
  vermillion= '#D55E00',
  sky       = '#56B4E9',
  grey      = '#999999',
  yellow    = '#F0E442',
  pink      = '#CC79A7'
)
OPT <- list(adamw = CB$blue, muon = CB$vermillion, sgdm = CB$grey)

# ── Theme (identical to make_C_figs_r.R::paper_theme) ─────────────────────────
paper_theme <- function(base_size = 9) {
  theme_minimal(base_size = base_size, base_family = 'TeX Gyre Termes') +
    theme(
      panel.grid.minor    = element_blank(),
      panel.grid.major.x  = element_blank(),
      panel.grid.major.y  = element_line(linewidth = 0.25, colour = '#d9dde3'),
      axis.title          = element_text(family = 'TeX Gyre Termes', colour = '#1a202c', size = base_size),
      axis.text           = element_text(family = 'TeX Gyre Termes', colour = '#2d3748', size = base_size - 0.5),
      plot.title          = element_text(family = 'TeX Gyre Termes', face = 'plain', colour = '#111827',
                                         size = base_size + 1),
      plot.subtitle       = element_text(family = 'TeX Gyre Termes', colour = '#1a1a1a', size = base_size - 0.5),
      legend.position     = 'top',
      legend.title        = element_text(family = 'TeX Gyre Termes', size = base_size - 0.5),
      legend.text         = element_text(family = 'TeX Gyre Termes', size = base_size - 0.5),
      strip.text          = element_text(family = 'TeX Gyre Termes', face = 'plain', colour = '#1a202c'),
      plot.margin         = margin(5.5, 6, 5.5, 6),
      plot.tag            = element_text(family = 'TeX Gyre Termes', face = 'bold', size = 11),
      plot.tag.position   = c(0.02, 0.98)
    )
}

# ── save_both: ragg 300dpi PNG + svglite SVG + vector tier (== make_C_figs_r.R) ─
save_both <- function(p, name, w = 5.0, h = 4.2) {
  png_path <- file.path(fig_dir, paste0(name, '.png'))
  svg_path <- file.path(evd_dir, paste0(name, '.svg'))
  ragg::agg_png(png_path, width = w, height = h, units = 'in', res = 300, scaling = 1)
  print(p); dev.off()
  svglite::svglite(svg_path, width = w, height = h)
  print(p); dev.off()
  emit_vector(p, name, w, h)   # vector tier: tikz (.tex) or cairo_pdf (.pdf)
  sz <- file.info(png_path)$size
  cat(sprintf('  saved %s  (%d bytes)\n', png_path, sz))
}

read_json_r <- function(path, simplifyVector = TRUE) {
  txt <- paste(readLines(path, warn = FALSE), collapse = '\n')
  txt <- gsub('\\bNaN\\b', 'null', txt)
  txt <- gsub('\\bInfinity\\b', 'null', txt)
  jsonlite::fromJSON(txt, simplifyVector = simplifyVector)
}

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0) y else x

cat('=== make_gap20260705_C_figs_r.R ===\n')
gapC <- read_json_r(file.path(res_dir, 'ieee_gap_20260705', 'C', 'gapC_verdict.json'),
                    simplifyVector = FALSE)

# ════════════════════════════════════════════════════════════════════════════
# 1. C_targetfamily — G1 order intrinsic beyond equal-weight target
# Rank index (0..1 order recovery, signed -1..1) per target family x optimizer.
# The inverse-weighted (high-degree-dominant) headline: order STILL ascends
# (+0.61 AdamW, +0.83 Muon), so the low-to-high staircase is not an artifact of
# the equal-weight construction. SGDM invw is fit-gated (fit 0.11) -> hollow.
# ════════════════════════════════════════════════════════════════════════════
cat('1. C_targetfamily\n')

cells <- gapC$c1_targets$cells
fit_gate <- as.numeric(gapC$c1_targets$fit_gate %||% 0.85)
fam_lab <- c(geom = 'Geom.\ndecay', gapped = 'Gapped\n{1,3,4}', invw = 'Inverse-\nwt.')
opt_lab <- c(adamw = 'AdamW', muon = 'Muon', sgdm = 'SGDM')

tf <- bind_rows(lapply(names(cells), function(k) {
  c <- cells[[k]]
  data.frame(
    family = as.character(c$family),
    opt    = as.character(c$optimizer),
    rank   = as.numeric(c$rank_index),
    rstd   = as.numeric(c$rank_std),
    n      = as.integer(c$n),
    fit    = as.numeric(c$fit_med),
    stringsAsFactors = FALSE)
})) %>%
  mutate(se        = rstd / sqrt(n),
         fitgated  = fit < fit_gate,
         family    = factor(family, levels = c('geom', 'gapped', 'invw'),
                             labels = fam_lab[c('geom', 'gapped', 'invw')]),
         opt       = factor(opt, levels = c('adamw', 'muon', 'sgdm'),
                             labels = opt_lab[c('adamw', 'muon', 'sgdm')]))

dodge <- position_dodge(width = 0.62)
p_tf <- ggplot(tf, aes(family, rank, colour = opt, group = opt)) +
  geom_hline(yintercept = 0, colour = '#374151', linewidth = 0.5) +
  geom_hline(yintercept = c(-1, 1), colour = '#c9ced6', linewidth = 0.3, linetype = 'dotted') +
  geom_errorbar(aes(ymin = pmax(rank - se, -1), ymax = pmin(rank + se, 1)),
                width = 0.18, linewidth = 0.6, position = dodge) +
  geom_point(aes(shape = fitgated, fill = opt), size = 2.6, stroke = 0.9, position = dodge) +
  scale_colour_manual(values = c('AdamW' = OPT$adamw, 'Muon' = OPT$muon, 'SGDM' = OPT$sgdm),
                      name = NULL) +
  scale_fill_manual(values = c('AdamW' = OPT$adamw, 'Muon' = OPT$muon, 'SGDM' = OPT$sgdm),
                    guide = 'none') +
  scale_shape_manual(values = c('FALSE' = 21, 'TRUE' = 1),
                     labels = c('FALSE' = 'fit passed', 'TRUE' = 'fit-gated'),
                     name = NULL,
                     guide = guide_legend(override.aes =
                       list(fill = c('#333333', 'white'), colour = '#333333', size = 2.4))) +
  annotate('text', x = 1.95, y = -0.30, hjust = 0.5, size = 2.3, colour = '#374151',
           label = 'Inverse-weighted: order still\nascends (+0.61 AdamW, +0.83 Muon)') +
  coord_cartesian(ylim = c(-0.75, 1.05)) +
  labs(title = '(a) Staircase rank index by target family and optimizer',
       x = NULL, y = 'Staircase rank index (1 = ascending)') +
  paper_theme(9) +
  theme(legend.position = 'top', legend.box = 'horizontal',
        legend.box.spacing = unit(1, 'pt'),
        legend.margin      = margin(0, 0, 0, 0),
        plot.margin        = margin(2, 6, 5.5, 6))

# (b) fit by cell with the fit gate: the inverse-weighted SGDM cell is
#     gated out honestly, every reported cell fits
p_tf_b <- ggplot(tf, aes(family, fit, colour = opt, group = opt)) +
  geom_hline(yintercept = fit_gate, colour = '#888', linetype = 'dashed', linewidth = 0.6) +
  geom_point(size = 2.4, position = dodge) +
  geom_line(linewidth = 0.7, position = dodge) +
  scale_colour_manual(values = c('AdamW' = OPT$adamw, 'Muon' = OPT$muon, 'SGDM' = OPT$sgdm),
                      name = NULL) +
  annotate('text', x = 0.7, y = fit_gate, label = sprintf('fit gate %.2f', fit_gate),
           hjust = 0, vjust = -0.6, size = 2.4, colour = '#374151') +
  coord_cartesian(ylim = c(0, 1.05)) +
  labs(title = '(b) Fit per cell (gate check)',
       x = NULL, y = 'Median fit correlation') +
  paper_theme(9) +
  theme(legend.position = 'none')

# (c) per-degree onset (median half-time) by family: the staircase timing is
#     family-robust, not an equal-weight artifact
half_rows <- bind_rows(lapply(names(cells), function(k) {
  c <- cells[[k]]
  hm <- c$half_med
  if (is.null(hm)) return(NULL)
  bind_rows(lapply(names(hm), function(d) {
    data.frame(family = as.character(c$family), opt = as.character(c$optimizer),
               degree = factor(as.integer(d), levels = sort(as.integer(names(hm)))),
               half = as.numeric(hm[[d]]), stringsAsFactors = FALSE)
  }))
})) %>%
  mutate(family = factor(family, levels = c('geom', 'gapped', 'invw'),
                         labels = fam_lab[c('geom', 'gapped', 'invw')]),
         opt = factor(opt, levels = c('adamw', 'muon', 'sgdm'),
                      labels = opt_lab[c('adamw', 'muon', 'sgdm')]))
p_tf_c <- ggplot(half_rows, aes(degree, half, colour = opt, group = opt)) +
  geom_line(linewidth = 0.8, position = dodge) +
  geom_point(size = 2.2, position = dodge) +
  facet_wrap(~ family) +
  scale_colour_manual(values = c('AdamW' = OPT$adamw, 'Muon' = OPT$muon, 'SGDM' = OPT$sgdm),
                      name = NULL) +
  scale_y_log10() +
  labs(title = '(c) Per-degree onset by family',
       x = NULL, y = 'Median half-time (log)') +
  paper_theme(9) +
  theme(legend.position = 'none',
        axis.text.x = element_text(size = 6.5, angle = 45, hjust = 1),
        strip.text = element_text(size = 7.5))

# (d) rank vs fit: the ordering does not track fit quality (no monotone
#     relation across the 9 cells)
p_tf_d <- ggplot(tf, aes(fit, rank, colour = opt, shape = family)) +
  geom_hline(yintercept = 0, colour = '#888', linetype = 'dashed', linewidth = 0.5) +
  geom_point(size = 2.6) +
  scale_colour_manual(values = c('AdamW' = OPT$adamw, 'Muon' = OPT$muon, 'SGDM' = OPT$sgdm),
                      name = NULL, guide = 'none') +
  scale_shape_manual(values = c(15, 16, 17), name = NULL) +
  coord_cartesian(xlim = c(0, 1.05), ylim = c(-0.75, 1.05)) +
  labs(title = '(d) Rank vs fit (not a fit artifact)',
       x = 'Median fit correlation', y = 'Rank index') +
  paper_theme(9) +
  theme(legend.position = 'top')

p_targetfamily <- compose_C_four_panel(
  list(p_tf, p_tf_b, p_tf_c, p_tf_d),
  'C_targetfamily', 7.2, 5.4, root)
save_both(p_targetfamily, 'C_targetfamily', 7.2, 5.4)

# ════════════════════════════════════════════════════════════════════════════
# 2. C_ownership_localize — Muon two-path redundancy localized to layer 1
# Median final degree-4 Walsh correlation by freeze arm under Muon. Floor control
# (both frozen) = +0.002 dies (< 0.15 threshold) -> the redundancy is a genuine
# two-path hidden-network property, not a leaky freeze. Layer-0 freezes preserve
# (l0attn +0.50, l0mlp +0.47 ~= unfrozen) while layer-1 freezes kill/reduce
# (l1attn +0.02, l1mlp +0.18) -> degree 4 lives in LAYER 1 under Muon.
# ════════════════════════════════════════════════════════════════════════════
cat('2. C_ownership_localize\n')

arms <- gapC$c2_ownership$arms
thr  <- as.numeric(gapC$c2_ownership$acquire_thresh %||% 0.15)
arm_order <- c('both', 'l0attn', 'l0mlp', 'l1attn', 'l1mlp')
arm_lab <- c(both = 'Both frozen', l0attn = 'L0 attn', l0mlp = 'L0 MLP',
             l1attn = 'L1 attn', l1mlp = 'L1 MLP')
# grouping: floor control (both), layer-0 preserved, layer-1 kills/reduces
arm_grp <- c(both = 'Floor control', l0attn = 'Layer 0 (preserved)',
             l0mlp = 'Layer 0 (preserved)', l1attn = 'Layer 1 (kills degree 4)',
             l1mlp = 'Layer 1 (kills degree 4)')

oz <- bind_rows(lapply(arm_order, function(a) {
  data.frame(arm = a, deg4 = as.numeric(arms[[a]]$deg4_med), stringsAsFactors = FALSE)
})) %>%
  mutate(grp = factor(arm_grp[arm],
                      levels = c('Floor control', 'Layer 0 (preserved)', 'Layer 1 (kills degree 4)')),
         arm = factor(arm, levels = arm_order, labels = arm_lab[arm_order]))

# near-zero bars (both +0.002, l1attn +0.02) render as invisible: add a baseline
# tick + value label so they read as measured zero, not missing data (matches the
# C_ownership_optaxis zero-label convention).
zero_bars <- oz %>% filter(deg4 < 0.05) %>% mutate(xpos = as.integer(arm))

# Freeze-arm palette (bundle decision 2026-07-10, unified 2026-07-16): freeze
# figures stay OFF the optimizer palette (blue/vermillion/grey = AdamW/Muon/SGDM).
# black = control, skyblue = preserved arms, purple = suppressing arms — matching
# C_dissoc(b)/C_ownership_optaxis/C_ownership_lr_sensitivity (black/purple/skyblue).
grp_cols <- c('Floor control' = '#000000',
              'Layer 0 (preserved)' = '#56B4E9',
              'Layer 1 (kills degree 4)' = '#7B3294')

p_oz <- ggplot(oz, aes(arm, deg4, fill = grp)) +
  geom_col(width = 0.66, alpha = 0.9) +
  geom_hline(yintercept = 0, colour = '#374151', linewidth = 0.5) +
  geom_hline(yintercept = thr, colour = '#374151', linewidth = 0.55, linetype = 'dashed') +
  geom_segment(data = zero_bars,
               aes(x = xpos - 0.12, xend = xpos + 0.12, y = 0, yend = 0),
               inherit.aes = FALSE, colour = '#374151', linewidth = 1.6) +
  geom_text(data = oz, aes(label = sprintf('%+.3f', deg4)),
            vjust = -0.5, size = 2.5, colour = '#1a202c', nudge_x = -0.12) +
  scale_fill_manual(values = grp_cols, name = NULL) +
  coord_cartesian(ylim = c(-0.02, 0.60)) +
  labs(title = '(a) Degree-4 Walsh correlation by freeze arm (Muon)',
       x = NULL, y = 'Degree-4 final Walsh correlation') +
  paper_theme(9) +
  theme(legend.position = 'top', legend.box.spacing = unit(1, 'pt'),
        legend.margin = margin(0, 0, 0, 0),
        legend.text   = element_text(size = 7.5),
        axis.text.x   = element_text(size = 6.5, angle = 35, hjust = 1))

# (b) per-seed degree-4 by arm (n = 15; the distributions behind the medians)
oz_seeds <- bind_rows(lapply(arm_order, function(a) {
  vals <- as.numeric(unlist(arms[[a]]$deg4_all))
  data.frame(arm = a, deg4 = vals, seed = seq_along(vals) - 1L, stringsAsFactors = FALSE)
})) %>%
  mutate(grp = factor(arm_grp[arm],
                      levels = c('Floor control', 'Layer 0 (preserved)', 'Layer 1 (kills degree 4)')),
         arm = factor(arm, levels = arm_order, labels = arm_lab[arm_order]))
p_oz_b <- ggplot(oz_seeds, aes(arm, deg4, colour = grp)) +
  geom_hline(yintercept = thr, colour = '#888', linetype = 'dashed', linewidth = 0.6) +
  stat_summary(fun = median, geom = 'crossbar', width = 0.5, colour = '#1a1a1a') +
  geom_point(position = position_jitter(width = 0.07, height = 0, seed = 67),
             size = 1.4, alpha = 0.75, show.legend = FALSE) +
  scale_colour_manual(values = grp_cols, guide = 'none') +
  labs(title = '(b) Per-seed degree-4 (n = 15/arm)',
       x = NULL, y = 'Degree-4 final Walsh corr.') +
  paper_theme(9) +
  theme(axis.text.x = element_text(size = 6.5, angle = 35, hjust = 1))

# (c) fit control: freezing any sub-network leaves overall fit high, so the
#     degree-4 drops in (a)-(b) are selective, not training failures
fit_df <- bind_rows(lapply(arm_order, function(a) {
  data.frame(arm = a, fit = as.numeric(arms[[a]]$fit_med), stringsAsFactors = FALSE)
})) %>% mutate(grp = factor(arm_grp[arm],
                            levels = c('Floor control', 'Layer 0 (preserved)', 'Layer 1 (kills degree 4)')),
               arm = factor(arm, levels = arm_order, labels = arm_lab[arm_order]))
p_oz_c <- ggplot(fit_df, aes(arm, fit, fill = grp)) +
  geom_col(width = 0.66, alpha = 0.9) +
  geom_text(aes(label = sprintf('%.2f', fit)), vjust = -0.4, size = 2.6) +
  scale_fill_manual(values = grp_cols, guide = 'none') +
  coord_cartesian(ylim = c(0, 1.05)) +
  labs(title = '(c) Overall fit preserved under every freeze',
       x = NULL, y = 'Median final fit') +
  paper_theme(9) +
  theme(axis.text.x = element_text(size = 6.5, angle = 35, hjust = 1))

# (d) acquisition count per arm (seeds above the 0.15 threshold): the L1-MLP
#     arm keeps a bimodal fraction above threshold — seed-structured, not uniform
acq <- oz_seeds %>% group_by(arm, grp) %>%
  summarise(k = sum(deg4 > thr), n = n(), .groups = 'drop')
p_oz_d <- ggplot(acq, aes(arm, k / n, fill = grp)) +
  geom_col(width = 0.66, alpha = 0.9) +
  geom_text(aes(label = paste0(k, '/', n)), vjust = -0.4, size = 2.6) +
  scale_fill_manual(values = grp_cols, guide = 'none') +
  coord_cartesian(ylim = c(0, 1.05)) +
  labs(title = '(d) Seeds above acquisition threshold',
       x = NULL, y = 'Fraction of 15 seeds') +
  paper_theme(9) +
  theme(axis.text.x = element_text(size = 6.5, angle = 35, hjust = 1))

p_ownership_localize <- compose_C_four_panel(
  list(p_oz, p_oz_b, p_oz_c, p_oz_d),
  'C_ownership_localize', 7.2, 5.4, root)
save_both(p_ownership_localize, 'C_ownership_localize', 7.2, 5.4)

cat('=== done make_gap20260705_C_figs_r.R ===\n')

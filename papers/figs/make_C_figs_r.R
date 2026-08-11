#!/usr/bin/env Rscript
# make_C_figs_r.R — Professional R/ggplot2 renderer for all 10 manuscript figures.
# Uses same data sources as the Python generators (verified; read from JSON verdicts
# and raw .jsonl files). Outputs PNG to papers/figs/<name>.png and SVG to
# papers/figs/evidence_r/<name>.svg.
#
# Design standard applied:
#  - Panel labels (a)(b)... via patchwork plot_annotation(tag_levels='a') + bold left-aligned
#  - No big composite embedded titles; short per-panel titles only
#  - Verbose subtitles (seeds/config notes) moved to LaTeX \caption
#  - Render at 5.0in width, fonts >=8pt, heights <=9.5in

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

# ── Colour palette (colourblind-safe, matches figstyle.py) ──────────────────
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

# Freeze-arm palette (bundle decision 2026-07-10, unified 2026-07-16): deliberately
# OFF the optimizer palette. black = unfrozen baseline, purple = attention frozen,
# skyblue = MLP frozen. Shared by C_dissoc(b), C_ownership_optaxis,
# C_ownership_lr_sensitivity (and C_ownership_localize in the gap-battery script).
FREEZE <- list(none = '#000000', attn = '#7B3294', mlp = '#56B4E9')

# ── 10^{-k} axis tick labels for log scales (replaces "1e-02"-style labels) ──
# tikzDevice sanitizes "$" etc., so raw "$10^{-2}$" cannot pass through a label.
# Instead emit Unicode superscript characters and register PER-CHARACTER sanitize
# replacements (tikzDevice's sanitizer is strictly character-wise: it strsplits
# the string, so multi-char tokens never match). Appended AFTER fig_pipeline.R's
# map; C-figure-specific, additive, a no-op for labels without these characters.
# The PNG/SVG raster tiers render the Unicode superscripts directly.
options(
  tikzSanitizeCharacters   = c(getOption('tikzSanitizeCharacters'),
                               '⁻', '²', '³', '⁴', '⁵', '⁶'),
  tikzReplacementCharacters = c(getOption('tikzReplacementCharacters'),
                                '$^{-}$', '$^{2}$', '$^{3}$', '$^{4}$', '$^{5}$', '$^{6}$')
)
pow10_labels <- function(x) {
  sup <- c('2' = '⁻²', '3' = '⁻³', '4' = '⁻⁴', '5' = '⁻⁵', '6' = '⁻⁶')
  vapply(x, function(v) {
    if (is.na(v)) return('')
    paste0('10', sup[[as.character(round(-log10(v)))]])
  }, character(1))
}

# ── Theme ────────────────────────────────────────────────────────────────────
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

# ── save_both: ragg 300dpi PNG + svglite SVG ────────────────────────────────
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

# Helper: relaxed JSON reader (handles NaN/Infinity in some files)
read_json_r <- function(path, simplifyVector = TRUE) {
  txt <- paste(readLines(path, warn = FALSE), collapse = '\n')
  txt <- gsub('\\bNaN\\b', 'null', txt)
  txt <- gsub('\\bInfinity\\b', 'null', txt)
  jsonlite::fromJSON(txt, simplifyVector = simplifyVector)
}

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0) y else x

cat('=== make_C_figs_r.R ===\n')

# ════════════════════════════════════════════════════════════════════════════
# 1. C_case — full per-degree staircase, one seed, unfrozen vs attn-frozen
# Two panels (one per arm); within each, the deg-1..4 Walsh-correlation
# trajectories. Reader sees (a) the unfrozen arm acquire all four degrees in
# staircase order, (b) the frozen arm acquire deg 1-3 but block deg-4.
# Data: experiments/results/arch_staircase/d128_l2_h4_{none,attn}_s0.jsonl
# ════════════════════════════════════════════════════════════════════════════
cat('1. C_case\n')

# Read full per-degree (1..4) trajectory; recompute finals from raw.
read_traj_alldeg <- function(path) {
  lines <- readLines(path, warn = FALSE)
  rows  <- lapply(lines, function(l) {
    d <- tryCatch(jsonlite::fromJSON(l, simplifyVector = TRUE), error = function(e) NULL)
    if (is.null(d) || !('step' %in% names(d))) return(NULL)
    data.frame(step = d$step,
               deg  = 1:4,
               corr = c(d$deg_corr[['1']], d$deg_corr[['2']],
                        d$deg_corr[['3']], d$deg_corr[['4']]),
               stringsAsFactors = FALSE)
  })
  bind_rows(Filter(Negate(is.null), rows))
}

base_as   <- file.path(res_dir, 'arch_staircase')
traj_none <- read_traj_alldeg(file.path(base_as, 'd128_l2_h4_none_s0.jsonl')) %>%
  mutate(arm = 'Unfrozen')
traj_attn <- read_traj_alldeg(file.path(base_as, 'd128_l2_h4_attn_s0.jsonl')) %>%
  mutate(arm = 'Attention frozen')
traj_df   <- bind_rows(traj_none, traj_attn) %>%
  mutate(arm = factor(arm, levels = c('Unfrozen', 'Attention frozen')),
         deg = factor(deg, levels = 1:4,
                      labels = c('degree 1', 'degree 2', 'degree 3', 'degree 4')))

# Recompute the cited final degree-4 values from raw (NUMBERS RED LINE):
final_none_d4 <- traj_none %>% filter(deg == 4) %>% pull(corr) %>% tail(1)
final_attn_d4 <- traj_attn %>% filter(deg == 4) %>% pull(corr) %>% tail(1)
cat(sprintf('   recomputed: unfrozen deg-4 final = %+.4f (cite +0.427)\n', final_none_d4))
cat(sprintf('   recomputed: frozen   deg-4 final = %+.4f (cite -0.003)\n', final_attn_d4))

# Per-degree finals for both arms (for honest caption reporting)
finals_tbl <- traj_df %>% group_by(arm, deg) %>%
  summarise(final = tail(corr, 1), .groups = 'drop')
print(finals_tbl)

# Degree colours: sequential blue->red so acquisition order reads as a ramp.
deg_cols <- c('degree 1' = CB$sky, 'degree 2' = CB$green,
              'degree 3' = CB$orange, 'degree 4' = CB$vermillion)

# Label the cited deg-4 endpoints, one per arm. Placed at the SAME anchor in both
# facets (top-RIGHT, where the region above ~0.65 is empty in both facets) so the
# labels are symmetric and never overlap the curves (the degree-1 curve spikes to
# 1.0 on the LEFT, so the top-left corner is occupied).
endpts <- data.frame(
  arm   = factor(c('Unfrozen', 'Attention frozen'),
                 levels = c('Unfrozen', 'Attention frozen')),
  x     = c(3950, 3950),
  y     = c(0.86, 0.86),
  label = c('degree-4 final +0.427',
            sprintf('degree-4 final %+.3f', final_attn_d4))
)

p_case <- ggplot(traj_df, aes(step, corr, colour = deg)) +
  geom_hline(yintercept = 0, colour = '#999999', linewidth = 0.5, linetype = 'dashed') +
  geom_line(linewidth = 1.1) +
  geom_text(data = endpts,
            aes(x = x, y = y, label = label),
            colour = 'black', size = 2.9, hjust = 1, vjust = 1, inherit.aes = FALSE) +
  facet_wrap(~ arm, nrow = 1) +
  scale_colour_manual(values = deg_cols, name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0.02, 0.04))) +
  coord_cartesian(xlim = c(0, 4000), ylim = c(-0.12, 1.05)) +
  labs(x = 'Training step', y = 'Walsh correlation by degree',
       title = '(a)-(b) Per-degree acquisition, one seed (0)') +
  paper_theme(9) +
  theme(panel.grid.major.x = element_line(linewidth = 0.2, colour = '#e6e9ee'))

# (c) degree-4 zoom: the blocked degree in detail (onset and plateau)
traj_d4 <- traj_df %>% filter(deg == 'degree 4')
p_case_c <- ggplot(traj_d4, aes(step, corr, colour = arm)) +
  geom_hline(yintercept = 0, colour = '#999999', linewidth = 0.5, linetype = 'dashed') +
  geom_line(linewidth = 1.1) +
  scale_colour_manual(values = c('Unfrozen' = '#000000', 'Attention frozen' = '#7B3294'),
                      name = 'Freeze arm') +
  coord_cartesian(ylim = c(-0.12, 0.65)) +
  labs(x = 'Training step', y = 'Degree-4 Walsh correlation',
       # 3-across row: this panel is ~1.5in wide; a single-line title overflows
       # right and overprints panel (d)'s title. Two lines keep it in-bounds
       # (plain element_text: \n wraps on both tikz and cairo tiers).
       title = '(c) Degree-4 zoom\n(the blocked degree)') +
  paper_theme(9) +
  theme(legend.position = 'bottom',
        plot.title = element_text(lineheight = 0.9))

# (d) per-seed finals across all 15 seeds: the freeze block on degree 4 is
#     seed-consistent; degrees 1-3 are matched across arms
seed_rows <- bind_rows(lapply(0:14, function(s) {
  bind_rows(lapply(c(none = 'Unfrozen', attn = 'Attention frozen'), function(arm_lab) {
    tag <- names(which(c(none = 'Unfrozen', attn = 'Attention frozen') == arm_lab))
    f <- file.path(base_as, sprintf('d128_l2_h4_%s_s%d.jsonl', tag, s))
    if (!file.exists(f)) return(NULL)
    tr <- read_traj_alldeg(f)
    if (!nrow(tr)) return(NULL)
    tr %>% group_by(deg) %>% summarise(final = tail(corr, 1), .groups = 'drop') %>%
      mutate(arm = arm_lab, seed = s)
  }))
})) %>%
  mutate(arm = factor(arm, levels = c('Unfrozen', 'Attention frozen')),
         deg = factor(deg, levels = 1:4,
                      labels = c('degree 1', 'degree 2', 'degree 3', 'degree 4')))
p_case_d <- ggplot(seed_rows, aes(arm, final, colour = arm)) +
  stat_summary(fun = median, geom = 'crossbar', width = 0.45, colour = '#1a1a1a') +
  geom_point(position = position_jitter(width = 0.05, height = 0, seed = 41),
             size = 1.5, alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~ deg, nrow = 1) +
  scale_colour_manual(values = c('Unfrozen' = '#000000', 'Attention frozen' = '#7B3294'),
                      guide = 'none') +
  labs(x = NULL, y = 'Final Walsh correlation',
       title = '(d) Per-seed finals (n = 15)') +
  paper_theme(9) +
  theme(axis.text.x = element_text(size = 6.8, angle = 20, hjust = 1))

save_both((p_case | p_case_c | p_case_d) + plot_layout(nrow = 1, widths = c(2.2, 1, 1.6)),
          'C_case', 7.2, 3.6)

# ════════════════════════════════════════════════════════════════════════════
# 2. C_dissoc — 4-panel composite dissociation figure
# Data: tost_si.json, arch_staircase/d128_l2_h4_*.jsonl, induction_verdicts.json,
#       subspace_verdicts.json
# ════════════════════════════════════════════════════════════════════════════
cat('2. C_dissoc\n')

# Panel (a): SI equivalence strip-chart (TOST result)
tost <- read_json_r(file.path(res_dir, 'figures-004', 'tost_si.json'),
                    simplifyVector = FALSE)
lin <- tost$tests$muon_vs_adamw$linear

muon_si  <- as.numeric(unlist(tost$si_per_seed$muon))
adamw_si <- as.numeric(unlist(tost$si_per_seed$adamw))
si_df <- bind_rows(
  data.frame(opt = 'Muon',  si = muon_si),
  data.frame(opt = 'AdamW', si = adamw_si)
) %>% mutate(opt = factor(opt, levels = c('Muon', 'AdamW')))
means_si <- si_df %>% group_by(opt) %>% summarise(m = mean(si), .groups = 'drop')

# CI/delta annotation string
ci <- as.numeric(unlist(lin$ci90))
ann_label <- sprintf('Δ = %+.2f\n90%% CI [%.2f, %.2f]\np = %.3f',
                     as.numeric(lin$diff), ci[1], ci[2], as.numeric(lin$p_tost))

pA <- ggplot(si_df, aes(opt, si, colour = opt)) +
  geom_jitter(width = 0.12, height = 0, size = 2, alpha = 0.75) +
  geom_crossbar(data = means_si, aes(y = m, ymin = m, ymax = m),
                width = 0.35, linewidth = 1.2, middle.linewidth = 1.2) +
  geom_hline(yintercept = 1.0, colour = '#888888', linetype = 'dashed', linewidth = 0.6) +
  annotate('text', x = 0.55, y = 8.2,
           label = ann_label, hjust = 0, vjust = 1, size = 2.1, colour = '#1a202c') +
  scale_colour_manual(values = c('Muon' = OPT$muon, 'AdamW' = OPT$adamw), guide = 'none') +
  coord_cartesian(xlim = c(0.4, 3.0), ylim = c(-0.3, 8.4)) +
  labs(title = 'SI equivalence', x = NULL, y = 'Staircase index SI') +
  paper_theme(8) +
  theme(axis.text.x = element_text(size = 7))

# Panel (b): deg-4 final Walsh corr by freeze arm
read_final_deg4 <- function(pattern) {
  paths <- Sys.glob(pattern)
  vals  <- c()
  for (p in paths) {
    lines <- readLines(p, warn = FALSE)
    for (l in rev(lines)) {
      d <- tryCatch(jsonlite::fromJSON(l, simplifyVector = TRUE), error = function(e) NULL)
      if (!is.null(d) && '_summary' %in% names(d)) {
        v <- d[['_summary']][['final_deg_corr']][['4']]
        if (!is.null(v)) vals <- c(vals, as.numeric(v))
        break
      }
    }
  }
  vals
}

none_d4 <- read_final_deg4(file.path(base_as, 'd128_l2_h4_none_s*.jsonl'))
attn_d4 <- read_final_deg4(file.path(base_as, 'd128_l2_h4_attn_s*.jsonl'))
mlp_d4  <- read_final_deg4(file.path(base_as, 'd128_l2_h4_mlp_s*.jsonl'))

freeze_df <- bind_rows(
  data.frame(arm = 'None', deg4 = none_d4),
  data.frame(arm = 'Attn', deg4 = attn_d4),
  data.frame(arm = 'MLP',  deg4 = mlp_d4)
) %>% mutate(arm = factor(arm, levels = c('None', 'Attn', 'MLP')))
freeze_means <- freeze_df %>% group_by(arm) %>% summarise(m = mean(deg4), .groups = 'drop')
freeze_cols <- c('None' = FREEZE$none, 'Attn' = FREEZE$attn,
                 'MLP' = FREEZE$mlp)

pB <- ggplot(freeze_df, aes(arm, deg4, colour = arm)) +
  geom_jitter(width = 0.12, height = 0, size = 2, alpha = 0.65) +
  geom_crossbar(data = freeze_means, aes(y = m, ymin = m, ymax = m),
                width = 0.35, linewidth = 1.2, middle.linewidth = 1.2) +
  geom_hline(yintercept = 0, colour = '#888888', linetype = 'dashed', linewidth = 0.6) +
  scale_colour_manual(values = freeze_cols, guide = 'none') +
  coord_cartesian(ylim = c(-0.12, 0.65)) +
  labs(title = 'Freeze arms', x = NULL, y = 'Degree-4 final Walsh corr.') +
  paper_theme(8) +
  theme(axis.text.x = element_text(size = 7))

# Panel (c): emergence steps by context length (bar chart)
# M6 replot: use fine-arm per-seed emergence (induction_fine) for Muon×{64,128,256}
# and AdamW×{64,128}; AdamW L256 + SGDM from the main grid (induction_emergence).
# Identical source to C_timing / Table 2: Muon 30/40/70, AdamW 48/90/260, SGDM
# 290/480/480. This resolves the former grid-floored tie at L=64,128 with no data
# value altered (bars now show the same means already reported in Table 2).
lens <- c(64, 128, 256)
read_fine_emergence_c <- function(path) {
  lines <- readLines(path, warn = FALSE)
  for (l in rev(lines)) {
    d <- tryCatch(jsonlite::fromJSON(l, simplifyVector = TRUE), error = function(e) NULL)
    if (!is.null(d) && '_summary' %in% names(d)) {
      em <- d[['_summary']][['emergence_step']]
      if (!is.null(em) && !is.na(em)) return(as.numeric(em))
    }
  }
  NA_real_
}
c_fine_dir <- file.path(res_dir, 'induction_fine')
c_main_dir <- file.path(res_dir, 'induction_emergence')
c_main_all <- bind_rows(lapply(Sys.glob(file.path(c_main_dir, '*.jsonl')), function(p) {
  lines <- readLines(p, warn = FALSE)
  for (l in rev(lines)) {
    d <- tryCatch(jsonlite::fromJSON(l, simplifyVector = TRUE), error = function(e) NULL)
    if (!is.null(d) && '_summary' %in% names(d)) {
      s <- d[['_summary']]; em <- s[['emergence_step']]
      if (!is.null(em) && !is.na(em))
        return(data.frame(opt = s$optimizer, L = as.integer(s$seq_len),
                          emergence_step = as.numeric(em), stringsAsFactors = FALSE))
    }
  }
  NULL
})) %>% filter(!is.na(emergence_step))

emerge_mean_c <- function(opt, L) {
  use_fine <- opt %in% c('muon', 'adamw') && !(opt == 'adamw' && L == 256)
  if (use_fine) {
    vals <- c()
    for (s in 0:4) {
      fn <- file.path(c_fine_dir, sprintf('%s_L%d_fine_s%d.jsonl', opt, L, s))
      if (file.exists(fn)) vals <- c(vals, read_fine_emergence_c(fn))
    }
    vals <- vals[!is.na(vals)]
  } else {
    vals <- c_main_all %>% filter(opt == !!opt, L == !!L) %>% pull(emergence_step)
  }
  if (length(vals) == 0) NA_real_ else mean(vals)
}

emerge_df <- bind_rows(lapply(c('adamw', 'muon', 'sgdm'), function(opt) {
  data.frame(opt = opt, L = lens,
             step = sapply(lens, function(L) emerge_mean_c(opt, L)))
})) %>% mutate(opt = factor(opt, levels = c('adamw', 'muon', 'sgdm')),
               L   = factor(L))

pC <- ggplot(emerge_df, aes(L, step, fill = opt)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.72) +
  scale_fill_manual(values = c(adamw = OPT$adamw, muon = OPT$muon, sgdm = OPT$sgdm),
                    labels = c('AdamW', 'Muon', 'SGDM'), name = 'Optimizer') +
  coord_cartesian(ylim = c(0, 580)) +
  labs(title = 'ICL emergence step', x = 'Context length L', y = 'Emergence step') +
  paper_theme(8) +
  theme(axis.text.x = element_text(size = 7.5))

# Panel (d): subspace capture vs random baseline
sv   <- read_json_r(file.path(res_dir, 'figures-010', 'subspace_verdicts.json'),
                    simplifyVector = FALSE)
main_sv <- sv$main
L_names  <- names(main_sv)
cap_vals <- sapply(L_names, function(k) main_sv[[k]]$captured_mean)
rand_base <- 0.0061

sub_df <- data.frame(L = factor(L_names, levels = L_names), captured = cap_vals)

pD <- ggplot(sub_df, aes(L, captured)) +
  geom_point(colour = CB$blue, size = 3, alpha = 0.85) +
  geom_hline(yintercept = rand_base, colour = '#b91c1c', linewidth = 1.0,
             linetype = 'dashed') +
  annotate('text', x = 1.2, y = rand_base * 1.7,
           label = 'Random\nbaseline', colour = 'black', size = 2.3,
           hjust = 0, lineheight = 0.95) +
  scale_y_log10(breaks = 10^-(2:5), labels = pow10_labels) +
  coord_cartesian(ylim = c(5e-6, 1e-2)) +
  labs(title = 'Ansatz-direction capture', x = 'Context length L',
       y = 'Captured variance (log scale)') +
  paper_theme(8) +
  theme(axis.text.x = element_text(size = 6.8))

p_dissoc <- compose_C_four_panel(list(pA, pB, pC, pD),
                                 'C_dissoc', 5.0, 5.6, root)
save_both(p_dissoc, 'C_dissoc', 5.0, 5.6)

# ════════════════════════════════════════════════════════════════════════════
# 3. C_scale — staircase depth threshold at L=24/d=256
# Data: experiments/results/arch_staircase_scaleup/*.jsonl
# ════════════════════════════════════════════════════════════════════════════
cat('3. C_scale\n')

scaleup_dir <- file.path(res_dir, 'arch_staircase_scaleup')
files_su    <- Sys.glob(file.path(scaleup_dir, '*.jsonl'))

read_last_summary <- function(path) {
  lines <- readLines(path, warn = FALSE)
  for (l in rev(lines)) {
    d <- tryCatch(jsonlite::fromJSON(l, simplifyVector = TRUE), error = function(e) NULL)
    if (!is.null(d) && '_summary' %in% names(d)) return(d[['_summary']])
  }
  NULL
}

scale_rows <- lapply(files_su, function(p) {
  b <- basename(p)
  m <- regmatches(b, regexpr('^(adamw|muon)_L24_l(\\d+)_s(\\d+)', b))
  if (length(m) == 0) return(NULL)
  parts <- strsplit(m, '_')[[1]]
  opt   <- parts[1]
  nl    <- as.integer(sub('l', '', parts[3]))
  su    <- read_last_summary(p)
  if (is.null(su)) return(NULL)
  dc <- su[['final_deg_corr']]
  data.frame(opt = opt, n_layers = nl,
             deg3 = as.numeric(dc[['3']]),
             deg4 = as.numeric(dc[['4']]),
             stringsAsFactors = FALSE)
})
scale_df <- bind_rows(Filter(Negate(is.null), scale_rows))
scale_med <- scale_df %>%
  group_by(opt, n_layers) %>%
  summarise(med_deg3 = median(deg3), med_deg4 = median(deg4), .groups = 'drop') %>%
  mutate(label = paste0(c(adamw = 'AdamW', muon = 'Muon')[opt], '\n',
                         n_layers, ' layer'))

scale_long <- scale_med %>%
  pivot_longer(c(med_deg3, med_deg4), names_to = 'degree', values_to = 'corr') %>%
  mutate(degree = recode(degree, med_deg3 = 'Degree 3', med_deg4 = 'Degree 4'),
         label  = factor(label, levels = unique(label)))

# Single-panel: short title, no verbose subtitle
p_scale <- ggplot(scale_long, aes(label, corr, fill = degree)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7, alpha = 0.88) +
  scale_fill_manual(values = c('Degree 3' = CB$grey, 'Degree 4' = CB$vermillion),
                    name = NULL) +
  labs(title = '(a) Final Walsh correlation (L=24, d=256)',
       x = NULL, y = 'Final Walsh correlation') +
  paper_theme(10)

# (b) per-seed strips behind the (a) medians
scale_seed_long <- scale_df %>%
  pivot_longer(c(deg3, deg4), names_to = 'degree', values_to = 'corr') %>%
  mutate(degree = recode(degree, deg3 = 'Degree 3', deg4 = 'Degree 4'),
         opt = recode(opt, adamw = 'AdamW', muon = 'Muon'),
         nl_f = factor(paste0(n_layers, 'L'), levels = c('1L', '2L')))
p_scale_b <- ggplot(scale_seed_long, aes(nl_f, corr, colour = degree)) +
  stat_summary(fun = median, geom = 'crossbar', width = 0.5, colour = '#1a1a1a',
               position = position_dodge(width = 0.7)) +
  geom_point(position = position_jitterdodge(jitter.width = 0.07, dodge.width = 0.7,
                                             seed = 43),
             size = 1.5, alpha = 0.8) +
  facet_wrap(~ opt) +
  scale_colour_manual(values = c('Degree 3' = CB$grey, 'Degree 4' = CB$vermillion),
                      name = NULL) +
  labs(title = '(b) Per-seed correlations (L=24, d=256)',
       x = NULL, y = 'Final Walsh correlation') +
  paper_theme(9) +
  theme(legend.position = 'top')

# (c) the 24-configuration scale grid: final degree-4 correlation per config
#     (L x d x depth x optimizer), from the scalegrid arm summaries
grid_dir <- file.path(res_dir, 'arch_staircase_scale')
grid_rows <- lapply(Sys.glob(file.path(grid_dir, 'scalegrid_*.jsonl')), function(p) {
  b <- basename(p)
  m <- regmatches(b, regexpr('^scalegrid_L(\\d+)_d(\\d+)_l(\\d+)_fnone_(adamw|muon)', b))
  if (length(m) == 0) return(NULL)
  parts <- strsplit(m, '_')[[1]]
  su <- read_last_summary(p)
  if (is.null(su)) return(NULL)
  dc <- su[['final_deg_corr']]
  data.frame(L = as.integer(sub('L', '', parts[2])),
             d = as.integer(sub('d', '', parts[3])),
             n_layers = as.integer(sub('l', '', parts[4])),
             opt = parts[5],
             deg4 = as.numeric(dc[['4']]), stringsAsFactors = FALSE)
})
grid_df <- bind_rows(Filter(Negate(is.null), grid_rows)) %>%
  group_by(L, d, n_layers, opt) %>%
  summarise(med_deg4 = median(deg4), .groups = 'drop') %>%
  mutate(config = factor(sprintf('L%d,d%d,%dL', L, d, n_layers)),
         opt = recode(opt, adamw = 'AdamW', muon = 'Muon'))
p_scale_c <- ggplot(grid_df, aes(config, med_deg4, fill = opt)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.7) +
  scale_fill_manual(values = c(AdamW = OPT$adamw, Muon = OPT$muon), name = NULL) +
  labs(title = '(c) Scale grid: degree-4 by configuration',
       x = NULL, y = 'Median degree-4 corr.') +
  paper_theme(9) +
  theme(axis.text.x = element_text(size = 6.2, angle = 45, hjust = 1),
        legend.position = 'top')

# (d) the L=24 freeze audit (8 seeds): attention-freeze kills degree 4 under
#     AdamW (8/8 fail) but RESCUES it under Muon (rank index +1.000, 8/8)
frz_rows <- lapply(Sys.glob(file.path(grid_dir, 'freeze8_*.jsonl')), function(p) {
  b <- basename(p)
  m <- regmatches(b, regexpr('^freeze8_L24_d256_l2_fattn_(adamw|muon)_s(\\d+)', b))
  if (length(m) == 0) return(NULL)
  parts <- strsplit(m, '_')[[1]]
  su <- read_last_summary(p)
  if (is.null(su)) return(NULL)
  dc <- su[['final_deg_corr']]
  data.frame(opt = parts[2], seed = as.integer(parts[3]),
             deg4 = as.numeric(dc[['4']]), stringsAsFactors = FALSE)
})
frz_df <- bind_rows(Filter(Negate(is.null), frz_rows)) %>%
  mutate(opt = recode(opt, adamw = 'AdamW', muon = 'Muon'))
p_scale_d <- ggplot(frz_df, aes(opt, deg4, colour = opt)) +
  geom_hline(yintercept = 0.3, linetype = 'dashed', colour = '#888', linewidth = 0.6) +
  stat_summary(fun = median, geom = 'crossbar', width = 0.45, colour = '#1a1a1a') +
  geom_point(position = position_jitter(width = 0.06, height = 0, seed = 43),
             size = 2.0, alpha = 0.85, show.legend = FALSE) +
  scale_colour_manual(values = c(AdamW = OPT$adamw, Muon = OPT$muon), guide = 'none') +
  labs(title = '(d) L=24 attn-freeze audit (8 seeds): reversal',
       x = NULL, y = 'Degree-4 corr. (attn frozen)') +
  paper_theme(9)

save_both((p_scale | p_scale_b | p_scale_c | p_scale_d) +
            plot_layout(nrow = 2) & theme(legend.position = 'top'),
          'C_scale', 7.2, 5.4)

# ════════════════════════════════════════════════════════════════════════════
# 4. C_ownership_lr_sensitivity — Muon lr sweep: attn freeze vs unfrozen
# Data: experiments/results/arch_staircase_optaxis_lrsweep/lrsweep_verdict.json
# ════════════════════════════════════════════════════════════════════════════
cat('4. C_ownership_lr_sensitivity\n')

lrsweep <- read_json_r(
  file.path(res_dir, 'arch_staircase_optaxis_lrsweep', 'lrsweep_verdict.json'),
  simplifyVector = FALSE)

lr_rows <- lapply(lrsweep$by_lr, function(row) {
  lr  <- as.numeric(row$muon_lr)
  bind_rows(lapply(c('none', 'attn'), function(fz) {
    m <- row[[fz]]
    data.frame(
      muon_lr     = lr,
      freeze      = fz,
      med_deg4    = as.numeric(m$median_deg4),
      q25         = as.numeric(m$q25_deg4),
      q75         = as.numeric(m$q75_deg4),
      vals        = I(list(as.numeric(unlist(m$deg4_values))))
    )
  }))
})
lr_df <- bind_rows(lr_rows) %>%
  mutate(freeze = factor(freeze, levels = c('none', 'attn'),
                         labels = c('Unfrozen', 'Attn frozen')),
         lr_label = factor(sprintf('%.2g', muon_lr),
                           levels = sprintf('%.2g', sort(unique(muon_lr)))))

# Scatter of raw seed values
lr_scatter <- bind_rows(mapply(function(row) {
  lr <- as.numeric(row$muon_lr)
  bind_rows(lapply(c('none', 'attn'), function(fz) {
    vals <- as.numeric(unlist(row[[fz]]$deg4_values))
    data.frame(muon_lr  = lr,
               lr_label = sprintf('%.2g', lr),
               freeze   = factor(ifelse(fz == 'none', 'Unfrozen', 'Attn frozen'),
                                  levels = c('Unfrozen', 'Attn frozen')),
               deg4 = vals)
  }))
}, lrsweep$by_lr, SIMPLIFY = FALSE))

deltas <- lr_df %>%
  pivot_wider(id_cols = lr_label, names_from = freeze, values_from = med_deg4) %>%
  mutate(delta = `Attn frozen` - Unfrozen, y_pos = 0.65)

# Single-panel: short title, verbose subtitle moved to caption
p_lr <- ggplot(lr_df, aes(lr_label, med_deg4, colour = freeze, group = freeze)) +
  geom_hline(yintercept = 0, colour = '#d1d5db', linewidth = 0.5) +
  geom_errorbar(aes(ymin = q25, ymax = q75), width = 0.18,
                position = position_dodge(width = 0.35), linewidth = 0.45) +
  geom_line(position = position_dodge(width = 0.35), linewidth = 1.2) +
  geom_point(position = position_dodge(width = 0.35), size = 3) +
  geom_point(data = lr_scatter,
             aes(lr_label, deg4, colour = freeze),
             size = 1.6, alpha = 0.45,
             position = position_jitterdodge(jitter.width = 0.08, dodge.width = 0.35)) +
  geom_text(data = deltas,
            aes(lr_label, pmin(y_pos, 0.64), label = sprintf('Δ=%+.3f', delta),
                colour = NULL),
            colour = '#374151', size = 2.7, inherit.aes = FALSE) +
  scale_colour_manual(values = c('Unfrozen' = FREEZE$none, 'Attn frozen' = FREEZE$attn),
                      name = NULL) +
  coord_cartesian(ylim = c(-0.08, 0.72)) +
  labs(title = '(a) Muon attn-freeze: degree-4 across learning rates',
       x = 'Muon learning rate', y = 'Final degree-4 Walsh corr. (median ± IQR)') +
  paper_theme(9)

# (b) per-seed values at the native lr (n = 3/arm, seeds 5-7)
native_lr <- lr_scatter$muon_lr[1]
p_lr_b <- ggplot(lr_scatter, aes(lr_label, deg4, colour = freeze)) +
  stat_summary(fun = median, geom = 'crossbar', width = 0.5, colour = '#1a1a1a',
               position = position_dodge(width = 0.6)) +
  geom_point(position = position_jitterdodge(jitter.width = 0.06, dodge.width = 0.6,
                                             seed = 47),
             size = 1.8, alpha = 0.85) +
  scale_colour_manual(values = c('Unfrozen' = FREEZE$none, 'Attn frozen' = FREEZE$attn),
                      name = NULL) +
  labs(title = '(b) Per-seed degree-4 by lr',
       x = 'Muon learning rate', y = 'Final degree-4 Walsh corr.') +
  paper_theme(9) +
  theme(legend.position = 'top')

# (c) paired delta (frozen - unfrozen) per lr with seed-level spread
delta_seeds <- lr_scatter %>%
  pivot_wider(id_cols = c(muon_lr, lr_label), names_from = freeze, values_from = deg4,
              values_fn = median) %>%
  mutate(delta = `Attn frozen` - Unfrozen)
p_lr_c <- ggplot(delta_seeds, aes(lr_label, delta)) +
  geom_hline(yintercept = 0, colour = '#888', linetype = 'dashed', linewidth = 0.6) +
  geom_point(size = 2.6, colour = '#000000') +
  geom_line(aes(group = 1), colour = '#000000', linewidth = 0.8) +
  labs(title = '(c) Freeze delta per lr (median-of-seeds)',
       x = 'Muon learning rate', y = 'Δ degree-4 (frozen - unfrozen)') +
  paper_theme(9)

# (d) verdict robustness: sign and size of the delta hold across the lr grid —
#     the two-path redundancy is not a single-lr artifact
verdict_df <- deltas %>% mutate(robust = delta > 0.1)
p_lr_d <- ggplot(deltas, aes(lr_label, delta)) +
  geom_hline(yintercept = 0, colour = '#888', linetype = 'dashed', linewidth = 0.6) +
  geom_col(fill = CB$sky, width = 0.55) +
  geom_text(aes(label = sprintf('%+.2f', delta)), vjust = -0.4, size = 2.7) +
  labs(title = '(d) Delta robust across the lr grid',
       x = 'Muon learning rate', y = 'Δ degree-4 (median)') +
  paper_theme(9)

save_both((p_lr | p_lr_b | p_lr_c | p_lr_d) + plot_layout(nrow = 2) &
            theme(legend.position = 'top'),
          'C_ownership_lr_sensitivity', 7.2, 5.4)

# ════════════════════════════════════════════════════════════════════════════
# 5. C_ownership_optaxis — 017 freeze deltas across optimizers
# Data: experiments/results/figures-017/optaxis_verdict.json
# ════════════════════════════════════════════════════════════════════════════
cat('5. C_ownership_optaxis\n')

oax <- read_json_r(file.path(res_dir, 'figures-017', 'optaxis_verdict.json'),
                   simplifyVector = FALSE)
fr  <- oax$freeze_deltas_deg4
opts_oa <- c('adamw', 'muon', 'sgdm')

oa_rows <- bind_rows(lapply(opts_oa, function(o) {
  bind_rows(lapply(c('none', 'attn', 'mlp'), function(fz) {
    data.frame(opt   = o,
               freeze = fz,
               deg4  = as.numeric(fr[[o]][[paste0(fz, '_deg4')]]))
  }))
})) %>%
  mutate(opt    = factor(opt, levels = c('adamw', 'muon', 'sgdm'),
                          labels = c('AdamW', 'Muon', 'SGDM')),
         freeze = factor(freeze, levels = c('none', 'attn', 'mlp'),
                          labels = c('None', 'Attn frozen', 'MLP frozen')))

delta_df <- data.frame(
  opt   = factor(c('AdamW', 'Muon', 'SGDM'), levels = c('AdamW', 'Muon', 'SGDM')),
  delta = sapply(opts_oa, function(o) {
    v <- fr[[o]][['delta_attn_minus_none']]
    if (is.null(v)) NA else as.numeric(v)
  }),
  y_pos = max(sapply(opts_oa, function(o) as.numeric(fr[[o]]$none_deg4))) +
    c(0.10, 0.03, 0.10)   # stagger: adjacent same-height labels would touch
)

# Mark the attn-frozen bars that are measured ~0 (AdamW -0.0003, SGDM -0.0017):
# these render as invisible zero-height bars and could read as missing data, when
# in fact for AdamW the collapse to ~0 IS the result. Annotate with a "0.00" label
# and a small baseline tick at the attn bar's dodged x-position. (Muon's attn bar
# is ~0.50 and renders normally, so it gets no marker.)
attn_dodge <- 0.72 / 3   # bar-centre offset for the middle (attn) bar is ~0
zero_attn <- oa_rows %>%
  filter(freeze == 'Attn frozen', abs(deg4) < 0.02) %>%
  mutate(xpos = as.integer(opt))   # attn is the centre bar => no horizontal shift

# Single-panel: short title, subtitle removed (moved to caption)
p_optax <- ggplot(oa_rows, aes(opt, deg4, fill = freeze)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.72, alpha = 0.88) +
  geom_hline(yintercept = 0, colour = '#374151', linewidth = 0.5) +
  # baseline tick + label so measured-zero attn bars are not read as missing
  geom_segment(data = zero_attn,
               aes(x = xpos - 0.10, xend = xpos + 0.10, y = 0, yend = 0),
               inherit.aes = FALSE, colour = FREEZE$attn, linewidth = 1.6) +
  geom_text(data = zero_attn,
            aes(x = xpos, y = 0.035, label = '0.00'),
            inherit.aes = FALSE, colour = 'black', size = 2.5,
            vjust = 0) +
  geom_text(data = delta_df,
            aes(opt, y_pos, label = sprintf('Δattn=%+.3f', delta)),
            colour = '#374151', size = 2.2, inherit.aes = FALSE) +
  scale_fill_manual(values = c('None' = FREEZE$none, 'Attn frozen' = FREEZE$attn,
                                'MLP frozen' = FREEZE$mlp), name = NULL) +
  coord_cartesian(ylim = c(-0.06, 0.68)) +
  labs(title = '(a) Degree-4 final Walsh correlation by freeze arm and optimizer',
       x = NULL, y = 'Degree-4 final Walsh correlation') +
  paper_theme(9) +
  theme(legend.position = 'top',
        legend.box.spacing = unit(1, 'pt'),
        legend.margin      = margin(0, 0, 0, 0),
        plot.margin        = margin(2, 6, 5.5, 6))

# (b) per-seed degree-4 by freeze arm and optimizer (raw jsonl).
# Data routing: AdamW runs (15 seeds) live in arch_staircase/ with NO
# optimizer suffix; the Muon/SGDM optaxis runs (5 seeds each) live in
# arch_staircase_optaxis/ with a _muon/_sgdm suffix. Loading all three from
# one directory pattern silently drops AdamW.
oax_dir <- file.path(res_dir, 'arch_staircase_optaxis')
arch_dir <- file.path(res_dir, 'arch_staircase')
oax_seeds <- bind_rows(lapply(opts_oa, function(o) {
  base_dir <- if (o == 'adamw') arch_dir else oax_dir
  seed_range <- if (o == 'adamw') 0:14 else 0:4
  bind_rows(lapply(c('none', 'attn', 'mlp'), function(fz) {
    bind_rows(lapply(seed_range, function(s) {
      fname <- if (o == 'adamw') {
        sprintf('d128_l2_h4_%s_s%d.jsonl', fz, s)
      } else {
        sprintf('d128_l2_h4_%s_s%d_%s.jsonl', fz, s, o)
      }
      f <- file.path(base_dir, fname)
      if (!file.exists(f)) return(NULL)
      su <- read_last_summary(f)
      if (is.null(su)) return(NULL)
      dc <- su[['final_deg_corr']]
      data.frame(opt = o, freeze = fz, seed = s,
                 deg4 = as.numeric(dc[['4']]), stringsAsFactors = FALSE)
    }))
  }))
})) %>%
  mutate(opt    = factor(opt, levels = c('adamw', 'muon', 'sgdm'),
                          labels = c('AdamW', 'Muon', 'SGDM')),
         freeze = factor(freeze, levels = c('none', 'attn', 'mlp'),
                          labels = c('None', 'Attn', 'MLP')))
p_optax_b <- ggplot(oax_seeds, aes(freeze, deg4, colour = freeze)) +
  stat_summary(fun = median, geom = 'crossbar', width = 0.5, colour = '#1a1a1a') +
  geom_point(position = position_jitter(width = 0.06, height = 0, seed = 51),
             size = 1.3, alpha = 0.75, show.legend = FALSE) +
  facet_wrap(~ opt) +
  scale_colour_manual(values = c('None' = FREEZE$none, 'Attn' = FREEZE$attn,
                                  'MLP' = FREEZE$mlp), guide = 'none') +
  labs(title = '(b) Per-seed degree-4 (AdamW n=15, Muon/SGDM n=5)',
       x = NULL, y = 'Degree-4 final Walsh corr.') +
  paper_theme(9) +
  theme(axis.text.x = element_text(size = 6.8, angle = 35, hjust = 1))

# (c) paired freeze-vs-none delta per optimizer (median over seeds) — the
#     AdamW-specificity verdict: Δattn is large-negative only under AdamW
delta_seeds <- oax_seeds %>%
  pivot_wider(id_cols = c(opt, seed), names_from = freeze, values_from = deg4,
              values_fn = median) %>%
  mutate(delta_attn = Attn - None)
p_optax_c <- ggplot(delta_seeds, aes(opt, delta_attn, colour = opt)) +
  geom_hline(yintercept = 0, colour = '#888', linetype = 'dashed', linewidth = 0.6) +
  stat_summary(fun = median, geom = 'crossbar', width = 0.45, colour = '#1a1a1a') +
  geom_point(position = position_jitter(width = 0.06, height = 0, seed = 51),
             size = 1.5, alpha = 0.8, show.legend = FALSE) +
  scale_colour_manual(values = c(AdamW = OPT$adamw, Muon = OPT$muon, SGDM = OPT$sgdm), guide = 'none') +
  labs(title = '(c) Paired Δ attn-freeze (per seed)',
       x = NULL, y = 'Δ degree-4 (frozen - none)') +
  paper_theme(9)

# (d) matched-fit control: degrees 1-3 stay acquired under every optimizer at
#     the matched point (the freeze dissociation is not a fit failure)
mp <- oax$matched_pointwise_deg_corr
mp_rows <- bind_rows(lapply(c('adamw', 'muon', 'sgdm'), function(o) {
  v <- as.numeric(mp[[o]]$deg_corr_med)
  data.frame(opt = o, degree = factor(paste0('deg ', 1:4), levels = paste0('deg ', 1:4)),
             none = v, stringsAsFactors = FALSE)
})) %>% mutate(opt = factor(opt, levels = c('adamw', 'muon', 'sgdm'),
                            labels = c('AdamW', 'Muon', 'SGDM')))
p_optax_d <- ggplot(mp_rows, aes(degree, none, fill = opt)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.7) +
  scale_fill_manual(values = c(AdamW = OPT$adamw, Muon = OPT$muon, SGDM = OPT$sgdm), name = NULL) +
  labs(title = '(d) Matched-fit control (unfrozen arms)',
       x = NULL, y = 'Walsh corr. at matched point') +
  paper_theme(9) +
  theme(legend.position = 'bottom', legend.direction = 'horizontal')

p_ownership_optaxis <- compose_C_four_panel(
  list(p_optax, p_optax_b, p_optax_c, p_optax_d),
  'C_ownership_optaxis', 7.2, 5.6, root)
save_both(p_ownership_optaxis, 'C_ownership_optaxis', 7.2, 5.6)

# ════════════════════════════════════════════════════════════════════════════
# 6. C_arch_depth — P1 width/depth axes + P2 freeze arms (3-panel)
# Data: experiments/results/figures-017/arch_verdicts.json
# ════════════════════════════════════════════════════════════════════════════
cat('6. C_arch_depth\n')

av  <- read_json_r(file.path(res_dir, 'figures-017', 'arch_verdicts.json'),
                   simplifyVector = FALSE)

width_df <- bind_rows(lapply(names(av$p1_width_axis), function(d) {
  w <- av$p1_width_axis[[d]]
  data.frame(d_model   = as.integer(d),
             rank_index = as.numeric(w$rank_index),
             rank_std   = as.numeric(w$rank_std),
             fit        = as.numeric(w$fit))
}))

depth_df <- bind_rows(lapply(names(av$p1_depth_axis), function(L) {
  a <- av$p1_depth_axis[[L]]
  data.frame(n_layers   = as.integer(L),
             rank_index = as.numeric(a$rank_index),
             rank_std   = as.numeric(a$rank_std),
             fit        = as.numeric(a$fit))
}))

freeze_labels <- c(none = 'None\n(unfrozen)', attn = 'Attn\nfrozen', mlp  = 'MLP\nfrozen')
fz_df <- bind_rows(lapply(c('none', 'attn', 'mlp'), function(fr_name) {
  fm <- av$p2_freeze_arms[[fr_name]]
  data.frame(
    freeze     = freeze_labels[fr_name],
    fit_median = as.numeric(fm$fit_median),
    rank_index = as.numeric(fm$rank_index)
  )
})) %>% pivot_longer(c(fit_median, rank_index), names_to = 'metric', values_to = 'val') %>%
  mutate(freeze = factor(freeze, levels = freeze_labels),
         metric = recode(metric, fit_median = 'Fit corr.', rank_index = 'Rank index'))

# Panels (a)(b)(c) via patchwork tags; subtitles kept as short config notes
pWidth <- ggplot(width_df, aes(d_model, rank_index)) +
  geom_errorbar(aes(ymin = rank_index - rank_std, ymax = rank_index + rank_std),
                width = 0.15, linewidth = 0.45, colour = CB$blue) +
  geom_line(colour = CB$blue, linewidth = 1.0) +
  geom_point(colour = CB$blue, size = 3) +
  scale_x_log10(breaks = c(64, 128, 256, 512)) +
  coord_cartesian(ylim = c(-1.05, 1.25)) +
  labs(title = 'Width axis',
       x = expression(d[model]), y = 'Staircase rank index') +
  paper_theme(8)

pDepth <- ggplot(depth_df, aes(n_layers, rank_index)) +
  geom_errorbar(aes(ymin = rank_index - rank_std, ymax = rank_index + rank_std),
                width = 0.15, linewidth = 0.45, colour = CB$orange) +
  geom_line(colour = CB$orange, linewidth = 1.0) +
  geom_point(colour = CB$orange, size = 3) +
  scale_x_continuous(breaks = c(1, 2, 4)) +
  coord_cartesian(ylim = c(-1.05, 1.25)) +
  labs(title = 'Depth axis',
       x = 'Number of layers', y = 'Staircase rank index') +
  paper_theme(8)

# Explicit "measured = 0" marker for the attn-frozen rank-index bar: its value is
# ~7e-18 (machine zero), so the dodged bar is invisible and could read as missing
# data. Place a small tick + "0.00 (no staircase order)" label at the bar's x slot.
attn_rank_x   <- which(levels(fz_df$freeze) == freeze_labels['attn'])  # group index
rank_dodge    <- 0.72 / 4                                              # half bar offset
zero_marker <- data.frame(
  x     = attn_rank_x + rank_dodge
)

# Label ALL three rank-index bars with their value (symmetric, black, plain) so the
# measured-zero attn bar is decodable without a one-off coloured note. The
# "no staircase order" reading of the 0.00 bar is moved to the caption.
rank_labels <- fz_df %>%
  filter(metric == 'Rank index') %>%
  mutate(x     = as.integer(freeze) + rank_dodge,
         label = sprintf('%.2f', val))

pFreeze <- ggplot(fz_df, aes(freeze, val, fill = metric)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.72, alpha = 0.88) +
  geom_hline(yintercept = 0, colour = '#374151', linewidth = 0.4) +
  # tiny visible baseline tick so the measured-zero bar is not read as missing
  geom_segment(data = zero_marker,
               aes(x = x - 0.09, xend = x + 0.09, y = 0, yend = 0),
               inherit.aes = FALSE, colour = '#374151', linewidth = 1.4) +
  geom_text(data = rank_labels, aes(x = x, y = val, label = label),
            inherit.aes = FALSE, colour = 'black', size = 2.6, vjust = -0.5) +
  scale_fill_manual(values = c('Fit corr.' = CB$green, 'Rank index' = CB$vermillion),
                    name = NULL) +
  coord_cartesian(ylim = c(0, 1.05), clip = 'off') +
  labs(title = 'Freeze arms',
       x = NULL, y = 'Fit corr. / rank index') +
  paper_theme(8) +
  theme(legend.position = 'bottom')

# Use patchwork tags for (a)(b)(c); remove big composite title
# (d) fit is NOT the threshold driver: overall fit stays high at L=1 where the
#     staircase order collapses — the depth threshold lives in the ordering,
#     not in the fit
pFit <- ggplot(depth_df, aes(n_layers, fit)) +
  geom_errorbar(aes(ymin = fit - rank_std, ymax = fit + rank_std),
                width = 0.15, linewidth = 0.45, colour = CB$green) +
  geom_line(colour = CB$green, linewidth = 1.0) +
  geom_point(colour = CB$green, size = 3) +
  scale_x_continuous(breaks = c(1, 2, 4)) +
  coord_cartesian(ylim = c(0, 1.05)) +
  labs(title = 'Fit vs depth (control)',
       x = 'Number of layers', y = 'Final fit correlation') +
  paper_theme(8)

p_arch <- compose_C_four_panel(list(pWidth, pDepth, pFreeze, pFit),
                               'C_arch_depth', 7.2, 5.4, root)
save_both(p_arch, 'C_arch_depth', 7.2, 5.4)

# ════════════════════════════════════════════════════════════════════════════
# 7. C_staircase_rho — degree-k |corr| trajectories (seed-mean, 3 optimizers)
# Data: experiments/results/degree_staircase/{opt}_staircase_s*.jsonl
# ════════════════════════════════════════════════════════════════════════════
cat('7. C_staircase_rho\n')

deg_dir <- file.path(res_dir, 'degree_staircase')
DEGREES <- c(1, 2, 3, 4)

load_rho_trajectories <- function(opt) {
  paths <- Sys.glob(file.path(deg_dir, paste0(opt, '_staircase_s*.jsonl')))
  seed_trajs <- lapply(paths, function(p) {
    lines <- readLines(p, warn = FALSE)
    rows  <- lapply(lines, function(l) {
      d <- tryCatch(jsonlite::fromJSON(l, simplifyVector = TRUE), error = function(e) NULL)
      if (is.null(d) || !('step' %in% names(d))) return(NULL)
      dc <- d[['deg_corr']]
      row <- data.frame(step = d$step)
      for (k in DEGREES) row[[paste0('d', k)]] <- abs(as.numeric(dc[[as.character(k)]]))
      row
    })
    bind_rows(Filter(Negate(is.null), rows))
  })
  # align by step index then compute seed mean
  min_len <- min(sapply(seed_trajs, nrow))
  arr     <- lapply(seed_trajs, function(t) t[seq_len(min_len), ])
  steps   <- arr[[1]]$step
  mats    <- lapply(arr, function(t) as.matrix(t[, -1]))
  means   <- Reduce('+', mats) / length(mats)
  colnames(means) <- paste0('d', DEGREES)
  cbind(data.frame(step = steps, opt = opt), as.data.frame(means))
}

rho_list <- lapply(c('adamw', 'sgdm', 'muon'), load_rho_trajectories)
rho_df   <- bind_rows(rho_list) %>%
  pivot_longer(cols = starts_with('d'), names_to = 'degree', values_to = 'corr') %>%
  mutate(degree = factor(sub('d', 'deg ', degree), levels = paste0('deg ', DEGREES)),
         opt    = factor(opt, levels = c('adamw', 'sgdm', 'muon'),
                          labels = c('AdamW', 'SGDM', 'Muon')))

deg_ls <- c('deg 1' = 'solid', 'deg 2' = 'dashed', 'deg 3' = 'dotdash', 'deg 4' = 'dotted')

# Build one ggplot per optimizer; three near-identical panels are laid out as a
# single ROW sharing one Degree legend (guides='collect'). Inner y-axis titles
# dropped so the narrow panels keep enough plot width; the y scale is shared.
make_rho_panel <- function(opt_label, opt_colour, show_ylab = TRUE) {
  df_sub <- rho_df %>% filter(opt == opt_label)
  ggplot(df_sub, aes(step, corr, linetype = degree)) +
    geom_line(linewidth = 0.85, colour = opt_colour) +
    scale_linetype_manual(values = deg_ls, name = 'Degree') +
    # neutral legend-key colour so the three per-optimizer panels emit an
    # IDENTICAL Degree guide -> guides='collect' merges them into one legend
    guides(linetype = guide_legend(
             override.aes = list(colour = '#333333'), nrow = 1)) +
    coord_cartesian(ylim = c(0, 1.0)) +
    labs(title = opt_label, x = 'Training step',
         y = if (show_ylab) '|Walsh correlation|' else NULL) +
    paper_theme(9)
}

p_rho_adamw <- make_rho_panel('AdamW', OPT$adamw, show_ylab = TRUE)
p_rho_sgdm  <- make_rho_panel('SGDM',  OPT$sgdm,  show_ylab = FALSE)
p_rho_muon  <- make_rho_panel('Muon',  OPT$muon,  show_ylab = FALSE)

# (d) onset-step distribution per degree and optimizer (step at which each
#     degree's |corr| first crosses 0.3, per seed) — the staircase ORDER is
#     optimizer-invariant while the onset TIMING shifts
onset_rows <- bind_rows(lapply(c('adamw', 'sgdm', 'muon'), function(opt) {
  paths <- Sys.glob(file.path(deg_dir, paste0(opt, '_staircase_s*.jsonl')))
  bind_rows(lapply(paths, function(p) {
    lines <- readLines(p, warn = FALSE)
    rows <- lapply(lines, function(l) {
      d <- tryCatch(jsonlite::fromJSON(l, simplifyVector = TRUE), error = function(e) NULL)
      if (is.null(d) || !('step' %in% names(d))) return(NULL)
      dc <- d[['deg_corr']]
      data.frame(step = d$step,
                 d1 = abs(as.numeric(dc[['1']])), d2 = abs(as.numeric(dc[['2']])),
                 d3 = abs(as.numeric(dc[['3']])), d4 = abs(as.numeric(dc[['4']])))
    })
    tr <- bind_rows(Filter(Negate(is.null), rows))
    if (!nrow(tr)) return(NULL)
    bind_rows(lapply(1:4, function(k) {
      col <- paste0('d', k)
      hit <- tr$step[tr[[col]] >= 0.3]
      data.frame(opt = opt, degree = factor(paste0('deg ', k), levels = paste0('deg ', 1:4)),
                 onset = if (length(hit)) min(hit) else NA_real_)
    }))
  }))
})) %>% mutate(opt = factor(opt, levels = c('adamw', 'sgdm', 'muon'),
                            labels = c('AdamW', 'SGDM', 'Muon')))
p_rho_d <- ggplot(onset_rows, aes(degree, onset, colour = opt)) +
  stat_summary(fun = median, geom = 'crossbar', width = 0.5, colour = '#1a1a1a',
               position = position_dodge(width = 0.7)) +
  geom_point(position = position_jitterdodge(jitter.width = 0.06, dodge.width = 0.7,
                                             seed = 53),
             size = 1.3, alpha = 0.7) +
  scale_colour_manual(values = c(AdamW = OPT$adamw, SGDM = OPT$sgdm, Muon = OPT$muon),
                      name = NULL) +
  labs(title = 'Onset step per degree (|corr| ≥ 0.3)',
       x = NULL, y = 'Onset step') +
  paper_theme(9) +
  theme(legend.position = 'bottom', legend.direction = 'horizontal')

p_rho <- compose_C_four_panel(
  list(p_rho_adamw, p_rho_sgdm, p_rho_muon, p_rho_d),
  'C_staircase_rho', 7.2, 5.4, root)
save_both(p_rho, 'C_staircase_rho', 7.2, 5.4)

# ════════════════════════════════════════════════════════════════════════════
# 8. C_staircase_si — SI scatter/strip by optimizer
# Data: experiments/results/figures-004/staircase_verdicts.json
# ════════════════════════════════════════════════════════════════════════════
cat('8. C_staircase_si\n')

sv4 <- read_json_r(file.path(res_dir, 'figures-004', 'staircase_verdicts.json'),
                   simplifyVector = FALSE)

si_strip_rows <- bind_rows(lapply(c('muon', 'adamw', 'sgdm'), function(opt) {
  si_vals <- as.numeric(unlist(sv4[[opt]]$SI_per_seed))
  data.frame(opt = opt, si = si_vals)
})) %>% mutate(opt = factor(opt, levels = c('adamw', 'sgdm', 'muon'),
                              labels = c('AdamW', 'SGDM', 'Muon')))

si_means <- si_strip_rows %>% group_by(opt) %>% summarise(m = mean(si), .groups = 'drop')

# Single-panel: short title; verbose mean values moved to caption
p_si <- ggplot(si_strip_rows, aes(opt, si, colour = opt)) +
  geom_jitter(width = 0.12, height = 0, size = 2.5, alpha = 0.8) +
  geom_crossbar(data = si_means, aes(y = m, ymin = m, ymax = m),
                width = 0.35, linewidth = 1.4, middle.linewidth = 1.4) +
  geom_hline(yintercept = 1.0, colour = '#888888', linetype = 'dashed', linewidth = 0.7) +
  annotate('text', x = 2.55, y = 0.55, label = 'SI = 1 (simultaneous)',
           colour = 'black', size = 2.4, hjust = 0.5, family = 'TeX Gyre Termes') +
  scale_colour_manual(values = c('AdamW' = OPT$adamw, 'SGDM' = OPT$sgdm,
                                  'Muon'  = OPT$muon), guide = 'none') +
  scale_y_continuous(breaks = seq(0, 8, 2)) +
  coord_cartesian(ylim = c(0, 8)) +
  labs(title = 'Staircase index SI by optimizer',
       x = NULL, y = 'Staircase index SI') +
  paper_theme(9)
# ── 4-panel composite: (a) SI strip, (b) TOST primary, (c) point-null robustness,
#    (d) median-vs-mean robustness. TOST = pre-specified primary; one-sample = secondary.
sii <- fromJSON(file.path(res_dir, 'figures-004', 'si_inference.json'),
                simplifyVector = FALSE)

# (a) SI strip chart (existing panel, unchanged content)
p_a <- p_si + labs(title = 'SI by optimizer') + paper_theme(8.5)

# (b) TOST equivalence: three pairwise diffs with 90% CI vs ±1.5 / ±2.0 margins.
tost15 <- sii$tost_margin_1.5
pair_lab <- c(muon_vs_adamw = 'Muon - AdamW', adamw_vs_sgdm = 'AdamW - SGDM',
              muon_vs_sgdm = 'Muon - SGDM')
tost_df <- bind_rows(lapply(names(pair_lab), function(k) {
  v <- tost15[[k]]
  data.frame(pair = pair_lab[[k]], diff = v$diff, lo = v$ci_lo, hi = v$ci_hi,
             p15 = v$p_tost)
}))
# corrected verdicts from the sweep: all three pass at 2.0; only AdamW–SGDM at 1.5 (Bonferroni)
tost_df$verdict <- c('2.0 only', '1.5 + 2.0', '2.0 only')
tost_df$pair <- factor(tost_df$pair, levels = unname(pair_lab))
p_b <- ggplot(tost_df, aes(y = pair)) +
  annotate('rect', xmin = -2, xmax = 2, ymin = -Inf, ymax = Inf, fill = '#0072B2', alpha = 0.06) +
  annotate('rect', xmin = -1.5, xmax = 1.5, ymin = -Inf, ymax = Inf, fill = '#0072B2', alpha = 0.10) +
  geom_vline(xintercept = 0, colour = '#888888', linewidth = 0.5) +
  geom_vline(xintercept = c(-1.5, 1.5), colour = '#0072B2', linetype = 'dashed', linewidth = 0.6) +
  geom_vline(xintercept = c(-2, 2), colour = '#0072B2', linetype = 'dotted', linewidth = 0.6) +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.18, linewidth = 0.7, colour = 'black') +
  geom_point(aes(x = diff), size = 2.6, colour = 'black') +
  geom_text(aes(x = hi, label = verdict), hjust = -0.2, size = 2.3,
            family = 'TeX Gyre Termes', colour = '#1a202c') +
  coord_cartesian(xlim = c(-2.4, 3.6)) +
  labs(title = 'TOST equivalence (primary)', x = 'Pairwise SI difference', y = NULL) +
  paper_theme(8.5)

# (c) Point-null robustness: per-optimizer mean ± SD vs SI = 1, one-sample t annotated.
os <- sii$one_sample_vs_1
os_df <- bind_rows(lapply(c('adamw', 'sgdm', 'muon'), function(o) {
  v <- os[[o]]
  data.frame(opt = o, mean = v$mean, sd = v$sd_ddof1, t = v$t)
})) %>% mutate(opt = factor(opt, levels = c('adamw', 'sgdm', 'muon'),
                            labels = c('AdamW', 'SGDM', 'Muon')))
p_c <- ggplot(os_df, aes(opt, mean, colour = opt)) +
  geom_hline(yintercept = 1.0, colour = '#888888', linetype = 'dashed', linewidth = 0.7) +
  geom_linerange(aes(ymin = mean - sd, ymax = mean + sd), linewidth = 0.7) +
  geom_point(size = 2.8) +
  geom_text(aes(label = sprintf('t(14)=%.2f', t), y = mean + sd),
            vjust = -0.6, size = 2.3, family = 'TeX Gyre Termes', colour = '#1a202c') +
  scale_colour_manual(values = c('AdamW' = OPT$adamw, 'SGDM' = OPT$sgdm, 'Muon' = OPT$muon),
                      guide = 'none') +
  coord_cartesian(ylim = c(0, 6)) +
  labs(title = 'Point-null check (secondary robustness)', x = NULL, y = 'Mean SI ± SD') +
  paper_theme(8.5)

# (d) Median vs mean robustness; flag the Muon SI = 7.35 outlier.
mm_df <- bind_rows(lapply(c('adamw', 'sgdm', 'muon'), function(o) {
  seeds <- as.numeric(unlist(sii$regression_check[[o]]$per_seed))
  data.frame(opt = o, mean = mean(seeds), median = median(seeds),
             mx = max(seeds))
})) %>% mutate(opt = factor(opt, levels = c('adamw', 'sgdm', 'muon'),
                            labels = c('AdamW', 'SGDM', 'Muon')))
mm_long <- mm_df %>% pivot_longer(c(mean, median), names_to = 'stat', values_to = 'val')
p_d <- ggplot(mm_long, aes(opt, val, shape = stat, group = stat)) +
  geom_point(aes(colour = opt), size = 2.8, position = position_dodge(width = 0.3)) +
  geom_point(data = mm_df %>% filter(opt == 'Muon'),
             aes(x = opt, y = mx), shape = 4, size = 3, colour = OPT$muon, inherit.aes = FALSE) +
  annotate('text', x = 3.35, y = 6.9, label = 'outlier 7.35',
           colour = OPT$muon, size = 2.3, hjust = 1, family = 'TeX Gyre Termes') +
  annotate('text', x = 0.62, y = 7.9, label = 'circle = mean, triangle = median',
           colour = '#1a202c', size = 2.3, hjust = 0, family = 'TeX Gyre Termes') +
  scale_colour_manual(values = c('AdamW' = OPT$adamw, 'SGDM' = OPT$sgdm, 'Muon' = OPT$muon),
                      guide = 'none') +
  scale_shape_manual(values = c(mean = 16, median = 17), guide = 'none') +
  coord_cartesian(ylim = c(0, 8.4)) +
  labs(title = 'Mean vs median (robustness)', x = NULL, y = 'SI') +
  paper_theme(8.5)

p_si4 <- compose_C_four_panel(list(p_a, p_b, p_c, p_d),
                              'C_staircase_si', 5.0, 4.8, root)
save_both(p_si4, 'C_staircase_si', 5.0, 4.8)

# ════════════════════════════════════════════════════════════════════════════
# 9. C_subspace — ansatz-direction capture (log scale) + PCA dim vs L
# Data: experiments/results/figures-010/subspace_verdicts.json
#       + individual runs for scatter
# ════════════════════════════════════════════════════════════════════════════
cat('9. C_subspace\n')

sv10 <- read_json_r(file.path(res_dir, 'figures-010', 'subspace_verdicts.json'),
                    simplifyVector = FALSE)

# Rebuild per-seed scatter from raw jsonl for subspace
sub_rawdir <- file.path(res_dir, 'induction_subspace')
sub_raw    <- lapply(Sys.glob(file.path(sub_rawdir, '*.jsonl')), function(p) {
  lines <- readLines(p, warn = FALSE)
  for (l in rev(lines)) {
    d <- tryCatch(jsonlite::fromJSON(l, simplifyVector = TRUE), error = function(e) NULL)
    if (!is.null(d) && '_summary' %in% names(d)) {
      s <- d[['_summary']]
      return(data.frame(seq_len = as.integer(s$seq_len),
                        captured = as.numeric(s$final_captured_total),
                        pca_dim  = as.numeric(s$final_pca_dim)))
    }
  }
  NULL
})
sub_raw_df <- bind_rows(Filter(Negate(is.null), sub_raw)) %>%
  mutate(L_label = paste0('L', seq_len))

rand_base_sub <- 0.0061

pSub_cap <- ggplot(sub_raw_df, aes(factor(seq_len), captured)) +
  geom_jitter(width = 0.15, height = 0, colour = CB$blue, alpha = 0.55, size = 2.4) +
  geom_hline(yintercept = rand_base_sub, colour = CB$vermillion,
             linetype = 'dashed', linewidth = 1.0) +
  annotate('text', x = length(unique(sub_raw_df$seq_len)) - 0.2,
           y = rand_base_sub * 1.5,
           label = sprintf('Random baseline\n(%.4f)', rand_base_sub),
           colour = 'black', size = 2.4, hjust = 1) +
  scale_y_log10(breaks = 10^-(2:5), labels = pow10_labels) +
  coord_cartesian(ylim = c(5e-6, 1e-2)) +
  labs(title = 'Ansatz-direction capture', x = 'Context length L',
       y = 'Captured variance (log scale)') +
  paper_theme(8)

pSub_pca <- ggplot(sub_raw_df, aes(factor(seq_len), pca_dim)) +
  geom_jitter(width = 0.15, height = 0.05, colour = CB$blue, alpha = 0.55, size = 2.4) +
  geom_hline(yintercept = 3, colour = CB$vermillion,
             linetype = 'dashed', linewidth = 1.0) +
  annotate('text', x = length(unique(sub_raw_df$seq_len)) - 0.2,
           y = 3.25, label = 'Claimed dim = 3',
           colour = 'black', size = 2.4, hjust = 1) +
  labs(title = 'Effective update dimension', x = 'Context length L',
       y = 'PCA components (90% var.)') +
  paper_theme(8)

# (c) optimizer-invariance of the null: ansatz-direction capture under all
#     three optimizers (optaxis arm) stays at the random baseline
oax_sub_dir <- file.path(res_dir, 'induction_subspace_optaxis')
oax_sub <- bind_rows(lapply(Sys.glob(file.path(oax_sub_dir, '*.jsonl')), function(p) {
  b <- basename(p)
  m <- regmatches(b, regexpr('_(adamw|muon|sgdm)\\.jsonl$', b))
  if (length(m) == 0) return(NULL)
  opt <- sub('\\.jsonl$', '', sub('.*_', '', b))
  lines <- readLines(p, warn = FALSE)
  for (l in rev(lines)) {
    d <- tryCatch(jsonlite::fromJSON(l, simplifyVector = TRUE), error = function(e) NULL)
    if (!is.null(d) && '_summary' %in% names(d)) {
      s <- d[['_summary']]
      return(data.frame(opt = opt, captured = as.numeric(s$final_captured_total)))
    }
  }
  NULL
}))
sub_all <- bind_rows(
  sub_raw_df %>% mutate(opt = 'adamw') %>% select(opt, captured),
  oax_sub) %>%
  mutate(opt = factor(opt, levels = c('adamw', 'muon', 'sgdm'),
                      labels = c('AdamW', 'Muon', 'SGDM')))
pSub_c <- ggplot(sub_all, aes(opt, captured, colour = opt)) +
  geom_hline(yintercept = rand_base_sub, colour = CB$vermillion,
             linetype = 'dashed', linewidth = 0.8) +
  stat_summary(fun = median, geom = 'crossbar', width = 0.45, colour = '#1a1a1a') +
  geom_point(position = position_jitter(width = 0.07, height = 0, seed = 57),
             size = 1.7, alpha = 0.75, show.legend = FALSE) +
  scale_colour_manual(values = c(AdamW = OPT$adamw, Muon = OPT$muon, SGDM = OPT$sgdm),
                      guide = 'none') +
  scale_y_log10(breaks = 10^-(2:5), labels = pow10_labels) +
  coord_cartesian(ylim = c(5e-6, 1e-2)) +
  labs(title = 'Capture is ~0 under every optimizer',
       x = NULL, y = 'Captured variance (log)') +
  paper_theme(8)

# (d) mapping-validity: low capture is not an artifact of a high-dimensional
#     surrogate — captured variance vs effective dimension per run
pSub_d <- ggplot(sub_raw_df, aes(pca_dim, captured)) +
  geom_hline(yintercept = rand_base_sub, colour = CB$vermillion,
             linetype = 'dashed', linewidth = 0.8) +
  geom_point(colour = CB$blue, alpha = 0.6, size = 2.0) +
  scale_y_log10(breaks = 10^-(2:5), labels = pow10_labels) +
  labs(title = 'Capture vs surrogate dimension',
       x = 'PCA components (90% var.)', y = 'Captured variance (log)') +
  paper_theme(8)

# 2x2 composite with patchwork tags; remove verbose composite title
p_subspace <- (pSub_cap | pSub_pca | pSub_c | pSub_d) +
  plot_layout(nrow = 2) +
  plot_annotation(
    tag_levels = 'a', tag_prefix = '(', tag_suffix = ')',
    theme = paper_theme(9) + theme(
      plot.tag          = element_text(face = 'bold', size = 11),
      plot.tag.position = c(0.02, 0.98)
    ))
save_both(p_subspace, 'C_subspace', 7.2, 5.4)

# ════════════════════════════════════════════════════════════════════════════
# 10. C_timing — ICL emergence step vs context length (log-log)
# Data: fine-arm (induction_fine) for Muon×all L, AdamW×L{64,128};
#       main grid (induction_emergence) for AdamW L256 and SGDM×all L.
# Fine-arm means: Muon 30/40/70, AdamW 48/90; SGDM 290/480/480, AdamW256 260
# Ratios AdamW/Muon: 1.6×/2.3×/3.7×; SGDM/Muon: ~9.7×/12×/6.9×
# ════════════════════════════════════════════════════════════════════════════
cat('10. C_timing\n')

lens_t <- c(64L, 128L, 256L)

# ── Helper: read emergence_step from last _summary in a jsonl file ──────────
read_fine_emergence <- function(path) {
  lines <- readLines(path, warn = FALSE)
  for (l in rev(lines)) {
    d <- tryCatch(jsonlite::fromJSON(l, simplifyVector = TRUE), error = function(e) NULL)
    if (!is.null(d) && '_summary' %in% names(d)) {
      em <- d[['_summary']][['emergence_step']]
      if (!is.null(em) && !is.na(em)) return(as.numeric(em))
    }
  }
  NA_real_
}

# ── Collect fine-arm per-seed values ────────────────────────────────────────
fine_dir <- file.path(res_dir, 'induction_fine')

# Muon: L64, L128, L256 (fine)
# AdamW: L64, L128 (fine); L256 from main grid (induction_emergence)
fine_seeds <- list()
for (opt in c('muon', 'adamw')) {
  L_fine <- if (opt == 'muon') c(64L, 128L, 256L) else c(64L, 128L)
  for (L in L_fine) {
    vals <- c()
    for (s in 0:4) {
      fn <- file.path(fine_dir, sprintf('%s_L%d_fine_s%d.jsonl', opt, L, s))
      if (file.exists(fn)) vals <- c(vals, read_fine_emergence(fn))
    }
    fine_seeds[[paste0(opt, '_L', L)]] <- vals[!is.na(vals)]
  }
}

# ── Main grid: AdamW L256 + SGDM all L ─────────────────────────────────────
emerge_rawdir <- file.path(res_dir, 'induction_emergence')
emerge_raw_all <- bind_rows(lapply(Sys.glob(file.path(emerge_rawdir, '*.jsonl')), function(p) {
  lines <- readLines(p, warn = FALSE)
  for (l in rev(lines)) {
    d <- tryCatch(jsonlite::fromJSON(l, simplifyVector = TRUE), error = function(e) NULL)
    if (!is.null(d) && '_summary' %in% names(d)) {
      s <- d[['_summary']]
      em <- s[['emergence_step']]
      if (!is.null(em) && !is.na(em))
        return(data.frame(opt = s$optimizer, L = as.integer(s$seq_len),
                          emergence_step = as.numeric(em),
                          stringsAsFactors = FALSE))
    }
  }
  NULL
})) %>% filter(!is.na(emergence_step))

# Store main-grid values for SGDM and AdamW L256
for (opt in c('sgdm')) {
  for (L in c(64L, 128L, 256L)) {
    vals <- emerge_raw_all %>% filter(opt == !!opt, L == !!L) %>% pull(emergence_step)
    fine_seeds[[paste0(opt, '_L', L)]] <- vals[!is.na(vals)]
  }
}
# AdamW L256 from main grid
vals256 <- emerge_raw_all %>% filter(opt == 'adamw', L == 256L) %>% pull(emergence_step)
fine_seeds[['adamw_L256']] <- vals256[!is.na(vals256)]

# ── Build timing summary dataframe ──────────────────────────────────────────
timing_df <- bind_rows(lapply(c('adamw', 'muon', 'sgdm'), function(opt) {
  bind_rows(lapply(lens_t, function(L) {
    vals <- fine_seeds[[paste0(opt, '_L', L)]]
    if (length(vals) == 0) return(NULL)
    data.frame(opt    = opt, L = L,
               em_mean = mean(vals),
               em_std  = if (length(vals) > 1) sd(vals) else 0,
               stringsAsFactors = FALSE)
  }))
})) %>% mutate(opt = factor(opt, levels = c('adamw', 'muon', 'sgdm'),
                              labels = c('AdamW', 'Muon', 'SGDM')))

cat('C_timing summary (fine-arm merged):\n')
print(timing_df)

# ── Per-seed scatter: fine-arm where available, main grid for rest ───────────
# Fine-arm seeds (Muon all L, AdamW L64/128)
fine_scatter_rows <- bind_rows(lapply(c('muon', 'adamw'), function(opt) {
  L_fine <- if (opt == 'muon') c(64L, 128L, 256L) else c(64L, 128L)
  bind_rows(lapply(L_fine, function(L) {
    vals <- fine_seeds[[paste0(opt, '_L', L)]]
    if (length(vals) == 0) return(NULL)
    data.frame(opt = opt, L = L, emergence_step = vals, stringsAsFactors = FALSE)
  }))
}))

# Main-grid seeds for AdamW L256 + SGDM
main_scatter_rows <- emerge_raw_all %>%
  filter((opt == 'adamw' & L == 256L) | opt == 'sgdm')

emerge_scatter <- bind_rows(fine_scatter_rows, main_scatter_rows) %>%
  mutate(opt = factor(opt, levels = c('adamw', 'muon', 'sgdm'),
                       labels = c('AdamW', 'Muon', 'SGDM')))

# ── Plot ─────────────────────────────────────────────────────────────────────
p_timing <- ggplot(timing_df, aes(L, em_mean, colour = opt, group = opt)) +
  geom_point(data = emerge_scatter,
             aes(L, emergence_step, colour = opt),
             position = position_jitter(width = 0.04, height = 0, seed = 7),
             size = 1.5, alpha = 0.4, inherit.aes = FALSE) +
  geom_errorbar(aes(ymin = em_mean - em_std, ymax = em_mean + em_std),
                width = 0.15, linewidth = 0.45) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 3.2) +
  scale_x_log10(breaks = c(64, 128, 256)) +
  scale_y_log10(breaks = c(30, 50, 100, 200, 300, 500),
                labels = c('30', '50', '100', '200', '300', '500')) +
  coord_cartesian(ylim = c(20, 700)) +
  scale_colour_manual(values = c('AdamW' = OPT$adamw, 'SGDM' = OPT$sgdm,
                                  'Muon'  = OPT$muon), name = 'Optimizer') +
  labs(title = '(a) ICL emergence step vs context length (log-log)',
       x = 'Context length L', y = 'ICL emergence step') +
  paper_theme(9) +
  # legend dedup: (b) carries the identical Optimizer legend
  theme(legend.position = 'none')

# (b) matched-cadence control: emergence read on ONE shared evaluation grid
#     (eval_every = 100, 10 seeds/cell) — the cadence-confound objection
THR <- 0.5
read_emergence_step <- function(path) {
  lines <- readLines(path, warn = FALSE)
  for (l in lines) {
    d <- tryCatch(jsonlite::fromJSON(l, simplifyVector = TRUE), error = function(e) NULL)
    if (is.null(d) || '_meta' %in% names(d)) next
    if (!is.null(d[['icl_score']]) && !is.na(d[['icl_score']]) && d[['icl_score']] >= THR)
      return(as.numeric(d[['step']]))
  }
  NA_real_
}
mg_rows <- bind_rows(lapply(c('adamw', 'muon', 'sgdm'), function(opt) {
  bind_rows(lapply(lens_t, function(L) {
    paths <- Sys.glob(file.path(emerge_rawdir, sprintf('%s_L%d_s*.jsonl', opt, L)))
    vals  <- sapply(paths, read_emergence_step)
    vals  <- vals[!is.na(vals)]
    if (length(vals) == 0) return(NULL)
    data.frame(opt = opt, L = L, emergence_step = vals, stringsAsFactors = FALSE)
  }))
}))
mg_sum <- mg_rows %>%
  group_by(opt, L) %>%
  summarise(m = mean(emergence_step),
            s = if (n() > 1) sd(emergence_step) else 0, .groups = 'drop') %>%
  mutate(opt = factor(opt, levels = c('adamw', 'muon', 'sgdm'),
                      labels = c('AdamW', 'Muon', 'SGDM')))
p_timing_b <- ggplot(mg_sum, aes(L, m, colour = opt, group = opt)) +
  geom_errorbar(aes(ymin = pmax(m - s, 1), ymax = m + s),
                width = 0.06, linewidth = 0.45) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 3.0) +
  annotate('text', x = 95, y = 150,
           label = 'Muon & AdamW\nat grid floor (100 steps)',
           hjust = 0.5, size = 2.3, colour = '#1a1a1a', family = 'TeX Gyre Termes',
           lineheight = 0.95) +
  scale_x_log10(breaks = c(64, 128, 256)) +
  scale_y_log10(breaks = c(100, 200, 300, 500),
                labels = c('100', '200', '300', '500')) +
  coord_cartesian(ylim = c(80, 650)) +
  scale_colour_manual(values = c('AdamW' = OPT$adamw, 'SGDM' = OPT$sgdm,
                                  'Muon'  = OPT$muon), name = 'Optimizer') +
  labs(title = '(b) Matched-cadence control (eval every 100)',
       x = 'Context length L', y = 'Emergence step (log)') +
  paper_theme(9)

# (c) acceleration ratios with bootstrap CIs (per-seed resampled median ratio)
ratio_rows <- bind_rows(lapply(lens_t, function(L) {
  mu <- fine_seeds[[paste0('muon_L', L)]]
  bind_rows(lapply(c('adamw', 'sgdm'), function(o) {
    v <- fine_seeds[[paste0(o, '_L', L)]]
    if (!length(v) || !length(mu)) return(NULL)
    set.seed(L)
    boot <- replicate(20000, {
      mean(sample(v, replace = TRUE)) / mean(sample(mu, replace = TRUE))
    })
    data.frame(L = L, pair = paste0(c(adamw = 'AdamW', sgdm = 'SGDM')[o], ' / Muon'),
               ratio = mean(v) / mean(mu),
               lo = unname(quantile(boot, 0.025)), hi = unname(quantile(boot, 0.975)))
  }))
}))
p_timing_c <- ggplot(ratio_rows, aes(factor(L), ratio, colour = pair)) +
  geom_hline(yintercept = 1, linetype = 'dashed', colour = '#888', linewidth = 0.6) +
  geom_pointrange(aes(ymin = lo, ymax = hi),
                  position = position_dodge(width = 0.5), linewidth = 0.8, size = 0.7) +
  scale_colour_manual(values = c('AdamW / Muon' = OPT$adamw, 'SGDM / Muon' = OPT$sgdm),
                      name = NULL) +
  labs(title = '(c) Acceleration ratios (95% bootstrap CI)',
       x = 'Context length L', y = 'Emergence-step ratio') +
  paper_theme(9) +
  theme(legend.position = 'top')

# (d) per-seed emergence steps (fine grid; the raw distributions)
p_timing_d <- ggplot(emerge_scatter, aes(factor(L), emergence_step, colour = opt)) +
  stat_summary(fun = median, geom = 'crossbar', width = 0.5, colour = '#1a1a1a',
               position = position_dodge(width = 0.7)) +
  geom_point(position = position_jitterdodge(jitter.width = 0.05, dodge.width = 0.7,
                                             seed = 7),
             size = 1.3, alpha = 0.7) +
  scale_colour_manual(values = c('AdamW' = OPT$adamw, 'SGDM' = OPT$sgdm,
                                  'Muon'  = OPT$muon), name = 'Optimizer') +
  scale_y_log10() +
  labs(title = '(d) Per-seed emergence steps',
       x = 'Context length L', y = 'Emergence step (log)') +
  paper_theme(9) +
  theme(legend.position = 'none')

p_timing4 <- compose_C_four_panel(
  list(p_timing, p_timing_b, p_timing_c, p_timing_d),
  'C_timing', 7.2, 5.4, root)
save_both(p_timing4, 'C_timing', 7.2, 5.4)

cat('\n=== All 10 figures written ===\n')

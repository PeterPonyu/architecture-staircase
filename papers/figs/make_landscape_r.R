#!/usr/bin/env Rscript
# Polished R/ggplot2 renderer for the five literature-landscape opener figures.
# Matches the paper house style (paper_theme + CB palette + 300dpi PNG / SVG),
# so the openers are visually consistent with every other figure in the papers.
# All numbers are verified literature values (see per-figure comments / captions).

ver <- paste(R.version$major, sub('\\..*', '', R.version$minor), sep = '.')
userlib <- file.path(Sys.getenv('HOME'), 'R', 'x86_64-pc-linux-gnu-library', ver)
.libPaths(c(userlib, .libPaths()))

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(ggrepel)
  library(scales); library(ragg); library(svglite)
})

fig_dir <- normalizePath('.')                       # run from papers/figs
evid    <- file.path(fig_dir, 'evidence_r'); dir.create(evid, showWarnings = FALSE)

paper_theme <- function(base_size = 9) {
  theme_minimal(base_size = base_size, base_family = 'TeX Gyre Termes') +
    theme(
      panel.grid.minor   = element_blank(),
      panel.grid.major   = element_line(linewidth = 0.25, colour = '#d9dde3'),
      axis.title  = element_text(size = base_size, colour = '#1a1a1a'),
      axis.text   = element_text(size = base_size - 0.5, colour = '#1a1a1a'),
      plot.title  = element_text(colour = '#111827', size = base_size + 1.5),
      plot.subtitle = element_text(colour = '#444444', size = base_size - 0.5),
      legend.position = 'top', legend.title = element_text(size = base_size - 1),
      legend.text = element_text(size = base_size - 1),
      plot.margin = margin(6, 8, 6, 6)
    )
}
CB <- list(blue='#1f77b4', vermillion='#d62728', orange='#ff7f0e',
           green='#2ca02c', purple='#9467bd', grey='#7f7f7f', ink='#1a1a1a')

save_both <- function(p, name, w = 6.6, h = 3.2) {
  ragg::agg_png(file.path(fig_dir, paste0(name, '.png')), width = w, height = h,
                units = 'in', res = 300, scaling = 1); print(p); dev.off()
  svglite::svglite(file.path(evid, paste0(name, '.svg')), width = w, height = h)
  print(p); dev.off(); emit_vector(p, name, w, h); cat('saved', name, '\n')
}
source(file.path(fig_dir, 'fig_pipeline.R'))  # emit_vector(): tikz/.tex + cairo_pdf/.pdf

## ---- A: grokking-acceleration landscape --------------------------------
dfa <- data.frame(
  label  = c('Tveit et al. 2025\noptimizer swap (Muon)',
             'Grokfast 2024\ngradient filtering',
             'NeuralGrok 2025\ngradient transform',
             'GrokAlign 2025\nJacobian regulariser',
             'This work\nnorm-cap (causal)'),
  factor = c(1.49, 50, 2.95, 7.56, 10), y = c(5, 4, 3, 2, 1),
  kind   = c('Optimizer swap', 'Gradient filtering', 'Gradient filtering',
             'Regulariser / norm', 'This work (causal)'),
  tag    = c('1.49×', '>50×', '2.95×', '7.56×', '~10×'))
pal_a <- c('Optimizer swap'=CB$blue, 'Gradient filtering'=CB$green,
           'Regulariser / norm'=CB$orange, 'This work (causal)'=CB$vermillion)
pA <- ggplot(dfa, aes(factor, y, colour = kind)) +
  geom_vline(xintercept = 1, linetype = 'dotted', colour = '#9aa0a6') +
  geom_segment(aes(x = 1, xend = factor, y = y, yend = y), linewidth = 0.8, alpha = 0.45) +
  geom_point(size = 3.6) +
  geom_text(aes(label = tag), family = 'TeX Gyre Termes', vjust = -1.0, size = 3.1, fontface = 'plain', colour = 'black') +
  scale_x_log10(limits = c(0.9, 130), breaks = c(1, 2, 5, 10, 20, 50, 100)) +
  scale_y_continuous(breaks = dfa$y, labels = dfa$label, limits = c(0.5, 5.5)) +
  scale_colour_manual(values = pal_a, guide = 'none') +
  labs(x = 'grokking acceleration factor (× faster, log scale)', y = NULL) +
  paper_theme() +
  theme(plot.margin = margin(5.5, 7, 5.5, 14))
save_both(pA, 'A_landscape', w = 6.7, h = 3.1)

## ---- B: delay-lambda exponent meta-forest ------------------------------
dfb <- data.frame(
  study = c('Omnigrok (Liu 2023)', 'Multi-task tri (Xu 2026)',
            'Multi-task dual (Xu 2026)', 'Weight-decay regimes (Verma 2026)',
            'Norm-separation law (Truong/Khanh 2026)',
            'This work — fp32', 'This work — bf16'),
  # This-work fp32: balanced (level-means) fit over all 43 defloored fp32 runs.
  # The original nine-run headline population gives -0.50 +- 0.09; deleting one
  # lambda level at a time spans -0.37 to -0.52. The band below is drawn to
  # -0.35 so it contains every population the fp32 arm supports.
  slope = c(-1.00, -0.97, -0.87, -0.86, -0.78, -0.46, 0.00),
  se    = c(NA, NA, NA, NA, NA, 0.07, 0.12),
  kind  = c('Prior work','Prior work','Prior work','Prior work','Prior work',
            'This work','Artifact'),
  note  = c('asserted','derived','derived','derived','derived','n=43','bf16 artifact'),
  y = 7:1)
pal_b <- c('Prior work'=CB$grey, 'This work'=CB$blue, 'Artifact'=CB$green)
pB <- ggplot(dfb, aes(slope, y, colour = kind)) +
  annotate('rect', xmin = -1, xmax = -0.35, ymin = 0.4, ymax = 7.6, fill = CB$blue, alpha = 0.06) +
  geom_vline(xintercept = -1, linetype = 'dashed', colour = '#444444', linewidth = 0.4) +
  geom_vline(xintercept = 0, linetype = 'dotted', colour = '#9aa0a6') +
  annotate('text', x = -1, y = 7.75, label = 'theory −1', family = 'TeX Gyre Termes',
           size = 2.6, vjust = 0, colour = 'black') +
  geom_errorbarh(aes(xmin = slope - se, xmax = slope + se), height = 0.18, linewidth = 0.7, na.rm = TRUE) +
  geom_point(aes(shape = kind), size = 3.2) +
  geom_text(aes(label = ifelse(note == '', sprintf('%+.2f', slope), sprintf('%+.2f (%s)', slope, note))),
            family = 'TeX Gyre Termes', size = 2.7, vjust = -1.1, colour = 'black') +
  scale_x_continuous(limits = c(-1.25, 0.35), breaks = seq(-1.2, 0.2, 0.2),
                     labels = function(z) sprintf('%.1f', z)) +
  scale_y_continuous(breaks = dfb$y, labels = dfb$study, limits = c(0.4, 8.1)) +
  scale_colour_manual(values = pal_b, guide = 'none') +
  scale_shape_manual(values = c('Prior work'=16, 'This work'=15, 'Artifact'=18), guide = 'none') +
  labs(x = expression('grokking delay vs '*lambda*'   (log-log exponent)'), y = NULL) +
  paper_theme()
save_both(pB, 'B_metaslope', w = 6.8, h = 3.3)

## ---- C: sequential-acquisition order coverage --------------------------
segs <- data.frame(
  label = c('Abbe 2021/23 (theory)', 'Saxe 2014 (linear)', 'Rende 2024 (NL)',
            'Varre 2025 (n-gram)', 'This work (Boolean)'),
  lo = c(1, 1, 3, 1, 1), hi = c(33, 33, 33, 3, 4), y = c(6, 5, 3, 2, 1),
  kind = c('Theory', 'Theory', 'Prior (empirical)', 'Prior (empirical)', 'This work'))
pts <- data.frame(x = c(3, 3, 9, 33), y = c(4, 3, 3, 3),
                  kind = c('Prior (empirical)', 'mark', 'mark', 'mark'))
yax <- data.frame(y = 6:1, lab = c('Abbe 2021/23 (theory)', 'Saxe 2014 (linear)',
                  'Barak 2022', 'Rende 2024 (NL)', 'Varre 2025 (n-gram)', 'This work (Boolean)'))
labs_c <- data.frame(x = c(1, 1, 3, 3, 1, 1), y = c(6.36, 5.36, 4.0, 3.36, 2.36, 1.36),
                     lab = c('any order', 'sequential SVD modes', 'single k-parity',
                             'orders 3, 9, 33', 'n-grams 1–3', 'degrees 1→4'),
                     hj  = c(0, 0, -0.15, 0, 0, 0),
                     kind = c('Theory', 'Theory', 'Prior (empirical)', 'Prior (empirical)',
                              'Prior (empirical)', 'This work'))
pal_c <- c('Theory'=CB$grey, 'Prior (empirical)'=CB$blue, 'This work'=CB$vermillion)
pC <- ggplot() +
  geom_segment(data = segs, aes(x = lo, xend = hi, y = y, yend = y, colour = kind),
               linewidth = 5, lineend = 'round', alpha = 0.85) +
  geom_point(data = subset(pts, kind == 'Prior (empirical)'), aes(x, y), colour = CB$blue, size = 3.4) +
  geom_point(data = subset(pts, kind == 'mark'), aes(x, y), shape = 21, fill = 'white',
             colour = CB$blue, size = 1.7, stroke = 1) +
  geom_text(data = labs_c, aes(x, y, label = lab), colour = 'black', family = 'TeX Gyre Termes',
            size = 2.8, hjust = labs_c$hj, show.legend = FALSE) +
  scale_x_log10(limits = c(0.9, 42), breaks = c(1, 2, 3, 4, 9, 33)) +
  scale_y_continuous(breaks = yax$y, labels = yax$lab, limits = c(0.4, 6.6)) +
  scale_colour_manual(values = pal_c, guide = 'none') +
  labs(x = 'interaction order / monomial degree (log)', y = NULL) +
  paper_theme()
save_both(pC, 'C_landscape', w = 6.7, h = 3.4)

## ---- E1: free-repetition R_free vs scale -------------------------------
dfe1 <- data.frame(
  x = c(2.5e6, 5.6e6, 9.9e6, 5e6, 8.7e9, 1.2e11, 1.5e8),
  r = c(10, 4, 4, 4, 4, 4, 16),
  grp = c('This work: synthetic (small-cap corner)', 'This work: synthetic (larger cells)',
          'This work: synthetic (larger cells)', 'This work: WikiText-103 byte bridge',
          'Muennighoff 2023 (LLM)', 'Galactica 2022 (120B)', 'Chen 2026 (decay onset)'))
pal_e1 <- c('This work: synthetic (small-cap corner)'=CB$vermillion,
            'This work: synthetic (larger cells)'='#e8888a',
            'This work: WikiText-103 byte bridge'=CB$orange,
            'Muennighoff 2023 (LLM)'=CB$blue,
            'Galactica 2022 (120B)'=CB$purple,
            'Chen 2026 (decay onset)'=CB$grey)
shp_e1 <- c('This work: synthetic (small-cap corner)'=15, 'This work: synthetic (larger cells)'=15,
            'This work: WikiText-103 byte bridge'=17, 'Muennighoff 2023 (LLM)'=16,
            'Galactica 2022 (120B)'=18, 'Chen 2026 (decay onset)'=4)
pE1 <- ggplot(dfe1, aes(x, r, colour = grp, shape = grp)) +
  annotate('rect', xmin = 1.3e5, xmax = 3e10, ymin = 0, ymax = 4, fill = CB$green, alpha = 0.06) +
  annotate('text', x = 1.6e5, y = 4.4, hjust = 0, label = '≈4-epoch free floor',
           family = 'TeX Gyre Termes', size = 2.6, colour = 'black') +
  # Half-life reference converted from a full-width dashed line (which spanned the
  # otherwise-empty upper band) to a compact annotation on the Chen 2026 marker,
  # whose decay-onset value (r=16) already IS the repeated-token half-life.
  annotate('text', x = 2.6e8, y = 16, hjust = 0, family = 'TeX Gyre Termes', size = 2.5, colour = 'black',
           label = 'half-life ≈16 epochs') +
  geom_point(size = 3.3) +
  scale_x_log10(limits = c(1.3e5, 3e11), breaks = c(1e6, 1e8, 1e10, 1e11),
                labels = c('1e6','1e8','1e10','1e11')) +
  scale_y_continuous(limits = c(0, 17)) +
  scale_colour_manual(values = pal_e1, name = NULL) +
  scale_shape_manual(values = shp_e1, name = NULL) +
  guides(colour = guide_legend(nrow = 2, byrow = TRUE),
         shape  = guide_legend(nrow = 2, byrow = TRUE)) +
  labs(x = 'model scale (parameters, log)', y = expression(R[free]*' (free-repetition epochs)')) +
  paper_theme() + theme(legend.position = 'bottom',
                        legend.box = 'horizontal',
                        legend.key.height = unit(0.30, 'cm'),
                        legend.key.width = unit(0.30, 'cm'),
                        legend.text = element_text(size = 6.3),
                        legend.margin = margin(2, 2, 0, 2))
save_both(pE1, 'E1_landscape', w = 6.9, h = 3.9)

## ---- E2: in-context-emergence claim landscape --------------------------
claims <- data.frame(x = c(0, 1, 2), y = 1,
                     lab = c('Olsson 2022', 'Lee 2023', 'Wang 2025'))
ours <- data.frame(x = c(0, 2), y = 0, kind = c('positive', 'negative'),
                   lab = c('emerges ≈1850 steps\n(clean positive)',
                           'calibration-negative\ndetector 42/45 → 1/45 (τ=0.7)'))
pE2 <- ggplot() +
  geom_point(data = claims, aes(x, y), colour = CB$blue, size = 3.6) +
  geom_text(data = claims, aes(x, y, label = lab),
            family = 'TeX Gyre Termes', size = 2.7, colour = 'black', vjust = -1.0) +
  geom_point(data = ours, aes(x, y, shape = kind, colour = kind), size = 4) +
  geom_text(data = ours, aes(x, y, label = lab), colour = 'black', family = 'TeX Gyre Termes',
            size = 2.7, vjust = 1.7, show.legend = FALSE) +
  annotate('text', x = 1, y = 0, label = '(no calibrated\nnegative existed)', family = 'TeX Gyre Termes',
           size = 2.5, colour = 'black', vjust = 1.7) +
  scale_shape_manual(values = c('positive'=15, 'negative'=4), guide = 'none') +
  scale_colour_manual(values = c('positive'=CB$green, 'negative'=CB$vermillion), guide = 'none') +
  scale_x_continuous(breaks = c(0, 1, 2), labels = c('induction / copy', 'in-context RL', 'in-context TD'),
                     limits = c(-0.5, 2.5)) +
  scale_y_continuous(breaks = c(0, 1), labels = c('this work\n(calibrated)', 'published\nclaim'),
                     limits = c(-0.62, 1.42)) +
  labs(x = NULL, y = NULL) +
  paper_theme() + theme(panel.grid.major.x = element_blank(),
                        plot.margin = margin(3, 8, 3, 6))
save_both(pE2, 'E2_landscape', w = 7.0, h = 2.1)

cat('all landscape figures rendered\n')

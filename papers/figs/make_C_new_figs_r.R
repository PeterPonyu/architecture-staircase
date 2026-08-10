#!/usr/bin/env Rscript
# make_C_new_figs_r.R — NEW standalone Paper C figures from already-measured data.
# All plot-only (no GPU). Numbers computed directly from raw .jsonl logs and the
# figures-004 verdict JSON, cross-checked against findings-004/010/013/017.
#
# Figures produced:
#   C_pure3_speedup  (body)     — pure degree-3 timing: Muon ~6.6x faster than AdamW;
#                                  SGDM fails to learn (fit ~ 0).
#   C_ladder_necessity (body)   — final fit_corr x optimizer, pure-deg3 vs nested
#                                  staircase; SGDM fails pure-deg3 but the ladder
#                                  bootstraps it to a high overall fit.
#   C_matched_grid_timing (appendix) — matched-cadence (eval_every=100) 10-seed
#                                  emergence step vs L, 3 optimizers; removes the
#                                  cadence-confound objection to the fine-grid timing.
#
# Design standard (shared with make_C_figs_r.R, width bumped to 6.5in unified column):
#   - Panel labels (a)(b) bold left-aligned via patchwork
#   - base_family TeX Gyre Termes, ragg::agg_png 300dpi, base_size ~9 (>=8pt effective)
#   - ZERO dev-leakage in titles/axes/legend (descriptive science titles only)

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
})

root    <- normalizePath(file.path(getwd()))
fig_dir <- file.path(root, 'papers', 'figs')
res_dir <- file.path(root, 'experiments', 'results')
source(file.path(fig_dir, 'fig_pipeline.R'))  # emit_vector(): tikz/.tex + cairo_pdf/.pdf
source(file.path(fig_dir, 'C_panel_contract.R'))

# ── Colour palette (matches make_C_figs_r.R) ────────────────────────────────
CB <- list(
  blue       = '#0072B2',
  orange     = '#E69F00',
  green       = '#009E73',
  vermillion = '#D55E00',
  sky        = '#56B4E9',
  grey       = '#999999'
)
OPT <- list(AdamW = CB$blue, Muon = CB$vermillion, SGDM = CB$grey)

paper_theme <- function(base_size = 9) {
  theme_minimal(base_size = base_size, base_family = 'TeX Gyre Termes') +
    theme(
      panel.grid.minor   = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(linewidth = 0.25, colour = '#d9dde3'),
      axis.title         = element_text(family = 'TeX Gyre Termes', colour = '#1a202c', size = base_size),
      axis.text          = element_text(family = 'TeX Gyre Termes', colour = '#2d3748', size = base_size - 0.5),
      plot.title         = element_text(family = 'TeX Gyre Termes', face = 'plain', colour = '#111827',
                                        size = base_size + 1),
      plot.subtitle      = element_text(family = 'TeX Gyre Termes', colour = '#1a1a1a', size = base_size - 0.5),
      legend.position    = 'top',
      legend.title       = element_text(family = 'TeX Gyre Termes', size = base_size - 0.5),
      legend.text        = element_text(family = 'TeX Gyre Termes', size = base_size - 0.5),
      strip.text         = element_text(family = 'TeX Gyre Termes', face = 'plain', colour = '#1a202c'),
      plot.margin        = margin(5.5, 6, 5.5, 6),
      plot.tag           = element_text(family = 'TeX Gyre Termes', face = 'bold', size = 11),
      plot.tag.position  = c(0.02, 0.98)
    )
}

save_png <- function(p, name, w = 6.5, h = 4.0) {
  png_path <- file.path(fig_dir, paste0(name, '.png'))
  ragg::agg_png(png_path, width = w, height = h, units = 'in', res = 300, scaling = 1)
  print(p); dev.off()
  emit_vector(p, name, w, h)   # vector tier: tikz (.tex) or cairo_pdf (.pdf)
  cat(sprintf('  saved %s  (%d bytes)\n', png_path, file.info(png_path)$size))
}

# Read the last _summary record from a degree_staircase jsonl run.
read_last_summary <- function(path) {
  lines <- readLines(path, warn = FALSE)
  for (l in rev(lines)) {
    d <- tryCatch(jsonlite::fromJSON(l, simplifyVector = TRUE), error = function(e) NULL)
    if (!is.null(d) && '_summary' %in% names(d)) return(d[['_summary']])
  }
  NULL
}

deg_dir <- file.path(res_dir, 'degree_staircase')
OPTS    <- c('adamw', 'muon', 'sgdm')
OPTLAB  <- c(adamw = 'AdamW', muon = 'Muon', sgdm = 'SGDM')

cat('=== make_C_new_figs_r.R ===\n')

# ════════════════════════════════════════════════════════════════════════════
# Collect per-seed pure-deg3 and nested-staircase summaries (single source of
# truth for C1 + C2), straight from raw logs.
# ════════════════════════════════════════════════════════════════════════════
collect <- function(opt, kind) {
  paths <- Sys.glob(file.path(deg_dir, sprintf('%s_%s_s*.jsonl', opt, kind)))
  rows  <- lapply(paths, function(p) {
    s <- read_last_summary(p)
    if (is.null(s)) return(NULL)
    fd <- s[['final_deg_corr']]
    ht <- s[['half_times']]
    data.frame(
      opt      = opt,
      kind     = kind,
      fit      = as.numeric(s[['final_fit_corr']]),
      t3       = if (!is.null(ht[['3']])) as.numeric(ht[['3']]) else NA_real_,
      d1       = if (!is.null(fd[['1']])) as.numeric(fd[['1']]) else NA_real_,
      d2       = if (!is.null(fd[['2']])) as.numeric(fd[['2']]) else NA_real_,
      d3       = if (!is.null(fd[['3']])) as.numeric(fd[['3']]) else NA_real_,
      d4       = if (!is.null(fd[['4']])) as.numeric(fd[['4']]) else NA_real_,
      stringsAsFactors = FALSE
    )
  })
  bind_rows(Filter(Negate(is.null), rows))
}

pure3  <- bind_rows(lapply(OPTS, collect, kind = 'pure3'))
nested <- bind_rows(lapply(OPTS, collect, kind = 'staircase'))

cat('\n-- C1/C2 verification (computed from raw _summary) --\n')
pure3 %>% group_by(opt) %>%
  summarise(n = n(), T3_mean = mean(t3), fit_mean = mean(fit), .groups = 'drop') %>%
  as.data.frame() %>% print()
nested %>% group_by(opt) %>%
  summarise(n = n(), fit_mean = mean(fit),
            d1 = mean(d1), d2 = mean(d2), d3 = mean(d3), d4 = mean(d4),
            .groups = 'drop') %>% as.data.frame() %>% print()

# ════════════════════════════════════════════════════════════════════════════
# C_pure3_speedup — isolate the TIMING mechanism on a single-degree target.
# AdamW and Muon both fit perfectly (fit ~ 1.0); Muon ~6.6x faster.
# SGDM never learns (fit ~ 0), so its half-time is a failed-to-learn artifact and
# is shown as a failure marker, NOT a misleading "fast" time.
# ════════════════════════════════════════════════════════════════════════════
cat('\n1. C_pure3_speedup\n')

learn_thr <- 0.5  # fit_corr above which the target is considered learned
p3 <- pure3 %>%
  mutate(learned = fit > learn_thr,
         optf = factor(OPTLAB[opt], levels = c('AdamW', 'Muon', 'SGDM')))

# timing strip uses only the optimizers that actually learned the target
timing <- p3 %>% filter(learned)
fail   <- p3 %>% filter(!learned)

timing_means <- timing %>% group_by(optf) %>%
  summarise(m = mean(t3), .groups = 'drop')

m_adamw <- timing_means$m[timing_means$optf == 'AdamW']
m_muon  <- timing_means$m[timing_means$optf == 'Muon']
speedup <- m_adamw / m_muon

ann_speed <- sprintf('Muon %.0f±%.0f steps\nAdamW %.0f steps\n(%.1f× slower)',
                     m_muon, sd(timing$t3[timing$optf == 'Muon']),
                     m_adamw, speedup)

p_pure3 <- ggplot(timing, aes(optf, t3, colour = optf)) +
  geom_crossbar(data = timing_means, aes(y = m, ymin = m, ymax = m),
                width = 0.34, linewidth = 1.0) +
  geom_jitter(width = 0.12, height = 0, size = 2.6, alpha = 0.8) +
  # SGDM: failed-to-learn marker at the floor (kept inside the panel: centred
  # between Muon and SGDM at reduced size so it cannot clip the panel edge)
  geom_text(data = data.frame(x = 2.62),
            aes(x = x, y = 55), inherit.aes = FALSE,
            label = 'SGDM fails to learn\n(final fit ≈ 0)',
            colour = 'black', size = 3.0, lineheight = 0.95, family = 'TeX Gyre Termes') +
  annotate('text', x = 2.3, y = 1080, label = ann_speed,
           hjust = 0.5, size = 3.2, colour = 'black', lineheight = 0.95,
           family = 'TeX Gyre Termes') +
  scale_colour_manual(values = c('AdamW' = OPT$AdamW, 'Muon' = OPT$Muon,
                                 'SGDM' = OPT$SGDM), guide = 'none') +
  scale_x_discrete(drop = FALSE) +
  coord_cartesian(ylim = c(0, 1650)) +
  labs(title = '(a) Time to acquire a single degree-3 target',
       x = NULL, y = 'Acquisition half-time (steps)') +
  # Displayed at 0.82\linewidth on a 6.5in design (resizebox scale 0.82) while
  # most figures render at ~1.3x; base_size 13 (+ the geom_text/annotate size
  # bumps above) lands the printed text at body-comparable size.
  paper_theme(10)

# (b) bootstrap speedup: resampled AdamW/Muon half-time ratio
set.seed(97)
sp_boot <- replicate(20000, {
  mean(sample(timing$t3[timing$optf == 'AdamW'], replace = TRUE)) /
    mean(sample(timing$t3[timing$optf == 'Muon'], replace = TRUE))
})
sp_df <- data.frame(ratio = sp_boot)
p_pure3_b <- ggplot(sp_df, aes(ratio)) +
  geom_vline(xintercept = 1, linetype = 'dashed', colour = '#888', linewidth = 0.6) +
  geom_histogram(bins = 40, fill = CB$blue, colour = 'white', linewidth = 0.2) +
  geom_vline(xintercept = unname(quantile(sp_boot, c(0.025, 0.975))),
             linetype = 'dotted', colour = '#1a1a1a') +
  labs(title = sprintf('(b) Speedup %.1fx [95%% CI %.1f-%.1f]',
                       speedup, unname(quantile(sp_boot, 0.025)),
                       unname(quantile(sp_boot, 0.975))),
       x = 'AdamW / Muon half-time ratio', y = 'Bootstrap count') +
  paper_theme(9)

# (c) matched-fit control: both optimizers fit the target perfectly, so the
#     timing difference is not a fit-quality confound (SGDM shown for the floor)
p_pure3_c <- ggplot(p3, aes(optf, fit, colour = optf)) +
  stat_summary(fun = median, geom = 'crossbar', width = 0.45, colour = '#1a1a1a') +
  geom_point(position = position_jitter(width = 0.07, height = 0, seed = 61),
             size = 2.0, alpha = 0.8, show.legend = FALSE) +
  scale_colour_manual(values = c('AdamW' = OPT$AdamW, 'Muon' = OPT$Muon,
                                 'SGDM' = OPT$SGDM), guide = 'none') +
  coord_cartesian(ylim = c(-0.05, 1.05)) +
  labs(title = '(c) Matched fit at measurement',
       x = NULL, y = 'Final fit correlation') +
  paper_theme(9)

# (d) nested-ladder per-degree half-times (onset ordering under each optimizer)
deg_ht <- bind_rows(lapply(OPTS, function(opt) {
  paths <- Sys.glob(file.path(deg_dir, sprintf('%s_staircase_s*.jsonl', opt)))
  bind_rows(lapply(paths, function(p) {
    s <- read_last_summary(p)
    if (is.null(s)) return(NULL)
    ht <- s[['half_times']]
    bind_rows(lapply(1:4, function(k) {
      v <- ht[[as.character(k)]]
      data.frame(opt = opt, degree = factor(paste0('deg ', k), levels = paste0('deg ', 1:4)),
                 ht = if (!is.null(v)) as.numeric(v) else NA_real_)
    }))
  }))
})) %>% filter(!is.na(ht)) %>%
  mutate(optf = factor(OPTLAB[opt], levels = c('AdamW', 'Muon', 'SGDM')))
p_pure3_d <- ggplot(deg_ht, aes(degree, ht, colour = optf)) +
  stat_summary(fun = median, geom = 'crossbar', width = 0.5, colour = '#1a1a1a',
               position = position_dodge(width = 0.7)) +
  geom_point(position = position_jitterdodge(jitter.width = 0.06, dodge.width = 0.7,
                                             seed = 61),
             size = 1.4, alpha = 0.7) +
  scale_colour_manual(values = c('AdamW' = OPT$AdamW, 'Muon' = OPT$Muon,
                                 'SGDM' = OPT$SGDM), name = NULL) +
  labs(title = '(d) Per-degree half-times on the ladder',
       x = NULL, y = 'Half-time (steps)') +
  paper_theme(9) +
  theme(legend.position = 'top')

p_pure3_four <- compose_C_four_panel(
  list(p_pure3, p_pure3_b, p_pure3_c, p_pure3_d),
  'C_pure3_speedup', 7.2, 5.4, root)
save_png(p_pure3_four, 'C_pure3_speedup', 7.2, 5.4)

# ════════════════════════════════════════════════════════════════════════════
# C_ladder_necessity — final fit_corr x optimizer, pure-deg3 vs nested staircase.
# SGDM fails the isolated degree-3 target but the nested ladder bootstraps it to a
# high overall fit (Prop G1). Companion panel shows SGDM's per-degree breakdown:
# degrees 1-3 acquired on the ladder, degree 4 largely not.
# ════════════════════════════════════════════════════════════════════════════
cat('\n2. C_ladder_necessity\n')

# Panel (a): grouped bars, mean fit +/- SD, two bars per optimizer
fit_summary <- bind_rows(
  pure3  %>% mutate(target = 'Single degree-3'),
  nested %>% mutate(target = 'Nested staircase')
) %>%
  group_by(opt, target) %>%
  summarise(m = mean(fit), s = sd(fit), n = n(), .groups = 'drop') %>%
  mutate(optf   = factor(OPTLAB[opt], levels = c('AdamW', 'Muon', 'SGDM')),
         target = factor(target, levels = c('Single degree-3', 'Nested staircase')))

pA_ladder <- ggplot(fit_summary, aes(optf, m, fill = target)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.72, alpha = 0.9) +
  geom_errorbar(aes(ymin = pmax(m - s, -0.05), ymax = m + s),
                position = position_dodge(width = 0.8), width = 0.18, linewidth = 0.45) +
  geom_hline(yintercept = 0, colour = '#374151', linewidth = 0.4) +
  annotate('text', x = 3, y = -0.12, label = 'isolated target\nfails',
           colour = 'black', size = 2.6, lineheight = 0.9) +
  scale_fill_manual(values = c('Single degree-3' = CB$orange,
                               'Nested staircase' = CB$blue), name = NULL) +
  coord_cartesian(ylim = c(-0.18, 1.05)) +
  labs(title = 'Final fit: isolated target vs full ladder',
       x = NULL, y = 'Final fit correlation') +
  paper_theme(9)

# Panel (b): SGDM per-degree final corr on the nested ladder (the bootstrap)
sgdm_deg <- nested %>% filter(opt == 'sgdm') %>%
  select(d1, d2, d3, d4) %>%
  pivot_longer(everything(), names_to = 'degree', values_to = 'corr') %>%
  mutate(degree = recode(degree, d1 = 'Deg 1', d2 = 'Deg 2',
                         d3 = 'Deg 3', d4 = 'Deg 4'))
sgdm_deg_mean <- sgdm_deg %>% group_by(degree) %>%
  summarise(m = mean(corr), .groups = 'drop')

pB_ladder <- ggplot(sgdm_deg, aes(degree, corr)) +
  geom_jitter(width = 0.12, height = 0, colour = OPT$SGDM, alpha = 0.55, size = 2) +
  geom_crossbar(data = sgdm_deg_mean, aes(y = m, ymin = m, ymax = m),
                width = 0.4, linewidth = 1.1, colour = '#444444') +
  geom_hline(yintercept = 0, colour = '#888888', linetype = 'dashed', linewidth = 0.5) +
  coord_cartesian(ylim = c(-0.1, 0.8)) +
  labs(title = 'SGDM per-degree acquisition on the ladder',
       x = NULL, y = 'Final Walsh correlation') +
  paper_theme(9)

# (c) per-seed fit strips by target and optimizer (the distributions behind (a))
fit_seeds <- bind_rows(
  pure3  %>% mutate(target = 'Single degree-3'),
  nested %>% mutate(target = 'Nested staircase')
) %>% mutate(optf = factor(OPTLAB[opt], levels = c('AdamW', 'Muon', 'SGDM')),
             target = factor(target, levels = c('Single degree-3', 'Nested staircase')))
pC_ladder <- ggplot(fit_seeds, aes(target, fit, colour = optf)) +
  stat_summary(fun = median, geom = 'crossbar', width = 0.5, colour = '#1a1a1a',
               position = position_dodge(width = 0.75)) +
  geom_point(position = position_jitterdodge(jitter.width = 0.07, dodge.width = 0.75,
                                             seed = 63),
             size = 1.4, alpha = 0.7) +
  scale_colour_manual(values = c('AdamW' = OPT$AdamW, 'Muon' = OPT$Muon,
                                 'SGDM' = OPT$SGDM), name = NULL) +
  coord_cartesian(ylim = c(-0.18, 1.05)) +
  labs(title = 'Per-seed fit by target',
       x = NULL, y = 'Final fit correlation') +
  paper_theme(9) +
  theme(axis.text.x = element_text(size = 7.5, angle = 15, hjust = 1),
        legend.position = 'top')

# (d) per-degree finals for ALL optimizers on the ladder (the bootstrap is
#     optimizer-shared at degrees 1-3; degree 4 splits them)
all_deg <- nested %>%
  select(opt, d1, d2, d3, d4) %>%
  pivot_longer(c(d1, d2, d3, d4), names_to = 'degree', values_to = 'corr') %>%
  mutate(degree = recode(degree, d1 = 'Deg 1', d2 = 'Deg 2',
                         d3 = 'Deg 3', d4 = 'Deg 4'),
         optf = factor(OPTLAB[opt], levels = c('AdamW', 'Muon', 'SGDM')))
pD_ladder <- ggplot(all_deg, aes(degree, corr, colour = optf)) +
  stat_summary(fun = median, geom = 'crossbar', width = 0.5, colour = '#1a1a1a',
               position = position_dodge(width = 0.75)) +
  geom_point(position = position_jitterdodge(jitter.width = 0.07, dodge.width = 0.75,
                                             seed = 63),
             size = 1.3, alpha = 0.65) +
  scale_colour_manual(values = c('AdamW' = OPT$AdamW, 'Muon' = OPT$Muon,
                                 'SGDM' = OPT$SGDM), name = NULL) +
  labs(title = 'Per-degree finals, all optimizers',
       x = NULL, y = 'Final Walsh correlation') +
  paper_theme(9) +
  theme(legend.position = 'none')

p_ladder <- compose_C_four_panel(
  list(pA_ladder, pB_ladder, pC_ladder, pD_ladder),
  'C_ladder_necessity', 7.2, 5.4, root)
save_png(p_ladder, 'C_ladder_necessity', 7.2, 5.4)

# ════════════════════════════════════════════════════════════════════════════
# C_matched_grid_timing (appendix) — emergence step vs L, all 3 optimizers, on a
# SINGLE shared evaluation cadence (eval_every=100, 10 seeds/cell). This removes
# the cadence-confound objection to the fine-grid timing table: at matched cadence
# SGDM is unambiguously slowest and AdamW separates from Muon at long context;
# Muon and AdamW are grid-floored equal at L=64/128 (the reason the fine grid was
# used for those cells in the main body).
# ════════════════════════════════════════════════════════════════════════════
cat('\n3. C_matched_grid_timing\n')

emerge_dir <- file.path(res_dir, 'induction_emergence')
THR <- 0.5  # emergence threshold = icl_score crossing (matches _meta)

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

LENS <- c(64L, 128L, 256L)
emerge_rows <- bind_rows(lapply(OPTS, function(opt) {
  bind_rows(lapply(LENS, function(L) {
    paths <- Sys.glob(file.path(emerge_dir, sprintf('%s_L%d_s*.jsonl', opt, L)))
    vals  <- sapply(paths, read_emergence_step)
    vals  <- vals[!is.na(vals)]
    if (length(vals) == 0) return(NULL)
    data.frame(opt = opt, L = L, emergence_step = vals, stringsAsFactors = FALSE)
  }))
}))

emerge_summary <- emerge_rows %>%
  group_by(opt, L) %>%
  summarise(m = mean(emergence_step),
            s = if (n() > 1) sd(emergence_step) else 0,
            .groups = 'drop') %>%
  mutate(optf = factor(OPTLAB[opt], levels = c('AdamW', 'Muon', 'SGDM')))

cat('   matched-grid emergence (eval_every=100, 10 seeds):\n')
print(as.data.frame(emerge_summary[, c('opt', 'L', 'm', 's')]))

emerge_scatter <- emerge_rows %>%
  mutate(optf = factor(OPTLAB[opt], levels = c('AdamW', 'Muon', 'SGDM')))

p_grid <- ggplot(emerge_summary, aes(L, m, colour = optf, group = optf)) +
  geom_point(data = emerge_scatter,
             aes(L, emergence_step, colour = optf),
             position = position_jitter(width = 0.04, height = 0, seed = 11),
             size = 1.4, alpha = 0.35, inherit.aes = FALSE) +
  geom_errorbar(aes(ymin = pmax(m - s, 1), ymax = m + s),
                width = 0.06, linewidth = 0.45) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 3.0) +
  annotate('text', x = 64, y = 130,
           label = 'Muon & AdamW at grid floor (100 steps)',
           hjust = 0.1, size = 2.6, colour = '#1a1a1a') +
  scale_x_log10(breaks = c(64, 128, 256)) +
  scale_y_log10(breaks = c(100, 200, 300, 500),
                labels = c('100', '200', '300', '500')) +
  coord_cartesian(ylim = c(80, 650)) +
  scale_colour_manual(values = c('AdamW' = OPT$AdamW, 'Muon' = OPT$Muon,
                                 'SGDM' = OPT$SGDM), name = 'Optimizer') +
  labs(title = 'Matched-cadence in-context emergence step vs context length',
       x = 'Context length L', y = 'Emergence step (log scale)') +
  paper_theme(9)
save_png(p_grid, 'C_matched_grid_timing', 5.0, 3.4)

cat('\n=== All new Paper C figures written ===\n')

#!/usr/bin/env python3
"""Analyze G002 Muon ownership lr-sensitivity sweep and render C-paper figure."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median, mean

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'experiments' / 'results' / 'arch_staircase_optaxis_lrsweep'
FIGS = ROOT / 'papers' / 'figs'
FIG = FIGS / 'C_ownership_lr_sensitivity.png'
VERDICT = RESULTS / 'lrsweep_verdict.json'


def load_summary(path: Path) -> dict:
    last = ''
    with path.open() as f:
        for line in f:
            if line.strip():
                last = line
    obj = json.loads(last)
    if '_summary' not in obj:
        raise RuntimeError(f'incomplete jsonl: {path}')
    return obj['_summary']


def q(vals, pct):
    vals = sorted(vals)
    if not vals:
        return float('nan')
    pos = (len(vals)-1) * pct
    lo = math.floor(pos); hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi-pos) + vals[hi] * (pos-lo)


def main() -> int:
    rows = []
    for p in sorted(RESULTS.glob('muon_l2_*_lr*_s*.jsonl')):
        s = load_summary(p)
        rows.append({
            'path': str(p.relative_to(ROOT)),
            'muon_lr': float(s['muon_lr']),
            'freeze': s['freeze'],
            'seed': int(s['seed']),
            'final_fit_corr': float(s['final_fit_corr']),
            'final_deg4': float(s['final_deg_corr']['4']),
            'n_learned': int(s['n_learned']),
        })
    if len(rows) != 18:
        raise RuntimeError(f'expected 18 rows, found {len(rows)}')
    lrs = sorted({r['muon_lr'] for r in rows})
    freezes = ['none', 'attn']
    agg = []
    for lr in lrs:
        by = {fz: [r for r in rows if r['muon_lr'] == lr and r['freeze'] == fz]
              for fz in freezes}
        if any(len(v) != 3 for v in by.values()):
            raise RuntimeError(f'missing cells for lr={lr}: {[len(by[f]) for f in freezes]}')
        m = {}
        for fz, vals in by.items():
            deg = [r['final_deg4'] for r in vals]
            fit = [r['final_fit_corr'] for r in vals]
            m[fz] = {
                'median_deg4': median(deg),
                'mean_deg4': mean(deg),
                'q25_deg4': q(deg, 0.25),
                'q75_deg4': q(deg, 0.75),
                'median_fit_corr': median(fit),
                'min_fit_corr': min(fit),
                'max_fit_corr': max(fit),
                'seeds': [r['seed'] for r in vals],
                'deg4_values': deg,
                'fit_corr_values': fit,
            }
        agg.append({
            'muon_lr': lr,
            'none': m['none'],
            'attn': m['attn'],
            'delta_attn_minus_none_deg4': m['attn']['median_deg4'] - m['none']['median_deg4'],
            'delta_attn_minus_none_fit': m['attn']['median_fit_corr'] - m['none']['median_fit_corr'],
        })
    overall = {
        'n_cells': len(rows),
        'lrs': lrs,
        'seeds': sorted({r['seed'] for r in rows}),
        'max_abs_delta_deg4_by_lr': max(abs(a['delta_attn_minus_none_deg4']) for a in agg),
        'median_abs_delta_deg4_by_lr': median([abs(a['delta_attn_minus_none_deg4']) for a in agg]),
        'attn_min_median_deg4': min(a['attn']['median_deg4'] for a in agg),
        'none_min_median_deg4': min(a['none']['median_deg4'] for a in agg),
        'interpretation': (
            'Muon attention-freeze does not reproduce the AdamW attention bottleneck. '
            'At lr 0.01 and 0.02 the median degree-4 delta is near zero; at lr 0.04, '
            'the unfrozen arm is less stable while the attention-frozen arm still learns degree 4.'
        ),
    }
    verdict = {'rows': rows, 'by_lr': agg, 'overall': overall}
    RESULTS.mkdir(parents=True, exist_ok=True)
    VERDICT.write_text(json.dumps(verdict, indent=2) + '\n')

    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.size': 9.5,
        'axes.labelsize': 10.5,
        'axes.titlesize': 11.0,
        'legend.fontsize': 9.5,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'figure.dpi': 150,
    })
    x = list(range(len(lrs)))
    fig, ax = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
    for fz, color, marker, label in [
        ('none', '#3b6fb6', 'o', 'unfrozen'),
        ('attn', '#d55e00', 's', 'attention frozen'),
    ]:
        y = [a[fz]['median_deg4'] for a in agg]
        yerr = [[a[fz]['median_deg4'] - a[fz]['q25_deg4'] for a in agg],
                [a[fz]['q75_deg4'] - a[fz]['median_deg4'] for a in agg]]
        ax.errorbar(x, y, yerr=yerr, marker=marker, linewidth=2.4, capsize=4,
                    color=color, label=label)
        for xi, a in zip(x, agg):
            vals = a[fz]['deg4_values']
            jitter = -0.045 if fz == 'none' else 0.045
            ax.scatter([xi + jitter] * len(vals), vals, s=22, color=color, alpha=0.45,
                       edgecolors='none')
    for xi, a in zip(x, agg):
        d = a['delta_attn_minus_none_deg4']
        ax.annotate(f"Δ={d:+.3f}", xy=(xi, 0.632),
                    ha='center', va='bottom', fontsize=9.5)
    ax.axhline(0.0, color='0.75', linewidth=1)
    ax.set_xticks(x, [f"{lr:g}" for lr in lrs])
    ax.set_xlabel('Muon learning rate')
    ax.set_ylabel('Final degree-4 Walsh corr. (median ± IQR)')
    ax.set_title('Muon lr sweep: attention freeze does not block degree 4')
    ax.set_ylim(-0.08, 0.70)
    ax.grid(axis='y', color='0.90', linewidth=0.8)
    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.01, 0.05))
    ax.text(0.99, 0.02, '3 seeds/arm; d=128, L=2, 4000 steps', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8.5, color='0.35')
    fig.savefig(FIG, dpi=240, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    print(json.dumps({'verdict': str(VERDICT), 'figure': str(FIG), 'overall': overall}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Draw paper_v2 figures from paper_v2/data/derived.json (run paper_v2.data.derive first).

    uv run --extra dev python -m paper_v2.figures.make_figures
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
D = json.loads((HERE.parent / 'data/derived.json').read_text())
C = D['confirmation']
TASKS = ('access', 'routing')
PLOT_TASKS = ('routing', 'access')  # routing (competent) first
FAMILIES = ('superseded', 'updated_other', 'entity_mention', 'unassigned')
PLOT_FAMILIES = ('superseded', 'updated_other', 'static_other', 'entity_mention', 'unassigned')
LABELS = {'superseded': 'Superseded assignment', 'updated_other': 'Other attribute, updated',
          'static_other': 'Other attribute, not updated',
          'entity_mention': 'Mentioned by entity', 'unassigned': 'Unattached word'}
COLORS = {'superseded': '#1f4e79', 'updated_other': '#7f7f7f', 'static_other': '#5b9bd5', 'entity_mention': '#c55a11', 'unassigned': '#bfbfbf'}
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
                     'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'STIXGeneral'],
                     'mathtext.fontset': 'stix'})
INK, GRID = '#222222', '#e6e6e6'
# Main-text figures: one colour per condition, one marker per series; dots with interval whiskers.
COND = {'Main design': ('#1f3b5c', 'o'), 'Balanced': ('#7aa6d6', 's'), 'Thinking': ('#4f9a74', 'D')}


def tidy(ax, axis='y'):
    """Light grid behind the data, thin ink spines, no tick marks."""
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(INK)
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(length=0, labelsize=7, colors=INK)


def dot(ax, x, y, b, color, marker, horizontal=False, **kw):
    lo, hi = b['ci']
    if horizontal:
        ax.plot([lo, hi], [y, y], color=color, lw=1.1, solid_capstyle='round', zorder=2)
    else:
        ax.plot([x, x], [lo, hi], color=color, lw=1.1, solid_capstyle='round', zorder=2)
    ax.plot(*((b['mean'], y) if horizontal else (x, b['mean'])), marker=marker, color=color, ms=3.6,
            mec='white', mew=0.5, ls='none', zorder=3, **kw)


def err(b):
    return [[b['mean'] - b['ci'][0]], [b['ci'][1] - b['mean']]]


def constructions(conf=None, path='constructions.pdf', static=None):
    """static: the model's static-other results (routing only; paper_v2/sections/A-protocol-history.tex, part B)."""
    conf = conf or C
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for ax, (title, get, xlabel) in zip(axes, [
            ('(a) Decision preference shift $D_c$',
             lambda t, f: conf['constructions'][f'{t}:{f}'], 'nats (wrong minus correct action log-odds shift)'),
            ('(b) Generated wrong-action switches',
             lambda t, f: {'mean': conf['switches'][f'{t}:all'][f'switch_{f}']['mean'],
                           'ci': conf['switches'][f'{t}:all'][f'switch_{f}']['confidence_interval']},
             'net switch rate')]):
        for i, t in enumerate(PLOT_TASKS):
            for j, f in enumerate(PLOT_FAMILIES):
                y = i * 6 + j
                if f == 'static_other':
                    if not static or t != 'routing':
                        continue
                    b = static['constructions'][f]['D' if ax is axes[0] else 'switch']
                else:
                    b = get(t, f)
                ax.barh(y, b['mean'], color=COLORS[f], height=0.8, label=LABELS[f] if i == 0 else None)
                ax.errorbar(b['mean'], y, xerr=err(b), color='black', lw=0.8, capsize=1.5)
        ax.set_yticks([2, 8], ['Routing', 'Access\n(not competent)'])
        ax.invert_yaxis()
        ax.axvline(0, color='black', lw=0.6)
        ax.set_title(title, loc='left')
        ax.set_xlabel(xlabel)
    axes[1].set_yticklabels([])  # task labels shown once, on the left panel
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc='lower center', ncol=3, fontsize=7)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(HERE / path)


def competence():
    cells = [(t, c) for t in PLOT_TASKS for c in ('sequential', 'interleaved', 'competing')]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    xs = range(len(cells))
    comp = C['competence']
    hist = C['historical_retrieval']
    series = [
        ('Current state, asked directly', [comp[f'{t}:{c}']['direct_accuracy'] for t, c in cells], 'o', '#000000'),
        ('Initial state, asked directly', [hist[t]['mean'] for t, c in cells], 's', '#7f7f7f'),
        ('Decision, no history', [comp[f'{t}:{c}']['current_only_accuracy'] for t, c in cells], 'D', '#2e75b6'),
        ('Decision, old value implies correct action',
         [C['member_accuracy'][f'{t}:{c}:superseded']['old_implies_correct']['mean'] for t, c in cells], '^', '#70ad47'),
        ('Decision, old value implies wrong action',
         [C['member_accuracy'][f'{t}:{c}:superseded']['old_implies_wrong']['mean'] for t, c in cells], 'v', '#c00000'),
    ]
    for k, (label, ys, m, col) in enumerate(series):
        ax.plot([x + (k - 2) * 0.08 for x in xs], ys, m, color=col, label=label, ms=4)
    ax.axhline(0.95, color='black', lw=0.6, ls=':', label='Threshold, retrieval (0.95)')
    ax.axhline(0.90, color='#2e75b6', lw=0.6, ls=':', label='Threshold, decision (0.90)')
    ax.axvline(2.5, color='black', lw=0.4)
    ax.set_xticks(list(xs), [f'{t}\n{c}' for t, c in cells])
    ax.set_ylabel('strict accuracy')
    ax.set_ylim(0.6, 1.0)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=6.5, loc='lower center', ncol=3, columnspacing=1.0, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(HERE / 'competence.pdf')


def replication():
    splits = [('v1b_development', 'v1b dev'), ('v1b_gate', 'v1b gate'), ('v2_development', 'v2 dev'),
              ('v2_gate', 'v2 gate'), ('confirmation', 'v2 confirmation\n(held-out template)')]
    stats = [('D_superseded', '$D_{\\mathrm{S}}$'), ('Delta_semantic', '$\\Delta_{\\mathrm{SO}}$'),
             ('Delta_mention', '$\\Delta_{\\mathrm{SM}}$')]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.2), sharey=True)
    for ax, (key, title) in zip(axes, stats):
        for i, (s, _) in enumerate(splits):
            src = C['primary'] if s == 'confirmation' else D['replication'][s]
            for k, (t, m) in enumerate(zip(TASKS, ('o', 's'))):
                b = src[t][key]
                ax.errorbar(b['mean'], i + (k - 0.5) * 0.25, xerr=err(b), fmt=m, ms=3.5, lw=0.8,
                            color='#1f4e79' if t == 'access' else '#c55a11',
                            label=t.capitalize() if (i == 0 and ax is axes[0]) else None)
        ax.axvline(0, color='black', lw=0.6)
        ax.set_title(title)
        ax.set_xlabel('nats')
    axes[0].set_yticks(range(len(splits)), [l for _, l in splits])
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(HERE / 'replication.pdf')


def conditions():
    """Qwen3-8B routing: main design, balanced subset and thinking mode, per construction (main-text figure)."""
    B, T = D.get('balanced'), D.get('think', {}).get('Qwen3-8B')
    if not B or not T:
        return
    sw = C['switches']['routing:all']
    rows = {'Main design': ({f: C['constructions'][f'routing:{f}'] for f in FAMILIES},
                            {f: {'mean': sw[f'switch_{f}']['mean'], 'ci': sw[f'switch_{f}']['confidence_interval']}
                             for f in FAMILIES}),
            'Balanced': ({f: B['constructions'][f]['D'] for f in FAMILIES},
                         {f: B['constructions'][f]['switch'] for f in FAMILIES}),
            'Thinking': (None, {f: T['constructions'][f]['switch'] for f in FAMILIES})}
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.75))
    for k, (ax, title) in enumerate(zip(axes, ['(a) Score shift $D$ (nats)', '(b) Net switch rate'])):
        conds = [c for c in rows if rows[c][k] is not None]
        for i, c in enumerate(conds):
            for j, f in enumerate(FAMILIES):
                dot(ax, j + (i - (len(conds) - 1) / 2) * 0.22, None, rows[c][k][f], *COND[c],
                    label=c if j == 0 else None)
        ax.set_xticks(range(len(FAMILIES)), ['S', 'O', 'M', 'U'])
        ax.set_xlim(-0.5, len(FAMILIES) - 0.5)
        ax.axhline(0, color=INK, lw=0.6, zorder=1)
        ax.spines['bottom'].set_visible(False)
        ax.set_title(title, loc='left', fontsize=7.5, color=INK)
        tidy(ax)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(-0.035, None)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, loc='upper center', ncol=3, handletextpad=0.2,
               columnspacing=1.2, borderaxespad=0.1)
    fig.tight_layout(pad=0.2, w_pad=1.0, rect=(0, 0, 1, 0.9))
    fig.savefig(HERE / 'conditions.pdf')


MODEL_ORDER = ('Qwen3-8B', 'Qwen3-14B', 'Gemma 4 12B', 'Qwen3.5 9B', 'Granite 4.1 8B')


def models():
    """Routing, each confirmed model's competent cells: D_S and net switch rates S and M (main-text figure)."""
    ms = [m for m in MODEL_ORDER if D['models'].get(m, {}).get('confirmation', {}).get('competent_confirmation', {}).get('routing')]
    get = lambda m: D['models'][m]['confirmation']['competent_confirmation']['routing']
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.7), sharey=True)
    for i, m in enumerate(ms):
        c = get(m)
        dot(axes[0], None, i, c['D_superseded'], COLORS['superseded'], 'o', horizontal=True)
        dot(axes[1], None, i - 0.14, c['switch_superseded'], COLORS['superseded'], 'o', horizontal=True,
            label='S, superseded' if i == 0 else None)
        dot(axes[1], None, i + 0.14, c['switch_entity_mention'], COLORS['entity_mention'], 'D', horizontal=True,
            label='M, mention' if i == 0 else None)
    for ax, title in zip(axes, ['(a) $D_{\\mathrm{S}}$ (nats)', '(b) Net switch rate']):
        ax.axvline(0, color=INK, lw=0.6, zorder=1)
        ax.spines['left'].set_visible(False)
        ax.set_title(title, loc='left', fontsize=7.5, color=INK)
        tidy(ax, axis='x')
    axes[0].set_yticks(range(len(ms)), ms)
    axes[0].set_ylim(len(ms) - 0.5, -0.5)
    axes[0].set_xlim(left=0)
    axes[1].set_xlim(left=-0.01)
    fig.legend(*axes[1].get_legend_handles_labels(), frameon=False, fontsize=7, loc='upper center', ncol=2,
               handletextpad=0.2, columnspacing=1.2, borderaxespad=0.1)
    fig.tight_layout(pad=0.2, w_pad=0.8, rect=(0, 0, 1, 0.88))
    fig.savefig(HERE / 'models.pdf')


if __name__ == '__main__':
    S = D.get('static_other', {})
    constructions(static=S.get('Qwen3-8B'))
    for m, entry in D['models'].items():  # per-model appendix figures for every other confirmed model
        if m != 'Qwen3-8B' and 'confirmation' in entry:
            constructions(entry['confirmation'], static=S.get(m), path=f"constructions_{m.lower().replace(' ', '_').replace('-', '_')}.pdf")
    competence()
    replication()
    conditions()
    models()

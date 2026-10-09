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
FAMILIES = ('superseded', 'updated_other', 'entity_mention', 'unassigned')
LABELS = {'superseded': 'Superseded assignment', 'updated_other': 'Other attribute, updated',
          'entity_mention': 'Mentioned by entity', 'unassigned': 'Unattached word'}
COLORS = {'superseded': '#1f4e79', 'updated_other': '#7f7f7f', 'entity_mention': '#c55a11', 'unassigned': '#bfbfbf'}
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})


def err(b):
    return [[b['mean'] - b['ci'][0]], [b['ci'][1] - b['mean']]]


def constructions():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for ax, (title, get, xlabel) in zip(axes, [
            ('(a) Decision preference shift $D_c$',
             lambda t, f: C['constructions'][f'{t}:{f}'], 'nats (wrong minus correct action log-odds shift)'),
            ('(b) Generated wrong-action switches',
             lambda t, f: {'mean': C['switches'][f'{t}:all'][f'switch_{f}']['mean'],
                           'ci': C['switches'][f'{t}:all'][f'switch_{f}']['confidence_interval']},
             'net switches per history')]):
        for i, t in enumerate(TASKS):
            for j, f in enumerate(FAMILIES):
                y = i * 5 + j
                b = get(t, f)
                ax.barh(y, b['mean'], color=COLORS[f], height=0.8, label=LABELS[f] if i == 0 else None)
                ax.errorbar(b['mean'], y, xerr=err(b), color='black', lw=0.8, capsize=1.5)
        ax.set_yticks([1.5, 6.5], ['Access', 'Routing'])
        ax.invert_yaxis()
        ax.axvline(0, color='black', lw=0.6)
        ax.set_title(title, loc='left')
        ax.set_xlabel(xlabel)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc='lower center', ncol=4, fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(HERE / 'constructions.pdf')


def competence():
    cells = [(t, c) for t in TASKS for c in ('sequential', 'interleaved', 'competing')]
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
    ax.axhline(0.95, color='black', lw=0.4, ls=':')
    ax.axhline(0.90, color='#2e75b6', lw=0.4, ls=':')
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
    stats = [('D_superseded', '$D_{\\mathrm{superseded}}$'), ('Delta_semantic', '$\\Delta_{\\mathrm{semantic}}$'),
             ('Delta_mention', '$\\Delta_{\\mathrm{mention}}$')]
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


if __name__ == '__main__':
    constructions()
    competence()
    replication()

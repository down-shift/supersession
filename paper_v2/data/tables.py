"""Write the cross-model LaTeX tables from paper_v2/data/derived.json (run paper_v2.data.derive first).

    uv run --extra dev python -m paper_v2.data.tables

Outputs paper_v2/data/tables/{cross_model,models_gate,joint_by_model}.tex, \\input by the manuscript, so these
numbers are never typed by hand.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / 'derived.json').read_text())
ORDER = ['Qwen3-8B', 'Qwen3-14B', 'Gemma 3 4B', 'Gemma 3 12B', 'Granite 3.1 8B', 'Mistral 7B v0.3', 'Falcon3 7B']
CELLS = [('routing', 'sequential'), ('routing', 'interleaved'), ('routing', 'competing'),
         ('access', 'sequential'), ('access', 'interleaved'), ('access', 'competing')]
ABBR = {'sequential': 'seq.', 'interleaved': 'int.', 'competing': 'comp.'}


def num(x, signed=False):
    return f'{x:+.3f}' if signed else f'{x:.3f}'


def est(b, signed=True):
    lo, hi = b['ci']
    return f"${num(b['mean'], signed)}$ {{\\scriptsize $[{num(lo, signed)}, {num(hi, signed)}]$}}"


def models():
    return [m for m in ORDER if m in D['models']]


def gate_rule_note(m, conf):
    """Entries under the amendment's gate rule, when it selects different cells from the confirmation rule."""
    parts = []
    for t in ('routing', 'access'):
        g, c = conf['competent'][t], conf['competent_confirmation'][t]
        if (g and g['cells']) == (c and c['cells']):
            continue
        if g is None:
            parts.append(f'{t}: no gate-competent cell')
            continue
        cells = '+'.join(ABBR[x] for x in g['cells'])
        parts.append(f"{t} {cells} ($n={g['n']}$): $D_{{\\mathrm{{superseded}}}}={num(g['D_superseded']['mean'], True)}$, "
                     f"$\\Delta_{{\\mathrm{{semantic}}}}={num(g['Delta_semantic']['mean'], True)}$, "
                     f"$\\Delta_{{\\mathrm{{mention}}}}={num(g['Delta_mention']['mean'], True)}$, "
                     f"net switch rate ${num(g['switch_superseded']['mean'], True)}$")
    return f"For {m}, the gate rule instead selects " + '; '.join(parts) + '.' if parts else \
        f'For {m}, the gate rule selects the same cells.'


def cross_model():
    lines = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
             r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{@{}llccccc@{}}', r'\toprule',
             r'Model & Task (competent cells) & $n$ & $D_{\mathrm{superseded}}$ & $\Delta_{\mathrm{semantic}}$ & '
             r'$\Delta_{\mathrm{mention}}$ & Net switch rate \\', r'\midrule']
    notes = []
    for m in models():
        entry = D['models'][m]
        conf = entry.get('confirmation')
        if not conf:
            status = 'not eligible: no competent cell at the gate' if not entry['any_score_pass'] else 'confirmation not run'
            lines.append(f'{m} & \\multicolumn{{6}}{{l}}{{{status}}} \\\\')
            continue
        for t in ('routing', 'access'):
            c = conf['competent_confirmation'][t]
            if c is None:
                lines.append(f'{m} & {t} & \\multicolumn{{5}}{{l}}{{no competent cell}} \\\\')
                continue
            cells = ', '.join(ABBR[x] for x in c['cells'])
            lines.append(f"{m} & {t} ({cells}) & {c['n']} & {est(c['D_superseded'])} & {est(c['Delta_semantic'])} & "
                         f"{est(c['Delta_mention'])} & {est(c['switch_superseded'])} \\\\")
        notes.append(gate_rule_note(m, conf))
        lines.append(r'\midrule')
    if lines[-1] == r'\midrule':
        lines.pop()
    lines += [r'\bottomrule', r'\end{tabular}}',
              r'\caption{Confirmation results per model, on the task-cells that met the competence criteria '
              r'\emph{on the confirmation data} (the rule used in the text and in Table~\ref{tab:competence}). '
              r'$D$ and $\Delta$ in nats; $D_{\mathrm{superseded}}$ and $\Delta_{\mathrm{semantic}}$ with 97.5\% '
              r'intervals, $\Delta_{\mathrm{mention}}$ and the superseded net switch rate with 95\% intervals (history '
              r'bootstrap). Models are reported separately, never pooled. The replication amendment defined competent '
              r'cells by each model\textquoteright s frozen gate (Table~\ref{tab:models-gate}); gate-rule '
              r'estimates are in Appendix~\ref{app:tables}.}',
              r'\label{tab:cross-model}', r'\end{table*}']
    return '\n'.join(lines) + '\n'


def diff_note():
    out = []
    for m in models():
        conf = D['models'][m].get('confirmation')
        if not conf:
            continue
        g = set(D['models'][m]['competent_cells'])
        c = set(conf['confirmation_competent_cells'])
        for cell in sorted(g ^ c):
            t, k = cell.split(':')
            out.append(f"{m} {t} {ABBR[k]} is {'competent' if cell in g else 'not competent'} here and "
                       f"{'competent' if cell in c else 'not competent'} on the confirmation data")
    return ': ' + '; '.join(out) if out else ''


def models_gate():
    lines = [r'\begin{table}[h]', r'\centering', r'\small', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{@{}llcccc@{}}', r'\toprule',
             r'Model & Cell & Direct & No hist. & History & Comp. \\', r'\midrule']
    for m in models():
        cells = D['models'][m]['gate_cells']
        for i, (t, c) in enumerate(CELLS):
            x = cells[f'{t}:{c}']
            lines.append(f"{m if i == 0 else ''} & {t} {ABBR[c]} & {num(x['direct_accuracy'])} & "
                         f"{num(x['current_only_accuracy'])} & {num(x['historical_accuracy'])} & "
                         f"{'yes' if x['score_eligible'] else 'no'} \\\\")
        lines.append(r'\midrule')
    lines.pop()
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{Frozen gate split (384 histories, identical rows for every model): strict direct '
              r'current-state retrieval, current-only decision accuracy, decision accuracy with history, and whether '
              r'the cell is competent (direct $\ge 0.95$, no history $\ge 0.90$, invalid $\le 0.05$). This gate decided '
              r'eligibility to generate confirmation data. Labels computed on the confirmation data '
              r'(Table~\ref{tab:competence}, Table~\ref{tab:cross-model}) can differ' + diff_note() + '.}',
              r'\label{tab:models-gate}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def joint_by_model():
    lines = [r'\begin{table}[h]', r'\centering', r'\small', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{@{}llccc@{}}', r'\toprule', r'Model & Construction & Kept & Net switch rate & Switched \\',
             r'\midrule']
    labels = {'superseded': 'superseded', 'updated_other': 'other attr.', 'entity_mention': 'mention',
              'unassigned': 'unattached'}
    for m in models():
        conf = D['models'][m].get('confirmation')
        if not conf:
            continue
        for i, f in enumerate(labels):
            j = conf['joint_correctness'][f'routing:{f}']
            lines.append(f"{m if i == 0 else ''} & {labels[f]} & {j['n_retained']}/{j['n_total']} & "
                         f"{est(j['switch'])} & {j['to_wrong']} / {j['to_correct']} \\\\")
        lines.append(r'\midrule')
    lines.pop()
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{Routing histories the model demonstrably tracks (criteria of Table~\ref{tab:joint}), per '
              r'model and construction. Condition (ii), initial-state retrieval, is asked only on the superseded '
              r'members and is applied at the history level to all four constructions. Net switch rate with 95\% history-bootstrap intervals; switched: histories '
              r'switched to the wrong / to the correct action.}',
              r'\label{tab:joint-by-model}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def gate_rule():
    """Appendix paragraph: compact entries under the amendment's gate rule, where it differs."""
    notes = [gate_rule_note(m, D['models'][m]['confirmation']) for m in models() if D['models'][m].get('confirmation')]
    return (r'\paragraph{Gate-rule estimates.} The replication amendment defined competent cells by each '
            r'model\textquoteright s frozen gate. Table~\ref{tab:cross-model} uses the cells competent on the '
            r'confirmation data, the rule used throughout the text. ' + ' '.join(notes) + '\n')


def main():
    out = HERE / 'tables'
    out.mkdir(exist_ok=True)
    for name, fn in [('cross_model', cross_model), ('models_gate', models_gate), ('joint_by_model', joint_by_model),
                     ('gate_rule', gate_rule)]:
        (out / f'{name}.tex').write_text(fn())
        print(out / f'{name}.tex')


if __name__ == '__main__':
    main()

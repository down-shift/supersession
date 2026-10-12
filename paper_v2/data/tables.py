"""Write the cross-model LaTeX tables from paper_v2/data/derived.json (run paper_v2.data.derive first).

    uv run --extra dev python -m paper_v2.data.tables

Outputs paper_v2/data/tables/*.tex, \\input by the
manuscript, so these numbers are never typed by hand.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / 'derived.json').read_text())
ORDER = ['Qwen3-8B', 'Qwen3-14B', 'Gemma 3 4B', 'Gemma 3 12B', 'Granite 3.1 8B', 'Mistral 7B v0.3', 'Falcon3 7B',
         'Ministral 3 14B', 'Gemma 4 12B', 'Qwen3.5 9B', 'Granite 4.1 8B']
CELLS = [('routing', 'sequential'), ('routing', 'interleaved'), ('routing', 'competing'),
         ('access', 'sequential'), ('access', 'interleaved'), ('access', 'competing')]
ABBR = {'sequential': 'seq.', 'interleaved': 'int.', 'competing': 'comp.'}


def num(x, signed=False):
    return f'{x:+.3f}' if signed else f'{x:.3f}'


def est(b, signed=True):
    lo, hi = b['ci']
    return f"${num(b['mean'], signed)}$ $[{num(lo, signed)}, {num(hi, signed)}]$"


def pt(b, signed=True):
    """Point estimate only; grey when its interval includes zero (intervals are in the appendix tables)."""
    lo, hi = b.get('ci') or b['confidence_interval']
    v = f"${num(b['mean'], signed)}$"
    return f'\\textcolor{{gray}}{{{v}}}' if lo <= 0 <= hi else v


def models():
    return [m for m in ORDER if m in D['models']]



def bounds(b, signed=True):
    lo, hi = b['ci']
    return f"$[{num(lo, signed)}, {num(hi, signed)}]$"


def interval_tables(confirmation_selected=False):
    rule = 'competent_confirmation' if confirmation_selected else 'competent'
    base = 'tab:cross-model-confirmation' if confirmation_selected else 'tab:cross-model-ci'
    groups = [
        ('scores', [('D_superseded', r'$D_{\mathrm{S}}$'), ('Delta_semantic', r'$\Delta_{\mathrm{SO}}$'),
                    ('Delta_mention', r'$\Delta_{\mathrm{SM}}$')]),
        ('switches', [('switch_superseded', 'S'), ('switch_diff_attribute', 'S $-$ O'),
                      ('switch_diff_mention_minus_superseded', 'M $-$ S')])]
    blocks = []
    for kind, metrics in groups:
        lines = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{5pt}',
                 r'\begin{tabular}{@{}llrlrr@{}}', r'\toprule',
                 r'Model & Task / cells & $n$ & Quantity & Estimate & Interval \\', r'\midrule']
        for m in models():
            conf = D['models'][m].get('confirmation')
            if not conf:
                continue
            first_model = True
            for task in ('routing', 'access'):
                c = conf[rule][task]
                if c is None:
                    continue
                cells = ','.join({'sequential': 's', 'interleaved': 'i', 'competing': 'c'}[k] for k in c['cells'])
                for i, (key, label) in enumerate(metrics):
                    b = c[key]
                    lines.append(' & '.join([m if first_model else '', f'{task} {cells}' if i == 0 else '',
                                             str(c['n']) if i == 0 else '', label,
                                             f"${num(b['mean'], True)}$", bounds(b)]) + r' \\')
                    first_model = False
            lines.append(r'\midrule')
        lines[-1] = r'\bottomrule'
        label = base if kind == 'scores' else base + '-switches'
        selection = ('Post hoc confirmation-selected robustness estimates' if confirmation_selected
                     else r'Primary confirmation estimates on frozen-gate-selected cells (Table~\ref{tab:cross-model})')
        caption = selection + (r': score shifts in nats. Intervals are 97.5\% for $D_{\mathrm{S}}$ and '
                               r'$\Delta_{\mathrm{SO}}$, 95\% for $\Delta_{\mathrm{SM}}$.' if kind == 'scores'
                               else r': net switch rates and paired differences, with 95\% intervals.')
        caption += r' Cells: s = sequential, i = interleaved, c = competing; $n$ counts histories.'
        lines += [r'\end{tabular}', r'\caption{' + caption + '}', r'\label{' + label + '}', r'\end{table*}']
        blocks.append('\n'.join(lines))
    return '\n\n'.join(blocks) + '\n'

def cross_model(ci=False, confirmation_selected=False):
    """Gate-selected primary estimates; optional post hoc confirmation selection. ci=True adds intervals."""
    if ci:
        return interval_tables(confirmation_selected)
    f = pt
    lines = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}']
    if ci:
        lines.append(r'\resizebox{\textwidth}{!}{%')
    lines += [r'\begin{tabular}{@{}llrcccccc@{}}', r'\toprule',
              r' & & & \multicolumn{3}{c}{Scores (nats)} & \multicolumn{3}{c}{Net switch rate} \\',
              r'\cmidrule(lr){4-6}\cmidrule(lr){7-9}',
              r'Model & Cells & $n$ & $D_{\mathrm{S}}$ & $\Delta_{\mathrm{SO}}$ & $\Delta_{\mathrm{SM}}$ & S & '
              r'S $-$ O & M $-$ S \\', r'\midrule']
    for m in models():
        conf = D['models'][m].get('confirmation')
        if not conf:  # models without a confirmation are listed in the gate table
            continue
        for t in ('routing', 'access'):
            c = conf['competent_confirmation' if confirmation_selected else 'competent'][t]
            if c is None:
                continue
            cells = t + ' ' + ', '.join(ABBR[x] for x in c['cells'])
            lines.append(f"{m} & {cells} & {c['n']} & {f(c['D_superseded'])} & {f(c['Delta_semantic'])} & "
                         f"{f(c['Delta_mention'])} & {f(c['switch_superseded'])} & {f(c['switch_diff_attribute'])} & "
                         f"{f(c['switch_diff_mention_minus_superseded'])} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}' + ('}' if ci else '')]
    if confirmation_selected:
        lines.append(r'\caption{Post hoc robustness comparison: confirmation estimates on cells selected by '
                     r'confirmation competence. Primary gate-selected estimates are in Table~\ref{tab:cross-model-ci}. '
                     r'Intervals: 97.5\% for $D_{\mathrm{S}}$ and $\Delta_{\mathrm{SO}}$, 95\% otherwise.}')
        lines += [r'\label{tab:cross-model-confirmation}', r'\end{table*}']
    elif ci:
        lines.append(r'\caption{Table~\ref{tab:cross-model} with intervals: 97.5\% for $D_{\mathrm{S}}$ and '
                     r'$\Delta_{\mathrm{SO}}$, 95\% otherwise (history bootstrap). Cells are selected by the frozen gate.}')
        lines += [r'\label{tab:cross-model-ci}', r'\end{table*}']
    else:
        lines.append(r'\caption{Primary confirmation estimates on cells selected by each model\textquoteright s '
                     r'frozen gate. Access in Qwen3-8B fails confirmation competence. Grey: interval includes zero '
                     r'(Tables~\ref{tab:cross-model-ci} and~\ref{tab:cross-model-ci-switches}); post hoc confirmation selection: '
                     r'Table~\ref{tab:cross-model-confirmation}.}')
        lines += [r'\label{tab:cross-model}', r'\end{table*}']
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
    lines = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{5pt}', r'\begin{tabular}{@{}llcccc@{}}', r'\toprule',
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
              r'\caption{Frozen gate split (384 histories, the same rows for every model): accuracy and competence '
              r'per cell. This gate decided eligibility for confirmation.}',
              r'\label{tab:models-gate}', r'\end{table*}']
    return '\n'.join(lines) + '\n'


def joint_by_model():
    lines = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{5pt}', r'\begin{tabular}{@{}llccc@{}}', r'\toprule', r'Model & Construction & Kept & Net switch rate & Switched \\',
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
              r'\caption{Routing histories each model tracks (criteria of \S\ref{sec:results-h1}), with 95\% intervals. '
              r'Retained sets differ by construction; Table~\ref{tab:tracked-models} uses a common set.}',
              r'\label{tab:joint-by-model}', r'\end{table*}']
    return '\n'.join(lines) + '\n'


def macros():
    """Confirmation competence discrepancies and Qwen3-14B paired switch differences."""
    c = D['models']['Qwen3-14B']['confirmation']['competent']['routing']
    inline = lambda b: f"${num(b['mean'], True)}$ $[{num(b['ci'][0], True)}, {num(b['ci'][1], True)}]$"
    return '\n'.join([
        r'\newcommand{\gateConfDiffs}{Competence labels on the confirmation data can differ from the gate labels '
        r'(Table~\ref{tab:models-gate})' + diff_note() + '.}',
        r'\newcommand{\swattrfourteen}{' + inline(c['switch_diff_attribute']) + '}',
        r'\newcommand{\swmentionfourteen}{' + inline(c['switch_diff_mention_minus_superseded']) + '}']) + '\n'


TRACKED_ROWS = [('superseded', 'S'), ('updated_other', 'O'), ('entity_mention', 'M'), ('unassigned', 'U')]
TRACKED_DIFFS = [('mention_minus_superseded', 'M $-$ S'), ('superseded_minus_other', 'S $-$ O'),
                 ('other_minus_unattached', 'O $-$ U')]
TRACKED_CAPTION = (r'Net switch rates on the routing histories tracked under all four constructions (post hoc), with '
                   r'95\% intervals; lower rows are paired differences. S, O, M, U: \S\ref{sec:design}.')


def tracked():
    """Post hoc, Qwen3-8B (main text): constructions compared on the routing histories tracked under all four."""
    J = D['models']['Qwen3-8B']['confirmation']['joint_intersection']['routing']
    lines = [r'\begin{table}[t]', r'\centering', r'\small', r'\begin{tabular}{@{}lcclc@{}}', r'\toprule',
             r'Constr. & Net switch & W/C & Pair & Difference \\', r'\midrule']
    rows = [(f"{label} & {pt(J['constructions'][f]['switch'])} & "
             f"{J['constructions'][f]['to_wrong']}/{J['constructions'][f]['to_correct']}") for f, label in TRACKED_ROWS]
    diffs = [f"{label} & {pt(J['differences'][k]['switch'])}" for k, label in TRACKED_DIFFS] + [' & ']
    lines += [f'{r} & {d}' + r' \\' for r, d in zip(rows, diffs)]
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{Qwen3-8B, routing: the ' + f"{J['n_retained']} of {J['n_total']}" + r' histories tracked under '
              r'all four constructions (post hoc). W/C: switched to wrong / correct. Grey: interval includes zero '
              r'(Table~\ref{tab:tracked-models}).}', r'\label{tab:tracked}', r'\end{table}']
    return '\n'.join(lines) + '\n'



def tracked_models():
    """Common tracked sets, with models arranged vertically to retain readable intervals."""
    lines = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{8pt}',
             r'\begin{tabular}{@{}llrr@{}}', r'\toprule',
             r'Model / retained histories & Construction or difference & Net switch & 95\% interval \\', r'\midrule']
    for m in models():
        conf = D['models'][m].get('confirmation')
        if not conf:
            continue
        j = conf['joint_intersection']['routing']
        rows = [(label, j['constructions'][key]['switch']) for key, label in TRACKED_ROWS]
        rows += [(label, j['differences'][key]['switch']) for key, label in TRACKED_DIFFS]
        for i, (label, b) in enumerate(rows):
            model = m if i == 0 else (f"$n={j['n_retained']}/{j['n_total']}$" if i == 1 else '')
            lines.append(' & '.join([model, label, f"${num(b['mean'], True)}$", bounds(b)]) + r' \\')
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines += [r'\end{tabular}', r'\caption{' + TRACKED_CAPTION + '}',
              r'\label{tab:tracked-models}', r'\end{table*}']
    return '\n'.join(lines) + '\n'


def static_other():
    """Never-updated controls: models form row groups, with separate estimate and CI columns."""
    data = D.get('static_other', {})
    lines = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{5pt}',
             r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
             r'Construction / contrast & $D$ (nats) & 95\% interval & Net switch & 95\% interval \\', r'\midrule']
    for m in models():
        if m not in data:
            continue
        lines.append(r'\multicolumn{5}{@{}l}{\textbf{' + m + r'}} \\')
        rows = [(label, data[m]['constructions'][key]) for key, label in [
            ('superseded', 'Superseded'), ('updated_other', 'Other attribute, updated'),
            ('static_other', 'Other attribute, not updated'), ('entity_mention', 'Mention'), ('unassigned', 'Unattached')]]
        rows += [(label, data[m]['contrasts'][key]) for key, label in [
            ('Delta_update', r'$\Delta_{\mathrm{ON}}$ (O $-$ N)'),
            ('Delta_attribute', r'$\Delta_{\mathrm{SO}}$ (S $-$ O)'),
            ('superseded_minus_static', r'$\Delta_{\mathrm{SN}}$ (S $-$ N)')]]
        for label, b in rows:
            lines.append(' & '.join([label, f"${num(b['D']['mean'], True)}$", bounds(b['D']),
                                     f"${num(b['switch']['mean'], True)}$", bounds(b['switch'])]) + r' \\')
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines += [r'\end{tabular}', r'\caption{Never-updated other attribute (N) on the routing confirmation '
              r'histories ($n=384$), with 95\% intervals. Prediction recorded before scoring: '
              r'$\Delta_{\mathrm{ON}} \le 0$.}', r'\label{tab:static-other}', r'\end{table*}']
    return '\n'.join(lines) + '\n'


def conditions():
    """One row group per generation condition, without scaling the table."""
    b, t = D.get('balanced'), D.get('think', {}).get('Qwen3-8B')
    if not b or not t:
        return ''
    constructions = D['confirmation']['constructions']
    switches = D['confirmation']['switches']['routing:all']
    labels = [('superseded', 'Superseded'), ('updated_other', 'Other attribute, updated'),
              ('entity_mention', 'Mention'), ('unassigned', 'Unattached')]
    lines = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{5pt}',
             r'\begin{tabular}{@{}lrrrrr@{}}', r'\toprule',
             r'Construction & $D$ (nats) & 95\% interval & Net switch & 95\% interval & W/C \\', r'\midrule']
    for group, n in [('Main design, no thinking', 384), ('Balanced, no thinking', 192), ('Main design, thinking', 384)]:
        lines.append(r'\multicolumn{6}{@{}l}{\textbf{' + group + f'}} ($n={n}$)' + r'} \\')
        for key, label in labels:
            if group == 'Main design, no thinking':
                d = constructions[f'routing:{key}']
                w = switches[f'switch_{key}']
                sw = {'mean': w['mean'], 'ci': w['confidence_interval']}
                counts = switches['switch_counts'][key]
            else:
                c = (b if group == 'Balanced, no thinking' else t)['constructions'][key]
                d = c.get('D')
                sw, counts = c['switch'], c
            lines.append(' & '.join([label, f"${num(d['mean'], True)}$" if d else '--', bounds(d) if d else '--',
                                     f"${num(sw['mean'], True)}$", bounds(sw),
                                     f"{counts['to_wrong']}/{counts['to_correct']}"]) + r' \\')
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines += [r'\end{tabular}', r'\caption{Values behind Figure~\ref{fig:conditions} (Qwen3-8B, routing), '
              r'with 95\% intervals. W/C: histories switched to wrong / correct. Thinking has no scores; '
              r'balanced and original histories differ, so their comparison is descriptive.}',
              r'\label{tab:conditions}', r'\end{table*}']
    return '\n'.join(lines) + '\n'

def main():
    out = HERE / 'tables'
    out.mkdir(exist_ok=True)
    for name, fn in [('cross_model', cross_model), ('cross_model_ci', lambda: cross_model(ci=True)), ('cross_model_confirmation', lambda: cross_model(ci=True, confirmation_selected=True)), ('models_gate', models_gate), ('joint_by_model', joint_by_model),
                     ('tracked', tracked), ('tracked_models', tracked_models), ('static_other', static_other), ('conditions', conditions), ('macros', macros)]:
        (out / f'{name}.tex').write_text(fn())
        print(out / f'{name}.tex')


if __name__ == '__main__':
    main()

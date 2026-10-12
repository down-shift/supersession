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
    return f"${num(b['mean'], signed)}$ {{\\scriptsize $[{num(lo, signed)}, {num(hi, signed)}]$}}"


def pt(b, signed=True):
    """Point estimate only; grey when its interval includes zero (intervals are in the appendix tables)."""
    lo, hi = b.get('ci') or b['confidence_interval']
    v = f"${num(b['mean'], signed)}$"
    return f'\\textcolor{{gray}}{{{v}}}' if lo <= 0 <= hi else v


def models():
    return [m for m in ORDER if m in D['models']]


def cross_model(ci=False, confirmation_selected=False):
    """Gate-selected primary estimates; optional post hoc confirmation selection. ci=True adds intervals."""
    f = est if ci else pt
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
                     r'(Table~\ref{tab:cross-model-ci}); post hoc confirmation selection: '
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
    lines = [r'\begin{table}[h]', r'\centering', r'\small', r'\setlength{\tabcolsep}{3pt}',
             r'\resizebox{\columnwidth}{!}{%', r'\begin{tabular}{@{}llcccc@{}}', r'\toprule',
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
    lines += [r'\bottomrule', r'\end{tabular}}',
              r'\caption{Frozen gate split (384 histories, the same rows for every model): accuracy and competence '
              r'per cell. This gate decided eligibility for confirmation.}',
              r'\label{tab:models-gate}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def joint_by_model():
    lines = [r'\begin{table}[h]', r'\centering', r'\small', r'\setlength{\tabcolsep}{3pt}',
             r'\resizebox{\columnwidth}{!}{%', r'\begin{tabular}{@{}llccc@{}}', r'\toprule', r'Model & Construction & Kept & Net switch rate & Switched \\',
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
    lines += [r'\bottomrule', r'\end{tabular}}',
              r'\caption{Routing histories each model tracks (criteria of \S\ref{sec:results-h1}), with 95\% intervals. '
              r'Retained sets differ by construction; Table~\ref{tab:tracked-models} uses a common set.}',
              r'\label{tab:joint-by-model}', r'\end{table}']
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
    """The same comparison for every confirmed model (Appendix B)."""
    ms = [m for m in models() if D['models'][m].get('confirmation')]
    J = {m: D['models'][m]['confirmation']['joint_intersection']['routing'] for m in ms}
    lines = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
             r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{@{}l' + 'c' * len(ms) + '@{}}', r'\toprule',
             ' & '.join([''] + ms) + r' \\',
             ' & '.join([''] + [f"$n={J[m]['n_retained']}/{J[m]['n_total']}$" for m in ms]) + r' \\', r'\midrule']
    for f, label in TRACKED_ROWS:
        lines.append(' & '.join([label] + [est(J[m]['constructions'][f]['switch']) for m in ms]) + r' \\')
    lines.append(r'\midrule')
    for k, label in TRACKED_DIFFS:
        lines.append(' & '.join([label] + [est(J[m]['differences'][k]['switch']) for m in ms]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\caption{' + TRACKED_CAPTION + '}',
              r'\label{tab:tracked-models}', r'\end{table*}']
    return '\n'.join(lines) + '\n'


def static_other():
    """Five constructions on the confirmation routing histories (paper_v2/sections/A-protocol-history.tex, part B)."""
    S = D.get('static_other', {})
    ms = [m for m in models() if m in S]
    lines = [r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{3pt}',
             r'\resizebox{\columnwidth}{!}{%', r'\begin{tabular}{@{}l' + 'cc' * len(ms) + '@{}}', r'\toprule',
             ' & '.join([''] + [f'\\multicolumn{{2}}{{c}}{{{m}}}' for m in ms]) + r' \\',
             ' & '.join([''] + ['$D$', 'Net switch'] * len(ms)) + r' \\', r'\midrule']
    for f, label in [('superseded', 'Superseded'), ('updated_other', 'Other attr., updated'),
                     ('static_other', 'Other attr., not updated'), ('entity_mention', 'Mention'), ('unassigned', 'Unattached')]:
        lines.append(' & '.join([label] + [x for m in ms for x in (est(S[m]['constructions'][f]['D']),
                                                                    est(S[m]['constructions'][f]['switch']))]) + r' \\')
    lines.append(r'\midrule')
    for k, label in [('Delta_update', '$\\Delta_{\\mathrm{ON}}$ (updated $-$ not updated)'),
                     ('Delta_attribute', '$\\Delta_{\\mathrm{SO}}$ (superseded $-$ updated)'),
                     ('superseded_minus_static', '$\\Delta_{\\mathrm{SN}}$ (superseded $-$ not updated)')]:
        lines.append(' & '.join([label] + [x for m in ms for x in (est(S[m]['contrasts'][k]['D']),
                                                                    est(S[m]['contrasts'][k]['switch']))]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}}',
              r'\caption{Never-updated other attribute (N) on the routing confirmation histories ($n=384$), with 95\% '
              r'intervals. Prediction recorded before scoring: $\Delta_{\mathrm{ON}} \le 0$.}',
              r'\label{tab:static-other}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def conditions():
    """Qwen3-8B routing: sealed confirmation, balanced subset (paper_v2/sections/A-protocol-history.tex) and thinking mode
    (paper_v2/sections/A-protocol-history.tex), per construction."""
    B, T = D.get('balanced'), D.get('think', {}).get('Qwen3-8B')
    if not B or not T:
        return ''
    C, S = D['confirmation']['constructions'], D['confirmation']['switches']['routing:all']
    sw = lambda b: {'mean': b['mean'], 'ci': b['confidence_interval']}
    cnt = lambda c: f"{c['to_wrong']}/{c['to_correct']}"
    lines = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
             r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{@{}lccccccccc@{}}', r'\toprule',
             r' & \multicolumn{3}{c}{Main design, no thinking ($n=384$)} & \multicolumn{3}{c}{Balanced, no thinking '
             r'($n=192$)} & \multicolumn{2}{c}{Main design, thinking ($n=384$)} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}',
             r'Construction & $D$ & Net switch & W/C & $D$ & Net switch & W/C & Net switch & W/C \\', r'\midrule']
    for f, label in [('superseded', 'Superseded'), ('updated_other', 'Other attr., updated'),
                     ('entity_mention', 'Mention'), ('unassigned', 'Unattached')]:
        b, t = B['constructions'][f], T['constructions'][f]
        lines.append(' & '.join([label, est(C[f'routing:{f}']), est(sw(S[f'switch_{f}'])), cnt(S['switch_counts'][f]),
                                 est(b['D']), est(b['switch']), cnt(b), est(t['switch']), cnt(t)]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}}',
              r'\caption{Values behind Figure~\ref{fig:conditions} (Qwen3-8B, routing), with 95\% intervals. W/C: '
              r'histories switched to the wrong / to the correct action.}',
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

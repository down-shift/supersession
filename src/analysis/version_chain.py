"""History-paired chain relevance, descriptive contrasts, and competence."""
from collections import defaultdict
import math
import numpy as np
from src.data.version_chain import audit
from src.analysis.metrics import bootstrap_mean_ci


def identity_transfer(base, edit):
    if base['pair_id'] != edit['pair_id'] or base['pair_direction'] != 0 or edit['pair_direction'] != 1:
        raise ValueError('E requires baseline/edit members of one matched pair')
    source, replacement = base['source_value'], base['replacement_value']
    b, e = base['candidate_logits'], edit['candidate_logits']
    return float((e[replacement]-e[source]) - (b[replacement]-b[source]))


def validate_scores(dataset, scores):
    audit(dataset)
    expected = {r['example_id']: r for r in dataset}
    if len(scores) != len(expected) or {s['example_id'] for s in scores} != set(expected):
        raise ValueError('scores must cover every dataset member exactly once')
    for s in scores:
        if any(s.get(k) != v for k, v in expected[s['example_id']].items()):
            raise ValueError('score semantic metadata differs from dataset')
        if (set(s['candidate_logits']) != set(s['candidate_values']) or
            not all(math.isfinite(float(v)) for v in s['candidate_logits'].values()) or
            s['full_vocab_rank'] < 1 or s['candidate_rank'] < 1 or
            s['full_vocab_next_token_accuracy'] not in (0, 1) or s['accuracy'] not in (0, 1)):
            raise ValueError('invalid logits/competence diagnostics')


def relevance(dataset, scores):
    validate_scores(dataset, scores)
    pairs = defaultdict(dict)
    for s in scores:
        pairs[s['pair_id']][s['pair_direction']] = s
    cells = defaultdict(dict)
    effects = []
    for p in pairs.values():
        b, e = p[0], p[1]
        effect = identity_transfer(b, e)
        key = (b['history_id'], b['depth'], b['edited_variable'], b['version_index'])
        cells[key][b['query']] = effect
        effects.append({k: b[k] for k in ('pair_id','history_id','depth','edited_variable','version_index','age_from_current','query')}
                       | {'identity_transfer_E': effect})
    rows = []
    symmetric = defaultdict(list)
    for (hid, depth, variable, index), queries in cells.items():
        other = 'z' if variable == 'x' else 'x'
        value = queries[variable]-queries[other]
        row = {'history_id': hid, 'depth': depth, 'axis': variable,
               'version_index': index, 'age_from_current': depth-index, 'R': value}
        rows.append(row)
        symmetric[(hid, depth, index)].append(value)
    for (hid, depth, index), values in symmetric.items():
        if len(values) != 2:
            raise ValueError('missing x/z relevance cell')
        rows.append({'history_id': hid, 'depth': depth, 'axis': 'symmetric',
                     'version_index': index, 'age_from_current': depth-index, 'R': float(np.mean(values))})
    return effects, rows


def competence(dataset, scores, threshold=0.98):
    validate_scores(dataset, scores)
    # Repeated baseline prompts across edit pairs are not independent trials.
    baselines = {}
    for s in scores:
        if s['pair_direction'] == 0:
            key = (s['history_id'], s['query'])
            if key in baselines and baselines[key]['candidate_logits'] != s['candidate_logits']:
                raise ValueError('identical baseline prompts have inconsistent logits')
            baselines[key] = s
    details = []
    cells = defaultdict(list)
    for s in scores:
        if s['pair_direction'] == 0:
            continue
        cells[(s['depth'], s['query'], s['orientation'], tuple(s['block_order']), 'edited')].append(s)
    for s in baselines.values():
        cells[(s['depth'], s['query'], s['orientation'], tuple(s['block_order']), 'baseline')].append(s)
    for key, ss in sorted(cells.items(), key=str):
        details.append({'depth':key[0], 'query':key[1], 'orientation':key[2],
            'block_order':list(key[3]), 'member':key[4], 'n':len(ss),
            'full_vocab_accuracy':float(np.mean([s['full_vocab_next_token_accuracy'] for s in ss])),
            'candidate_accuracy':float(np.mean([s['accuracy'] for s in ss])),
            'full_vocab_rank_max':max(s['full_vocab_rank'] for s in ss)})
    summaries = []
    qualified = []
    prefix_passed = True
    for depth in sorted({s['depth'] for s in scores}):
        ss = [s for s in baselines.values() if s['depth'] == depth]
        depth_cells = [c for c in details if c['depth'] == depth]
        passed = len(depth_cells) == 16 and all(c['full_vocab_accuracy'] >= threshold and
                      c['candidate_accuracy'] >= threshold for c in depth_cells)
        prefix_passed = prefix_passed and passed
        if prefix_passed:
            qualified.append(depth)
        previous, oldest = [], []
        for s in ss:
            chain = s['versions'][s['query']]
            l = s['candidate_logits']
            previous.append(l[chain[-1]]-l[chain[-2]])
            oldest.append(l[chain[-1]]-l[chain[0]])
        edited = [s for s in scores if s['depth'] == depth and s['pair_direction']==1]
        summaries.append({'depth':depth, 'baseline_query_count':len(ss),
            'full_vocab_accuracy':float(np.mean([s['full_vocab_next_token_accuracy'] for s in ss])),
            'candidate_accuracy':float(np.mean([s['accuracy'] for s in ss])),
            'full_vocab_rank_mean':float(np.mean([s['full_vocab_rank'] for s in ss])),
            'full_vocab_rank_max':max(s['full_vocab_rank'] for s in ss),
            'current_minus_previous_mean':float(np.mean(previous)),
            'current_minus_oldest_mean':float(np.mean(oldest)), 'competence_passed':passed,
            'edited_full_vocab_accuracy':float(np.mean([s['full_vocab_next_token_accuracy'] for s in edited])),
            'edited_candidate_accuracy':float(np.mean([s['accuracy'] for s in edited])),
            'edited_full_vocab_rank_mean':float(np.mean([s['full_vocab_rank'] for s in edited])),
            'edited_full_vocab_rank_max':max(s['full_vocab_rank'] for s in edited)})
    return {'minimum_cell_accuracy':threshold, 'qualified_depths':qualified,
            'depths':summaries, 'cells':details,
            'rule':'baseline and edited accuracy >= threshold in every query/orientation/block-order cell; cap at first failed depth'}


def bootstrap(values, draws=2000, seed=0):
    # Callers supply exactly one already-paired observation per independent history.
    x = np.asarray(values, dtype=float)
    stat = bootstrap_mean_ci(x,n_boot=draws,seed=seed)
    return {'mean':stat['mean'],'median':float(np.median(x)),
            'ci_low':stat['ci_low'],'ci_high':stat['ci_high'],'n_histories':len(x)}


def summarize(rows, draws=2000, seed=0):
    groups = defaultdict(list)
    contrasts = defaultdict(list)
    by_history = defaultdict(dict)
    for r in rows:
        groups[(r['depth'], r['axis'], r['version_index'], r['age_from_current'])].append(r['R'])
        by_history[(r['history_id'], r['depth'], r['axis'])][r['version_index']] = r['R']
    for (_, depth, axis), versions in by_history.items():
        contrasts[(depth, axis, 'current_minus_previous')].append(versions[depth]-versions[depth-1])
        if depth >= 2:
            contrasts[(depth, axis, 'previous_minus_older_obsolete_mean')].append(
                versions[depth-1]-float(np.mean([versions[i] for i in range(depth-1)])))
            for i in range(depth-1):
                contrasts[(depth, axis, f'obsolete_v{i+1}_minus_v{i}')].append(versions[i+1]-versions[i])
    summary = [{'depth':d,'axis':a,'version_index':i,'age_from_current':age,
                **bootstrap(v,draws,seed)} for (d,a,i,age),v in sorted(groups.items())]
    comparison = [{'depth':d,'axis':a,'contrast':c,**bootstrap(v,draws,seed)}
                  for (d,a,c),v in sorted(contrasts.items())]
    return summary, comparison

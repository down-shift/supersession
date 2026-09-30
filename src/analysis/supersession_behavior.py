"""History-paired control/status identity-transfer effects and diagnostics."""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

from src.analysis.metrics import bootstrap_mean_ci, trimmed_mean
from src.data.supersession_behavior import audit_behavior_dataset, CONDITIONS, FIELDS

METRICS = {'controls': ('R_live', 'R_superseded', 'R_irrelevant'),
           'controls_counterbalanced': ('R_live', 'R_superseded', 'R_irrelevant', 'R_irrelevant_counterbalanced'),
           'status': ('R_accepted_current', 'R_superseded_initial', 'R_retained_initial_after_rejection', 'R_rejected_update')}
DEFINITIONS = {
    'identity_transfer': '(replacement_after - source_after) - (replacement_before - source_before)',
    'R_live': 'mean of x/z live-bound-value edit E(relevant current query) minus E(other current query)',
    'R_superseded': 'mean of x/z obsolete-initial-value edit E(relevant current query) minus E(other current query)',
    'R_irrelevant': 'half of [E(edit first unassigned mention | query x)-E(same edit | query z) + E(edit second unassigned mention | query z)-E(same edit | query x)]; x/z label analytic slots, not bindings or variable mentions',
    'R_irrelevant_counterbalanced': 'average each initial-x/initial-z edit and query E over both unassigned slot orders (xz,zx), then compute symmetric x/z query relevance contrasts; semantic identity is independent of slot position',
    'R_accepted_current': 'symmetric relevance contrast for proposed-value edits in accepted/YES histories',
    'R_superseded_initial': 'symmetric relevance contrast for initial-value edits in accepted/YES histories',
    'R_retained_initial_after_rejection': 'symmetric relevance contrast for initial-value edits in rejected/NO histories',
    'R_rejected_update': 'symmetric relevance contrast for proposed-value edits in rejected/NO histories',
    'status_2x2_acceptance_effect': 'For each variable, average the two within-history differences in query-specific relevance R when changing only its proposed update from rejected to accepted, holding the other variable status fixed',
    'correct_answer_logit_change': 'edited minus baseline logit of the fixed baseline correct-answer token; semantic answer may change for live/current edits',
    'correct_minus_stale_change': 'edited minus baseline correct-answer/obsolete margin following the semantic roles in each member; before/after token identities saved explicitly; null when no obsolete value exists',
    'fixed_baseline_correct_minus_stale_change': 'edited minus baseline margin with both token identities fixed to the baseline correct answer and baseline obsolete value',
}


def matched_edit_effect(base, edit):
    """Keep both logit components with fixed source/replacement orientation."""
    if base['pair_direction'] != 0 or edit['pair_direction'] != 1 or base['pair_id'] != edit['pair_id']:
        raise ValueError('matched effect requires baseline 0 and edited 1 from the same pair')
    source, replacement = base['source_value'], base['replacement_value']
    lb, le = base['candidate_logits'], edit['candidate_logits']
    for value in (source, replacement, base['answer'], edit['answer'], base['stale_value'], edit['stale_value']):
        if value is not None and (value not in lb or value not in le):
            raise ValueError(f'paired candidate logits missing required value {value!r}')
    for logits in (lb, le):
        if not all(math.isfinite(float(v)) for v in logits.values()):
            raise ValueError('nonfinite candidate logits are not valid trials')
    before, after = lb[source], le[source]
    rb, ra = lb[replacement], le[replacement]
    answer, stale = base['answer'], base['stale_value']
    edited_answer, edited_stale = edit['answer'], edit['stale_value']
    if (stale is None) != (edited_stale is None):
        raise ValueError('value edit changed whether an obsolete binding exists')
    row = {k: base[k] for k in ('history_id', 'condition', 'experiment_kind', 'pair_id', 'edited_field',
                              'edited_variable', 'query', 'query_id', 'edit_status', 'source_value', 'replacement_value')}
    row['unassigned_slot_order'] = base.get('unassigned_slot_order')
    row.update(source_logit_before=before, source_logit_after=after, replacement_logit_before=rb, replacement_logit_after=ra,
               source_logit_change=after-before, replacement_logit_change=ra-rb,
               identity_transfer=(ra-after)-(rb-before),
               correct_answer_value_used=answer, edited_answer_value=edit['answer'],
               correct_answer_logit_before=lb[answer], correct_answer_logit_after=le[answer],
               correct_answer_logit_change=le[answer]-lb[answer],
               edited_answer_logit_before=lb[edit['answer']], edited_answer_logit_after=le[edit['answer']],
               semantic_target_logit_change=le[edited_answer]-lb[answer],
               stale_value_used=stale, stale_value_before=stale, stale_value_after=edited_stale,
               correct_minus_stale_before=lb[answer]-lb[stale] if stale is not None else None,
               correct_minus_stale_after=le[edited_answer]-le[edited_stale] if stale is not None else None,
               correct_minus_stale_change=(le[edited_answer]-le[edited_stale])-(lb[answer]-lb[stale]) if stale is not None else None,
               fixed_baseline_correct_minus_stale_before=lb[answer]-lb[stale] if stale is not None else None,
               fixed_baseline_correct_minus_stale_after=le[answer]-le[stale] if stale is not None else None,
               fixed_baseline_correct_minus_stale_change=(le[answer]-le[stale])-(lb[answer]-lb[stale]) if stale is not None else None,
               target_rank_before=base['full_vocab_rank'], target_rank_after=edit['full_vocab_rank'],
               candidate_rank_before=base['candidate_rank'], candidate_rank_after=edit['candidate_rank'],
               full_vocab_next_token_correct_before=base['full_vocab_next_token_accuracy'],
               full_vocab_next_token_correct_after=edit['full_vocab_next_token_accuracy'],
               pair_both_full_vocab_next_token_correct=int(bool(base['full_vocab_next_token_accuracy']) and bool(edit['full_vocab_next_token_accuracy'])))
    return row


def audited_effects(dataset, scored, kind):
    audit_behavior_dataset(dataset, kind)
    expected = {r['example_id']: r for r in dataset}
    seen, groups = set(), defaultdict(dict)
    for score in scored:
        eid = score.get('example_id')
        if eid not in expected or eid in seen:
            raise ValueError('unexpected or duplicate scored example_id')
        seen.add(eid)
        if any(score.get(k) != v for k, v in expected[eid].items()):
            raise ValueError(f'scored semantic metadata differs from dataset at {eid}')
        for key in ('candidate_logits', 'candidate_probabilities', 'full_vocab_rank', 'candidate_rank', 'accuracy', 'full_vocab_next_token_accuracy', 'prompt'):
            if key not in score:
                raise ValueError(f'scored record missing {key}')
        if not isinstance(score['prompt'], str) or score['full_vocab_rank'] < 1 or score['candidate_rank'] < 1:
            raise ValueError('invalid prompt/rank diagnostics')
        if score['accuracy'] not in (0, 1) or score['full_vocab_next_token_accuracy'] not in (0, 1):
            raise ValueError('invalid correctness diagnostics')
        groups[score['pair_id']][score['pair_direction']] = score
    if seen != set(expected):
        raise ValueError('incomplete scored dataset: missing pair directions or query cells')
    return [matched_edit_effect(m[0], m[1]) for _, m in sorted(groups.items())]


def _metric(effect, kind):
    if kind in ('controls', 'controls_counterbalanced'):
        condition = effect['condition']
        if condition == 'irrelevant_counterbalanced': condition = 'irrelevant'
        status = {'live': 'live_current', 'superseded': 'superseded_initial', 'irrelevant': 'irrelevant_occurrence'}[condition]
        if effect['edit_status'] != status:
            raise ValueError('control edit semantic status mismatch')
        return f'R_{effect["condition"]}'
    field = effect['edited_field']
    initial = field in ('initial_x', 'initial_z')
    if effect['condition'] == 'accepted':
        expected = 'superseded_initial' if initial else 'accepted_current'
        metric = 'R_superseded_initial' if initial else 'R_accepted_current'
    else:
        expected = 'accepted_current' if initial else 'rejected_update'
        metric = 'R_retained_initial_after_rejection' if initial else 'R_rejected_update'
    if effect['edit_status'] != expected:
        raise ValueError('accepted/rejected edit semantic status mismatch')
    return metric


def history_contrasts(effects, kind):
    """All cells required; primary contrasts never depend on model correctness."""
    if kind not in METRICS:
        raise ValueError('only controls/status analysis is supported')
    grouped = defaultdict(dict)
    for effect in effects:
        if effect['experiment_kind'] != kind or effect['condition'] not in CONDITIONS[kind]:
            raise ValueError('mixed experiment kinds or invalid condition')
        field = effect['edited_field']
        if field not in (FIELDS[:2] if kind == 'controls' else FIELDS) or effect['edited_variable'] != field[-1] or effect['query'] not in ('x', 'z'):
            raise ValueError('invalid relevance contrast cell')
        metric = _metric(effect, kind)
        if kind == 'controls_counterbalanced' and effect['condition'] == 'irrelevant_counterbalanced': metric = 'R_irrelevant_counterbalanced'
        cell = (metric, effect['edited_variable'], effect['query'], effect.get('unassigned_slot_order'))
        hid = effect['history_id']
        if cell in grouped[hid]:
            raise ValueError('duplicate history-level relevance cell')
        grouped[hid][cell] = float(effect['identity_transfer'])
    if not grouped:
        raise ValueError('no paired effects')
    output = []
    for hid, cells in sorted(grouped.items()):
        required = {(m, v, q, slot) for m in METRICS[kind] for v in ('x', 'z') for q in ('x', 'z')
                    for slot in (('xz', 'zx') if m == 'R_irrelevant_counterbalanced' else (None,))}
        if set(cells) != required:
            raise ValueError(f'history {hid}: missing or unexpected query/edit cells')
        row = {'history_id': hid}
        for metric in METRICS[kind]:
            def val(variable, query):
                if metric == 'R_irrelevant_counterbalanced':
                    return float(np.mean([cells[(metric, variable, query, slot)] for slot in ('xz', 'zx')]))
                return cells[(metric, variable, query, None)]
            row[metric+'_x'] = val('x', 'x') - val('x', 'z')
            row[metric+'_z'] = val('z', 'z') - val('z', 'x')
            row[metric] = .5*(row[metric+'_x'] + row[metric+'_z'])
        if kind in ('controls', 'controls_counterbalanced'):
            differences = (('R_live', 'R_superseded'), ('R_superseded', 'R_irrelevant'), ('R_live', 'R_irrelevant'))
        else:
            differences = (('R_accepted_current', 'R_rejected_update'), ('R_superseded_initial', 'R_retained_initial_after_rejection'))
        for a, b in differences:
            row[f'{a}_minus_{b}'] = row[a]-row[b]
        if kind == 'controls_counterbalanced':
            for a, b in (('R_live', 'R_irrelevant_counterbalanced'), ('R_superseded', 'R_irrelevant_counterbalanced')):
                row[f'{a}_minus_{b}'] = row[a]-row[b]
        output.append(row)
    return output


def status_2x2_contrasts(effects):
    """Per-history local acceptance effects; status pattern order is x then z."""
    grouped = defaultdict(dict)
    expected_status = {'YY': {'x':'accepted_current','z':'accepted_current'}, 'YN': {'x':'accepted_current','z':'rejected_update'},
                       'NY': {'x':'rejected_update','z':'accepted_current'}, 'NN': {'x':'rejected_update','z':'rejected_update'}}
    for e in effects:
        condition, variable, query = e['condition'], e['edited_variable'], e['query']
        if e['experiment_kind'] != 'status_2x2' or condition not in expected_status:
            raise ValueError('expected independent status_2x2 effects')
        if e['edited_field'] != f'proposed_{variable}':
            continue
        expected = expected_status[condition][variable]
        if e['edit_status'] != expected:
            raise ValueError('status_2x2 semantic status mismatch')
        key = (condition, variable, query)
        if key in grouped[e['history_id']]: raise ValueError('duplicate status_2x2 query/edit cell')
        grouped[e['history_id']][key] = float(e['identity_transfer'])
    rows=[]
    for hid, cells in sorted(grouped.items()):
        required={(c,v,q) for c in ('YY','YN','NY','NN') for v in ('x','z') for q in ('x','z')}
        if set(cells)!=required: raise ValueError(f'history {hid}: missing status_2x2 query/edit cell')
        r={ 'history_id': hid }
        for variable, patterns in (('x',(('YY','NY'),('YN','NN'))),('z',(('YY','YN'),('NY','NN')))):
            for i,(yes,no) in enumerate(patterns,1):
                relevant = cells[(yes,variable,variable)]-cells[(yes,variable,'z' if variable=='x' else 'x')]
                rejected = cells[(no,variable,variable)]-cells[(no,variable,'z' if variable=='x' else 'x')]
                r[f'R_{variable}_local_{i}']=relevant-rejected
            r[f'R_{variable}_acceptance_effect']=.5*(r[f'R_{variable}_local_1']+r[f'R_{variable}_local_2'])
        r['R_acceptance_effect_symmetric']=.5*(r['R_x_acceptance_effect']+r['R_z_acceptance_effect'])
        rows.append(r)
    if not rows: raise ValueError('no status_2x2 effects')
    return rows


def summarize_histories(values, seed=73021, n_boot=2000):
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('history summary requires finite values from complete histories')
    ci = bootstrap_mean_ci(x, n_boot=n_boot, seed=seed)
    return {'n_histories': len(x), 'mean': float(x.mean()), 'median': float(np.median(x)),
            'trimmed_mean_10pct': trimmed_mean(x, .1), 'fraction_positive': float(np.mean(x > 0)),
            'ci95_cluster_bootstrap': [ci['ci_low'], ci['ci_high']]}


def competence(scored, effects, threshold=.99):
    """Report every condition, with all baseline/edit outcomes retained."""
    groups = defaultdict(list)
    for r in scored:
        groups[(r['condition'], r['edit_status'])].append(r)
    result = []
    for (condition, status), rows in sorted(groups.items()):
        pairs = [r for r in effects if r['condition'] == condition and r['edit_status'] == status]
        accuracy = float(np.mean([r['full_vocab_next_token_accuracy'] for r in rows]))
        ranks = [r['full_vocab_rank'] for r in rows]
        result.append({'condition': condition, 'edit_status': status, 'n_scored_members': len(rows),
                       'n_pairs': len(pairs), 'full_vocab_next_token_accuracy': accuracy,
                       'candidate_restricted_accuracy': float(np.mean([r['accuracy'] for r in rows])),
                       'target_rank_mean': float(np.mean(ranks)), 'target_rank_median': float(np.median(ranks)),
                       'target_rank_max': int(max(ranks)),
                       'both_answers_correct_count': sum(r['pair_both_full_vocab_next_token_correct'] for r in pairs),
                       'both_answers_correct_fraction': float(np.mean([r['pair_both_full_vocab_next_token_correct'] for r in pairs])),
                       'diagnostic_accuracy_threshold': threshold, 'low_competence': accuracy < threshold})
    return result

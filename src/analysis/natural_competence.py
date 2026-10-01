"""Shared unique-prompt competence gate and complete semantic cell audit."""
import math
from collections import Counter
from itertools import product

GATE_VERSION = 'natural_competence_v2'
GATE = {'full_vocab_accuracy_min': .99, 'candidate_accuracy_min': .99,
        'mean_candidate_rank_max': 1.01, 'stale_margin_min': 0.0}
STATUSES = {'live': 'live_current', 'superseded': 'superseded_initial',
            'irrelevant_counterbalanced': 'irrelevant_occurrence'}


def expected_cells(templates):
    return {(template, condition, query, orientation, variable, status, direction, slot)
            for template in templates for condition, status in STATUSES.items()
            for query, orientation, variable, direction in product(('x', 'z'), (0, 1), ('x', 'z'), (0, 1))
            for slot in (('xz', 'zx') if condition == 'irrelevant_counterbalanced' else ('none',))}


def evaluate_competence(dataset, scores):
    """Count a prompt once within each semantic cell represented by its rows."""
    if not dataset:
        raise ValueError('empty competence dataset')
    expected = {r['example_id']: r for r in dataset}
    actual = {r['example_id']: r for r in scores}
    if len(expected) != len(dataset) or len(actual) != len(scores) or set(actual) != set(expected):
        raise ValueError('scored example_id set does not exactly match dataset')
    templates = sorted({r['prompt_variant'] for r in dataset})
    cells, unique, condition_unique = {}, {}, {}
    for row in dataset:
        score = actual[row['example_id']]
        if any(score.get(k) != v for k, v in row.items()):
            raise ValueError(f'scored metadata mismatch at {row["example_id"]}')
        signature = (row['prompt_variant'], score['prompt'])
        logits = score['candidate_logits']
        generated = score.get('generated_first_token')
        parsed = (str(generated).strip() in row['candidate_values']) if generated is not None else None
        target_rank = score.get('full_vocab_rank')
        wrong_token = str(generated).strip() if generated is not None and not score['full_vocab_next_token_accuracy'] else None
        diagnostics = (score['full_vocab_next_token_accuracy'], score['accuracy'], score['candidate_rank'],
                       logits[row['answer']], logits[row['stale_value']] if row.get('stale_value') else None,
                       target_rank, parsed, wrong_token)
        if any(not math.isfinite(float(x)) for x in diagnostics if isinstance(x, (int, float))):
            raise ValueError('nonfinite competence diagnostics')
        if (diagnostics[0] not in (0, 1) or diagnostics[1] not in (0, 1) or diagnostics[2] < 1 or
                (target_rank is not None and target_rank < 1)):
            raise ValueError('invalid competence accuracy/rank')
        if signature in unique and unique[signature] != diagnostics:
            raise ValueError('duplicate prompt has inconsistent scoring diagnostics')
        unique[signature] = diagnostics
        condition_unique.setdefault((row['prompt_variant'], row['condition']), {})[signature] = diagnostics
        if row['condition'] == 'irrelevant':
            continue
        cell = (row['prompt_variant'], row['condition'], row['query'], row['orientation'],
                row['edited_variable'], row['edit_status'], row['pair_direction'],
                row.get('unassigned_slot_order') or 'none')
        cells.setdefault(cell, {})[signature] = diagnostics
    required = expected_cells(templates)
    if set(cells) != required:
        raise ValueError(f'incomplete competence cell set: missing={sorted(required-set(cells))}, unexpected={sorted(set(cells)-required)}')
    summary = {}
    for key, prompts in sorted(cells.items()):
        values = list(prompts.values())
        margins = [d[3]-d[4] for d in values if d[4] is not None]
        target_ranks = [d[5] for d in values if d[5] is not None]
        parse_values = [d[6] for d in values if d[6] is not None]
        wrong_tokens = Counter(d[7] for d in values if d[7])
        summary['|'.join(map(str, key))] = {
            'n_unique_prompts': len(values),
            'full_vocab_accuracy': sum(d[0] for d in values)/len(values),
            'candidate_accuracy': sum(d[1] for d in values)/len(values),
            'mean_candidate_rank': sum(d[2] for d in values)/len(values),
            'mean_full_vocab_target_rank': sum(target_ranks)/len(target_ranks) if target_ranks else None,
            'median_full_vocab_target_rank': sorted(target_ranks)[len(target_ranks)//2] if target_ranks else None,
            'max_full_vocab_target_rank': max(target_ranks) if target_ranks else None,
            'parse_rate': sum(parse_values)/len(parse_values) if parse_values else None,
            'common_wrong_tokens': [{'token': token, 'count': count} for token, count in wrong_tokens.most_common(5)],
            'mean_current_minus_stale_logit_margin': sum(margins)/len(margins) if margins else None}
    passed = all(v['full_vocab_accuracy'] >= GATE['full_vocab_accuracy_min'] and
                 v['candidate_accuracy'] >= GATE['candidate_accuracy_min'] and
                 v['mean_candidate_rank'] <= GATE['mean_candidate_rank_max'] and
                 (v['mean_current_minus_stale_logit_margin'] is None or
                  v['mean_current_minus_stale_logit_margin'] > GATE['stale_margin_min'])
                 for v in summary.values())
    failed_cells = []
    for key, metrics in summary.items():
        failures = []
        if metrics['full_vocab_accuracy'] < GATE['full_vocab_accuracy_min']:
            failures.append('full_vocab_accuracy')
        if metrics['candidate_accuracy'] < GATE['candidate_accuracy_min']:
            failures.append('candidate_accuracy')
        if metrics['mean_candidate_rank'] > GATE['mean_candidate_rank_max']:
            failures.append('mean_candidate_rank')
        margin = metrics['mean_current_minus_stale_logit_margin']
        if margin is not None and margin <= GATE['stale_margin_min']:
            failures.append('current_minus_stale_margin')
        if failures:
            failed_cells.append({'cell': key, 'failed_criteria': failures})
    condition_summary = {}
    for (template, condition), prompts in sorted(condition_unique.items()):
        values = list(prompts.values())
        ranks = [d[5] for d in values if d[5] is not None]
        parses = [d[6] for d in values if d[6] is not None]
        wrong_tokens = Counter(d[7] for d in values if d[7])
        condition_summary[f'{template}|{condition}'] = {
            'n_unique_prompts': len(values),
            'full_vocab_accuracy': sum(d[0] for d in values)/len(values),
            'candidate_accuracy_diagnostic': sum(d[1] for d in values)/len(values),
            'parse_rate': sum(parses)/len(parses) if parses else None,
            'mean_full_vocab_target_rank': sum(ranks)/len(ranks) if ranks else None,
            'median_full_vocab_target_rank': sorted(ranks)[len(ranks)//2] if ranks else None,
            'max_full_vocab_target_rank': max(ranks) if ranks else None,
            'common_wrong_tokens': [{'token': token, 'count': count} for token, count in wrong_tokens.most_common(10)]}
    return {'pass': passed, 'summary': summary, 'condition_summary': condition_summary,
            'failed_cells': failed_cells, 'templates': templates,
            'n_unique_prompts': len(unique), 'expected_cell_count': len(required),
            'dataset_seed': sorted({r['seed'] for r in dataset})}


def concrete_history_signature(row):
    """Identify the full generated history while ignoring template replication."""
    import json
    return json.dumps({'initial_values': row['matching_values'],
                       'replacement_values': row['replacement_values'],
                       'entities': row.get('entities', row.get('variables')),
                       'attribute': row.get('attribute'), 'orientation': row.get('orientation')}, sort_keys=True)

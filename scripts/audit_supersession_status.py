#!/usr/bin/env python3
"""CPU-only error audit and all-NO-correct sensitivity for saved status scores."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from src.data.io import read_jsonl, sha256_file
from src.analysis.supersession_behavior import audited_effects, history_contrasts, summarize_histories
from src.utils import provenance, save_json


def categorize_no_errors(dataset, scored, token_ids):
    by_id = {r['example_id']: r for r in dataset}
    if len(by_id) != len(dataset) or set(by_id) != {r['example_id'] for r in scored}:
        raise ValueError('dataset/scored IDs are not a one-to-one match')
    id_to_value = {v: k for k, v in token_ids.items()}
    rows = []
    for score in scored:
        if score['condition'] != 'rejected' or score['full_vocab_next_token_accuracy']:
            continue
        ex = by_id[score['example_id']]
        q = ex['query']
        proposed = ex['semantic_values'][f'proposed_{q}']
        retained = ex['semantic_values'][f'initial_{q}']
        predicted = score.get('greedy_token_id')
        token_value = id_to_value.get(predicted)
        is_candidate = token_value in ex['candidate_values']
        category = ('rejected_proposed_value' if predicted == token_ids[proposed] else
                    'retained_initial_value' if predicted == token_ids[retained] else
                    'other_candidate' if is_candidate else 'non_candidate')
        rows.append({'history_id': ex['history_id'], 'example_id': ex['example_id'], 'pair_id': ex['pair_id'],
                     'query': q, 'edited_field': ex['edited_field'], 'pair_direction': ex['pair_direction'],
                     'correct_answer': ex['answer'], 'proposed_value': proposed, 'retained_initial_value': retained,
                     'predicted_candidate': token_value if is_candidate else None, 'predicted_token_id': predicted, 'error_category': category,
                     'prompt': score['prompt']})
    return rows


def all_no_members_correct_histories(scored):
    by_history = defaultdict(list)
    for r in scored:
        if r['condition'] == 'rejected':
            by_history[r['history_id']].append(r)
    if not by_history or any(len(rows) != 16 for rows in by_history.values()):
        raise ValueError('each status history must have all 16 rejected/NO baseline-edit members')
    return {h for h, rows in by_history.items() if all(r['full_vocab_next_token_accuracy'] == 1 for r in rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True); p.add_argument('--behavior', required=True)
    p.add_argument('--token-ids', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--seed', type=int, default=73021); p.add_argument('--bootstrap-draws', type=int, default=2000)
    a = p.parse_args()
    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'{out} is nonempty; choose a fresh audit path')
    ds, scores = read_jsonl(a.dataset), read_jsonl(a.behavior)
    token_map = json.loads(Path(a.token_ids).read_text())['token_ids']
    if any(r.get('experiment_kind') != 'status' for r in ds): raise ValueError('status audit requires the original matched status dataset')
    errors = categorize_no_errors(ds, scores, token_map)
    effects = audited_effects(ds, scores, 'status')
    histories = history_contrasts(effects, 'status')
    valid = all_no_members_correct_histories(scores)
    sensitivity = [h for h in histories if h['history_id'] in valid]
    if len(sensitivity) != len(valid): raise ValueError('status contrast histories and complete-NO histories differ')
    metric_pairs = (('R_accepted_current', 'R_rejected_update'),
                    ('R_superseded_initial', 'R_retained_initial_after_rejection'))
    selected = {}
    for a_metric, b_metric in metric_pairs:
        for suffix in ('_x', '_z'):
            key = f'{a_metric}_minus_{b_metric}{suffix}'
            vals = [h[a_metric+suffix]-h[b_metric+suffix] for h in sensitivity]
            selected[key] = summarize_histories(vals, a.seed, a.bootstrap_draws)
        key = f'{a_metric}_minus_{b_metric}_symmetric'
        selected[key] = summarize_histories([h[a_metric]-h[b_metric] for h in sensitivity], a.seed, a.bootstrap_draws)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(errors).to_csv(out/'no_condition_prediction_errors.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(out/'complete_no_history_sensitivity.csv', index=False)
    summary = {'analysis': 'secondary_sensitivity_complete_NO_histories_only',
               'primary_all_trials_analysis_replaced': False, 'n_histories_total': len(histories),
               'n_histories_all_NO_members_correct': len(valid), 'n_incorrect_NO_predictions': len(errors),
               'error_category_counts': {key: sum(r['error_category'] == key for r in errors) for key in
                                         ('rejected_proposed_value', 'retained_initial_value', 'other_candidate', 'non_candidate')},
               'required_NO_members_per_history': 16, 'contrasts': selected,
               'bootstrap_unit': 'history_id', 'bootstrap_seed': a.seed, 'bootstrap_draws': a.bootstrap_draws,
               'source_sha256': {'dataset': sha256_file(a.dataset), 'behavior': sha256_file(a.behavior), 'token_map': sha256_file(a.token_ids)},
               'sensitivity_histories': sorted(valid), 'provenance': provenance({'seed': a.seed}, a.dataset)}
    save_json(summary, out/'summary.json')
    print(f'audited {len(errors)} incorrect NO members; complete-NO sensitivity n={len(valid)}; wrote {out}')


if __name__ == '__main__': main()

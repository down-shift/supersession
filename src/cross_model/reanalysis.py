"""Read-only diagnostics and decompositions of complete saved v1 confirmations."""
import math
from collections import defaultdict
from pathlib import Path

from src.analysis.supersession_behavior import audit_behavior_dataset, history_contrasts
from src.cross_model.protocol import VALUES, digest, read_sealed
from src.cross_model.robustness_analysis import (normalized, paired_effects, query_relevance,
                                                summary, summarize_rows)
from src.cross_model.score_checks import checked_scores
from src.data.io import read_jsonl, sha256_file
from src.utils import provenance


def _competitor(masses, answer):
    return max((v for v in VALUES if v != answer), key=masses.get)


def pair_diagnostics(base, edit, ranks):
    b, e = base['semantic_log_mass'], edit['semantic_log_mass']
    source, replacement = base['source_value'], base['replacement_value']
    target_b, target_e = base['answer'], edit['answer']
    competitor_b, competitor_e = _competitor(b, target_b), _competitor(e, target_e)
    result = {k: base[k] for k in ('pair_id', 'history_id', 'condition', 'orientation',
                                  'edited_variable', 'query', 'source_value', 'replacement_value')}
    result.update(unassigned_slot_order=base.get('unassigned_slot_order'),
                  baseline_answer=target_b, edited_answer=target_e,
                  baseline_competitor=competitor_b, edited_competitor=competitor_e,
                  answer_identity_changed=target_b != target_e,
                  rank_before=ranks[base['example_id']], rank_after=ranks[edit['example_id']],
                  semantic_correct_before=int(ranks[base['example_id']] == 1),
                  semantic_correct_after=int(ranks[edit['example_id']] == 1),
                  answer_log_mass_before=b[target_b], answer_log_mass_after=e[target_e],
                  member_target_score_change=e[target_e]-b[target_b],
                  fixed_baseline_answer_score_change=e[target_b]-b[target_b],
                  decision_margin_before=b[target_b]-b[competitor_b],
                  decision_margin_after=e[target_e]-e[competitor_e],
                  decision_margin_change=(e[target_e]-e[competitor_e])-(b[target_b]-b[competitor_b]),
                  member_target_vs_fixed_baseline_competitor_change=(e[target_e]-e[competitor_b])-(b[target_b]-b[competitor_b]),
                  fixed_baseline_target_vs_fixed_competitor_change=(e[target_b]-e[competitor_b])-(b[target_b]-b[competitor_b]),
                  source_log_mass_before=b[source], source_log_mass_after=e[source],
                  replacement_log_mass_before=b[replacement], replacement_log_mass_after=e[replacement],
                  source_score_change=e[source]-b[source], replacement_score_change=e[replacement]-b[replacement],
                  identity_transfer=(e[replacement]-e[source])-(b[replacement]-b[source]),
                  prefix_mass={'source': {'value': source, 'baseline': math.exp(b[source]), 'edited': math.exp(e[source])},
                               'replacement': {'value': replacement, 'baseline': math.exp(b[replacement]), 'edited': math.exp(e[replacement])},
                               'member_target': {'baseline_value': target_b, 'edited_value': target_e,
                                                 'baseline': math.exp(b[target_b]), 'edited': math.exp(e[target_e])}},
                  semantic_log_mass_before=b, semantic_log_mass_after=e)
    stale_b, stale_e = base.get('stale_value'), edit.get('stale_value')
    if (stale_b is None) != (stale_e is None):
        raise ValueError('paired edit changed stale-binding applicability')
    result['current_over_stale_margin_before'] = b[target_b]-b[stale_b] if stale_b else None
    result['current_over_stale_margin_after'] = e[target_e]-e[stale_e] if stale_e else None
    if 'surface_likelihoods' in base:
        result['surface_likelihoods'] = {
            'source': {'value': source, 'baseline': base['surface_likelihoods'][source], 'edited': edit['surface_likelihoods'][source]},
            'replacement': {'value': replacement, 'baseline': base['surface_likelihoods'][replacement], 'edited': edit['surface_likelihoods'][replacement]},
            'member_target': {'baseline': base['surface_likelihoods'][target_b], 'edited': edit['surface_likelihoods'][target_e]}}
    return result


def _history_mean(records, getter):
    values = defaultdict(list)
    for row in records:
        value = getter(row)
        if value is not None:
            values[row['history_id']].append(value)
    per_history = [{'history_id': hid, 'mean': sum(v)/len(v)} for hid, v in sorted(values.items())]
    return {'summary': summary([r['mean'] for r in per_history]), 'history_rows': per_history}


def _condition_diagnostics(condition, pairs, actual):
    members = [s for s in actual.values() if s['condition'] == condition]
    metrics = ('member_target_score_change', 'fixed_baseline_answer_score_change', 'decision_margin_change',
               'member_target_vs_fixed_baseline_competitor_change', 'fixed_baseline_target_vs_fixed_competitor_change',
               'decision_margin_before', 'decision_margin_after', 'answer_log_mass_before', 'answer_log_mass_after',
               'source_log_mass_before', 'source_log_mass_after', 'replacement_log_mass_before', 'replacement_log_mass_after',
               'source_score_change', 'replacement_score_change',
               'current_over_stale_margin_before', 'current_over_stale_margin_after')
    report = {metric: _history_mean(pairs, lambda r, m=metric: r[m]) for metric in metrics}
    report['n_pairs'] = len(pairs)
    report['answer_identity_changed_pairs'] = sum(p['answer_identity_changed'] for p in pairs)
    report['member_target_score_change_definition'] = (
        'edited correct-answer S minus baseline correct-answer S; answer identities can differ in live associated-query pairs'
        if condition == 'live' else 'edited minus baseline S of the same fixed current-answer identity')
    report['decision_margin_definition'] = 'each member correct-answer S minus its own strongest incorrect semantic candidate S'
    for direction, label in ((0, 'baseline'), (1, 'edited')):
        group = [s for s in members if s['pair_direction'] == direction]
        unique = {}
        for s in group:
            key = s['prompt']
            data = (s['answer'], s['semantic_log_mass'], s['_recomputed_rank'])
            if key in unique and unique[key] != data:
                raise ValueError('identical saved prompt has inconsistent semantic masses/answer')
            unique[key] = data
        report[label] = {'n_members': len(group), 'n_unique_prompts': len(unique),
                         'semantic_accuracy': sum(s['_recomputed_rank'] == 1 for s in group)/len(group),
                         'unique_prompt_semantic_accuracy': sum(d[2] == 1 for d in unique.values())/len(unique),
                         'mean_candidate_rank': sum(s['_recomputed_rank'] for s in group)/len(group)}
    report['candidate_ranking_changes'] = {
        'rank_changed_pairs': sum(p['rank_before'] != p['rank_after'] for p in pairs),
        'correct_to_incorrect_pairs': sum(p['semantic_correct_before'] and not p['semantic_correct_after'] for p in pairs),
        'incorrect_to_correct_pairs': sum(not p['semantic_correct_before'] and p['semantic_correct_after'] for p in pairs),
        'both_correct_pairs': sum(p['semantic_correct_before'] and p['semantic_correct_after'] for p in pairs)}
    report['absolute_bounded_prefix_mass'] = {
        role: {label: _history_mean(pairs, lambda r, ro=role, la=label: r['prefix_mass'][ro][la])
               for label in ('baseline', 'edited')}
        for role in ('member_target', 'source', 'replacement')}
    return report


def _relevance(effects, field='identity_transfer'):
    return history_contrasts([{**e, 'identity_transfer': e[field]} for e in effects], 'controls_counterbalanced')


def order_relevance(effects):
    grouped = defaultdict(dict)
    for e in effects:
        if e['condition'] != 'irrelevant_counterbalanced':
            continue
        key = (e['unassigned_slot_order'], e['edited_variable'], e['query'])
        if key in grouped[e['history_id']]:
            raise ValueError('duplicate irrelevant order/edit/query cell')
        grouped[e['history_id']][key] = e
    required = {(o, v, q) for o in ('xz', 'zx') for v in ('x', 'z') for q in ('x', 'z')}
    output = []
    for hid, cells in sorted(grouped.items()):
        if set(cells) != required:
            raise ValueError('incomplete irrelevant matched set; both mention orders and queries are required')
        row = {'history_id': hid}
        for order in ('xz', 'zx'):
            for field in ('identity_transfer', 'replacement_change', 'negative_source_change'):
                values = {(v, q): cells[(order, v, q)][field] for v in ('x', 'z') for q in ('x', 'z')}
                row[f'R_{order}_{field}'] = query_relevance(values)
                for v in ('x', 'z'):
                    other = 'z' if v == 'x' else 'x'
                    row[f'R_{order}_{field}_{v}'] = values[(v, v)] - values[(v, other)]
        output.append(row)
    return output


def reanalyze_rows(rows, scores, *, require_surfaces=True):
    audit_behavior_dataset(rows, 'controls_counterbalanced')
    actual, paired, ranks = checked_scores(rows, scores, require_surfaces=require_surfaces)
    if any(s.get('score_kind') != 'confirmatory' for s in scores):
        raise ValueError('saved-score reanalysis requires confirmatory scores')
    actual = {eid: {**s, '_recomputed_rank': ranks[eid]} for eid, s in actual.items()}
    details = [pair_diagnostics(pair[0], pair[1], ranks) for _, pair in sorted(paired.items())]
    effects = paired_effects(rows, scores)
    for e in effects:
        e['negative_source_change'] = -e['source_change']
    histories = _relevance(effects)
    source_rows = _relevance(effects, 'negative_source_change')
    replacement_rows = _relevance(effects, 'replacement_change')
    for total, source, replacement in zip(histories, source_rows, replacement_rows):
        if any(not math.isclose(total[k], source[k]+replacement[k], abs_tol=1e-10, rel_tol=1e-10)
               for k in total if k != 'history_id'):
            raise ValueError('source/replacement relevance decomposition is inconsistent')
    primary_key = 'R_superseded_minus_R_irrelevant_counterbalanced'
    denominator_key = 'R_live_minus_R_irrelevant_counterbalanced'
    by_history = {r['history_id']: r for r in rows}
    eligible = {r['history_id'] for r in histories}
    for row in rows:
        if row['condition'] == 'superseded' and ranks[row['example_id']] != 1:
            eligible.discard(row['history_id'])
    conditioned_effects = [e for e in effects if e['history_id'] in eligible]
    conditioned_histories = _relevance(conditioned_effects) if eligible else []
    # Selection is only on supersession competence; every matched control/query/member
    # of each selected history is retained, including both irrelevant mention orders.
    secondary = {'label': 'secondary; conditioned on all superseded members being semantically correct within a complete history',
                 'eligibility_definition': 'strict rank one for every superseded query/edit/baseline/edited member; no filtering of matched control members',
                 'included_history_ids': sorted(eligible), 'n_histories': len(eligible),
                 'n_matched_members': sum(row['history_id'] in eligible for row in rows),
                 'history_rows': conditioned_histories, 'statistics': summarize_rows(conditioned_histories),
                 'primary_contrast': summary([r[primary_key] for r in conditioned_histories])}
    return {
        'protocol': 'cross_model_v1_saved_score_reanalysis', 'revision': 'reporting_correction_20261003',
        'all_trial_primary': True, 'n_trials': len(rows), 'n_pairs': len(details), 'n_histories': len(histories),
        'rank_audit': {'recomputed_from_masses': len(scores),
                       'stored_ranks_compared': sum(s.get('semantic_rank') is not None for s in scores),
                       'stored_rank_mismatches': 0, 'ties_policy': 'strict rank one; ties count as incorrect'},
        'condition_diagnostics': {c: _condition_diagnostics(c, [p for p in details if p['condition'] == c], actual)
                                  for c in ('live', 'superseded', 'irrelevant', 'irrelevant_counterbalanced')},
        'history_rows': histories, 'statistics': summarize_rows(histories),
        'complete_relevance_decomposition': {
            'definition': 'E = replacement score change - source score change; R and every higher contrast decompose linearly into replacement and negative-source components',
            'replacement_component': {'history_rows': replacement_rows, 'statistics': summarize_rows(replacement_rows)},
            'negative_source_component': {'history_rows': source_rows, 'statistics': summarize_rows(source_rows)},
            'surface_note': 'per-surface likelihoods are retained in pair details; logsumexp across surfaces is not an additive mediation decomposition'},
        'counterbalanced_unassigned_R_by_mention_order': {
            'definition': 'R separately for xz/zx: .5*((E_xx-E_xz)+(E_zz-E_zx)); never average raw E across queries',
            'history_rows': order_relevance(effects), 'statistics': summarize_rows(order_relevance(effects))},
        'ordinary_irrelevant_R': summary([r['R_irrelevant'] for r in histories]),
        'normalized_primary_relative_to_live_minus_irrelevant': normalized(
            [r[primary_key] for r in histories], [r[denominator_key] for r in histories]),
        'strata': {'orientation': {str(o): summary([r[primary_key] for r in histories if by_history[r['history_id']]['orientation'] == o]) for o in (0, 1)},
                   'semantic_variable': {v: summary([r[f'R_superseded_{v}'] - r[f'R_irrelevant_counterbalanced_{v}'] for r in histories]) for v in ('x', 'z')}},
        'secondary_conditioned_supersession_fully_correct': secondary,
        'paired_effect_details': details,
        'aggregation': 'each pair E is edited minus baseline log(source/replacement odds); average counterbalanced orders within history before symmetric entity/query R; all bootstrap draws resample histories',
        'probability_note': 'absolute masses are bounded prefix-event probabilities; no EOS, termination or candidate renormalization',
        'new_relational_contrasts': {'available': False, 'reason': 'the six-condition relational design was not collected in cross_model_v1'},
    }


def reanalyze(dataset_path, scores_path):
    """Require original saved hashes and sealed provenance, without loading weights."""
    dataset_path, scores_path = Path(dataset_path), Path(scores_path)
    data_info = read_sealed(str(dataset_path)+'.provenance.json')
    score_info = read_sealed(str(scores_path)+'.provenance.json')
    if data_info['stage'] != 'confirmatory' or score_info['stage'] != 'confirmatory':
        raise ValueError('saved artifacts are not confirmatory')
    if score_info['scores_sha256'] != sha256_file(scores_path):
        raise ValueError('saved score artifact hash mismatch')
    for info in (data_info, score_info):
        if info['provenance']['dataset_sha256'] != sha256_file(dataset_path):
            raise ValueError('saved dataset artifact hash mismatch')
    scientific = ('model_id', 'model_revision', 'tokenizer_id', 'tokenizer_revision', 'protocol', 'contract_sha256', 'candidate_map_sha256')
    if any(data_info['provenance'].get(k) != score_info['provenance'].get(k) for k in scientific):
        raise ValueError('original dataset/score scientific provenance mismatch')
    result = reanalyze_rows(read_jsonl(dataset_path), read_jsonl(scores_path))
    result.update(dataset_sha256=sha256_file(dataset_path), scores_sha256=sha256_file(scores_path),
                  dataset_path=str(dataset_path.resolve()), scores_path=str(scores_path.resolve()),
                  source_provenance={'dataset': data_info, 'scores': score_info,
                                     'dataset_sidecar_sha256': sha256_file(str(dataset_path)+'.provenance.json'),
                                     'scores_sidecar_sha256': sha256_file(str(scores_path)+'.provenance.json')},
                  reanalysis_provenance=provenance(score_info['provenance']['config'], dataset_path),
                  reanalysis_code_sha256=digest({str(p): sha256_file(p) for p in
                      map(Path, ('src/cross_model/reanalysis.py', 'src/cross_model/robustness_analysis.py',
                                 'src/cross_model/score_checks.py', 'src/analysis/supersession_behavior.py',
                                 'scripts/robustness_v2.py'))}))
    return result


def markdown_report(result):
    """Compact paper-review table; detailed saved terms stay in the JSON artifact."""
    def interval(stats):
        if stats['mean'] is None:
            return 'n/a'
        low, high = stats['ci95_cluster_bootstrap']
        return f'{stats["mean"]:.4f} [{low:.4f}, {high:.4f}]'

    model = result.get('source_provenance', {}).get('scores', {}).get('provenance', {}).get('model_id', 'saved model')
    lines = [f'# Corrected saved-score reanalysis: {model}', '',
             'Revision: reporting_correction_20261003. No new inference. All trials remain in the primary estimate.', '',
             f'{result["n_histories"]} histories, {result["n_pairs"]} pairs, {result["n_trials"]} members. '
             'Strict ranks are recomputed from bounded semantic masses; saved ranks agree.', '',
             'All intervals below are 95% paired history-bootstrap intervals (2,000 draws; seed 73021).', '',
             '| Contrast | Mean [95% CI], nats |', '|---|---:|']
    for key in ('R_superseded_minus_R_irrelevant_counterbalanced', 'R_live_minus_R_superseded'):
        lines.append(f'| {key} | {interval(result["statistics"]["results"][key])} |')
    lines += ['', '| Condition | Baseline accuracy | Edited accuracy | Current target prefix mass, baseline/edit | Source prefix mass, baseline/edit | Replacement prefix mass, baseline/edit |',
              '|---|---:|---:|---:|---:|---:|']
    for condition, report in result['condition_diagnostics'].items():
        probabilities = report['absolute_bounded_prefix_mass']
        def masses(role):
            return '/'.join(f'{probabilities[role][member]["summary"]["mean"]:.4g}' for member in ('baseline', 'edited'))
        lines.append(f'| {condition} | {report["baseline"]["semantic_accuracy"]:.4%} | {report["edited"]["semantic_accuracy"]:.4%} | {masses("member_target")} | {masses("source")} | {masses("replacement")} |')
    lines += ['', 'Masses are absolute bounded prefix probabilities, averaged within histories; no candidate renormalization or EOS scoring.',
              'The live answer identity can change after its assignment is edited. Other conditions retain the same correct current answer.', '',
              '| Condition | Member-target score change | Member-specific decision-margin change | Fixed-baseline-competitor diagnostic |',
              '|---|---:|---:|---:|']
    for condition, report in result['condition_diagnostics'].items():
        fields = ('member_target_score_change', 'decision_margin_change', 'member_target_vs_fixed_baseline_competitor_change')
        lines.append(f'| {condition} | ' + ' | '.join(interval(report[f]['summary']) for f in fields) + ' |')
    lines += ['', 'Decision margins choose the strongest incorrect candidate separately in each member.',
              'The fixed-baseline-competitor column is a separate diagnostic; in live pairs that competitor may become the correct edited answer.', '',
              '| Counterbalanced unassigned mention order | Query-specific R, mean [95% CI] |', '|---|---:|']
    order = result['counterbalanced_unassigned_R_by_mention_order']['statistics']['results']
    for label in ('xz', 'zx'):
        lines.append(f'| {label} | {interval(order[f"R_{label}_identity_transfer"])} |')
    secondary = result['secondary_conditioned_supersession_fully_correct']
    lines += ['', f'Secondary, conditioned table: {secondary["n_histories"]} complete histories whose superseded members are all correct. '
              'All their matched control members, both queries and both irrelevant orders are retained.', '',
              f'Secondary primary contrast: {interval(secondary["primary_contrast"])}.', '']
    ratio = result['normalized_primary_relative_to_live_minus_irrelevant']
    if ratio['available']:
        lo, hi = ratio['ci95_history_bootstrap']
        lines += [f'Frozen normalized ratio of means: {ratio["ratio_of_means"]:.4f} [{lo:.4f}, {hi:.4f}].']
    else:
        lines += [f'Frozen normalized ratio unavailable: {ratio["reason"]}.']
    lines += ['', 'The JSON retains per-history/variable/orientation distributions, all four source/replacement mass terms, additive relevance decompositions and per-surface likelihoods.',
              'The new relational conditions were not collected in v1 and are unavailable from these artifacts.',
              'Earlier preview/reanalysis reports from the faulty implementation are superseded and must not enter the paper.', '']
    return '\n'.join(lines)

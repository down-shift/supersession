"""History-level sequence-mass and separately labeled original raw-logit effects."""
from collections import defaultdict
import numpy as np

from src.analysis.supersession_behavior import history_contrasts, summarize_histories
from src.cross_model.protocol import VALUES


def contrasts(rows, scores, field='semantic_log_mass'):
    expected = {r['example_id']: r for r in rows}; seen = set(); pairs = defaultdict(dict)
    for s in scores:
        eid = s['example_id']
        if eid in seen or eid not in expected or any(s.get(k) != v for k, v in expected[eid].items()):
            raise ValueError('causal dataset/score mismatch')
        seen.add(eid)
        if set(s[field]) != set(VALUES): raise ValueError('complete candidate scores required')
        pairs[s['pair_id']][s['pair_direction']] = s
    if seen != set(expected): raise ValueError('incomplete causal scores')
    effects = []
    for members in pairs.values():
        b, e = members[0], members[1]; source, replacement = b['source_value'], b['replacement_value']
        effect = {k: b[k] for k in ('history_id', 'condition', 'experiment_kind', 'edited_field',
                  'edited_variable', 'query', 'edit_status', 'unassigned_slot_order')}
        effect['identity_transfer'] = ((e[field][replacement]-e[field][source]) -
                                      (b[field][replacement]-b[field][source]))
        effects.append(effect)
    return history_contrasts(effects, 'controls_counterbalanced')


def summarize(history_rows):
    names = [k for k in history_rows[0] if k != 'history_id']
    result = {k: summarize_histories([r[k] for r in history_rows]) for k in names}
    # Ratio of model means with paired bootstrap; never average per-history ratios.
    live = np.array([r['R_live_minus_R_irrelevant_counterbalanced'] for r in history_rows])
    primary = np.array([r['R_superseded_minus_R_irrelevant_counterbalanced'] for r in history_rows])
    normalized = {'available': False, 'reason': 'irrelevant-corrected live mean lower 95% history-bootstrap bound <= 1 nat'}
    if result['R_live_minus_R_irrelevant_counterbalanced']['ci95_cluster_bootstrap'][0] > 1:
        rng = np.random.default_rng(73021)
        indices = rng.integers(0, len(live), size=(2000, len(live)))
        denominators = live[indices].mean(1)
        if np.all(denominators > 1):
            ratios = primary[indices].mean(1)/denominators
            normalized = {'available': True, 'definition':'mean(superseded-irrelevant_cb) / mean(live-irrelevant_cb)', 'ratio_of_means': float(primary.mean()/live.mean()),
                          'ci95_history_bootstrap': list(map(float, np.quantile(ratios, [.025, .975])))}
        else:
            normalized = {'available': False, 'reason': 'a paired bootstrap irrelevant-corrected live denominator <= 1 nat'}
    return {'results': result, 'normalized_primary_relative_to_live_minus_irrelevant': normalized,
            'bootstrap_unit': 'history', 'bootstrap_draws': 2000, 'bootstrap_seed': 73021}


def strata(rows, history_rows):
    by_id = {r['history_id']: r for r in rows}
    primary = 'R_superseded_minus_R_irrelevant_counterbalanced'
    orientation = {str(o): summarize_histories([r[primary] for r in history_rows
                     if by_id[r['history_id']]['orientation'] == o]) for o in (0, 1)}
    variables = {v: summarize_histories([r['R_superseded_'+v]-r['R_irrelevant_counterbalanced_'+v]
                                      for r in history_rows]) for v in ('x', 'z')}
    # Membership strata overlap. They are descriptive, never independent samples.
    vocabulary = {}
    for value in VALUES:
        selected = [r[primary] for r in history_rows if value in by_id[r['history_id']]['matching_values'].values()]
        vocabulary[value] = summarize_histories(selected) if selected else {'n_histories': 0}
    return {'orientation': orientation, 'semantic_variable': variables,
            'vocabulary_history_membership': vocabulary, 'vocabulary_strata_overlap': True}

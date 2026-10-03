"""The v1 replacement/source relevance algebra, with independently crossed orders."""
from collections import defaultdict
import numpy as np

from src.analysis.metrics import trimmed_mean
from src.cross_model.robustness_v2 import CONDITIONS, required_cells
from src.cross_model.score_checks import checked_scores

BOOTSTRAP_SEED = 73021
BOOTSTRAP_DRAWS = 2000
CONTROLS = ('early_unassigned', 'entity_mention', 'other_attribute', 'late_unassigned')


def summary(values):
    """Same history bootstrap as v1, vectorized for saved-score reports."""
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {'n_histories': 0, 'mean': None, 'ci95_cluster_bootstrap': None}
    if not np.isfinite(x).all():
        raise ValueError('history summary requires finite values')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(x), size=(BOOTSTRAP_DRAWS, len(x)))
    means = x[indices].mean(axis=1)
    return {'n_histories': len(x), 'mean': float(x.mean()), 'median': float(np.median(x)),
            'trimmed_mean_10pct': trimmed_mean(x, .1), 'fraction_positive': float(np.mean(x > 0)),
            'ci95_cluster_bootstrap': list(map(float, np.quantile(means, [.025, .975])))}


def summarize_rows(history_rows):
    if not history_rows:
        return {'n_histories': 0, 'results': {}}
    keys = set(history_rows[0]) - {'history_id'}
    if any(set(r) - {'history_id'} != keys for r in history_rows):
        raise ValueError('history statistic columns do not match')
    return {'n_histories': len(history_rows),
            'results': {k: summary([r[k] for r in history_rows]) for k in sorted(keys)},
            'bootstrap_unit': 'history', 'bootstrap_draws': BOOTSTRAP_DRAWS,
            'bootstrap_seed': BOOTSTRAP_SEED}


def normalized(primary, corrected_live):
    """Exactly the already frozen v1 ratio-of-means denominator guard."""
    if not len(primary):
        return {'available': False, 'reason': 'no complete histories'}
    if len(primary) != len(corrected_live):
        raise ValueError('normalized sensitivity requires paired histories')
    reason = 'irrelevant-corrected live mean lower 95% history-bootstrap bound <= 1 nat'
    if summary(corrected_live)['ci95_cluster_bootstrap'][0] <= 1:
        return {'available': False, 'reason': reason}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(primary), size=(BOOTSTRAP_DRAWS, len(primary)))
    denominator = np.asarray(corrected_live)[indices].mean(axis=1)
    if not np.all(denominator > 1):
        return {'available': False, 'reason': 'a paired bootstrap irrelevant-corrected live denominator <= 1 nat'}
    ratios = np.asarray(primary)[indices].mean(axis=1) / denominator
    return {'available': True, 'ratio_of_means': float(np.mean(primary) / np.mean(corrected_live)),
            'ci95_history_bootstrap': list(map(float, np.quantile(ratios, [.025, .975]))),
            'definition': 'ratio of model means; never mean of per-history ratios',
            'bootstrap_draws': BOOTSTRAP_DRAWS, 'bootstrap_seed': BOOTSTRAP_SEED}


def paired_effects(rows, scores):
    _, pairs, _ = checked_scores(rows, scores)
    effects = []
    for pid, pair in sorted(pairs.items()):
        base, edit = pair[0], pair[1]
        source, replacement = base['source_value'], base['replacement_value']
        before, after = base['semantic_log_mass'], edit['semantic_log_mass']
        effect = {k: base[k] for k in ('history_id', 'condition', 'edited_variable', 'query')}
        effect.update(pair_id=pid,
                      source_change=after[source] - before[source],
                      replacement_change=after[replacement] - before[replacement],
                      identity_transfer=(after[replacement] - after[source]) - (before[replacement] - before[source]))
        for key in ('historical_entity_order', 'current_entity_order', 'unassigned_slot_order',
                    'experiment_kind', 'edited_field', 'edit_status'):
            if key in base:
                effect[key] = base[key]
        effects.append(effect)
    return effects


def query_relevance(values):
    """Symmetric entity-specific E: identical to history_contrasts in v1."""
    return .5 * ((values[('x', 'x')] - values[('x', 'z')]) +
                 (values[('z', 'z')] - values[('z', 'x')]))


def contrasts(rows, scores):
    effects = paired_effects(rows, scores)
    grouped = defaultdict(dict)
    for e in effects:
        key = (e['condition'], e['historical_entity_order'], e['current_entity_order'],
               e['edited_variable'], e['query'])
        if key in grouped[e['history_id']]:
            raise ValueError('duplicate history/order/edit/query effect')
        grouped[e['history_id']][key] = e['identity_transfer']
    expected = {cell[:-1] for cell in required_cells()}
    output = []
    for hid, cells in sorted(grouped.items()):
        if set(cells) != expected:
            raise ValueError('incomplete history condition/order/edit/query product')
        out = {'history_id': hid}
        for condition in CONDITIONS:
            by_order = {(h, c): {(v, q): cells[(condition, h, c, v, q)]
                                     for v in ('x', 'z') for q in ('x', 'z')}
                        for h in (0, 1) for c in (0, 1)}
            # Average nuisance orders within the history before each relevance contrast.
            averaged = {(v, q): float(np.mean([x[(v, q)] for x in by_order.values()]))
                        for v in ('x', 'z') for q in ('x', 'z')}
            for v in ('x', 'z'):
                other = 'z' if v == 'x' else 'x'
                out[f'R_{condition}_{v}'] = averaged[(v, v)] - averaged[(v, other)]
                for q in ('x', 'z'):
                    out[f'E_{condition}_edit_{v}_query_{q}'] = averaged[(v, q)]
            out[f'R_{condition}'] = query_relevance(averaged)
            for (h, c), values in by_order.items():
                out[f'R_{condition}_h{h}_c{c}'] = query_relevance(values)
            for label, orders in (('aligned', ((0, 0), (1, 1))), ('reversed', ((0, 1), (1, 0)))):
                out[f'R_{condition}_{label}'] = float(np.mean([query_relevance(by_order[o]) for o in orders]))
            out[f'R_{condition}_aligned_minus_reversed'] = out[f'R_{condition}_aligned'] - out[f'R_{condition}_reversed']
        out['R_live_minus_R_superseded'] = out['R_live'] - out['R_superseded']
        for control in CONTROLS:
            out[f'R_superseded_minus_R_{control}'] = out['R_superseded'] - out[f'R_{control}']
            out[f'R_live_minus_R_{control}'] = out['R_live'] - out[f'R_{control}']
            for label in ('aligned', 'reversed', 'x', 'z'):
                out[f'R_superseded_minus_R_{control}_{label}'] = out[f'R_superseded_{label}'] - out[f'R_{control}_{label}']
        output.append(out)
    return output


def report(rows, scores):
    if any(r.get('cross_model_stage') != 'confirmatory' for r in rows) or any(s.get('score_kind') != 'confirmatory' for s in scores):
        raise ValueError('causal reporting is restricted to all-trial confirmation scores')
    history_rows = contrasts(rows, scores)
    by_id = {r['history_id']: r for r in rows}
    return {'history_rows': history_rows, 'sequence_mass': summarize_rows(history_rows),
            'primary_contrasts': [f'R_superseded_minus_R_{c}' for c in CONTROLS[:3]],
            'continuity_contrast': 'R_superseded_minus_R_late_unassigned',
            'orientation': {str(o): summarize_rows([r for r in history_rows if by_id[r['history_id']]['orientation'] == o]) for o in (0, 1)},
            'estimand': 'cross_model_sequence_mass_v1; bounded surface-class prefix events without termination',
            'aggregation': 'mean E over all four independent order cells within history, then symmetric x/z query relevance; contrasts and bootstrap on histories',
            'all_trial_estimate': True,
            'live_order_caveat': 'live has one assignment block; current order labels duplicate it and are not an independent physical order factor'}

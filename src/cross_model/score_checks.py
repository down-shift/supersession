"""Shared validation of saved bounded semantic masses and paired score identities."""
import math
from collections import defaultdict

from src.cross_model.protocol import VALUES


def strict_rank(masses, answer):
    if set(masses) != set(VALUES) or any(not math.isfinite(v) for v in masses.values()):
        raise ValueError('complete finite bounded semantic masses required')
    if answer not in masses:
        raise ValueError('answer outside the fixed semantic candidate universe')
    return 1 + sum(v >= masses[answer] for k, v in masses.items() if k != answer)


def checked_scores(rows, scores, *, require_surfaces=False):
    """Reject duplicates before building maps; recompute ranks, never trust them."""
    if not rows or len(rows) != len(scores):
        raise ValueError('dataset/score count mismatch or empty input')
    dataset_ids = [r['example_id'] for r in rows]
    score_ids = [s['example_id'] for s in scores]
    if len(set(dataset_ids)) != len(dataset_ids) or len(set(score_ids)) != len(score_ids):
        raise ValueError('duplicate dataset/score example ID')
    if set(dataset_ids) != set(score_ids):
        raise ValueError('dataset/score ID mismatch')
    actual = {s['example_id']: s for s in scores}
    pairs, ranks = defaultdict(dict), {}
    for row in rows:
        score = actual[row['example_id']]
        if any(score.get(k) != value for k, value in row.items()):
            raise ValueError(f'score metadata mismatch at {row["example_id"]}')
        rank = strict_rank(score['semantic_log_mass'], row['answer'])
        if score.get('semantic_rank') is not None and score['semantic_rank'] != rank:
            raise ValueError(f'stored semantic rank differs from recomputed strict rank at {row["example_id"]}')
        if score.get('semantic_accuracy') is not None and score['semantic_accuracy'] != int(rank == 1):
            raise ValueError(f'stored semantic accuracy differs from recomputed strict rank at {row["example_id"]}')
        ranks[row['example_id']] = rank
        if require_surfaces:
            from src.cross_model.scoring import logsumexp
            surfaces = score.get('surface_likelihoods')
            if not isinstance(surfaces, dict) or set(surfaces) != set(VALUES):
                raise ValueError('saved surface likelihoods missing; do not regenerate with inference')
            for value, events in surfaces.items():
                if not events or len({tuple(e['ids']) for e in events}) != len(events):
                    raise ValueError('missing or duplicated saved surface events')
                mass = logsumexp([e['log_probability'] for e in events])
                if not math.isclose(mass, score['semantic_log_mass'][value], abs_tol=1e-8, rel_tol=1e-8):
                    raise ValueError('saved semantic mass differs from saved surface likelihoods')
        pid, direction = row['pair_id'], row['pair_direction']
        if type(direction) is not int or direction not in (0, 1):
            raise ValueError('pair_direction must mean baseline=0 or edited=1')
        if direction in pairs[pid]:
            raise ValueError('duplicate pair member')
        if row['example_id'] != f'{pid}:{direction}':
            raise ValueError('example ID does not match pair/member identity')
        pairs[pid][direction] = score
    for pid, pair in pairs.items():
        if set(pair) != {0, 1}:
            raise ValueError(f'incomplete pair {pid}')
        for key in ('history_id', 'condition', 'edited_variable', 'query', 'source_value', 'replacement_value'):
            if pair[0][key] != pair[1][key]:
                raise ValueError(f'pair member metadata mismatch: {key}')
    return actual, pairs, ranks

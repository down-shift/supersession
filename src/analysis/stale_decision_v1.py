"""Independent paired history bootstrap and secondary grouped prediction."""
from collections import defaultdict
import math
import numpy as np
from src.data.stale_decision_v1 import validate
from src.experiments.stale_decision_v1 import checked_scores


def bootstrap(values, seed=81004, draws=2000, level=.95):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all() or draws < 1:
        raise ValueError('invalid history bootstrap input')
    rng = np.random.default_rng(seed)
    means = x[rng.integers(len(x), size=(draws, len(x)))].mean(axis=1)
    return {'mean': float(x.mean()), 'confidence_interval': np.quantile(means, [(1-level)/2, 1-(1-level)/2]).tolist(),
            'n_histories': len(x), 'seed': seed, 'draws': draws, 'confidence_level': level}


def cells(rows, scores):
    checked_scores(rows, scores)
    byid = {s['example_id']: s for s in scores}; pairs = defaultdict(dict); out = []
    for r in rows:
        pairs[r['pair_id']][r['member']] = r
    for members in pairs.values():
        a = members[0]
        if a['family'] == 'current_only':
            continue
        b = members[1]; sa, sb = byid[a['example_id']], byid[b['example_id']]
        correct, wrong = a['answer'], a['wrong_action']
        # Live control uses fixed baseline action orientation; ground truth flips.
        la = sa['action_logp'][wrong] - sa['action_logp'][correct]
        lb = sb['action_logp'][wrong] - sb['action_logp'][correct]
        source, replacement = a['old_values']
        def edit_effect(field):
            return (sb[field][replacement] - sb[field][source]) - (sa[field][replacement] - sa[field][source])
        out.append({k: a[k] for k in ('history_id', 'family', 'task', 'difficulty', 'vocabulary', 'order_stratum', 'template_id')} |
                   {'D': la-lb, 'R_easy_local': edit_effect('easy_target')-edit_effect('easy_other'),
                    'wrong_rate_delta': int(sa['generated']['classification'] == 'wrong')-int(sb['generated']['classification'] == 'wrong'),
                    'strict_accuracy_delta': int(sa['generated']['classification'] == 'correct')-int(sb['generated']['classification'] == 'correct')})
    return out


def gates(rows, scores):
    checked_scores(rows, scores)
    byid = {s['example_id']: s for s in scores}
    result = {}
    for task in ('access', 'routing'):
        for difficulty in ('sequential', 'interleaved', 'competing'):
            rr = [r for r in rows if r['task'] == task and r['difficulty'] == difficulty]
            if not rr:
                continue
            ss = [byid[r['example_id']] for r in rr]
            controls = [byid[r['example_id']] for r in rr if r['family'] == 'current_only']
            hist = [byid[r['example_id']] for r in rr if r['family'] in ('superseded', 'updated_other', 'entity_mention', 'unassigned')]
            easy = np.mean([s['easy_generated']['classification'] == 'correct' for s in ss])
            downstream = np.mean([s['generated']['classification'] == 'correct' for s in controls])
            invalid = np.mean([s['generated']['classification'] == 'invalid' for s in ss])
            easy_invalid = np.mean([s['easy_generated']['classification'] == 'invalid' for s in ss])
            ha = np.mean([s['generated']['classification'] == 'correct' for s in hist])
            competence = easy >= .95 and downstream >= .90 and invalid <= .05 and easy_invalid <= .05
            result[task + ':' + difficulty] = {'direct_accuracy': float(easy), 'current_only_accuracy': float(downstream),
                'invalid_rate': float(invalid), 'easy_invalid_rate': float(easy_invalid), 'historical_accuracy': float(ha),
                'score_eligible': bool(competence), 'behavior_eligible': bool(competence and .65 <= ha <= .95),
                'n': len(rr), 'n_current_only': len(controls), 'n_historical': len(hist)}
    return {'cells': result, 'any_score_pass': any(v['score_eligible'] for v in result.values()),
            'none_pass_rule': 'stop confirmation; report feasibility failure; any redesign requires v2'}


def prediction(rows, scores, effects):
    """Nested-free fixed L2 grouped CV. Incremental held-out log loss is secondary."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    byid = {s['example_id']: s for s in scores}
    relevance = {(e['history_id'], e['family']): e['R_easy_local'] for e in effects}
    x, y, groups = [], [], []
    for r in rows:
        if r['family'] not in ('superseded', 'updated_other', 'entity_mention', 'unassigned'):
            continue
        s = byid[r['example_id']]; lp = s['easy_target']; old = r['old_values'][r['member']]
        confidence = math.exp(lp[r['correct_state']]) / sum(math.exp(v) for v in lp.values())
        # Confidence normalized within bounded retrieval candidate universe, explicitly not valid mass.
        nuisance = [int(r['task'] == 'routing'), int(r['difficulty'] == 'interleaved'),
                    int(r['difficulty'] == 'competing'), int(r['vocabulary'] == 'opaque'),
                    int(r['order_stratum'] == 'reversed')]
        nuisance += [int(r['family'] == f) for f in ('updated_other', 'entity_mention', 'unassigned')]
        x.append([lp[r['correct_state']]-lp[old], lp[old], confidence,
                  s['history_position'], s['prompt_tokens'], *nuisance, relevance[r['history_id'], r['family']]])
        y.append(int(s['generated']['classification'] != 'correct'))  # Invalid included as failure.
        groups.append(r['history_id'])
    x, y, groups = np.array(x), np.array(y), np.array(groups)
    if len(set(y)) < 2 or len(set(groups)) < 5:
        return {'available': False, 'reason': 'insufficient outcome variation/history groups; no H3 claim'}
    predictions = [np.zeros(len(y)), np.zeros(len(y))]
    for train, test in GroupKFold(5).split(x, y, groups):
        if len(set(y[train])) < 2:
            return {'available': False, 'reason': 'one-class training fold; no H3 claim'}
        for j, xx in enumerate((x[:, :-1], x)):
            estimator = make_pipeline(StandardScaler(), LogisticRegression(C=1., solver='lbfgs', max_iter=2000))
            estimator.fit(xx[train], y[train]); predictions[j][test] = estimator.predict_proba(xx[test])[:, 1]
    losses = []
    for p in predictions:
        p = np.clip(p, 1e-9, 1-1e-9)
        losses.append(-(y*np.log(p)+(1-y)*np.log(1-p)))
    history_improvements = [float((losses[0]-losses[1])[groups == g].mean()) for g in sorted(set(groups))]
    return {'available': True, 'outcome': 'strict downstream failure including invalid',
            'baseline_log_loss': float(losses[0].mean()), 'plus_R_log_loss': float(losses[1].mean()),
            'incremental_log_loss_improvement': bootstrap(history_improvements),
            'method': 'fixed C=1 L2 logistic; 5-fold history-grouped CV; no tuning; paired history bootstrap',
            'interpretation': 'secondary within-split prediction; held-out-template confirmation must be reported separately'}


def analyze(rows, scores):
    validate(rows); effects = cells(rows, scores)
    history = defaultdict(dict)
    for e in effects:
        history[e['history_id']][e['family']] = e
    reports = {}
    primary = {}
    for task in ('access', 'routing'):
        hh = [h for h in history.values() if h['superseded']['task'] == task]
        primary[task] = {'D_superseded': bootstrap([h['superseded']['D'] for h in hh], level=.975),
                         'Delta_semantic': bootstrap([h['superseded']['D']-h['updated_other']['D'] for h in hh], level=.975),
                         'Delta_mention': bootstrap([h['superseded']['D']-h['entity_mention']['D'] for h in hh])}
    # Preserve model/task/difficulty/vocabulary/order/template strata, never pool architectures.
    for task in ('access', 'routing'):
        for difficulty in ('sequential', 'interleaved', 'competing'):
            for vocabulary in ('natural', 'opaque'):
                for order in ('aligned', 'reversed', 'all'):
                    hh = [h for h in history.values() if h['superseded']['task'] == task and
                          h['superseded']['difficulty'] == difficulty and h['superseded']['vocabulary'] == vocabulary and
                          (order == 'all' or h['superseded']['order_stratum'] == order)]
                    if not hh:
                        continue
                    key = ':'.join((task, difficulty, vocabulary, order))
                    stats = {f'D_{f}': bootstrap([h[f]['D'] for h in hh]) for f in ('superseded', 'updated_other', 'entity_mention', 'unassigned', 'live')}
                    stats['Delta_semantic'] = bootstrap([h['superseded']['D']-h['updated_other']['D'] for h in hh])
                    stats['Delta_mention'] = bootstrap([h['superseded']['D']-h['entity_mention']['D'] for h in hh])
                    stats['wrong_rate_delta_superseded'] = bootstrap([h['superseded']['wrong_rate_delta'] for h in hh])
                    stats['strict_accuracy_delta_superseded'] = bootstrap([h['superseded']['strict_accuracy_delta'] for h in hh])
                    reports[key] = stats
    byid = {s['example_id']: s for s in scores}; generated = {}
    for family in ('superseded', 'updated_other', 'entity_mention', 'unassigned', 'current_only', 'live'):
        ss = [byid[r['example_id']] for r in rows if r['family'] == family]
        generated[family] = {'n': len(ss), **{k: sum(s['generated']['classification'] == k for s in ss)/len(ss) for k in ('correct', 'wrong', 'invalid')},
                             'mean_canonical_valid_mass': float(np.mean([s['canonical_valid_mass'] for s in ss])),
                             'relaxed_accuracy_diagnostic': float(np.mean([byid[r['example_id']]['generated']['relaxed_action'] == r['answer'] for r in rows if r['family'] == family]))}
    return {'model_id': scores[0]['model_id'], 'primary': primary, 'strata': reports, 'generation': generated,
            'history_effects': effects, 'gates': gates(rows, scores), 'H3_secondary': prediction(rows, scores, effects)}

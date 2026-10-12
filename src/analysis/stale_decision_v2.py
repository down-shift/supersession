"""stale_decision_v2 analysis: identical to the v1b analysis, applied to v2 data and scores
(docs/stale_decision_v2.md). The v1 analysis functions are run with v2's data validator and score checks."""
from collections import Counter, defaultdict
from contextlib import contextmanager
import math

import numpy as np

from src.analysis import stale_decision_v1 as v1a
from src.data.stale_decision_v2 import validate
from src.experiments.stale_decision_v2 import checked_scores

HISTORICAL = ('superseded', 'updated_other', 'entity_mention', 'unassigned')


@contextmanager
def v1b_validation():
    """The v1 analysis functions validate scores with v1's strict checks (log probability exactly <= 0).
    For v1b they must use v1b's checks, which add the documented float32 rounding tolerance; nothing else
    in the v1 analysis changes. The substitution is undone on exit."""
    original = v1a.checked_scores, v1a.validate
    v1a.checked_scores, v1a.validate = checked_scores, validate
    try:
        yield
    finally:
        v1a.checked_scores, v1a.validate = original


def _logistic_cv(x, y, groups, seed=81004):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    p = np.zeros(len(y))
    for train, test in GroupKFold(5).split(x, y, groups):
        if len(set(y[train])) < 2:
            return None
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1., solver='lbfgs', max_iter=2000))
        m.fit(x[train], y[train])
        p[test] = m.predict_proba(x[test])[:, 1]
    return np.clip(p, 1e-9, 1 - 1e-9)


def _metrics(y, p, groups):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return {'auroc': float(roc_auc_score(y, p)), 'average_precision': float(average_precision_score(y, p)),
            'prevalence': float(y.mean()), 'brier': float(brier_score_loss(y, p)), 'log_loss': float(loss.mean()),
            '_loss': loss}


def prediction(rows, scores, effects, task=None, outcome='failure'):
    """H3 with predictors available before the downstream question is asked.

    pre_downstream: easy current-historical margin, historical candidate log mass, easy confidence,
    edited-token position, downstream prompt length, design indicators. plus_R adds R_easy_local.
    contemporaneous: downstream correct-action confidence alone, a labelled comparison that is NOT
    available before evaluating the downstream question and never enters the pre-downstream models.
    outcome: 'failure' (wrong or invalid) or 'wrong' (wrong only)."""
    byid = {s['example_id']: s for s in scores}
    relevance = {(e['history_id'], e['family']): e['R_easy_local'] for e in effects}
    pre, plus, contemp, y, groups = [], [], [], [], []
    for r in rows:
        if (task is not None and r['task'] != task) or r['family'] not in HISTORICAL:
            continue
        s = byid[r['example_id']]
        lp = s['easy_target']
        old = r['old_values'][r['member']]
        confidence = math.exp(lp[r['correct_state']]) / sum(math.exp(v) for v in lp.values())
        nuisance = [int(r['task'] == 'routing'), int(r['difficulty'] == 'interleaved'),
                    int(r['difficulty'] == 'competing'), int(r['vocabulary'] == 'opaque'),
                    int(r['order_stratum'] == 'reversed')] + [int(r['family'] == f) for f in HISTORICAL[1:]]
        base = [lp[r['correct_state']] - lp[old], lp[old], confidence, s['history_position'], s['prompt_tokens'], *nuisance]
        pre.append(base)
        plus.append(base + [relevance[r['history_id'], r['family']]])
        a = s['action_logp']
        contemp.append([math.exp(a[r['answer']]) / sum(math.exp(v) for v in a.values())])
        cls = s['generated']['classification']
        y.append(int(cls != 'correct') if outcome == 'failure' else int(cls == 'wrong'))
        groups.append(r['history_id'])
    y, groups = np.array(y), np.array(groups)
    if len(set(y)) < 2 or len(set(groups)) < 5:
        return {'available': False, 'events': int(y.sum()), 'n': int(len(y)),
                'reason': 'insufficient outcome variation or history groups; no H3 claim'}
    out = {'available': True, 'outcome': outcome, 'events': int(y.sum()), 'n': int(len(y)),
           'method': 'fixed C=1 L2 logistic; 5-fold history-grouped CV; no tuning; paired history bootstrap'}
    losses = {}
    for name, x in (('pre_downstream', pre), ('pre_downstream_plus_R', plus), ('contemporaneous_action_confidence', contemp)):
        p = _logistic_cv(np.array(x, dtype=float), y, groups)
        if p is None:
            return {'available': False, 'reason': 'one-class training fold; no H3 claim', 'events': int(y.sum())}
        m = _metrics(y, p, groups)
        losses[name] = m.pop('_loss')
        out[name] = m
    gain = [float((losses['pre_downstream'] - losses['pre_downstream_plus_R'])[groups == g].mean()) for g in sorted(set(groups))]
    out['R_incremental_log_loss_improvement'] = v1a.bootstrap(gain)
    out['note'] = ('contemporaneous_action_confidence uses the downstream forward pass and is a comparison, '
                   'not a predictor available before the downstream question')
    return out


def historical_retrieval(rows, scores, task=None):
    byid = {s['example_id']: s for s in scores}
    per_history = defaultdict(list)
    for r in rows:
        if r['family'] == 'superseded' and (task is None or r['task'] == task):
            per_history[r['history_id']].append(byid[r['example_id']]['historical']['generated']['classification'])
    if not per_history:
        return {'n_histories': 0}
    out = {}
    for name in ('correct', 'wrong', 'invalid'):
        out[name] = v1a.bootstrap([sum(c == name for c in v) / len(v) for v in per_history.values()])
    return out


def generated_switches(rows, scores):
    """Secondary (added after development, before the frozen gate): paired generated wrong-action switch per
    history, P(wrong | old implies wrong) - P(wrong | old implies correct), by task x construction and by task
    x difficulty x construction, and switch-based Delta_semantic / Delta_mention; 95% history bootstrap."""
    byid = {s['example_id']: s for s in scores}
    per = defaultdict(dict)
    for r in rows:
        if r['family'] in HISTORICAL:
            per[(r['task'], r['difficulty'], r['family'], r['history_id'])][r['member']] = int(
                byid[r['example_id']]['generated']['classification'] == 'wrong')
    switch = {k: v[0] - v[1] for k, v in per.items()}
    out = {}
    for task in ('access', 'routing'):
        for difficulty in (None, 'sequential', 'interleaved', 'competing'):
            def values(family):
                return {h: d for (t, dif, f, h), d in switch.items()
                        if t == task and f == family and (difficulty is None or dif == difficulty)}
            fam = {f: values(f) for f in HISTORICAL}
            key = task + (':' + difficulty if difficulty else ':all')
            out[key] = {f'switch_{f}': v1a.bootstrap(list(v.values())) for f, v in fam.items()}
            hs = sorted(fam['superseded'])
            out[key]['switch_Delta_semantic'] = v1a.bootstrap([fam['superseded'][h] - fam['updated_other'][h] for h in hs])
            out[key]['switch_Delta_mention'] = v1a.bootstrap([fam['superseded'][h] - fam['entity_mention'][h] for h in hs])
            out[key]['switch_counts'] = {f: {'to_wrong': sum(x == 1 for x in v.values()), 'to_correct': sum(x == -1 for x in v.values()),
                                             'n_histories': len(v)} for f, v in fam.items()}
    return out


def classify_errors(rows, scores):
    """What a wrong downstream answer implies: which state's action it matches."""
    byid = {s['example_id']: s for s in scores}
    counts = defaultdict(Counter)
    for r in rows:
        s = byid[r['example_id']]
        cls = s['generated']['classification']
        key = f"{r['task']}:{r['difficulty']}:{r['family']}"
        if cls == 'correct':
            counts[key]['correct'] += 1
        elif cls == 'invalid':
            counts[key]['invalid:' + ('cap' if not s['terminated'] else 'format')] += 1
        else:
            implied = [st for st, act in r['policy'].items() if act == r['wrong_action']]
            old = r['old_values'][r['member']] if r['family'] != 'current_only' else None
            other_current = r['currents'][1]
            tags = []
            if old in implied:
                tags.append('matches_edited_old_value')
            if other_current in implied:
                tags.append('matches_other_entity_current')
            counts[key]['wrong:' + ('+'.join(tags) or 'unattributed')] += 1
    return {k: dict(v) for k, v in sorted(counts.items())}


def analyze(rows, scores):
    validate(rows)
    checked_scores(rows, scores)
    with v1b_validation():
        base = v1a.analyze(rows, scores)
    effects = base['history_effects']
    h3 = {}
    for outcome in ('failure', 'wrong'):
        h3[outcome] = {'overall': prediction(rows, scores, effects, outcome=outcome),
                       'by_task': {t: prediction(rows, scores, effects, task=t, outcome=outcome) for t in ('access', 'routing')}}
    base.pop('H3_secondary')
    base['H3_v1b'] = h3
    base['historical_retrieval'] = {'overall': historical_retrieval(rows, scores),
                                    'by_task': {t: historical_retrieval(rows, scores, t) for t in ('access', 'routing')}}
    base['error_classes'] = classify_errors(rows, scores)
    base['generated_switches_secondary'] = generated_switches(rows, scores)
    base['protocol'] = 'stale_decision_v2'
    return base

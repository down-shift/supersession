"""Derive every number shown in paper_v2 from the sealed stale_decision records.

    uv run --extra dev python -m paper_v2.data.derive

Inputs are the completed analysis reports and the confirmation score file (verified against its completion
record). Output: paper_v2/data/derived.json with input hashes. Bootstrap: the protocol's history bootstrap
(seed 81004, 2,000 draws, 95% unless stated).
"""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from src.analysis.stale_decision_v1 import bootstrap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs'
CONF = OUT / 'stale_decision_v2/confirmation'
REPORTS = {
    'v1b_development': OUT / 'stale_decision_v1b/dev/development_report.json',
    'v1b_gate': OUT / 'stale_decision_v1b/gate/gate_report.json',
    'v2_development': OUT / 'stale_decision_v2/dev/development_report.json',
    'v2_gate': OUT / 'stale_decision_v2/gate/gate_report.json',
    'v2_confirmation': CONF / 'confirmation_report.json',
    'gemma_development': OUT / 'stale_decision_v2m/gemma3_4b/development_report.json',
    'gemma_gate': OUT / 'stale_decision_v2m/gemma3_4b/gate/gate_report.json',
}
TASKS = ('access', 'routing')
CELLS = ('sequential', 'interleaved', 'competing')
FAMILIES = ('superseded', 'updated_other', 'entity_mention', 'unassigned')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ci(b):
    return {'mean': b['mean'], 'ci': b['confidence_interval'], 'level': b['confidence_level'], 'n': b['n_histories']}


def construction_effects(report):
    """D per construction, per task and per task x cell, from the report's per-history effects."""
    by = defaultdict(list)
    for h in report['history_effects']:
        for key in (h['task'], f"{h['task']}:{h['difficulty']}"):
            by[(key, h['family'])].append(h['D'])
    return {f'{k}:{f}': ci(bootstrap(v)) for (k, f), v in sorted(by.items())}


def member_accuracy():
    """Generated accuracy of the superseded construction by member, per cell, from the raw confirmation scores."""
    complete = json.loads((CONF / 'confirmation_scores.jsonl.complete.json').read_text())
    if sha(CONF / 'confirmation_scores.jsonl') != complete['scores_sha256']:
        raise ValueError('confirmation scores differ from their completion record')
    rows = {r['example_id']: r for r in map(json.loads, (CONF / 'confirmation.jsonl').open())}
    acc = defaultdict(lambda: defaultdict(list))
    for s in map(json.loads, (CONF / 'confirmation_scores.jsonl').open()):
        r = rows[s['example_id']]
        if r['family'] in FAMILIES:
            correct = float(s['generated']['classification'] == 'correct')
            acc[f"{r['task']}:{r['difficulty']}:{r['family']}"][r['member']].append((r['history_id'], correct))
    out = {}
    for key, members in sorted(acc.items()):
        wrong = dict(members[0])   # member 0: obsolete value implies the wrong action
        right = dict(members[1])   # member 1: obsolete value implies the correct action
        ids = sorted(wrong)
        out[key] = {'old_implies_wrong': ci(bootstrap([wrong[i] for i in ids])),
                    'old_implies_correct': ci(bootstrap([right[i] for i in ids]))}
    return out, complete['scores_sha256']


def summary(report):
    p = report['primary']
    return {t: {k: ci(v) for k, v in p[t].items()} for t in TASKS}


def main():
    reports = {k: json.loads(p.read_text()) for k, p in REPORTS.items()}
    conf = reports['v2_confirmation']
    members, scores_sha = member_accuracy()
    derived = {
        'inputs': {k: {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for k, p in REPORTS.items()},
        'confirmation_scores_sha256': scores_sha,
        'confirmation': {
            'primary': summary(conf),
            'constructions': construction_effects(conf),
            'switches': conf['generated_switches_secondary'],
            'competence': conf['gates']['cells'],
            'historical_retrieval': {t: ci(conf['historical_retrieval']['by_task'][t]['correct']) for t in TASKS} |
                                    {'overall': ci(conf['historical_retrieval']['overall']['correct'])},
            'member_accuracy': members,
            'strata': {k: {m: ci(v[m]) for m in ('D_superseded', 'Delta_semantic', 'Delta_mention')}
                       for k, v in conf['strata'].items() if k.endswith(':all')},
            'H3': {o: {'overall': conf['H3_v1b'][o]['overall'], **conf['H3_v1b'][o]['by_task']} for o in conf['H3_v1b']},
            'error_classes': {k: v for k, v in conf['error_classes'].items() if k.split(':')[2] == 'superseded'},
        },
        'replication': {k: summary(reports[k]) for k in ('v1b_development', 'v1b_gate', 'v2_development', 'v2_gate')},
        'replication_switches': {k: {t: reports[k]['generated_switches_secondary'][f'{t}:all'] for t in TASKS}
                                 for k in ('v1b_gate', 'v2_development', 'v2_gate')},  # v1b development predates this analysis
        'gates': {k: reports[k]['gates'] for k in ('v1b_gate', 'v2_gate', 'gemma_development', 'gemma_gate')},
    }
    for o in derived['confirmation']['H3'].values():
        for v in o.values():  # log loss of the constant (prevalence-only) predictor, for scale
            p = v['events'] / v['n']
            v['prevalence_only_log_loss'] = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    r = defaultdict(list)
    for h in conf['history_effects']:
        if h['family'] == 'superseded':
            r[h['task']].append(h['R_easy_local'])
    derived['confirmation']['R_easy_local_superseded'] = {t: ci(bootstrap(v)) for t, v in sorted(r.items())}
    path = ROOT / 'paper_v2/data/derived.json'
    path.write_text(json.dumps(derived, indent=1, sort_keys=True) + '\n')
    print(path)


if __name__ == '__main__':
    main()

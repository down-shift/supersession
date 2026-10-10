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
# Every model attempted (docs/stale_decision_v2_models.md, docs/stale_decision_v2_replication.md). Reports that do
# not exist yet are skipped, so the derivation can be rerun as models finish.
V2R = OUT / 'stale_decision_v2r'
MODEL_RUNS = {
    'Qwen3-8B': {'gate': OUT / 'stale_decision_v2/gate/gate_report.json', 'confirmation': CONF / 'confirmation_report.json'},
    'Gemma 3 4B': {'gate': OUT / 'stale_decision_v2m/gemma3_4b/gate/gate_report.json'},
    **{label: {'gate': V2R / f'{name}/gate/gate_report.json',
               'confirmation': V2R / f'{name}/confirmation/confirmation_report.json'}
       for label, name in [('Qwen3-14B', 'qwen3_14b'), ('Gemma 3 12B', 'gemma3_12b'),
                           ('Granite 3.1 8B', 'granite31_8b'), ('Mistral 7B v0.3', 'mistral7b'),
                           ('Falcon3 7B', 'falcon3_7b')]},
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


def checked_scores(conf_dir):
    """Rows and score records of a confirmation run, after checking the scores against their completion record."""
    complete = json.loads((conf_dir / 'confirmation_scores.jsonl.complete.json').read_text())
    if sha(conf_dir / 'confirmation_scores.jsonl') != complete['scores_sha256']:
        raise ValueError('confirmation scores differ from their completion record')
    rows = {r['example_id']: r for r in map(json.loads, (conf_dir / 'confirmation.jsonl').open())}
    return rows, list(map(json.loads, (conf_dir / 'confirmation_scores.jsonl').open())), complete['scores_sha256']


def member_accuracy(conf_dir=CONF):
    """Generated accuracy of each construction by member, per cell, from the raw confirmation scores."""
    rows, scores, scores_sha = checked_scores(conf_dir)
    acc = defaultdict(lambda: defaultdict(list))
    for s in scores:
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
    return out, scores_sha


def run_stats(conf_dir):
    """Logged wall-clock time and mean prompt length of a confirmation run (Appendix C)."""
    timing = json.loads((conf_dir / 'confirmation_scores.jsonl.timing.json').read_text())
    _, scores, _ = checked_scores(conf_dir)
    return {'gpu_hours': timing['seconds'] / 3600, 'seconds_per_record': timing['seconds_per_record'],
            'mean_prompt_tokens': sum(x['prompt_tokens'] for x in scores) / len(scores)}


def competent_effects(report, cells):
    """Compact cross-model entries: confirmation histories pooled over the model's gate-competent cells of a task."""
    out = {}
    for t in TASKS:
        keep = {c.split(':')[1] for c in cells if c.startswith(f'{t}:')}
        if not keep:
            out[t] = None
            continue
        by = defaultdict(dict)
        for h in report['history_effects']:
            if h['task'] == t and h['difficulty'] in keep:
                by[h['history_id']][h['family']] = h
        hh = list(by.values())
        out[t] = {'cells': sorted(keep), 'n': len(hh),
                  'D_superseded': ci(bootstrap([h['superseded']['D'] for h in hh], level=.975)),
                  'Delta_semantic': ci(bootstrap([h['superseded']['D'] - h['updated_other']['D'] for h in hh], level=.975)),
                  'Delta_mention': ci(bootstrap([h['superseded']['D'] - h['entity_mention']['D'] for h in hh])),
                  'switch_superseded': ci(bootstrap([h['superseded']['wrong_rate_delta'] for h in hh])),
                  'switch_entity_mention': ci(bootstrap([h['entity_mention']['wrong_rate_delta'] for h in hh]))}
    return out


def model_summaries():
    out, inputs = {}, {}
    for label, paths in MODEL_RUNS.items():
        if not paths['gate'].exists():
            continue
        gate = json.loads(paths['gate'].read_text())
        inputs[f'{label}:gate'] = {'path': str(paths['gate'].relative_to(ROOT)), 'sha256': sha(paths['gate'])}
        cells = sorted(k for k, c in gate['gates']['cells'].items() if c['score_eligible'])
        entry = {'gate_cells': gate['gates']['cells'], 'competent_cells': cells, 'any_score_pass': gate['gates']['any_score_pass']}
        cpath = paths.get('confirmation')
        if cpath and cpath.exists():
            conf = json.loads(cpath.read_text())
            inputs[f'{label}:confirmation'] = {'path': str(cpath.relative_to(ROOT)), 'sha256': sha(cpath)}
            conf_cells = sorted(k for k, c in conf['gates']['cells'].items() if c['score_eligible'])
            entry['confirmation'] = {'primary': summary(conf), 'competent': competent_effects(conf, cells),
                                     'confirmation_competent_cells': conf_cells,
                                     'competent_confirmation': competent_effects(conf, conf_cells),
                                     'switches': conf['generated_switches_secondary'],
                                     'competence': conf['gates']['cells'],
                                     'historical_retrieval': {t: ci(conf['historical_retrieval']['by_task'][t]['correct'])
                                                              for t in TASKS},
                                     'H3': {t: conf['H3_v1b']['failure']['by_task'][t] for t in TASKS} |
                                           {'overall': conf['H3_v1b']['failure']['overall']},
                                     'constructions': construction_effects(conf),
                                     'joint_correctness': joint_correctness(cpath.parent),
                                     'member_accuracy': member_accuracy(cpath.parent)[0],
                                     'run': run_stats(cpath.parent)}
            # the per-history switch must reproduce the report's switch rate
            for t in TASKS:
                hh = [h for h in conf['history_effects'] if h['task'] == t and h['family'] == 'superseded']
                mean = sum(h['wrong_rate_delta'] for h in hh) / len(hh)
                if abs(mean - conf['generated_switches_secondary'][f'{t}:all']['switch_superseded']['mean']) > 1e-12:
                    raise ValueError('per-history switches do not reproduce the report')
        out[label] = entry
    return out, inputs


def joint_correctness(conf_dir=CONF):
    """Plan amendment A: decision effects restricted to histories where the model demonstrably tracks the
    record. A history is kept for construction c if (i) direct current-state retrieval is correct under both
    members of c, (ii) initial-state retrieval is correct under both superseded members of the history, and
    (iii) the history's current-only decision is correct. Reports retained n, D_c, net switch rate and counts,
    95% history bootstrap; per task and per task-cell."""
    rows, scores, _ = checked_scores(conf_dir)
    hist = defaultdict(dict)  # history_id -> {(family, member): (row, score)}
    for sc in scores:
        r = rows[sc['example_id']]
        hist[r['history_id']][(r['family'], r['member'])] = (r, sc)
    groups = defaultdict(lambda: defaultdict(list))
    totals = defaultdict(int)
    for items in hist.values():
        r0 = items[('superseded', 0)][0]
        initial_ok = all(items[('superseded', m)][1]['historical']['generated']['classification'] == 'correct' for m in (0, 1))
        current_only_ok = items[('current_only', 0)][1]['generated']['classification'] == 'correct'
        for f in FAMILIES:
            (a, sa), (b, sb) = items[(f, 0)], items[(f, 1)]
            retrieval_ok = all(x['easy_generated']['classification'] == 'correct' for x in (sa, sb))
            for key in (f"{r0['task']}:{f}", f"{r0['task']}:{r0['difficulty']}:{f}"):
                totals[key] += 1
                if not (retrieval_ok and initial_ok and current_only_ok):
                    continue
                wrong, correct = a['wrong_action'], a['answer']
                la = sa['action_logp'][wrong] - sa['action_logp'][correct]
                lb = sb['action_logp'][wrong] - sb['action_logp'][correct]
                sw = int(sa['generated']['classification'] == 'wrong') - int(sb['generated']['classification'] == 'wrong')
                groups[key]['D'].append(la - lb)
                groups[key]['switch'].append(sw)
    out = {}
    for key in sorted(totals):
        g = groups[key]
        n = len(g['switch'])
        out[key] = {'n_total': totals[key], 'n_retained': n,
                    'D': ci(bootstrap(g['D'])) if n else None,
                    'switch': ci(bootstrap(g['switch'])) if n else None,
                    'to_wrong': sum(1 for x in g['switch'] if x == 1),
                    'to_correct': sum(1 for x in g['switch'] if x == -1)}
    return out


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
    derived['confirmation']['joint_correctness'] = joint_correctness()
    derived['models'], model_inputs = model_summaries()
    derived['inputs'].update(model_inputs)
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

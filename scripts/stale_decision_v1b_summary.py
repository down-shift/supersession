"""Readable summary of a stale_decision_v1b analysis report (no inference, no new estimates).

    python -m scripts.stale_decision_v1b_summary REPORT.json
"""
import json
import sys


def ci(e):
    lo, hi = e['confidence_interval']
    return f"{e['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main(path):
    r = json.load(open(path))
    print('model', r['model_id'])
    print('\n== Gates (feasibility, all cells)')
    for k, v in r['gates']['cells'].items():
        print(f"{k:24s} direct {v['direct_accuracy']:.3f}  current-only {v['current_only_accuracy']:.3f}  "
              f"with-history {v['historical_accuracy']:.3f}  invalid {v['invalid_rate']:.3f}  "
              f"score-eligible {v['score_eligible']}  behavior-eligible {v['behavior_eligible']}")
    print('\n== Primary (97.5% for D_superseded and Delta_semantic; 95% for Delta_mention)')
    for task, v in r['primary'].items():
        print(task, {k: ci(e) for k, e in v.items()})
    print('\n== D by construction, all orders (95%)')
    for k, v in r['strata'].items():
        if k.endswith(':all'):
            print(f"{k:36s}", '  '.join(f"{f.replace('D_', '')} {ci(v[f])}" for f in
                                        ('D_superseded', 'D_updated_other', 'D_entity_mention', 'D_unassigned', 'D_live')))
    print('\n== Generated outcomes by task and construction (history-averaged)')
    for k, v in r['generation']['by_task_family'].items():
        if v.get('n_histories'):
            print(f"{k:28s} correct {v['correct']['mean']:.3f}  wrong {v['wrong']['mean']:.3f}  invalid {v['invalid']['mean']:.3f}")
    print('\n== Historical retrieval (superseded construction)')
    print(json.dumps({k: {m: round(e['mean'], 3) for m, e in v.items()} if 'correct' in v else v
                      for k, v in [('overall', r['historical_retrieval']['overall'])] +
                      list(r['historical_retrieval']['by_task'].items())}))
    print('\n== Wrong-answer attribution')
    for k, v in r['error_classes'].items():
        if any(x.startswith('wrong') for x in v):
            print(k, v)
    print('\n== H3 (development, within-split CV)')
    for outcome, v in r['H3_v1b'].items():
        o = v['overall']
        if not o.get('available'):
            print(outcome, 'unavailable:', o.get('reason'), 'events', o.get('events'))
            continue
        print(outcome, 'events', o['events'], 'of', o['n'])
        for name in ('pre_downstream', 'pre_downstream_plus_R', 'contemporaneous_action_confidence'):
            m = o[name]
            print(f"  {name:36s} AUROC {m['auroc']:.3f}  AP {m['average_precision']:.3f} (prev {m['prevalence']:.3f})  "
                  f"Brier {m['brier']:.4f}  logloss {m['log_loss']:.4f}")
        print('  R incremental log-loss gain', ci(o['R_incremental_log_loss_improvement']))


if __name__ == '__main__':
    main(sys.argv[1])

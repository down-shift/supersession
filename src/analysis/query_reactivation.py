"""Post-hoc query-reactivation analysis for existing four-query outputs.

This module is explicitly exploratory: it does not change the frozen experiment.
All aggregation and resampling units are histories.
"""
from collections import defaultdict
import numpy as np
from src.analysis.metrics import trimmed_mean


def summarize_histories(values, seed=20261002, n_boot=10000, permutation=True):
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite history statistic; histories must not be silently excluded")
    if n_boot < 1:
        raise ValueError("bootstrap draws must be positive")
    if not len(x):
        return {"n_histories": 0, "mean": None, "median": None,
                "trimmed_mean_10pct": None, "fraction_positive": None,
                "ci95_cluster_bootstrap": None,
                "sign_flip_permutation_p_two_sided": None}
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(x, (n_boot, len(x)), replace=True), axis=1)
    signs = rng.choice([-1.0, 1.0], (n_boot, len(x)))
    p = (np.sum(np.abs(np.mean(signs*x, axis=1)) >= abs(x.mean()))+1)/(n_boot+1)
    return {"n_histories": int(len(x)), "mean": float(x.mean()),
            "median": float(np.median(x)), "trimmed_mean_10pct": trimmed_mean(x, .1),
            "fraction_positive": float(np.mean(x > 0)),
            "ci95_cluster_bootstrap": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "sign_flip_permutation_p_two_sided": float(p) if permutation else None}


def derive_effects(pair_rows):
    """Calculate unchanged identity_transfer from saved matched logits."""
    pairs = defaultdict(dict)
    for row in pair_rows:
        direction = row.get('pair_direction')
        if direction not in (0, 1) or direction in pairs[row['pair_id']]:
            raise ValueError(f"invalid or duplicate pair direction: {row.get('pair_id')}")
        pairs[row['pair_id']][direction] = row
    cells = defaultdict(dict)
    for pid, members in pairs.items():
        if set(members) != {0, 1}:
            raise ValueError(f"pair {pid} must contain directions 0 and 1")
        base, edit = members[0], members[1]
        if any(base.get(k) != edit.get(k) for k in (
                'history_id', 'edited_binding', 'query_id', 'source_value',
                'replacement_value', 'matching_values', 'variables', 'orientation')):
            raise ValueError(f"pair {pid} changes history/binding/query")
        source, replacement = base['source_value'], base['replacement_value']
        if source == replacement:
            raise ValueError(f"pair {pid} source and replacement must differ")
        lb, le = base['candidate_logits'], edit['candidate_logits']
        if set(lb) != set(le) or not np.isfinite(list(lb.values()) + list(le.values())).all():
            raise ValueError(f"pair {pid} has invalid or nonfinite candidate logits")
        value = (le[replacement]-le[source])-(lb[replacement]-lb[source])
        key = (base['edited_binding'], base['query_id'])
        hid = base['history_id']
        if key in cells[hid]:
            raise ValueError(f"duplicate history edit/query cell for {hid}: {key}")
        cells[hid][key] = float(value)
    return cells


def score_protocol_record(row, raw_scorer, baseline_cache, stage):
    """Apply the isolated protocol's stage routing to one raw scored example."""
    scored = raw_scorer(row)
    if stage != 'confirmatory':
        for key in ('candidate_logits','candidate_probabilities','identity_transfer','matched_edit_effect'):
            scored.pop(key, None)
        scored['causal_effects_computed'] = False
        return scored
    if row.get('pair_direction') == 0:
        baseline_cache[row['pair_id']] = scored
        return scored
    baseline = baseline_cache.get(row['pair_id'])
    if baseline is None:
        raise ValueError('edited member lacks baseline')
    source, replacement = row['source_value'], row['replacement_value']
    lb, le = baseline['candidate_logits'], scored['candidate_logits']
    scored['identity_transfer'] = float((le[replacement]-le[source])-(lb[replacement]-lb[source]))
    scored['causal_effects_computed'] = True
    return scored


def resume_baseline_cache(completed, stage):
    """Rebuild only baseline pairs; edited checkpoints must have a saved baseline."""
    if stage != 'confirmatory':
        return {}
    cache = {}
    for row in completed:
        if row.get('pair_direction') == 0:
            if row['pair_id'] in cache:
                raise ValueError(f"duplicate baseline checkpoint: {row['pair_id']}")
            cache[row['pair_id']] = row
        elif row.get('pair_direction') == 1 and row.get('pair_id') not in cache:
            raise ValueError(f"checkpoint edited member lacks baseline: {row.get('pair_id')}")
    return cache


def evaluate_competence_gate(rows, scores, threshold=.97):
    """Aggregate current/historical competence; strata remain diagnostic only."""
    if threshold != .97:
        raise ValueError('query_reactivation_v1 frozen gate threshold is 0.97')
    expected={r['example_id']:r for r in rows}; actual={r['example_id']:r for r in scores}
    if len(expected)!=len(rows) or len(actual)!=len(scores) or set(expected)!=set(actual):
        raise ValueError('gate scores must exactly cover the dataset')
    tasks={'current':[],'historical':[]}; diagnostics=defaultdict(list)
    for eid,row in expected.items():
        score=actual[eid]
        if any(score.get(k)!=v for k,v in row.items()): raise ValueError('gate score metadata differs from dataset')
        if any(k in score for k in ('identity_transfer','matched_edit_effect')) or score.get('causal_effects_computed') is not False:
            raise ValueError('gate score contains or permits causal effects')
        acc=score.get('full_vocab_next_token_accuracy')
        if acc not in (0,1): raise ValueError('invalid full-vocabulary competence score')
        task='current' if row['query_id'].startswith('current_') else 'historical'
        tasks[task].append(acc)
        variable=row['query_id'][-1]
        diagnostics[(task,variable,row['orientation'])].append(acc)
    summary={k:{'n':len(v),'full_vocab_accuracy':float(np.mean(v))} for k,v in sorted(tasks.items())}
    by_cell={'|'.join(map(str,k)):{'n':len(v),'full_vocab_accuracy':float(np.mean(v))} for k,v in sorted(diagnostics.items())}
    passed=all(summary[k]['n']>0 and summary[k]['full_vocab_accuracy']>=threshold for k in tasks)
    return {'pass':passed,'threshold':threshold,'aggregate_task_competence':summary,
            'query_variable_orientation_diagnostics':by_cell,
            'decision_scope':'aggregate current and historical competence only; x/z/orientation cells diagnostic, not veto criteria',
            'causal_effects_computed':False}


def history_contrasts(pair_rows):
    cells = derive_effects(pair_rows)
    required = [(b, q) for b in ('old_x','old_z')
                for q in ('current_x','initial_x','current_z','initial_z')]
    output = []
    for hid, e in sorted(cells.items()):
        if any(k not in e for k in required):
            raise ValueError(f"history {hid} missing required edit/query cells")
        old_current = .5*((e['old_x','current_x']-e['old_x','current_z'])+
                          (e['old_z','current_z']-e['old_z','current_x']))
        old_historical = .5*((e['old_x','initial_x']-e['old_x','initial_z'])+
                             (e['old_z','initial_z']-e['old_z','initial_x']))
        current_cells=[('current_x',q) for q in ('current_x','current_z')]+[('current_z',q) for q in ('current_x','current_z')]
        current_selectivity = (.5*((e['current_x','current_x']-e['current_x','current_z'])+
                                   (e['current_z','current_z']-e['current_z','current_x']))
                               if all(k in e for k in current_cells) else None)
        output.append({'history_id': hid, 'R_old_current': old_current,
                       'R_old_historical': old_historical,
                       'Delta_reactivate': old_historical-old_current,
                       'R_current_binding_selectivity': current_selectivity})
    return output


def summarize(rows, seed=20261002, n_boot=10000):
    keys = ('R_old_current','R_old_historical','Delta_reactivate','R_current_binding_selectivity')
    return {'analysis_status':'POST-HOC / EXPLORATORY for existing data; fresh query_reactivation_v1 data are confirmatory only under frozen protocol',
            'bootstrap_unit':'history_id','n_histories':len(rows),
            'estimands':{k:summarize_histories([r[k] for r in rows if r[k] is not None],seed,n_boot) for k in keys}}

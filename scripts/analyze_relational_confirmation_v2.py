"""Read-only extended reanalysis of saved relational v2 confirmation bundles."""
import csv
import json
from pathlib import Path

from src.cross_model.protocol import read_sealed, digest
from src.cross_model.robustness_v2 import CONDITIONS, concrete_signature
from src.cross_model.robustness_analysis import (report, summary, order_contrasts,
                                                  condition_order_values, paired_effects)
from src.cross_model.score_checks import checked_scores
from src.data.io import read_jsonl, sha256_file

ROOT = Path('outputs/cross_model_relational_v2/factorial_relation_counterbalanced_runtimefix_20261004')
OUT = ROOT / 'extended_analysis_v6'
FREEZE_SHA = 'f2981fa0f05a5e29e9478f8a94987e7578b2ce3667d1c941e8ce31cf8cdfef9a'


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def run_model(slug):
    d = ROOT / slug
    ds, ss = d/'confirmatory.jsonl', d/'confirmatory_scores.jsonl'
    dp, sp, old = (read_sealed(d/'confirmatory.jsonl.provenance.json'),
                   read_sealed(d/'confirmatory_scores.jsonl.provenance.json'),
                   read_sealed(d/'confirmatory_analysis.json'))
    if sha256_file(ROOT/'protocol_freeze.json') != FREEZE_SHA:
        raise ValueError('runtimefix freeze SHA-256 mismatch')
    if dp['freeze_sha256'] != FREEZE_SHA:
        raise ValueError('dataset freeze binding mismatch')
    rows, scores = read_jsonl(ds), read_jsonl(ss)
    if sha256_file(ds) != old['dataset_sha256'] or sha256_file(ds) != dp['provenance']['dataset_sha256']:
        raise ValueError('dataset hash binding mismatch')
    if sha256_file(ss) != old['scores_sha256'] or sha256_file(ss) != sp['scores_sha256']:
        raise ValueError('score hash binding mismatch')
    if old['dataset_sha256'] != dp['provenance']['dataset_sha256']:
        raise ValueError('analysis/dataset provenance mismatch')
    if dp['gate_sha256'] != sha256_file(d/'migrated/gate_report.json') or dp['preflight_sha256'] != sha256_file(d/'migrated/preflight.json'):
        raise ValueError('gate or preflight binding mismatch')
    checked_scores(rows, scores)
    if len(rows) != 18432 or len({r['history_id'] for r in rows}) != 96:
        raise ValueError('confirmation row/history count mismatch')
    fresh = report(rows, scores)
    if fresh['history_rows'] != old['history_rows']:
        raise ValueError('saved analysis differs from recomputed frozen estimates')
    actual, pairs, ranks = checked_scores(rows, scores)
    history = fresh['history_rows']
    competence = []
    member_groups = {}
    for r in rows:
        s = actual[r['example_id']]
        key = (r['condition'], r['query'], r['edited_variable'], r['historical_entity_order'],
               r['current_entity_order'], r['pair_direction'])
        member_groups.setdefault(key, []).append((r['history_id'], int(ranks[r['example_id']] == 1),
                                                   s['semantic_log_mass'][r['answer']] - s['semantic_log_mass'][r['stale_value']] if r['stale_value'] else None))
    for key, vals in sorted(member_groups.items()):
        margins = [x[2] for x in vals if x[2] is not None]
        competence.append({'condition':key[0], 'query':key[1], 'edited_variable':key[2],
                           'historical_order':key[3], 'current_order':key[4], 'pair_direction':key[5],
                           'n_rows':len(vals), 'candidate_accuracy':sum(x[1] for x in vals)/len(vals),
                           'mean_current_minus_stale':sum(margins)/len(margins) if margins else None})
    correctness = {}
    for pid, pair in pairs.items():
        rows_by_id = {r['example_id']: r for r in rows}
        baseline, edited = rows_by_id[pid+':0'], rows_by_id[pid+':1']
        correctness[pid] = {'history_id':baseline['history_id'], 'condition':baseline['condition'],
                            'baseline_correct':int(ranks[pid+':0']==1), 'edited_correct':int(ranks[pid+':1']==1)}
    fully_correct = {hid for hid in {r['history_id'] for r in rows}
                     if all(ranks[r['example_id']] == 1 for r in rows if r['history_id'] == hid)}
    conditioned = [r for r in history if r['history_id'] in fully_correct]
    out = {'model':slug, 'dataset_sha256':sha256_file(ds), 'scores_sha256':sha256_file(ss),
           'freeze_sha256':FREEZE_SHA, 'n_rows':len(rows), 'n_histories':96,
           'bootstrap_draws':2000, 'bootstrap_seed':73021,
           'recomputed_analysis':fresh, 'order_specific_contrasts':order_contrasts(history),
           'condition_order_history_values':condition_order_values(history),
           'competence_cells':competence,
           'correctness_pair_changes':correctness,
           'correctness_conditioned_history_counts':{c:sum(1 for x in correctness.values() if x['condition']==c and x['baseline_correct'] and x['edited_correct']) for c in CONDITIONS},
           'complete_history_correctness_conditioned_secondary':{'n_histories':len(conditioned), 'history_ids':sorted(fully_correct),
              'sequence_mass':__import__('src.cross_model.robustness_analysis',fromlist=['summarize_rows']).summarize_rows(conditioned) if conditioned else {'n_histories':0,'results':{}}},
           'surface_score_interpretation':'Saved bounded surface-class continuation event probabilities are available; these are prefix scores without termination and not unrestricted answer probabilities.'}
    outdir = OUT/slug; outdir.mkdir(parents=True, exist_ok=False)
    (outdir/'report.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    write_csv(outdir/'per_history.csv', history)
    write_csv(outdir/'condition_order_per_history.csv', condition_order_values(history))
    write_csv(outdir/'competence_cells.csv', competence)
    write_csv(outdir/'paired_correctness.csv', list(correctness.values()))
    write_csv(outdir/'paired_score_changes.csv', paired_effects(rows, scores))
    return out


def main():
    if OUT.exists(): raise FileExistsError(f'preserve existing analysis directory: {OUT}')
    freeze = read_sealed(ROOT/'protocol_freeze.json')
    if digest(freeze) != digest(read_sealed(ROOT/'protocol_freeze.json')):
        raise ValueError('freeze seal verification failed')
    ledger = json.loads(Path('configs/cross_model_relational_v2_geometryfix_exclusions/prior_history_exclusions.json').read_text())
    excluded = ledger['resolved_confirmation_exclusion']
    source = Path(excluded['source_dataset'])
    if sha256_file(source) != excluded['source_dataset_sha256']:
        raise ValueError('stopped-history exclusion source hash mismatch')
    stopped_rows = read_jsonl(source)
    stopped = {concrete_signature(r) for r in stopped_rows if r.get('matching_values')}
    if stopped != set(excluded['physical_signatures']) or len(stopped) != excluded['histories']:
        raise ValueError('stopped-history exclusion ledger mismatch')
    OUT.mkdir(parents=True)
    try:
        results = [run_model(x) for x in ('qwen3_8b','gemma3_4b')]
        confirm = [read_jsonl(ROOT/m/'confirmatory.jsonl') for m in ('qwen3_8b','gemma3_4b')]
        current = [{concrete_signature(r) for r in rs} for rs in confirm]
        overlap = [len(stopped & values) for values in current]
        if overlap != [0, 0] or current[0] != current[1]:
            raise ValueError('confirmation histories overlap stopped source or differ between models')
        (OUT/'audit.json').write_text(json.dumps({'freeze_sha256':FREEZE_SHA, 'models':[x['model'] for x in results], 'stopped_source_sha256':excluded['source_dataset_sha256'], 'stopped_signature_count':len(stopped), 'stopped_overlap_by_model':overlap, 'confirmation_signature_sets_match':True, 'complete':True}, indent=2)+'\n')
    except Exception:
        # Keep failed attempt for audit; no original artifact is touched.
        raise


if __name__ == '__main__': main()

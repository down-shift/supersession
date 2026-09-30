#!/usr/bin/env python3
"""Accuracy-only competence gate report; it deliberately never computes causal R."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.io import read_jsonl, sha256_file
from src.data.supersession_behavior import audit_status_2x2_gate_dataset
from src.utils import provenance, save_json

FORBIDDEN={'identity_transfer','matched_edit_effect','candidate_logits','candidate_probabilities',
           'source_logit_before','source_logit_after','replacement_logit_before','replacement_logit_after'}

def summarize_gate(dataset, scores, threshold=.99):
    audit_status_2x2_gate_dataset(dataset)
    expected={r['example_id']:r for r in dataset}
    if len(expected)!=len(dataset) or {r.get('example_id') for r in scores}!=set(expected):
        raise ValueError('gate scored records do not exactly cover dataset IDs')
    if len(scores)!=len(expected): raise ValueError('duplicate gate score IDs')
    groups={c:[] for c in ('YY','YN','NY','NN')}
    for row in scores:
        eid=row['example_id']
        if any(k in row for k in FORBIDDEN): raise ValueError('gate scores contain causal/effect data; this analysis accepts competence-only rows')
        if any(row.get(k)!=v for k,v in expected[eid].items()): raise ValueError(f'gate semantic metadata changed for {eid}')
        for key in ('prompt','full_vocab_next_token_accuracy','candidate_accuracy','target_rank','candidate_target_rank'):
            if key not in row: raise ValueError(f'gate score missing {key}')
        if row['full_vocab_next_token_accuracy'] not in (0,1) or row['candidate_accuracy'] not in (0,1):
            raise ValueError('gate accuracy fields must be binary')
        groups[row['condition']].append(row)
    result=[]; query_diagnostics=[]
    for condition,rows in groups.items():
        if len(rows)!=48 or len({r['history_id'] for r in rows})!=24 or {r['query'] for r in rows}!={'x','z'}:
            raise ValueError(f'{condition} gate cell must have 48 query examples from all 24 histories')
        result.append({'status_cell':condition,'n_histories':24,'n_scored_prompts':len(rows),
            'minimum_correct_for_threshold':int(np.ceil(threshold*len(rows))),
            'full_vocab_accuracy':float(np.mean([r['full_vocab_next_token_accuracy'] for r in rows])),
            'candidate_accuracy':float(np.mean([r['candidate_accuracy'] for r in rows])),
            'target_rank_mean':float(np.mean([r['target_rank'] for r in rows])),
            'target_rank_median':float(np.median([r['target_rank'] for r in rows])),
            'candidate_target_rank_mean':float(np.mean([r['candidate_target_rank'] for r in rows]))})
        strata={}
        for row in rows:
            literal=row['variables'][0 if row['query']=='x' else 1]
            position='first' if row['query']=='x' else 'second'
            strata.setdefault((row['query'],int(row['orientation']),literal,position),[]).append(row)
        for (query,orientation,literal,position),members in sorted(strata.items()):
            query_diagnostics.append({'status_cell':condition,'query_role':query,'literal_variable':literal,
                'block_position':position,'orientation':orientation,'n_histories':len(members),
                'full_vocab_accuracy':float(np.mean([r['full_vocab_next_token_accuracy'] for r in members])),
                'candidate_accuracy':float(np.mean([r['candidate_accuracy'] for r in members])),
                'target_rank_mean':float(np.mean([r['target_rank'] for r in members])),
                'target_rank_median':float(np.median([r['target_rank'] for r in members]))})
    failed=[r['status_cell'] for r in result if r['full_vocab_accuracy'] < threshold]
    return {'threshold':threshold,'failure_rule':'any status cell below threshold is a prompt-design failure',
            'gate_pass':not failed,'failure_cells':failed,'status_cells':result,
            'query_orientation_diagnostics':query_diagnostics}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--behavior',required=True)
    p.add_argument('--output-dir',required=True); p.add_argument('--threshold',type=float,default=.99)
    a=p.parse_args()
    if not 0<=a.threshold<=1: raise ValueError('--threshold must be in [0,1]')
    out=Path(a.output_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'{out} is nonempty; choose a fresh output directory')
    dataset,scores=read_jsonl(a.dataset),read_jsonl(a.behavior)
    run_manifest=Path(a.behavior+'.run.json')
    if run_manifest.exists() and json.loads(run_manifest.read_text()).get('dataset_sha256')!=sha256_file(a.dataset):
        raise ValueError('gate scoring manifest dataset hash differs from supplied gate dataset')
    summary=summarize_gate(dataset,scores,a.threshold)
    summary.update({'analysis':'status_2x2_competence_gate_only','causal_R_inspected':False,
                    'dataset_sha256':sha256_file(a.dataset),'scores_sha256':sha256_file(a.behavior),
                    'provenance':provenance({},a.dataset)})
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summary['status_cells']).to_csv(out/'status_cell_competence.csv',index=False)
    pd.DataFrame(summary['query_orientation_diagnostics']).to_csv(out/'status_query_orientation_diagnostics.csv',index=False)
    save_json(summary,out/'competence_gate.json')
    if summary['gate_pass']:
        print(f'competence gate PASSED: every status cell >= {a.threshold:.1%}')
    else:
        print(f"PROMPT-DESIGN FAILURE: status cells below {a.threshold:.1%}: {', '.join(summary['failure_cells'])}")
        raise SystemExit(2)


if __name__=='__main__': main()

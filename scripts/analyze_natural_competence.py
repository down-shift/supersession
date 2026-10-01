#!/usr/bin/env python3
"""Competence-only development or provenance-bound frozen gate; never computes R."""
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from src.data.io import read_jsonl, sha256_file
from src.utils import save_json

GATE = {'full_vocab_accuracy_min': .99, 'candidate_accuracy_min': .99,
        'mean_candidate_rank_max': 1.01, 'stale_margin_min': 0.0}

def history_signature(row):
    return json.dumps({'values':row['matching_values'], 'entities':row.get('entities', row.get('variables')),
                       'attribute':row.get('attribute'), 'orientation':row.get('orientation')}, sort_keys=True)

p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True); p.add_argument('--behavior', required=True)
p.add_argument('--stage', choices=('development', 'frozen_gate'), required=True)
p.add_argument('--output', required=True); p.add_argument('--token-ids', required=True)
p.add_argument('--config', required=True); p.add_argument('--selected-template')
p.add_argument('--prior-dataset', action='append', default=[])
a = p.parse_args()
dataset, scores = read_jsonl(a.dataset), read_jsonl(a.behavior)
expected = {r['example_id']:r for r in dataset}
actual = {r.get('example_id'):r for r in scores}
if len(actual) != len(scores) or set(actual) != set(expected):
    raise ValueError('scored example_id set does not exactly match dataset')
score_by_id = actual
for eid, row in expected.items():
    score = score_by_id[eid]
    if any(score.get(k) != row.get(k) for k in ('condition','history_id','prompt_variant','answer','pair_id','pair_direction')):
        raise ValueError(f'scored metadata mismatch at {eid}')

# A prompt may be repeated across independent edit pairs. Count it once, and
# fail if those duplicate measurements disagree.
unique = {}
for row in dataset:
    score = score_by_id[row['example_id']]
    signature = (row['prompt_variant'], score['prompt'])
    diagnostics = (score['full_vocab_next_token_accuracy'], score['accuracy'], score['candidate_rank'],
                  score['candidate_logits'].get(row['answer']), score['candidate_logits'].get(row.get('stale_value')) if row.get('stale_value') else None)
    if signature in unique and unique[signature][1] != diagnostics:
        raise ValueError('duplicate prompt has inconsistent scoring diagnostics')
    unique[signature] = (row, diagnostics)

cells=defaultdict(list)
for row, d in unique.values():
    if row['condition'] == 'irrelevant':
        continue  # uncounterbalanced diagnostic is never gate-blocking
    key=(row['prompt_variant'],row['condition'],row['query'],row['orientation'],row['edited_variable'],row['edit_status'],
         row['pair_direction'],row.get('unassigned_slot_order') or 'none')
    margin=(d[3]-d[4]) if d[4] is not None else None
    cells[key].append((d[0],d[1],d[2],margin))
required_conditions={'live','superseded','irrelevant_counterbalanced'}
if not cells or {k[1] for k in cells} != required_conditions:
    raise ValueError('gate dataset must contain live, superseded, and counterbalanced-irrelevant cells')
summary={ '|'.join(map(str,k)):{'n_unique_prompts':len(v), 'full_vocab_accuracy':float(np.mean([x[0] for x in v])),
         'candidate_accuracy':float(np.mean([x[1] for x in v])), 'mean_candidate_rank':float(np.mean([x[2] for x in v])),
         'mean_current_minus_stale_logit_margin':(float(np.mean([x[3] for x in v if x[3] is not None])) if any(x[3] is not None for x in v) else None)}
         for k,v in sorted(cells.items())}
passed=bool(summary) and all(v['full_vocab_accuracy']>=GATE['full_vocab_accuracy_min'] and
     v['candidate_accuracy']>=GATE['candidate_accuracy_min'] and v['mean_candidate_rank']<=GATE['mean_candidate_rank_max'] and
     (v['mean_current_minus_stale_logit_margin'] is None or v['mean_current_minus_stale_logit_margin']>GATE['stale_margin_min'])
     for v in summary.values())
templates=sorted({r['prompt_variant'] for r in dataset})
if a.stage=='frozen_gate' and (not a.selected_template or templates != [a.selected_template]):
    raise ValueError('frozen gate must use exactly --selected-template')
score_prov_path=Path(a.behavior+'.provenance.json')
if not score_prov_path.exists(): raise ValueError('scoring provenance sidecar is required')
score_prov=json.loads(score_prov_path.read_text())
if score_prov.get('dataset_sha256') != sha256_file(a.dataset): raise ValueError('scoring provenance dataset hash mismatch')
renderer_hash=hashlib.sha256(Path('src/data/supersession_behavior.py').read_bytes()).hexdigest()
history_sigs=sorted({history_signature(r) for r in dataset})
prior_hashes=[]
if a.stage=='frozen_gate':
    seen=set()
    for path in a.prior_dataset:
        prior_hashes.append(sha256_file(path))
        current={history_signature(row) for row in read_jsonl(path)}
        if seen & current or current & set(history_sigs): raise ValueError('frozen gate overlaps an earlier-stage concrete history')
        seen |= current
    if not a.prior_dataset: raise ValueError('frozen gate requires development --prior-dataset files')

artifact={'stage':'frozen_competence_gate' if a.stage=='frozen_gate' else 'development_competence_only',
 'pass':passed if a.stage=='frozen_gate' else None,'preregistered_gate':GATE,'causal_metrics_inspected':False,
 'n_unique_prompts':len(unique),'selected_templates':templates,'selected_template':a.selected_template,
 'by_template_condition_query_orientation_edit_status_pair_direction_slot':summary,
 'history_signatures':history_sigs,'dataset_path':str(Path(a.dataset).resolve()),'behavior_path':str(Path(a.behavior).resolve()),
 'dataset_sha256':sha256_file(a.dataset),'behavior_sha256':sha256_file(a.behavior),
 'scoring_provenance_sha256':sha256_file(score_prov_path),'token_map_sha256':sha256_file(a.token_ids),
 'config_sha256':sha256_file(a.config),'renderer_sha256':renderer_hash,
 'model_tokenizer':{k:score_prov.get(k) for k in ('model_revision','tokenizer_revision','model_id','tokenizer_id')},
 'seed':score_prov.get('seed'),'prior_dataset_sha256':prior_hashes,
 'prior_dataset_paths':[str(Path(x).resolve()) for x in a.prior_dataset]}
if a.stage=='frozen_gate':
    artifact['artifact_sha256']=hashlib.sha256(json.dumps(artifact,sort_keys=True,separators=(',',':')).encode()).hexdigest()
save_json(artifact,a.output)
print(json.dumps({'stage':artifact['stage'],'pass':artifact['pass'],'cells':len(summary),'unique_prompts':len(unique)},sort_keys=True))

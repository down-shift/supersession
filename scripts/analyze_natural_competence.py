#!/usr/bin/env python3
"""Competence-only development or provenance-bound frozen gate; never computes R."""
import argparse, hashlib, json
from pathlib import Path
from src.data.io import read_jsonl, sha256_file
from src.utils import save_json

from src.analysis.natural_competence import GATE, GATE_VERSION, concrete_history_signature, evaluate_competence

p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True); p.add_argument('--behavior', required=True)
p.add_argument('--stage', choices=('development', 'frozen_gate'), required=True)
p.add_argument('--output', required=True); p.add_argument('--token-ids', required=True)
p.add_argument('--config', required=True); p.add_argument('--selected-template')
p.add_argument('--prior-dataset', action='append', default=[])
a = p.parse_args()
if Path(a.output).exists(): raise FileExistsError('preserve the existing analysis; choose a fresh output path')
dataset, scores = read_jsonl(a.dataset), read_jsonl(a.behavior)
evaluation = evaluate_competence(dataset, scores)
summary = evaluation['summary']
passed = evaluation['pass']
templates = evaluation['templates']
dataset_seeds = evaluation['dataset_seed']
if a.stage=='frozen_gate' and (not a.selected_template or templates != [a.selected_template]):
    raise ValueError('frozen gate must use exactly --selected-template')
score_prov_path=Path(a.behavior+'.provenance.json')
if not score_prov_path.exists(): raise ValueError('scoring provenance sidecar is required')
score_prov=json.loads(score_prov_path.read_text())
if score_prov.get('dataset_sha256') != sha256_file(a.dataset): raise ValueError('scoring provenance dataset hash mismatch')
if score_prov.get('dataset_seed') != dataset_seeds: raise ValueError('scoring provenance dataset seeds mismatch')
renderer_hash=hashlib.sha256(Path('src/data/supersession_behavior.py').read_bytes()).hexdigest()
history_sigs=sorted({concrete_history_signature(r) for r in dataset})
prior_hashes=[]
if a.stage=='frozen_gate':
    seen=set()
    for path in a.prior_dataset:
        prior_hashes.append(sha256_file(path))
        current={concrete_history_signature(row) for row in read_jsonl(path)}
        if seen & current or current & set(history_sigs): raise ValueError('frozen gate overlaps an earlier-stage concrete history')
        seen |= current
    if not a.prior_dataset: raise ValueError('frozen gate requires development --prior-dataset files')

artifact={'stage':'frozen_competence_gate' if a.stage=='frozen_gate' else 'development_competence_only',
 'pass':passed if a.stage=='frozen_gate' else None,'preregistered_gate':GATE,'causal_metrics_inspected':False,
 'gate_version':GATE_VERSION, 'expected_cell_count':evaluation['expected_cell_count'],
 'n_unique_prompts':evaluation['n_unique_prompts'],'selected_templates':templates,'selected_template':a.selected_template,
 'by_template_condition_query_orientation_edit_status_pair_direction_slot':summary,
 'history_signatures':history_sigs,'dataset_path':str(Path(a.dataset).resolve()),'behavior_path':str(Path(a.behavior).resolve()),
 'dataset_sha256':sha256_file(a.dataset),'behavior_sha256':sha256_file(a.behavior),
 'scoring_provenance_sha256':sha256_file(score_prov_path),'token_map_sha256':sha256_file(a.token_ids),
 'config_sha256':sha256_file(a.config),'renderer_sha256':renderer_hash,
 'competence_code_sha256':sha256_file('src/analysis/natural_competence.py'),
 'model_tokenizer':{k:score_prov.get(k) for k in ('model_revision','tokenizer_revision','model_id','tokenizer_id')},
 'seed':dataset_seeds[0] if len(dataset_seeds)==1 else None, 'dataset_seed':dataset_seeds,
 'scoring_config_seed':score_prov.get('seed'),'prior_dataset_sha256':prior_hashes,
 'prior_dataset_paths':[str(Path(x).resolve()) for x in a.prior_dataset]}
if a.stage=='frozen_gate':
    artifact['artifact_sha256']=hashlib.sha256(json.dumps(artifact,sort_keys=True,separators=(',',':')).encode()).hexdigest()
save_json(artifact,a.output)
print(json.dumps({'stage':artifact['stage'],'pass':artifact['pass'],'cells':len(summary),'unique_prompts':evaluation['n_unique_prompts']},sort_keys=True))

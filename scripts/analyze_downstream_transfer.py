#!/usr/bin/env python3
"""Analyze confirmatory downstream derived-code transfer."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.io import read_jsonl,sha256_file
from src.data.downstream_transfer import verify_sealed_artifact,template_hash
from src.analysis.downstream_transfer import history_contrasts,summarize_histories,current_answer_stability
from src.utils import save_json
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scores',required=True);p.add_argument('--dataset',required=True);p.add_argument('--gate',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--seed',type=int,default=20261033);p.add_argument('--bootstrap-draws',type=int,default=10000);a=p.parse_args()
out=Path(a.output_dir)
if out.exists() and any(out.iterdir()): raise FileExistsError('choose an empty output directory')
rows=read_jsonl(a.dataset);scores=read_jsonl(a.scores);gate=verify_sealed_artifact(json.loads(Path(a.gate).read_text()));side=json.loads(Path(a.scores+'.provenance.json').read_text())
dsprov=json.loads(Path(a.dataset+'.provenance.json').read_text())
if gate.get('pass') is not True or gate.get('template_sha256')!=template_hash(): raise ValueError('invalid frozen gate')
if side.get('stage')!='confirmatory' or side.get('dataset_sha256')!=sha256_file(a.dataset) or side.get('gate_sha256')!=sha256_file(a.gate): raise ValueError('confirmatory score lineage mismatch')
if gate.get('dataset_sha256')!=sha256_file(gate.get('dataset_path','')) or gate.get('scores_sha256')!=sha256_file(gate.get('scores_path','')): raise ValueError('frozen gate dataset/scores changed')
if side.get('scoring_code_sha256')!=gate.get('scoring_code_sha256') or side.get('scoring_version')!=gate.get('scoring_version'): raise ValueError('confirmatory scoring differs from frozen gate')
if side.get('token_audit_sha256')!=gate.get('token_audit_sha256'): raise ValueError('confirmatory tokenizer audit differs from frozen gate')
for key in ('config_sha256','values_sha256','codes_sha256','model_revision','tokenizer_revision'):
 if side.get(key)!=gate.get(key) or dsprov.get(key)!=gate.get(key): raise ValueError(f'confirmatory frozen artifact mismatch: {key}')
if dsprov.get('gate_sha256')!=sha256_file(a.gate) or dsprov.get('token_audit_sha256')!=gate.get('token_audit_sha256'): raise ValueError('confirmatory dataset gate/tokenizer lineage mismatch')
if {r['example_id'] for r in rows}!={r['example_id'] for r in scores}: raise ValueError('scores do not exactly cover confirmatory dataset')
hist=history_contrasts(scores);stability=current_answer_stability(rows,scores);out.mkdir(parents=True,exist_ok=True)
stability_by_history={}
for r in stability:
 bucket=stability_by_history.setdefault(r['history_id'],{})
 for k in ('baseline_accuracy','edited_accuracy','baseline_correct_logprob','edited_correct_logprob','baseline_margin','edited_margin','delta_correct_logprob','delta_margin'):
  bucket.setdefault(k,[]).append(r[k])
for bucket in stability_by_history.values():
 for k,values in bucket.items(): bucket[k]=sum(values)/len(values)
summary={'protocol':'downstream_transfer_v1','primary':'R_stale_derived','estimands':{k:summarize_histories([r[k] for r in hist],a.seed,a.bootstrap_draws) for k in ('R_stale_derived','R_live_derived')},
 'current_answer_stability':{k:summarize_histories([r[k] for r in stability_by_history.values()],a.seed,a.bootstrap_draws) for k in ('baseline_accuracy','edited_accuracy','baseline_correct_logprob','edited_correct_logprob','baseline_margin','edited_margin','delta_correct_logprob','delta_margin')},
 'n_histories':len(hist),'bootstrap_unit':'history_id','scores_sha256':sha256_file(a.scores),'dataset_sha256':sha256_file(a.dataset)}
save_json(summary,out/'downstream_transfer_summary.json');save_json(hist,out/'downstream_transfer_per_history.json');save_json(stability,out/'downstream_transfer_answer_stability.json')
print(f'wrote analysis for {len(hist)} histories')

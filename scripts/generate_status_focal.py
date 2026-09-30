#!/usr/bin/env python3
"""Generate focal validity prompt development, frozen competence, or confirmatory prompts."""
import argparse,json
from pathlib import Path
from src.data.status_focal import (VARIANTS,generate,audit,verify_competence_artifact,
    focal_history_signatures,focal_template_hash)
from src.data.io import write_jsonl,read_jsonl,sha256_file
from src.data.status_prompt_gate import history_signatures
from src.utils import load_config

LEGACY=('outputs/supersession/status.jsonl','outputs/supersession/status_2x2.jsonl',
        'outputs/supersession/status_2x2_competence_gate.jsonl')

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=('development','frozen_gate','confirmatory'),required=True);p.add_argument('--config',default='configs/four_query_288.yaml');p.add_argument('--values',required=True);p.add_argument('--token-ids',required=True);p.add_argument('--output',required=True);p.add_argument('--n',type=int);p.add_argument('--seed',type=int,required=True);p.add_argument('--variant',choices=VARIANTS,action='append');p.add_argument('--passing-gate');p.add_argument('--selection');p.add_argument('--exclude-dataset',action='append',default=[])
 a=p.parse_args();out=Path(a.output)
 if out.exists() or Path(str(out)+'.provenance.json').exists():raise FileExistsError(f'{out} exists; choose a fresh output')
 variants=tuple(a.variant or VARIANTS); cfg=load_config(a.config); selected_artifacts=[]
 if a.stage=='confirmatory':
  if not a.passing_gate:raise ValueError('confirmatory generation requires an explicitly supplied passing frozen-gate artifact')
  gate,gate_rows=verify_competence_artifact(a.passing_gate,'frozen_gate',a.config,a.token_ids)
  variants=(gate['selected_variant'],); selected_artifacts.append((a.passing_gate,gate,gate_rows))
  generation=json.loads(Path(gate['dataset_path']+'.provenance.json').read_text()); selection_path=generation.get('selection_path')
  if not selection_path:raise ValueError('frozen-gate generation provenance lacks its development selection')
  selection,dev_rows=verify_competence_artifact(selection_path,'development',a.config,a.token_ids)
  selected_artifacts.append((selection_path,selection,dev_rows))
  if a.seed==gate.get('seed'):raise ValueError('confirmatory generation requires a new seed')
  if a.seed==selection.get('seed'):raise ValueError('confirmatory generation must use a fresh seed')
 elif a.stage=='frozen_gate':
  if not a.selection:raise ValueError('frozen_gate requires a development selection artifact')
  selection,dev_rows=verify_competence_artifact(a.selection,'development',a.config,a.token_ids)
  selected_artifacts.append((a.selection,selection,dev_rows))
  if a.seed==selection.get('seed'):raise ValueError('frozen competence gate requires a new seed')
  variants=(selection['selected_variant'],)
 n=a.n if a.n is not None else (96 if a.stage=='confirmatory' else 24)
 proposals=json.loads(Path(a.values).read_text());token_doc=json.loads(Path(a.token_ids).read_text());values=[v for v in proposals if v in token_doc['token_ids']]
 if len(values)<5:raise ValueError('fewer than five validated one-token candidates remain')
 if selected_artifacts and list(values)!=selected_artifacts[0][1].get('candidate_values',list(values)):raise ValueError('candidate values differ from selected/gated artifact')
 prior_focal=[str(p) for p in Path('outputs/supersession').glob('status_focal*.jsonl')] if a.stage=='development' else []
 excluded_paths=list(LEGACY if a.stage=='development' else ()) + prior_focal + a.exclude_dataset
 excluded_rows=[r for path in sorted(set(excluded_paths)) if Path(path).exists() for r in read_jsonl(path)]
 for _,_,old_rows in selected_artifacts:excluded_rows.extend(old_rows)
 legacy_compatible=[r for r in excluded_rows if set(('initial_x','initial_z','proposed_x','proposed_z'))<=set(r.get('matching_values',{}))]
 signatures=history_signatures(legacy_compatible)|focal_history_signatures(excluded_rows)
 rows=generate(a.stage,n,values,a.seed,variants,signatures)
 write_jsonl(rows,out)
 sidecar={'experiment_kind':'status_focal','stage':a.stage,'seed':a.seed,'n_histories':n,'variants':variants,'dataset_sha256':sha256_file(out),'token_map_sha256':sha256_file(a.token_ids),'template_sha256':focal_template_hash(),'candidate_values':values,'excluded_datasets':[{'path':p,'sha256':sha256_file(p)} for p in excluded_paths if Path(p).exists()],'excluded_history_count':len(signatures),'selection_path':str(Path(a.selection).resolve()) if a.selection else (str(Path(selection_path).resolve()) if a.stage=='confirmatory' else None),'frozen_gate_path':str(Path(a.passing_gate).resolve()) if a.passing_gate else None,'frozen_gate_sha256':sha256_file(a.passing_gate) if a.passing_gate else None,'config_sha256':sha256_file(a.config)}
 Path(str(out)+'.provenance.json').write_text(json.dumps(sidecar,indent=2)+'\n')
 print(f'wrote {len(rows)} records to {out}')
if __name__=='__main__':main()

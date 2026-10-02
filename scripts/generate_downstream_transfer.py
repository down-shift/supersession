#!/usr/bin/env python3
"""Create isolated downstream_transfer_v1 stages without overwriting artifacts."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.io import read_jsonl,write_jsonl,sha256_file
from src.data.downstream_transfer import generate,template_hash,verify_sealed_artifact
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('development','frozen_gate','confirmatory'),required=True);p.add_argument('--output',required=True);p.add_argument('--prior-dataset',action='append',default=[]);p.add_argument('--gate');p.add_argument('--token-audit');p.add_argument('--config',default='configs/downstream_transfer_v1.yaml');p.add_argument('--values-json',default='configs/downstream_transfer_values.json');p.add_argument('--codes-json',default='configs/downstream_transfer_codes.json')
a=p.parse_args();out=Path(a.output)
if any(Path(str(out)+s).exists() for s in ('','.provenance.json')): raise FileExistsError('preserve prior outputs; choose a new path')
values=json.loads(Path(a.values_json).read_text());codes=json.loads(Path(a.codes_json).read_text());prior=[read_jsonl(x) for x in a.prior_dataset];excluded=[r for ds in prior for r in ds];gate=None
if a.stage=='confirmatory':
 if not a.gate: raise ValueError('confirmatory generation requires --gate')
 gate=verify_sealed_artifact(json.loads(Path(a.gate).read_text()))
 if gate.get('pass') is not True: raise ValueError('frozen gate failed')
 for key,path in [('config_sha256',a.config),('values_sha256',a.values_json),('codes_sha256',a.codes_json)]:
  if gate.get(key)!=sha256_file(path): raise ValueError(f'frozen {key} changed')
 if gate.get('template_sha256')!=template_hash(): raise ValueError('frozen template changed')
 if gate.get('dataset_sha256')!=sha256_file(gate.get('dataset_path','')) or gate.get('scores_sha256')!=sha256_file(gate.get('scores_path','')): raise ValueError('frozen gate dataset/scores changed')
 if not a.token_audit or gate.get('token_audit_sha256')!=sha256_file(a.token_audit): raise ValueError('confirmatory generation requires the frozen tokenizer audit')
 excluded+=read_jsonl(gate['dataset_path'])
rows=generate(a.stage,{'development':24,'frozen_gate':24,'confirmatory':96}[a.stage],values,codes,excluded=excluded);write_jsonl(rows,out)
prov={'protocol':'downstream_transfer_v1','stage':a.stage,'dataset_path':str(out.resolve()),'dataset_sha256':sha256_file(out),'template_sha256':template_hash(),'config_sha256':sha256_file(a.config),'values_sha256':sha256_file(a.values_json),'codes_sha256':sha256_file(a.codes_json),'token_audit_sha256':sha256_file(a.token_audit) if a.token_audit else None,'prior_dataset_paths':[str(Path(x).resolve()) for x in a.prior_dataset],'prior_dataset_sha256':[sha256_file(x) for x in a.prior_dataset],'gate_path':str(Path(a.gate).resolve()) if a.gate else None,'gate_sha256':sha256_file(a.gate) if a.gate else None}
Path(str(out)+'.provenance.json').write_text(json.dumps(prov,indent=2)+'\n');print(f'wrote {len(rows)} {a.stage} records')

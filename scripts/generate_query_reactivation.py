#!/usr/bin/env python3
"""Generate isolated query_reactivation_v1 stages; never overwrites artifacts."""
import argparse, json
from pathlib import Path
from src.data.io import read_jsonl, write_jsonl, sha256_file
from src.data.query_reactivation import VALUES, generate, template_hash, verify_sealed_artifact, verify_gate_values

p=argparse.ArgumentParser(description=__doc__); p.add_argument('--stage',choices=('development','frozen_gate','confirmatory'),required=True)
p.add_argument('--output',required=True); p.add_argument('--n',type=int)
p.add_argument('--values-json',default='configs/query_reactivation_values.json'); p.add_argument('--prior-dataset',action='append',default=[])
p.add_argument('--gate')
a=p.parse_args(); out=Path(a.output)
if out.exists() or Path(str(out)+'.provenance.json').exists(): raise FileExistsError('preserve existing artifacts; choose a new path')
values=json.loads(Path(a.values_json).read_text())
if values != list(VALUES): raise ValueError('values file differs from the single frozen query_reactivation_v1 vocabulary')
prior=[read_jsonl(x) for x in a.prior_dataset]
excluded=[r for rows in prior for r in rows]
gate=None
if a.stage=='confirmatory':
 if not a.gate: raise ValueError('confirmatory generation requires a frozen gate artifact')
 gate=verify_sealed_artifact(json.loads(Path(a.gate).read_text()))
 if gate.get('stage')!='frozen_gate' or gate.get('pass') is not True: raise ValueError('invalid/failed frozen gate')
 verify_gate_values(values,gate)
 excluded += read_jsonl(gate['dataset_path'])
 if gate['template_sha256']!=template_hash() or gate['dataset_sha256']!=sha256_file(gate['dataset_path']): raise ValueError('gate provenance changed')
rows=generate(a.stage,a.n or {'development':24,'frozen_gate':24,'confirmatory':96}[a.stage],values,excluded=excluded)
write_jsonl(rows,out)
prov={'protocol':'query_reactivation_v1','stage':a.stage,'seed':rows[0]['seed'],'dataset_path':str(out.resolve()),
 'dataset_sha256':sha256_file(out),'template_sha256':template_hash(),'values_sha256':sha256_file(a.values_json),
 'prior_dataset_paths':[str(Path(x).resolve()) for x in a.prior_dataset],
 'prior_dataset_sha256':[sha256_file(x) for x in a.prior_dataset],
 'gate_path':str(Path(a.gate).resolve()) if a.gate else None,
 'gate_sha256':sha256_file(a.gate) if a.gate else None}
Path(str(out)+'.provenance.json').open('x').write(json.dumps(prov,indent=2)+'\n')
print(f'wrote {len(rows)} {a.stage} records')

#!/usr/bin/env python3
"""Generate focal validity prompt development, frozen competence, or confirmatory prompts."""
import argparse,json
from pathlib import Path
from src.data.status_focal import VARIANTS,generate,audit,require_passing_frozen_gate
from src.data.io import write_jsonl

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=('development','frozen_gate','confirmatory'),required=True);p.add_argument('--values',required=True);p.add_argument('--token-ids',required=True);p.add_argument('--output',required=True);p.add_argument('--n',type=int);p.add_argument('--seed',type=int,required=True);p.add_argument('--variant',choices=VARIANTS,action='append');p.add_argument('--passing-gate');p.add_argument('--selection')
 a=p.parse_args();out=Path(a.output)
 if out.exists() or Path(str(out)+'.provenance.json').exists():raise FileExistsError(f'{out} exists; choose a fresh output')
 variants=tuple(a.variant or VARIANTS)
 if a.stage=='confirmatory':
  if not a.passing_gate:raise ValueError('confirmatory generation requires an explicitly supplied passing frozen-gate artifact')
  gate=json.loads(Path(a.passing_gate).read_text())
  variants=(require_passing_frozen_gate(gate),)
  if a.seed==gate.get('seed'):raise ValueError('confirmatory generation requires a new seed')
 elif a.stage=='frozen_gate':
  if not a.selection:raise ValueError('frozen_gate requires a development selection artifact')
  selection=json.loads(Path(a.selection).read_text())
  if selection.get('stage')!='development' or selection.get('selected_variant') not in VARIANTS:raise ValueError('invalid development selection artifact')
  if a.seed==selection.get('seed'):raise ValueError('frozen competence gate requires a new seed')
  variants=(selection['selected_variant'],)
 n=a.n if a.n is not None else (96 if a.stage=='confirmatory' else 24)
 proposals=json.loads(Path(a.values).read_text());token_doc=json.loads(Path(a.token_ids).read_text());values=[v for v in proposals if v in token_doc['token_ids']]
 if len(values)<5:raise ValueError('fewer than five validated one-token candidates remain')
 rows=generate(a.stage,n,values,a.seed,variants)
 write_jsonl(rows,out);Path(str(out)+'.provenance.json').write_text(json.dumps({'experiment_kind':'status_focal','stage':a.stage,'seed':a.seed,'n_histories':n,'variants':variants,'passing_gate':str(Path(a.passing_gate).resolve()) if a.passing_gate else None},indent=2)+'\n')
 print(f'wrote {len(rows)} records to {out}')
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Recompute competence and freeze prompt selection or frozen-gate artifact."""
import argparse,json
from pathlib import Path
from src.data.io import read_jsonl,sha256_file
from src.data.status_focal import (audit,gate_summary,focal_template_hash,
    development_variant_diagnostic,DEVELOPMENT_MIN_ACCURACY)

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--scores',required=True);p.add_argument('--output',required=True);p.add_argument('--select',action='store_true');a=p.parse_args()
 ds=read_jsonl(a.dataset);ss=read_jsonl(a.scores);stage=audit(ds);summary=gate_summary(ds,ss)
 out=Path(a.output)
 if out.exists():raise FileExistsError(f'{out} exists; preserve artifacts and choose a fresh path')
 prov_path=Path(a.scores+'.provenance.json')
 if not prov_path.is_file():raise ValueError('scoring provenance is required to freeze selection/gate')
 prov=json.loads(prov_path.read_text())
 if prov.get('dataset_sha256')!=sha256_file(a.dataset) or prov.get('template_sha256')!=focal_template_hash():raise ValueError('scoring provenance does not match dataset/template')
 variants={r['prompt_variant'] for r in ds}
 if stage=='development' and a.select:
  diagnostics={v:development_variant_diagnostic(ds,ss,v) for v in sorted(variants)}
  eligible={v for v,d in diagnostics.items() if d['passed']}
  if not eligible:
   print(json.dumps({'stage':'development','passed':False,'selected_variant':None,
    'selection_written':False,'development_min_accuracy':DEVELOPMENT_MIN_ACCURACY,
    'variant_diagnostics':diagnostics},indent=2))
   return False
  def key(v):
   cs=[c for c in summary['cells'] if c['cell'][0]==v]
   return (min(c['full_vocab_accuracy'] for c in cs),-max(c['target_rank_max'] for c in cs),min(c['candidate_accuracy'] for c in cs),-max(c['candidate_rank_max'] for c in cs))
  selected=max(sorted(eligible),key=key)
 elif stage=='frozen_gate':selected=ds[0]['prompt_variant']
 else:raise ValueError('development requires --select; frozen gate must contain one variant')
 if stage=='frozen_gate' and not summary['passed']:raise ValueError('frozen gate failed; no passing artifact written')
 development_qualification=development_variant_diagnostic(ds,ss,selected) if stage=='development' else None
 doc={'stage':stage,'passed':development_qualification['passed'] if stage=='development' else summary['passed'],'causal_effects_computed':False,'competence_summary':summary,
  'selected_variant':selected,'candidate_values':ds[0]['candidate_values'],'seed':ds[0]['seed'],
  'dataset_path':str(Path(a.dataset).resolve()),'dataset_sha256':sha256_file(a.dataset),
  'scores_path':str(Path(a.scores).resolve()),'scores_sha256':sha256_file(a.scores),
  'scores_provenance_path':str(prov_path.resolve()),'scores_provenance_sha256':sha256_file(prov_path),
  'scoring_provenance':prov,'template_sha256':focal_template_hash(),
  'alignment_eligible_variants':sorted(v for v in variants if all(r['validity_alignment_passed'] for r in ss if r['prompt_variant']==v)),
  'variant_metrics_used':['full_vocab_next_token_accuracy','target_rank','candidate_accuracy','candidate_target_rank'],
  'development_qualification':development_qualification,
  'development_min_accuracy':DEVELOPMENT_MIN_ACCURACY if stage=='development' else None,
  'selection_rule':'all cells >= 0.98 accuracy and rank 1; competence metrics only' if stage=='development' else 'all focal competence cells must be perfect',
  'selection_path':json.loads(Path(a.dataset+'.provenance.json').read_text()).get('selection_path') if stage=='frozen_gate' and Path(a.dataset+'.provenance.json').exists() else None}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(doc,indent=2)+'\n')
 print(json.dumps({'stage':stage,'selected_variant':selected,'passed':doc['passed'],'complete_cells':summary['complete_cells']}))
if __name__=='__main__':main()

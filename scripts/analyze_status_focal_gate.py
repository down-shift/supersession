#!/usr/bin/env python3
"""Recompute competence and freeze prompt selection or frozen-gate artifact."""
import argparse,json
from pathlib import Path
from src.data.io import read_jsonl,sha256_file
from src.data.status_focal import audit,gate_summary,focal_template_hash

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--scores',required=True);p.add_argument('--output',required=True);p.add_argument('--select',action='store_true');a=p.parse_args()
 ds=read_jsonl(a.dataset);ss=read_jsonl(a.scores);stage=audit(ds);summary=gate_summary(ds,ss)
 prov_path=Path(a.scores+'.provenance.json')
 if not prov_path.is_file():raise ValueError('scoring provenance is required to freeze selection/gate')
 prov=json.loads(prov_path.read_text())
 if prov.get('dataset_sha256')!=sha256_file(a.dataset) or prov.get('template_sha256')!=focal_template_hash():raise ValueError('scoring provenance does not match dataset/template')
 variants={r['prompt_variant'] for r in ds}
 if stage=='development' and a.select:
  eligible={v for v in variants if all(r['validity_alignment_passed'] for r in ss if r['prompt_variant']==v)}
  if not eligible:raise ValueError('no prompt variant passed the focal status token-alignment audit')
  def key(v):
   cs=[c for c in summary['cells'] if c['cell'][0]==v]
   return (min(c['full_vocab_accuracy'] for c in cs),-max(c['target_rank_max'] for c in cs),min(c['candidate_accuracy'] for c in cs),-max(c['candidate_rank_max'] for c in cs))
  selected=max(sorted(eligible),key=key)
 elif stage=='frozen_gate':selected=ds[0]['prompt_variant']
 else:raise ValueError('development requires --select; frozen gate must contain one variant')
 if stage=='frozen_gate' and not summary['passed']:raise ValueError('frozen gate failed; no passing artifact written')
 doc={'stage':stage,'passed':summary['passed'],'causal_effects_computed':False,'competence_summary':summary,
  'selected_variant':selected,'candidate_values':ds[0]['candidate_values'],'seed':ds[0]['seed'],
  'dataset_path':str(Path(a.dataset).resolve()),'dataset_sha256':sha256_file(a.dataset),
  'scores_path':str(Path(a.scores).resolve()),'scores_sha256':sha256_file(a.scores),
  'scores_provenance_path':str(prov_path.resolve()),'scores_provenance_sha256':sha256_file(prov_path),
  'scoring_provenance':prov,'template_sha256':focal_template_hash(),
  'alignment_eligible_variants':sorted(v for v in variants if all(r['validity_alignment_passed'] for r in ss if r['prompt_variant']==v)),
  'variant_metrics_used':['full_vocab_next_token_accuracy','target_rank','candidate_accuracy','candidate_target_rank'],
  'selection_rule':'lexicographic competence metrics only' if stage=='development' else 'all focal competence cells must be perfect',
  'selection_path':json.loads(Path(a.dataset+'.provenance.json').read_text()).get('selection_path') if stage=='frozen_gate' and Path(a.dataset+'.provenance.json').exists() else None}
 out=Path(a.output)
 if out.exists():raise FileExistsError(f'{out} exists; preserve artifacts and choose a fresh path')
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(doc,indent=2)+'\n')
 print(json.dumps({'stage':stage,'selected_variant':selected,'passed':summary['passed'],'complete_cells':summary['complete_cells']}))
if __name__=='__main__':main()

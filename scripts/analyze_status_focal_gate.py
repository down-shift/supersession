#!/usr/bin/env python3
"""Competence-only summaries and immutable gate artifacts for status_focal."""
import argparse,json
from pathlib import Path
from src.data.io import read_jsonl
from src.data.status_focal import audit,gate_summary,VARIANTS

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--scores',required=True);p.add_argument('--output',required=True);p.add_argument('--select',action='store_true',help='development only: choose best wording by competence metrics');a=p.parse_args()
 ds=read_jsonl(a.dataset);ss=read_jsonl(a.scores);stage=audit(ds);summary=gate_summary(ds,ss)
 if stage=='development' and a.select:
  variants={r['prompt_variant'] for r in ds}
  # Ranking is fixed before looking: accuracy first, then target rank, then candidate accuracy/rank.
  def key(v):
   rr=[x for x in summary['cells'] if x['cell'][0]==v]
   return (min(x['full_vocab_accuracy'] for x in rr),-max(x['target_rank_max'] for x in rr),min(x['candidate_accuracy'] for x in rr),-max(x['candidate_rank_max'] for x in rr))
  selected=max(sorted(variants),key=key)
 elif stage=='frozen_gate': selected=ds[0]['prompt_variant']
 else: raise ValueError('--select is required for development; frozen gate artifact is the pass artifact')
 doc={**summary,'seed':ds[0]['seed'],'selected_variant':selected,'dataset_path':str(Path(a.dataset).resolve()),'scores_path':str(Path(a.scores).resolve()),'variant_metrics_used':['full_vocab_accuracy','target_rank_max','candidate_accuracy','candidate_rank_max'],'selection_rule':'lexicographic competence only' if stage=='development' else 'single frozen wording','prompt_variant_count':len({r['prompt_variant'] for r in ds})}
 Path(a.output).write_text(json.dumps(doc,indent=2)+'\n')
 print(json.dumps({'stage':stage,'selected_variant':selected,'passed':summary['passed'],'complete_cells':summary['complete_cells']}))
if __name__=='__main__':main()

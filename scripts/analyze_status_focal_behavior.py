#!/usr/bin/env python3
"""History-paired focal validity effects from matched edit records."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from src.analysis.metrics import bootstrap_mean_ci
from src.data.io import read_jsonl,sha256_file

def analyze(rows,seed=20261010,n_boot=10000):
 f=pd.DataFrame(rows);need={'history_id','focal_variable','focal_valid','focal_position','query_role','edited_field','identity_transfer_E','baseline_competence','edited_competence','baseline_candidate_logits','edited_candidate_logits'}
 if not need<=set(f):raise ValueError(f'missing behavioral fields: {sorted(need-set(f))}')
 if f['history_id'].nunique()!=96 or len(f)!=96*2*2*2*2:raise ValueError('confirmatory behavioral file must contain 96 complete matched histories')
 if f.duplicated(['history_id','focal_valid','focal_position','query_role','edited_field']).any():raise ValueError('duplicate behavioral cell')
 for hid,g in f.groupby('history_id'):
  focal=g['focal_variable'].unique()
  if len(focal)!=1:raise ValueError(f'{hid}: focal variable changes within history')
  expected={(valid,pos,query,field) for valid in (True,False) for pos in ('focal_first','focal_second') for query in ('focal','other') for field in (f'proposed_{focal[0]}',f'initial_{focal[0]}')}
  actual=set(zip(g['focal_valid'],g['focal_position'],g['query_role'],g['edited_field']))
  if actual!=expected:raise ValueError(f'{hid}: incomplete focal validity/edit/query/order cells')
 idx=['history_id','focal_variable','focal_valid','focal_position','edited_field']
 e=f.pivot(index=idx,columns='query_role',values='identity_transfer_E').reset_index()
 if not {'focal','other'}<=set(e):raise ValueError('each cell requires focal and other query roles')
 e['R']=e['focal']-e['other']
 # Competence remains descriptive output, never used to choose prompt wording here.
 c=f.groupby(idx,as_index=False).agg(full_vocab_accuracy=('baseline_competence',lambda x:np.mean([v['full_vocab_next_token_accuracy'] for v in x])),
   candidate_accuracy=('baseline_competence',lambda x:np.mean([v['candidate_accuracy'] for v in x])),
   target_rank_mean=('baseline_competence',lambda x:np.mean([v['target_rank'] for v in x])),
   candidate_rank_mean=('baseline_competence',lambda x:np.mean([v['candidate_target_rank'] for v in x])))
 e=e.merge(c,on=idx,validate='one_to_one')
 # Pair each validity condition, then collapse position counterbalance within history.
 contrasts=[]
 for (hid,focal,field),g in e.groupby(['history_id','focal_variable','edited_field']):
  wide=g.pivot(index='focal_position',columns='focal_valid',values='R')
  if not {True,False}<=set(wide):raise ValueError('matched APPLIED/IGNORED R cells are incomplete')
  if field==f'proposed_{focal}': delta=wide[True]-wide[False];name='delta_R_proposed'
  elif field==f'initial_{focal}': delta=wide[False]-wide[True];name='delta_R_initial'
  else:raise ValueError('unexpected edited field')
  for pos,val in delta.items():contrasts.append({'history_id':hid,'focal_variable':focal,'metric':name,'focal_position':pos,'value':float(val)})
 h=pd.DataFrame(contrasts)
 # Average the independently counterbalanced focal-first/second values within each history.
 h=h.groupby(['history_id','focal_variable','metric'],as_index=False).value.mean()
 h['axis']=h['focal_variable']; sym=h.groupby(['history_id','metric'],as_index=False).value.mean();sym['axis']='symmetric';h=pd.concat([h,sym],ignore_index=True)
 summary=[]
 for (metric,axis),g in h.groupby(['metric','axis'],sort=True):
  v=g.value.to_numpy(float);ci=bootstrap_mean_ci(v,n_boot=n_boot,seed=seed+len(metric)+len(axis))
  summary.append({'metric':metric,'axis':axis,'n_histories':len(v),'mean':float(v.mean()),'median':float(np.median(v)),'ci95':[ci['ci_low'],ci['ci_high']]})
 return e,h,summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--behavior',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--bootstrap-seed',type=int,default=20261010);p.add_argument('--bootstrap-draws',type=int,default=10000);a=p.parse_args()
 rows=read_jsonl(a.behavior);prov=Path(a.behavior+'.provenance.json')
 if not prov.is_file() or json.loads(prov.read_text()).get('purpose')!='confirmatory_behavioral_matched_edit':raise ValueError('confirmatory behavior provenance is required')
 e,h,s=analyze(rows,a.bootstrap_seed,a.bootstrap_draws);out=Path(a.output_dir)
 if out.exists() and any(out.iterdir()):raise FileExistsError(f'{out} is nonempty')
 out.mkdir(parents=True,exist_ok=True);e.to_csv(out/'status_focal_R_by_history_cell.csv',index=False);h.to_csv(out/'status_focal_validity_effects_by_history.csv',index=False)
 (out/'status_focal_validity_effect_summary.json').write_text(json.dumps({'analysis_label':'confirmatory_status_focal_behavior','metrics':{'delta_R_proposed':'R(APPLIED proposed_focal) - R(IGNORED proposed_focal)','delta_R_initial':'R(retained initial_focal) - R(superseded initial_focal)','R':'identity transfer E queried at focal variable minus E queried at other variable'},'bootstrap_unit':'history_id','bootstrap_seed':a.bootstrap_seed,'bootstrap_draws':a.bootstrap_draws,'raw_logits_and_competence_retained_in_source':True,'rows':s},indent=2)+'\n')
if __name__=='__main__':main()

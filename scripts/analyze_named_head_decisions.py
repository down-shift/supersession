#!/usr/bin/env python3
"""Exploratory history-bootstrap summary of saved named-head decision logits."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from src.analysis.metrics import bootstrap_mean_ci
NAMED={(32,8):'primary',(34,1):'primary',(34,28):'primary',(33,12):'secondary',(34,29):'negative_control',(35,27):'negative_control'}

FIELDS=('baseline_current_minus_old','patched_current_minus_old','patch_delta_current_minus_old',
        'baseline_current_logit','baseline_old_logit','patched_current_logit','patched_old_logit')

def summarize(rows,seed=20261001,n_boot=10000):
    f=pd.DataFrame(rows); required={'history_id','layer','head','edited_binding','direction',*FIELDS}
    if not required<=set(f): raise ValueError(f'missing fields: {sorted(required-set(f))}')
    if {(int(x),int(y)) for x,y in zip(f['layer'],f['head'])}!=set(NAMED): raise ValueError('expected six named heads')
    # Collapse positions and donor directions within history before treating histories as independent.
    keys=['history_id','layer','head','edited_binding']
    h=f.groupby(keys,as_index=False)[list(FIELDS)].mean()
    h['axis']=h['edited_binding'].str[-1]
    if not set(h['axis'])<=set('xz'): raise ValueError('edited binding must identify x/z')
    metrics={'patch_delta_current_minus_old':'patch decision margin change',
      'patched_current_change':'patched minus baseline current logit',
      'patched_obsolete_change':'patched minus baseline obsolete logit'}
    h['patched_current_change']=h['patched_current_logit']-h['baseline_current_logit']
    h['patched_obsolete_change']=h['patched_old_logit']-h['baseline_old_logit']
    records=[]
    for (layer,head),g in h.groupby(['layer','head'],sort=True):
        rec={'layer':int(layer),'head':int(head),'role':NAMED[(int(layer),int(head))],'n_histories':g.history_id.nunique()}
        for metric in metrics:
            for axis,part in [('x',g[g['axis']=='x']),('z',g[g['axis']=='z']),('symmetric',g.groupby('history_id')[metric].mean().reset_index())]:
                vals=part[metric].to_numpy(float)
                ci=bootstrap_mean_ci(vals,n_boot=n_boot,seed=seed+int(layer)*100+int(head)*7+len(metric)+len(axis))
                rec[f'{metric}_{axis}_mean']=float(vals.mean()); rec[f'{metric}_{axis}_median']=float(np.median(vals))
                rec[f'{metric}_{axis}_ci95']=[ci['ci_low'],ci['ci_high']]
        records.append(rec)
    return h,records

def main():
    p=argparse.ArgumentParser();p.add_argument('--patches',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--seed',type=int,default=20261001);p.add_argument('--bootstrap-draws',type=int,default=10000);a=p.parse_args()
    rows=[json.loads(x) for x in Path(a.patches).read_text().splitlines() if x.strip()]
    hist,summary=summarize(rows,a.seed,a.bootstrap_draws);out=Path(a.output_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'{out} is nonempty')
    out.mkdir(parents=True,exist_ok=True);hist.to_csv(out/'named_head_decisions_by_history.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'named_head_decision_summary.csv',index=False)
    (out/'analysis.json').write_text(json.dumps({'analysis_label':'exploratory_validity_dependent_version_selection_diagnostics','interpretation':'exploratory only; does not establish version selection','bootstrap_unit':'history_id','seed':a.seed,'draws':a.bootstrap_draws,'reserve_used':False,'heads':{f'L{l}H{h}':v for (l,h),v in NAMED.items()},'rows':summary},indent=2)+'\n')
if __name__=='__main__':main()

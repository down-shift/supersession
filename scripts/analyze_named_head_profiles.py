#!/usr/bin/env python3
"""Summarize the explicitly named exploratory profiles from 24 discovery histories."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from src.analysis.metrics import bootstrap_mean_ci
from scripts.analyze_attention_head_patching import history_head_profiles

NAMED = {(32,8):'primary', (34,1):'primary', (34,28):'primary', (33,12):'secondary', (34,29):'negative_control', (35,27):'negative_control'}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--patches',required=True); p.add_argument('--output-dir',required=True)
    p.add_argument('--seed',type=int,default=20260930); p.add_argument('--bootstrap-draws',type=int,default=10000); a=p.parse_args()
    rows=[json.loads(line) for line in Path(a.patches).read_text().splitlines() if line.strip()]
    if not rows: raise ValueError('patch file is empty')
    manifest_path=Path(str(a.patches)+'.provenance.json')
    if not manifest_path.is_file(): raise ValueError('named head patches require their runner provenance sidecar')
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('stage')!='discovery' or not manifest.get('exploratory'):
        raise ValueError('named head analysis requires an exploratory discovery run')
    if {tuple(x) for x in manifest.get('profile_heads',[])}!=set(NAMED):
        raise ValueError('runner provenance does not match the six predeclared named heads')
    actual={(int(r['layer']),int(r['head'])) for r in rows}
    if actual != set(NAMED): raise ValueError(f'profile head set differs from predeclared six heads: {sorted(actual)}')
    directions={'baseline_to_edited','edited_to_baseline'}
    cells={}
    for r in rows:
        k=(r['history_id'],int(r['layer']),int(r['head']),r['edited_binding'],r['query_id'])
        cells.setdefault(k,set()).add(r['direction'])
    if any(d!=directions for d in cells.values()): raise ValueError('each raw head cell must retain both donor/recipient directions')
    by_profile={(layer,head):{(b,q) for (hid,l,h,b,q) in cells if l==layer and h==head} for layer,head in NAMED}
    expected_cells={('old_x','current_x'),('old_x','current_z'),('old_x','initial_x'),('old_x','initial_z'),
                    ('old_z','current_x'),('old_z','current_z'),('old_z','initial_x'),('old_z','initial_z'),
                    ('current_x','current_x'),('current_x','current_z'),('current_z','current_x'),('current_z','current_z')}
    if any(c!=expected_cells for c in by_profile.values()): raise ValueError('named head profile is missing required history/query cells')
    history=history_head_profiles(rows)
    if history.history_id.nunique()!=24 or set(history.history_id)!={str(x) for x in manifest.get('history_ids',[])}: raise ValueError('named profile must use exactly the 24 stage1_discovery histories')
    if history.groupby(['layer','head']).history_id.nunique().ne(24).any(): raise ValueError('each named head needs complete profiles for all 24 discovery histories')
    out=Path(a.output_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'{out} is nonempty; choose a fresh analysis path')
    out.mkdir(parents=True,exist_ok=True)
    history['profile_role']=[NAMED[(int(l),int(h))] for l,h in zip(history.layer,history.head)]
    history.to_csv(out/'named_head_profiles_by_history.csv',index=False)
    metrics=['R_stale','R_historical','R_current']
    records=[]
    for (layer,head),g in history.groupby(['layer','head'],sort=True):
        record={'layer':int(layer),'head':int(head),'profile_role':NAMED[(int(layer),int(head))],'n_histories':len(g)}
        for metric in metrics:
            for suffix in ('_x','_z',''):
                col=metric+suffix; x=g[col].to_numpy(float)
                ci=bootstrap_mean_ci(x,n_boot=a.bootstrap_draws,seed=a.seed+layer*100+head*3+len(suffix))
                name=metric+suffix if suffix else metric+'_symmetric'
                record[name+'_mean']=float(np.mean(x)); record[name+'_median']=float(np.median(x))
                record[name+'_fraction_positive']=float(np.mean(x>0)); record[name+'_ci95']=[ci['ci_low'],ci['ci_high']]
        records.append(record)
    pd.DataFrame(records).to_csv(out/'named_head_profile_summary.csv',index=False)
    (out/'analysis.json').write_text(json.dumps({'analysis_label':'exploratory_named_head_functional_profile','history_ids':sorted(history.history_id.unique()),'heads':{f'L{l}H{h}':role for (l,h),role in NAMED.items()},'definitions':{'R_stale':'old_x query-x minus query-z and symmetric old_z contrast','R_historical':'old initial value queried at historical query position minus other variable historical query, symmetric x/z','R_current':'current value queried at current query position minus other variable current query, symmetric x/z'},'profiles':records,'automatic_head_selection':False,'reserve_used':False,'bootstrap_unit':'history_id','bootstrap_seed':a.seed,'bootstrap_draws':a.bootstrap_draws},indent=2)+'\n')
    print(f'wrote named head profiles for {history.history_id.nunique()} discovery histories to {out}')

if __name__=='__main__': main()

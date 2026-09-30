#!/usr/bin/env python3
"""Summarize exploratory block-component R_x by layer with history bootstrap."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def history_rx(rows):
    f=pd.DataFrame(rows)
    required={'history_id','layer','component','query_id','edited_binding','direction','patch_delta_toward_donor'}
    if not required.issubset(f.columns): raise ValueError(f'patch records missing {sorted(required-set(f.columns))}')
    f=f[f.edited_binding.eq('old_x') & f.query_id.isin(['current_x','current_z'])]
    directions=f.groupby(['history_id','layer','component','query_id','direction'],as_index=False).patch_delta_toward_donor.mean()
    effects=directions.groupby(['history_id','layer','component','query_id'],as_index=False).patch_delta_toward_donor.mean()
    wide=effects.pivot(index=['history_id','layer','component'],columns='query_id',values='patch_delta_toward_donor').reset_index()
    if not {'current_x','current_z'}.issubset(wide.columns): raise ValueError('each layer/component/history needs both current query cells')
    wide=wide.dropna(subset=['current_x','current_z'])
    wide['R_x_patch']=wide.current_x-wide.current_z
    return wide

def summarize(history,seed=20260930,n_boot=10000):
    rng=np.random.default_rng(seed); result=[]
    for (component,layer),g in history.groupby(['component','layer'],sort=True):
        x=g.R_x_patch.to_numpy(float); n=len(x)
        means=np.mean(rng.choice(x,size=(n_boot,n),replace=True),axis=1)
        result.append({'component':component,'layer':int(layer),'n_histories':n,'mean':float(np.mean(x)),'median':float(np.median(x)),'fraction_positive':float(np.mean(x>0)),'ci95_low':float(np.quantile(means,.025)),'ci95_high':float(np.quantile(means,.975))})
    return pd.DataFrame(result)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--patches',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--bootstrap-seed',type=int,default=20260930); p.add_argument('--bootstrap-draws',type=int,default=10000); a=p.parse_args()
    rows=[json.loads(line) for line in Path(a.patches).read_text().splitlines() if line.strip()]
    hist=history_rx(rows); summary=summarize(hist,a.bootstrap_seed,a.bootstrap_draws)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    hist.to_csv(out/'component_Rx_by_history.csv',index=False)
    summary.to_csv(out/'component_Rx_by_layer.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,6))
    for component,g in summary.groupby('component',sort=False):
        g=g.sort_values('layer'); x=g.layer.to_numpy(); y=g['mean'].to_numpy(); lo=g.ci95_low.to_numpy(); hi=g.ci95_high.to_numpy()
        ax.plot(x,y,marker='o',label=component); ax.fill_between(x,lo,hi,alpha=.16)
    ax.axhline(0,color='black',linewidth=.7); ax.set(xlabel='Transformer layer',ylabel='Mean history-level $R_{x,patch}$ (logits)',title='Exploratory block-component patching at final_preanswer'); ax.legend(); fig.tight_layout(); fig.savefig(out/'component_Rx_by_layer.png',dpi=180); plt.close(fig)
    doc={'analysis_label':'exploratory','exploratory':True,'metric':'R_x_patch = mean effect(old_x/current_x) - mean effect(old_x/current_z), with bidirectional donor-oriented effects averaged within history first','bootstrap_unit':'history_id','bootstrap_seed':a.bootstrap_seed,'bootstrap_draws':a.bootstrap_draws,'n_histories':int(hist.history_id.nunique()),'rows':summary.to_dict(orient='records'),'interpretation':'Descriptive component trajectories only; no component/layer is selected automatically.'}
    (out/'component_summary.json').write_text(json.dumps(doc,indent=2)+'\n')
if __name__=='__main__': main()

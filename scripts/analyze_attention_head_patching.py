#!/usr/bin/env python3
"""Analyze exploratory per-query-head R_x effects or frozen reserve head sets."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def history_head_rx(rows):
    f=pd.DataFrame(rows)
    required={'history_id','layer','head','query_id','edited_binding','direction','patch_delta_toward_donor'}
    if not required.issubset(f.columns): raise ValueError(f'head records missing {sorted(required-set(f.columns))}')
    f=f[f.edited_binding.eq('old_x') & f.query_id.isin(['current_x','current_z'])]
    direction=f.groupby(['history_id','layer','head','query_id','direction'],as_index=False).patch_delta_toward_donor.mean()
    effect=direction.groupby(['history_id','layer','head','query_id'],as_index=False).patch_delta_toward_donor.mean()
    wide=effect.pivot(index=['history_id','layer','head'],columns='query_id',values='patch_delta_toward_donor').reset_index()
    if not {'current_x','current_z'}.issubset(wide.columns): raise ValueError('each head/history needs both old_x query cells')
    wide=wide.dropna(subset=['current_x','current_z']); wide['R_x_patch']=wide.current_x-wide.current_z
    return wide

def history_head_profiles(rows):
    """History-level stale, valid-history, and current binding query contrasts."""
    f=pd.DataFrame(rows); required={'history_id','layer','head','query_id','edited_binding','direction','patch_delta_toward_donor'}
    if not required.issubset(f.columns): raise ValueError(f'head records missing {sorted(required-set(f.columns))}')
    d=f.groupby(['history_id','layer','head','edited_binding','query_id','direction'],as_index=False).patch_delta_toward_donor.mean()
    e=d.groupby(['history_id','layer','head','edited_binding','query_id'],as_index=False).patch_delta_toward_donor.mean()
    def contrast(binding, rel, irr, name):
        x=e[e.edited_binding.eq(binding)].pivot(index=['history_id','layer','head'],columns='query_id',values='patch_delta_toward_donor')
        if rel not in x or irr not in x: raise ValueError(f'incomplete head-analysis cells for {binding}/{rel}/{irr}')
        return (x[rel]-x[irr]).rename(name).reset_index()
    keys=['history_id','layer','head']
    out=contrast('old_x','current_x','current_z','R_stale_x').merge(contrast('old_z','current_z','current_x','R_stale_z'),on=keys)
    out=out.merge(contrast('old_x','initial_x','initial_z','R_historical_x'),on=keys).merge(contrast('old_z','initial_z','initial_x','R_historical_z'),on=keys)
    out=out.merge(contrast('current_x','current_x','current_z','R_current_x'),on=keys).merge(contrast('current_z','current_z','current_x','R_current_z'),on=keys)
    out['R_stale']=.5*(out.R_stale_x+out.R_stale_z); out['R_historical']=.5*(out.R_historical_x+out.R_historical_z); out['R_current']=.5*(out.R_current_x+out.R_current_z)
    out['generic_binding_score']=.5*(out.R_historical+out.R_current); out['stale_minus_generic_binding']=out.R_stale-out.generic_binding_score
    return out

def _trimmed_mean(x,proportion=.1):
    x=np.sort(np.asarray(x,dtype=float)); k=int(np.floor(len(x)*proportion))
    return float(np.mean(x[k:len(x)-k])) if k and len(x)>2*k else float(np.mean(x))

def summarize_heads(history,seed=20260930,n_boot=10000):
    rng=np.random.default_rng(seed); records=[]
    for (layer,head),g in history.groupby(['layer','head'],sort=True):
        values=g.R_x_patch.to_numpy(float); n=len(values)
        draws=np.mean(rng.choice(values,size=(n_boot,n),replace=True),axis=1)
        records.append({'layer':int(layer),'head':int(head),'n_histories':n,'mean':float(np.mean(values)),'median':float(np.median(values)),'trimmed_mean_10pct':_trimmed_mean(values),'fraction_positive':float(np.mean(values>0)),'ci95_low':float(np.quantile(draws,.025)),'ci95_high':float(np.quantile(draws,.975))})
    return pd.DataFrame(records)

def compare_whole_attention(history,whole_path):
    if not whole_path: return None
    whole=pd.read_csv(whole_path)
    required={'history_id','layer','component','R_x_patch'}
    if not required.issubset(whole.columns): raise ValueError(f'whole-attention input missing {sorted(required-set(whole.columns))}')
    whole=whole[whole.component.eq('attention_output')][['history_id','layer','R_x_patch']].rename(columns={'R_x_patch':'whole_attention_R_x_patch'})
    sums=history.groupby(['history_id','layer'],as_index=False).R_x_patch.sum().rename(columns={'R_x_patch':'sum_individual_head_R_x_patch'})
    paired=sums.merge(whole,on=['history_id','layer'],how='inner',validate='one_to_one')
    if paired.empty: raise ValueError('no matching history/layer rows in whole-attention comparison input')
    paired['sum_minus_whole_attention']=paired.sum_individual_head_R_x_patch-paired.whole_attention_R_x_patch
    return paired.groupby('layer',as_index=False).agg(n_histories=('history_id','nunique'),sum_individual_heads_mean=('sum_individual_head_R_x_patch','mean'),whole_attention_output_mean=('whole_attention_R_x_patch','mean'),mean_sum_minus_whole_attention=('sum_minus_whole_attention','mean'),median_sum_minus_whole_attention=('sum_minus_whole_attention','median'))

def reserve_summary(rows,seed=20260930,n_boot=10000):
    f=pd.DataFrame(rows)
    keys={'history_id','edited_binding','query_id','direction','patch_delta_toward_donor','head_set'}
    if not keys.issubset(f.columns): raise ValueError(f'reserve records missing {sorted(keys-set(f.columns))}')
    f['head_set_key']=f.head_set.map(lambda x:json.dumps(x,sort_keys=True))
    d=f.groupby(['history_id','head_set_key','edited_binding','query_id','direction'],as_index=False).patch_delta_toward_donor.mean()
    e=d.groupby(['history_id','head_set_key','edited_binding','query_id'],as_index=False).patch_delta_toward_donor.mean()
    contrasts=[]
    for binding,relevant,other,label in [('old_x','current_x','current_z','R_x'),('old_z','current_z','current_x','R_z')]:
        q=e[e.edited_binding.eq(binding)].pivot(index=['history_id','head_set_key'],columns='query_id',values='patch_delta_toward_donor').reset_index()
        if not {relevant,other}.issubset(q.columns): raise ValueError(f'reserve records incomplete for {binding} query pair')
        q[label]=q[relevant]-q[other]; contrasts.append(q[['history_id','head_set_key',label]])
    both=contrasts[0].merge(contrasts[1],on=['history_id','head_set_key'],validate='one_to_one')
    both['symmetric_R']=.5*(both.R_x+both.R_z)
    rng=np.random.default_rng(seed); summary=[]
    for key,g in both.groupby('head_set_key'):
        rec={'head_set':json.loads(key),'n_histories':g.history_id.nunique()}
        for metric in ('R_x','R_z','symmetric_R'):
            x=g[metric].to_numpy(float); draws=np.mean(rng.choice(x,size=(n_boot,len(x)),replace=True),axis=1)
            rec.update({f'{metric}_mean':float(x.mean()),f'{metric}_median':float(np.median(x)),f'{metric}_ci95_low':float(np.quantile(draws,.025)),f'{metric}_ci95_high':float(np.quantile(draws,.975))})
        summary.append(rec)
    return both,pd.DataFrame(summary)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--patches',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--whole-attention-by-history'); p.add_argument('--bootstrap-seed',type=int,default=20260930); p.add_argument('--bootstrap-draws',type=int,default=10000); a=p.parse_args()
    rows=[json.loads(line) for line in Path(a.patches).read_text().splitlines() if line.strip()]
    if not rows: raise ValueError('patch file is empty')
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    if 'head_set' in rows[0]:
        per_history,summary=reserve_summary(rows,a.bootstrap_seed,a.bootstrap_draws)
        per_history.to_csv(out/'frozen_head_set_R_by_history.csv',index=False); summary.to_csv(out/'frozen_head_set_summary.csv',index=False)
        (out/'frozen_head_set_summary.json').write_text(json.dumps({'analysis_label':'reserve_confirmation','bootstrap_unit':'history_id','rows':summary.to_dict(orient='records')},indent=2)+'\n')
        return
    try:
        profiles=history_head_profiles(rows)
        profiles.to_csv(out/'head_mechanistic_profile_by_history.csv',index=False)
    except ValueError:
        # Backward-compatible analysis of archived, narrower scans. New discovery
        # outputs are required to contain every profile cell before interpretation.
        profiles=None
    history=history_head_rx(rows); summary=summarize_heads(history,a.bootstrap_seed,a.bootstrap_draws)
    layer_heads=rows[0].get('num_attention_heads')
    if layer_heads is not None:
        expected={(int(r.layer),h) for r in summary.itertuples() for h in range(int(layer_heads))}
        actual=set(zip(summary['layer'].astype(int),summary['head'].astype(int)))
        if expected!=actual: raise ValueError('head scan is incomplete: not every configured query head has records at each scanned layer')
    if (summary.n_histories!=history.history_id.nunique()).any(): raise ValueError('some head/layer cells are missing histories')
    comparison=compare_whole_attention(history,a.whole_attention_by_history)
    history.to_csv(out/'head_Rx_by_history.csv',index=False); summary.to_csv(out/'head_Rx_by_layer.csv',index=False)
    if comparison is not None: comparison.to_csv(out/'head_sum_vs_attention.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    pivot=summary.pivot(index='layer',columns='head',values='mean').sort_index()
    vmax=float(np.nanmax(np.abs(pivot.to_numpy()))) or 1.
    fig,ax=plt.subplots(figsize=(max(10,pivot.shape[1]*.35),5.5)); image=ax.imshow(pivot.to_numpy(),aspect='auto',origin='lower',cmap='coolwarm',norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax))
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns); ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index); ax.set(xlabel='Query-head index',ylabel='Transformer layer',title='Exploratory mean history-level $R_{x,patch}$ by Qwen3 query head'); fig.colorbar(image,ax=ax,label='Mean $R_{x,patch}$ (logits)'); fig.tight_layout(); fig.savefig(out/'head_Rx_heatmap.png',dpi=180); plt.close(fig)
    doc={'analysis_label':'exploratory','exploratory':True,'metric':'history-level R_x_patch = effect(old_x/current_x) - effect(old_x/current_z), each cell averaged across both donor-oriented directions first','profile_definitions':{'R_stale':'mean symmetric query relevance contrast for old_x/current_x vs old_x/current_z and old_z/current_z vs old_z/current_x','R_historical':'mean symmetric query relevance contrast for old_x/initial_x vs old_x/initial_z and symmetric z cells','R_current':'mean symmetric query relevance contrast for current_x/current_x vs current_x/current_z and symmetric z cells','generic_binding_score':'mean(R_historical,R_current)','stale_minus_generic_binding':'R_stale - generic_binding_score; exploratory diagnostic only'},'profile_cells_complete':profiles is not None,'bootstrap_unit':'history_id','bootstrap_seed':a.bootstrap_seed,'bootstrap_draws':a.bootstrap_draws,'n_histories':int(history.history_id.nunique()),'n_layers':int(summary['layer'].nunique()),'n_heads_per_layer':{str(k):int(v) for k,v in summary.groupby('layer')['head'].nunique().items()},'individual_head_effects_are_additive':False,'interpretation':'Per-head patch effects are single-head interventions in the intact network. Their sum is descriptive only and is not a decomposition; joint all-head intervention should reproduce whole-attention-output patching by construction and is covered by a unit test. No head is selected automatically.','whole_attention_comparison':None if comparison is None else comparison.to_dict(orient='records'),'rows':summary.to_dict(orient='records')}
    (out/'head_summary.json').write_text(json.dumps(doc,indent=2)+'\n')

if __name__=='__main__': main()

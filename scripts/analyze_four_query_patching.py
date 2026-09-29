#!/usr/bin/env python3
"""History-level analysis of donor-oriented four-query residual patches."""
import argparse, collections, json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.io import read_jsonl
from src.utils import save_json
from src.analysis.metrics import trimmed_mean
from src.experiments.patching import r_x_patch, symmetric_r_patch

def stats(values,seed=0,n_boot=10000,n_perm=10000):
 x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]; n=len(x)
 if not n:return {'n_histories':0,'mean':None,'median':None,'trimmed_mean_10pct':None,'fraction_positive':None,'ci95_cluster_bootstrap':None,'sign_flip_p_two_sided':None}
 rng=np.random.default_rng(seed); boot=np.mean(rng.choice(x,(n_boot,n),replace=True),axis=1)
 observed=abs(float(x.mean())); flips=rng.choice(np.array([-1.,1.]),(n_perm,n)); p=(1+np.sum(np.abs(np.mean(flips*x,axis=1))>=observed))/(n_perm+1)
 return {'n_histories':n,'mean':float(x.mean()),'median':float(np.median(x)),'trimmed_mean_10pct':float(trimmed_mean(x,.1)),'fraction_positive':float(np.mean(x>0)),'ci95_cluster_bootstrap':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'sign_flip_p_two_sided':float(p)}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--patches',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--stage',choices=('discovery','heldout'),required=True); p.add_argument('--seed',type=int,default=20260929); p.add_argument('--selection-site',default='final_preanswer'); p.add_argument('--region-width',type=int,choices=(3,),default=3); a=p.parse_args()
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); rows=read_jsonl(a.patches)
 required={'history_id','query_id','edited_binding','direction','layer','patch_delta_toward_donor'}
 if rows and not required.issubset(rows[0]): raise ValueError(f'missing patch fields: {sorted(required-set(rows[0]))}')
 for r in rows:
  r.setdefault('position',0)
  r.setdefault('site_role',r.get('site','other_position')); r.setdefault('site_label',r.get('site','other_position'))
  aliases={'donor_to_recipient':'baseline_to_edited','recipient_to_donor':'edited_to_baseline'}
  r['direction']=aliases.get(r['direction'],r['direction'])
 frame=pd.DataFrame(rows)
 # Each absolute position remains its own cell in all-position data. Semantic
 # multi-token spans are averaged within direction by role.
 allpos=bool(frame.site_role.eq('other_position').any())
 keys=['history_id','edited_binding','query_id','site_role','layer']+(['position'] if allpos else [])
 d=frame.groupby(keys+['direction'],dropna=False).patch_delta_toward_donor.mean().reset_index()
 pivot=d.pivot(index=keys,columns='direction',values='patch_delta_toward_donor').reset_index()
 needed={'baseline_to_edited','edited_to_baseline'}
 if not needed.issubset(pivot.columns): raise ValueError('both donor-oriented patch directions are required')
 pivot['mean_donor_oriented_patch_effect']=pivot[list(needed)].mean(axis=1)
 pivot['effect']=pivot['mean_donor_oriented_patch_effect']
 if allpos:
  tokenmeta=frame.groupby('position',as_index=False).agg(
   token_id=('token_id',lambda x:'|'.join(map(str,sorted(set(x.dropna()))))),
   token_text=('token_text',lambda x:' | '.join(sorted(set(str(v) for v in x.dropna())))))
  pivot=pivot.merge(tokenmeta,on='position',how='left')
 if a.stage=='heldout':
  derived=[]
  groupkeys=['history_id','site_role','layer']+(['position'] if allpos else [])
  for key,g in pivot.groupby(groupkeys,dropna=False):
   by={(r.edited_binding,r.query_id):float(r.effect) for r in g.itertuples()}
   if all(k in by for k in [('old_x','current_x'),('old_x','current_z'),('old_z','current_z'),('old_z','current_x')]):
    entry=dict(zip(groupkeys,key if isinstance(key,tuple) else (key,)))
    entry.update({'measure':'primary_contrasts','R_x_patch':r_x_patch(by['old_x','current_x'],by['old_x','current_z']),'R_z_patch':r_x_patch(by['old_z','current_z'],by['old_z','current_x'])})
    entry['symmetric_R_patch']=symmetric_r_patch(entry['R_x_patch'],entry['R_z_patch']); derived.append(entry)
  if derived:
   # Separate derived contrasts make the held-out history-level primary easy to audit.
   pd.DataFrame(derived).to_csv(out/'heldout_R_by_history.csv',index=False)
 pivot.to_csv(out/('heldout_patch_effects_by_history.csv' if a.stage=='heldout' else 'patch_effects_by_history.csv'),index=False)
 def cell(binding,query,layer,role,position=None):
  f=pivot[(pivot.edited_binding==binding)&(pivot.query_id==query)&(pivot.layer==layer)&(pivot.site_role==role)]
  if position is not None:f=f[f.position==position]
  return dict(zip(f.history_id,f.effect))
 layers=sorted(int(x) for x in pivot.layer.unique()); roles=sorted(set(pivot.site_role)); results={}; perlayer={}
 positions=sorted(int(x) for x in pivot.position.unique()) if allpos else [None]
 for role in roles:
  for layer in layers:
   for position in positions:
    c={(b,q):cell(b,q,layer,role,position) for b in ('old_x','current_x','old_z','current_z') for q in ('current_x','initial_x','current_z','initial_z')}
    rxs={h:r_x_patch(c['old_x','current_x'][h],c['old_x','current_z'][h]) for h in set(c['old_x','current_x'])&set(c['old_x','current_z'])}
    rzs={h:r_x_patch(c['old_z','current_z'][h],c['old_z','current_x'][h]) for h in set(c['old_z','current_z'])&set(c['old_z','current_x'])}
    both=set(rxs)&set(rzs); sym={h:symmetric_r_patch(rxs[h],rzs[h]) for h in both}
    sxids=set(c['old_x','initial_x'])&set(c['old_x','current_x'])&set(c['current_x','current_x'])&set(c['current_x','initial_x'])
    szids=set(c['old_z','initial_z'])&set(c['old_z','current_z'])&set(c['current_z','current_z'])&set(c['current_z','initial_z'])
    sx={h:.5*((c['old_x','initial_x'][h]-c['old_x','current_x'][h])+(c['current_x','current_x'][h]-c['current_x','initial_x'][h])) for h in sxids}
    sz={h:.5*((c['old_z','initial_z'][h]-c['old_z','current_z'][h])+(c['current_z','current_z'][h]-c['current_z','initial_z'][h])) for h in szids}
    key=f'{role}/layer_{layer}'+(f'/position_{position}' if position is not None else ''); results[key]={'R_x_patch':stats(list(rxs.values()),a.seed),'R_z_patch':stats(list(rzs.values()),a.seed+1),'symmetric_R_patch':stats(list(sym.values()),a.seed+2),'S_x_control':stats(list(sx.values()),a.seed+3),'S_z_control':stats(list(sz.values()),a.seed+4),'cells_present':{f'{b}/{q}':len(v) for (b,q),v in c.items()}}
    perlayer[(role,layer,position)]=rxs
 region_primary={}; region_rows=[]
 if a.stage=='heldout':
  regionkeys=['history_id','edited_binding','query_id','site_role']+(['position'] if allpos else [])
  grouped=pivot.groupby(regionkeys,dropna=False).effect.agg(['mean','count']).reset_index()
  if not grouped.empty and not (grouped['count']==len(layers)).all(): raise ValueError('held-out data are incomplete across the frozen layer region')
  for role in sorted(set(grouped.site_role)):
   subset=grouped[grouped.site_role==role]
   positions_for_role=sorted(int(x) for x in subset.position.unique()) if allpos else [None]
   for position in positions_for_role:
    f=subset if position is None else subset[subset.position==position]
    cells={}
    for b in ('old_x','old_z','current_x','current_z'):
     for q in ('current_x','initial_x','current_z','initial_z'):
      z=f[(f.edited_binding==b)&(f.query_id==q)]; cells[(b,q)]=dict(zip(z.history_id,z['mean']))
    rxids=set(cells['old_x','current_x'])&set(cells['old_x','current_z']); rzids=set(cells['old_z','current_z'])&set(cells['old_z','current_x'])
    rx={h:r_x_patch(cells['old_x','current_x'][h],cells['old_x','current_z'][h]) for h in rxids}; rz={h:r_x_patch(cells['old_z','current_z'][h],cells['old_z','current_x'][h]) for h in rzids}
    both=set(rx)&set(rz); sy={h:symmetric_r_patch(rx[h],rz[h]) for h in both}; key=role+(f'/position_{position}' if position is not None else '')
    region_primary[key]={'R_x_patch':stats(list(rx.values()),a.seed),'R_z_patch':stats(list(rz.values()),a.seed+1),'symmetric_R_patch':stats(list(sy.values()),a.seed+2),'frozen_layers':layers}
    for h in both: region_rows.append({'history_id':h,'layer':'selected_region','site_role':role,'position':position,'edited_binding':'symmetric','query_id':'symmetric_R_patch','R_x_patch':rx[h],'R_z_patch':rz[h],'symmetric_R_patch':sy[h]})
 if a.stage=='discovery':
  sel={}; siteframe=pivot[(pivot.site_role==a.selection_site)]
  actual_layers=sorted(int(x) for x in siteframe.layer.unique()); by={}
  for layer in actual_layers:
   c={(b,q):cell(b,q,layer,a.selection_site) for b,q in [('old_x','current_x'),('old_x','current_z')]}
   ids=set(c['old_x','current_x'])&set(c['old_x','current_z']); by[layer]={h:r_x_patch(c['old_x','current_x'][h],c['old_x','current_z'][h]) for h in ids}
  windows=[]
  for i in range(len(actual_layers)-a.region_width+1):
   win=actual_layers[i:i+a.region_width]
   if win==list(range(win[0],win[0]+a.region_width)):
    idsets=[set(by[l]) for l in win]; common=set.intersection(*idsets) if idsets else set()
    vals={h:float(np.mean([by[l][h] for l in win])) for h in common}
    windows.append((float(np.mean(list(vals.values()))) if vals else float('-inf'),win[0],win,len(common)))
  if not windows: raise ValueError('no complete contiguous three-layer selection window')
  best=max(windows,key=lambda x:(x[0],-x[1]))
  sel={'selection_statistic':{'name':'R_x_patch','formula':'old_x/current_x - old_x/current_z','site':a.selection_site,'window_width':a.region_width,'selected_layers':best[2],'tie_break':'lowest_start_layer'},'per_layer_R_x_patch':{str(l):stats(list(by[l].values()),a.seed) for l in actual_layers},'selected_window_mean_R_x_patch':best[0],'complete_histories_in_window':best[3]}
  save_json(sel,out/'discovery_selection.json')
 summary={'stage':a.stage,'results':results,'selected_region_primary':region_primary if a.stage=='heldout' else None,'primary':'R_x_patch at discovery; selected-region symmetric_R_patch at heldout','cells_present':sorted({f"{r.edited_binding}/{r.query_id}" for r in frame.itertuples()}),'bootstrap_unit':'history_id','direction_order':'donor-oriented each direction, then averaged','selection':sel if a.stage=='discovery' else None}
 save_json(summary,out/'four_query_patch_summary.json' if a.stage=='discovery' else out/'heldout_patch_summary.json')
 if allpos:
  # Absolute position is part of the analysis key; never pool token locations.
  hx=frame[frame.edited_binding=='old_x']; g=hx.groupby(['layer','position','token_id','token_text','site_role'],dropna=False).patch_delta_toward_donor.mean().reset_index()
  # form R at history level then aggregate by (layer,position)
  ps=pivot[pivot.edited_binding=='old_x']; wide=ps.pivot(index=['history_id','layer','position','token_id','token_text','site_role'],columns='query_id',values='effect').reset_index()
  wide['R_x_patch']=wide['current_x']-wide['current_z']; agg=wide.groupby(['layer','position','token_id','token_text','site_role']).R_x_patch.agg(['mean','count']).reset_index(); agg.to_csv(out/'all_positions_Rx.csv',index=False)
  try:
   import os
   mplconfig=out/'.mplconfig'; mplconfig.mkdir(exist_ok=True); os.environ.setdefault('MPLCONFIGDIR',str(mplconfig))
   import matplotlib.pyplot as plt
   mat=agg.pivot(index='layer',columns='position',values='mean').sort_index(); fig,ax=plt.subplots(figsize=(max(10,mat.shape[1]*.25),max(4,mat.shape[0]*.22))); im=ax.imshow(mat,aspect='auto',origin='lower',interpolation='nearest'); ax.set(xlabel='input token position',ylabel='transformer layer'); fig.colorbar(im,ax=ax,label='mean history-level R_x_patch'); fig.tight_layout(); fig.savefig(out/'all_positions_Rx.png',dpi=160); plt.close(fig)
  except ImportError: pass
  (out/'all_positions_token_legend.csv').write_text(agg[['position','token_id','token_text','site_role']].drop_duplicates().sort_values('position').to_csv(index=False),encoding='utf8')
 # compact per-history layer/site table for R contrasts
 outrows=[]
 for (role,layer,position),vals in perlayer.items():
  for h,v in vals.items():outrows.append({'history_id':h,'layer':layer,'site_role':role,'position':position,'edited_binding':'old_x','query_id':'current_x-current_z','mean_donor_oriented_patch_effect':v,'R_x_patch':v})
 pd.DataFrame(outrows).to_csv(out/'patch_Rx_by_layer_site.csv',index=False)
 if region_rows: pd.DataFrame(region_rows).to_csv(out/'heldout_R_by_history.csv',index=False)
 print(f'wrote {a.stage} patch analysis to {out}')
if __name__=='__main__': main()
